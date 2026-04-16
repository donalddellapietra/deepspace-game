# Render Perf Isolation Playbook

This is the current debugging procedure for render-performance regressions in
the deep-layers refactor branch. The goal is to isolate one cost source at a
time and avoid mixing renderer cost, harness artifacts, and CPU debug overhead.

## Rules

1. Use the deterministic render harness first.
2. Change one variable at a time.
3. Measure sequentially, never with multiple GPU runs in parallel.
4. Use screenshots to confirm that a perf fix did not reintroduce visual
   regressions.
5. Separate GPU cost from CPU cost before reasoning about "render slowness".
6. Never rely only on warm-up-gated FPS metrics. Always enforce a hard
   startup-inclusive stall gate with:
   - `--max-any-frame-ms`
   - `--max-frame-gap-ms --frame-gap-warmup-frames 2`

## Baseline Harness

Use the plain world first. It is the smallest controlled environment and avoids
sphere-specific traversal noise.

Example baseline command:

```bash
cargo run --bin deepspace-game -- \
  --render-harness \
  --plain-world \
  --plain-layers 20 \
  --spawn-depth 20 \
  --harness-width 1280 \
  --harness-height 720 \
  --exit-after-frames 2 \
  --timeout-secs 4
```

For live-loop perf checks (window/surface path), always include:

```bash
--max-any-frame-ms 250 --max-frame-gap-ms 400 --frame-gap-warmup-frames 2
```

Keep timeout budgets tight. Prefer `4s` unless the scenario has a concrete
reason to run longer.

## Isolation Order

### 1. Confirm visual correctness first

- Run with `--screenshot`.
- Verify the image before trusting any timing number.
- If the image is wrong, fix correctness before continuing perf work.

### 2. Establish render-target scaling

Run the same scene at multiple harness sizes:

- `64x64`
- `320x180`
- `1280x720`

Interpretation:

- If time is flat across sizes, the bottleneck is not pixel shading.
- If time scales with size, the bottleneck is mostly GPU work.

### 3. Disable non-render subsystems

Use harness flags to strip optional work:

- `--disable-highlight`
- `--suppress-startup-logs`
- `--force-visual-depth N`

Interpretation:

- If `disable-highlight` helps, the CPU cursor/highlight path is the issue.
- If `force-visual-depth` helps, the depth budget is the issue.

### 4. Compare shallow vs deep with the same local budget

Always compare multiple anchor depths under the same forced visual depth.

Example:

```bash
--spawn-depth 6  --force-visual-depth 5
--spawn-depth 20 --force-visual-depth 5
--spawn-depth 30 --force-visual-depth 5
```

Interpretation:

- If costs converge, the asymmetry is in `visual_depth()` or another zoom-driven
  depth budget.
- If deep layers are still slower, the issue is elsewhere.

### 5. Check packer size separately from shader cost

For Cartesian runs, log packed node counts and compare:

- full/exact tree pack
- LOD-preserving pack

Interpretation:

- If packed nodes scale with layer depth and frame time scales with them, the
  packer/buffer size is leaking absolute depth into the renderer.

### 6. Split CPU frame phases

Once GPU cost is under control, split these independently:

- update
- upload_tree_lod
- frame_aware_raycast
- set_highlight
- render_offscreen / render

Do not treat "highlight" as one opaque bucket if it is still expensive.

## Current Findings

At the time of writing:

- The gray deep-layer plain bug came from descending into child nodes using the
  wrong entry box size.
- Offscreen harness cost was initially misleading because the harness was not
  forcing the actual renderer resolution.
- Full Cartesian exact packing created a large remaining layer asymmetry.
- Restoring the LOD-preserving Cartesian packer brought deep-layer render cost
  much closer to shallow-layer cost.
- The remaining obvious cost center is the CPU highlight/raycast path, not the
  GPU renderer.

## Deep Per-Phase Breakdown (2026-04)

The harness now captures every measurable phase per frame, not just averages.
See `scripts/perf-breakdown.sh` and the `--perf-trace PATH` flag.

### New per-frame signals

| signal | origin | meaning |
|---|---|---|
| `update_ms`, `camera_write_ms` | `App::update` | player + camera buffer write |
| `pack_ms` | `upload_tree_lod` | `pack_tree_lod_selective` CPU cost |
| `ribbon_build_ms` | `upload_tree_lod` | `build_ribbon` CPU cost |
| `tree_write_ms`, `ribbon_write_ms` | `Renderer::update_{tree,ribbon}` | `queue.write_buffer` cost |
| `bind_group_rebuild_ms` | buffer upload | rebuild when buffer outgrew allocation |
| `highlight_ms`, `highlight_raycast_ms`, `highlight_set_ms` | `update_highlight` | already existed |
| `render_{encode,submit,wait}_ms` | `render_offscreen` | CPU-side cost of each phase |
| `gpu_pass_ms` | Metal `TIMESTAMP_QUERY` | GPU ray-march pass start→end |
| `submitted_done_ms` | `queue.on_submitted_work_done` | true submit→GPU-done |
| `packed_node_count`, `ribbon_len`, `effective_visual_depth`, `reused_gpu_tree` | `App` state | what the frame was asked to do |

The harness emits three summary lines at end-of-run (`render_harness_timing`,
`render_harness_worst`, `render_harness_workload`) and, when `--perf-trace PATH`
is set, one CSV row per frame for post-hoc analysis.

### Known caveats

- **Apple Silicon Metal timestamp quirk.** Per-pass timestamps occasionally
  report `end_tick < start_tick` for passes under ~1 ms. The harness takes the
  absolute value and clamps physically impossible values (>5 s) to `None` in
  the CSV. `gpu_pass_samples` in the summary reflects the valid count.
- **`gpu_pass_ms` undercounts TBDR resolve.** Apple's tile-based deferred
  renderer does the tile→main-memory resolve *after* the `endOfPassWrite`
  timestamp sample. Resolve time shows up in `submitted_done_ms` but not in
  `gpu_pass_ms`. For "how long did the GPU really take", use `submitted_done_ms`.

### Matrix results (plain world, warmup=10, 60 frames)

Resolution sweep at spawn_depth=6:

| size | render_total | render_wait | submitted_done | gpu_pass |
|---|---|---|---|---|
| 64×64     | 5.3 ms  | 5.2 ms  | 5.2 ms  | 1.8 ms  |
| 320×180   | 4.2 ms  | 4.1 ms  | 4.1 ms  | 1.4 ms  |
| 640×360   | 7.9 ms  | 7.8 ms  | 7.9 ms  | 3.4 ms  |
| 1280×720  | 21.4 ms | 21.3 ms | 21.3 ms | 10.0 ms |
| 1920×1080 | 40.7 ms | 40.5 ms | 40.5 ms | 16.7 ms |

Depth sweep at 1280×720:

| depth | packed_nodes | ribbon_len | render_total | gpu_pass |
|---|---|---|---|---|
| 3  | 15 | 0  | 9.1 ms  | 3.3 ms  |
| 6  | 40 | 3  | 21.3 ms | 8.4 ms  |
| 10 | 58 | 7  | 21.7 ms | 9.2 ms  |
| 14 | 66 | 11 | 21.4 ms | 9.7 ms  |
| 17 | 76 | 14 | 21.7 ms | 11.7 ms |

World preset at 1280×720, spawn_depth=6:

| preset | packed_nodes | render_total | gpu_pass |
|---|---|---|---|
| plain  | 40  | 21.7 ms | 9.9 ms |
| sphere | 779 | 9.3 ms  | 3.8 ms |

### Conclusions

- **CPU phases are sub-percent.** `update + upload + highlight + encode + submit`
  sum to ≤ 0.2 ms at every resolution and depth. The CPU is not the bottleneck.
- **`render_wait_ms` ≈ `submitted_done_ms`.** The CPU-side poll accurately
  reflects GPU-done time on Metal. No hidden CPU stall.
- **`gpu_pass_ms` is ~half of `submitted_done_ms`.** The remaining ~10–12 ms at
  1280×720 is outside the per-pass timestamp window. On Apple Silicon this is
  almost entirely the TBDR tile-resolve phase. Lowering the render target
  resolution directly lowers this cost.
- **Depth-independent shader cost from layer 6+.** With LOD working, shader
  time plateaus around ~9–12 ms at 1280×720 regardless of anchor depth — a big
  improvement over the old "deeper = slower" behavior.
- **Sphere world is ~2.3× faster than plain world at depth 6** despite 20× more
  packed nodes. Sky-dominant framings cheap-out early in the DDA; the flat
  plain at depth 6 is the worst-case workload (every pixel hits the surface).

### How to use the harness

Run the full matrix:

```bash
scripts/perf-breakdown.sh all
```

Or just one dimension:

```bash
scripts/perf-breakdown.sh resolution
scripts/perf-breakdown.sh depth
scripts/perf-breakdown.sh world
```

CSVs land under `tmp/perf/<label>.csv`. Columns are documented in
`src/app/test_runner/runner.rs::PerfTraceWriter::new`.

For a single ad-hoc run:

```bash
cargo run --bin deepspace-game -- \
  --render-harness --plain-world --plain-layers 20 \
  --spawn-depth 6 --harness-width 1280 --harness-height 720 \
  --exit-after-frames 60 --timeout-secs 7 \
  --perf-trace tmp/perf/my_run.csv --perf-trace-warmup 10 \
  --suppress-startup-logs
```

The three summary lines (`render_harness_timing`, `render_harness_worst`,
`render_harness_workload`) print at end of run.

## Anti-Patterns

Do not do these:

- running multiple perf harnesses in parallel and comparing their times
- trusting a perf number before checking the screenshot
- changing render budget and packer strategy in the same experiment
- mixing overlay/native-window issues into harness-based renderer debugging
- leaving verbose deep-path debug logging on while trying to compare CPU timings
- trusting `min-fps`/`min-cadence-fps` alone while warm-up hides startup stalls
