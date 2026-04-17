# Prompt: Migrate the ray-march from fragment shader to compute shader

Goal: test the hypothesis that rewriting `march_cartesian` as a compute shader with threadgroup cooperation will improve Fragment Occupancy (currently ~12%) on Apple Silicon and produce measurable FPS gains.

This is a research task, not a sure thing. Treat the steps below as *experiments that could be wrong*. Stop and report back if a hypothesis is falsified along the way.

## What we believe (but haven't proven)

- Current `march_cartesian.wgsl` uses ~260 bytes of per-thread state (5 stack arrays + assorted locals). On Apple M2 Max, this keeps Fragment Occupancy at ~12%, and the shader is register-pressure-limited rather than memory- or compute-bound.
- Moving per-depth stack state from registers to threadgroup memory (via compute shader) **might** recover occupancy.
- Tiles of nearby pixels **might** cooperatively walk overlapping tree regions, cutting redundant global-memory fetches via a shared per-tile tree cache.

Neither is guaranteed. Compute shaders on Apple Silicon have their own constraints:
- Threadgroup memory has its own occupancy cost (shared across SIMD groups in the group).
- Compute dispatch adds per-frame overhead (not free).
- Thread divergence inside a workgroup can still serialize execution.
- Fragment shaders may have Apple-specific rasterizer fast paths we lose.

The question we want answered: **does this actually help for our workload?**

## What to read first

Required context (read in full before making any changes):

- `docs/testing/perf-occupancy-diagnosis.md` — the measurement that motivated this work and the working theory of the bottleneck.
- `docs/testing/gpu-telemetry.md` — how to measure Fragment Occupancy / ALU Utilization / buffer limiters.
- `assets/shaders/march.wgsl` — the current ray-march shader. Note the `s_*` stack arrays and `cur_occupancy`/`cur_first_child` scalar cache.
- `assets/shaders/main.wgsl` — fragment shader entry point that calls `march()`.
- `src/renderer/init.rs` + `src/renderer/draw.rs` — where the fragment render pass is set up and executed.
- `src/renderer/buffers.rs` — tree / ribbon / uniforms buffer upload.

## Baseline (capture before any changes)

Before touching code, capture these numbers. They anchor the "did this help?" question for every step.

```bash
cargo build --bin deepspace-game --release

# Baseline A: slow-soldier at retina
scripts/capture-gpu-trace.sh baseline-soldier-1440 -- \
    --render-harness --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 8 --spawn-xyz 1.15 1.1 1.04 --spawn-depth 5 \
    --disable-overlay --harness-width 2560 --harness-height 1440 \
    --exit-after-frames 300 --timeout-secs 15 --suppress-startup-logs
scripts/parse-metal-trace.py tmp/trace/baseline-soldier-1440.trace \
    > tmp/trace/baseline-soldier-1440.txt

# Baseline B: INV8 empty-heavy zoom-in
scripts/capture-gpu-trace.sh baseline-inv8 -- \
    --render-harness --plain-world --plain-layers 40 --spawn-depth 3 \
    --spawn-pitch -1.0 --script "wait:60,zoom_in:10,wait:240" \
    --harness-width 1920 --harness-height 1080 \
    --exit-after-frames 400 --timeout-secs 15 --suppress-startup-logs \
    --disable-overlay
scripts/parse-metal-trace.py tmp/trace/baseline-inv8.trace \
    > tmp/trace/baseline-inv8.txt

# Baseline C: Menger fractal
scripts/capture-gpu-trace.sh baseline-menger -- \
    --render-harness --menger-world --plain-layers 15 \
    --harness-width 1920 --harness-height 1080 \
    --exit-after-frames 300 --timeout-secs 15 --suppress-startup-logs \
    --disable-overlay
scripts/parse-metal-trace.py tmp/trace/baseline-menger.trace \
    > tmp/trace/baseline-menger.txt

# Baseline D: standard render_harness_timing numbers
for scenario in soldier-1440 inv8 menger; do
  for i in 1 2 3 4 5; do
    # Repeat the harness args from above for each scenario.
    # Grep render_harness_timing for gpu_pass / submitted_done / total.
    ...
  done
done
```

Record, as a table in the eventual commit message or a `tmp/compute-migration-baseline.md`:

| scenario | fragment occupancy | alu utilization | buffer read limiter | gpu_pass_ms mean | submitted_done_ms mean | total_ms mean |
|---|---|---|---|---|---|---|

## Proposed migration plan

Do these in order. **Each step is its own commit** with a clear title. After each commit, re-run the telemetry captures and compare against baseline. If the numbers don't move the expected direction, stop and report — the hypothesis for that step didn't hold.

### Step 1: Minimal compute-shader port

**Goal**: equivalent visual output, measurable occupancy change (either direction).

- Add a new compute shader `assets/shaders/march_compute.wgsl` that mirrors the existing `main.wgsl` fragment entry, but as a compute `@workgroup_size(8, 8, 1)` or `(16, 16, 1)` entry with `@group(0) @binding(X) var output_color: texture_storage_2d<rgba8unorm, write>`.
- Do NOT yet use threadgroup memory for tree state. Keep per-thread state identical to the fragment shader. This isolates "compute dispatch pipeline cost" from "threadgroup optimization benefit."
- Add a new render path in `src/renderer/` that dispatches the compute shader into a storage texture, then blits that texture to the surface via a fullscreen fragment pass (or `copyTextureToTexture` if wgpu supports it for the surface format).
- Gate behind a CLI flag like `--renderer compute` vs default `fragment`. Don't delete the fragment path — keep it as a control.

**Tests after Step 1**:
- Screenshots for plain_d8, sphere, zoom3, menger, slow-soldier must be pixel-identical to the fragment-shader version. Minor colorspace differences are a correctness bug — track them down.
- Timing + occupancy telemetry on all 4 scenarios.
- `cargo test --lib gpu` still passes.

**Expected outcome**: occupancy MIGHT go up (fewer per-thread registers because no fragment-interpolator pressure) or MIGHT go down (compute has its own register constraints). FPS should stay within ±15% of baseline. If it's way worse, the compute dispatch overhead is eating the wins, which is itself useful information.

### Step 2: Move per-depth stack to threadgroup memory

Only if Step 1 compiled and rendered correctly. This is where the hypothesis actually gets tested.

- Move `s_node_idx`, `s_cell`, `s_side_dist`, `s_node_origin`, `s_cell_size` into a threadgroup-scoped array indexed by `local_invocation_index`. Each thread gets its own "row" in threadgroup memory.
- Keep `cur_occupancy`/`cur_first_child` as per-thread registers (scalar cache from `bf7ff20`).

**Tests after Step 2**:
- Same screenshots + telemetry + unit tests as Step 1.
- Specifically check Fragment Occupancy → Compute Occupancy in the telemetry summary.
- **This is the step that either validates or refutes the hypothesis.** If Compute Occupancy stays around 12%, threadgroup memory isn't helping (maybe Apple's scheduler has other constraints we don't understand). Stop and report.

**Expected outcome**: Compute Occupancy goes up to the 25-50% range. FPS improves 1.5-3× on the slow scenarios. If it only improves 1.1× or gets worse, the bet doesn't hold.

### Step 3 (optional, only if Step 2 succeeded): tile-local tree cache

This is genuinely speculative — don't do this unless Steps 1-2 delivered measurable improvement.

- In each workgroup, pre-fetch the likely-accessed tree nodes into threadgroup memory at the start of the pass.
- Thread 0 of each group does the ray-box intersection between the tile's AABB and the tree's ribbon shells; identifies the subset of nodes the tile's pixels will traverse.
- Other threads wait on a barrier, then read from threadgroup memory instead of global `tree[]` buffer for the cached nodes.

**Tests after Step 3**:
- Same screenshot suite; correctness must be preserved.
- Check Buffer Read Limiter should DROP (fewer global fetches). Threadgroup/Imageblock Load Limiter should increase.
- FPS: unclear — highly workload-dependent.

## What could make this fail

Be prepared to abandon any step and report clearly. Possible failure modes:

1. **Compute shader pipeline overhead is >2 ms per frame.** Dispatch + barrier + blit costs swamp any per-pixel gains. Verdict: fragment shader stays, abandon compute path.

2. **Threadgroup memory has its own occupancy cost we weren't expecting.** Compute Occupancy stays stuck below 20%. Verdict: the bottleneck is NOT register pressure in the way we thought. Need a different theory (divergence? instruction latency? SLC thrashing?).

3. **Compute shader visual output has correctness bugs we can't trace.** Specifically subtle divergence between fragment and compute outputs at pixel edges. Treat this as a blocker for merging.

4. **Apple Silicon compute shader storage textures have constraints we don't know.** E.g. some formats aren't writeable from compute. Test with rgba8unorm first, not more exotic formats.

5. **Shared-memory usage across the workgroup serializes threads unexpectedly.** Check with the telemetry — look at whether Fragment/Compute Occupancy rises but ALU Utilization doesn't.

## Testing matrix to run at each step

| scenario | resolution | expected occupancy direction | expected FPS direction |
|---|---|---|---|
| slow-soldier (zoom=4 inside body) | 2560×1440 | UP from 12% | UP from 43 FPS |
| INV8 empty-heavy zoom-in | 1920×1080 | UP or same | UP or same |
| Menger fractal | 1920×1080 | UP or same | roughly same (already cheap) |
| plain-d8 steady | 1920×1080 | UP or same | UP |

## What "success" looks like at the end

After Steps 1-2, reproducing the metrics:
- Fragment/Compute Occupancy ≥ 30% on the slow-soldier scenario (up from 12%).
- Slow-soldier FPS ≥ 80 (up from 43) at 2560×1440.
- No regression on any other scenario (±10% of baseline is fine).
- All screenshots pixel-identical.
- All unit tests passing.

If Step 3 is attempted and works:
- Buffer Read Limiter drops 30-50%.
- Slow-soldier FPS ≥ 120 at 2560×1440.

## What "failure" looks like

If after Step 2 the numbers don't move meaningfully:
- Revert the branch (keep fragment shader path as default).
- Write up what the telemetry showed. Specifically: what was the actual per-counter distribution? Did occupancy move? Did buffer-read limiter move?
- Propose an alternative theory (maybe it's divergence, not register pressure? Or maybe Apple's fragment rasterizer has fast paths we lose in compute?).

Negative results are valuable. The worst outcome is "tried it, didn't work, abandoned without understanding why."

## Out of scope for this work

Don't do any of these as part of this migration. They're follow-ups if Steps 1-2 succeed:

- Temporal upscaling / TAA
- Per-node empty-run metadata (INV9)
- Sparse-vs-dense re-evaluation
- Streaming / LOD beyond what the packer already does
- UI/overlay changes
- Platform testing beyond macOS (if Step 1-2 works on macOS we'll port, but start focused)

## Don't

- Don't delete the fragment shader path until compute has been running in the live game for a week.
- Don't skip the baseline capture. Without it, we can't tell if anything changed.
- Don't merge with a visual regression. Screenshots must match.
- Don't promise the 7× number in any commit message or doc. We don't know if the hypothesis holds. Report measured gains honestly, including null results.
- Don't couple this work with any other optimization. If you also add INV9 halfway through, we can't tell which change moved the numbers.

## If you get stuck

Publish the telemetry comparison (before/after tables of the counters) and escalate. Specifically useful data to share:

- Fragment Occupancy mean before / Compute Occupancy mean after.
- ALU Utilization mean before / mean after.
- Buffer Read Limiter mean before / after.
- Frame time breakdown (gpu_pass vs submitted_done vs render_wait).
- Screenshot diff if any.

The answer might be "the hypothesis was wrong and here's what the numbers say instead" — that's a perfectly good outcome.
