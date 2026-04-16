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
  --timeout-secs 6
```

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

## Anti-Patterns

Do not do these:

- running multiple perf harnesses in parallel and comparing their times
- trusting a perf number before checking the screenshot
- changing render budget and packer strategy in the same experiment
- mixing overlay/native-window issues into harness-based renderer debugging
- leaving verbose deep-path debug logging on while trying to compare CPU timings
