# Prompt: Migrate the ray-march from sparse-tree DDA to a mipmapped brickmap

Goal: test whether rewriting the inner ray-march to walk a two-level
sparse-brick + dense-mipmapped-voxel grid (a "brickmap") closes the
2–5× speedup needed to hit 60 FPS on lower-end hardware.

This is a research task framed as falsifiable experiments. Each phase
below is a hypothesis that can hold or be refuted. Stop and report
back if a phase's hypothesis doesn't hold.

## Context: what we've measured, what's still broken

The sparse-tree DDA is now ALU-limited at 96% of frames on Apple
Silicon (measured via Metal `Top Performance Limiter`). Our
compute-shader + slim-stack + register-shadow path hits 113 FPS at
2560×1440 on M2 Max — well above the 80 FPS target in the previous
compute-shader-migration prompt, but still only ~3-4× above the
theoretical 0.7 ms floor because the inner DDA iterations form a
serial dependency chain. Each iteration depends on the previous
(cur_cell → slot → occupancy-test → axis-pick → advance → new
cur_cell), so Apple's in-order GPU scheduler can't hide the 4-cycle
ALU latency between dependent ops. This shows up as 44% ALU
utilization even at saturated-ALU-limiter — the ALU is busy, but
half-stalled by the chain.

Things we tried, and what they taught us:

1. **Stack-slim (scalarize 3 of 5 stack arrays)** — big win, +54%.
   Cut per-iteration ALU work.
2. **Register-shadow (keep current-depth cell in registers)** —
   modest +8%. Eliminated workgroup-memory round-trips.
3. **Workgroup size 32 vs 64** — no change. Proved occupancy isn't
   the limiter; register file is.
4. **Private-array stack** — regressed by 24%. Spills hit local
   memory harder than the workgroup path we replaced.
5. **fp16 DDA state** — null result. Throughput doubles, but
   dependency-chain latency is unchanged on Apple GPUs, and per-op
   precision loss produced visible pixel divergence.

Cumulative conclusion: **the shader's inner loop is dependency-chain
latency-bound, not throughput-bound, not bandwidth-bound, not
occupancy-bound.** We can't parallelize the chain. We can only
shorten it (fewer iterations per ray) or make each link lighter
(fewer ALU ops per iteration). Stack-slim and register-shadow
already did the viable work on the second path. The remaining 2–5×
requires the first.

For context see:

- `docs/testing/perf-occupancy-diagnosis.md` — initial
  occupancy-register-pressure story (partially superseded by the
  ALU-limiter finding).
- `docs/testing/compute-migration-results.md` — measured numbers
  through the four optimization commits to date.
- `docs/prompts/compute-shader-migration.md` — predecessor research
  prompt; structure and epistemology worth matching here.

## The proposal: brickmap + per-ray mip LOD

A brickmap replaces the sparse-tree inner walk with two structurally
simpler steps:

1. **Top level (sparse)** — a flat index of "bricks" (fixed-size
   voxel chunks, say 32³) keyed by brick coordinates. Entries are
   either `EMPTY` or a pointer to brick-data.

2. **Inside a brick (dense, mipmapped)** — each brick-data carries
   the full 32³ voxel grid plus its mipmap pyramid down to 1×1×1.
   Each mip voxel pre-stores the averaged color/ID of its underlying
   native-level subtree.

At render time, per pixel:

- Compute the target mip level from the projected pixel size:
  `mip = log2(ray_distance × pixel_size_world / native_voxel_size)`.
- March the ray through the brick index (sparse, few steps for a
  scene). For each populated brick entered, march through the mip
  level's dense voxel grid inside it (~1-10 steps at the selected
  LOD).
- Each voxel at the selected mip level is already the "finished"
  representative color — no tree descent, no mip cascade at read
  time.

**The hypothesis**: each inner-loop iteration drops from ~40 ALU ops
(OOB + slot + popcount + tag + descent/pop dance) to ~15 ALU ops
(OOB + one 3D-indexed load + hit test), AND the total iterations per
ray drop from ~32 to ~15 because there's no per-subtree
descend/pop bookkeeping. Both effects compound; combined speedup
could be 2-5×.

The dependency chain still exists — we're still walking a DDA. But
each link is dramatically cheaper and there are fewer links.

### Why this preserves the game's layer UX

The sparse tree stays as the authoritative world storage. The
brickmap is a **render cache** covering the 5-7 levels of scale
currently visible around the camera's anchor depth — an "anchor
window." When the camera zooms past the window's edges, the
existing ribbon-pop machinery transforms the ray into the adjacent
window's coord system, same as it pops between tree depths today.
Windows are baked from the tree; no structural change to the tree.

### Why edits stay cheap

Single-voxel place/break touches exactly one brick. Updating the
brickmap costs the log₂(brick_size) mip levels' worth of local
averages — ~6 writes for a 32³ brick, ~100 µs total. Can be done
synchronously in the same frame as the tree update. The 1-2 frame
render-staleness problem ("you broke a block but it still looks
there") does NOT apply here because the cascade is fast.

Bulk edits (explosions, structure placement affecting thousands of
voxels) can still be batched across frames — apply a 2 ms budget of
edits per frame, queue the rest. From the player's perspective this
reads as an expanding effect, which is usually good-feeling.

## What to read first

Required (full read before touching code):

- `assets/shaders/march_cartesian.wgsl` — the DDA kernel being
  replaced. Note the per-iteration structure and stack scalarization.
- `assets/shaders/march.wgsl` — outer loop with ribbon pop; this
  stays, just invokes a new inner kernel.
- `assets/shaders/bindings.wgsl` — current buffer layout, esp. the
  `tree[]` interleaved header format and the `ribbon[]` structure.
- `src/renderer/init.rs`, `draw.rs`, `buffers.rs` — GPU resource
  setup and the compute-pipeline plumbing a brickmap pipeline can
  parallel.
- `src/world/gpu/` — tree packer. The brickmap baker sits next to
  this, reading the same tree representation and emitting brickmap
  windows.
- `docs/testing/compute-migration-results.md` — performance baselines
  on the slow-soldier scene. Must not regress beyond `±10%` on any
  scenario at any phase (budget for measurement noise).

## Baseline (capture before any changes)

Before building anything, record:

```bash
cargo build --bin deepspace-game --release

# A: slow-soldier at retina (motivating scenario)
for i in 1 2 3 4 5; do
  timeout 30 ./target/release/deepspace-game \
    --render-harness --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 8 --spawn-xyz 1.15 1.1 1.04 --spawn-depth 5 \
    --disable-overlay --shader-stats \
    --harness-width 2560 --harness-height 1440 \
    --exit-after-frames 300 --timeout-secs 20 \
    --suppress-startup-logs 2>&1 \
    | grep -E "render_harness_timing|render_harness_shader"
done

# B: same scenario at 1080p (lower-end-hardware proxy)
# ... same args, harness-width 1920, harness-height 1080 ...

# C: Metal GPU Counters trace
scripts/capture-gpu-trace.sh baseline-brickmap-slow-soldier -- \
    --render-harness --vox-model assets/vox/soldier_729.vxs \
    ...same args as A...
scripts/parse-metal-trace.py tmp/trace/baseline-brickmap-slow-soldier.trace
```

Record, as a table in the commit message or a
`tmp/brickmap-baseline.md`:

| scenario | resolution | gpu_pass_ms | submitted_done_ms | total_ms | Top-Limiter |
|---|---|---|---|---|---|

`Top Performance Limiter` should be ~96% `ALU Limiter` (confirms the
motivating diagnosis before we change anything).

## Proposed phases

Do these in order. **Each phase is one commit with a clear title.**
After each commit, re-run the baselines and compare. If the numbers
don't move the expected direction, stop and report — the hypothesis
for that phase didn't hold.

### Phase 1: brickmap for one scene, one anchor, no editing

**Goal**: prove the shader + memory layout work and deliver the
inner-loop speedup. Scope it ruthlessly — one scene, one fixed
anchor depth, no dynamics. This is the first falsification test.

- Add a Rust-side offline baker that takes a `VoxelTree` + anchor
  depth + brick-size parameter and emits a brickmap window: a flat
  sparse brick index + dense mipmapped voxel blobs per populated
  brick. Choose brick size up front (likely 16³ or 32³; prototype
  with one value and justify in the commit).
- GPU-resident representation: two storage buffers — a `brick_index`
  (coord → brick pointer or EMPTY) and a `brick_data` blob holding
  all mip pyramids tiled linearly. The shader indexes `brick_index`
  with brick coords, then dereferences into `brick_data`.
- New shader `assets/shaders/march_brickmap.wgsl` replacing
  `march_cartesian` for the Cartesian path. Keep `march.wgsl`
  (outer ribbon loop) unchanged; it just calls the new kernel.
- Gate behind a CLI flag like `--renderer brickmap` vs default
  `compute`. DO NOT delete the sparse-tree path.
- Only test the slow-soldier scene at the single anchor depth the
  camera spawns at. Everything else can regress or be gated off.

**Tests after Phase 1**:
- Screenshot for slow-soldier must be pixel-close to the current
  shader — not bit-identical (mip averaging differs from
  representative_block), but perceptually indistinguishable. Define
  a tolerance (e.g. mean diff ≤ 2.0 per channel, max single-pixel
  diff ≤ 10 per channel).
- `render_harness_timing` mean across 5 runs at 1440p and 1080p.
- GPU counter trace: Top Performance Limiter distribution.
- Memory: report brickmap size in bytes; must fit inside a 256 MB
  budget for this scene.
- `cargo test --lib` still passes.

**Expected outcome**: slow-soldier at 1440p drops from ~8 ms to
~3-4 ms (2-2.5× speedup). ALU utilization climbs toward 70%+ (the
chain is shorter, so the same ALU serves more productive work).
Fragment/Compute Occupancy likely stays similar — register pressure
is a secondary concern now. If total_ms doesn't drop at least 30%,
the hypothesis is wrong and we stop. If it drops 50%+, we proceed.

### Phase 2: synchronous single-voxel edits

Only proceed if Phase 1 delivered the speedup.

- Implement brick-local mip cascade: given a voxel update at
  `(x, y, z)`, rebuild the log₂(brick_size) mip levels containing
  that voxel. Upload via `queue.write_buffer` surgically (one brick
  sub-region per edit).
- Wire into existing break/place flow: the tree and brickmap update
  in lockstep within the same edit tick.
- Profile single-edit cost (target <200 µs including upload).
- Profile bulk-edit throughput (target 1000 edits per frame at 60
  FPS, i.e. <2 ms for the bulk-edit batch).

**Tests after Phase 2**:
- Visual: break and place cycles update within the frame they fire
  (no stale-pixel glitch).
- Perf regression: steady-state FPS unchanged vs Phase 1.
- Edit latency: p99 single-edit within 300 µs.

**Expected outcome**: edits are a non-event in the frame budget.
If a single edit costs >1 ms, something's wrong in the upload path.

### Phase 3: anchor-window swapping

Only proceed if Phase 2 is correct.

- Define how many anchor windows coexist (probably 3: current + one
  neighbor in each zoom direction, pre-baked when the camera is
  within ~half a window of a boundary).
- Background baker thread: given a tree + anchor depth, produce a
  brickmap window asynchronously. Shouldn't touch the render thread.
- On zoom past a threshold, swap the active window pointer. Ribbon
  pop into the neighbor window works unchanged — the outer `march`
  loop just sees a different brick index / brick data pair per
  invocation.
- Profile window swap latency: should be free (pointer swap, not
  rebuild).
- Profile continuous-zoom FPS: no hitches >1 frame at any scale.

**Tests after Phase 3**:
- Recorded zoom sequence (5-second continuous zoom across 4-5 scale
  boundaries) at 60 FPS measured end-to-end.
- Memory: multi-window brickmap fits in 512 MB for the soldier
  scene.
- Background baker CPU usage: report mean % of one core during a
  zoom session.

**Expected outcome**: zoom UX matches today's quality (smooth, no
stutter) and 60 FPS at 1080p on an M1 Air or comparable target
machine — or at minimum, M2 Max hits ~2 ms at 1080p, predicting
60 FPS on lower hardware.

### Phase 4 (optional): streaming / eviction

If the target is infinite procedural worlds, this phase is needed
later. Not for the initial migration. Hypothesis: LRU eviction of
distant anchor windows keeps resident memory bounded even as the
camera explores unbounded worlds. Scope out separately.

## What could make this fail

Each phase has specific failure modes that should short-circuit the
plan, not get worked around.

1. **Phase 1 speedup doesn't materialize.** If total_ms drops <30%
   in Phase 1, the theory that "cheaper iterations × fewer
   iterations" compounds is wrong for our workload. Revert, write
   up the counter-evidence (maybe the sparse tree is already cache-
   friendly in ways brickmap isn't), and explore the remaining
   levers (INV9 empty-run skip, TAAU upsampling).

2. **Memory blow-up.** If a single anchor window exceeds the
   256 MB budget (or whatever ceiling is realistic for lower-end
   GPUs), the brick size is wrong or the mip pyramid too deep.
   Evaluate: larger bricks (fewer mip tips but more empty waste)
   vs smaller bricks (more index waste). There is no free lunch
   here; if nothing fits, the brickmap approach doesn't fit this
   scene's content density.

3. **Mip averaging produces visually bad LOD.** For voxel scenes,
   simple box-average of RGB at each mip level can look
   fundamentally wrong — especially at edges where you need either
   "dominant color" or "median color" not "average color." If LOD
   transitions are visibly ugly after Phase 1, tune the averager
   before proceeding; worst case, use "representative color of
   most-common subtree voxel" like our current `representative_block`
   instead of arithmetic mean.

4. **Per-ray mip selection causes temporal instability.** Small
   camera motion → ray crosses a mip boundary → color pops. Can
   happen without trilinear blending at mip edges. Mitigations are
   standard (trilinear-in-xyz × linear-in-mip-level). If still bad,
   add TAA on top (orthogonal work).

5. **Synchronous edits can't hit the latency target.** If
   `queue.write_buffer` sub-region uploads show >1 ms latency on
   any platform, the assumption that single edits are "free"
   breaks and edits must become async. That's a separate design
   rework.

6. **Anchor window swap has visible hitch.** If the swap itself
   (pointer swap, not bake) takes >1 ms, something's wrong in the
   bind-group refresh path. Investigate before accepting.

## Testing matrix at each phase

| scenario | resolution | purpose |
|---|---|---|
| slow-soldier (zoom=4 inside body) | 2560×1440 | motivating scenario; must drop ≥30% in Phase 1 |
| slow-soldier | 1920×1080 | lower-end-hardware proxy; must drop ≥30% in Phase 1 |
| INV8 empty-heavy zoom-in | 1920×1080 | regression check; ±10% of current is fine |
| Menger fractal | 1920×1080 | high-detail stress; ±10% of current |
| plain_d8 steady | 1920×1080 | nominal path; ±10% |

The slow-soldier drop is the load-bearing data. Others are
regression gates.

## What "success" looks like

After Phase 1:
- slow-soldier submitted_done_ms ≤ 4.0 ms at 2560×1440.
- slow-soldier submitted_done_ms ≤ 2.5 ms at 1920×1080.
- Top Performance Limiter still ALU, but ALU Utilization ≥ 65%.
- Screenshot perceptually matches baseline; no visible LOD pops on
  quasi-static camera.
- All other scenarios within ±10% of baseline.
- Memory budget respected.

After Phase 3:
- Continuous zoom across 3-4 scale boundaries, no hitches, 60 FPS at
  1080p on the target lower-end device (or M2 Max 1080p ≤ 2.5 ms,
  extrapolating to 60 FPS at 1/4 compute).

## What "failure" looks like

If Phase 1 doesn't deliver:
- Revert the branch; keep the compute-shader path as default.
- Write up measured counter distributions BEFORE and AFTER the
  brickmap path. Identify what moved and what didn't. Propose an
  alternative theory (maybe dependency-chain latency can't be
  shortened via inner-loop simplification; maybe the sparse tree's
  representation_block fast-path is already collapsing empty runs
  more effectively than we thought).

Negative results matter. "Tried brickmaps, net zero, here's what
shifted in the counters" is a perfectly good outcome — and it rules
out a class of solutions.

## Out of scope for this work

Do NOT mix these with the brickmap migration; they're separate
follow-ups:

- Temporal upscaling / TAAU (already exists on another branch).
- Resolution scaling (already lands elsewhere).
- INV9 empty-run metadata on the existing tree path.
- Reorganizing the tree storage itself.
- Streaming / eviction for unbounded worlds (Phase 4 covers this
  later if needed).
- Sphere / face / body dispatch changes — the brickmap replaces only
  `march_cartesian`, not the outer ribbon dispatch.

## Don't

- Don't delete the sparse-tree Cartesian path until the brickmap has
  been running in the live game across all scenes for a week.
- Don't target infinite worlds in Phase 1; bounded scenes only.
- Don't skip the baseline capture. Without it, we can't tell whether
  the brickmap helped or hurt.
- Don't accept a Phase 1 that improves slow-soldier but regresses
  other scenarios. The testing matrix is a regression gate.
- Don't promise specific speedup numbers in commit messages before
  measurement. Report measured results honestly.
- Don't couple this with any other optimization. Each phase must be
  evaluable in isolation.

## If you get stuck

Publish the counter comparison (before/after distributions for
`Top Performance Limiter`, `ALU Utilization`, `Compute Occupancy`,
`Buffer Read Limiter`). Share the screenshot diff and the memory
footprint. Specifically helpful data:

- Per-iteration AIR-IR ALU counts for the brickmap kernel vs the
  existing sparse-tree kernel (use `naga` + `xcrun metal -S` as in
  the current tooling).
- Shader stats counters: avg_steps, avg_empty, avg_descend,
  avg_lod_terminal on the same scene before/after. If iteration
  count didn't drop, the "fewer iterations" theory failed and we
  need to understand why.
- A side-by-side counter reading: if the ALU Limiter share stays at
  95%+ AND ALU Utilization doesn't climb, the simplified inner
  loop didn't actually simplify — probably the compiler emitted
  more shuffle/convert ops than we expect.

The honest answer might be "brickmap buys 1.5× not 2.5×, and we
need another lever to hit the full target." That's useful
information.
