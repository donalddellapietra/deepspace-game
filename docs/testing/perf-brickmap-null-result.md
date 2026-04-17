# Perf experiment: brickmap-as-NodeKind — null result

Implementation of the spec in [`docs/prompts/brickmap-impl.md`](../prompts/brickmap-impl.md):
a new `NodeKind::Brick` that replaces a 3-level Cartesian subtree with a flat
27³ dense voxel grid, dispatched from `march_cartesian` on the same axis as
`CubedSphereBody`. Phase 1 target: ≥15% improvement in `submitted_done_ms` on
soldier scene at 2560×1440, pixel-identical to baseline.

**Result: bricks regress every measured workload by 1% – 123%. The approach
does not fit this codebase.** The code landed; brick emission is disabled via
a 2.0 density threshold (`BRICK_DENSITY_THRESHOLD` in `src/world/gpu/pack.rs`).
The dispatch path stays in the shader so re-enabling for future investigation
is a one-constant change.

## The hypothesis

From the implementation prompt:

> Think "Amanatides & Woo" DDA but in a dense 27³ grid with no tree, no
> popcount, no rank. […] 27³ (3 levels) averages ~12-18 cells crossed per
> ray — this is what makes the "flat DDA > recursive DDA" dep-chain win
> materialize.

The binding-constraint claim being tested: **the recursive DDA's inner loop is
dep-chain-bound**, and a flat DDA breaks the chain.

## What would falsify it

If `submitted_done_ms` doesn't improve ≥15% on soldier, one of:

1. The dep chain isn't the binding constraint (register pressure is).
2. The per-cell win is real but small enough to be lost to brick-code
   register tax.
3. The recursive DDA has already absorbed the dep-chain win via its own
   optimizations (AABB cull, empty-repr bypass, scalar header cache).

All three turned out to be true.

## Experiment setup

Two binaries, identical shader source, identical dispatch path in `march_cartesian`,
differing only in whether `pack_tree*` emits bricks:

- **"no-bricks"** — `BRICK_DENSITY_THRESHOLD = 2.0` (impossible), 0 bricks emit.
  Shader still has `if kind == 3u { march_brick(...) }` branch, never taken.
- **"bricks"** — `BRICK_DENSITY_THRESHOLD = 0.30`, dense subtrees become bricks.

Both branches bypass `pack_tree_lod_selective` (temporarily switched to
`pack_tree`) and bump `MAX_STACK_DEPTH` to 12, so LOD flattening isn't
confounding the comparison. Pack-level and shader-level LOD are the dominant
perf levers in this codebase; they had to be disabled to isolate the brick
effect.

Harness: `--render-harness`, `--disable-overlay`, `--harness-width 2560
--harness-height 1440`, `--exit-after-frames 300`. Measured `submitted_done_ms`
(CPU-observed GPU completion time, not the misleading `gpu_pass` timestamp —
see "Measurement caveat" below).

## Results

| Workload | no-bricks | bricks | Δ | bricks emitted | brick density |
|---|---|---|---|---|---|
| Soldier (LOD on, as spec'd) | 6.40 ms | 14.26 ms | **+123%** | 450 | ~11% avg |
| Menger default spawn | 5.16 ms | 5.21 ms | +1% | 1 | 40% |
| Menger zoom (spawn-depth 14) | 4.39 ms | 9.18 ms | **+109%** | 1 (hit constantly) | 40% |
| Plain default (depth 8) | 3.99 ms | 5.14 ms | +29% | 6 | ~30–70% |
| Plain zoom (spawn-depth 14) | 3.71 ms | 5.15 ms | +39% | 5 | ~30–70% |

The Menger zoom row is the most decisive: **one** 40%-dense brick, hit by a
large fraction of camera rays (camera sitting inside the sponge, rays exit
through brick content), doubles the frame time. The brickmap hypothesis
predicts this should be the best case for brick wins. It isn't.

## Analysis

### Three costs, each small, summing to the regression

**1. Dispatch tax (always paid).** Even when 0 bricks emit, the shader path
carries `if kind == 3u { march_brick(...) } ...`. Dead code still counts: the
WGSL→Metal compiler reasons about register live-ranges across the whole
function body, and march_brick's locals (inv_dir, side_dist, cell, normal,
iter) compete with the outer DDA's register pool. Soldier with 0 bricks
measures ~7.59 ms vs 6.40 ms baseline = **+1.2 ms from dead code alone**.

**2. Sparse-brick empty-traversal penalty.** At threshold=0.05 on soldier,
450 bricks emit at 11% average density. A ray through a mostly-empty brick
walks ~27+ cells one DDA iteration at a time. The recursive DDA would have
skipped those empty sub-regions via the empty-representative fast path in
~5 effective steps. **~6× more iterations per ray** in sparse bricks.

**3. Warp divergence on brick entry.** Rays within a warp now diverge: some
are in `march_brick`, some in `march_cartesian`. Apple Silicon's TBDR
warp-width serialization hits hardest when different threads in a warp
execute different functions. The `gpu_pass` timestamp under-reports this
(see measurement caveat); the true cost shows up only in `submitted_done`.

### Why none of the three can be fixed in isolation

- **Fixing (1) means removing the dispatch path**, which defeats the purpose.
- **Fixing (2) means raising the density threshold** until only dense bricks
  emit — but the workloads tested don't contain dense-enough subtrees at
  depth-from-leaves==3. Soldier is hollow (surface shells, empty interior);
  Menger is 40% at that depth; plain world has dense layers but they dedup
  into a handful of library nodes.
- **Fixing (3) means per-warp coherence control** — not a WGSL primitive.

### The deeper reason: the baseline isn't what brickmap papers compare against

Published brickmap work (Crassin's GigaVoxels, Reichl et al) compares against
**naive sparse octree DDA**: per-cell popcount, cold `tree[]` reads, no
empty-skip at the inner loop. Against that baseline, brickmaps win 2–5×.

The recursive DDA on `occupancy-stack-slim` has already absorbed those wins:

| Brickmap paper's claimed win | How we already captured it |
|---|---|
| Flat inner loop, no popcount chain | Scalar header cache (`cur_occupancy`, `cur_first_child` held in registers across inner loop) |
| Skip empty subtree in one step | Empty-representative fast path (tag=2 child with `repr==255` → advance parent DDA) |
| Skip whole subtree if ray misses | AABB cull per child (`_pad`-encoded content AABB, ray-box per descend) |
| Shorter dep chain per iteration | Branchless min-axis in DDA advance |

The brick has nothing left to give. Its *theoretical* per-cell cost advantage
(one u8 extract vs popcount+rank+storage load) is real — but register-
pressure-wise, it *adds* code without removing any, because the recursive
DDA still has to exist for the non-bricked 24 of each node's children.

## Measurement caveat: `gpu_pass` vs `submitted_done`

Apple Silicon is tile-based-deferred-rendering (TBDR). Render-pass-boundary
timestamps (`gpu_pass_ms` in our harness) measure the interval between the
`@begin` and `@end` of the command encoding, *not* actual fragment shader
execution wall-clock. The fragment work happens lazily during the pass, and
finishes when the pass presents or the fence is signaled.

Consequence: `gpu_pass` reports +12–26% regression, `submitted_done` (CPU
callback on `queue.on_submitted_work_done`) reports +30–110%. Always trust
`submitted_done` for end-to-end perf on Apple Silicon; `gpu_pass` is useful
only as a component cost attribution within a single frame.

## Files touched

Code landed on branch `brickmap-attempt-1` (off `occupancy-stack-slim`):

- `assets/shaders/brick.wgsl` (new) — flat 27³ inner DDA.
- `assets/shaders/march.wgsl` — `kind == 3u` dispatch in `march_cartesian`.
- `assets/shaders/bindings.wgsl` — `NodeKindGpu` kind=3 doc, `BRICK_DIM`/
  `BRICK_VOXELS`/`BRICK_U32S` constants, `@binding(8) brick_data`.
- `src/world/gpu/pack.rs` — `OrderedEntry::Brick` variant, `try_emit_brick`,
  `flatten_brick_recursive`. Density threshold set to 2.0 (disabled).
- `src/world/gpu/types.rs` — `GpuNodeKind::brick(offset)` constructor.
- `src/renderer/{init,buffers,mod}.rs` — binding 8 plumbing.
- `src/shader_compose.rs` — `brick.wgsl` registered.
- All `pack_tree*` call sites updated for the 5-tuple return type.

## How to re-enable for future investigation

```rust
// src/world/gpu/pack.rs
const BRICK_DENSITY_THRESHOLD: f32 = 0.30;  // or 0.50 for dense-only
```

Before measuring, also bypass LOD in `src/app/edit_actions/upload.rs`
(switch `pack_tree_lod_selective` → `pack_tree`) and bump
`MAX_STACK_DEPTH` in `assets/shaders/bindings.wgsl` to avoid LOD
confounding the result. Otherwise the per-pack LOD flattening will
flatten out any dense subtree before it reaches the brick emission
check.

## What would have to be true for bricks to win here

At least one of:

1. **A workload with genuinely dense content at depth-from-leaves==3.** Our
   default scenes are all hollow / sparse / uniform-deduped. A scene with
   solid volumetric interiors (medical CT, smoke simulation, thick terrain)
   would exercise the brickmap's strength. We don't have such scenes.
2. **The dispatch tax removed via a different encoding.** Bricks as a tag
   variant (tag=3 in the packed word, no `kind` lookup) instead of a
   NodeKind. That saves one `node_kinds[]` storage load per non-empty
   child — but our inner loop already has `node_kinds` access for the
   sphere-body dispatch, so this isn't likely to win.
3. **Compute shader dispatch with per-tile coherence.** The fragment-shader
   path forces all threads in a warp through the same code; a compute
   dispatch could batch brick-entering rays separately. Major rewrite.

None of these make brickmap the right next experiment. The perf roadmap
should look elsewhere.

## Prior art in this tree

- `compute-shader-migration` branch commit "First-class NodeKind::Brick
  with tunable side (3, 9, 27)" is the previous brick attempt the prompt
  mentions. It shipped a single-depth (3³) dense-child shortcut that
  collapsed one tree level into 7 u32s. Per the prompt: "measured zero
  perf win." This attempt used the proper 27³ design but landed in the
  same measurement ditch.
- The `INV1`-`INV9` perf investigation series concluded that
  **register pressure on Apple M-series is the binding constraint**, not
  dep chain length. That finding correctly predicts this null result:
  any shader code addition that doesn't strictly reduce register count is
  neutral-to-negative.
