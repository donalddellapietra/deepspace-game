# Brickmap-as-NodeKind implementation prompt

## Your assignment

Implement base-3 brickmaps on a new worktree branched off `.claude/worktrees/occupancy-stack-slim`. The previous attempt on `compute-shader-migration` failed by shipping a single-depth (3³) dense-child shortcut that wasn't really a brickmap — it collapsed one tree level into 7 u32s and measured zero perf win. **Do not repeat that design.**

This prompt spells out the correct design. If any detail here conflicts with what you think is better, push back before writing code. Don't silently redesign.

## Architectural constraints (non-negotiable)

The codebase has a strict rule: **every layer of the recursive structure is identical, no special leaf layer, no scale-specific special-casing.** The ONE exception is content-motivated node kinds (`NodeKind::CubedSphereBody`, `NodeKind::CubedSphereFace`) that dispatch to their own DDA.

Brickmaps must fit that exception pattern, not violate it:

- **`NodeKind::Brick`** is a new node kind, analogous to `CubedSphereBody`. A brick can appear at ANY depth in the tree where pack-time determines the subtree is dense enough — it is NOT "the leaf layer."
- **No asymmetry between scales.** A brick at depth 3 and a brick at depth 10 behave identically; the only difference is the world-size of the cells inside.
- **Everything is base-3.** No 16³, no 32³. Bricks are 3^N cells on a side. See "Brick size" below.
- **The outer (cross-brick) traversal is the existing recursive DDA, unchanged.** Only the inner (within-brick) traversal is flat.

## Brick design

### Node kind dispatch

Add `NodeKind::Brick` to `NodeKindGpu` (see `assets/shaders/bindings.wgsl:67-72`). Top-level `march()` in `march.wgsl` already dispatches on `current_kind`; add a branch that calls `march_brick(...)` in the same slot as `sphere_in_cell(...)`. Inside a Cartesian node's descend, check the child's kind; if `Brick`, call `march_brick` instead of pushing the stack.

### Brick size

Each brick is **27³ voxels** (= 19683 voxels, 3 levels of tree collapsed). This is the default. Store as a compile-time constant `BRICK_DIM: u32 = 27u`.

Rationale:
- 3³ (previous attempt) is too small — only 1 level collapsed, outer dep chain unchanged.
- 9³ (2 levels) is lightweight (729 B/brick) but brick entries are still rare in the inner DDA.
- 27³ (3 levels) averages ~12-18 cells crossed per ray — this is what makes the "flat DDA > recursive DDA" dep-chain win materialize.
- 81³ is too much memory per brick.

Per-brick storage at 27³ × 1 byte block_id = **19.2 KB/brick**. A soldier model at 729³ packs to 3³ = 27 bricks = 520 KB. Plausible.

### Inner traversal

Flat 3D DDA inside the brick. Think "Amanatides & Woo" DDA but in a dense 27³ grid with no tree, no popcount, no rank:

```
fn march_brick(
    brick_data_offset: u32,    // offset into brick_data[] buffer
    brick_world_min: vec3<f32>, // world coords of brick's [0,0,0] corner
    brick_world_size: f32,      // world edge length of brick
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
) -> HitResult {
    // Standard 3D DDA. Cell coords in [0, BRICK_DIM)³.
    // Loop: check occupancy at brick_data[offset + z*729 + y*27 + x].
    // If != 0, hit. Else advance cur_side_dist, cur_cell on min axis.
    // On OOB, miss — caller pops / continues outer DDA.
}
```

Keep the inner DDA aggressively simple. Use our existing `min_axis_mask` helper. Empty cells = u8 0; block cells = u8 palette_index. No 27-bit occupancy masks — full dense storage is the whole point.

### Storage layout

**Separate buffer**, not inline in `tree[]`. Add `@group(0) @binding(N) var<storage, read> brick_data: array<u32>;` — each u32 packs 4 block_ids (to stay naturally aligned). Pack `brick_data_offset` into the child tag/packed word (see `assets/shaders/tree.wgsl:54-56` for existing packed-u32 layout — the `_pad` bits are available).

Rationale: inline-in-tree couples brick memory to the sparse tree's addressing. Separate buffer allows pool allocation, eviction, and independent streaming of bricks.

### Mipmap (optional, Phase 2)

Per-ray in-brick LOD: when the brick is far from the camera, walk at reduced resolution. Base-3 mip levels:

- Level 0: 27³ (27 cells/axis)
- Level 1: 9³ (cells cover 3× more world)
- Level 2: 3³

Mip cell value = mode or volume-weighted representative of its 3³ children. LOD selection per ray: same formula as `lod_pixels` in `march_cartesian`, applied to the per-cell world size at the current mip.

**Phase 2** — skip in Phase 1. First land a correct-but-single-resolution brick, measure, then add mip.

## Pack-time: when does a subtree become a brick?

At pack time (`src/world/gpu/pack.rs`), when packing a subtree, decide: emit as sparse `NodeKind::Cartesian` (existing behavior) or as `NodeKind::Brick`?

Criteria (all must hold):
1. **Size**: subtree contains ≤ 27³ voxels worth of content. I.e., subtree depth from leaves is ≤ 3.
2. **Density**: subtree has > 5% non-empty voxels. Empty/near-empty stays sparse (storage win).
3. **Not a sphere body / face**: existing node-kind dispatch takes priority.

If all hold, flatten the subtree into a 27³ voxel grid, allocate a `brick_id`, append packed voxels to `brick_data[]`, and emit a `NodeKind::Brick` with that `brick_id`.

**Don't force-brick at a fixed depth.** Let density decide. This matches the "no special leaf layer" rule — bricks appear where content is dense, regardless of scale.

## Phases

### Phase 1: Correctness, default world

- Pack logic with density-triggered brick emission
- Inner `march_brick` in WGSL
- `NodeKind::Brick` dispatch in outer `march()`
- Edit path: placing/removing a block inside a brick writes directly to `brick_data[]` at the indexed offset (O(1))
- Edit path: adding a block where no brick exists yet still goes down the sparse path (bricks only emerge at pack time initially)

Test:
- Render soldier scene at 2560×1440. **Must be pixel-identical to current baseline** (or within 1-pixel-max-6-intensity tolerance, same as we've been using). Any visible regression is a blocker.
- `submitted_done_ms` measured and reported vs. baseline 6.39 ms

### Phase 2: Edit-path brick creation

- When a block is placed in a subtree that becomes dense enough mid-session, allocate a new brick and replace the sparse subtree
- When all blocks in a brick are removed, either keep as empty brick (defer re-sparsify) or re-sparsify inline

### Phase 3: Mipmap + per-ray LOD

(Do NOT attempt in Phase 1. Ship Phase 1, measure, then decide.)

## What success looks like

**Phase 1 success criteria**:
1. Soldier scene renders pixel-identical (or within tolerance) to current baseline.
2. `submitted_done_ms` on soldier scene improves by ≥ 15%. If it doesn't, report why — the hypothesis is that flat inner DDA cuts the dep chain enough to matter. If avg_steps drops significantly but wall-clock doesn't, we've learned the chain wasn't the binding constraint after all.
3. Plain-d8, sphere, zoom3 scenes render pixel-identical (no visual regressions elsewhere).
4. `cargo test --lib gpu` passes.

**Anti-goal: do not ship unless all four hold.** If bricks regress even one canonical scene, bricks are wrong for this workload and we need to know that cleanly.

## What to avoid (lessons from the failed attempt)

1. Do NOT make bricks a single-depth collapse (3³ = one tree level). That's not a brickmap, and it doesn't shorten the outer dep chain.
2. Do NOT put diagnostic register-pressure probes (`w0..w15: vec4<f32>` with keep-live guards) in the production shader. Use a separate diagnostic file if you need to probe.
3. Do NOT forget to add constants (`BRICK_FLAG_BIT`, `BRICK_EMPTY_BT`, shader_stats atomics) to `bindings.wgsl` when you reference them in `march_cartesian.wgsl`. The abandoned attempt broke compilation by forgetting this.
4. Do NOT add kill-switch env vars (`DEEPSPACE_NO_BRICK`) as a substitute for landing correctly. Either it works or it doesn't — no half-states.
5. Do NOT leave `*.bak` files in git. Use git stash.

## Measurement methodology

Register pressure can be measured via `assets/shaders/measure_compute.wgsl` + `/tmp/query_pipeline_stats.swift` (exists on `occupancy-stack-slim` branch). Use this BEFORE submitting to know whether your cuts actually moved register count. Don't guess.

Wall-clock: use `scripts/dev.sh` or the release harness:

```
./target/release/deepspace-game --render-harness --vox-model assets/vox/soldier_729.vxs \
  --plain-layers 8 --spawn-xyz 1.15 1.1 1.04 --spawn-depth 5 --disable-overlay \
  --harness-width 2560 --harness-height 1440 --exit-after-frames 300 --timeout-secs 15 \
  --suppress-startup-logs 2>&1 | grep submitted_done
```

Report `submitted_done_ms` before and after.

## Architectural red lines

Before writing any code, confirm you understand:

1. Bricks are a NodeKind, appear at any depth, no scale-specific handling.
2. Inner brick DDA is flat; outer tree DDA is recursive and unchanged.
3. Everything is base-3 — no 16³, no 32³.
4. Editing within a brick is O(1) voxel write; no sparse-tree rebuild per edit.
5. Sphere bodies and faces are untouched.

If any of these is ambiguous or you want to push back, STOP and surface it before implementing.

## Deliverable

A landable diff on a new worktree that:
- Adds `NodeKind::Brick` to the existing kind dispatch
- Implements density-triggered brick packing in `pack.rs`
- Implements flat `march_brick` in WGSL
- Wires a `brick_data` storage buffer through the renderer binding group
- Passes screenshot A/B on plain_d8, sphere, zoom3, soldier
- Reports soldier-scene `submitted_done_ms` before and after

Commit incrementally. Test after each commit. Don't let the branch drift into "debugging scaffolding" territory — if Phase 1 can't be made to work within ~500 LoC plus tests, back out and report what's blocking.
