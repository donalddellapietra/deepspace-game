# Sphere break at >1 layer below top layer renders as solid grass

Status: open. Last updated 2026-05-01. Worktree: `sphere-mercator-1-1`.

## Symptom

On a `--wrapped-planet --planet-render-sphere` world, breaking a block at
the default zoom (slab natural depth) creates a visible hole. Zooming in
deeper and breaking again — i.e. an edit at `anchor_depth > slab_natural`
— writes the correct hole into the tree but renders identically to the
unedited terrain. The hole is invisible.

## Root cause: render and edit are not symmetric on the sphere

The CPU edit path descends INTO the slab cell's anchor block when
`edit_depth > slab_natural` (see `cpu_raycast_sphere_uv` extra-levels
loop, src/world/raycast/mod.rs:238-276). It writes a Child::Empty at
the right sub-cell.

The GPU sphere DDA does NOT. `sphere_uv_in_cell`
(assets/shaders/march.wgsl:1071) marches at fixed slab granularity
(`dims_x × dims_y × dims_z`) and reads each cell with
`sample_slab_cell` (assets/shaders/march.wgsl:1013). That helper walks
exactly `slab_depth` levels and stops:

```wgsl
if tag == 1u { return block_type; }
// Last walked level: this child is a Node anchor block; its
// packed block_type is the uniform subtree's representative.
if level == slab_depth - 1u { return block_type; }
```

When the sub-cell hole turns the anchor block from uniform GRASS into a
non-uniform Node, the pack format encodes it as `tag=2,
block_type=representative_block`. `representative_block` is the most
common non-empty block — 1 / 27 sub-cells empty leaves the majority
(GRASS) untouched, so the slab cell still reads as GRASS to the
shader.

The asymmetry is acknowledged in the most recent commit on this
branch (28f4c8a, revert):

> The CPU variable-depth break (`cpu_raycast_sphere_uv` with
> `max_depth`) is KEPT — sub-cell breaks still write the correct path
> into the tree. The only caveat: a small sub-cell hole inside a slab
> cell isn't visible until that cell becomes non-uniform enough that
> its `representative_block` changes. Proper LOD-driven descent into
> non-uniform cells (mirroring Cartesian's `lod_pixels < 1.0` gate) is
> a separate follow-up.

This document is that follow-up.

## What the data layer looks like

Reproduced by `world::raycast::tests::diagnosis_sphere_deep_break_invisible_at_slab_granularity`
(src/world/raycast/mod.rs). Same +X-equator break, varying `edit_depth`:

| edit_depth | path len | slab cell at depth 5 (what shader sees) |
|---:|---:|---|
| 5 | 5 | `Child::Empty` → shader returns `REPRESENTATIVE_EMPTY` → hole visible |
| 6 | 6 | `Mixed-Node(rep_block=2)` → shader returns 2 (GRASS) → hole invisible |
| 7 | 7 | `Mixed-Node(rep_block=2)` → invisible |
| 8 | 8 | `Mixed-Node(rep_block=2)` → invisible |

Slab natural depth = embedding(2) + slab(3) = 5. Any `edit_depth > 5`
carves at sub-cell granularity inside the anchor block, where the
shader can't see.

## How Cartesian gets it right

`march_cartesian` (assets/shaders/march.wgsl:170-315) walks the tree
recursively. On a `tag=2` Node child it:

1. Computes `lod_pixels = child_cell_size / ray_dist *
   screen_height / (2 * tan(fov/2))`.
2. If `lod_pixels < LOD_PIXEL_THRESHOLD` OR stack at max → terminate
   with the representative (`child_bt`) at that level.
3. Otherwise → push the child frame onto `s_node_idx[]` / `s_cell[]`
   and continue DDA inside the child node at finer granularity.

Critically, the pack format auto-flattens uniform Cartesian subtrees
into `tag=1 Block(uniform_type)`. So a uniform anchor block packs as
ONE Block entry — Cartesian renders it as a single voxel at any zoom
level above the LOD floor. After an edit makes the anchor non-uniform,
it packs as `tag=2`, and the recursive descent reveals sub-cells.

Sphere mode currently has no analogue of step 3. It walks at slab
granularity and stops.

## Fix sketch (mirroring Cartesian)

Inside `sphere_uv_in_cell`, when the slab cell at the natural sample
position is a non-uniform Node (i.e. `tag=2, block_type=representative
≠ REPRESENTATIVE_EMPTY`):

1. Compute the slab cell's projected size on screen at the current `t`.
   The sphere cell is curved; using the cell's max-axis arc length
   (`r_step`, or `r * lon_step` at the equator, etc.) is a reasonable
   single-number proxy.
2. If projected size < `LOD_PIXEL_THRESHOLD` → terminate with
   `representative_block` (current behavior; correct LOD).
3. Else → run a sub-cell DDA INSIDE the anchor block in the same
   (lon, lat, r) parameterisation, with cell sizes
   `(lon_step / 3, lat_step / 3, r_step / 3)` and one extra level of
   slot lookup into the child node. Recurse / loop until LOD or stack
   ceiling reached.

The key constraint identified by the previous attempt (commit 268d3de,
reverted in 28f4c8a):

> Same render at every zoom — broken. At zoom-in the bevel grid
> sub-divided uniform anchor cells, making them look like fields of
> small individual blocks even where the underlying data was a single
> uniform Cartesian Node.

The previous attempt scaled `dims` by `3^extra` based on
`uniforms.max_depth`. That sub-divided EVERY cell, including uniform
ones. The fix is to gate the sub-division on tree structure
(`tag=2` non-uniform Node), not on user max_depth — exactly what
Cartesian does.

## What needs to change

- `assets/shaders/march.wgsl::sphere_uv_in_cell`: on a non-uniform
  slab cell hit, run a sub-cell DDA in the anchor's subtree at finer
  cell sizes, with the same (lon, lat, r) DDA primitives. LOD-gated by
  projected sub-cell size, not by `uniforms.max_depth`.
- `assets/shaders/march.wgsl::sample_slab_cell`: split into two
  helpers — one returns `(tag, block_type, child_node_idx)` so the
  caller can decide whether to descend; the other (or the same, with
  args) walks the anchor's subtree at a given `(sub_lon, sub_lat,
  sub_r)` triple.
- `src/world/raycast/cpu_raycast_sphere_uv`: the CPU mirror needs the
  same sub-cell descent with LOD gating, so click-targeting matches
  what the shader renders. (Today the CPU uses `max_depth` as the
  cap; the GPU should use LOD; and they need to land on the same cell
  for the cursor to be honest.)

## Out of scope here

Edit at `cy` axis (radial layer) within the natural slab is not
affected — those are real slab cells and `sample_slab_cell` returns
Empty after an edit. Only edits BELOW slab natural depth fall through
the gap.

## Files

- src/world/raycast/mod.rs:61-287 — `cpu_raycast_sphere_uv` (CPU edit hit, with `extra_levels` sub-cell descent)
- src/world/raycast/cartesian.rs:121-189 — Cartesian recursive descent (recipe to mirror)
- src/world/edit.rs:14-90 — `break_block` / `place_child` (writes the edit)
- assets/shaders/march.wgsl:1013-1054 — `sample_slab_cell` (slab-only lookup; needs anchor-descent extension)
- assets/shaders/march.wgsl:1071-1275 — `sphere_uv_in_cell` (slab DDA; needs sub-cell DDA branch)
- assets/shaders/march.wgsl:170-315 — `march_cartesian` (the working recipe)
- src/world/gpu/pack.rs:192-230 — pack semantics for `tag=1` (uniform-flatten) vs `tag=2` (non-uniform with `representative`)
