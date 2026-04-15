# Unified DDA Refactor — Scope

The renderer currently has two descent loops that exist for historical
reasons, not architectural ones: a Cartesian DDA stack and a sphere-face
DDA. They share a tree, a buffer, and a NodeKind dispatch — but the
inner loops are duplicated and operate in different precision regimes.

This refactor collapses them into **one cell-local DDA** parameterized
by per-NodeKind transform metadata. The virtual-root precision fix
falls out for free, the two-stack/two-budget split disappears, and
layer-1 zoom works at any tree depth.

This is the architecture the engine has been trying to be since the
anchor refactor; we just haven't paid the cost to get there.

## Core idea (one paragraph)

Every cell, regardless of NodeKind, is a unit cube `[0, 3)³` in its
own local frame. The DDA loop only ever marches through unit cubes.
On cell entry, the current NodeKind's transform is applied: for
Cartesian, identity; for CubedSphereBody/Face, a Jacobian computed
once at the cell's center that maps world-space ray direction into
local (u_ea, v_ea, r_n) coords. Inside the cell, the DDA primitive
is identical: integer cell boundaries, exact at any depth. On
descend, the ray rebases into the child's local frame (multiply by
3, integer offset). On exit, pop to parent and continue. The
"virtual root" is just the deepest cell containing the camera —
same DDA, just a different starting frame.

## What lives where in the new design

### Per-NodeKind transform (new)

For each NodeKind, a small bundle of math that answers:

- **`world_to_local(world_pos, cell_meta) → local_pos`** — used once
  on cell entry to find where the ray currently sits in the cell.
- **`local_dir(world_dir, cell_meta) → local_dir`** — Jacobian-applied
  ray direction. For Cartesian this is just `world_dir / cell_size`.
  For face cells it's the per-cell linearization of the EA projection.
- **`child_descend(local_pos, local_dir, slot) → (new_local_pos,
  new_local_dir)`** — rebase coords into child's frame. Pure integer
  arithmetic for all kinds (multiply by 3, subtract integer offset).
- **`parent_ascend(local_pos, local_dir, slot_in_parent) → (...)`** —
  inverse of descend. Used on cell exit when the ray pops upward.

### Unified DDA loop (rewritten)

State per stack level:

```wgsl
struct CellFrame {
    node_idx: u32,        // tree node at this level
    kind: u32,            // NodeKind discriminant (cached)
    cell: vec3<i32>,      // current 3×3×3 sub-cell, [0..2]
    local_oc: vec3<f32>,  // ray origin in local coords
    local_dir: vec3<f32>, // ray direction in local coords (linearized for non-Cartesian)
    side_dist: vec3<f32>, // DDA stepping state
    transform_meta: vec4<f32>, // per-kind extras (face id, inner_r, etc.)
}

var stack: array<CellFrame, MAX_VISUAL_DEPTH>;
var depth: u32 = 0u;
```

Loop body:

```
1. Look up child at current cell via slot index.
2. If terminal (Block/Empty): handle hit or empty-step.
3. If Node child:
     a. Read child's NodeKind from node_kinds buffer.
     b. Decide LOD: child sub-pixel → flatten to representative block.
     c. Else descend: rebase coords via NodeKind's child_descend, push frame.
4. If cell exits [0, 3)³: pop frame (parent_ascend). If at depth 0: ray miss.
5. Else: DDA-step within current cell (side_dist comparison, increment cell index).
```

One loop. One stack. NodeKind dispatch only inside steps 3a/3c (cell
descent transform) and the initial frame setup. Everything else is
NodeKind-agnostic.

### Virtual root selection (new CPU code)

`render_frame()` already exists and is sphere-aware-able. New behavior:

- Walk camera's path from absolute root.
- Stop at the **deepest cell whose 3 cell-widths in world units still
  comfortably contain the visible scene** (~= camera's anchor minus a
  small K, capped by ancestor-ribbon length we're willing to pack).
- This cell becomes the virtual root. NodeKind doesn't matter — the
  unified DDA handles any kind as a starting frame.

### Ancestor ribbon (new GPU + CPU code)

For rays that exit the virtual root (looking at the sky/horizon):

- CPU packs the chain `virtual_root → absolute_root` as a sequence of
  parent-frame entries: each ancestor's NodeKind, its slot containing
  the descendant, plus its 26 sibling children (LOD-flattened to
  representative blocks).
- Shader pops via `parent_ascend`, continues DDA in the parent's
  frame. At most ~63 pops to reach absolute root.
- Sky path triggers when a ray exits the absolute root.

### NodeKind transform implementations

| Kind | `world_to_local` | `local_dir` Jacobian | `child_descend` |
|------|------------------|----------------------|-----------------|
| Cartesian | `(world - origin) / size` | `world_dir / size` | trivial: `local * 3 - integer_slot * 2` |
| CubedSphereBody | dispatch on ray-shell intersection → which face | based on entry face's axes | trivial within a face slot |
| CubedSphereFace | EA projection with per-cell Jacobian linearization | analytical Jacobian of EA at cell center | `local * 3 + (2 - slot * 2)` per axis |

The Jacobian for CubedSphereFace is the only nontrivial math. It's a
3×3 matrix of partial derivatives of `(world_x, world_y, world_z)`
w.r.t. `(u_ea, v_ea, r_n)`, evaluated at the cell's center. The
inverse Jacobian sends world ray direction into local. Curvature
error within one cell is `O(cell_size²)` — invisible at any depth.

## What goes away

- `walk_face_subtree` — replaced by unified DDA descending into face
  cells via the same code path as Cartesian.
- `sphere_in_cell` — replaced by NodeKind transform metadata + entry
  setup; no separate routine.
- `MAX_STACK_DEPTH` (16) and the hardcoded `d <= 22u` face loop bound
  — replaced by `MAX_VISUAL_DEPTH` (~16, screen-resolution bounded).
- `cs_edit_depth` clamp `[1, 14]` — gone, no precision wall.
- `RENDER_FRAME_MAX_DEPTH = 0` — gone, frame can lift to any depth.
- The "two precision regimes" (Cartesian camera coords, face EA
  cells_d quantization) — gone, all coords are cell-local f32.

## Files touched

- `assets/shaders/ray_march.wgsl` — full DDA rewrite (~600 → ~700
  lines). New stack frame struct, unified loop, NodeKind transform
  helpers, ancestor pop. Net biggest change.
- `src/world/gpu.rs` — pack ancestor ribbon, emit per-cell transform
  metadata in `node_kinds` buffer if needed.
- `src/world/anchor.rs` — virtual-root selection: `render_frame()`
  picks deepest cell containing camera, no NodeKind restriction.
- `src/app/mod.rs` — lift `RENDER_FRAME_MAX_DEPTH`, simplify camera
  position handoff.
- `src/app/edit_actions.rs` — drop `cs_edit_depth` clamp; CPU edit
  raycast uses unified path-walk.
- `src/world/edit.rs` — `cpu_raycast` mirrors the unified DDA in
  Rust: same NodeKind dispatch, same cell-local descent. Cursor
  highlight derives from CPU walk.
- `src/world/spherical_worldgen.rs` — lift `PlanetSetup.depth` cap
  if any; planets can now have arbitrarily deep face subtrees.
- `src/renderer.rs` — uniforms reshuffle: `MAX_STACK_DEPTH` constant
  → `MAX_VISUAL_DEPTH`, ancestor ribbon size, etc.

## Order of work (within the single diff)

1. **CPU virtual-root packer.** Pack the camera's deepest-cell
   subtree + ancestor ribbon into the GPU buffer. Test offline that
   the pack is correct (BFS traversal counts, no orphans).
2. **Shader unified DDA — Cartesian-only first.** Rewrite `march()`
   with the new stack frame and unified loop, but only the Cartesian
   transform path wired up. Spheres render as black during this phase
   — that's the "intermediate broken state" the diff goes through
   internally, but the diff doesn't ship until step 4 lands.
3. **Shader CubedSphereBody + Face transforms.** Add the Jacobian
   math, NodeKind dispatch in cell-entry setup. Spheres render again.
4. **CPU edit raycast mirror.** Update `cpu_raycast` to use the same
   unified descent. Edits land at any depth without a clamp.
5. **Ancestor pop in shader.** Wire up `parent_ascend` for rays
   exiting the virtual root upward. Sky path fires when ray exits
   absolute root.
6. **Cap removal pass.** Delete `cs_edit_depth` clamp,
   `RENDER_FRAME_MAX_DEPTH`, `MAX_STACK_DEPTH`, hardcoded `22u`. Set
   `MAX_VISUAL_DEPTH = 16`.
7. **Test pass.** Boot game. Visual smoke: spawn looks correct, zoom
   to layer 1, edit at deep zoom, look at sky, look at horizon.
   Run `cargo check`, run existing test suite.

This is a single diff per your "no intermediate visual states" rule.
The broken-internally states in steps 2–3 are within the diff, never
on a commit.

## Risks

- **Jacobian sign conventions.** The 6-face EA projection has 6 sets
  of axis sign/swap conventions. Getting one wrong renders a face
  upside-down or mirrored. Mitigation: derive the Jacobian by
  symbolic differentiation of the existing `face_*_axis` functions in
  the shader, write a unit test in Rust that compares finite-
  difference world position vs. analytical Jacobian at sample points.
- **Face-seam transitions.** When a ray walks across the surface and
  hits a cube edge, the unified DDA must pop out of the current face
  cell and push into the neighbor face's frame. The 24-case face
  transition table (6 faces × 4 edges) needs to be in WGSL. Existing
  Rust `cubesphere::face_uv_to_dir` has the math; it just needs to
  be ported as a switch-case shader function. Risk: bugs at seams
  show as thin artifact lines.
- **Sphere outer-shell exit.** When a ray inside the sphere exits
  outward, it pops back to the body's parent (Cartesian) frame and
  continues there. The body's bounding cell IS the parent's child;
  the ray is already in that frame's coords post-pop. Risk: getting
  the t-parameter handoff right (ray didn't restart, it continued).
- **Precision at the absolute root.** When the ancestor ribbon's
  parent_ascend reaches absolute-root scale (~3^23 of virtual-root
  units), the local coords get large. f32 still represents them
  fine, but precision degrades. For sky/horizon rays this is
  acceptable — at that scale we're picking a sky-color blend, not
  cell-precise hits.
- **Performance.** Per-cell transform compute is O(1) but adds a few
  dozen ops per cell entry vs today's hardcoded Cartesian step.
  Likely net-positive because we eliminate the giant
  `sphere_in_cell` analytical plane intersections. To verify with
  the existing test_runner harness during step 7.

## Estimated size

- Shader: ~600 lines net new (rewriting ~400 of the existing 672).
- CPU pack: ~150 lines (ancestor ribbon + virtual root walker).
- CPU raycast mirror: ~100 lines (refactor of `cpu_raycast`).
- Tests: ~50 lines (Jacobian sanity checks, virtual-root pack
  invariants).
- Total diff: ~1000 lines added, ~800 deleted, single commit.

## Test plan

- **Unit (Rust).** Jacobian symbolic vs. finite-difference at sample
  face points across all 6 faces. Virtual-root pack: total node
  count, no orphans, ancestor ribbon length matches camera depth.
- **CPU mirror.** `cpu_raycast` returns same cell at same depth as
  visual ray for known-good test scenes (bench against the current
  raycast on shallow scenes that both can handle).
- **Visual smoke.** Boot the game in headless via the existing
  `test_runner` harness with `--timeout-secs 5`. Snapshot at spawn,
  zoom to layer 5, layer 10, layer 1. Compare against current
  baseline at layers where current code works.
- **Edit smoke.** Place and break at layer 1. Highlight at deep zoom
  follows cursor. Saved meshes still load/place.

## What this does NOT do

- Multi-planet rendering. The architecture supports it for free
  (planets are just nodes), but this diff doesn't add planets to
  worldgen. That's a separate worldgen change.
- Physics / collision at arbitrary depth. Player physics still uses
  the existing path machinery; the rendering refactor doesn't touch
  collision.
- New NodeKinds (cylinders, icosahedra, etc.). The architecture
  supports them as drop-in transform implementations. Not in scope.

## When to start

When you're ready to commit a focused session of ~half a day of
implementation + ~couple hours of visual debugging on the Jacobian
and seam transitions. This is one of the bigger diffs in the
engine's history; it's the right one to take, but it deserves a
session that isn't time-boxed.
