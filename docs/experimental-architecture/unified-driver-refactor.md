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

Every cell occupies its own local coordinate frame. The DDA loop
structure is uniform — same stack, same descend/ascend mechanism,
same outer flow — but the **boundary test inside each step is
parameterized by NodeKind**. Cartesian cells use planar boundary
tests on all three axes. CubedSphereFace cells use planar tests on
u/v (this is what the current code already does — `n_u_lo` is a
plane through the body center) and analytical ray-vs-sphere on the
r-axis (also what the current code already does, via
`ray_sphere_after`). On cell entry, NodeKind metadata sets up the
local frame: Cartesian uses cell origin + size; face uses the
existing `face_*_axis` orientation plus inner_r/outer_r. On descend,
the ray rebases into the child's local frame via integer arithmetic
(multiply by 3, integer offset for Cartesian; the same plus axis
remap on face seams). On exit, pop to parent and continue. The
"virtual root" is just the deepest cell containing the camera —
same DDA loop, just a different starting frame. Cell-local coords
keep f32 precision exact at any tree depth.

**What "unified" means precisely:** one loop, one stack, one
descend/ascend protocol. Kind-specific behavior is metadata +
small switches inside the boundary-test step. This is structurally
different from today's two parallel routines (each with its own
stack management, entry/exit handling, and outer flow); it's not
"one identical boundary primitive for every cell shape" — that
would require linearizing curved cells, which is unnecessary
because today's planar/spherical mix already works.

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

| Kind | Cell entry setup | Boundary test (u, v, r) | Descend / Ascend |
|------|------------------|--------------------------|------------------|
| Cartesian | origin, size | planar / planar / planar | `local * 3 - integer_slot * 2` per axis |
| CubedSphereBody | analytical ray-vs-cs_outer sphere → entry face | n/a (single layer dispatching to face) | descend into face slot |
| CubedSphereFace | face_axis vectors + inner_r, outer_r, current u/v cell EA bounds | planar / planar / **spherical (`ray_sphere_after`)** | `local * 3 + (2 - slot * 2)` for u/v; ditto for r in normalized shell coords |

Notes:

- **Planar u/v boundaries on face cells are not new.** The current
  shader already constructs them via `n_u_lo = u_axis -
  ea_to_cube(u_lo_ea) * n_axis` — a plane through the body center
  that intersects the sphere as a great-circle-like arc. The unified
  DDA reuses this construction; what changes is that `u_lo_ea` comes
  from cell-local descent state (precision-safe at any depth)
  instead of `cells_d = pow(3, depth)` quantization.
- **Spherical r-axis is a per-NodeKind specialization, not a
  weakness.** The current shader uses `ray_sphere_after` for the
  r-boundaries of face cells; the unified loop keeps this — the
  boundary-test step branches on NodeKind to pick planar vs.
  spherical for each axis. This is the kind-aware behavior NodeKind
  exists to express.
- **No global Jacobian is needed.** Earlier drafts proposed a 3×3
  per-cell Jacobian to linearize the curved face geometry into a
  pure box DDA. On reflection that linearization isn't necessary:
  the boundary tests for face cells already work in world coords
  with the existing analytical math. The win from cell-local frames
  is precision (cell bounds expressed via local descent state, not
  3^d quantization), not geometric simplification.

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

- **Face axis sign conventions.** The 6 faces have specific
  orientations baked into `face_normal`, `face_u_axis`, `face_v_axis`
  in the current shader. The unified loop reuses these; getting them
  wrong renders a face inverted. Mitigation: keep the existing
  axis-vector functions verbatim and verify in Rust with finite-
  difference checks against sample world positions before porting
  any new code.
- **Face-seam transitions.** When a ray walks across the surface and
  hits a cube edge, the unified DDA must pop out of the current face
  cell and push into the neighbor face's frame. The 24-case face
  transition table (6 faces × 4 edges) needs to be in WGSL. Existing
  Rust `cubesphere::face_uv_to_dir` has the math; it just needs to
  be ported as a switch-case shader function. Risk: bugs at seams
  show as thin artifact lines along cube edges of the planet. This
  is the same special case the current code handles implicitly by
  re-running `pick_face` every step; the unified version makes it
  explicit.
- **Sphere outer-shell exit.** Same special case the current code
  has. Today, `sphere_in_cell` returns "miss" when the ray exits
  cs_outer outward, and the caller advances Cartesian DDA past the
  body cell. The unified loop does the same: r-axis exit at outer
  boundary → pop face frame → pop body frame → continue Cartesian
  DDA in the body's parent. Risk: getting the t-parameter handoff
  right (ray continues from exit point, doesn't restart).
- **Precision at the absolute root.** When the ancestor ribbon's
  parent_ascend reaches absolute-root scale, the local coords inflate
  by 3^(ancestor depth). f32 still represents them fine for sky/
  horizon picking — at that scale we're not doing cell-precise hits,
  just direction-based color blends.
- **Performance — Cartesian-content overhead.** Today's pure
  Cartesian DDA step is ~10 ops. Unified adds NodeKind read + branch
  before the boundary test; estimate ~15 ops per step (~50% overhead
  for pure Cartesian content). Acceptable trade for the precision
  win, but verify with `test_runner` during step 7. For sphere
  content the unified loop should be net-positive vs. today (we
  eliminate the giant per-step `sphere_in_cell` setup).
- **Stack pressure.** Per-frame state grows from ~44 to ~100 bytes
  (NodeKind discriminant + face id + radii + local oc/dir cached at
  cell entry). 16 frames × 100 = 1600 bytes per pixel. Within
  typical GPU per-thread budgets (~4 KB). Verify no register spill
  in the wgpu-naga compile stats.

## Issues that aren't actually issues

These came up during scope review and were rejected after closer
inspection. Documenting so they don't recur:

- **"Linearization at shallow face cells will facet the planet."**
  No: the planet's silhouette comes from analytical `cs_outer`
  sphere intersection, not from u/v cell boundaries. The current
  code already uses planar u/v boundaries (great-circle-like arcs
  through the body center); the unified loop reuses them.
- **"Stack budget for visual depth + ancestor chain overflows."**
  No: ancestor pop is sequential, not cumulative. Peak stack depth
  equals max single-descent path, ~16.
- **"3×3 per-cell Jacobian needed for face cells."** No: not needed
  if we keep the existing analytical boundary tests. The Jacobian
  was an early sketch for a fully-linear box DDA, which isn't the
  design we're shipping.
- **"CPU/GPU drift risk from dual-implementing transforms."** No
  worse than today; current code has the same dual implementation
  for both Cartesian and sphere math.

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
