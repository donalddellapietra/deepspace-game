# Sphere shell → Cartesian gnomonic — rewrite proposal

## Context

The current sphere rendering uses equal-angle cubed-sphere geometry.
Each face subtree is a 27-way branching tree in `(u, v, r)` coords.
The shader's `sphere_in_cell` walks this tree via a DDA that computes
ray-plane intersections using plane normals derived from `ea_to_cube()`
(i.e., `tan(c · π/4)`). This is numerically stable down to ~d=9 but
breaks at d≥10:

- Plane normals for adjacent cells differ by ~1e-5 in the n-axis
  component. For rays at grazing angles, `t_u_lo / t_v_lo / t_r_lo`
  are numerically close; `argmin` flips unstably across adjacent
  pixel rays.
- Visible symptom: mode-4 stripes on the ground + "hollow" placed
  blocks. Documented in `docs/spheres/d10-hollow-block-debug-state.md`.

Every fix that stays inside the equal-angle mapping (pinning
pick_face, quantizing cell bounds, forcing max walker depth,
disabling pack flatten, reformulating the plane normals) has been
tried in worktree `sphere-attempt-2-2-3-2` and rejected. The bug is
intrinsic to evaluating trig in the DDA's inner loop at that scale.

Problem gets worse — not just d=10 but any deeper depth would break
similarly.

## Scope simplification

We only need a **shell** planet, not a solid body down to the core.
Players cannot dig through to the other side; the inner shell is the
floor. This removes "how do rays behave in the hollow interior" as
a concern and lets the crust be bounded.

## Proposal

**Replace the equal-angle + ray-plane DDA with a Cartesian DDA in
gnomonic face-local `(u, v, r)` coordinates.**

Each cube face becomes a flat Cartesian body with:

- `u, v ∈ [0, 1)` across the face (linear fractions — gnomonic).
- `r ∈ [0, 1)` across the shell thickness (linear between inner_r
  and outer_r).

Cell boundaries are **axis-aligned in `(u, v, r)`**: at every depth,
cell bounds are exact rationals `(k/3^d, (k+1)/3^d)` on each axis.
No trig anywhere in the DDA.

The sphere's curvature only appears at two points:
1. **Ray entry**: convert world-ray to face-local via gnomonic
   projection (ratio `x/y`, `z/y` for PosY, etc. — linear, no trig).
2. **Shading**: convert hit `(u, v, r)` back to world-XYZ for the
   hit position, and compute the outward radial direction for
   surface lighting. `cubesphere::face_space_to_body_point` already
   does this; reuse unchanged.

### Why this solves BOTH stripes AND d=30+

- **Stripes at d=10**: gone because the DDA is bit-exact integer
  slot-pick in gnomonic face-local coords. No trig in the hot path.
- **d=30+ depth support**: preserved because the DDA descends in
  frame-local coords (same technique `march_cartesian` uses today).
  Each descent rescales `[0, 1)³` of the current cell, keeping all
  values O(1) regardless of absolute depth.

The two failure modes are independent, and this rewrite fixes both
in one stroke by reusing the already-working `march_cartesian`
infrastructure.

### Trade-off: cell angular uniformity

Equal-angle cubed-sphere (current): every cell subtends the same
solid angle on the sphere. Visually uniform at all face positions.

Gnomonic (proposed): cells at face corners subtend ~2× the solid
angle of cells at face center. That's "visible" in principle but
cosmetic at typical close-range gameplay — the view never sees
the whole face at once, and the local cell sizes in a given view
are similar.

Given the current architecture flat-out does not render d≥10
correctly, cosmetic non-uniformity is a cheap price.

## What changes in code

### WGSL (`assets/shaders/sphere.wgsl`)

Replace `sphere_in_cell`'s body entirely. The new implementation:

1. **Find which cube face the ray enters**. Same `pick_face(n)` or
   a ray-box intersection against each of 6 faces.
2. **Transform ray to face-local gnomonic coords**. For PosY face:
   `u(t) = (ray_origin.x + ray_dir.x · t) / (ray_origin.y + ray_dir.y · t)`
   `v(t) = (ray_origin.z + ray_dir.z · t) / (ray_origin.y + ray_dir.y · t)`
   `r(t) = (|ray(t)| − inner_r) / (outer_r − inner_r)`
   These are rational functions of `t` — solvable in closed form for
   `(u, v, r) = k/3^d` crossings.

3. **Run 3D Cartesian DDA in face-local space**. This is the same
   shape as `march_cartesian`:
   - Track current cell `(u_cell, v_cell, r_cell)` as u32 slot
     indices at the current frame depth.
   - At each step, compute `t_next_u / t_next_v / t_next_r` for the
     next axis boundary and take the min.
   - Walker descent on non-empty children uses the EXISTING integer
     slot-pick walker (`walk_face_subtree`) — it already works.
   - On empty cell, step to next boundary.
4. **Exit conditions**:
   - `r_cell` leaves `[0, 1)`: ray exits the shell. Terminate.
   - `u_cell` or `v_cell` leaves `[0, 1)`: ray crosses to adjacent
     face. Transition to that face's subtree and restart the DDA
     with the reflected `(u, v)` in the new face's coord system.

5. **Hit shading**: on non-empty cell, compute the hit position in
   world coords via `face_space_to_body_point`, use the outward
   radial direction as the surface normal (smooth sphere normal).
   For placed blocks where per-face cube shading matters, pick the
   normal from `last_side` (the axis we crossed to enter the hit
   cell) — but in face-local space these are `u`/`v`/`r` axes, not
   the current world-space `u_axis`/`v_axis`/`n_axis`.

### Tree layer (no changes expected)

The tree stores content in `(u, v, r)` subtrees already. No
structural change. `cs_raycast` (CPU mirror) similarly rewritten but
tests continue to use the same tree API.

### Out of scope for first iteration

- Face-to-face transition at cube edges: implementable but complex.
  Start with "ray terminates on lateral exit" — renders as "shell
  edge" (acceptable for north-pole scene). Add edge-crossing in a
  second pass.
- Ribbon / LOD representative changes: should work unchanged because
  the walker output format doesn't change.

## Estimated scope

- New WGSL: ~200 LoC to replace `sphere_in_cell`.
- Deleted WGSL: ~350 LoC (the equal-angle plane-math, ray-plane
  intersections, bevel-layered shading that relies on `cube_to_ea`).
- CPU mirror in `src/world/raycast/sphere.rs` follows the same
  shape: ~250 new LoC, ~400 deleted.
- Existing `walk_face_subtree` reused as-is.
- Existing tests continue to pass on the tree layer; new shader
  validation via `shader_compose::tests::naga_validates_main_shader`.

## Migration sequence

One big commit. No shim layer, no transitional representation —
per the "no intermediate visual states" memory:

1. Replace `sphere_in_cell` with the new gnomonic Cartesian body.
2. Replace `cs_raycast` CPU mirror.
3. Update `cubesphere::body_point_to_face_space` to use gnomonic
   (removes the `atan` / `cube_to_ea` chain; just `x/y`, `z/y`).
4. Remove `ea_to_cube` / `cube_to_ea` from all call sites.
5. Run: `cargo test --lib`, `scripts/repro-sphere-d10-bug.sh`.
6. Verify the d=10 case visually matches a d=8 case (both clean,
   no stripes).

## Risk

The gnomonic mapping changes the **on-sphere position of every
existing voxel** by the cube-corner distortion factor. For a new
sphere world this is fine. For PERSISTED worlds with content, the
content would shift — but sphere worlds aren't persisted yet in
this project (only demo_sphere with procedural terrain).

## Success criterion

`scripts/repro-sphere-d10-bug.sh 0 4`:
- Mode 0 after-place screenshot: **no horizontal stripes on the
  ground**, placed block renders as a recognizable shape.
- Mode 4 after-place screenshot: winning plane is either uniform
  r_lo or a small set of sibling cells — no whole-screen striping.

Both at `--spawn-xyz 1.5 1.7993 1.4988 --spawn-depth 10` (the user's
bug repro) AND at other spawn positions.
