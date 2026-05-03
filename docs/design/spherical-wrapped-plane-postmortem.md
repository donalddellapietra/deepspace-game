# SphericalWrappedPlane — failed attempt postmortem

**Status:** abandoned. Working tree restored to the dodecahedron-test
commit (`1987893`). Sphere code (`src/world/bootstrap/spherical_wrapped_planet.rs`,
`src/world/raycast/spherical_wrapped_planet.rs`, `NodeKind::SphericalWrappedPlane`,
shader `march_spherical_wrapped_plane`, `cell_offset` field on TBs)
all reverted. The dodecahedron-test preset survives because it uses
TBs at fixed slot positions — no sphere tiling required.

This document records what was tried, what worked briefly, and the
specific reasons the design didn't converge — for the next attempt.

## Goal

A UV sphere built from per-cell `TangentBlock` rotations, with the
existing flat `WrappedPlane` slab as ground truth. Apply the lessons
from the TB-rotation work: stored rotations (not recomputed per
fragment), cell-local arithmetic, single boundary rule applied
symmetrically across shader / CPU raycast / anchor descent.

## What was attempted (commits `074636ea..c53752b`)

### 1. `cell_offset` field on `NodeKind::TangentBlock` (commits `083d918`, `9a00989`)

Extended TB with a per-cell displacement from natural slot centre,
in parent-frame `[0, 3)³` units. Plumbed through `TbBoundary`,
`GpuNodeKind`, WGSL `NodeKindGpu`, every TB-boundary site (shader
TB child dispatch, ribbon pop, CPU raycast TB sites, anchor
descend / pop, zoom_in / zoom_out, in_frame_rot). With default
zero, no behaviour change.

### 2. SphericalWrappedPlane node kind + sphere DDA shader (`ed002cd`)

Added `NodeKind::SphericalWrappedPlane { dims, slab_depth, body_radius_cells, lat_max }`,
`GpuNodeKind` packing for kind=3, WGSL `NodeKindGpu` mirror, and
`march_spherical_wrapped_plane` shader function: ray-vs-sphere,
walk shells, find cell at `(lon_idx, lat_idx, r_idx)` via
`sample_slab_cell`, dispatch into cell's TB.

### 3. Bootstrap (`563f8d7`, then re-pointed to use new kind)

`spherical_wrapped_planet_world` builds the standard `[27, 2, 14]`
slab but with each cell wrapped in a `TangentBlock` carrying:
- `rotation = R_y(lon_c) · R_x(-lat_c)` — tangent basis at cell centre.
- `cell_offset = (sphere_pos − natural_centre) / cell_size_wp` — moves
  the cell from its flat slab position onto the sphere surface.

### 4. Render-frame discipline (`9f2743d`)

`path_is_strict_descendant_of_spherical_wrapped_plane` predicate +
truncate-while-descendant rule: render frame must be at-or-above
the SphericalWP, never below (otherwise a Cartesian descendant
would dispatch `march_cartesian` against the sphere-positioned
cells and miss most of them).

### 5. CPU raycast mirror (`db1ad98`, `7a5348d`)

`cpu_raycast_spherical_wrapped_planet` mirroring the shader's DDA.
Required tracking the displaced-vs-not distinction in `TbBoundary`
because for displaced cells the shader uses `tb_scale = 1` (no
inscribed-cube shrink) while `TbBoundary::from_kind` always applies
the shrink. Worked around by inlining the rotation in the CPU
sphere raycast.

## What worked

Briefly, at commit `7a5348d`, the sphere rendered as a clear solid
sphere with visible grass / dirt / stone surfaces. CPU raycast
fired correctly enough to break some cells on the sphere.

## Why it broke down

The fundamental issue is geometric:

> **Tangent cubes do not tile a sphere without overlap or gaps.**

A unit cube's spatial diagonal has length √3. Adjacent cells on the
sphere are spaced one cell-arc-length apart (≈ cell_size). Two
adjacent rotated cubes' AABBs each extend `cell_size · max_axis_extent`
≈ `cell_size · √2` in render frame. They **always overlap** in
their AABBs, geometrically.

Three options were tried, each with a fatal limitation:

### Option A: Inscribed-cube shrink (`tb_scale = 1/√max_extent`)

The default shrink that makes the rotated cube fit inside its
slot's `[0, 3)³`. Applied to displaced cells: the rotated cube
shrinks to ~71% of `cell_size` along each axis. Adjacent cells'
AABBs are now tangent (no overlap) but their CONTENT cubes leave
~30% gaps between them on the sphere surface.

**Visual:** sphere appears as scattered patches with sky between cells.
The user explicitly rejected this: *"the sphere is full of holes,
mostly sky between visible cells."*

### Option B: No shrink (`tb_scale = 1`), no AABB compensation

Cubes render at full `cell_size` extent. Adjacent cubes' AABBs
overlap by ~30% in render. `march_in_tangent_cube`'s storage `[0, 3)³`
clips at AABB corners, leaving triangular cuts on each cube's
surface.

**Visual:** sphere covered in triangular fragments. The user's
*"there is something fundamentally just wrong about this — the
blocks are moving in space and are cut off by their edges"*.

### Option C: AABB enlargement to fit rotated cube (`aabb_factor = 1/inscribed_scale`)

Enlarge each cell's render-frame AABB to fully contain the rotated
cube. Cubes render solidly. But adjacent cells' AABBs now overlap
by 40%+ in render. The first-hit DDA returns the closest cube's
content. When a cell is BROKEN, adjacent cells' content protrudes
into the broken cell's region — the user can't see what's underneath.

**Visual:** sphere renders solidly but breaking cells doesn't
expose interior content. The user's *"the exterior of the sphere
interferes with and covers up the internal structure of the sphere"*.

### Option D: Sphere-cell mask (Option C + reject hits in neighbours' bins)

After the inner DDA returns a hit, recompute the hit's spherical
coords and reject if `(cx, cy, cz)` of the hit doesn't match the
cell we tested. This correctly identifies overlap regions and
prevents adjacent cells from covering broken cells.

**Visual:** small gaps reappear at cell-edge overlap regions —
because both cells reject the same overlap region as "not mine",
and the geometric reality is that no cell's content covers that
exact arc-difference. The user noted this still left visible
fragmentation.

## Other concrete bugs encountered along the way

Worth recording so they're not re-derived:

- **Bootstrap r_c was outside body shell.** Original formula
  `r_c = body_radius + (r_idx + 0.5)·cell_size` placed cells
  OUTSIDE the body shell. The shader's DDA expects them inside
  `[body_radius − N_r·cell_size, body_radius]`. Cells rendered but
  at world positions inconsistent with the ray's spherical hit,
  making blocks appear to drift as the camera moved. Fixed in
  `9cfc390` to `r_c = body_radius − (N_r − r_idx − 0.5)·cell_size`.

- **Lon convention mismatch.** Bootstrap originally used
  `lon_c = (lon_idx + 0.5) · 2π/N_lng` (cell 0 at +X axis) while
  the shader's `u = (lon + π) / (2π)` puts cell 0 at −X. Fixed in
  `ed002cd` by aligning the bootstrap to the shader.

- **`TbBoundary::new` always applies inscribed shrink.** GPU
  packing uses `tb_scale = 1` for displaced cells (cell_offset != 0)
  but `TbBoundary::from_kind` and `TbBoundary::new` always compute
  `tb_scale = inscribed_cube_scale(R)`. CPU raycast mismatched the
  shader for displaced cells. Worked around by inlining rotation
  in the sphere CPU raycast.

- **Render frame drifting below SphericalWP.** Default truncate
  rule put the render frame at `anchor.depth() − K`. As the camera
  zoomed in, the render frame descended into Cartesian descendants
  of SphericalWP, where `march_cartesian` ran against
  sphere-positioned cells and missed most of them. Required a
  second `truncate-while-strict-descendant-of-SphericalWP` rule on
  top of the existing `truncate-while-TB`.

## Lessons for the next attempt

1. **Tangent cubes ≠ sphere tiles.** The cube approximation has
   inherent overlap / gap tradeoffs that no amount of shader
   massaging removes. A real sphere primitive would need spherical
   patches (curved tile geometry) or fundamentally different
   storage (e.g., subdivided icosahedron with per-face Cartesian
   trees, like a cubed sphere but icosahedral).

2. **Storage layout vs render layout.** Storing cells in a flat
   `[27, 2, 14]` slab and *positioning* them on a sphere via
   `cell_offset` creates a permanent CPU/GPU coordinate mismatch
   that has to be re-validated at every API boundary. The slot
   DDA, the sphere DDA, the anchor descent, and the CPU raycast
   all need to agree on what world space a cell occupies. Each
   surface is a place where the agreement can quietly break.

3. **`cell_offset` was the wrong primitive.** It tries to
   generalise TB by adding translation, but TB's `R^T·(p − pivot)/scale`
   rule assumes the cube fits inside the parent slot's
   axis-aligned bound. Cells displaced to sphere positions have
   no axis-aligned-bound semantics — they only have spherical
   neighbour relationships.

4. **The sphere-mercator-1 reference was right about one thing.**
   It computed per-fragment tangent bases from `(lon_c, lat_c)` —
   "render-time reinterpretation". We rejected that as the BS2
   anti-pattern. But once we tried storing rotations on cells,
   we ran into the AABB problem, and the only way out was a
   sphere-cell mask that re-derives the spherical region anyway.
   The render-time reinterpretation might actually be the right
   approach for this geometry; the precision wall it hits is at
   deep zoom, and a depth-bounded sphere subframe works around
   that. The sphere-mercator-1 work has unfinished precision
   investigations that may yet bear fruit.

## What survives this branch

- `dodecahedron_test` preset: 12 fixed-position TBs at cube-edge
  slots, each rotated to a regular-dodecahedron face normal.
  Validates the TB rotation primitive against twelve distinct
  non-axis-aligned rotations without requiring sphere tiling.
- `rotated_cube_test` preset: single TB primitive.
- All TB infrastructure pre-`cell_offset`: `TbBoundary`,
  `frame_path_chain`, the shader's TB child dispatch + ribbon
  pop, the CPU mirrors, the truncate-while-TB render-frame rule.
- The `sphere-mercator-1` worktree (separate branch) for
  render-time-reinterpretation reference.
