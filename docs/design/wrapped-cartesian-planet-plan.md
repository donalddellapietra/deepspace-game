# Wrapped Cartesian Planet — High-Level Implementation Plan

This is the coordinator's high-level plan. Phase 0a will refine this into a
detailed implementation plan with file:line references and concrete diffs.

## Goal

Player in space → travels to planet surface → looks like a planet at every
altitude. No atmosphere. Pole handling required. Reference behavior: when you
spawn a cube in space and travel to it, it just works. The planet should match
that, end-to-end.

## Architecture in one paragraph

A planet is a flat Cartesian voxel slab embedded as ordinary children inside a
27³ anchor cell. Surrounding cells in that 27³ are empty (sparse occupancy =
unset bits — no new "banned" primitive needed). The slab dimensions are roughly
2:1:shallow (e.g. 20×10×2 cells at depth 20 inside a 27³ at depth 22). The X
axis (longitude) wraps. The two end-rows in Y (latitude) are non-buildable
polar strips. Curvature is applied as render-time per-step ray bending
parameterized by camera altitude. Storage and simulation stay flat Cartesian
forever; only the marcher sees curvature.

## Why this architecture (the three problems it eliminates)

1. **Seams.** Every interior cell has exactly 6 neighbors. X-wrap is
   topologically clean (column 0 ↔ column W−1). Polar strips are banned, not
   seamed.
2. **Deep-depth scalability.** The slab is a Cartesian subtree. It inherits
   the WorldPos anchor+offset precision discipline that already renders
   correctly at arbitrary depth. No f32 absolute coordinates anywhere in the
   data path. No equal-angle remap, no face-space coords.
3. **Stretched non-90° voxels.** Voxels are perfect axis-aligned cubes at
   every depth. Curvature lives in the ray, not the cell.

## Phase plan

### Phase 0 — Setup (initial opus agents)

- **0a:** Detailed implementation plan refining this doc — file:line refs,
  concrete changes per phase, full test plan.
- **0b:** Remove legacy cubed-sphere code (cubesphere.rs, raycast/sphere.rs,
  NodeKind::CubedSphere* variants, sphere shaders, in_face_subtree pack
  gating, sphere_in_cell shader dispatch, SphereFrame, body_local AABB).
- **0c:** Port the visual harness, silhouette/curvature image-analysis tests,
  altitude-stepped screenshot tooling from sphere-attempt-2-2-3-2.

### Phase 1 — Slab as Cartesian content (no wrap, no curvature)

- 1a: `NodeKind::WrappedPlane { dims: [u32; 3] }` (or wrap-metadata on a
  Cartesian root — implementer decides).
- 1b: Worldgen produces a hardcoded 20×10×2 slab inside an empty 27³.
- 1c: Renders as a flat patch in space. Visual harness screenshots top-down
  and edge-on.
- 1d: Walking onto the slab from spawn behaves like walking onto any
  Cartesian region.

### Phase 2 — X-wrap (full implementation, single bundled commit)

- 2a: Wrap-aware `step_neighbor_cartesian` at slab root depth.
- 2b: Wrap-aware shader marcher pop on X-exit at slab root depth.
- 2c: Validation: walk east → return west; ray launched east hits west side.

### Phase 3 — Render-time curvature (coordinator-owned)

- 3a: `k(altitude)` uniform + smooth altitude→k curve.
- 3b: Per-step ray bending in the shader marcher (parabolic at moderate
  altitude, true spherical at high altitude).
- 3c: Two-ray gameplay picking — straight ray for placement/breaking, curved
  ray for visuals.
- 3d: Visual harness with screenshots at evenly-spaced altitude steps from
  surface to orbit.
- 3e: Edge-on horizon test — silhouette must be a circle from orbit.

### Phase 4 — Poles

- 4a: Worldgen marks the top/bottom Y rows non-buildable; edits in those
  regions are rejected.
- 4b: Pole rendering: simple opaque/impostor fill — figure out what looks
  reasonable from orbit and from the boundary on the ground.
- 4c: Validate ground view at high latitude looking toward the pole, and
  orbit view of polar regions.

### Phase 5 — Polish

- 5a: Tune `k(altitude)` by stepwise visual harness. The transition zone is
  the highest-risk artifact.
- 5b: Verify deep-depth rendering still works on the slab (zoom into a
  single voxel on the slab surface).
- 5c: Verify entities work on the slab.
- 5d: Verify edits/placement work, and are rejected on pole strips.

## Validation gates (every phase)

- `cargo build --release` green.
- `cargo test --release` green in this worktree.
- Visual harness screenshots committed to `tmp/` in this worktree.
- Coordinator review pass before commit.

## Coordinator-owned work

- Phase 3 entirely (render-time curvature in the shader).
- Code review of every opus-agent diff.
- Plan refinement after Phase 0.

## Deferred / out of scope

- Atmosphere.
- Lighting propagation across the X-wrap boundary (covered in Phase 5 if
  cheap, otherwise deferred).
- Multiple planets in one world.
- Real-time tile streaming for very large slabs.
