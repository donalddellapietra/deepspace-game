# Unified slot-path DDA — new sphere architecture

## Premise

One DDA primitive handles every rendering layer, Cartesian and UVR.
No `SphereSubFrame`, no `m_truncated` threshold, no Jacobian stored
per-cell, no two-march-paths-with-a-transition.

State per DDA iteration:

- `slot_path: u8[]` — symbolic chain from world root down through
  body node + face slot + UVR levels. Integer-typed, no precision wall.
- `residual: [f32; 3] ∈ [0, 1)³` — position inside the current cell's
  local [0, 1) box. Magnitude always O(1).
- `rd_body: [f32; 3]` — held constant per face; rotated on cross-face
  transitions.

Per step:

1. Tree-lookup `slot_path` → block / empty / descend.
2. If hit: materialize `body_pt` lazily from slot_path (Horner sum
   of `slot·(1/3)^k` → `(un, vn, rn)` → `face_space_to_body_point`),
   return `HitInfo`.
3. Else: compute `rd_local = J_hat⁻¹ · rd_body` once for this cell
   (J_hat is the O(1) *shape* Jacobian — scale factored out).
   Find axis-exit `t_min` against residual ∈ [0, 1) walls.
   `residual += rd_local · t_min`. All operands O(1).
4. Cross the wall: `slot_path.step_neighbor(axis, sign)` (integer
   carry; bubble up on overflow). `residual[axis]` = 0 or 1−ε on
   opposite side. On face-root overflow, cross-face rotate `rd_body`
   into neighbor face's basis, descend new face slot.

The cartesian ribbon-pop in `raycast/cartesian.rs` already implements
this primitive for cartesian frames. The rewrite is: extend that
primitive to recognize `NodeKind::CubedSphereBody` / `CubedSphereFace`
and dispatch the remap only at those boundaries.

## Files to delete outright

- `src/world/raycast/sphere.rs` — body-XYZ march (`cs_raycast`,
  `walk_face_subtree`, `FaceWindow`, `face_lod_depth`). Replaced by
  unified DDA.
- `src/world/raycast/sphere_sub.rs` — local-frame march
  (`cs_raycast_local`, `walk_from_deep_sub_frame`, `walk_sub_frame`).
  The abandoned sub-frame path. Delete.
- `docs/design/sphere-ribbon-pop-two-step.md`
- `docs/design/sphere-ribbon-pop-impl-plan.md`
- `docs/design/sphere-ribbon-pop-gpu-port.md`
- `docs/design/sphere-ribbon-pop-uvr-state.md`
- `docs/design/sphere-ribbon-pop-proposal.md`
- `docs/design/sphere-shader-bug-repro.md`
- `docs/problems/sphere-dig-down-saturation.md` — the bug class this
  architecture eliminates; keep the observation in git history.

## Files to strip (retain, remove items)

### `src/world/raycast/mod.rs`

- Remove `mod sphere`, `mod sphere_sub`.
- Remove `pub use sphere::{FaceWindow, LodParams}` — move `LodParams`
  to `cartesian.rs` or the new unified module if still used.
- Remove `cpu_raycast_in_sub_frame`.
- From `SphereHitCell`, remove:
  - `sub_c_body: [f32; 3]`
  - `sub_j_cols: [[f32; 3]; 3]`
  - `sub_local_lo: [f32; 3]`
  - `sub_local_size: f32`

  The Jacobian-escape-hatch fields are the admission that the hit
  coords `u_lo/v_lo/r_lo` aren't trustable at deep m. In the new
  architecture the hit carries `slot_path` + a tail residual; the
  AABB derives its 8 corners from that, each in body-XYZ via
  `face_space_to_body_point` (each an O(1) computation). No
  Jacobian needed downstream.

### `src/world/cubesphere.rs`

Delete:
- `face_frame_jacobian` — the per-cell basis builder
- `mat3_inv` — inverse of the per-cell Jacobian
- `mat3_mul_vec`, `Mat3` — helpers only `mat3_inv` used

Keep:
- `face_space_to_body_point` — hit materialization
- `body_point_to_face_space` — ray-sphere entry (once per ray)
- `face_uv_to_dir`, `ea_to_cube`, `cube_to_ea` — shared primitives
- `FACE_SLOTS`, `Face` — tree topology

### `src/app/frame.rs`

- Delete `SphereSubFrame` struct, its impl (`with_neighbor_stepped`,
  `depth_levels`, …).
- Delete `ActiveFrameKind::SphereSub` variant.
- Delete `uvr_corner()` helper.
- Collapse `compute_render_frame` to produce `Cartesian` or `Body` —
  or reduce the enum to just a `render_path: Path` + `logical_path`.
  The sphere-specific distinction falls out of the node kinds on the
  path, not a frame-level tag.

### `src/world/anchor.rs`

- Delete `WorldPos::in_sub_frame` (only `SphereSubFrame` consumer).
- Delete `Transition::SphereEntry { body_path }` variant (dead since
  `maybe_enter_sphere` removal in `eed141b`).
- `SphereState.uvr_path` stays — it's the semantic extension of the
  anchor's slot path into the face subtree. In the new architecture
  `slot_path = body_path + face_slot + uvr_path` concatenated; there's
  no renderer-side distinction between the cartesian prefix and the
  UVR suffix.

### `src/app/edit_actions/upload.rs`

- Delete the `ActiveFrameKind::SphereSub(sub)` match arm.
- Delete the `set_root_kind_sphere_sub(…)` call site.
- Uniform upload becomes: `set_frame_root(bfs_idx)` + whatever
  cube-face metadata the shader needs (face index, inner_r, outer_r
  — already available on the `CubedSphereBody` node).

### `src/renderer/mod.rs`, `init.rs`, `buffers.rs`

Delete:
- `set_root_kind_sphere_sub` method
- `ROOT_KIND_SPHERE_SUB` constant (reduce `root_kind` to
  `{ CARTESIAN, SPHERE_BODY }` or eliminate entirely — the shader can
  inspect NodeKind at the render root)
- `sub_uvr_slots: array<vec4<u32>, 16>` uniform
- `sub_meta: vec4<u32>` uniform (face / prefix_len / face_root_depth)
- `sub_face_corner: vec4<f32>` uniform
- All the CPU-side buffer-packing code that writes the above

### `src/shader_compose.rs`

- Delete the sub-uvr-slots compose entries (4 uses of
  `sub_uvr_slots`/`sub_meta`).

### `assets/shaders/bindings.wgsl`

- Remove `sub_uvr_slots`, `sub_meta`, `sub_face_corner` fields from
  `Uniforms`.

### `assets/shaders/march.wgsl`

- Delete the entire `ribbon_level == 0u && uniforms.root_kind ==
  ROOT_KIND_SPHERE_SUB` branch (lines ~835–870).
- The `sphere_in_cell` dispatch (line ~566, ~889, ~898) gets replaced
  by the unified DDA entry point.

### `assets/shaders/sphere.wgsl`

This file shrinks dramatically. Delete:
- `face_frame_jacobian_shader` + `FaceFrameJac` struct
- `mat3_inv_shader`, `mat3_inv_scaled_shader` + `Mat3Columns`
- `walk_face_subtree` (CPU mirror deleted too)
- `walk_from_deep_sub_frame_dyn`
- `sphere_in_sub_frame` — the whole sub-frame entry (lines ~1040+)
- `bevel_layered_local` (the sub-frame-local bevel variant; keep the
  body bevel `bevel_layered` and reformulate it on slot_path +
  residual)
- `coords_to_slot`, `slot_to_coords` (if only sub-frame code uses them)
- `sphere_in_cell` in its current body-XYZ form

Keep (reformulated to consume slot_path + residual):
- Ray-sphere entry (finds initial slot_path)
- Shell clip (inner_r / outer_r bounds)
- Per-cell walker — rewritten to slot-path form
- `bevel_layered` — rewritten

### Tests

- `tests/e2e_sphere_descent.rs` — retain as validation harness;
  rewrite the assertion paths since `SphereSubFrame` is gone.
- `src/world/raycast/sphere_sub.rs::tests` — delete with the file.
- `src/world/raycast/sphere.rs::tests` — delete with the file.
- New unit tests for the unified DDA: depth 5, 15, 25, 35 all hit
  through the same code path; face-seam crossing; dug pit visible
  from inside a deep sub-cell.

## Files to add

- `src/world/raycast/unified.rs` (or extend `cartesian.rs`) — the
  unified DDA primitive, with sphere remap hooks at NodeKind
  boundaries.
- `assets/shaders/unified.wgsl` (or extend `tree.wgsl`) — shader
  mirror.

## What stays untouched

- `src/world/raycast/cartesian.rs` — the primitive this architecture
  generalizes.
- `src/world/tree.rs` — Path, slot_coords, NodeId, NodeLibrary.
- `src/world/cubesphere.rs::{face_uv_to_dir, face_space_to_body_point,
  body_point_to_face_space, ea_to_cube, cube_to_ea, Face, FACE_SLOTS}`.
- Entity system, heightmap, atmosphere, UI, physics.
- `assets/shaders/{main, ray_prim, tree, taa_resolve,
  entity_raster}.wgsl`.

## Staged commit plan

The architecture rewrite must land as one diff (see memory: "no
intermediate visual states"). But the preparatory deletion work can
stage ahead of it:

1. **Pre-strip** (green at head): delete unused `SphereHitCell`
   Jacobian fields, delete `cpu_raycast_in_sub_frame`, delete the
   march.wgsl SPHERE_SUB branch, delete the upload.rs / renderer
   sub-frame wiring. After this, `sphere_sub.rs` is unreferenced.
2. **Delete sphere_sub.rs + SphereSubFrame + shader mirror** (green):
   nothing consumes it anymore.
3. **Unified DDA rewrite** (one commit, may be broken mid-diff):
   replace `cs_raycast` and `walk_face_subtree` with the unified
   primitive; rewrite shader `sphere_in_cell` to the slot-path form.

Stages 1–2 are pure subtractions; the risk is concentrated in
stage 3. Expected size:
- Stage 1: ~-400 LoC
- Stage 2: ~-1800 LoC (sphere_sub.rs + shader sub-frame sections)
- Stage 3: ~-600 LoC from `sphere.rs` + ~+800 LoC new unified DDA +
  ~-400 LoC shader + ~+500 LoC shader unified march

Net: ~-1900 LoC. Smaller codebase, no architectural branch.

## Open questions (for the critique agent)

1. Does `rd_local = J_hat⁻¹ · rd_body · (1/s)` stay f32-stable when
   `residual += rd_local · t` is computed per DDA step, with
   `rd_local ~ O(3^m)` and `t ~ O(1/3^m)`? The individual step is
   O(1), but does it compound over many steps?
2. `alpha_u = (π/2) / cos²(e_u · π/4)` blows up as `e_u → ±1` (face
   edge). Does the scale-factor-out trick `J = s · J_hat` actually
   hold near face edges, or does J_hat acquire its own O(1/cos²)
   scale that compounds with s?
3. Cross-face transitions at cube edges: feasible as a slot-path
   bubble-up + face rotation? Or does grazing-angle thrashing
   between adjacent faces require separate handling?
4. LOD depth (`face_lod_depth(t, shell, ...)`) currently uses body-
   XYZ ray distance. In slot-path form, does `t · |rd_body|` give
   the same screen-projected pixel size? Sanity-check.
5. What exactly does the shader need to compute mid-DDA that requires
   body-XYZ vs. what can be deferred to hit-materialization? Audit
   each consumer.
