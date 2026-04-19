# Sphere locality refactor — plan

**Status:** planned, not yet implemented.

Companion to `docs/principles/no-absolute-coordinates.md` and
`docs/principles/locality-prime-directive.md`. Identifies the
remaining violations in the sphere path and the shape of the fix.

## The precision bug it fixes

At `anchor_depth ≥ 26`, `hit_aabb_body_local` collapses a face
subtree cell's 8 corners to a single f32 point:

- `face_cell_bounds(hit.path)` returns `(u_lo, v_lo, r_lo, size)`
  in face-normalized `[0, 1]³` — precision-safe via Kahan
  compensation at any depth.
- `face_space_to_body_point(face, u, v, r, inner_r, outer_r,
  body_size=3.0)` then scales these into **body-frame absolute
  coordinates** centered at `(1.5, 1.5, 1.5)`.
- A depth-26 cell has size `≈ 3⁻²⁴ ≈ 5e-12` in body-frame.
  `f32::ULP` at body-frame `1.5` is `≈ 2e-7`. The cell's extent is
  ~5 orders of magnitude below the precision available to
  represent it relative to body-center, so `1.5 + 5e-12 == 1.5`
  and all 8 corners land on the same `f32` triple.
- The highlight AABB collapses to a zero-extent box. The shader's
  `hit_pos ∈ AABB` check never fires; the cursor disappears.

This is the specific symptom of a structural violation: the sphere
pipeline expresses cell geometry in **body-frame absolute
coordinates** (the body is always at `[0, 3]³`, centered at
`(1.5, 1.5, 1.5)`) while the Cartesian pipeline already lives in
render-frame-local coordinates. Per the locality principle, a
consistent local-frame treatment at every subsystem removes the
precision wall.

## What's absolute today

Sphere-path sites that still use body-frame constants:

- `src/world/aabb.rs::hit_aabb_body_local` — accumulates 8 corners
  at body-center `(1.5, 1.5, 1.5)` with radius `outer_r * 3.0`.
- `src/world/aabb.rs::hit_aabb` (test-only) — same body-center
  offset walk.
- `src/app/mod.rs::gpu_camera_for_frame` — Sphere branch uses
  `camera.position.in_frame(&sphere.body_path)`. Cartesian branch
  uses `frame.render_path`. Asymmetric (violates principle #7).
- `src/app/edit_actions/upload.rs::set_root_kind_face` — ships
  `pop_pos = in_frame(&body_path)`, so the shader's
  `root_face_pop_pos` uniform is in body-frame.
- `src/app/edit_actions/mod.rs::frame_aware_raycast` — Sphere
  branch uses `body_path` + `cpu_raycast_in_sphere_frame`.
  Cartesian uses `render_path` + `cpu_raycast_in_frame`.
- `src/world/raycast/mod.rs::cpu_raycast_in_sphere_frame` — calls
  `sphere::cs_raycast_in_body` with `body_origin=[0,0,0],
  body_size=3.0` hardcoded.
- `src/world/raycast/sphere.rs::cs_raycast_in_body` — inner DDA
  uses `cs_center = body_origin + body_size * 0.5` with these
  fixed constants. `t` is body-frame distance.
- `assets/shaders/sphere.wgsl::march_face_root` /
  `sphere_in_cell` — `let cs_center = vec3<f32>(1.5);` and
  `let cs_outer = uniforms.root_radii.y * 3.0;` assume camera and
  ray are in body-frame.
- `assets/shaders/face_walk.wgsl::walk_face_subtree` — takes
  `(un_in, vn_in, rn_in)` in absolute face `[0, 1]`. Precision is
  `1 / ULP ≈ 1e7` steps; cells at depth ≥ 14 fall below this and
  round into their neighbours.

## What "symmetric render-frame" means

The target state:

- Camera position on the GPU and the CPU raycast origin are
  **both** `camera.position.in_frame(&active_frame.render_path)`
  for every kind.
- `active_frame.render_path` is deep (anchor_depth − 3 by
  default), so the cam magnitude stays `∈ [0, WORLD_SIZE)` with
  body-frame-equivalent precision at any depth — one ULP is
  ~2e-7 × WORLD_SIZE regardless of where the render frame sits
  in the global tree.
- AABBs, hit-points, highlight uniforms all live in that same
  frame.
- Sphere-specific curvature math (cube-to-equal-area, shell
  intersection) only runs at the body level, reached via ribbon
  pop. Inside the face subtree's render frame, the ray walks the
  tree with **face-local `(u, v, r)`** coordinates. The face
  subtree stores children at `slot_index(us, vs, rs)`, so the
  walker's slot-pick formula is `slot_index(floor(u * 3), floor(v
  * 3), floor(r * 3))` — structurally identical to the Cartesian
  walker but with face-axis-aware ray stepping.

## Why a naive unification is wrong

Attempting to just swap `sphere.body_path` → `frame.render_path`
in `gpu_camera_for_frame` and run the existing Cartesian walker
over a face subtree produces incorrect hits. The face subtree
stores children at `slot_index(us, vs, rs)` where the `(u, v, r)`
axes are **permuted relative to body `(x, y, z)`** depending on
which face:

| Face | u-axis | v-axis | r-axis |
|---|---|---|---|
| PosX | -Z | +Y | +X |
| NegX | +Z | +Y | -X |
| PosY | +X | -Z | +Y |
| NegY | +X | +Z | -Y |
| PosZ | +X | +Y | +Z |
| NegZ | -X | +Y | -Z |

A ray going straight down (body -Y) above the PosY face has its
`body_y` component aligned with the face's `r`-axis. But a
Cartesian walker picks `slot_index(floor(x), floor(y), floor(z))`
and treats `body_y` as `vs`. For PosY that reads the wrong cell
on every step (tested empirically: 6 of 40 descent iterations
return plausible-looking hits by coincidence, then the walker
wanders off).

So the face walker needs **face-local ray coordinates**:
project the camera position and ray direction onto the face's
`(u, v, r)` basis before walking. Within a sufficiently deep
render frame the cube-to-ea projection is locally linear
(distortion is `O(face_window_size²)` — negligible past
`face_depth ≥ 1`), so the CPU can precompute a
`(cam_uvr_local, ray_dir_uvr_local)` pair and the walker is a
plain DDA in face-local `[0, 3]³` coordinates.

At `face_depth = 0` (render frame is the whole face) the
linearization breaks down; that case keeps the current
sphere-in-cell math but re-expresses `cs_center`, `cs_outer`,
`cs_inner` as render-frame-local constants derived from
`WorldPos::offset_from`, not the hardcoded `(1.5, 1.5, 1.5)`.

## Proposed order of changes

This is a structural rewrite, not a patch stack. All the pieces
have to land together for any test to produce correct output;
splitting it into "always-green" intermediate commits inserts
shim layers that themselves become bugs. See the
`Don't force incremental green` feedback in auto-memory.

1. **AABB**: delete `hit_aabb_body_local` / `hit_aabb`; all
   callers use `hit_aabb_in_frame_local(hit, render_path)`.
2. **Face-local CPU ray**: new helper
   `world::cubesphere_local::body_to_face_local(cam_body,
    ray_dir_body, face) -> (cam_uvr, ray_dir_uvr)` applies the
   face's axis permutation + `cube_to_ea` + radial map.
3. **CPU raycast**: `frame_aware_raycast` always uses
   `render_path`. A single `cpu_raycast_in_frame` dispatches on
   the frame root's `NodeKind`:
   - `CubedSphereBody` → `cs_raycast_in_body` with
     render-frame-local `cs_center` / `cs_outer` / `cs_inner`
     computed via `offset_from`.
   - `CubedSphereFace` at `face_depth = 0` → same as above but
     in face-root-frame.
   - `CubedSphereFace` at `face_depth ≥ 1` → face-local-DDA
     walker (the face-permuted Cartesian walker).
   - Everything else → Cartesian walker.
4. **GPU camera**: `gpu_camera_for_frame` always uses
   `render_path`. `set_root_kind_face` ships the face-local
   sphere constants (or omits them for `face_depth ≥ 1`, where
   no sphere math runs).
5. **Shader**: `main.wgsl` / `sphere.wgsl` / `face_walk.wgsl` use
   the same dispatch: for `face_depth ≥ 1`, `march_face_root` is
   replaced by a face-local-DDA walker that works in
   `[0, WORLD_SIZE)³` render-frame coords. For `face_depth = 0`
   the existing `march_face_root` stays but reads render-frame
   sphere constants from uniforms instead of the hardcoded
   `vec3<f32>(1.5)`.
6. **Retire body-frame helpers**: delete
   `cpu_raycast_in_sphere_frame`, the `body_origin=[0,0,0],
    body_size=3.0` signature on `cs_raycast_in_body`, the
   body-frame `cs_center`/`cs_outer` hardcoding in the shader.
7. **Tests**: extend
   `sphere_cursor_hit_point_is_inside_aabb_after_wall_dig` to
   sweep `anchor_depth ∈ [5, 40]`; add a descent variant that
   checks `sphere_zoom_invariance` continues to hold at
   `tree_depth = 60`.

## What's landing now

Just the documentation. The implementation is intentionally
deferred: the above touches ~8 files across Rust + WGSL, and the
face-axis-permutation helper and shader rewrite each need their
own debugging pass. Landing them piecewise produces a broken
renderer for the duration; landing them together is the single
big commit this refactor needs.

## Scope that stays out

- No change to content addressing, face subtree storage layout,
  or SDF evaluation. The data on disk stays the same — only the
  coordinate-system contract between CPU, GPU, and AABB changes.
- No change to `WorldPos::add_local` / `renormalize_cartesian`.
  The anchor-relative offset semantics for face subtree nodes
  (which the coordinates doc says *should* be `(u, v, r)`) is a
  separate refactor; for now the face-local ray helper does the
  face permutation at the boundary, not inside `WorldPos`.
