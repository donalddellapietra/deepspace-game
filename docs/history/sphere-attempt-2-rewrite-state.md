# Sphere-attempt-2 rewrite — session state

## Where we are

On branch `sphere-attempt-2`, branched from `entities-on-fractal`.

- **Commit `1d184e8` (pushed):** full strip of all sphere / cubed-sphere
  code, Cartesian pipeline green. 127 library tests passing.
- **Uncommitted WIP** from the rewrite pass: see "What's written" below.

## What's written (uncommitted)

- **`src/world/cubesphere.rs`** (~330 LoC, fresh) — geometry helpers
  (`Face`, `FACE_SLOTS`, `CORE_SLOT`, `pick_face`, `ea_to_cube` /
  `cube_to_ea`, `face_uv_to_dir`, `body_point_to_face_space`,
  `face_space_to_body_point`, `ray_outer_sphere_hit`,
  `find_body_ancestor_in_path`) + worldgen (`PlanetSetup`,
  `demo_planet`, `insert_spherical_body`, `install_at_root_center`).
  Consolidates the failed branch's three files (`cubesphere.rs` +
  `cubesphere_local.rs` + `spherical_worldgen.rs`, ~760 LoC) into
  one focused module. Unit tests for face/EA round-trips and body
  insertion.

- **`src/world/raycast/sphere.rs`** (~200 LoC, fresh) — unified
  `cs_raycast` taking `Option<FaceWindow>`. `None` = whole-body
  march (called from Cartesian DDA when it descends into a
  `CubedSphereBody` child); `Some(FaceWindow)` = bounded face-window
  march (called when the render frame root lives inside a face
  subtree). `walk_face_subtree` descends the face subtree to
  `max_face_depth`, synthesising `EMPTY_NODE`-tagged placement
  entries so placement depth doesn't collapse to wherever the empty
  chain happened to end.

- **`src/world/tree.rs`** — `NodeKind` re-gained `CubedSphereBody {
  inner_r, outer_r }` and `CubedSphereFace { face }` variants with
  proper `Hash` + `Eq` impls.

- **`src/world/mod.rs`** — `pub mod cubesphere;` restored.

## What's left (compile-breaking)

These errors will surface on next `cargo check`. All are local match
arms / dispatch points that don't know about the re-added `NodeKind`
variants:

1. **`src/world/gpu/types.rs`** — `GpuNodeKind::from_node_kind` match
   is non-exhaustive. Restore two arms:
   ```rust
   NodeKind::CubedSphereBody { inner_r, outer_r } => Self {
       kind: 1, face: 0, inner_r, outer_r,
   },
   NodeKind::CubedSphereFace { face } => Self {
       kind: 2, face: face as u32, inner_r: 0.0, outer_r: 0.0,
   },
   ```

2. **`src/world/gpu/pack.rs`** — uniform-flatten gate exempts sphere
   nodes. Add back the `matches!(node.kind, NodeKind::Cartesian)`
   gate around the flatten logic (roughly line 212 of the strip
   state).

3. **`src/world/raycast/mod.rs`** — wire in the sphere module and
   dispatch at frame root:
   ```rust
   mod sphere;
   ```
   `cpu_raycast_in_frame` must dispatch on `frame_kind`:
   - `CubedSphereBody { inner_r, outer_r }` → `sphere::cs_raycast(...)` with `window = None`.
   - Otherwise → existing Cartesian path.

   Also add back `cpu_raycast_in_sphere_frame` (or fold into
   `cpu_raycast_in_frame` via an optional window param). Callers
   from `edit_actions/mod.rs` pass face-window when render root is
   a face node.

4. **`src/world/raycast/cartesian.rs`** — dispatch into sphere at
   descent boundary: when walking into a `Child::Node(child_id)`
   whose `NodeKind` is `CubedSphereBody`, hand off to
   `sphere::cs_raycast` with body origin / size from the DDA's
   current cell state, `window = None`, and the current `path` as
   `ancestor_path`. On hit, return that HitInfo up. On miss,
   advance the DDA past the body cell.

5. **`src/world/aabb.rs`** — `hit_aabb` and `hit_aabb_in_frame_local`
   must handle hit paths that cross a `CubedSphereBody` ancestor.
   Original behavior: walk path up to the body, then compute the
   hit cell's 8 corners in body-local coords via
   `face_space_to_body_point` and bound them. Restore
   `hit_aabb_body_local` for the highlight path (see `edit_actions`
   below).

## What's left (end-to-end UX parity)

6. **`assets/shaders/sphere.wgsl`** (fresh, ~250 LoC target) — GPU
   mirror of `cs_raycast`. One `sphere_in_cell(body_origin,
   body_size, inner_r, outer_r, window_bounds, window_active)`
   function that dispatches both unbounded body-march and
   face-window march via a single flag. Inline the small face-math
   helpers (`cube_to_ea`, `ea_to_cube`, `pick_face`, `face_normal`,
   `face_u_axis`, `face_v_axis`) and `walk_face_subtree` — no need
   for separate `face_math.wgsl` / `face_walk.wgsl` files.

7. **`src/shader_compose.rs`** — add `("sphere.wgsl", include_str!(
   "../assets/shaders/sphere.wgsl"))` to the SOURCES table.

8. **`assets/shaders/main.wgsl`** + **`measure_compute.wgsl`** —
   re-add `#include "sphere.wgsl"`.

9. **`assets/shaders/march.wgsl`** — restore NodeKind dispatch:
   - Inside `march_cartesian`'s tag=2 descent branch, when
     `node_kinds[child_idx].kind == 1u` (BODY): call
     `sphere_in_cell` with body origin/size from the current DDA
     state, `window_active = false`. On hit, propagate. On miss,
     advance DDA.
   - At the top-level `march`, dispatch on `uniforms.root_kind`:
     - CARTESIAN → `march_cartesian(current_idx, ...)`.
     - BODY → `sphere_in_cell` with body filling the `[0, 3)^3`
       render frame (`body_origin = vec3(0.0)`, `body_size = 3.0`),
       `window_active = false`.
     - FACE → `sphere_in_cell` with body filling frame,
       `window_active = true`, bounds from uniforms.

10. **`assets/shaders/bindings.wgsl`** — restore `ROOT_KIND_BODY =
    1u` and `ROOT_KIND_FACE = 2u` constants.

11. **`src/renderer/mod.rs`** — restore `ROOT_KIND_BODY` /
    `ROOT_KIND_FACE` pub consts + `set_root_kind_body(inner_r,
    outer_r)` / `set_root_kind_face(inner_r, outer_r, face_id,
    bounds, pop_pos)` methods. The uniform struct fields
    (`root_radii`, `root_face_meta`, `root_face_bounds`,
    `root_face_pop_pos`) were preserved as seam during strip — they
    already exist; just stop writing zeros.

## App-layer integration

12. **`src/app/frame.rs`** — re-add `ActiveFrameKind::Body { inner_r,
    outer_r }` and `ActiveFrameKind::Sphere(SphereFrame)`. **No
    extraneous fields on `SphereFrame`**: just what the uniforms
    and CPU raycast need — `body_path: Path`, `face: Face`,
    `inner_r: f32`, `outer_r: f32`, `face_u_min`, `face_v_min`,
    `face_r_min`, `face_size`. Drop `face_root_id`,
    `body_node_id`, `frame_path`, `face_depth` from the failed
    branch (redundant with existing frame machinery — derived
    on-demand from `render_path` when needed).

    `compute_render_frame` walks the render path. When it enters
    a `CubedSphereBody`, stash radii + body path. When it enters
    a `CubedSphereFace`, start tracking the window `(face, 0, 0,
    0, 1)`. Subsequent Cartesian descents inside the face subtree
    update the window via slot-coords: `size /= 3; u_min += us *
    size; v_min += vs * size; r_min += rs * size`. Result kind:
    `Cartesian` / `Body` / `Sphere(SphereFrame)`.

13. **`src/app/mod.rs`** — re-add `planet_path: Option<Path>` on App.
    `render_frame_kind` dispatches on `ActiveFrameKind` to
    `NodeKind`. `gpu_camera_for_frame` handles `Sphere` by
    transforming through the body path (not render_path).

14. **`src/app/edit_actions/mod.rs`** — `frame_aware_raycast`
    dispatches on `ActiveFrameKind`:
    - `Cartesian` | `Body { .. }` → `cpu_raycast_in_frame`.
    - `Sphere(sphere)` → `cpu_raycast_in_sphere_frame` (or fold:
      `cpu_raycast_in_frame` with window param).

15. **`src/app/edit_actions/highlight.rs`** — sphere hits use
    `aabb::hit_aabb_body_local` (highlight is drawn in body-local
    coords for sphere to avoid f32 precision loss in the popped
    render frame). Cartesian uses `hit_aabb_in_frame_local`.

16. **`src/app/edit_actions/zoom.rs`** — restore
    `camera_local_sphere_focus_path` (picks the face subtree
    nearest the camera's forward-ray to anchor the render frame).
    Restore sphere arms in `camera_fits_frame` and
    `frame_projected_pixels`.

17. **`src/app/edit_actions/upload.rs`** — restore
    `set_root_kind_body` / `set_root_kind_face` dispatch in the
    match on `self.active_frame.kind`. Beam-prepass currently
    `matches!(.., Cartesian)`; keep that gate (beam is Cartesian-
    only; sphere has its own fast-path via cubemap partitioning).

18. **`src/app/event_loop.rs`** — sphere `camera_local` for debug
    overlay uses `sphere.body_path`, not `render_path`.

19. **`src/world/bootstrap.rs`** — re-add `WorldPreset::DemoSphere`
    variant + `bootstrap_demo_sphere_world` function. The
    bootstrap calls `cubesphere::install_at_root_center`, sets
    `planet_path: Some(body_path)`. Update `surface_y_for_preset`
    to return `None` for DemoSphere.

20. **`src/world/bootstrap.rs` `WorldBootstrap` struct** — re-add
    `planet_path: Option<Path>` field (was stripped). Update all
    fractal `WorldBootstrap { ... }` constructors to set
    `planet_path: None`.

21. **`src/app/test_runner/config.rs`** — re-add `"--sphere-world"
    => cfg.world_preset = WorldPreset::DemoSphere` + usage text.

## Tests to add (minimum for confidence)

- `cubesphere::tests` — already written (face math, round-trips,
  insertion).
- `raycast::tests::planet_world_raycast_hits_sphere` — ray from
  outside the planet body hits the outer shell.
- `raycast::tests::planet_dig_to_core` — successive break edits
  down through the shell reach the core boundary.
- `edit::tests::propagate_edit_preserves_node_kinds_through_sphere_path` —
  the critical regression: `propagate_edit` already preserves
  NodeKind on rebuild; this test guards that invariant. (Already
  written in the failed branch; port forward.)

## Build/test verification

```bash
cd /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/sphere-attempt-2
cargo check --workspace --all-targets
cargo test --lib
# Headless render for visual sanity:
./scripts/dev.sh -- --sphere-world --render-harness --screenshot tmp/sphere.png --exit-after-frames 60
```

## Design principles locked in

1. **Sphere nodes are first-class Cartesian-tree citizens.** Tree
   storage, pack, ribbon, edit propagation, entity overlay — all
   unchanged from Cartesian. Only the DDA dispatches on NodeKind.

2. **No parallel raycast function.** The existing
   `cpu_raycast_in_frame` handles sphere via NodeKind dispatch at
   frame root and at descent. If a face-window context is needed
   (render root inside face subtree), it's passed as an optional
   param — not a duplicate function.

3. **Face-window bounds are derived from the render path, not
   stored in a `SphereFrame` field bundle.** `compute_render_frame`
   walks the path and accumulates `(face, u_min, v_min, r_min,
   size)` from slot coordinates during Cartesian descents inside
   a face subtree. No `face_depth`, `frame_path`,
   `body_node_id`, `face_root_id` fields on the frame — all
   recoverable from the path.

4. **One shader sphere march, not three.** Failed branch had
   `sphere_in_cell`, `sphere_in_face_window`, `march_face_root`.
   Rewrite: one `sphere_in_cell(... , window_active: bool,
   window: vec4<f32>)`.

5. **`cs_edit_depth == edit_depth`.** No divergence between
   Cartesian and sphere edit depths.

6. **Core is uniform-stone Cartesian.** Sphere digging reaches
   `r = inner_r`; below that the body's center slot holds a
   uniform-stone subtree. Accept the resolution discontinuity at
   the inner shell boundary — invisible in practice.

## Known pitfalls from the failed branch

- **NodeKind preservation on edit.** `propagate_edit` MUST
  reinsert ancestors with `insert_with_kind` preserving the
  original kind. `edit.rs` already does this (comment in the file
  references the regression). Don't regress.

- **Dedup across NodeKinds.** The NodeKind is part of the content
  hash. Two nodes with identical children but different kinds do
  not dedup. Tree tests verify.

- **`uniform_type` for CubedSphereBody.** The body is NOT uniform
  (it has 6 face children + 1 core), so `UNIFORM_MIXED` is correct.
  Pack must not try to flatten it to a single block.
