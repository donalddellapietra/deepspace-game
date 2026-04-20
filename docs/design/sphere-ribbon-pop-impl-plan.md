# Sphere ribbon-pop — implementation plan

Supersedes the earlier `sphere-ribbon-pop-proposal.md` with one
correction and concrete execution steps.

## Correction (from numeric verification)

`tests/sphere_ribbon_pop_precision.rs` proves the factored form
`t(K) = −(A + K·a)/(B + K·b)` is **insufficient** — at face-subtree
depth ≥ 20, `K·a` falls below f32 relative eps of `A`, so the final
sum collapses. The **relative form** survives:

    t(K) − t(0) = K · (A·b − B·a) / (B · (B + K·b))

This is the delta‑t between two adjacent boundary planes. The DDA
must track per‑cell relative t, not accumulate an absolute t in
body‑XYZ. This is the sphere analog of Cartesian ribbon‑pop, which
the earlier proposal left implicit.

## Architecture

### Render frame at face‑subtree depth N

New frame kind: `SphereSub`. Carries:

    body_path: Path
    face:      Face
    un_corner, vn_corner, rn_corner: f32   // absolute [0,1] face coords
    frame_size: f32                          // = 1/3^N
    inner_r, outer_r: f32                    // body-local radii

Plus, precomputed for O(1) per‑ray setup:

    c_body: [f32; 3]    // body-XYZ of local (0,0,0)
    J:      [[f32; 3]; 3]  // local→body Jacobian (linearized at corner)
    J_inv:  [[f32; 3]; 3]

### Local‑frame ribbon pop

Ray enters `SphereSub` frame as `(ro_local, rd_local)` with:

    ro_local = J_inv · (ro_body − c_body)
    rd_local = J_inv · rd_body

so the local frame maps to `[0, 3)³`. Cell boundaries in u and v are
at `local_u = 0, 1, 2, 3` — flat planes. Radial `r = R(K)` boundaries
are ellipsoids in local coords (J is generally non‑orthogonal, so
the body‑XYZ sphere transforms to an ellipsoid under J_inv); ray‑
ellipsoid intersection is a closed‑form quadratic, stable at any
depth because all quantities stay O(1) in local coords.

The DDA loop advances `ro_local` by `delta_t · rd_local` per cell
step. Both are O(1) in local coords, so the sum stays well within
f32 precision for 60+ face‑subtree levels.

### Plane/sphere intersection in local coords

**u/v planes (axis‑aligned in local):**

    local_u plane at K:  t_local = (K − ro_local.u) / rd_local.u

trivial, no warp, always representable.

**Radial boundaries (ellipsoid in local):**

The body‑XYZ sphere `|x − body_center|² = R²` in local coords
becomes `|c_body + J·x_local − body_center|² = R²`, which expands to
a quadratic in `t_local`:

    A·t² + 2·B·t + C = 0

with

    delta  = c_body + J·ro_local − body_center
    J·rd   = J · rd_local
    A = (J·rd) · (J·rd)
    B = delta · (J·rd)
    C = delta · delta − R²

All terms O(1) at any depth. Smaller root `t = (−B − √(B² − A·C))/A`.

### Why the relative form from the test applies

The test proved that `t(K) − t(0)` in body‑XYZ is precise. In local
coords, `t_local(K) − t_local(0) = (K − 0) / rd_local.u = K / rd_local.u`
 — directly representable. The relative form is baked into the
architecture: we *never* compute absolute body‑XYZ t inside the
local frame, so we never subtract two near‑equal f32s.

### Transition depth

- Face‑root (depth 1): existing exact `sphere_in_cell` march covers
  the whole face — no linearization.
- Depth 2: existing exact march still used; linearization error is
  1.2 % of cell width, borderline perceptible.
- Depth ≥ 3: `SphereSub` frame with local‑coord DDA. Linearization
  error is ≤ 0.14 % of cell width — sub‑pixel at typical FOV.

Seamlessness across transitions: the linearization error at the
frame corner matches the linearization error of the shallower frame
at that same point, to O(size_ea²). Transitions rebase one level at
a time, visible shift bounded by 0.14 % of a single cell width at
the transition depth.

## Execution order (single commit)

1. **`src/world/cubesphere.rs`** — add helpers:
   - `face_frame_jacobian(face, un_corner, vn_corner, rn_corner,
     frame_size, inner_r, outer_r, body_size) → (c_body, J)`.
   - `mat3_inv`, `mat3_mul_vec` small utilities (or reuse existing).

2. **`src/app/frame.rs`** — add `SphereSub` variant; teach
   `compute_render_frame` to descend through
   `CubedSphereFace` and through Cartesian‑kind nodes *inside* a
   face subtree (interpreting slots as UVR). Accumulate
   `un_corner, vn_corner, rn_corner, frame_size` along descent.
   Transition rule: depth ≥ 3 inside a face subtree → `SphereSub`;
   otherwise → existing `Body` (with exact march).

3. **`src/world/raycast/sphere.rs`** — split the existing
   `cs_raycast` into:
   - `cs_raycast_body(...)` — unchanged body‑local march (face‑root
     and shallow).
   - `cs_raycast_local(...)` — new, operates in a `SphereSub`
     frame's local coords.

4. **`src/world/raycast/mod.rs`** — dispatch `frame_kind`:
   - `Cartesian` → existing.
   - `Body { .. }` → `cs_raycast_body`.
   - `SphereSub(_)` → transform ray via `J_inv`, call
     `cs_raycast_local`.

5. **`assets/shaders/sphere.wgsl`** — add `sphere_in_sub_frame`,
   the GPU mirror of `cs_raycast_local`. Uniforms gain
   `sub_frame_c_body`, `sub_frame_J`, `sub_frame_J_inv`, and
   `sub_frame_face/corner/size` metadata.

6. **`assets/shaders/bindings.wgsl`** + **`src/renderer/mod.rs`** —
   add `ROOT_KIND_SPHERE_SUB = 3u`; renderer writes uniforms per
   frame kind.

7. **`src/world/aabb.rs`** — hit paths that descend through face
   subtree: `hit_aabb_body_local` already supports arbitrary
   face‑subtree depth via `face_space_to_body_point`. For
   `SphereSub` frames, transform hit cell corners into local coords
   via `J_inv` so the highlight renders in the same coord system as
   the sphere march.

8. **`src/app/edit_actions/*.rs`** — propagate `SphereSub` through
   frame‑aware raycast / highlight / upload / camera‑local /
   zoom. Camera transform for `SphereSub` uses `J_inv · (cam_body
   − c_body)`; other arms unchanged.

9. **Tests**:
   - Remove the `max_face_depth = 10, edit_depth = 30` caps from
     `planet_world_raycast_hits_sphere` and
     `planet_world_cartesian_descend_triggers_sphere_dispatch`.
   - Add a new `sphere_descent_to_depth_35` that anchors at a
     deep face‑subtree cell and verifies raycast + break succeed.
   - Add a visual regression test: render sphere at body‑root and
     at `SphereSub`‑depth‑3, diff silhouette curvature.

One commit: broken intermediate states are expected mid‑diff per
`feedback_no_intermediate_visual_states.md`. Compile green and
tests green at end.
