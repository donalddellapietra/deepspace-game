# Sphere ribbon-pop: two-step DDA for 30–40+ layer rendering

## Problem

At UVR depth ≥ ~15 on a cubed-sphere planet, the CPU + GPU sphere
march renders broken geometry (seen by the user as a smeared
diamond-shaped pit at "Layer 18" in the dig-down descent). Two root
causes:

1. The existing `cs_raycast` body march uses `ray_plane_t` with plane
   normals built from `u_axis − ea_to_cube(un_corner + K·d_ea) · n_axis`.
   At depth 20, adjacent cells' `un_corner` values collapse into the
   same f32 and the three candidate cell planes have identical normals.
   The DDA deadlocks or steps through the wrong cell.

2. The current SphereSub path bails out when the requested render
   path hits a broken (Child::Empty) cell — falls back to a
   shallow-resolved Node or to the body march. The fallback loses
   the tight linearization. At deep m, the sub-frame is evaluated
   at the shallow-resolved corner (not the camera's deep cell), so
   the Jacobian spans a large region with perceptible curvature,
   producing the smeared rendering.

## Architecture: two-step ribbon-pop

### Step 1 — intra-sub-frame DDA (this doc's scope)

The sub-frame is ALWAYS built at the CAMERA'S DEEP uvr_path depth
(`m_truncated`). The Jacobian is evaluated at that deep corner, so
linearization error is O((1/3^m)²) — geometrically negligible at
m ≥ 10.

The walker is sphere state symbolic:

- `SphereSubFrame.face_root_id` — NodeId of the face subtree root
  (always resolvable: body_path + face_slot).
- Walker pre-descends via `sphere.uvr_path[..m_truncated]` slots.
  No absolute face-normalized coords — pure slot-following.
  Child::Empty along the pre-descent → walker returns empty for
  this DDA step (the ray is in a dug region, DDA advances).
- After pre-descent reaches the sub-frame's Node, runs
  `walk_sub_frame` with the ray's local `[0, 3)³` coords. That
  walker already descends precisely via local-coord slot-floor
  arithmetic — O(1) operations at any depth.

DDA between cells INSIDE the sub-frame uses local `[0, 3)³` with the
constant sub-frame Jacobian. Ray position updates via
`local_new = local_old + rd_local · t_exit` where `rd_local = J_inv
· rd_body` (magnitude 3^m) and `t_exit = O(1/3^m)` — product is
O(1), precise in f32.

### Step 2 — sub-frame to neighbor (follow-up)

When a ray exits the sub-frame's local box, transition to the
neighbor sub-frame via symbolic path stepping + fresh Jacobian.
Needed for long rays that traverse multiple depth-m cells (rare in
typical gameplay; common for grazing shots).

NOT in this commit's scope. Ray exit at sub-frame boundary =
terminate the DDA (current behavior); it only misses hits for
atypical grazing rays.

## Precision analysis

At UVR depth m, with this architecture:

| Quantity | Magnitude | Precision (f32) | Notes |
|---|---|---|---|
| `uvr_path[k]` | slot 0..26 | exact | symbolic |
| `sphere.uvr_offset` | `[0,1)` | ~1e-7 | per-cell, cell-relative |
| `cam_local = in_sub_frame(sub)` | `[0,3)` | ~1e-7 | symbolic walk, no body subtraction |
| `J_inv` entries | O(3^m) | ~1e-7 rel | f64 mat_inv internally |
| `rd_local = J_inv · rd_body` | O(3^m) | ~1e-7 rel | single mat*vec |
| `t_exit = (3 − local) / rd_local` | O(1/3^m) | ~1e-7 rel | ratio of f32 |
| `local_new = local + rd_local · t` | O(1) | ~1e-7 abs | product cancels magnitude |
| walker slot pickup | integer | exact | floor(local) within [0,3) |

`un_corner` loses bits past m≈15 (f32 ULP of 0.5). That only affects
J's evaluation point, which is flat-approximate over the tiny
sub-frame region anyway — cells tile correctly in local coords, only
shading normals absorb the 1e-7 drift.

Scales cleanly to m ≈ 80 before `3^m > f32_max`.

## Implementation plan — Step 1

Files to modify:

- `src/app/frame.rs` — `SphereSubFrame`: add `face_root_id: NodeId`.
  `compute_render_frame`: always build sub-frame at deep
  `m_truncated`, resolving `face_root_id` from `body_path +
  face_slot` (always exists). Drop `resolve_node_prefix`
  fallback — no shallow-truncate path.

- `src/world/raycast/sphere_sub.rs` — `cs_raycast_local`: instead
  of calling `walk_sub_frame(library, sub_frame_node, …)`, call a
  new `walk_from_deep_sub_frame` that takes `face_root_id` + the
  sub-frame's `uvr_path_prefix` (derivable from `sub.render_path`),
  pre-descends the prefix slot-by-slot, handles Child::Empty
  mid-prefix by returning an empty walk, and once reaching the
  sub-frame's terminal Node calls `walk_sub_frame` for the inner
  DDA levels.

- `src/world/anchor.rs` — `new_with_sphere_resolved`: no changes;
  already extracts `uvr_path` correctly.

- `assets/shaders/sphere.wgsl` — mirror the `walk_from_deep_sub_frame`
  logic in `sphere_in_sub_frame`'s walker. Uniforms gain
  `face_root_idx` (the packed index of the face root node) and
  `uvr_path_slots` (the up-to-m_truncated slots the shader
  pre-descends before starting local-coord DDA).

- `src/renderer/*` — pack `face_root_idx` + `uvr_path_slots` into the
  sphere uniform block. Match the shader's expectation.

Test criteria:

- `tests/e2e_sphere_descent::sphere_dig_down_descent` continues
  passing at depth 25 (already green).
- Screenshot at `d{18..25}` shows CRISP cell walls for the dug pit
  (not the smeared diamond). Manual / visual check.
- Unit tests in `src/world/raycast/sphere_sub.rs` updated:
  `cs_raycast_local_hits_at_sub_depth_3` still passes; add a new
  test that passes with the sub-frame's deep Node broken (Empty
  mid-uvr_path) + a solid cell deeper in the sibling subtree.

## Step 2 — sub-frame to neighbor transitions

**Why it's required**: Step 1 alone renders nothing inside dug pits.
When the camera is at UVR depth 20 and the pit was dug at UVR depth 5,
the sub-frame's deep cell is entirely inside the broken region
(`Child::Empty` at depth 5 in the pre-descent). Walker returns
uniform-empty for the whole sub-frame. Rays exit the `[0, 3)³` local
box → DDA terminates → `None`. Nothing visible.

To render the pit walls (which are SIBLING cells at a SHALLOWER UVR
depth), the ray must exit the current sub-frame and enter the
neighbor sub-frame. That neighbor's uvr_path differs by one slot
step, has a fresh Jacobian, and may contain solid content.

**Math**: at cell exit via axis `k`, direction sign `s` ∈ {+1, −1}:
```
current: body_pos = c_cur + J_cur · local_cur           (local_cur[k] = 3 if s=+1, 0 if s=−1)
neighbor: body_pos = c_new + J_new · local_new          (local_new[k] = 0 if s=+1, 3 if s=−1)
c_new − c_cur ≈ s · 3 · J_cur[:, k]                     (column k of J_cur, × 3 local units)
→ local_new = J_new⁻¹ · J_cur · (local_cur − s · 3 · e_k)
```

`J_new⁻¹ · J_cur` is a small matrix close to identity: differs from `I`
by `O(1/3^m)` curvature. All magnitudes stay O(1) in local coords —
f32-precise at any depth.

Ray direction in new frame: `rd_local_new = J_new⁻¹ · rd_body`
(recomputed from body-dir; same unit rd_body).

### Symbolic path stepping

`uvr_path` is a `Path` whose slots use UVR semantics but the same
`slot_index(us, vs, rs)` packing as XYZ. `Path::step_neighbor_cartesian`
(renamed conceptually to `step_neighbor`) works unchanged — it's
slot-packed arithmetic, oblivious to XYZ vs UVR semantics.

Edge cases:
- `step_neighbor` bubbles up past uvr_path depth 0 (the face root
  boundary) → cross-face transition. NOT in this commit. Terminate
  the DDA in this case; the cross-face math needs face-to-face basis
  rotation, deferred to a follow-up.

### Incremental Jacobian update

`un_corner`, `vn_corner`, `rn_corner` update by `±frame_size` per
neighbor step (the `slot_index` axis semantics carry over):
- +u neighbor → `un_corner += frame_size`
- +v neighbor → `vn_corner += frame_size`
- +r neighbor → `rn_corner += frame_size`
- −axis → subtract instead.

Bubble-up preserves this: at the parent level, `frame_size` is 3x
larger, but slot moves from 2→3 (wrap to 0 with parent++), net delta
on the axis-sum is `−2·child_size + 3·child_size = +child_size` — same
as a simple increment at the leaf level.

`face_frame_jacobian` is re-evaluated at the new `(un, vn, rn)` to
produce fresh `J_new`, `c_body_new`, `J_new_inv` (via `mat3_inv` →
f64 internally).

### DDA loop structure

```
current_sub = sub (from compute_render_frame)
ro_local, rd_local = (cam_local, J_inv · rd_body)
loop:
  walk the sub-frame at current position → cell or uniform empty
  if hit → return HitInfo
  advance local t to cell exit (intra-cell or sub-frame-box exit)
  if pos still in [0, 3)³ → continue (next iter, same sub-frame)
  else:
    axis k, direction s = exit face
    step current_sub.uvr_path neighbor (axis k, direction s)
    if bubble past face root → terminate
    rebuild current_sub at new uvr_path (J, J_inv, c_body, un/vn/rn)
    local_new = J_new_inv · J_current · (local_exit − s·3·e_k)
    rd_local = J_new_inv · rd_body
    ro_local = local_new
    reset t = 0 (DDA restarts in new sub-frame's local parameter)
```

### Performance

Each neighbor transition does:
- O(m) to walk the pre-descent (optimizable via caching).
- O(1) to update un/vn/rn.
- O(1) `face_frame_jacobian` + `mat3_inv`.
- O(1) position + direction transfer.

For typical rendering scenarios (few cell transitions per ray), the
per-pixel cost is bounded. A future optimization: cache the
pre-descent path for neighboring rays (pixels near each other share
most prefix slots).

### Shader mirror

`assets/shaders/sphere.wgsl`'s `sphere_in_sub_frame` needs the same
loop: on box exit, step uvr_path neighbor, recompute J / c_body /
un_corner in-shader, transfer position, continue. Uniforms already
carry `uvr_prefix_slots[]` from Step 1; add space for the runtime-
mutated version (WGSL arrays can be mutated inside a function; the
initial values come from uniforms).

## Out of scope (follow-up)

- Cross-face transitions (ray crosses a cube-face seam). Needs
  per-face basis rotation to transform rd_body, position, and the
  walker's face root.
- Body-march rewrite. Unnecessary if SphereSub covers all
  deep-depth scenarios the user encounters in gameplay.
- Pre-descent caching for neighboring rays.
