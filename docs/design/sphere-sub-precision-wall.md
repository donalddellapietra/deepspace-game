# Sphere-sub precision wall — findings

Reproducible characterization of the "sphere only renders to layer
~10" issue described in the unified-DDA plan, using the new
`force_sphere_state` script command + `scripts/sphere_sub_screenshot.sh`.

## Reproduction

    scripts/sphere_sub_screenshot.sh sub_baseline 5
    scripts/sphere_sub_screenshot.sh sub_baseline 10
    scripts/sphere_sub_screenshot.sh sub_baseline 15
    scripts/sphere_sub_screenshot.sh sub_baseline 20
    scripts/sphere_sub_screenshot.sh sub_baseline 30
    scripts/sphere_sub_screenshot.sh sub_baseline 50

These produce `tmp/sub_baseline/sub-depth-N.png` with the camera
in `ActiveFrameKind::SphereSub` mode (forced via the harness's new
`force_sphere_state` script command, which dispatches the same code
path as the F9 debug key).

## Observed behaviour

| Depth | PNG bytes | Visual |
|-------|-----------|--------|
| 5     | 207526    | Crisp perspective grid of deep face-subtree cells |
| 10    | 52402     | Mostly uniform grey + sparse cyan dots (debug-paint sentinel for `t>=t_exit silent miss` per `sphere.wgsl:1332`) |
| 15    | 8584      | Solid sky — sphere not rendered at all |
| 20    | 5437      | Solid grey, no detail |
| 30    | 8584      | Solid sky |
| 50    | 5277      | Solid sky |

The wall is sharp and reproducible:
- depth 5  → fully working
- depth 10 → degraded; debug-paint shows precision-miss sentinel firing
- depth 15+ → ray entirely misses the body (or runs out of iteration budget)

The cyan sparkles at depth 10 are `sphere.wgsl::sphere_in_sub_frame`
hitting its case-5 termination (`if sign_s == 0` after the
out-of-box check), which the SPHERE_DEBUG_PAINT path tints cyan to
distinguish from sky (which would be the silent fall-through).

## What the recent unified changes did NOT fix

Steps 5a-5b of the unified-DDA work (commits `64ce3b1` and `7df88a0`)
removed `force_terminate` and added the skip-face-root pop loop, so
sub-frame misses CAN fall through to body-level march. The screenshots
above confirm this: at depth 10, parts of the frame are uniform grey
(body march is producing SOMETHING, not pure sky), and at depth 20
the whole frame is grey rather than sky. So fall-through is reaching
sphere_in_cell.

But sphere_in_cell at body level can't reproduce the fine cell-grid
detail that sphere_in_sub_frame produced at depth 5: the body march
LODs out at coarser resolution. Past depth 10 the WHOLE sphere
collapses into one uniform-coloured cell to body march, because the
body's content is many levels below the body cell itself.

## What an actual fix needs

The unified-DDA plan's Step 5d-5e: a real residual-DDA inside
sphere_in_sub_frame that doesn't accumulate body-XYZ position across
DDA steps. The current sphere_in_sub_frame uses sub-frame local
coords with rd_local at O(3^m) magnitude — at m ≥ ~10, the products
that compute exit times underflow or cancel.

### Root cause in one sentence

At m ≥ ~10, `pos = ro_local + rd_local * t` in the DDA step
(sphere.wgsl:1270) silently loses the smaller components of
`rd_local * t` because they're O(1/3^m · other-axis-scale) below the
f32 ULP of ro_local's O(1) components. The ray effectively stops
advancing on those axes, so the DDA either:
- Hits t >= t_exit without any axis crossing the box boundary
  (the "cyan sentinel" in SPHERE_DEBUG_PAINT); or
- Advances on ONE axis only, missing all cell boundaries on the
  others, and eventually returns a miss.

### Concrete port recipe for sphere_in_sub_frame

The CPU `unified_raycast` demonstrates the precision-stable pattern.
Port recipe:

1. State: per-cell `residual ∈ [0,1)³` (not sub-frame-local `[0,3)`
   pos). Each cell has its own basis.
2. Track `cell_slot: vec3<i32>` at current DDA level + `residual` at
   cell entry. DDA step computes `t_exit[k] = (target[k] - residual[k]) / rd_cell_local[k]`
   where `rd_cell_local = J_inv_cell · rd_body`.
3. `J_inv_cell` rebuilt at cell entry via `face_frame_jacobian_shader`
   at the cell's face-normalized corner. This keeps `J_inv_cell`'s
   magnitude bounded by the cell's face-subtree depth relative to
   the CURRENT cell (not the deep global depth), so its components
   stay at O(3^m_cell · body_size^-1) where m_cell is the walker's
   local descent from the sub-frame root — bounded.
4. `rd_body` itself is O(1), invariant. Never accumulate across steps.
5. Neighbor transition: `cell_slot[k] += sign`, `residual[k]` snaps
   to 0 or 1-eps on the crossed axis. If `cell_slot` leaves `[0,2]`,
   bubble up the UVR path (unchanged from current sphere_in_sub_frame).
6. Face-seam: unchanged pathway (currently returns miss → falls
   through to body march via the new pop loop from commit `7df88a0`).
7. Shading: hit-report `body_xyz = c_body_cell + J_cell · residual · 3`
   where `c_body_cell` is recomputed from face-space coords at hit
   time.

Most of these helpers exist in sphere.wgsl today
(`face_frame_jacobian_shader`, `mat3_inv_shader`, `mat3_mul_vec_shader`).
The `face_space_to_body_point` / `body_point_to_face_space` ports
added in commit `bcf5924` handle the shading-time body-XYZ
reconstruction.

The CPU `unified_raycast` in `src/world/raycast/unified.rs` already
uses cell-local residual ∈ [0,1)³ exclusively, with no f32 quantity
that scales with global tree depth. Porting that DDA structure to
WGSL — replacing the body of `sphere_in_sub_frame` with a residual
DDA — is the work that breaks past the layer-10 wall.

## Pre-existing scaffolding ready for the WGSL port

- `assets/shaders/sphere.wgsl`: `face_uv_to_dir`, `face_space_to_body_point`,
  `body_point_to_face_space` (commit `bcf5924`) — face-coord conversions,
  needed for a residual DDA inside a face subtree.
- `assets/shaders/sphere.wgsl`: existing `face_frame_jacobian_shader`
  + `mat3_inv_shader` + `mat3_mul_vec_shader` — Jacobian primitives.
- `assets/shaders/march.wgsl`: `unified_dda` single dispatch entry —
  the residual DDA replaces the call to `sphere_in_sub_frame` here.
