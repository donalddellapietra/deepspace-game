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
