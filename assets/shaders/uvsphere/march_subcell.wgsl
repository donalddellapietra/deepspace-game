// `march_uv_subcell`: sub-cell marcher.
//
// Used when the active render frame is a UV cell at body-tree depth
// `K ≥ 1`. The cell's `(phi_min, theta_min, r_min, dphi, dth, dr)`
// arrives via uniforms; the marcher operates entirely in the FRAME's
// cell-local fractions `(un_phi, un_theta, un_r) ∈ [0, 1]³`.
//
// Precision discipline:
//
// 1. State is `t` (world-time along the ray) and the FRAME-level
//    rates `d_un_*_frame` derived from the basis at the frame's
//    centre. The current `un_*_frame(t)` is `cam_un + d_un * t`.
//    Cam_un and `t` are O(1) values; the rates are constants. No
//    f32 catastrophic cancellation.
//
// 2. `un_*_frame` is recomputed from `t` once per iteration (NOT
//    derived from `world_pos → atan2 → phi_w → (phi_w − phi_min) /
//    frame_dphi`, which was the broken path). f32 precision in `t`
//    is ~1e-7 absolute and `d_un_*` is `O(1)`, so `un_*_frame` has
//    ~1e-7 absolute precision in `[0, 1]`.
//
// 3. The cell descent below the frame is capped at
//    `UV_SUBCELL_DESCENT` so the `un *= 3 - tier` amplification of
//    the frame-level `un_*` precision doesn't consume a leaf cell.
//    This caps the visible fineness at frame_depth +
//    UV_SUBCELL_DESCENT — the precision cliff, NOT a hardware limit.
//
// 4. Boundary nudge `ε` is in CELL-LOCAL FRACTION units (~1e-4 of
//    a cell), converted to world-time via the cell's per-axis rate.
//    A fixed world-distance ε would overshoot multiple cells at
//    deep descent.
//
// Termination:
// - Ray exits the body's outer shell (sky).
// - Ray exits the frame's `(φ, θ, r)` range — the marcher falls
//   back to `march_uv_sphere(body_node_idx, ...)` so the rest of
//   the body still renders. Without the fallback, a ray going
//   anywhere outside the camera's UV sub-cell reads as sky and the
//   sphere appears cut in half.
// - Hit a Block leaf — return hit.
// - Inner-core hit — return as stone.
// - Iteration cap.

fn march_uv_subcell(
    frame_node_idx: u32,
    body_node_idx: u32,
    body_inner_r: f32, body_outer_r: f32, body_theta_cap: f32,
    phi_min: f32, theta_min: f32, r_min: f32,
    frame_dphi: f32, frame_dth: f32, frame_dr: f32,
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 3.0;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    let body_size = 3.0;
    let center = vec3<f32>(body_size * 0.5);
    let oc = ray_origin - center;

    let outer_t = uv_ray_sphere(oc, ray_dir, body_outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y { return result; }
    let inside_outer = dot(oc, oc) <= body_outer_r * body_outer_r;
    var t: f32 = select(max(outer_t.x, 0.0001), 0.0001, inside_outer);
    let t_exit_outer = outer_t.y;

    // Frame-centre basis + Jacobian: constants for the whole march.
    let phi_c = phi_min + frame_dphi * 0.5;
    let theta_c = theta_min + frame_dth * 0.5;
    let r_c = r_min + frame_dr * 0.5;
    let basis = uv_basis_at(phi_c, theta_c, r_c);
    let d_un_frame = uv_d_un(basis, ray_dir, frame_dphi, frame_dth, frame_dr);

    // Cam_un_*: camera position projected onto the frame's [0, 1]³
    // fractions. Computed once at march entry from the camera's
    // body-frame cartesian coords (`ray_origin` IS the camera under
    // `gpu_camera_for_frame`, so `oc` here equals the camera offset
    // from body centre).
    let cam_r = max(length(oc), 1e-6);
    var cam_phi = atan2(oc.z, oc.x);
    if cam_phi < 0.0 { cam_phi += UV_TWO_PI; }
    let cam_theta = asin(clamp(oc.y / cam_r, -1.0, 1.0));
    let cam_un_phi = (cam_phi - phi_min) / max(frame_dphi, 1e-30);
    let cam_un_theta = (cam_theta - theta_min) / max(frame_dth, 1e-30);
    let cam_un_r = (cam_r - r_min) / max(frame_dr, 1e-30);

    var iter: u32 = 0u;
    loop {
        if iter >= UV_MAX_ITER { break; }
        iter += 1u;
        if t > t_exit_outer + 1e-4 { break; }

        // Sanity: ray inside body shell?
        let pos = ray_origin + ray_dir * t;
        let off = pos - center;
        let r_w = length(off);
        if r_w > body_outer_r * 1.0001 { break; }
        if r_w < body_inner_r * 0.9999 {
            result.hit = true;
            result.t = t;
            result.normal = off / max(r_w, 1e-6);
            result.color = palette[0u].rgb;
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }
        if abs(asin(clamp(off.y / max(r_w, 1e-6), -1.0, 1.0))) > body_theta_cap {
            break;
        }

        // FRAME-level cell-local fractions at this `t`. These are the
        // ONLY un_* values that drive descent — never the world phi_w
        // path. Each addition is O(1)+O(1), no cancellation.
        let un_phi_frame = cam_un_phi + d_un_frame.x * t;
        let un_theta_frame = cam_un_theta + d_un_frame.y * t;
        let un_r_frame = cam_un_r + d_un_frame.z * t;

        // Frame-exit: fall back to the body-root marcher so the rest
        // of the body still renders. The fall-back redoes ray-vs-body-
        // shell entry from the original ray, which is correct even
        // when the ray has already advanced inside the sub-cell — any
        // hit inside the sub-cell would have returned earlier in this
        // loop. Slight redundant work for sub-cell rays whose first
        // descent lands outside the frame; cleaner than threading a
        // `t_resume` through the body-root marcher.
        if un_phi_frame < -1e-6 || un_phi_frame > 1.0 + 1e-6
            || un_theta_frame < -1e-6 || un_theta_frame > 1.0 + 1e-6
            || un_r_frame < -1e-6 || un_r_frame > 1.0 + 1e-6 {
            return march_uv_sphere(body_node_idx, ray_origin, ray_dir);
        }

        // Descend within frame, capped.
        let d = uv_descend_cell(
            frame_node_idx,
            frame_dphi, frame_dth, frame_dr,
            un_phi_frame, un_theta_frame, un_r_frame,
            UV_SUBCELL_DESCENT,
        );

        if d.found_block {
            // For face/normal/bevel, use the world (phi_w, theta_w)
            // at the hit position. These are O(1), and the bevel only
            // needs them for the cell's surface normal and the
            // tangent-arc length comparison — neither path hits the
            // f32 cancellation cliff.
            var phi_w = atan2(off.z, off.x);
            if phi_w < 0.0 { phi_w += UV_TWO_PI; }
            let theta_w = asin(clamp(off.y / max(r_w, 1e-6), -1.0, 1.0));
            let face = uv_hit_face(
                off, r_w, theta_w, phi_w,
                d.un_phi, d.un_theta, d.un_r,
                d.dphi, d.dth, d.dr,
            );
            let bevel = uv_cell_bevel(d.un_phi, d.un_theta, d.un_r, face.axis);
            result.hit = true;
            result.t = t;
            result.normal = face.normal;
            result.color = palette[d.block_type].rgb * (0.7 + 0.3 * bevel);
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }

        // Empty cell: step ray to next axis crossing at the descent
        // depth's resolution. d_un at descent depth = d_un_frame * 3^D.
        let scale = pow(3.0, f32(d.depth));
        let d_un_phi_d = d_un_frame.x * scale;
        let d_un_theta_d = d_un_frame.y * scale;
        let d_un_r_d = d_un_frame.z * scale;
        let step = uv_cell_step(
            d.un_phi, d.un_theta, d.un_r,
            d_un_phi_d, d_un_theta_d, d_un_r_d,
        );
        if step.t > 1e20 { break; }
        let eps = uv_boundary_eps(d_un_phi_d, d_un_theta_d, d_un_r_d);
        t = t + step.t + eps;
    }
    return result;
}
