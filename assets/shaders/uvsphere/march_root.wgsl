// `march_uv_sphere`: body-root marcher.
//
// Used when the active render frame is the `UvSphereBody` node
// itself — the body's full `(φ, θ, r)` range is the cell. Per-cell
// basis is recomputed at each iteration's `(phi_w, theta_w, r_w)`
// because the basis varies sharply across the body (`dphi = 2π`).
//
// Precision: each iteration starts fresh from the world ray
// position, so f32 absolute body-frame ULPs (~1e-7) are the floor.
// Cells at body-tree depth N have arc-width `2π · r / 3^N`; at
// `r ≈ 0.5` and N ≈ 12, cell arc-width ≈ 1.2e-5, well above 1e-7.
// Past depth ≈ 14 precision degrades; that's the regime where
// the sub-cell marcher takes over via `compute_render_frame`'s
// `MAX_UV_FRAME_DEPTH` cap.

fn march_uv_sphere(
    body_node_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
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

    let body = node_kinds[body_node_idx];
    let inner_r_local = bitcast<f32>(body.param_a);
    let outer_r_local = bitcast<f32>(body.param_b);
    let theta_cap     = bitcast<f32>(body.param_c);
    let body_size = 3.0;
    let center = vec3<f32>(body_size * 0.5);
    let outer_r = outer_r_local * body_size;
    let inner_r = inner_r_local * body_size;
    let body_dphi = UV_TWO_PI;
    let body_dth = 2.0 * theta_cap;
    let body_dr = outer_r - inner_r;

    let oc_init = ray_origin - center;

    let outer_t = uv_ray_sphere(oc_init, ray_dir, outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y { return result; }
    let inside_outer = dot(oc_init, oc_init) <= outer_r * outer_r;
    var t: f32 = select(max(outer_t.x, 0.0001), 0.0001, inside_outer);
    let t_exit_outer = outer_t.y;

    var iter: u32 = 0u;
    loop {
        if iter >= UV_MAX_ITER { break; }
        iter += 1u;
        if t > t_exit_outer + 1e-4 { break; }

        let pos = ray_origin + ray_dir * t;
        let off = pos - center;
        let r_w = length(off);

        if r_w > outer_r * 1.0001 { break; }
        if r_w < inner_r * 0.9999 {
            // Inner-core hit.
            result.hit = true;
            result.t = t;
            result.normal = off / max(r_w, 1e-6);
            result.color = palette[0u].rgb;
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }

        let theta_w = asin(clamp(off.y / max(r_w, 1e-6), -1.0, 1.0));
        if abs(theta_w) > theta_cap { break; }
        var phi_w = atan2(off.z, off.x);
        if phi_w < 0.0 { phi_w += UV_TWO_PI; }

        // Body-cell un_*: normalize world (φ, θ, r) into [0, 1] of body.
        let un_phi_body = clamp(phi_w / body_dphi, 0.0, 1.0 - 1e-7);
        let un_theta_body = clamp(
            (theta_w + theta_cap) / body_dth, 0.0, 1.0 - 1e-7);
        let un_r_body = clamp(
            (r_w - inner_r) / body_dr, 0.0, 1.0 - 1e-7);

        let d = uv_descend_cell(
            body_node_idx, body_dphi, body_dth, body_dr,
            un_phi_body, un_theta_body, un_r_body,
            UV_MAX_DEPTH,
        );

        if d.found_block {
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

        // Empty cell — Jacobian DDA at the cell's depth.
        let basis = uv_basis_at(phi_w, theta_w, r_w);
        let d_un = uv_d_un(basis, ray_dir, d.dphi, d.dth, d.dr);
        let step = uv_cell_step(
            d.un_phi, d.un_theta, d.un_r,
            d_un.x, d_un.y, d_un.z,
        );
        if step.t > 1e20 { break; }
        let eps = uv_boundary_eps(d_un.x, d_un.y, d_un.z);
        t = t + step.t + eps;
    }
    return result;
}
