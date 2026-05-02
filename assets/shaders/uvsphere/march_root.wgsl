// `march_uv_sphere`: body-root marcher.
//
// Direct port of the CPU walker (`src/world/raycast/uvsphere.rs`)
// that handles deep-cell rendering correctly.
//
// Per-iteration loop:
// 1. World position at current `t`.
// 2. World `(φ, θ, r)` from the position.
// 3. Inner-core / outer-shell / θ-cap exit checks.
// 4. Tree descent that tracks ABSOLUTE cell bounds — no cell-local
//    `un *= 3 − tier` chain. The previous shader's chain amplified
//    a `1` ULP `phi_w` change to `3^K` ULPs of leaf cell, breaking
//    rendering at depth 10+ once a break had materialized cells
//    that deep.
// 5. On Block: face/bevel from absolute bounds, return hit.
// 6. On Empty: step `t` to the next cell boundary via ray-vs-
//    sphere/cone/φ-plane intersection (`uv_next_boundary`). All
//    intersections are in O(1) body-frame world coords; numerically
//    stable at any descent depth.

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

    let oc = ray_origin - center;

    let outer_t = uv_ray_sphere(oc, ray_dir, outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y { return result; }
    let inside_outer = dot(oc, oc) <= outer_r * outer_r;
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

        let d = uv_descend_cell(
            body_node_idx,
            0.0, UV_TWO_PI,
            -theta_cap, theta_cap,
            inner_r, outer_r,
            phi_w, theta_w, r_w,
            UV_MAX_DEPTH,
        );

        if d.found_block {
            let face = uv_hit_face(
                off, r_w, theta_w, phi_w,
                d.phi_lo, d.phi_hi,
                d.theta_lo, d.theta_hi,
                d.r_lo, d.r_hi,
            );
            let bevel = uv_cell_bevel_abs(
                phi_w, theta_w, r_w,
                d.phi_lo, d.phi_hi,
                d.theta_lo, d.theta_hi,
                d.r_lo, d.r_hi,
                face.axis,
            );
            result.hit = true;
            result.t = t;
            result.normal = face.normal;
            result.color = palette[d.block_type].rgb * (0.7 + 0.3 * bevel);
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }

        // Empty cell — step ray to next absolute cell boundary. No
        // cell-local fraction arithmetic; stable at any depth.
        let t_next = uv_next_boundary(
            oc, ray_dir, t,
            d.phi_lo, d.phi_hi,
            d.theta_lo, d.theta_hi,
            d.r_lo, d.r_hi,
        );
        if t_next > 1e20 { break; }
        let step = t_next - t;
        t = t_next + max(step * 1e-4, 1e-5);
    }
    return result;
}
