// `march_uv_sphere`: body-root marcher.
//
// Per-iteration loop:
// 1. World position at current `t` → world `(φ, θ, r)`.
// 2. Inner-core / outer-shell / θ-cap exit checks.
// 3. Tree descent with delta-tracked tier picking
//    (`uv_descend_cell` in `cell.wgsl`).
// 4. On Block: face normal + bevel from the LAST entry axis the
//    ray crossed to land in this cell — recorded by `uv_next_boundary`
//    on the previous iteration, or seeded to `+r` (outer-shell entry)
//    on the first iteration. NEVER auto-picked from arc distance —
//    that mis-picks near cell edges, producing triangular sub-cell
//    artifacts in the bevel.
// 5. On Empty: step to the next cell-face crossing via
//    `uv_next_boundary`; record the axis for the NEXT iteration's
//    face/bevel.

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

    // PROTOTYPE: fire ray vs the cartesian-block target. The OBB
    // REPLACES one specific cell (path [14, 21, 23], outer-r slot
    // at depth 3, lying on the grass band). If the ray hits the
    // OBB, render it directly — short-circuit the UV march. The
    // body's normal grass cell at this position is REPLACED, not
    // composited with, so we don't need the t-comparison: any pixel
    // that hits the OBB is "in" the cartesian-rendered cell.
    let proto = proto_ray_vs_obb(ray_origin, ray_dir, center);
    if proto.t < 1e20 {
        return proto_obb_render(ray_origin, ray_dir, center, proto);
    }

    let oc = ray_origin - center;

    let outer_t = uv_ray_sphere(oc, ray_dir, outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y { return result; }
    let inside_outer = dot(oc, oc) <= outer_r * outer_r;
    var t: f32 = select(max(outer_t.x, 0.0001), 0.0001, inside_outer);
    let t_exit_outer = outer_t.y;

    // Entry-axis tracking. Initial: when the ray enters the body
    // shell from outside, it crosses the outer sphere = +r face.
    // For inside-body cameras the seed value is unused (the first
    // iteration finds a Block immediately or steps to a new cell).
    var last_axis: u32 = 2u;
    var last_side: u32 = 1u;

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
            // Inner-core hit — radially-inward face.
            result.hit = true;
            result.t = t;
            result.normal = -off / max(r_w, 1e-6);
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
            let normal = uv_face_normal(
                off, r_w, theta_w, phi_w, last_axis, last_side,
            );
            let bevel = uv_cell_bevel(
                phi_w, theta_w, r_w,
                d.phi_lo, d.phi_hi,
                d.theta_lo, d.theta_hi,
                d.r_lo, d.r_hi,
                last_axis,
            );
            result.hit = true;
            result.t = t;
            result.normal = normal;
            result.color = palette[d.block_type].rgb * (0.7 + 0.3 * bevel);
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }

        // Empty cell — step to next cell-face crossing.
        let bd = uv_next_boundary(
            oc, ray_dir, t,
            d.phi_lo, d.phi_hi,
            d.theta_lo, d.theta_hi,
            d.r_lo, d.r_hi,
        );
        if bd.t > 1e20 { break; }
        let step = bd.t - t;
        t = bd.t + max(step * 1e-4, 1e-5);
        last_axis = bd.axis;
        last_side = bd.side;
    }

    return result;
}
