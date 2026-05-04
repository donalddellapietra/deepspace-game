fn ring_radius_after(oc: vec3<f32>, dir: vec3<f32>, radius: f32, after: f32) -> f32 {
    let a = dir.x * dir.x + dir.z * dir.z;
    if a < 1e-12 { return -1.0; }
    let b = 2.0 * (oc.x * dir.x + oc.z * dir.z);
    let c = oc.x * oc.x + oc.z * oc.z - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let inv_2a = 0.5 / a;
    let t0 = (-b - sq) * inv_2a;
    let t1 = (-b + sq) * inv_2a;
    let t_lo = min(t0, t1);
    let t_hi = max(t0, t1);
    if t_lo > after { return t_lo; }
    if t_hi > after { return t_hi; }
    return -1.0;
}

fn ring_meridian_after(oc: vec3<f32>, dir: vec3<f32>, angle: f32, after: f32) -> f32 {
    let n = vec3<f32>(-sin(angle), 0.0, cos(angle));
    let denom = dot(dir, n);
    if abs(denom) < 1e-12 { return -1.0; }
    let t = -dot(oc, n) / denom;
    if t > after { return t; }
    return -1.0;
}

fn ring_y_after(ray_origin: vec3<f32>, ray_dir: vec3<f32>, y: f32, after: f32) -> f32 {
    if abs(ray_dir.y) < 1e-12 { return -1.0; }
    let t = (y - ray_origin.y) / ray_dir.y;
    if t > after { return t; }
    return -1.0;
}

fn ring_coords(p: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
    let d = p - center;
    return vec3<f32>(atan2(d.z, d.x), length(d.xz), p.y);
}

fn ring_normal_for_cell(
    p: vec3<f32>,
    center: vec3<f32>,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
    y_lo: f32, y_hi: f32,
) -> vec3<f32> {
    let d = p - center;
    let rho = max(length(d.xz), 1e-6);
    let radial = vec3<f32>(d.x / rho, 0.0, d.z / rho);
    let theta = atan2(d.z, d.x);
    let dist_theta_lo = abs(theta - theta_lo) * rho;
    let dist_theta_hi = abs(theta - theta_hi) * rho;
    let dist_r_lo = abs(rho - r_lo);
    let dist_r_hi = abs(rho - r_hi);
    let dist_y_lo = abs(p.y - y_lo);
    let dist_y_hi = abs(p.y - y_hi);

    var best = dist_r_hi;
    var n = radial;
    if dist_r_lo < best { best = dist_r_lo; n = -radial; }
    if dist_y_hi < best { best = dist_y_hi; n = vec3<f32>(0.0, 1.0, 0.0); }
    if dist_y_lo < best { best = dist_y_lo; n = vec3<f32>(0.0, -1.0, 0.0); }
    if dist_theta_lo < best { best = dist_theta_lo; n = normalize(vec3<f32>(sin(theta_lo), 0.0, -cos(theta_lo))); }
    if dist_theta_hi < best { n = normalize(vec3<f32>(-sin(theta_hi), 0.0, cos(theta_hi))); }
    return n;
}

fn make_ring_hit(
    pos: vec3<f32>,
    center: vec3<f32>,
    t_param: f32,
    block_type: u32,
    theta: f32, rho: f32, y: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
    y_lo: f32, y_hi: f32,
    theta_step: f32, r_step: f32, y_step: f32,
) -> HitResult {
    var result: HitResult;
    let normal = ring_normal_for_cell(pos, center, theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi);
    let theta_in = clamp((theta - theta_lo) / theta_step, 0.0, 1.0);
    let r_in = clamp((rho - r_lo) / r_step, 0.0, 1.0);
    let y_in = clamp((y - y_lo) / y_step, 0.0, 1.0);
    let local = vec3<f32>(theta_in, r_in, y_in);
    let bevel = cube_face_bevel(local, normal);
    result.hit = true;
    result.t = t_param;
    result.color = palette[block_type].rgb * (0.82 + 0.18 * bevel);
    result.normal = normal;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = pos - local;
    result.cell_size = 1.0;
    return result;
}

fn ring_recompute_cell(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    t: f32,
    theta_org: f32,
    r_org: f32,
    y_org: f32,
    theta_step: f32,
    r_step: f32,
    y_step: f32,
) -> vec3<i32> {
    let uv = ring_coords(ray_origin + ray_dir * t, center);
    return vec3<i32>(
        i32(floor((uv.x - theta_org) / theta_step)),
        i32(floor((uv.y - r_org) / r_step)),
        i32(floor((uv.z - y_org) / y_step)),
    );
}

fn ring_point_in_shell(
    p: vec3<f32>,
    center: vec3<f32>,
    r_lo: f32,
    r_hi: f32,
    y_lo: f32,
    y_hi: f32,
) -> bool {
    let d = p - center;
    let rho = length(d.xz);
    return rho >= r_lo - 1e-5 && rho <= r_hi + 1e-5
        && p.y >= y_lo - 1e-5 && p.y <= y_hi + 1e-5;
}

fn ring_shell_consider_t(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    r_lo: f32,
    r_hi: f32,
    y_lo: f32,
    y_hi: f32,
    t: f32,
    best: f32,
) -> f32 {
    if t < 0.0 || t >= best { return best; }
    let p = ray_origin + ray_dir * (t + 1e-5);
    if ring_point_in_shell(p, center, r_lo, r_hi, y_lo, y_hi) {
        return t;
    }
    return best;
}

fn ring_shell_entry_after(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    oc: vec3<f32>,
    r_lo: f32,
    r_hi: f32,
    y_lo: f32,
    y_hi: f32,
    after: f32,
) -> f32 {
    let start_t = max(after, 0.0);
    if ring_point_in_shell(ray_origin + ray_dir * start_t, center, r_lo, r_hi, y_lo, y_hi) {
        return start_t;
    }

    var best = 1e20;

    let r0_a = ring_radius_after(oc, ray_dir, max(r_lo, 1e-5), after);
    best = ring_shell_consider_t(ray_origin, ray_dir, center, r_lo, r_hi, y_lo, y_hi, r0_a, best);
    let r0_b = ring_radius_after(oc, ray_dir, max(r_lo, 1e-5), r0_a + 1e-5);
    best = ring_shell_consider_t(ray_origin, ray_dir, center, r_lo, r_hi, y_lo, y_hi, r0_b, best);

    let r1_a = ring_radius_after(oc, ray_dir, r_hi, after);
    best = ring_shell_consider_t(ray_origin, ray_dir, center, r_lo, r_hi, y_lo, y_hi, r1_a, best);
    let r1_b = ring_radius_after(oc, ray_dir, r_hi, r1_a + 1e-5);
    best = ring_shell_consider_t(ray_origin, ray_dir, center, r_lo, r_hi, y_lo, y_hi, r1_b, best);

    let y0 = ring_y_after(ray_origin, ray_dir, y_lo, after);
    best = ring_shell_consider_t(ray_origin, ray_dir, center, r_lo, r_hi, y_lo, y_hi, y0, best);
    let y1 = ring_y_after(ray_origin, ray_dir, y_hi, after);
    best = ring_shell_consider_t(ray_origin, ray_dir, center, r_lo, r_hi, y_lo, y_hi, y1, best);

    return best;
}

fn ring_top_cell_x(p: vec3<f32>, center: vec3<f32>, dims_x: i32, theta_step: f32) -> i32 {
    let theta = atan2(p.z - center.z, p.x - center.x);
    return clamp(i32(floor((theta + 3.14159265) / theta_step)), 0, dims_x - 1);
}

fn ring_next_cell_t(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    oc: vec3<f32>,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
    y_lo: f32, y_hi: f32,
    after: f32,
    t_exit: f32,
) -> f32 {
    var t_next = t_exit + 1.0;
    let t_theta_lo = ring_meridian_after(oc, ray_dir, theta_lo, after);
    if t_theta_lo > 0.0 && t_theta_lo < t_next { t_next = t_theta_lo; }
    let t_theta_hi = ring_meridian_after(oc, ray_dir, theta_hi, after);
    if t_theta_hi > 0.0 && t_theta_hi < t_next { t_next = t_theta_hi; }
    let t_r_lo = ring_radius_after(oc, ray_dir, max(r_lo, 1e-5), after);
    if t_r_lo > 0.0 && t_r_lo < t_next { t_next = t_r_lo; }
    let t_r_hi = ring_radius_after(oc, ray_dir, r_hi, after);
    if t_r_hi > 0.0 && t_r_hi < t_next { t_next = t_r_hi; }
    let t_y_lo = ring_y_after(ray_origin, ray_dir, y_lo, after);
    if t_y_lo > 0.0 && t_y_lo < t_next { t_next = t_y_lo; }
    let t_y_hi = ring_y_after(ray_origin, ray_dir, y_hi, after);
    if t_y_hi > 0.0 && t_y_hi < t_next { t_next = t_y_hi; }
    return t_next;
}

const UV_RING_STACK_DEPTH: u32 = 24u;

fn march_uv_ring_anchor(
    anchor_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    oc: vec3<f32>,
    t_in: f32,
    t_exit: f32,
    slab_theta_lo: f32,
    slab_theta_step: f32,
    slab_r_lo: f32,
    slab_r_step: f32,
    slab_y_lo: f32,
    slab_y_step: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    var s_node_idx: array<u32, UV_RING_STACK_DEPTH>;
    var s_cell: array<u32, UV_RING_STACK_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;

    var cur_theta_org = slab_theta_lo;
    var cur_r_org = slab_r_lo;
    var cur_y_org = slab_y_lo;
    var cur_theta_step = slab_theta_step / 3.0;
    var cur_r_step = slab_r_step / 3.0;
    var cur_y_step = slab_y_step / 3.0;

    var t = max(t_in, 0.0) + 1e-5;
    s_cell[0] = pack_cell(clamp(
        ring_recompute_cell(
            ray_origin, ray_dir, center, t,
            cur_theta_org, cur_r_org, cur_y_org,
            cur_theta_step, cur_r_step, cur_y_step,
        ),
        vec3<i32>(0),
        vec3<i32>(2),
    ));

    var iters: u32 = 0u;
    loop {
        if iters > 2048u { break; }
        iters = iters + 1u;
        if t > t_exit { break; }

        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth = depth - 1u;
            cur_theta_step = cur_theta_step * 3.0;
            cur_r_step = cur_r_step * 3.0;
            cur_y_step = cur_y_step * 3.0;
            let popped = unpack_cell(s_cell[depth]);
            cur_theta_org = cur_theta_org - f32(popped.x) * cur_theta_step;
            cur_r_org = cur_r_org - f32(popped.y) * cur_r_step;
            cur_y_org = cur_y_org - f32(popped.z) * cur_y_step;
            s_cell[depth] = pack_cell(ring_recompute_cell(
                ray_origin, ray_dir, center, t,
                cur_theta_org, cur_r_org, cur_y_org,
                cur_theta_step, cur_r_step, cur_y_step,
            ));
            continue;
        }

        let theta_lo = cur_theta_org + f32(cell.x) * cur_theta_step;
        let theta_hi = theta_lo + cur_theta_step;
        let r_lo = cur_r_org + f32(cell.y) * cur_r_step;
        let r_hi = r_lo + cur_r_step;
        let y_lo = cur_y_org + f32(cell.z) * cur_y_step;
        let y_hi = y_lo + cur_y_step;
        let t_next = ring_next_cell_t(
            ray_origin, ray_dir, oc,
            theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi,
            t, t_exit,
        );

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;
        if (occ & bit) == 0u {
            t = t_next + max(min(min(cur_theta_step, cur_r_step), cur_y_step) * 1e-4, 1e-6);
            s_cell[depth] = pack_cell(ring_recompute_cell(
                ray_origin, ray_dir, center, t,
                cur_theta_org, cur_r_org, cur_y_org,
                cur_theta_step, cur_r_step, cur_y_step,
            ));
            continue;
        }

        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            let pos_h = ray_origin + ray_dir * t;
            let uv_h = ring_coords(pos_h, center);
            return make_ring_hit(
                pos_h, center, t, block_type,
                uv_h.x, uv_h.y, uv_h.z,
                theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi,
                cur_theta_step, cur_r_step, cur_y_step,
            );
        }

        if tag != 2u || block_type == 0xFFFEu {
            t = t_next + max(min(min(cur_theta_step, cur_r_step), cur_y_step) * 1e-4, 1e-6);
            s_cell[depth] = pack_cell(ring_recompute_cell(
                ray_origin, ray_dir, center, t,
                cur_theta_org, cur_r_org, cur_y_org,
                cur_theta_step, cur_r_step, cur_y_step,
            ));
            continue;
        }

        if depth + 1u >= UV_RING_STACK_DEPTH {
            let pos_h = ray_origin + ray_dir * t;
            let uv_h = ring_coords(pos_h, center);
            return make_ring_hit(
                pos_h, center, t, block_type,
                uv_h.x, uv_h.y, uv_h.z,
                theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi,
                cur_theta_step, cur_r_step, cur_y_step,
            );
        }

        let child_idx = tree[child_base + 1u];
        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_theta_org = theta_lo;
        cur_r_org = r_lo;
        cur_y_org = y_lo;
        cur_theta_step = cur_theta_step / 3.0;
        cur_r_step = cur_r_step / 3.0;
        cur_y_step = cur_y_step / 3.0;
        s_cell[depth] = pack_cell(clamp(
            ring_recompute_cell(
                ray_origin, ray_dir, center, t,
                cur_theta_org, cur_r_org, cur_y_org,
                cur_theta_step, cur_r_step, cur_y_step,
            ),
            vec3<i32>(0),
            vec3<i32>(2),
        ));
    }

    return result;
}

fn ring_point_in_top_cell(
    p: vec3<f32>,
    center: vec3<f32>,
    theta_lo: f32,
    theta_hi: f32,
    r_lo: f32,
    r_hi: f32,
    y_lo: f32,
    y_hi: f32,
) -> bool {
    let uv = ring_coords(p, center);
    return uv.x >= theta_lo - 1e-5 && uv.x <= theta_hi + 1e-5
        && uv.y >= r_lo - 1e-5 && uv.y <= r_hi + 1e-5
        && uv.z >= y_lo - 1e-5 && uv.z <= y_hi + 1e-5;
}

fn ring_top_entry_t(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    theta_lo: f32,
    theta_hi: f32,
    r_lo: f32,
    r_hi: f32,
    y_lo: f32,
    y_hi: f32,
) -> f32 {
    if ring_point_in_top_cell(ray_origin, center, theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi) {
        return 0.0;
    }
    let oc = ray_origin - center;
    var best = 1e20;
    let candidates = vec4<f32>(
        ring_meridian_after(oc, ray_dir, theta_lo, -1e-5),
        ring_meridian_after(oc, ray_dir, theta_hi, -1e-5),
        ring_radius_after(oc, ray_dir, r_lo, -1e-5),
        ring_radius_after(oc, ray_dir, r_hi, -1e-5),
    );
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let t = candidates[i];
        if t >= 0.0 && t < best {
            let p = ray_origin + ray_dir * (t + 1e-5);
            if ring_point_in_top_cell(p, center, theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi) {
                best = t;
            }
        }
    }
    let ty0 = ring_y_after(ray_origin, ray_dir, y_lo, -1e-5);
    if ty0 >= 0.0 && ty0 < best {
        let p = ray_origin + ray_dir * (ty0 + 1e-5);
        if ring_point_in_top_cell(p, center, theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi) {
            best = ty0;
        }
    }
    let ty1 = ring_y_after(ray_origin, ray_dir, y_hi, -1e-5);
    if ty1 >= 0.0 && ty1 < best {
        let p = ray_origin + ray_dir * (ty1 + 1e-5);
        if ring_point_in_top_cell(p, center, theta_lo, theta_hi, r_lo, r_hi, y_lo, y_hi) {
            best = ty1;
        }
    }
    return best;
}

fn march_uv_ring(
    ring_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    ray_origin: vec3<f32>,
    ray_dir_in: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let dims_x = i32(uniforms.slab_dims.x);
    let dims_z = i32(max(uniforms.slab_dims.z, 1u));
    let slab_depth = uniforms.slab_dims.w;
    if dims_x <= 0 { return result; }

    let center = body_origin + vec3<f32>(body_size * 0.5);
    let pi = 3.14159265;
    let theta_step = 2.0 * pi / f32(dims_x);
    let radius = body_size * 0.38;
    let side = max((2.0 * pi * radius / f32(dims_x)) * 0.95, body_size / 27.0);
    let half_side = side * 0.5;
    let r_lo = radius - half_side;
    let r_hi = radius + half_side;
    let height = side * f32(dims_z);
    let y_lo = center.y - height * 0.5;
    let y_hi = center.y + height * 0.5;
    let oc = ray_origin - center;

    var t = ring_shell_entry_after(
        ray_origin, ray_dir_in, center, oc,
        r_lo, r_hi, y_lo, y_hi,
        -1e-5,
    );

    for (var iter: u32 = 0u; iter < 128u; iter = iter + 1u) {
        if t >= 1e19 { break; }

        let probe_t = max(t, 0.0) + 1e-5;
        let probe = ray_origin + ray_dir_in * probe_t;
        if !ring_point_in_shell(probe, center, r_lo, r_hi, y_lo, y_hi) {
            t = ring_shell_entry_after(
                ray_origin, ray_dir_in, center, oc,
                r_lo, r_hi, y_lo, y_hi,
                probe_t,
            );
            continue;
        }

        let cell_x = ring_top_cell_x(probe, center, dims_x, theta_step);
        let cell_z = clamp(i32(floor((probe.y - y_lo) / side)), 0, dims_z - 1);
        let theta_lo = -pi + f32(cell_x) * theta_step;
        let theta_hi = theta_lo + theta_step;
        let cell_y_lo = y_lo + f32(cell_z) * side;
        let cell_y_hi = cell_y_lo + side;
        let t_next = ring_next_cell_t(
            ray_origin, ray_dir_in, oc,
            theta_lo, theta_hi, r_lo, r_hi, cell_y_lo, cell_y_hi,
            probe_t, 1e20,
        );
        let sample = sample_slab_cell(ring_idx, slab_depth, cell_x, 0, cell_z);

        if sample.tag == 2u {
            let sub = march_uv_ring_anchor(
                sample.child_idx,
                ray_origin, ray_dir_in, center, oc,
                probe_t,
                t_next,
                theta_lo, theta_step,
                r_lo, side,
                cell_y_lo, side,
            );
            if sub.hit { return sub; }
        } else if sample.block_type != 0xFFFEu {
            let pos_h = ray_origin + ray_dir_in * probe_t;
            let uv_h = ring_coords(pos_h, center);
            return make_ring_hit(
                pos_h, center, probe_t, sample.block_type,
                uv_h.x, uv_h.y, uv_h.z,
                theta_lo, theta_hi, r_lo, r_hi, cell_y_lo, cell_y_hi,
                theta_step, side, side,
            );
        }

        t = t_next + 1e-5;
    }

    return result;
}
