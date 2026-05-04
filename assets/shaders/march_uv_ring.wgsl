fn uv_ring_angle_in_cell(angle: f32, cell_x: i32, angle_step: f32) -> bool {
    let pi = 3.14159265;
    let lo = -pi + f32(cell_x) * angle_step;
    let hi = lo + angle_step;
    return angle >= lo - 1e-5 && angle <= hi + 1e-5;
}

fn uv_ring_point_in_cell(
    p: vec3<f32>,
    center: vec3<f32>,
    radius: f32,
    half_side: f32,
    cell_x: i32,
    angle_step: f32,
) -> bool {
    let d = p - center;
    let rho = length(d.xz);
    let angle = atan2(d.z, d.x);
    return rho >= radius - half_side - 1e-5
        && rho <= radius + half_side + 1e-5
        && d.y >= -half_side - 1e-5
        && d.y <= half_side + 1e-5
        && uv_ring_angle_in_cell(angle, cell_x, angle_step);
}

fn uv_ring_first_cylinder_t(
    oc: vec3<f32>,
    dir: vec3<f32>,
    cyl_radius: f32,
    after: f32,
) -> f32 {
    let a = dir.x * dir.x + dir.z * dir.z;
    if a < 1e-12 { return -1.0; }
    let b = 2.0 * (oc.x * dir.x + oc.z * dir.z);
    let c = oc.x * oc.x + oc.z * oc.z - cyl_radius * cyl_radius;
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

fn uv_ring_second_cylinder_t(
    oc: vec3<f32>,
    dir: vec3<f32>,
    cyl_radius: f32,
    after: f32,
) -> f32 {
    let a = dir.x * dir.x + dir.z * dir.z;
    if a < 1e-12 { return -1.0; }
    let b = 2.0 * (oc.x * dir.x + oc.z * dir.z);
    let c = oc.x * oc.x + oc.z * oc.z - cyl_radius * cyl_radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let inv_2a = 0.5 / a;
    let t0 = (-b - sq) * inv_2a;
    let t1 = (-b + sq) * inv_2a;
    let t_lo = min(t0, t1);
    let t_hi = max(t0, t1);
    if t_lo > after && t_hi > after { return t_hi; }
    return -1.0;
}

fn uv_ring_meridian_t(
    oc: vec3<f32>,
    dir: vec3<f32>,
    angle: f32,
    after: f32,
) -> f32 {
    let n = vec3<f32>(-sin(angle), 0.0, cos(angle));
    let denom = dot(dir, n);
    if abs(denom) < 1e-12 { return -1.0; }
    let t = -dot(oc, n) / denom;
    if t > after { return t; }
    return -1.0;
}

fn uv_ring_surface_normal(
    p: vec3<f32>,
    center: vec3<f32>,
    uv_min: vec3<f32>,
    uv_step: vec3<f32>,
) -> vec3<f32> {
    let d = p - center;
    let rho = max(length(d.xz), 1e-6);
    let radial = vec3<f32>(d.x / rho, 0.0, d.z / rho);
    let lo = uv_min.x;
    let hi = uv_min.x + uv_step.x;
    let angle = atan2(d.z, d.x);
    let dist_inner = abs(rho - uv_min.y);
    let dist_outer = abs(rho - (uv_min.y + uv_step.y));
    let dist_y_lo = abs(p.y - uv_min.z);
    let dist_y_hi = abs(p.y - (uv_min.z + uv_step.z));
    let dist_lo = abs(angle - lo) * rho;
    let dist_hi = abs(angle - hi) * rho;

    var best = dist_outer;
    var n = radial;
    if dist_inner < best { best = dist_inner; n = -radial; }
    if dist_y_hi < best { best = dist_y_hi; n = vec3<f32>(0.0, 1.0, 0.0); }
    if dist_y_lo < best { best = dist_y_lo; n = vec3<f32>(0.0, -1.0, 0.0); }
    if dist_lo < best { best = dist_lo; n = normalize(vec3<f32>(sin(lo), 0.0, -cos(lo))); }
    if dist_hi < best { n = normalize(vec3<f32>(-sin(hi), 0.0, cos(hi))); }
    return n;
}

fn uv_ring_point_in_bounds(
    p: vec3<f32>,
    center: vec3<f32>,
    uv_min: vec3<f32>,
    uv_step: vec3<f32>,
) -> bool {
    let d = p - center;
    let rho = length(d.xz);
    let angle = atan2(d.z, d.x);
    return angle >= uv_min.x - 1e-5
        && angle <= uv_min.x + uv_step.x + 1e-5
        && rho >= uv_min.y - 1e-5
        && rho <= uv_min.y + uv_step.y + 1e-5
        && p.y >= uv_min.z - 1e-5
        && p.y <= uv_min.z + uv_step.z + 1e-5;
}

fn uv_ring_next_boundary_t(
    oc: vec3<f32>,
    dir: vec3<f32>,
    center: vec3<f32>,
    uv_min: vec3<f32>,
    uv_step: vec3<f32>,
    after: f32,
) -> f32 {
    var t_next = 1e20;
    let eps = 1e-5;
    let t_a0 = uv_ring_meridian_t(oc, dir, uv_min.x, after);
    if t_a0 > 0.0 && t_a0 < t_next {
        let p = center + oc + dir * max(t_a0 - eps, after);
        if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_a0; }
    }
    let t_a1 = uv_ring_meridian_t(oc, dir, uv_min.x + uv_step.x, after);
    if t_a1 > 0.0 && t_a1 < t_next {
        let p = center + oc + dir * max(t_a1 - eps, after);
        if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_a1; }
    }
    let t_r0 = uv_ring_first_cylinder_t(oc, dir, max(uv_min.y, 1e-5), after);
    if t_r0 > 0.0 && t_r0 < t_next {
        let p = center + oc + dir * max(t_r0 - eps, after);
        if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_r0; }
    }
    let t_r1 = uv_ring_second_cylinder_t(oc, dir, max(uv_min.y, 1e-5), after);
    if t_r1 > 0.0 && t_r1 < t_next {
        let p = center + oc + dir * max(t_r1 - eps, after);
        if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_r1; }
    }
    let t_r2 = uv_ring_first_cylinder_t(oc, dir, uv_min.y + uv_step.y, after);
    if t_r2 > 0.0 && t_r2 < t_next {
        let p = center + oc + dir * max(t_r2 - eps, after);
        if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_r2; }
    }
    let t_r3 = uv_ring_second_cylinder_t(oc, dir, uv_min.y + uv_step.y, after);
    if t_r3 > 0.0 && t_r3 < t_next {
        let p = center + oc + dir * max(t_r3 - eps, after);
        if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_r3; }
    }
    if abs(dir.y) > 1e-8 {
        let t_y0 = (uv_min.z - center.y - oc.y) / dir.y;
        if t_y0 > after && t_y0 < t_next {
            let p = center + oc + dir * max(t_y0 - eps, after);
            if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_y0; }
        }
        let t_y1 = (uv_min.z + uv_step.z - center.y - oc.y) / dir.y;
        if t_y1 > after && t_y1 < t_next {
            let p = center + oc + dir * max(t_y1 - eps, after);
            if uv_ring_point_in_bounds(p, center, uv_min, uv_step) { t_next = t_y1; }
        }
    }
    return t_next;
}

fn uv_ring_uv_at(p: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
    let d = p - center;
    return vec3<f32>(atan2(d.z, d.x), length(d.xz), p.y);
}

const UV_RING_STACK_DEPTH: u32 = 24u;

fn march_uv_ring_subtree(
    root_node_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    uv_root_min: vec3<f32>,
    uv_root_step: vec3<f32>,
    t_enter: f32,
    t_exit: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    var s_node_idx: array<u32, UV_RING_STACK_DEPTH>;
    var s_uv_min: array<vec3<f32>, UV_RING_STACK_DEPTH>;
    var s_uv_step: array<vec3<f32>, UV_RING_STACK_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = root_node_idx;
    s_uv_min[0] = uv_root_min;
    s_uv_step[0] = uv_root_step;

    let oc = ray_origin - center;
    var t = max(t_enter, 0.0) + 1e-5;
    var iterations: u32 = 0u;
    loop {
        if iterations > 4096u || t > t_exit || depth >= UV_RING_STACK_DEPTH { break; }
        iterations = iterations + 1u;

        let node_min = s_uv_min[depth];
        let node_step = s_uv_step[depth];
        let p = ray_origin + ray_dir * t;
        if !uv_ring_point_in_bounds(p, center, node_min, node_step) {
            if depth == 0u { break; }
            depth = depth - 1u;
            continue;
        }

        let uv = uv_ring_uv_at(p, center);
        let rel = (uv - node_min) / node_step;
        let cell = vec3<i32>(
            clamp(i32(floor(rel.x * 3.0)), 0, 2),
            clamp(i32(floor(rel.y * 3.0)), 0, 2),
            clamp(i32(floor(rel.z * 3.0)), 0, 2),
        );
        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let child_step = node_step / 3.0;
        let child_min = node_min + vec3<f32>(cell) * child_step;

        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;
        if (occ & bit) == 0u {
            let t_next = uv_ring_next_boundary_t(oc, ray_dir, center, child_min, child_step, t);
            if t_next >= 1e19 { break; }
            t = t_next + max(min(min(child_step.x, child_step.y), child_step.z) * 1e-4, 1e-6);
            continue;
        }

        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u || depth + 1u >= UV_RING_STACK_DEPTH {
            let hit_world = ray_origin + ray_dir * t;
            let normal = uv_ring_surface_normal(hit_world, center, child_min, child_step);
            let local = clamp((uv_ring_uv_at(hit_world, center) - child_min) / child_step, vec3<f32>(0.0), vec3<f32>(1.0));
            let bevel = cube_face_bevel(local, normal);
            result.hit = true;
            result.t = t;
            result.color = palette[block_type].rgb * (0.82 + 0.18 * bevel);
            result.normal = normal;
            result.cell_min = hit_world - local;
            result.cell_size = 1.0;
            return result;
        }

        if tag != 2u || block_type == 0xFFFEu {
            let t_next = uv_ring_next_boundary_t(oc, ray_dir, center, child_min, child_step, t);
            if t_next >= 1e19 { break; }
            t = t_next + max(min(min(child_step.x, child_step.y), child_step.z) * 1e-4, 1e-6);
            continue;
        }

        depth = depth + 1u;
        s_node_idx[depth] = tree[child_base + 1u];
        s_uv_min[depth] = child_min;
        s_uv_step[depth] = child_step;
        t = t + max(min(min(child_step.x, child_step.y), child_step.z) * 1e-5, 1e-7);
    }

    return result;
}

fn uv_ring_cell_local(
    p: vec3<f32>,
    center: vec3<f32>,
    radius: f32,
    side: f32,
    cell_x: i32,
    angle_step: f32,
) -> vec3<f32> {
    let pi = 3.14159265;
    let d = p - center;
    let rho = length(d.xz);
    let angle = atan2(d.z, d.x);
    let cell_center_angle = -pi + (f32(cell_x) + 0.5) * angle_step;
    return vec3<f32>(
        ((angle - cell_center_angle) / angle_step) * 3.0 + 1.5,
        ((rho - radius) / side) * 3.0 + 1.5,
        (d.y / side) * 3.0 + 1.5,
    );
}

fn uv_ring_consider_t(
    candidate_t: f32,
    cell_x: i32,
    center: vec3<f32>,
    radius: f32,
    half_side: f32,
    angle_step: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    best_t: ptr<function, f32>,
) {
    if candidate_t < 0.0 || candidate_t >= (*best_t) {
        return;
    }
    let probe = ray_origin + ray_dir * (candidate_t + 1e-5);
    if uv_ring_point_in_cell(probe, center, radius, half_side, cell_x, angle_step) {
        *best_t = candidate_t;
    }
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
    let slab_depth = uniforms.slab_dims.w;
    if dims_x <= 0 { return result; }

    let center = body_origin + vec3<f32>(body_size * 0.5);
    let pi = 3.14159265;
    let angle_step = 2.0 * pi / f32(dims_x);
    let radius = body_size * 0.38;
    let side = max((2.0 * pi * radius / f32(dims_x)) * 0.95, body_size / 27.0);
    let half_side = side * 0.5;
    let oc = ray_origin - center;

    var best_t = 1e20;
    var best: HitResult = result;
    for (var cell_x: i32 = 0; cell_x < dims_x; cell_x = cell_x + 1) {
        let sample = sample_slab_cell(ring_idx, slab_depth, cell_x, 0, 0);
        if sample.block_type == 0xFFFEu {
            continue;
        }

        let lo = -pi + f32(cell_x) * angle_step;
        let hi = lo + angle_step;
        var cell_t = 1e20;

        if uv_ring_point_in_cell(ray_origin, center, radius, half_side, cell_x, angle_step) {
            cell_t = 0.0;
        }

        let t_outer0 = uv_ring_first_cylinder_t(oc, ray_dir_in, radius + half_side, -1e-5);
        uv_ring_consider_t(t_outer0, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);
        let t_outer1 = uv_ring_second_cylinder_t(oc, ray_dir_in, radius + half_side, -1e-5);
        uv_ring_consider_t(t_outer1, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);
        let inner_radius = max(radius - half_side, 1e-5);
        let t_inner0 = uv_ring_first_cylinder_t(oc, ray_dir_in, inner_radius, -1e-5);
        uv_ring_consider_t(t_inner0, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);
        let t_inner1 = uv_ring_second_cylinder_t(oc, ray_dir_in, inner_radius, -1e-5);
        uv_ring_consider_t(t_inner1, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);

        if abs(ray_dir_in.y) > 1e-8 {
            let t_y_lo = ((center.y - half_side) - ray_origin.y) / ray_dir_in.y;
            uv_ring_consider_t(t_y_lo, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);
            let t_y_hi = ((center.y + half_side) - ray_origin.y) / ray_dir_in.y;
            uv_ring_consider_t(t_y_hi, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);
        }

        let t_lo = uv_ring_meridian_t(oc, ray_dir_in, lo, -1e-5);
        uv_ring_consider_t(t_lo, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);
        let t_hi = uv_ring_meridian_t(oc, ray_dir_in, hi, -1e-5);
        uv_ring_consider_t(t_hi, cell_x, center, radius, half_side, angle_step, ray_origin, ray_dir_in, &cell_t);

        if cell_t < best_t {
            let cell_uv_min = vec3<f32>(lo, radius - half_side, center.y - half_side);
            let cell_uv_step = vec3<f32>(angle_step, side, side);
            if sample.tag == 2u {
                let sub = march_uv_ring_subtree(
                    sample.child_idx, ray_origin, ray_dir_in, center,
                    cell_uv_min, cell_uv_step, cell_t, 1e20,
                );
                if sub.hit && sub.t < best_t {
                    best_t = sub.t;
                    best = sub;
                }
                continue;
            }

            let hit_world = ray_origin + ray_dir_in * cell_t;
            let normal = uv_ring_surface_normal(hit_world, center, cell_uv_min, cell_uv_step);
            let local_in_cell = clamp(
                uv_ring_cell_local(hit_world, center, radius, side, cell_x, angle_step) / 3.0,
                vec3<f32>(0.0), vec3<f32>(1.0),
            );
            let bevel = cube_face_bevel(local_in_cell, normal);
            var out: HitResult;
            out.hit = true;
            out.t = cell_t;
            out.color = palette[sample.block_type].rgb * (0.82 + 0.18 * bevel);
            out.normal = normal;
            out.frame_level = 0u;
            out.frame_scale = 1.0;
            out.cell_min = hit_world - local_in_cell;
            out.cell_size = 1.0;
            best_t = cell_t;
            best = out;
        }
    }

    return best;
}
