#include "tree.wgsl"

const UV_EMPTY: u32 = 0xFFFEu;

struct UvWalkResult {
    block: u32,
    phi_lo: f32,
    theta_lo: f32,
    r_lo: f32,
    size: f32,
    ratio_depth: u32,
}

fn uv_slot_at(abs_c: f32, lo: f32, child_size: f32) -> u32 {
    return u32(clamp(floor((abs_c - lo) / child_size), 0.0, 2.0));
}

fn body_point_to_uv(
    point_body: vec3<f32>,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    body_size: f32,
) -> vec3<f32> {
    let center = vec3<f32>(body_size * 0.5);
    let offset = point_body - center;
    let r = max(length(offset), 1e-8);
    var phi = atan2(offset.z, offset.x);
    if phi < 0.0 { phi = phi + 6.28318530718; }
    let theta = acos(clamp(offset.y / r, -1.0, 1.0));
    let theta_span = max(3.14159265359 - 2.0 * theta_cap, 1e-6);
    return vec3<f32>(
        clamp(phi / 6.28318530718, 0.0, 0.9999999),
        clamp((theta - theta_cap) / theta_span, 0.0, 0.9999999),
        clamp((r / body_size - inner_r) / max(outer_r - inner_r, 1e-6), 0.0, 0.9999999),
    );
}

fn uv_to_body_point(
    phi_n: f32,
    theta_n: f32,
    r_n: f32,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    body_size: f32,
) -> vec3<f32> {
    let center = vec3<f32>(body_size * 0.5);
    let radius = (inner_r + r_n * (outer_r - inner_r)) * body_size;
    let phi = phi_n * 6.28318530718;
    let theta_span = max(3.14159265359 - 2.0 * theta_cap, 1e-6);
    let theta = theta_cap + theta_n * theta_span;
    let sin_theta = sin(theta);
    let dir = vec3<f32>(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
    return center + dir * radius;
}

fn walk_uv_subtree(
    root_idx: u32,
    phi_n: f32,
    theta_n: f32,
    r_n: f32,
    max_depth: u32,
) -> UvWalkResult {
    var res: UvWalkResult;
    res.block = UV_EMPTY;
    res.phi_lo = 0.0;
    res.theta_lo = 0.0;
    res.r_lo = 0.0;
    res.size = 1.0;
    res.ratio_depth = 0u;

    var node_idx = root_idx;
    var phi_lo = 0.0;
    var theta_lo = 0.0;
    var r_lo = 0.0;
    var size = 1.0;
    for (var d: u32 = 1u; d <= max_depth; d = d + 1u) {
        let base = node_offsets[node_idx];
        if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
        let occupancy = tree[base];
        let first_child = tree[base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let child_size = size / 3.0;
        let ps = uv_slot_at(phi_n, phi_lo, child_size);
        let ts = uv_slot_at(theta_n, theta_lo, child_size);
        let rs = uv_slot_at(r_n, r_lo, child_size);
        let slot = rs * 9u + ts * 3u + ps;
        let child_phi_lo = phi_lo + f32(ps) * child_size;
        let child_theta_lo = theta_lo + f32(ts) * child_size;
        let child_r_lo = r_lo + f32(rs) * child_size;
        let slot_bit = 1u << slot;
        if (occupancy & slot_bit) == 0u {
            res.phi_lo = child_phi_lo;
            res.theta_lo = child_theta_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            res.ratio_depth = d;
            return res;
        }
        let rank = countOneBits(occupancy & (slot_bit - 1u));
        let packed = tree[first_child + rank * 2u];
        let child_idx = tree[first_child + rank * 2u + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let tag = packed & 0xFFu;
        if tag == 1u {
            res.block = child_block_type(packed);
            res.phi_lo = child_phi_lo;
            res.theta_lo = child_theta_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            res.ratio_depth = d;
            return res;
        }
        if d == max_depth {
            res.block = child_block_type(packed);
            res.phi_lo = child_phi_lo;
            res.theta_lo = child_theta_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            res.ratio_depth = d;
            return res;
        }
        node_idx = child_idx;
        phi_lo = child_phi_lo;
        theta_lo = child_theta_lo;
        r_lo = child_r_lo;
        size = child_size;
    }
    return res;
}

fn uv_plane_intersection(
    ray_origin_body: vec3<f32>,
    ray_dir: vec3<f32>,
    phi: f32,
    body_size: f32,
) -> f32 {
    let center = vec3<f32>(body_size * 0.5);
    let rel = ray_origin_body - center;
    let normal = vec3<f32>(sin(phi), 0.0, -cos(phi));
    let denom = dot(normal, ray_dir);
    if abs(denom) <= 1e-8 { return 1e30; }
    return -dot(normal, rel) / denom;
}

fn uv_theta_intersection(
    ray_origin_body: vec3<f32>,
    ray_dir: vec3<f32>,
    theta: f32,
    body_size: f32,
) -> f32 {
    if abs(theta - 1.57079632679) <= 1e-6 {
        if abs(ray_dir.y) <= 1e-8 { return 1e30; }
        return (body_size * 0.5 - ray_origin_body.y) / ray_dir.y;
    }
    let center = vec3<f32>(body_size * 0.5);
    let p = ray_origin_body - center;
    let cos2 = cos(theta) * cos(theta);
    let sin2 = sin(theta) * sin(theta);
    let a = (ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z) * cos2 - ray_dir.y * ray_dir.y * sin2;
    if abs(a) <= 1e-8 { return 1e30; }
    let b = 2.0 * ((p.x * ray_dir.x + p.z * ray_dir.z) * cos2 - p.y * ray_dir.y * sin2);
    let c = (p.x * p.x + p.z * p.z) * cos2 - p.y * p.y * sin2;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return 1e30; }
    let sq = sqrt(disc);
    let t0 = (-b - sq) / (2.0 * a);
    let t1 = (-b + sq) / (2.0 * a);
    let expected_sign = sign(cos(theta));
    var best = 1e30;
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let t = select(t0, t1, i == 1u);
        if !(t == t) || abs(t) >= 1e29 { continue; }
        if expected_sign != 0.0 {
            let y = p.y + ray_dir.y * t;
            if sign(y) != expected_sign { continue; }
        }
        if t < best { best = t; }
    }
    return best;
}

fn uv_sphere_intersection(
    ray_origin_body: vec3<f32>,
    ray_dir: vec3<f32>,
    radius: f32,
    body_size: f32,
) -> vec2<f32> {
    let center = vec3<f32>(body_size * 0.5);
    let oc = ray_origin_body - center;
    let a = dot(ray_dir, ray_dir);
    if a <= 1e-12 { return vec2<f32>(1e30, 1e30); }
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return vec2<f32>(1e30, 1e30); }
    let sq = sqrt(disc);
    let denom = 0.5 / a;
    return vec2<f32>((-b - sq) * denom, (-b + sq) * denom);
}

fn uv_sphere_in_cell(
    body_root_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    inner_r: f32,
    outer_r: f32,
    theta_cap: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    max_depth: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let origin_body = ray_origin - body_origin;
    let outer_hit = uv_sphere_intersection(origin_body, ray_dir, outer_r * body_size, body_size);
    if outer_hit.x >= outer_hit.y || outer_hit.y < 0.0 { return result; }
    var cursor_t = min(max(outer_hit.x, 0.0) + 1e-4 * body_size, outer_hit.y);

    for (var iter: u32 = 0u; iter < 512u; iter = iter + 1u) {
        if cursor_t > outer_hit.y { return result; }
        let point_body = origin_body + ray_dir * cursor_t;
        let uv = body_point_to_uv(point_body, inner_r, outer_r, theta_cap, body_size);
        let walk = walk_uv_subtree(body_root_idx, uv.x, uv.y, uv.z, max_depth);
        if walk.block != UV_EMPTY {
            result.hit = true;
            result.t = cursor_t;
            result.color = palette[walk.block].rgb;
            let center = vec3<f32>(body_size * 0.5);
            result.normal = normalize(point_body - center);
            let p0 = uv_to_body_point(walk.phi_lo, walk.theta_lo, walk.r_lo, inner_r, outer_r, theta_cap, body_size);
            let p1 = uv_to_body_point(walk.phi_lo + walk.size, walk.theta_lo + walk.size, walk.r_lo + walk.size, inner_r, outer_r, theta_cap, body_size);
            result.cell_min = body_origin + min(p0, p1);
            result.cell_size = max(max(abs(p1.x - p0.x), abs(p1.y - p0.y)), abs(p1.z - p0.z));
            return result;
        }

        let phi_lo = walk.phi_lo * 6.28318530718;
        let phi_hi = (walk.phi_lo + walk.size) * 6.28318530718;
        let theta_span = max(3.14159265359 - 2.0 * theta_cap, 1e-6);
        let theta_lo = theta_cap + walk.theta_lo * theta_span;
        let theta_hi = theta_cap + (walk.theta_lo + walk.size) * theta_span;
        let r_lo = (inner_r + walk.r_lo * (outer_r - inner_r)) * body_size;
        let r_hi = (inner_r + (walk.r_lo + walk.size) * (outer_r - inner_r)) * body_size;
        var next_t = 1e30;
        let eps = walk.size * 1e-3 + 1e-6;
        for (var candidate_idx: u32 = 0u; candidate_idx < 6u; candidate_idx = candidate_idx + 1u) {
            var candidate_t = 1e30;
            if candidate_idx == 0u {
                candidate_t = uv_plane_intersection(origin_body, ray_dir, phi_lo, body_size);
            } else if candidate_idx == 1u {
                candidate_t = uv_plane_intersection(origin_body, ray_dir, phi_hi, body_size);
            } else if candidate_idx == 2u {
                candidate_t = uv_theta_intersection(origin_body, ray_dir, theta_lo, body_size);
            } else if candidate_idx == 3u {
                candidate_t = uv_theta_intersection(origin_body, ray_dir, theta_hi, body_size);
            } else if candidate_idx == 4u {
                if r_lo > 1e-6 {
                    let inner_hit = uv_sphere_intersection(origin_body, ray_dir, r_lo, body_size);
                    candidate_t = select(inner_hit.x, inner_hit.y, inner_hit.x <= cursor_t + 1e-6);
                }
            } else {
                let outer_cell_hit = uv_sphere_intersection(origin_body, ray_dir, r_hi, body_size);
                candidate_t = select(outer_cell_hit.x, outer_cell_hit.y, outer_cell_hit.x <= cursor_t + 1e-6);
            }
            if !(candidate_t == candidate_t) || candidate_t <= cursor_t + 1e-6 || candidate_t >= 1e29 {
                continue;
            }
            let candidate_p = origin_body + ray_dir * candidate_t;
            let candidate_uv = body_point_to_uv(candidate_p, inner_r, outer_r, theta_cap, body_size);
            let phi_delta = min(abs(candidate_uv.x - walk.phi_lo), abs(candidate_uv.x + 1.0 - walk.phi_lo));
            var valid = false;
            if candidate_idx < 2u {
                valid = candidate_uv.y >= walk.theta_lo - eps
                    && candidate_uv.y <= walk.theta_lo + walk.size + eps
                    && candidate_uv.z >= walk.r_lo - eps
                    && candidate_uv.z <= walk.r_lo + walk.size + eps;
            } else if candidate_idx < 4u {
                valid = phi_delta <= walk.size + eps
                    && candidate_uv.z >= walk.r_lo - eps
                    && candidate_uv.z <= walk.r_lo + walk.size + eps;
            } else {
                valid = phi_delta <= walk.size + eps
                    && candidate_uv.y >= walk.theta_lo - eps
                    && candidate_uv.y <= walk.theta_lo + walk.size + eps;
            }
            if valid {
                next_t = min(next_t, candidate_t);
            }
        }
        if next_t >= 1e29 { return result; }
        cursor_t = next_t + max(1e-4 * body_size / pow(3.0, f32(walk.ratio_depth)), 1e-6);
    }

    return result;
}
