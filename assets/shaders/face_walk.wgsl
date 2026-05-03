#include "bindings.wgsl"
#include "tree.wgsl"
#include "face_math.wgsl"

// Result of walking a face subtree: which terminal (block_id),
// what depth it sits at, AND the cell's bounds in face-normalized
// `(un, vn, rn) ∈ [0, 1]³` coords. Bounds come from incremental
// Kahan-compensated accumulation during descent — they don't suffer
// the precision wall that `cells_d = pow(3, depth)` quantization
// hits past depth 14.
struct FaceWalkResult {
    block: u32,
    depth: u32,
    u_lo: f32,  // cell's lo bound in normalized face EA u
    v_lo: f32,
    r_lo: f32,
    size: f32,  // cell width = 3^-depth (same on all axes)
}

const FACE_ROOT_LOD_THRESHOLD_PIXELS: f32 = 4.0;

// Walk a face subtree from the body node's face-center child slot,
// returning the terminal AND the cell's normalized bounds. Bounds
// are accumulated via Kahan compensation so cumulative error stays
// at ~1 ULP regardless of depth (vs. ~depth ULPs naive).
fn walk_face_subtree(body_node_idx: u32, face: u32,
                     un_in: f32, vn_in: f32, rn_in: f32,
                     depth_limit: u32) -> FaceWalkResult {
    var result: FaceWalkResult;
    result.u_lo = 0.0;
    result.v_lo = 0.0;
    result.r_lo = 0.0;
    result.size = 1.0;
    result.depth = 1u;

    let fs = face_slot(face);
    // Scalar-cache the body node's header: one load pair instead of
    // three (child_empty + child_packed + child_node_index each
    // re-read tree[h], tree[h+1] independently).
    let body_header_off = node_offsets[body_node_idx];
    let body_occupancy = tree[body_header_off];
    let body_first_child = tree[body_header_off + 1u];
    let body_bit = 1u << fs;
    if (body_occupancy & body_bit) == 0u {
        result.block = 0u;
        return result;
    }
    let body_rank = countOneBits(body_occupancy & (body_bit - 1u));
    let body_child_base = body_first_child + body_rank * 2u;
    let face_packed = tree[body_child_base];
    let face_tag = face_packed & 0xFFu;
    if face_tag == 1u {
        result.block = (face_packed >> 8u) & 0xFFFFu;
        return result;
    }
    var node = tree[body_child_base + 1u];
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);

    // Kahan-compensated boundary accumulators per axis.
    var u_sum: f32 = 0.0; var u_comp: f32 = 0.0;
    var v_sum: f32 = 0.0; var v_comp: f32 = 0.0;
    var r_sum: f32 = 0.0; var r_comp: f32 = 0.0;
    var size: f32 = 1.0;

    let limit = min(depth_limit, MAX_FACE_DEPTH);
    if limit <= 1u {
        let bt = (face_packed >> 8u) & 0xFFFFu;
        result.block = select(0u, bt, bt != 0xFFFEu);
        return result;
    }
    for (var d: u32 = 2u; d <= limit; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        // Scalar-cache node header: one rank, one packed load, one
        // node_index load — all reusing the same (occupancy,
        // first_child) locals.
        let header_off = node_offsets[node];
        let occupancy = tree[header_off];
        let first_child = tree[header_off + 1u];
        let bit = 1u << slot;
        let occupied = (occupancy & bit) != 0u;
        let rank = countOneBits(occupancy & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = select(0u, tree[child_base], occupied);
        let tag = packed & 0xFFu;

        // Boundary update: this step's child within the parent
        // contributes (size/3) * slot to the lo-bound, and shrinks
        // size by 3. Done with Kahan compensation.
        let step_size = size * (1.0 / 3.0);
        let u_add = step_size * f32(us);
        let v_add = step_size * f32(vs);
        let r_add = step_size * f32(rs);

        let yu = u_add - u_comp;
        let tu = u_sum + yu;
        u_comp = (tu - u_sum) - yu;
        u_sum = tu;

        let yv = v_add - v_comp;
        let tv = v_sum + yv;
        v_comp = (tv - v_sum) - yv;
        v_sum = tv;

        let yr = r_add - r_comp;
        let tr = r_sum + yr;
        r_comp = (tr - r_sum) - yr;
        r_sum = tr;

        size = step_size;

        if tag == 0u || tag == 1u {
            result.block = select(0u, (packed >> 8u) & 0xFFFFu, tag == 1u);
            result.depth = d;
            result.u_lo = u_sum + u_comp;
            result.v_lo = v_sum + v_comp;
            result.r_lo = r_sum + r_comp;
            result.size = size;
            return result;
        }
        if d >= limit {
            let bt = (packed >> 8u) & 0xFFFFu;
            result.block = select(0u, bt, bt != 0xFFFEu);
            result.depth = d;
            result.u_lo = u_sum + u_comp;
            result.v_lo = v_sum + v_comp;
            result.r_lo = r_sum + r_comp;
            result.size = size;
            return result;
        }
        node = tree[child_base + 1u];
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }

    // Hit max depth without terminal: report deepest LOD bounds.
    result.block = 0u;
    result.depth = limit;
    result.u_lo = u_sum + u_comp;
    result.v_lo = v_sum + v_comp;
    result.r_lo = r_sum + r_comp;
    result.size = size;
    return result;
}

fn sample_face_node(node_idx: u32,
                    un_in: f32, vn_in: f32, rn_in: f32,
                    depth_limit: u32) -> vec2<u32> {
    var node = node_idx;
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);
    let limit = min(depth_limit, MAX_FACE_DEPTH);
    for (var d: u32 = 1u; d <= limit; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        // Scalar-cache header: single rank computation reused for
        // packed + node_index loads.
        let header_off = node_offsets[node];
        let occupancy = tree[header_off];
        let bit = 1u << slot;
        if (occupancy & bit) == 0u {
            return vec2<u32>(0u, d);
        }
        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occupancy & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        if tag == 1u {
            return vec2<u32>((packed >> 8u) & 0xFFFFu, d);
        }
        node = tree[child_base + 1u];
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }
    return vec2<u32>(0u, limit);
}

fn walk_face_node(node_idx: u32,
                  un_in: f32, vn_in: f32, rn_in: f32,
                  ray_t: f32,
                  lod_scale: f32) -> FaceWalkResult {
    var result: FaceWalkResult;
    result.block = 0u;
    result.depth = 0u;
    result.u_lo = 0.0;
    result.v_lo = 0.0;
    result.r_lo = 0.0;
    result.size = 1.0;

    var node = node_idx;
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);

    var u_lo: f32 = 0.0;
    var v_lo: f32 = 0.0;
    var r_lo: f32 = 0.0;
    var size: f32 = 1.0;

    for (var d: u32 = 1u; d <= MAX_FACE_DEPTH; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;
        // Scalar-cache header: single rank computation reused for
        // packed + node_index loads.
        let header_off = node_offsets[node];
        let occupancy = tree[header_off];
        let first_child = tree[header_off + 1u];
        let bit = 1u << slot;
        let occupied = (occupancy & bit) != 0u;
        let rank = countOneBits(occupancy & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = select(0u, tree[child_base], occupied);
        let tag = packed & 0xFFu;

        let step_size = size * (1.0 / 3.0);
        u_lo = u_lo + step_size * f32(us);
        v_lo = v_lo + step_size * f32(vs);
        r_lo = r_lo + step_size * f32(rs);
        size = step_size;

        if tag == 0u || tag == 1u {
            result.block = select(0u, (packed >> 8u) & 0xFFFFu, tag == 1u);
            result.depth = d;
            result.u_lo = u_lo;
            result.v_lo = v_lo;
            result.r_lo = r_lo;
            result.size = size;
            return result;
        }

        let cell_size_local = 3.0 * size;
        let ray_dist = max(ray_t, 0.001);
        let lod_pixels = cell_size_local / ray_dist * lod_scale;
        let at_lod = lod_pixels < FACE_ROOT_LOD_THRESHOLD_PIXELS;
        let at_max = d >= uniforms.max_depth;
        if at_lod || at_max {
            let bt = (packed >> 8u) & 0xFFFFu;
            result.block = select(0u, bt, bt != 0xFFFEu);
            result.depth = d;
            result.u_lo = u_lo;
            result.v_lo = v_lo;
            result.r_lo = r_lo;
            result.size = size;
            return result;
        }

        node = tree[child_base + 1u];
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }

    result.depth = MAX_FACE_DEPTH;
    result.u_lo = u_lo;
    result.v_lo = v_lo;
    result.r_lo = r_lo;
    result.size = size;
    return result;
}

fn face_point_to_body_with_bounds(point: vec3<f32>, bounds: vec4<f32>) -> vec3<f32> {
    let face = uniforms.root_face_meta.x;
    let un = bounds.x + (point.x / 3.0) * bounds.w;
    let vn = bounds.y + (point.y / 3.0) * bounds.w;
    let rn = bounds.z + (point.z / 3.0) * bounds.w;
    let dir = face_uv_to_dir(face, un * 2.0 - 1.0, vn * 2.0 - 1.0);
    let radius_local = uniforms.root_radii.x + rn * (uniforms.root_radii.y - uniforms.root_radii.x);
    let body_local = vec3<f32>(0.5) + dir * radius_local;
    return body_local * 3.0;
}

fn root_face_point_to_body(point: vec3<f32>) -> vec3<f32> {
    return face_point_to_body_with_bounds(point, uniforms.root_face_bounds);
}

fn face_root_point_to_body(point: vec3<f32>) -> vec3<f32> {
    return face_point_to_body_with_bounds(point, vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

fn face_dir_to_body(origin: vec3<f32>, dir: vec3<f32>, bounds: vec4<f32>) -> vec3<f32> {
    let eps = max(bounds.w * 1e-3, 1e-5);
    let p0 = face_point_to_body_with_bounds(origin, bounds);
    let p1 = face_point_to_body_with_bounds(origin + dir * eps, bounds);
    let d = p1 - p0;
    if dot(d, d) < 1e-12 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(d);
}

fn face_local_normal_to_body(point: vec3<f32>, normal: vec3<f32>, bounds: vec4<f32>) -> vec3<f32> {
    let p = face_point_to_body_with_bounds(point, bounds);
    let dir = normalize(p - vec3<f32>(1.5));
    let face = uniforms.root_face_meta.x;
    let n_axis = face_normal(face);
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);
    if abs(normal.z) > 0.5 {
        return normalize(dir * sign(normal.z));
    }
    if abs(normal.y) > 0.5 {
        return normalize(v_axis * sign(normal.y));
    }
    let axis_dot = max(dot(dir, n_axis), 1e-5);
    let u_world = normalize(u_axis - n_axis * (dot(dir, u_axis) / axis_dot));
    return normalize(u_world * sign(normal.x));
}

fn face_box_to_body_bounds(hmin: vec3<f32>, hmax: vec3<f32>, bounds: vec4<f32>) -> mat2x3<f32> {
    var mn = vec3<f32>(1e20);
    var mx = vec3<f32>(-1e20);
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let corner = vec3<f32>(
            select(hmin.x, hmax.x, (i & 1u) != 0u),
            select(hmin.y, hmax.y, (i & 2u) != 0u),
            select(hmin.z, hmax.z, (i & 4u) != 0u),
        );
        let p = face_point_to_body_with_bounds(corner, bounds);
        mn = min(mn, p);
        mx = max(mx, p);
    }
    return mat2x3<f32>(mn, mx);
}
