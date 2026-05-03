#include "bindings.wgsl"
#include "tree.wgsl"
#include "face_math.wgsl"
#include "ray_prim.wgsl"
#include "face_walk.wgsl"

// Sphere-shell DDAs running inside a CubedSphereBody cell, plus the
// small cell-shading helpers they share with the fragment shader.

fn face_uv_for_normal(local: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let an = abs(normal);
    if an.x >= an.y && an.x >= an.z {
        return local.yz;
    }
    if an.y >= an.z {
        return local.xz;
    }
    return local.xy;
}

fn cube_face_bevel(local: vec3<f32>, normal: vec3<f32>) -> f32 {
    let uv = face_uv_for_normal(local, normal);
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.02, 0.14, edge);
}

fn sphere_cell_shape(cell_u: f32, cell_v: f32, cell_r: f32) -> f32 {
    let face_edge = min(
        min(cell_u, 1.0 - cell_u),
        min(cell_v, 1.0 - cell_v),
    );
    let bevel = smoothstep(0.02, 0.14, face_edge);
    _ = cell_r;
    return 0.78 + 0.22 * bevel;
}

fn march_face_root(
    root_node_idx: u32,
    ray_origin_body: vec3<f32>,
    ray_dir: vec3<f32>,
    bounds: vec4<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let face = uniforms.root_face_meta.x;
    let cs_center = vec3<f32>(1.5);
    let cs_outer = uniforms.root_radii.y * 3.0;
    let cs_inner = uniforms.root_radii.x * 3.0;
    let shell = cs_outer - cs_inner;
    let oc = ray_origin_body - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_face_id: u32 = 6u;
    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }
        let n = local / r;
        let hit_face = pick_face(n);
        if hit_face != face { break; }
        let n_axis = face_normal(face);
        let u_axis = face_u_axis(face);
        let v_axis = face_v_axis(face);
        let axis_dot = dot(n, n_axis);
        let cube_u = dot(n, u_axis) / axis_dot;
        let cube_v = dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);
        let un_abs = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let vn_abs = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let rn_abs = clamp((r - cs_inner) / shell, 0.0, 0.9999999);
        if un_abs < bounds.x || un_abs >= bounds.x + bounds.w ||
           vn_abs < bounds.y || vn_abs >= bounds.y + bounds.w ||
           rn_abs < bounds.z || rn_abs >= bounds.z + bounds.w {
            break;
        }

        let un_local = (un_abs - bounds.x) / bounds.w;
        let vn_local = (vn_abs - bounds.y) / bounds.w;
        let rn_local = (rn_abs - bounds.z) / bounds.w;
        let walk = sample_face_node(
            root_node_idx,
            un_local,
            vn_local,
            rn_local,
            uniforms.max_depth,
        );
        let block_id = walk.x;
        let term_depth = walk.y;
        let cells_d = pow3_u(term_depth);
        let iu = floor(un_local * cells_d);
        let iv = floor(vn_local * cells_d);
        let ir = floor(rn_local * cells_d);

        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_id {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            result.hit = true;
            result.t = t;
            let cell_u = un_local * cells_d - iu;
            let cell_v = vn_local * cells_d - iv;
            let cell_r = rn_local * cells_d - ir;
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let block_shape = sphere_cell_shape(cell_u, cell_v, cell_r);
            result.color = palette[block_id].rgb * (ambient + diffuse * 0.78) * axis_tint * block_shape;
            result.normal = hit_normal;
            return result;
        }

        let cell_lo = 1.0 / cells_d;
        let u_lo = iu / cells_d;
        let v_lo = iv / cells_d;
        let r_lo_local = ir / cells_d;
        let u_lo_ea = (bounds.x + u_lo * bounds.w) * 2.0 - 1.0;
        let u_hi_ea = (bounds.x + (u_lo + cell_lo) * bounds.w) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = (bounds.y + v_lo * bounds.w) * 2.0 - 1.0;
        let v_hi_ea = (bounds.y + (v_lo + cell_lo) * bounds.w) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + (bounds.z + r_lo_local * bounds.w) * shell;
        let r_hi = cs_inner + (bounds.z + (r_lo_local + cell_lo) * bounds.w) * shell;

        var t_next = t_exit + 1.0;
        var winning_face: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let cand_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if cand_u_lo > t && cand_u_lo < t_next { t_next = cand_u_lo; winning_face = 0u; }
        let cand_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if cand_u_hi > t && cand_u_hi < t_next { t_next = cand_u_hi; winning_face = 1u; }
        let cand_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if cand_v_lo > t && cand_v_lo < t_next { t_next = cand_v_lo; winning_face = 2u; }
        let cand_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if cand_v_hi > t && cand_v_hi < t_next { t_next = cand_v_hi; winning_face = 3u; }
        let cand_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if cand_r_lo > t && cand_r_lo < t_next { t_next = cand_r_lo; winning_face = 4u; }
        let cand_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if cand_r_hi > t && cand_r_hi < t_next { t_next = cand_r_hi; winning_face = 5u; }

        if t_next >= t_exit { break; }
        last_face_id = winning_face;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * cell_lo * bounds.w * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}

// Sphere DDA running inside one CubedSphereBody cell. The body cell
// is given in the render-frame's coords (origin + size); radii are
// scaled into the same frame. Returns hit/miss; on miss the caller
// continues the Cartesian DDA past the body cell.
fn sphere_in_cell(
    body_node_idx: u32,
    body_cell_origin: vec3<f32>,
    body_cell_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let cs_center = body_cell_origin + vec3<f32>(body_cell_size * 0.5);
    let cs_outer = outer_r_local * body_cell_size;
    let cs_inner = inner_r_local * body_cell_size;
    let shell = cs_outer - cs_inner;

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_face_id: u32 = 6u;
    var deepest_term_depth: u32 = 1u;
    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }

        let n = local / r;
        let face = pick_face(n);
        let n_axis = face_normal(face);
        let u_axis = face_u_axis(face);
        let v_axis = face_v_axis(face);
        let axis_dot = dot(n, n_axis);
        let cube_u = dot(n, u_axis) / axis_dot;
        let cube_v = dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);

        let un = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let vn = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let rn = clamp((r - cs_inner) / shell, 0.0, 0.9999999);

        let walk = walk_face_subtree(body_node_idx, face, un, vn, rn, uniforms.max_depth);
        let block_id = walk.block;
        let term_depth = walk.depth;

        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_id {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            let cell_color = palette[block_id].rgb;
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let cell_u = clamp((un - walk.u_lo) / walk.size, 0.0, 1.0);
            let cell_v = clamp((vn - walk.v_lo) / walk.size, 0.0, 1.0);
            let cell_r = clamp((rn - walk.r_lo) / walk.size, 0.0, 1.0);
            let block_shape = sphere_cell_shape(cell_u, cell_v, cell_r);
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            result.color = cell_color * (ambient + diffuse * 0.78) * axis_tint * block_shape;
            return result;
        }

        deepest_term_depth = max(deepest_term_depth, term_depth);
        // Cell bounds come from the walker's Kahan-compensated
        // accumulation. No more `floor(un * 3^depth)` quantization,
        // so depths past 14 stay precision-correct.
        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        var t_next = t_exit + 1.0;
        var winning_face: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let cand_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if cand_u_lo > t && cand_u_lo < t_next { t_next = cand_u_lo; winning_face = 0u; }
        let cand_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if cand_u_hi > t && cand_u_hi < t_next { t_next = cand_u_hi; winning_face = 1u; }
        let cand_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if cand_v_lo > t && cand_v_lo < t_next { t_next = cand_v_lo; winning_face = 2u; }
        let cand_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if cand_v_hi > t && cand_v_hi < t_next { t_next = cand_v_hi; winning_face = 3u; }
        let cand_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if cand_r_lo > t && cand_r_lo < t_next { t_next = cand_r_lo; winning_face = 4u; }
        let cand_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if cand_r_hi > t && cand_r_hi < t_next { t_next = cand_r_hi; winning_face = 5u; }

        if t_next >= t_exit { break; }
        last_face_id = winning_face;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * walk.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}

fn sphere_in_face_window(
    body_node_idx: u32,
    face: u32,
    face_u_min: f32,
    face_v_min: f32,
    face_r_min: f32,
    face_size: f32,
    face_depth: u32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin_body: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let body_origin = vec3<f32>(0.0);
    let body_size = 3.0;
    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;

    let oc = ray_origin_body - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_face_id: u32 = 6u;
    let depth_limit = min(MAX_FACE_DEPTH, face_depth + uniforms.max_depth);
    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }

        let n = local / r;
        let hit_face = pick_face(n);
        if hit_face != face { break; }
        let n_axis = face_normal(face);
        let u_axis = face_u_axis(face);
        let v_axis = face_v_axis(face);
        let axis_dot = dot(n, n_axis);
        let cube_u = dot(n, u_axis) / axis_dot;
        let cube_v = dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);

        let un_abs = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let vn_abs = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let rn_abs = clamp((r - cs_inner) / shell, 0.0, 0.9999999);
        if un_abs < face_u_min || un_abs >= face_u_min + face_size ||
           vn_abs < face_v_min || vn_abs >= face_v_min + face_size ||
           rn_abs < face_r_min || rn_abs >= face_r_min + face_size {
            break;
        }

        let walk = walk_face_subtree(body_node_idx, face, un_abs, vn_abs, rn_abs, depth_limit);
        let block_id = walk.block;
        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_id {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            let cell_u = clamp((un_abs - walk.u_lo) / walk.size, 0.0, 1.0);
            let cell_v = clamp((vn_abs - walk.v_lo) / walk.size, 0.0, 1.0);
            let cell_r = clamp((rn_abs - walk.r_lo) / walk.size, 0.0, 1.0);
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let block_shape = sphere_cell_shape(cell_u, cell_v, cell_r);
            result.color = palette[block_id].rgb * (ambient + diffuse * 0.78) * axis_tint * block_shape;
            return result;
        }

        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        var t_next = t_exit + 1.0;
        var winning_face: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let cand_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if cand_u_lo > t && cand_u_lo < t_next { t_next = cand_u_lo; winning_face = 0u; }
        let cand_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if cand_u_hi > t && cand_u_hi < t_next { t_next = cand_u_hi; winning_face = 1u; }
        let cand_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if cand_v_lo > t && cand_v_lo < t_next { t_next = cand_v_lo; winning_face = 2u; }
        let cand_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if cand_v_hi > t && cand_v_hi < t_next { t_next = cand_v_hi; winning_face = 3u; }
        let cand_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if cand_r_lo > t && cand_r_lo < t_next { t_next = cand_r_lo; winning_face = 4u; }
        let cand_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if cand_r_hi > t && cand_r_hi < t_next { t_next = cand_r_hi; winning_face = 5u; }

        if t_next >= t_exit { break; }
        last_face_id = winning_face;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * walk.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}
