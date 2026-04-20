#include "bindings.wgsl"
#include "tree.wgsl"
#include "face_math.wgsl"
#include "ray_prim.wgsl"
#include "face_walk.wgsl"

// Cubed-sphere DDA for rendering a `NodeKind::CubedSphereBody` cell
// as a curved sphere with `u/v/r`-indexed voxel content.
//
// The critical precision rule: `cs_center` is computed in the
// **caller's render frame** coords (not body-frame absolutes). The
// old code's `cs_center = vec3<f32>(1.5)` hardcode had a ULP wall at
// deep anchor because the camera's body-frame coordinates lost
// precision during path accumulation; with render-frame-local
// coords, `cs_center` tracks the body's position inside whatever
// frame the walker is currently rooted in, and `oc = ray_origin -
// cs_center` stays precision-bounded by the body's render-frame
// radius.

// Shading: simple per-cell lit color, shared by all the sphere
// leaf hits below.
fn sphere_shade(
    block_id: u32,
    hit_normal: vec3<f32>,
) -> vec3<f32> {
    let cell_color = palette.colors[block_id].rgb;
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
    let diffuse = max(dot(hit_normal, sun_dir), 0.0);
    let axis_tint = abs(hit_normal.y) * 1.0
                  + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
    let ambient = 0.22;
    return cell_color * (ambient + diffuse * 0.78) * axis_tint;
}

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

// Sphere DDA running inside one CubedSphereBody cell, expressed in
// the caller's render frame. `body_cell_origin` and `body_cell_size`
// give the body's bounding box in the render frame. All math
// operates on `oc = ray_origin - cs_center` where `cs_center` is
// also in the render frame — precision-safe at any anchor depth.
fn sphere_in_cell(
    body_node_idx: u32,
    body_cell_origin: vec3<f32>,
    body_cell_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    walker_max_depth: u32,
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

        let walk_depth = min(walker_max_depth, MAX_FACE_DEPTH);
        let walk = walk_face_subtree(body_node_idx, face, un, vn, rn, walk_depth);
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
            let hit_pos = ray_origin + ray_dir * t;
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            result.color = sphere_shade(block_id, hit_normal);
            result.cell_min = hit_pos - vec3<f32>(shell * walk.size * 0.5);
            result.cell_size = shell * walk.size;
            // Populate hit_path: first slot is the body's face-slot
            // (so the body→face transition is captured), followed
            // by the walker's internal (us, vs, rs) descent from
            // the face subtree root. The full body-rooted path is
            // [face_slot, us/vs/rs slots...]. Callers append this
            // to their own render/ribbon prefix for the path-prefix
            // match in `main.wgsl`.
            pack_slot_into_path(&result.hit_path, 0u, face_slot(face));
            var wu = un;
            var wv = vn;
            var wr = rn;
            for (var d: u32 = 0u; d < walk.depth; d = d + 1u) {
                let us = min(u32(wu * 3.0), 2u);
                let vs = min(u32(wv * 3.0), 2u);
                let rs = min(u32(wr * 3.0), 2u);
                let slot = rs * 9u + vs * 3u + us;
                pack_slot_into_path(&result.hit_path, d + 1u, slot);
                wu = wu * 3.0 - f32(us);
                wv = wv * 3.0 - f32(vs);
                wr = wr * 3.0 - f32(rs);
            }
            result.hit_path_depth = walk.depth + 1u;
            return result;
        }

        // Cell bounds in face-normalized space from the walker.
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
