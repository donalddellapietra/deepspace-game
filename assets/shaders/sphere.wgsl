#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"

// Cubed-sphere geometry + DDA. One WGSL file with the face-math
// helpers, the face-subtree walker, and the unified sphere march.
// The CPU mirror lives in `src/world/cubesphere.rs` +
// `src/world/raycast/sphere.rs`.

// ─────────────────────────────────────────────── face constants

// Face enum ↔ integer. 0..=5 in the same order as `Face::ALL`
// (PosX, NegX, PosY, NegY, PosZ, NegZ).

fn face_normal(f: u32) -> vec3<f32> {
    switch f {
        case 0u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 1u: { return vec3<f32>(-1.0,  0.0,  0.0); }
        case 2u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 3u: { return vec3<f32>( 0.0, -1.0,  0.0); }
        case 4u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        default: { return vec3<f32>( 0.0,  0.0, -1.0); }
    }
}

fn face_u_axis(f: u32) -> vec3<f32> {
    switch f {
        case 0u: { return vec3<f32>( 0.0,  0.0, -1.0); }
        case 1u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        case 2u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 3u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        case 4u: { return vec3<f32>( 1.0,  0.0,  0.0); }
        default: { return vec3<f32>(-1.0,  0.0,  0.0); }
    }
}

fn face_v_axis(f: u32) -> vec3<f32> {
    switch f {
        case 0u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 1u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        case 2u: { return vec3<f32>( 0.0,  0.0, -1.0); }
        case 3u: { return vec3<f32>( 0.0,  0.0,  1.0); }
        case 4u: { return vec3<f32>( 0.0,  1.0,  0.0); }
        default: { return vec3<f32>( 0.0,  1.0,  0.0); }
    }
}

/// Slot index inside a body cell for a given face's subtree.
/// Must match Rust's `cubesphere::FACE_SLOTS`.
fn face_slot(f: u32) -> u32 {
    switch f {
        case 0u: { return 14u; } // PosX: (2,1,1)
        case 1u: { return 12u; } // NegX: (0,1,1)
        case 2u: { return 16u; } // PosY: (1,2,1)
        case 3u: { return 10u; } // NegY: (1,0,1)
        case 4u: { return 22u; } // PosZ: (1,1,2)
        default: { return  4u; } // NegZ: (1,1,0)
    }
}

// ──────────────────────────────────────────── coord conversions

fn cube_to_ea(c: f32) -> f32 { return atan(c) * (4.0 / 3.14159265); }
fn ea_to_cube(c: f32) -> f32 { return tan(c * (3.14159265 / 4.0)); }

fn pick_face(n: vec3<f32>) -> u32 {
    let ax = abs(n.x);
    let ay = abs(n.y);
    let az = abs(n.z);
    if ax >= ay && ax >= az {
        return select(1u, 0u, n.x >= 0.0);
    } else if ay >= az {
        return select(3u, 2u, n.y >= 0.0);
    } else {
        return select(5u, 4u, n.z >= 0.0);
    }
}

// ──────────────────────────────────────── face-subtree walker

struct FaceWalkResult {
    block: u32,
    depth: u32,
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    size: f32,
}

/// Descend a face subtree from its root along `(un, vn, rn)` to the
/// terminal cell. Mirrors the CPU `walk_face_subtree` but without
/// `EMPTY_NODE` padding (the GPU doesn't need placement paths).
/// Empty-cell sentinel in `FaceWalkResult.block`. Palette index 0
/// is real (STONE), so we can't use 0 for "no hit". Matches
/// Rust's `REPRESENTATIVE_EMPTY`.
const FACE_WALK_EMPTY: u32 = 0xFFFEu;

fn walk_face_subtree(
    face_root_idx: u32,
    un_in: f32, vn_in: f32, rn_in: f32,
    max_depth: u32,
) -> FaceWalkResult {
    var res: FaceWalkResult;
    res.block = FACE_WALK_EMPTY;
    res.depth = 0u;
    res.u_lo = 0.0;
    res.v_lo = 0.0;
    res.r_lo = 0.0;
    res.size = 1.0;

    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);
    var node_idx = face_root_idx;
    var u_lo: f32 = 0.0;
    var v_lo: f32 = 0.0;
    var r_lo: f32 = 0.0;
    var size: f32 = 1.0;

    for (var d: u32 = 1u; d <= max_depth; d = d + 1u) {
        let base = node_offsets[node_idx];
        if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
        let occupancy = tree[base];
        let first_child = tree[base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }

        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;

        let child_size = size / 3.0;
        let child_u_lo = u_lo + f32(us) * child_size;
        let child_v_lo = v_lo + f32(vs) * child_size;
        let child_r_lo = r_lo + f32(rs) * child_size;

        // Is this slot populated?
        let mask = (occupancy >> slot) & 1u;
        if mask == 0u {
            // Empty cell — terminate.
            res.depth = d;
            res.u_lo = child_u_lo;
            res.v_lo = child_v_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            return res;
        }
        // Count 1-bits below `slot` to find child rank.
        let rank = countOneBits(occupancy & ((1u << slot) - 1u));
        let packed = tree[first_child + rank * 2u];
        let node_index = tree[first_child + rank * 2u + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }

        let tag = packed & 0xFFu;
        if tag == 1u {
            // Leaf block.
            res.block = child_block_type(packed);
            res.depth = d;
            res.u_lo = child_u_lo;
            res.v_lo = child_v_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            return res;
        }
        // Tag 2 → descend into node.
        if d == max_depth {
            // LOD-terminal. Use representative block; preserve the
            // empty sentinel when the subtree is all-empty.
            res.block = child_block_type(packed);
            res.depth = d;
            res.u_lo = child_u_lo;
            res.v_lo = child_v_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            return res;
        }
        node_idx = node_index;
        u_lo = child_u_lo;
        v_lo = child_v_lo;
        r_lo = child_r_lo;
        size = child_size;
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }
    res.u_lo = u_lo;
    res.v_lo = v_lo;
    res.r_lo = r_lo;
    res.size = size;
    res.depth = max_depth;
    return res;
}

// ───────────────────────────────────────────── cell-shape bevel

// Edge-dark band at normalized face edges for a single tree level.
fn bevel_level(un: f32, vn: f32, u_lo: f32, v_lo: f32, size: f32, cell_px: f32) -> f32 {
    if cell_px < 2.0 { return 1.0; }
    let cu = clamp((un - u_lo) / size, 0.0, 1.0);
    let cv = clamp((vn - v_lo) / size, 0.0, 1.0);
    let face_edge = min(min(cu, 1.0 - cu), min(cv, 1.0 - cv));
    let band_end = clamp(1.0 / cell_px, 0.0, 0.25);
    let b = smoothstep(0.0, band_end, face_edge);
    return 0.78 + 0.22 * b;
}

// Multi-level bevel overlay. Walks a few ancestors + descendants of
// the walker's cell so all voxel-grid levels visible to the pixel
// contribute a grid line.
fn bevel_layered(
    un: f32, vn: f32,
    u_lo: f32, v_lo: f32, size: f32,
    reference_scale: f32, ray_dist: f32, pixel_density: f32,
) -> f32 {
    let safe_dist = max(ray_dist, 1e-6);
    let base_px = size * reference_scale / safe_dist * pixel_density;
    var b: f32 = bevel_level(un, vn, u_lo, v_lo, size, base_px);

    var up_u = u_lo; var up_v = v_lo; var up_s = size; var up_px = base_px;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        up_s = up_s * 3.0;
        up_u = floor(up_u / up_s) * up_s;
        up_v = floor(up_v / up_s) * up_s;
        up_px = up_px * 3.0;
        b = b * bevel_level(un, vn, up_u, up_v, up_s, up_px);
    }

    var dn_u = u_lo; var dn_v = v_lo; var dn_s = size; var dn_px = base_px;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let cs = dn_s * (1.0 / 3.0);
        let cpx = dn_px * (1.0 / 3.0);
        if cpx < 2.0 { break; }
        let uf = clamp((un - dn_u) / dn_s, 0.0, 0.9999999);
        let vf = clamp((vn - dn_v) / dn_s, 0.0, 0.9999999);
        dn_u = dn_u + floor(uf * 3.0) * cs;
        dn_v = dn_v + floor(vf * 3.0) * cs;
        dn_s = cs;
        dn_px = cpx;
        b = b * bevel_level(un, vn, dn_u, dn_v, dn_s, dn_px);
    }
    return b;
}

fn depth_tint(rn: f32) -> f32 { return 0.55 + 0.45 * clamp(rn, 0.0, 1.0); }

// Per-ray LOD for the face walker. Matches the Cartesian
// `LOD_PIXEL_THRESHOLD` Nyquist gate.
fn face_lod_depth(ray_dist: f32, shell: f32) -> u32 {
    let pixel_density = uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
    let safe_dist = max(ray_dist, 1e-6);
    let ratio = shell * pixel_density / (safe_dist * max(LOD_PIXEL_THRESHOLD, 1e-6));
    if ratio <= 1.0 { return 1u; }
    let log3r = log2(ratio) * (1.0 / 1.5849625);
    return u32(clamp(1.0 + log3r, 1.0, f32(MAX_FACE_DEPTH)));
}

// ─────────────────────────────────────────── unified sphere DDA

/// Sphere shell DDA in one body cell. The body's local `[0, 1)³`
/// frame is mapped to `(body_origin, body_origin + body_size)³` in
/// the caller's coords. `inner_r`/`outer_r` are body-local radii.
///
/// When `window_active != 0`, hits are restricted to the face given
/// by `window_bounds.xyz + window_bounds.w` (u_min, v_min, r_min,
/// size) on `window_face`.
fn sphere_in_cell(
    body_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: vec3<f32>,
    ray_dir_in: vec3<f32>,
    window_active: u32,
    window_face: u32,
    window_bounds: vec4<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // `ray_sphere_after` + the cubemap-plane intersections assume
    // unit-length direction. The caller passes a non-unit vector
    // (camera.forward + right·ndc + up·ndc), so the quadratic
    // disc = b² − c would be scaled wrong for off-center pixels.
    // Renormalize up front; the returned `t` is in world units either
    // way because both walker and caller measure ray distance against
    // unit direction.
    let ray_dir = normalize(ray_dir_in);

    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;
    if shell <= 0.0 { return result; }

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    let pixel_density = uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
    var t = t_enter + eps_init;
    var steps = 0u;
    var last_side: u32 = 6u;
    let reference_scale = select(shell, shell * window_bounds.w, window_active != 0u);

    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }

        let n = local / r;
        let f = pick_face(n);
        if window_active != 0u && f != window_face { break; }

        let n_axis = face_normal(f);
        let u_axis = face_u_axis(f);
        let v_axis = face_v_axis(f);
        let axis_dot = dot(n, n_axis);
        if abs(axis_dot) < 1e-6 { break; }
        let cu = dot(n, u_axis) / axis_dot;
        let cv = dot(n, v_axis) / axis_dot;
        let un_abs = clamp((cube_to_ea(cu) + 1.0) * 0.5, 0.0, 0.9999999);
        let vn_abs = clamp((cube_to_ea(cv) + 1.0) * 0.5, 0.0, 0.9999999);
        let rn_abs = clamp((r - cs_inner) / shell, 0.0, 0.9999999);

        // Window clip.
        if window_active != 0u {
            if un_abs < window_bounds.x || un_abs >= window_bounds.x + window_bounds.w ||
               vn_abs < window_bounds.y || vn_abs >= window_bounds.y + window_bounds.w ||
               rn_abs < window_bounds.z || rn_abs >= window_bounds.z + window_bounds.w {
                break;
            }
        }

        // Locate the face root via body → face_slot child.
        let body_base = node_offsets[body_idx];
        let body_occ = tree[body_base];
        let body_first = tree[body_base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let fslot = face_slot(f);
        let fmask = (body_occ >> fslot) & 1u;
        if fmask == 0u { break; }
        let frank = countOneBits(body_occ & ((1u << fslot) - 1u));
        let face_node_idx = tree[body_first + frank * 2u + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 1u; }

        // Walker's UV/R in face-window-local frame when windowed, or
        // full-face frame otherwise.
        var walk_un = un_abs;
        var walk_vn = vn_abs;
        var walk_rn = rn_abs;
        if window_active != 0u {
            walk_un = (un_abs - window_bounds.x) / window_bounds.w;
            walk_vn = (vn_abs - window_bounds.y) / window_bounds.w;
            walk_rn = (rn_abs - window_bounds.z) / window_bounds.w;
        }

        let walk_depth = face_lod_depth(t, reference_scale);
        let w = walk_face_subtree(face_node_idx, walk_un, walk_vn, walk_rn, walk_depth);

        // 0xFFFEu is REPRESENTATIVE_EMPTY — an empty terminal.
        // Palette index 0 is a real block (STONE), so we can't use
        // zero as the empty sentinel.
        if w.block != FACE_WALK_EMPTY {
            // Hit. The previous step's `last_side` is the face we
            // crossed to exit the PREVIOUS cell; we entered THIS
            // cell through the geometrically-same boundary, but
            // that face's outward normal (from the hit cell's POV)
            // points back toward where the ray came from — the
            // opposite direction. So winning face 4 (crossed the
            // previous cell's r_lo going inward) lands on THIS
            // cell's r_hi face, outward normal +n; winning 0
            // (crossed prev u_lo going -u) lands on THIS cell's
            // u_hi, outward normal +u_axis; etc.
            var hit_normal: vec3<f32>;
            switch last_side {
                case 0u: { hit_normal =  u_axis; }
                case 1u: { hit_normal = -u_axis; }
                case 2u: { hit_normal =  v_axis; }
                case 3u: { hit_normal = -v_axis; }
                case 4u: { hit_normal =  n; }
                case 5u: { hit_normal = -n; }
                default: { hit_normal =  n; }
            }
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            let sun = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun), 0.0);
            let axis_tint = abs(hit_normal.y) + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let shape = bevel_layered(
                walk_un, walk_vn, w.u_lo, w.v_lo, w.size,
                reference_scale, t, pixel_density,
            );
            let tint = depth_tint(rn_abs);
            result.color = palette[w.block].rgb * (ambient + diffuse * 0.78) * axis_tint * shape * tint;
            return result;
        }

        // Empty cell — advance to next cell boundary via ray-plane /
        // ray-sphere intersections on the walker's 6 cell faces.
        let cell_u_lo_ea = w.u_lo * 2.0 - 1.0;
        let cell_u_hi_ea = (w.u_lo + w.size) * 2.0 - 1.0;
        let cell_v_lo_ea = w.v_lo * 2.0 - 1.0;
        let cell_v_hi_ea = (w.v_lo + w.size) * 2.0 - 1.0;
        // Window-local → absolute-face conversion for the radial
        // boundaries.
        let cell_r_lo_abs = select(w.r_lo, window_bounds.z + w.r_lo * window_bounds.w, window_active != 0u);
        let cell_r_hi_abs = select(w.r_lo + w.size, window_bounds.z + (w.r_lo + w.size) * window_bounds.w, window_active != 0u);
        let r_lo_world = cs_inner + cell_r_lo_abs * shell;
        let r_hi_world = cs_inner + cell_r_hi_abs * shell;

        let n_u_lo = u_axis - ea_to_cube(cell_u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(cell_u_hi_ea) * n_axis;
        let n_v_lo = v_axis - ea_to_cube(cell_v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(cell_v_hi_ea) * n_axis;

        var t_next = t_exit + 1.0;
        var winning: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let c_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if c_u_lo > t && c_u_lo < t_next { t_next = c_u_lo; winning = 0u; }
        let c_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if c_u_hi > t && c_u_hi < t_next { t_next = c_u_hi; winning = 1u; }
        let c_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if c_v_lo > t && c_v_lo < t_next { t_next = c_v_lo; winning = 2u; }
        let c_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if c_v_hi > t && c_v_hi < t_next { t_next = c_v_hi; winning = 3u; }
        let c_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo_world, t);
        if c_r_lo > t && c_r_lo < t_next { t_next = c_r_lo; winning = 4u; }
        let c_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi_world, t);
        if c_r_hi > t && c_r_hi < t_next { t_next = c_r_hi; winning = 5u; }

        if t_next >= t_exit { break; }
        last_side = winning;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * w.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}

// ─────────────────────── local-frame sphere sub-frame DDA

/// Terminal cell in the sub-frame's local `[0, 3)³` coords.
struct SubWalkResult {
    block: u32,
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    size: f32,
}

/// Walk a sub-frame subtree along local point `(u_l, v_l, r_l) ∈ [0, 3)³`
/// to `max_depth` levels. Mirrors CPU `walk_sub_frame` in
/// `src/world/raycast/sphere_sub.rs`.
fn walk_sub_frame(
    sub_frame_idx: u32,
    u_l_in: f32, v_l_in: f32, r_l_in: f32,
    max_depth: u32,
) -> SubWalkResult {
    var res: SubWalkResult;
    res.block = FACE_WALK_EMPTY;
    res.u_lo = 0.0;
    res.v_lo = 0.0;
    res.r_lo = 0.0;
    res.size = 3.0;

    let clamp_max = 0.9999999 * 3.0;
    var u_pt = clamp(u_l_in, 0.0, clamp_max);
    var v_pt = clamp(v_l_in, 0.0, clamp_max);
    var r_pt = clamp(r_l_in, 0.0, clamp_max);

    var node_idx = sub_frame_idx;
    var u_lo: f32 = 0.0;
    var v_lo: f32 = 0.0;
    var r_lo: f32 = 0.0;
    var size: f32 = 3.0;

    for (var d: u32 = 1u; d <= max_depth; d = d + 1u) {
        let base = node_offsets[node_idx];
        if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
        let occupancy = tree[base];
        let first_child = tree[base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }

        let child_size = size / 3.0;
        let us = min(u32((u_pt - u_lo) / child_size), 2u);
        let vs = min(u32((v_pt - v_lo) / child_size), 2u);
        let rs = min(u32((r_pt - r_lo) / child_size), 2u);
        let slot = rs * 9u + vs * 3u + us;
        let cu_lo = u_lo + f32(us) * child_size;
        let cv_lo = v_lo + f32(vs) * child_size;
        let cr_lo = r_lo + f32(rs) * child_size;

        let mask = (occupancy >> slot) & 1u;
        if mask == 0u {
            res.u_lo = cu_lo;
            res.v_lo = cv_lo;
            res.r_lo = cr_lo;
            res.size = child_size;
            return res;
        }
        let rank = countOneBits(occupancy & ((1u << slot) - 1u));
        let packed = tree[first_child + rank * 2u];
        let node_index = tree[first_child + rank * 2u + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let tag = packed & 0xFFu;
        if tag == 1u {
            res.block = child_block_type(packed);
            res.u_lo = cu_lo;
            res.v_lo = cv_lo;
            res.r_lo = cr_lo;
            res.size = child_size;
            return res;
        }
        if d == max_depth {
            res.block = child_block_type(packed);
            res.u_lo = cu_lo;
            res.v_lo = cv_lo;
            res.r_lo = cr_lo;
            res.size = child_size;
            return res;
        }
        node_idx = node_index;
        u_lo = cu_lo;
        v_lo = cv_lo;
        r_lo = cr_lo;
        size = child_size;
    }
    res.u_lo = u_lo;
    res.v_lo = v_lo;
    res.r_lo = r_lo;
    res.size = size;
    return res;
}

/// Fetch the i-th UVR pre-descent slot from the uniform array. One
/// u32 per vec4 slot — slot index `i` lives in element `(i/4, i%4)`.
fn sub_uvr_slot_at(i: u32) -> u32 {
    let row = uniforms.sub_uvr_slots[i / 4u];
    switch (i % 4u) {
        case 0u: { return row.x; }
        case 1u: { return row.y; }
        case 2u: { return row.z; }
        default: { return row.w; }
    }
}

/// Pre-descend from the face-subtree root (`face_root_idx`) along
/// `sub_meta.y` UVR slots, then dispatch `walk_sub_frame` at the
/// terminal Node. Mirrors CPU `walk_from_deep_sub_frame` in
/// `src/world/raycast/sphere_sub.rs`.
///
/// On `Child::Empty` / `Child::Block` / `Child::EntityRef` mid-prefix
/// the walker returns a uniform `SubWalkResult` covering the full
/// local `[0, 3)³` box (empty sentinel or the block type). The DDA
/// caller treats that as the whole sub-frame being one uniform cell.
fn walk_from_deep_sub_frame(
    face_root_idx: u32,
    u_l: f32, v_l: f32, r_l: f32,
    walker_limit: u32,
) -> SubWalkResult {
    var res: SubWalkResult;
    res.block = FACE_WALK_EMPTY;
    res.u_lo = 0.0;
    res.v_lo = 0.0;
    res.r_lo = 0.0;
    res.size = 3.0;

    let prefix_len = uniforms.sub_meta.y;
    var node_idx = face_root_idx;
    for (var i: u32 = 0u; i < prefix_len; i = i + 1u) {
        let slot = sub_uvr_slot_at(i);
        let base = node_offsets[node_idx];
        if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
        let occupancy = tree[base];
        let first_child = tree[base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let mask = (occupancy >> slot) & 1u;
        if mask == 0u {
            // Empty sub-cell mid-prefix → full sub-frame is empty.
            return res;
        }
        let rank = countOneBits(occupancy & ((1u << slot) - 1u));
        let packed = tree[first_child + rank * 2u];
        let node_index = tree[first_child + rank * 2u + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let tag = packed & 0xFFu;
        if tag == 1u {
            // Uniform-solid sub-frame (a Block cell deeper up the
            // chain collapses the whole local [0, 3)³ to one block).
            res.block = child_block_type(packed);
            return res;
        }
        if tag == 3u {
            // EntityRef cell mid-prefix — the sub-frame render
            // treats it as empty; the tag=3 dispatch is handled
            // elsewhere.
            return res;
        }
        // tag == 2u: descend into the Node child.
        node_idx = node_index;
    }

    // Pre-descent reached the deep terminal Node. Dispatch the
    // intra-cell walker from here.
    return walk_sub_frame(node_idx, u_l, v_l, r_l, walker_limit);
}

// ────────────── shader face-frame Jacobian (mirror of Rust)

/// Mirror of Rust `cubesphere::face_frame_jacobian`. Returns
/// `(c_body, J_col0, J_col1, J_col2)` at the face-frame corner
/// `(un, vn, rn)` of a cell of size `frame_size` in face-normalized
/// coords. Body-size is 3.0 (shader convention — the sphere body cell
/// fills `[0, 3)³`).
///
/// Columns of J are ∂body_pos / ∂(u_l, v_l, r_l) at the corner, with
/// `body_pos ≈ c_body + J · (u_l, v_l, r_l)` for local `(u_l, v_l, r_l)
/// ∈ [0, 3)³`. Used by the local-frame sub-frame DDA when stepping to
/// a neighbor sub-frame (the neighbor's J is recomputed at its own
/// corner).
struct FaceFrameJac {
    c_body: vec3<f32>,
    col_u: vec3<f32>,
    col_v: vec3<f32>,
    col_r: vec3<f32>,
}

fn face_frame_jacobian_shader(
    face: u32,
    un_corner: f32, vn_corner: f32, rn_corner: f32,
    frame_size: f32,
    inner_r: f32, outer_r: f32,
) -> FaceFrameJac {
    let body_size: f32 = 3.0;
    let center = vec3<f32>(body_size * 0.5);
    let n_axis = face_normal(face);
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);

    let e_u = un_corner * 2.0 - 1.0;
    let e_v = vn_corner * 2.0 - 1.0;
    let cu = ea_to_cube(e_u);
    let cv = ea_to_cube(e_v);
    let cos_u = cos(e_u * 0.7853981633974483); // π/4
    let cos_v = cos(e_v * 0.7853981633974483);
    let alpha_u = 1.5707963267948966 / (cos_u * cos_u); // π/2
    let alpha_v = 1.5707963267948966 / (cos_v * cos_v);

    let raw = n_axis + cu * u_axis + cv * v_axis;
    let nm = length(raw);
    let inv_nm = 1.0 / nm;
    let dir = raw * inv_nm;

    let r_body = (inner_r + rn_corner * (outer_r - inner_r)) * body_size;
    let dr_dbody = (outer_r - inner_r) * body_size;

    let c_body = center + dir * r_body;

    let s = frame_size / 3.0;
    let k_u = s * r_body * alpha_u * inv_nm;
    let k_v = s * r_body * alpha_v * inv_nm;
    let cu_nm = cu * inv_nm;
    let cv_nm = cv * inv_nm;
    let col_u = k_u * (u_axis - cu_nm * dir);
    let col_v = k_v * (v_axis - cv_nm * dir);
    let col_r = s * dr_dbody * dir;

    var out: FaceFrameJac;
    out.c_body = c_body;
    out.col_u = col_u;
    out.col_v = col_v;
    out.col_r = col_r;
    return out;
}

/// 3×3 inverse, stored column-major (columns in `col_u/v/r`). Returns
/// the inverse as three columns. f32 throughout — the sub-frame DDA
/// uses J_inv only for direction transforms in LOCAL coordinates; the
/// singular cases (determinant → 0) don't arise in the valid UVR
/// range.
struct Mat3Columns {
    col_u: vec3<f32>,
    col_v: vec3<f32>,
    col_r: vec3<f32>,
}

fn mat3_inv_shader(m: Mat3Columns) -> Mat3Columns {
    let a = m.col_u.x; let b = m.col_v.x; let c = m.col_r.x;
    let d = m.col_u.y; let e = m.col_v.y; let f = m.col_r.y;
    let g = m.col_u.z; let h = m.col_v.z; let i = m.col_r.z;
    let c00 =   e * i - f * h;
    let c01 = -(d * i - f * g);
    let c02 =   d * h - e * g;
    let c10 = -(b * i - c * h);
    let c11 =   a * i - c * g;
    let c12 = -(a * h - b * g);
    let c20 =   b * f - c * e;
    let c21 = -(a * f - c * d);
    let c22 =   a * e - b * d;
    let det = a * c00 + b * c01 + c * c02;
    let inv_det = 1.0 / det;
    var out: Mat3Columns;
    out.col_u = vec3<f32>(c00, c01, c02) * inv_det;
    out.col_v = vec3<f32>(c10, c11, c12) * inv_det;
    out.col_r = vec3<f32>(c20, c21, c22) * inv_det;
    return out;
}

fn mat3_mul_vec_shader(m: Mat3Columns, v: vec3<f32>) -> vec3<f32> {
    return m.col_u * v.x + m.col_v * v.y + m.col_r * v.z;
}

/// Upper bound on neighbor sub-frame transitions per ray — mirrors
/// CPU `MAX_NEIGHBOR_TRANSITIONS` in `src/world/raycast/sphere_sub.rs`.
const MAX_SPHERE_SUB_TRANSITIONS: u32 = 64u;

// ─── symbolic neighbor-step on UVR path (shader mirror of
// `Path::step_neighbor_cartesian`). Slot packing is identical to XYZ
// because UVR uses the same `slot_index(us, vs, rs) = rs*9 + vs*3 + us`
// formula — only the semantic axes differ.

/// Decompose a slot index into `(us, vs, rs)`, each ∈ 0..3.
fn slot_to_coords(slot: u32) -> vec3<u32> {
    let us = slot % 3u;
    let vs = (slot / 3u) % 3u;
    let rs = slot / 9u;
    return vec3<u32>(us, vs, rs);
}

fn coords_to_slot(us: u32, vs: u32, rs: u32) -> u32 {
    return rs * 9u + vs * 3u + us;
}

/// Return `(t_enter, t_exit)` for the ray crossing the local
/// `[0, 3)³` cube. `t_exit ≤ 0` → miss.
fn ray_sub_box_interval(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    var t_lo: f32 = -1e30;
    var t_hi: f32 =  1e30;
    for (var axis: u32 = 0u; axis < 3u; axis = axis + 1u) {
        let o = ro[axis];
        let d = rd[axis];
        if abs(d) < 1e-30 {
            if o < 0.0 || o >= 3.0 {
                return vec2<f32>(1e30, -1e30);
            }
            continue;
        }
        let t0 = (0.0 - o) / d;
        let t1 = (3.0 - o) / d;
        let a = min(t0, t1);
        let b = max(t0, t1);
        t_lo = max(t_lo, a);
        t_hi = min(t_hi, b);
    }
    return vec2<f32>(t_lo, t_hi);
}

/// Axis-exit t for a cell `[lo, lo+size]` along one axis. Returns
/// +∞ for rays parallel / going backward through both faces.
fn sub_axis_exit_t(p: f32, d: f32, lo: f32, hi: f32) -> f32 {
    if d > 1e-30  { return (hi - p) / d; }
    if d < -1e-30 { return (lo - p) / d; }
    return 1e30;
}

/// Helper that pre-descends `node` along `uvr_slots[0..prefix_len]`
/// and calls `walk_sub_frame` at the terminal Node. Unlike
/// `walk_from_deep_sub_frame`, this takes an explicit mutable-style
/// slot array (local f32 buffer, fed from the DDA's per-ray uvr
/// prefix that mutates on neighbor transitions).
///
/// Returns the SubWalkResult as if this were the starting face-root;
/// empty/solid cells mid-prefix collapse the whole `[0, 3)³` local box
/// to a single uniform cell (matching CPU `walk_from_deep_sub_frame`).
fn walk_from_deep_sub_frame_dyn(
    face_root_idx: u32,
    uvr_slots: array<u32, 64>,
    prefix_len: u32,
    u_l: f32, v_l: f32, r_l: f32,
    walker_limit: u32,
) -> SubWalkResult {
    var res: SubWalkResult;
    res.block = FACE_WALK_EMPTY;
    res.u_lo = 0.0;
    res.v_lo = 0.0;
    res.r_lo = 0.0;
    res.size = 3.0;

    var node_idx = face_root_idx;
    for (var i: u32 = 0u; i < prefix_len; i = i + 1u) {
        let slot = uvr_slots[i];
        let base = node_offsets[node_idx];
        if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
        let occupancy = tree[base];
        let first_child = tree[base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let mask = (occupancy >> slot) & 1u;
        if mask == 0u { return res; }
        let rank = countOneBits(occupancy & ((1u << slot) - 1u));
        let packed = tree[first_child + rank * 2u];
        let node_index = tree[first_child + rank * 2u + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }
        let tag = packed & 0xFFu;
        if tag == 1u {
            res.block = child_block_type(packed);
            return res;
        }
        if tag == 3u { return res; }
        node_idx = node_index;
    }
    return walk_sub_frame(node_idx, u_l, v_l, r_l, walker_limit);
}

/// Local-frame sphere DDA. GPU mirror of CPU `cs_raycast_local`.
/// `ray_origin_local`, `ray_dir_local` are already in sub-frame local
/// coords (J_inv applied CPU-side on the camera basis). On sub-frame
/// box exit the DDA transitions to the neighbor sub-frame via
/// symbolic UVR-path stepping + fresh Jacobian evaluation, mirroring
/// CPU `SphereSubFrame::with_neighbor_stepped` /
/// `cs_raycast_local`'s neighbor-step branch. Terminates when the
/// step would bubble past the face-root boundary (cross-face
/// transitions are deferred to a follow-up).
///
/// `face_root_idx` is the BFS index of the face-subtree root. The
/// walker pre-descends along `uniforms.sub_uvr_slots[..sub_meta.y]`
/// symbolically before running intra-cell DDA, so the sub-frame's
/// deep path may traverse `Child::Empty` links (dug regions) without
/// requiring a real Node at the terminal UVR depth.
fn sphere_in_sub_frame(
    face_root_idx: u32,
    ray_origin_local: vec3<f32>,
    ray_dir_local: vec3<f32>,
    walker_limit: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    if walker_limit == 0u { return result; }

    // --- Mutable per-ray sub-frame state (seeded from uniforms). ---
    let face = uniforms.sub_meta.x;
    let inner_r = uniforms.root_radii.x;
    let outer_r = uniforms.root_radii.y;
    // `uniforms.sub_meta.z` carries the face-root depth (see CPU
    // `SphereSubFrame::with_neighbor_stepped` + bindings.wgsl). The
    // shader reaches the same terminate condition via the bubble loop
    // below: when the UVR-prefix bubble-up reaches `depth == 0u` it
    // breaks out (`bubbled = false`) — matching CPU's cross-face
    // refusal in `with_neighbor_stepped`. No direct use of the value.
    let initial_prefix_len = uniforms.sub_meta.y;

    var un_corner = uniforms.sub_face_corner.x;
    var vn_corner = uniforms.sub_face_corner.y;
    var rn_corner = uniforms.sub_face_corner.z;
    let frame_size = uniforms.sub_face_corner.w;

    // Copy the uniform uvr prefix into a function-local array so we
    // can mutate it on neighbor transitions. Array length must match
    // the Rust-side `MAX_SPHERE_SUB_DEPTH` (64).
    var uvr_slots: array<u32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        uvr_slots[i] = 0u;
    }
    for (var i: u32 = 0u; i < initial_prefix_len; i = i + 1u) {
        uvr_slots[i] = sub_uvr_slot_at(i);
    }
    var uvr_prefix_len = initial_prefix_len;

    // J / J_inv start from uniforms.
    var j: Mat3Columns;
    j.col_u = uniforms.sub_j_col0.xyz;
    j.col_v = uniforms.sub_j_col1.xyz;
    j.col_r = uniforms.sub_j_col2.xyz;
    var j_inv: Mat3Columns;
    j_inv.col_u = uniforms.sub_j_inv_col0.xyz;
    j_inv.col_v = uniforms.sub_j_inv_col1.xyz;
    j_inv.col_r = uniforms.sub_j_inv_col2.xyz;

    // The caller already pre-multiplied the camera basis by J_inv, so
    // `ray_dir_local` IS the starting rd_local. Keep the rd_body
    // reference so we can recompute rd_local on neighbor transitions.
    // rd_body = J · rd_local (exact inverse; the caller's J_inv came
    // from this same matrix).
    let rd_body = mat3_mul_vec_shader(j, ray_dir_local);

    var ro_local = ray_origin_local;
    var rd_local = ray_dir_local;

    let interval0 = ray_sub_box_interval(ro_local, rd_local);
    var t_enter = interval0.x;
    var t_exit  = interval0.y;
    if t_exit <= 0.0 || t_enter >= t_exit { return result; }

    var t_span = max(abs(t_exit - t_enter), 1e-30);
    var t_nudge = t_span * 1e-5;
    var t = max(t_enter, 0.0) + t_nudge;

    var neighbor_transitions: u32 = 0u;

    var steps: u32 = 0u;
    loop {
        if steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let pos = ro_local + rd_local * t;

        let out_of_box =
            pos.x < 0.0 || pos.x >= 3.0 ||
            pos.y < 0.0 || pos.y >= 3.0 ||
            pos.z < 0.0 || pos.z >= 3.0 ||
            t >= t_exit;

        if out_of_box {
            // --- neighbor-transition path ---
            if neighbor_transitions >= MAX_SPHERE_SUB_TRANSITIONS { break; }

            // Pick exit axis / sign from pos.
            var axis_k: u32 = 0u;
            var sign_s: i32 = 0;
            var best_excess: f32 = -1.0;
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                let v = pos[k];
                var excess: f32;
                var sv: i32;
                if v >= 3.0 {
                    excess = v - 3.0;
                    sv = 1;
                } else if v < 0.0 {
                    excess = -v;
                    sv = -1;
                } else {
                    excess = -1.0;
                    sv = 0;
                }
                if excess > best_excess {
                    best_excess = excess;
                    axis_k = k;
                    sign_s = sv;
                }
            }
            if sign_s == 0 { break; }

            // Step the LAST slot of the UVR prefix along (axis_k,
            // sign_s), bubbling up through parents on slot overflow.
            // The slot-packing matches XYZ, so the arithmetic is
            // identical to `Path::step_neighbor_cartesian`.
            var depth: u32 = uvr_prefix_len;
            var bubbled = true;
            loop {
                if depth == 0u {
                    // No UVR slots left — the step bubbled past the
                    // face root; cross-face transition out of scope.
                    bubbled = false;
                    break;
                }
                let idx = depth - 1u;
                let slot = uvr_slots[idx];
                let coords = slot_to_coords(slot);
                var c0 = coords.x;
                var c1 = coords.y;
                var c2 = coords.z;
                // `axis_k` → which of (us, vs, rs) to step.
                var cur: i32;
                if axis_k == 0u { cur = i32(c0); }
                else if axis_k == 1u { cur = i32(c1); }
                else { cur = i32(c2); }
                let nxt = cur + sign_s;
                if nxt >= 0 && nxt <= 2 {
                    // Step in-place at this depth.
                    if axis_k == 0u { c0 = u32(nxt); }
                    else if axis_k == 1u { c1 = u32(nxt); }
                    else { c2 = u32(nxt); }
                    uvr_slots[idx] = coords_to_slot(c0, c1, c2);
                    // Parents untouched: re-wrap stopping here.
                    depth = idx + 1u;
                    break;
                }
                // Overflow: bubble up. After the parent step we'll
                // rewrite THIS slot to the wrapped value (2 if going
                // −, 0 if going +).
                depth = idx;
            }

            if !bubbled {
                // Bubble-up past the face-root boundary — cross-face
                // transition. Not implemented; terminate the DDA.
                break;
            }

            // On bubble-up, the depth-cursor walked back up to the
            // level that successfully stepped in-place, but any
            // deeper slots we skipped need to be re-wrapped to (0 or
            // 2) on the stepped axis. `depth` currently points just
            // past the in-place-stepped level; for any level below
            // that (index ≥ depth, < uvr_prefix_len), rewrap.
            let wrap_val: u32 = select(2u, 0u, sign_s > 0);
            for (var i2: u32 = depth; i2 < uvr_prefix_len; i2 = i2 + 1u) {
                let s2 = uvr_slots[i2];
                let c_ = slot_to_coords(s2);
                var d0 = c_.x;
                var d1 = c_.y;
                var d2 = c_.z;
                if axis_k == 0u { d0 = wrap_val; }
                else if axis_k == 1u { d1 = wrap_val; }
                else { d2 = wrap_val; }
                uvr_slots[i2] = coords_to_slot(d0, d1, d2);
            }

            // `face_root_depth` (on uniforms) mirrors CPU intent. The
            // shader reaches the same terminate condition via
            // `depth == 0u` in the bubble loop above.

            // Capture J_cur BEFORE we overwrite it below — needed
            // for the position transfer: local_new = J_new_inv · J_cur
            // · (local_cur − s·3·e_k).
            let j_cur = j;

            // Update corner coords incrementally on the stepped axis.
            let d_f = f32(sign_s);
            if axis_k == 0u {
                un_corner = un_corner + d_f * frame_size;
            } else if axis_k == 1u {
                vn_corner = vn_corner + d_f * frame_size;
            } else {
                rn_corner = rn_corner + d_f * frame_size;
            }

            // Recompute J / J_inv at the new corner.
            let ff = face_frame_jacobian_shader(
                face, un_corner, vn_corner, rn_corner, frame_size, inner_r, outer_r,
            );
            j.col_u = ff.col_u;
            j.col_v = ff.col_v;
            j.col_r = ff.col_r;
            j_inv = mat3_inv_shader(j);

            // Transfer position into the new basis via J_cur + J_inv.
            var shifted = pos;
            shifted[axis_k] = shifted[axis_k] - d_f * 3.0;
            var local_new = mat3_mul_vec_shader(j_inv, mat3_mul_vec_shader(j_cur, shifted));

            // Clamp the entry axis just inside the neighbor box.
            let eps_in = 3.0 * 1e-6;
            if sign_s == 1 {
                local_new[axis_k] = eps_in;
            } else {
                local_new[axis_k] = 3.0 - eps_in;
            }
            // Clamp non-entry axes into [0, 3) too.
            for (var k2: u32 = 0u; k2 < 3u; k2 = k2 + 1u) {
                if k2 == axis_k { continue; }
                if local_new[k2] < 0.0 { local_new[k2] = 0.0; }
                if local_new[k2] >= 3.0 { local_new[k2] = 3.0 - eps_in; }
            }

            // Re-transform rd_body into the new local basis.
            let rd_new = mat3_mul_vec_shader(j_inv, rd_body);

            ro_local = local_new;
            rd_local = rd_new;
            let interval = ray_sub_box_interval(ro_local, rd_local);
            let new_t_enter = interval.x;
            let new_t_exit  = interval.y;
            if new_t_exit <= 0.0 || new_t_enter >= new_t_exit { break; }
            t_span = max(abs(new_t_exit - new_t_enter), 1e-30);
            t_nudge = t_span * 1e-5;
            t = max(new_t_enter, 0.0) + t_nudge;
            t_exit = new_t_exit;

            neighbor_transitions = neighbor_transitions + 1u;
            continue;
        }

        let w = walk_from_deep_sub_frame_dyn(
            face_root_idx, uvr_slots, uvr_prefix_len,
            pos.x, pos.y, pos.z, walker_limit,
        );

        if w.block != FACE_WALK_EMPTY {
            // Shade the hit. Surface coloring uses simple radial
            // tinting + axis tint from the cell's exit face. The
            // full body-XYZ normal would require mapping local back
            // via the Jacobian, which the shader doesn't carry as a
            // matrix yet — `-rd_local` normalized suffices for
            // diffuse lighting (backface avoidance) since the ray
            // came from outside the cell.
            result.hit = true;
            result.t = t;
            let n_approx = normalize(-rd_local);
            result.normal = n_approx;
            let sun = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(n_approx, sun), 0.0);
            let ambient = 0.25;
            let rn_abs = rn_corner + w.r_lo * frame_size / 3.0;
            let tint = 0.55 + 0.45 * clamp(rn_abs, 0.0, 1.0);
            result.color = palette[w.block].rgb
                * (ambient + diffuse * 0.78) * tint;
            result.cell_min = vec3<f32>(w.u_lo, w.v_lo, w.r_lo);
            result.cell_size = w.size;
            return result;
        }

        // Advance past this empty cell. All six boundaries are
        // axis-aligned in local coords by the linearization.
        let t_u = sub_axis_exit_t(pos.x, rd_local.x, w.u_lo, w.u_lo + w.size);
        let t_v = sub_axis_exit_t(pos.y, rd_local.y, w.v_lo, w.v_lo + w.size);
        let t_r = sub_axis_exit_t(pos.z, rd_local.z, w.r_lo, w.r_lo + w.size);
        let t_min = min(min(t_u, t_v), t_r);
        if t_min <= 0.0 || t_min >= 1e29 {
            // Degenerate cell (parallel ray etc.) — force out-of-box
            // branch on the next iteration so we transition to the
            // neighbor or terminate.
            t = t_exit;
            continue;
        }
        t = t + t_min + t_nudge;
    }

    return result;
}
