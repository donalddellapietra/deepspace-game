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

/// Local-frame sphere DDA. GPU mirror of CPU `cs_raycast_local`.
/// `ray_origin_local`, `ray_dir_local` are already in sub-frame local
/// coords (J_inv applied CPU-side on the camera basis).
fn sphere_in_sub_frame(
    sub_frame_idx: u32,
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

    let interval = ray_sub_box_interval(ray_origin_local, ray_dir_local);
    let t_enter = interval.x;
    let t_exit  = interval.y;
    if t_exit <= 0.0 || t_enter >= t_exit { return result; }

    let t_span = max(abs(t_exit - t_enter), 1e-30);
    let t_nudge = t_span * 1e-5;
    var t = max(t_enter, 0.0) + t_nudge;

    var steps: u32 = 0u;
    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let pos = ray_origin_local + ray_dir_local * t;
        if pos.x < 0.0 || pos.x >= 3.0
            || pos.y < 0.0 || pos.y >= 3.0
            || pos.z < 0.0 || pos.z >= 3.0 { break; }

        let w = walk_sub_frame(sub_frame_idx, pos.x, pos.y, pos.z, walker_limit);

        if w.block != FACE_WALK_EMPTY {
            // Shade the hit. Surface coloring uses simple radial
            // tinting + axis tint from the cell's exit face. The
            // full body-XYZ normal would require mapping local back
            // via the Jacobian, which the shader doesn't carry as a
            // matrix yet — `-ray_dir_local` normalized suffices for
            // diffuse lighting (backface avoidance) since the ray
            // came from outside the cell.
            result.hit = true;
            result.t = t;
            let n_approx = normalize(-ray_dir_local);
            result.normal = n_approx;
            let sun = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(n_approx, sun), 0.0);
            let ambient = 0.25;
            let rn_abs = uniforms.sub_face_corner.z
                + w.r_lo * uniforms.sub_face_corner.w / 3.0;
            let tint = 0.55 + 0.45 * clamp(rn_abs, 0.0, 1.0);
            result.color = palette[w.block].rgb
                * (ambient + diffuse * 0.78) * tint;
            result.cell_min = vec3<f32>(w.u_lo, w.v_lo, w.r_lo);
            result.cell_size = w.size;
            return result;
        }

        // Advance past this empty cell. All six boundaries are
        // axis-aligned in local coords by the linearization.
        let t_u = sub_axis_exit_t(pos.x, ray_dir_local.x, w.u_lo, w.u_lo + w.size);
        let t_v = sub_axis_exit_t(pos.y, ray_dir_local.y, w.v_lo, w.v_lo + w.size);
        let t_r = sub_axis_exit_t(pos.z, ray_dir_local.z, w.r_lo, w.r_lo + w.size);
        let t_min = min(min(t_u, t_v), t_r);
        if t_min <= 0.0 || t_min >= 1e29 { break; }
        t = t + t_min + t_nudge;
    }

    return result;
}
