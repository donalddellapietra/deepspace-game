#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"

// Cubed-sphere geometry + DDA. One WGSL file with the face-math
// helpers, the face-subtree walker, and the unified sphere march.
// The CPU mirror lives in `src/world/cubesphere.rs` +
// `src/world/raycast/sphere.rs`.

/// When true, `sphere_in_sub_frame` paints solid debug colors per
/// control-flow branch so a developer can visually diagnose the
/// DDA loop without breakpointing a shader:
///   red    = initial ray-box interval miss (ray never entered)
///   blue   = neighbor-step transition consumed (every transition
///            fades the pixel darker, so repeated steps accumulate)
///   orange = cross-face terminate (bubble past face root)
///   green  = MAX_DDA_STEPS exhaust
///   yellow = MAX_SPHERE_SUB_TRANSITIONS exhaust
///   cyan   = `sign_s == 0` silent miss (t >= t_exit with no axis outside box)
///   white  = neighbor-transition ray-box interval miss
///   palette color = real hit (unchanged)
/// Left false by default; flip to true and reload the shader to
/// validate the geometry.
const SPHERE_DEBUG_PAINT: bool = false;

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
            // Neutralize `shade_pixel`'s `cube_face_bevel` — it picks
            // a cube face based on `result.normal` and projects
            // `(hit_pos - cell_min) / cell_size` onto that face's uv,
            // then darkens edges. For sphere hits the normal is
            // either a flat-face axis (±u/v/r) or the smooth radial
            // direction, and the cube_face_bevel's choice of uv
            // axes does NOT match the cell's face-normalized
            // (un, vn, rn) geometry — producing visible concentric-
            // circle banding across the curved sphere surface as
            // the radial direction sweeps between body axes. The
            // bevel here is already handled by `shape = bevel_layered`
            // above in face-normalized coords; shade_pixel's bevel
            // would double-apply darkening through the wrong axes.
            //
            // Trick: set cell_min/cell_size so `(hit_pos-cell_min)
            // / cell_size` = 0.5 for every pixel. cube_face_bevel
            // then gets uv=(0.5, 0.5) → edge=0.5 → smoothstep(0.02,
            // 0.14, 0.5) = 1.0 → no darkening applied.
            let cs = max(length(camera.forward), 1.0) * 1e3;
            result.cell_min = camera.pos + ray_dir * t - vec3<f32>(cs * 0.5);
            result.cell_size = cs;
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

/// Stable inverse of a matrix M = s · M_n where M_n has O(1) columns.
/// The face-frame Jacobian at depth m has M = (frame_size/3) · J_n:
/// `col_u/v/r` entries are O(1/3^m). At m ≥ 15 those entries fall
/// below f32 ULP near zero (~6e-8); cofactor products `e·i − f·h` in
/// `mat3_inv_shader` become two similar O(1/3^(2m)) values subtracted,
/// losing every significant digit — J_inv is garbage and the DDA
/// renders a collapsed smear past layer 20.
///
/// Fix: divide M by s first (elements become O(1), well-conditioned),
/// invert in f32 without precision loss, then multiply the result by
/// 1/s (since (s·M_n)^-1 = M_n^-1 / s). Net cost is two scalar
/// multiplies + one normal mat3_inv — identical register pressure.
fn mat3_inv_scaled_shader(m: Mat3Columns, s: f32) -> Mat3Columns {
    let inv_s = 1.0 / s;
    var mn: Mat3Columns;
    mn.col_u = m.col_u * inv_s;
    mn.col_v = m.col_v * inv_s;
    mn.col_r = m.col_r * inv_s;
    let mn_inv = mat3_inv_shader(mn);
    var out: Mat3Columns;
    out.col_u = mn_inv.col_u * inv_s;
    out.col_v = mn_inv.col_v * inv_s;
    out.col_r = mn_inv.col_r * inv_s;
    return out;
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

// ───────────────────────── sphere_in_sub_frame (rewrite) ──────────────
//
// GPU mirror of CPU `cs_raycast_local` in
// `src/world/raycast/sphere_sub.rs`. Written FROM SCRATCH to match
// that CPU spec line-by-line at deep UVR depths. The old hand-port had
// a precision collapse that returned smeared grey at depth ≥ ~5; this
// rewrite keeps every multi-precision step in the numerically-stable
// form the CPU version already uses.
//
// Precision model — the critical arithmetic identities:
//
//   `rd_body` is O(1).
//   `J_inv` entries are O(3^m) at UVR depth m.
//   `rd_local = J_inv · rd_body` is therefore O(3^m) — large but finite
//   in f32 for m up to ~25 (3^25 ≈ 8.5e11).
//
//   Inside `[0, 3)^3`, `ro_local` is O(1). `t_exit = (3 − ro_local)
//   / rd_local` is O(1) / O(3^m) = O(1/3^m) — representable in f32 at
//   any depth because it's a ratio.
//
//   `pos = ro_local + rd_local · t = O(1) + O(3^m)·O(1/3^m) = O(1)`.
//   The SUM of two values both ending up near O(1) is stable because
//   the O(3^m)·O(1/3^m) product was computed with f32 relative error,
//   not absolute error.
//
//   Neighbor transition: `local_new = J_new_inv · J_cur · (local_exit
//   − s·3·e_k)`. Each O(1/3^m) · O(3^m) · O(1) product chain stays
//   O(1) end-to-end — f32-stable at any depth.
//
// What was WRONG in the old code and is FIXED here:
//
//   * The old code returned DEBUG-WALKER pink/blue every iteration
//     whenever `w.size > 0.9`. That fired on the very first DDA step
//     of every ray, so the real DDA loop never executed. Rewrite
//     removes all debug early-returns; `SPHERE_DEBUG_PAINT` toggles
//     colour-coded painting on demand without short-circuiting the
//     real march.
//
//   * The old code had a spurious post-loop "DEBUG-SHADER-7 orange"
//     return that leaked every terminated-but-missed ray into the
//     framebuffer, swamping the real hit pixels. Rewrite returns a
//     proper miss unless SPHERE_DEBUG_PAINT wants a colour.
//
//   * The old code recovered `rd_body` via `J · ray_dir_local` inside
//     the function. The rewrite accepts `rd_body` as a dedicated
//     parameter so the caller can pass the exact body-frame value
//     (march.wgsl reconstructs it once, right at the dispatch site).
//     Shader precision work avoids repeating that inversion inside the
//     per-ray loop.
//
//   * The old code computed `rn_abs = rn_corner + w.r_lo * frame_size
//     / 3.0`. For m > 15 `frame_size / 3 = 1/3^(m+1)` is below the
//     ULP of `rn_corner` (~0.5, ULP ~6e-8), so the addition silently
//     drops the cell offset — every cell at the same shell reported
//     the same `rn_abs` and the tint collapsed. Rewrite carries
//     `frame_size/3` as a SEPARATE small term and only sums it when
//     we need the tint — the tint itself is tolerant of the 1e-7
//     truncation there, and the important DDA maths never forms that
//     sum.
//
// Parameters:
// * `face_root_idx` — BFS index of the face-subtree root node.
// * `ray_origin_local` — camera position in sub-frame local `[0,3)^3`.
//   Magnitude O(1). CPU-side sub-frame descent produced this without
//   any body-XYZ subtraction, so it's f32-precise at any depth.
// * `ray_dir_local` — camera-basis ray direction ALREADY rotated into
//   sub-frame local via J_inv. Magnitude O(3^m). f32-representable up
//   to m ≈ 25.
// * `rd_body` — the SAME ray direction expressed in body-XYZ, unit
//   length. Magnitude O(1). Used for re-deriving `rd_local` on every
//   neighbor transition (where J_inv changes).
// * `walker_limit` — max intra-cell walker descent depth.
fn sphere_in_sub_frame(
    face_root_idx: u32,
    ray_origin_local: vec3<f32>,
    ray_dir_local: vec3<f32>,
    rd_body: vec3<f32>,
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

    // --- Immutable subtree constants (seeded from uniforms). ---
    // `face`, `inner_r`, `outer_r` don't change across neighbor steps —
    // we stay inside one face subtree. `initial_prefix_len` is the UVR
    // prefix the walker inherits from CPU `compute_render_frame`; it
    // stays constant (neighbor-step preserves UVR path DEPTH via
    // slot-wrap, only rewrites the trailing slots).
    let face           = uniforms.sub_meta.x;
    let inner_r        = uniforms.root_radii.x;
    let outer_r        = uniforms.root_radii.y;
    let initial_prefix_len = uniforms.sub_meta.y;

    // --- Mutable per-neighbor-transition sub-frame state. ---

    // Face-normalised corner. Magnitude O(1) (∈ [0, 1)). On a
    // neighbor step we adjust one axis by ±frame_size = ±1/3^m, so at
    // deep m the increment is BELOW the f32 ULP of un_corner itself.
    // That's fine for J's evaluation point (J is nearly constant over
    // the sub-frame region anyway — curvature term O(1/3^(2m))), and
    // we NEVER form `un_corner + local * frame_size` inside the DDA
    // loop. See hit-shading block below for the one place we do and
    // why the tint is tolerant to the precision loss there.
    var un_corner  = uniforms.sub_face_corner.x;
    var vn_corner  = uniforms.sub_face_corner.y;
    var rn_corner  = uniforms.sub_face_corner.z;
    let frame_size = uniforms.sub_face_corner.w;

    // Copy the uniform UVR prefix into a function-local buffer so we
    // can mutate it on neighbor steps. Length 64 matches the CPU-side
    // MAX_SPHERE_SUB_DEPTH cap; unused tail stays zero and is never
    // read (guarded by `uvr_prefix_len`).
    var uvr_slots: array<u32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        uvr_slots[i] = 0u;
    }
    for (var i: u32 = 0u; i < initial_prefix_len; i = i + 1u) {
        uvr_slots[i] = sub_uvr_slot_at(i);
    }
    let uvr_prefix_len = initial_prefix_len;

    // J / J_inv seeded from uniforms. On every neighbor transition
    // they get recomputed from `face_frame_jacobian_shader` at the
    // new corner; `mat3_inv_shader` handles the 3×3 inversion in f32
    // (stable because J's columns are linearly independent —
    // `col_r` along the radial, `col_u`/`col_v` along face tangents).
    var j: Mat3Columns;
    j.col_u = uniforms.sub_j_col0.xyz;
    j.col_v = uniforms.sub_j_col1.xyz;
    j.col_r = uniforms.sub_j_col2.xyz;
    var j_inv: Mat3Columns;
    j_inv.col_u = uniforms.sub_j_inv_col0.xyz;
    j_inv.col_v = uniforms.sub_j_inv_col1.xyz;
    j_inv.col_r = uniforms.sub_j_inv_col2.xyz;

    // --- Ray state. `ro_local` is in the current sub-frame's local
    // coords (O(1)). `rd_local` is in the same frame (O(3^m)). On a
    // neighbor transition both are recomputed for the new frame;
    // `rd_body` is invariant across the whole DDA.
    var ro_local = ray_origin_local;
    var rd_local = ray_dir_local;

    // Initial ray-box interval inside `[0, 3)^3`.
    //
    // PRECISION NOTE — `ray_sub_box_interval` divides O(1)
    // boundary−ro_local by O(3^m) rd_local → result O(1/3^m). Both
    // operands are f32-representable; the division is one rounding
    // step. No catastrophic cancellation — numerator = boundary −
    // ro_local where both are O(1) but their DIFFERENCE can still be
    // O(1) (the whole box is size 3 in local).
    let interval0 = ray_sub_box_interval(ro_local, rd_local);
    var t_enter = interval0.x;
    var t_exit  = interval0.y;
    if t_exit <= 0.0 || t_enter >= t_exit {
        if SPHERE_DEBUG_PAINT {
            result.hit = true;
            result.t = 0.01;
            result.color = vec3<f32>(0.8, 0.1, 0.1); // red: interval miss
            result.normal = vec3<f32>(0.0, 1.0, 0.0);
            return result;
        }
        return result;
    }

    // PRECISION NOTE — `t_span`, `t_nudge`, `t`: all live at the same
    // O(1/3^m) scale as t_exit. Adding/subtracting two values of the
    // same scale is f32-safe because they share a common exponent.
    var t_span  = max(abs(t_exit - t_enter), 1e-30);
    var t_nudge = t_span * 1e-5;
    var t       = max(t_enter, 0.0) + t_nudge;

    var neighbor_transitions: u32 = 0u;
    var dda_steps: u32            = 0u;

    // --- Main DDA loop. Structure mirrors CPU `cs_raycast_local`. ---
    loop {
        if dda_steps >= 4096u {
            if SPHERE_DEBUG_PAINT {
                result.hit = true;
                result.t = 0.01;
                result.color = vec3<f32>(0.1, 0.9, 0.1); // green: dda cap
                result.normal = vec3<f32>(0.0, 1.0, 0.0);
                return result;
            }
            return result;
        }
        dda_steps = dda_steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        // PRECISION NOTE — `pos = ro_local + rd_local * t`:
        //   ro_local O(1), rd_local O(3^m), t O(1/3^m).
        //   rd_local * t = O(1). Adding to ro_local (also O(1)) is
        //   f32-stable; both exponents are near zero.
        let pos = ro_local + rd_local * t;

        let out_of_box =
            pos.x < 0.0 || pos.x >= 3.0 ||
            pos.y < 0.0 || pos.y >= 3.0 ||
            pos.z < 0.0 || pos.z >= 3.0 ||
            t >= t_exit;

        if out_of_box {
            // ------------------------------------------------------
            // Neighbor-transition branch. Steps the UVR path by one
            // slot along the exit axis, rebuilds J / J_inv / corner,
            // transfers position/direction into the neighbor's basis,
            // restarts the DDA in the new sub-frame. Mirrors CPU
            // `cs_raycast_local`'s out-of-box handler +
            // `SphereSubFrame::with_neighbor_stepped`.
            // ------------------------------------------------------
            if neighbor_transitions >= MAX_SPHERE_SUB_TRANSITIONS {
                if SPHERE_DEBUG_PAINT {
                    result.hit = true;
                    result.t = 0.01;
                    result.color = vec3<f32>(0.95, 0.95, 0.1); // yellow
                    result.normal = vec3<f32>(0.0, 1.0, 0.0);
                    return result;
                }
                return result;
            }

            // Pick exit axis / sign from `pos`. Prefer whichever axis
            // protrudes farthest past the box face — corner exits pick
            // the axis with the largest excursion so the neighbor step
            // lands on the geometrically correct face.
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
            if sign_s == 0 {
                // No axis outside the box — we hit the `t >= t_exit`
                // guard instead. Terminate the DDA; the ray left via
                // the sub-frame cap without a finite pos delta.
                if SPHERE_DEBUG_PAINT {
                    result.hit = true;
                    result.t = 0.01;
                    result.color = vec3<f32>(0.0, 0.9, 0.9); // cyan: t>=t_exit silent miss
                    result.normal = vec3<f32>(0.0, 1.0, 0.0);
                    return result;
                }
                return result;
            }

            // --- Bubble-up slot step. Mirror of
            //     `SphereSubFrame::with_neighbor_stepped` +
            //     `Path::step_neighbor_cartesian`.
            //
            // Pre-check: scan UVR slots (deepest first) for one whose
            // stepped-axis coord is NOT at the overflow boundary for
            // this direction. If all are pinned, the step would bubble
            // past the face root → cross-face transition, terminate.
            // ---------------------------------------------------------
            let boundary: u32 = select(0u, 2u, sign_s > 0);
            var can_step_in_face = false;
            if uvr_prefix_len > 0u {
                var ci: u32 = uvr_prefix_len;
                loop {
                    if ci == 0u { break; }
                    ci = ci - 1u;
                    let co = slot_to_coords(uvr_slots[ci]);
                    var coord: u32;
                    if axis_k == 0u { coord = co.x; }
                    else if axis_k == 1u { coord = co.y; }
                    else { coord = co.z; }
                    if coord != boundary { can_step_in_face = true; break; }
                }
            }
            if !can_step_in_face {
                if SPHERE_DEBUG_PAINT {
                    result.hit = true;
                    result.t = 0.01;
                    result.color = vec3<f32>(1.0, 0.55, 0.0); // orange
                    result.normal = vec3<f32>(0.0, 1.0, 0.0);
                    return result;
                }
                return result;
            }

            // Bubble step: find the deepest slot whose stepped-axis
            // coord can step in-place; write the new coord there; then
            // for every slot DEEPER than that one, wrap the
            // stepped-axis coord to `0` (going +) or `2` (going −).
            // This matches `Path::step_neighbor_cartesian`'s recursion
            // iteratively.
            var depth: u32 = uvr_prefix_len;
            loop {
                if depth == 0u { break; } // unreachable (pre-check above)
                let idx = depth - 1u;
                let co = slot_to_coords(uvr_slots[idx]);
                var c0 = co.x; var c1 = co.y; var c2 = co.z;
                var cur: i32;
                if axis_k == 0u { cur = i32(c0); }
                else if axis_k == 1u { cur = i32(c1); }
                else { cur = i32(c2); }
                let nxt = cur + sign_s;
                if nxt >= 0 && nxt <= 2 {
                    if axis_k == 0u { c0 = u32(nxt); }
                    else if axis_k == 1u { c1 = u32(nxt); }
                    else { c2 = u32(nxt); }
                    uvr_slots[idx] = coords_to_slot(c0, c1, c2);
                    depth = idx + 1u; // first untouched level
                    break;
                }
                depth = idx; // bubble up
            }
            // Rewrap any slots deeper than the one we just stepped.
            let wrap_val: u32 = select(2u, 0u, sign_s > 0);
            for (var i2: u32 = depth; i2 < uvr_prefix_len; i2 = i2 + 1u) {
                let co2 = slot_to_coords(uvr_slots[i2]);
                var d0 = co2.x; var d1 = co2.y; var d2 = co2.z;
                if axis_k == 0u { d0 = wrap_val; }
                else if axis_k == 1u { d1 = wrap_val; }
                else { d2 = wrap_val; }
                uvr_slots[i2] = coords_to_slot(d0, d1, d2);
            }

            // --- Basis update. Capture J_cur BEFORE overwriting it
            // so the position transfer has access to both matrices.
            let j_cur = j;

            // Incremental corner update. Corner is O(1), delta is
            // O(1/3^m). See precision note near un_corner's declaration.
            let d_f = f32(sign_s);
            if axis_k == 0u {
                un_corner = un_corner + d_f * frame_size;
            } else if axis_k == 1u {
                vn_corner = vn_corner + d_f * frame_size;
            } else {
                rn_corner = rn_corner + d_f * frame_size;
            }

            let ff = face_frame_jacobian_shader(
                face, un_corner, vn_corner, rn_corner, frame_size,
                inner_r, outer_r,
            );
            j.col_u = ff.col_u;
            j.col_v = ff.col_v;
            j.col_r = ff.col_r;
            // Stable scaled inverse — j's columns are O(frame_size/3)
            // = O(1/3^m). Naive mat3_inv loses f32 precision at m≥15
            // and the DDA collapses past layer 20. Passing frame_size/3
            // pre-scales j to O(1) for the cofactor computation.
            j_inv = mat3_inv_scaled_shader(j, frame_size / 3.0);

            // --- Position transfer:
            //     local_new = J_new_inv · J_cur · (local_exit − s·3·e_k)
            //
            // PRECISION NOTE — each factor:
            //   (local_exit − s·3·e_k): O(1). One axis zeroes (we're
            //       crossing it), others stay in [0, 3).
            //   J_cur · (…): O(1/3^m) · O(1) = O(1/3^m). Body-frame
            //       delta from the current sub-frame to the neighbor.
            //   J_new_inv · (…): O(3^m) · O(1/3^m) = O(1). The TWO
            //       large exponents cancel by multiplication (not by
            //       subtraction), which is f32-safe.
            //
            // So `local_new` stays O(1) in the neighbor frame. The
            // f32 relative error is O(1e-7), carried forward
            // unchanged — does NOT compound per neighbor step.
            var shifted = pos;
            shifted[axis_k] = shifted[axis_k] - d_f * 3.0;
            var local_new = mat3_mul_vec_shader(
                j_inv,
                mat3_mul_vec_shader(j_cur, shifted),
            );

            // Clamp the entry axis just inside the neighbor box on
            // the axis we crossed. Avoids an immediate re-exit on the
            // opposite face due to f32 drift in the matrix product.
            let eps_in: f32 = 3.0 * 1e-6;
            if sign_s == 1 {
                local_new[axis_k] = eps_in;
            } else {
                local_new[axis_k] = 3.0 - eps_in;
            }
            // Clamp non-crossed axes into [0, 3) too — corner-exit
            // drift otherwise kills the DDA in the neighbor.
            for (var k2: u32 = 0u; k2 < 3u; k2 = k2 + 1u) {
                if k2 == axis_k { continue; }
                if local_new[k2] < 0.0 { local_new[k2] = 0.0; }
                if local_new[k2] >= 3.0 { local_new[k2] = 3.0 - eps_in; }
            }

            // Recompute rd_local from body-frame reference (O(1)
            // input → O(3^m) output, one f32 matrix*vec). Avoids
            // compounding multiplicative drift across transitions —
            // every step gets a fresh local direction from rd_body.
            let rd_new = mat3_mul_vec_shader(j_inv, rd_body);

            ro_local = local_new;
            rd_local = rd_new;
            let interval = ray_sub_box_interval(ro_local, rd_local);
            let new_t_enter = interval.x;
            let new_t_exit  = interval.y;
            if new_t_exit <= 0.0 || new_t_enter >= new_t_exit {
                if SPHERE_DEBUG_PAINT {
                    result.hit = true;
                    result.t = 0.01;
                    result.color = vec3<f32>(1.0, 1.0, 1.0); // white: neighbor interval miss
                    result.normal = vec3<f32>(0.0, 1.0, 0.0);
                    return result;
                }
                return result;
            }
            t_span  = max(abs(new_t_exit - new_t_enter), 1e-30);
            t_nudge = t_span * 1e-5;
            t       = max(new_t_enter, 0.0) + t_nudge;
            t_exit  = new_t_exit;

            neighbor_transitions = neighbor_transitions + 1u;
            continue;
        }

        // ------------------------------------------------------------
        // Intra-sub-frame cell step. Walk the subtree at the current
        // `pos` to resolve the terminal walker cell; on solid hit,
        // shade and return; on empty, advance to the nearest cell
        // boundary and re-loop.
        // ------------------------------------------------------------
        let w = walk_from_deep_sub_frame_dyn(
            face_root_idx, uvr_slots, uvr_prefix_len,
            pos.x, pos.y, pos.z, walker_limit,
        );

        if w.block != FACE_WALK_EMPTY {
            // --- Hit shading. ---
            result.hit = true;
            result.t   = t;

            // Hit normal. Two different normals are needed here and
            // they disagreed in the previous code, producing visible
            // "concentric circle" artifacts at layers past ~12:
            //
            // * `diffuse_n` — a body-frame unit vector for the sun
            //   dot product. Body-frame because the sun direction is
            //   in world/body axes, not sub-frame-local. Using
            //   −rd_body normalized as the approximate surface normal
            //   (the ray came in along rd_body, so the face opposes).
            //
            // * `result.normal` — used by `shade_pixel`'s
            //   `cube_face_bevel` together with `result.cell_min` /
            //   `result.cell_size`. cell_min/size are in sub-frame
            //   LOCAL coords (walker-cell corner in [0,3)³), so the
            //   bevel's "which face did we hit" pick must also be in
            //   sub-frame local axes — otherwise the body-frame
            //   normal picks different cube faces as the body-space
            //   orientation rotates across the sphere, and each
            //   different face gives a different `uv`, which
            //   continuously varies → circular banding on a curved
            //   surface. Use −rd_local normalized (sub-frame-local)
            //   so cube_face_bevel's face pick matches the cell's
            //   actual local-axis face.
            let diffuse_n = normalize(-rd_body);
            let local_rd_len_sq = dot(rd_local, rd_local);
            var local_n: vec3<f32>;
            if local_rd_len_sq > 1e-30 {
                local_n = -rd_local / sqrt(local_rd_len_sq);
            } else {
                local_n = vec3<f32>(0.0, 1.0, 0.0);
            }
            result.normal = local_n;
            let sun = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(diffuse_n, sun), 0.0);
            let ambient: f32 = 0.25;

            // PRECISION NOTE — tint radial.
            //   `rn_corner` is O(1); `w.r_lo * frame_size / 3` is
            //   O(1/3^m). At deep m the sum loses the small term in
            //   f32 (ULP of rn_corner ~6e-8). We accept that drift:
            //   `tint = 0.55 + 0.45 * clamp(rn_abs, 0, 1)` is
            //   smooth over the whole shell, so a 1e-8 absolute drift
            //   in rn_abs shifts tint by ~5e-9 — invisible. We DO NOT
            //   use this sum anywhere else (DDA never needs abs coord).
            let rn_abs = rn_corner + w.r_lo * frame_size / 3.0;
            let tint = 0.55 + 0.45 * clamp(rn_abs, 0.0, 1.0);

            // Multi-scale cell-grid texture — mirrors sphere_in_cell's
            // `bevel_layered` call. Converts sub-frame local coords
            // into face-normalized (un, vn) so the existing
            // bevel_layered works unchanged.
            //
            // pos.x / pos.y are in sub-frame [0, 3) local; the
            // per-local-unit face-normalized step is frame_size/3.
            // walker cell face corner + size are similarly scaled.
            let fn_step = frame_size / 3.0;
            let hit_un = un_corner + pos.x * fn_step;
            let hit_vn = vn_corner + pos.y * fn_step;
            let cell_u_lo = un_corner + w.u_lo * fn_step;
            let cell_v_lo = vn_corner + w.v_lo * fn_step;
            let cell_face_size = w.size * fn_step;
            let shell_body = (outer_r - inner_r) * 3.0;
            let pixel_density = uniforms.screen_height
                / (2.0 * tan(camera.fov * 0.5));
            // `t` is in sub-frame local scale (O(1/3^m)); convert to
            // body units for the ray_dist arg by multiplying by the
            // face-local→body scale (fn_step * body_size / frame_size)
            // = body_size / 3. Works because rd_local has magnitude
            // O(3^m) = inverse of fn_step.
            let ray_dist_body = t * (3.0 * fn_step);
            let shape = bevel_layered(
                hit_un, hit_vn,
                cell_u_lo, cell_v_lo, cell_face_size,
                shell_body, ray_dist_body, pixel_density,
            );

            result.color = palette[w.block].rgb
                * (ambient + diffuse * 0.78) * tint * shape;
            result.cell_min = vec3<f32>(w.u_lo, w.v_lo, w.r_lo);
            result.cell_size = w.size;
            return result;
        }

        // --- Empty cell. Advance to this cell's nearest exit face.
        //
        // PRECISION NOTE — axis-exit t:
        //   Every boundary is axis-aligned in local coords (the
        //   Jacobian linearization makes `r_body = const` flatten to
        //   `r_local = const` too — col_r parallels body radial at
        //   the corner). `(boundary - pos[axis]) / rd_local[axis]`:
        //   numerator O(1), denominator O(3^m) → t O(1/3^m). f32
        //   representable. No cancellation (numerator is a signed O(1)
        //   difference, stable).
        let t_u = sub_axis_exit_t(pos.x, rd_local.x, w.u_lo, w.u_lo + w.size);
        let t_v = sub_axis_exit_t(pos.y, rd_local.y, w.v_lo, w.v_lo + w.size);
        let t_r = sub_axis_exit_t(pos.z, rd_local.z, w.r_lo, w.r_lo + w.size);
        let t_min = min(min(t_u, t_v), t_r);
        if t_min <= 0.0 || t_min >= 1e29 {
            // Degenerate step (zero-span cell, parallel ray, …).
            // Force out-of-box branch on the next iteration so the
            // neighbor-transition logic handles termination.
            t = t_exit;
            continue;
        }
        t = t + t_min + t_nudge;
    }
    // Unreachable in practice (every branch inside the loop either
    // `continue`s or `return`s). WGSL still requires a function-level
    // terminator, so return the default (miss) result.
    return result;
}
