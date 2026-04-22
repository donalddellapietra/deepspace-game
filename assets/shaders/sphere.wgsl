#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"
#include "sphere_debug.wgsl"

// Cubed-sphere geometry + DDA. One WGSL file with the face-math
// helpers, the face-subtree walker, and the unified sphere march.
// The CPU mirror lives in `src/world/cubesphere.rs` +
// `src/world/raycast/sphere.rs`.
//
// A runtime debug-paint mode is wired through
// `uniforms.sphere_debug_mode.x`; see `sphere_debug.wgsl` for the
// palette and `SPHERE_DEBUG_MODE_NAMES` (Rust) for the mode list.

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
    // Integer ratio form of the cell corner: `u_lo == f32(ratio_u) /
    // f32(3^ratio_depth)` bit-exact for ratio_depth ≤ WALK_SCALE_DEPTH
    // (= 15). Derived by integer divide inside the walker, so the
    // ratio is the *source of truth* and `u_lo`/`size` are the
    // reconstructed f32 view for downstream shading.
    ratio_u: u32,
    ratio_v: u32,
    ratio_r: u32,
    ratio_depth: u32,
}

/// Descend a face subtree from its root along `(un, vn, rn)` to the
/// terminal cell. Mirrors the CPU `walk_face_subtree` but without
/// `EMPTY_NODE` padding (the GPU doesn't need placement paths).
/// Empty-cell sentinel in `FaceWalkResult.block`. Palette index 0
/// is real (STONE), so we can't use 0 for "no hit". Matches
/// Rust's `REPRESENTATIVE_EMPTY`.
const FACE_WALK_EMPTY: u32 = 0xFFFEu;

/// Max depth at which the integer slot-pick is bit-exact. 3^15 =
/// 14_348_907 is the largest power of three that's representable as
/// an exact f32 integer (f32 mantissa = 23 bits + implicit 1; 3^16 =
/// 43M exceeds 2^24 and would round). The sole remaining precision
/// cost is in turning the f32 face-UV sample (`un_abs`) into its u32
/// scaled form — that's one f32 multiply with ≤ 0.5 ULP, so the
/// slot pick ends up bit-exact for all cells whose integer size is
/// ≥ 2 units (i.e., d ≤ 13 or so in practice). Walker callers that
/// pass `max_depth > WALK_SCALE_DEPTH` get clamped implicitly by the
/// POW3_TABLE index bound.
const WALK_SCALE_DEPTH: u32 = 15u;
const WALK_SCALE_U32: u32 = 14348907u;   // 3^15
const WALK_SCALE_F32: f32 = 14348907.0;

/// 3^d for d in [0, WALK_SCALE_DEPTH]. Module-scope const so it lives
/// in read-only memory and doesn't spill to registers. Every value is
/// exact in f32 (see WALK_SCALE_DEPTH doc).
const POW3_TABLE: array<u32, 16> = array<u32, 16>(
    1u, 3u, 9u, 27u,
    81u, 243u, 729u, 2187u,
    6561u, 19683u, 59049u, 177147u,
    531441u, 1594323u, 4782969u, 14348907u,
);

fn walk_face_subtree(
    face_root_idx: u32,
    un_scaled: u32, vn_scaled: u32, rn_scaled: u32,
    max_depth: u32,
) -> FaceWalkResult {
    // Integer slot-pick walker. The caller passes the face-UV sample
    // already scaled to `[0, WALK_SCALE_U32) = [0, 3^15)` u32s. Every
    // slot index at every depth is then computed by integer divide /
    // modulo — NO f32 rounding in the pick path. Compared to the
    // previous `floor((un_abs − u_lo) / child_size)` formulation,
    // this removes two f32 error sources:
    //   1. Iterative `child_size /= 3` drift (~6 ULP at d=10).
    //   2. `(un_abs − u_lo)` / `child_size` quotient jitter, which at
    //      d=10 peaked at ~6 % of a cell → systematic off-by-one on
    //      cells aligned to integer slot boundaries (= exactly where
    //      placed-block edges live → the "hollow" symptom).
    // f32 fields of FaceWalkResult (`u_lo`, `size`, ...) are still
    // derived from the integer ratio, so the bevel + plane-DDA paths
    // downstream see the cleanest possible `child_ratio_u / 3^d`
    // value — one f32 divide, 0.5 ULP total.
    var res: FaceWalkResult;
    res.block = FACE_WALK_EMPTY;
    res.depth = 0u;
    res.u_lo = 0.0;
    res.v_lo = 0.0;
    res.r_lo = 0.0;
    res.size = 1.0;
    res.ratio_u = 0u;
    res.ratio_v = 0u;
    res.ratio_r = 0u;
    res.ratio_depth = 0u;

    // Caller scales, but re-clamp here so the walker is self-defending
    // against an out-of-range upload (one cheap u32 min).
    let un_c = min(un_scaled, WALK_SCALE_U32 - 1u);
    let vn_c = min(vn_scaled, WALK_SCALE_U32 - 1u);
    let rn_c = min(rn_scaled, WALK_SCALE_U32 - 1u);
    // Clamp max_depth to the integer-pick horizon. Past d=15 the
    // cell_size_int drops below 1, and the pick becomes ambiguous —
    // but face_lod_depth in practice terminates well before here.
    let depth_cap = min(max_depth, WALK_SCALE_DEPTH);
    var node_idx = face_root_idx;
    var ratio_u: u32 = 0u;
    var ratio_v: u32 = 0u;
    var ratio_r: u32 = 0u;

    for (var d: u32 = 1u; d <= depth_cap; d = d + 1u) {
        let base = node_offsets[node_idx];
        if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
        let occupancy = tree[base];
        let first_child = tree[base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }

        // Integer cell-size at level d — exact for all d ≤ 15.
        let cell_size_int = POW3_TABLE[WALK_SCALE_DEPTH - d];
        let pow3_d = POW3_TABLE[d];

        // Slot pick via integer divide — exact: child_ratio_u is the
        // (d-level) integer coordinate of the cell containing the
        // sample. `us` is the 0..2 slot within the parent cell.
        let child_ratio_u = un_c / cell_size_int;
        let child_ratio_v = vn_c / cell_size_int;
        let child_ratio_r = rn_c / cell_size_int;
        let us = min(child_ratio_u - ratio_u * 3u, 2u);
        let vs = min(child_ratio_v - ratio_v * 3u, 2u);
        let rs = min(child_ratio_r - ratio_r * 3u, 2u);
        let slot = rs * 9u + vs * 3u + us;

        // f32-derived cell corner + size for downstream consumers.
        // Single divide each, 0.5 ULP error, independent of d.
        let inv_pow3_d = 1.0 / f32(pow3_d);
        let child_u_lo = f32(child_ratio_u) * inv_pow3_d;
        let child_v_lo = f32(child_ratio_v) * inv_pow3_d;
        let child_r_lo = f32(child_ratio_r) * inv_pow3_d;
        let child_size = inv_pow3_d;

        // Is this slot populated?
        let mask = (occupancy >> slot) & 1u;
        if mask == 0u {
            // Empty cell — terminate.
            res.depth = d;
            res.u_lo = child_u_lo;
            res.v_lo = child_v_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            res.ratio_u = child_ratio_u;
            res.ratio_v = child_ratio_v;
            res.ratio_r = child_ratio_r;
            res.ratio_depth = d;
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
            res.ratio_u = child_ratio_u;
            res.ratio_v = child_ratio_v;
            res.ratio_r = child_ratio_r;
            res.ratio_depth = d;
            return res;
        }
        // Tag 2 → descend into node.
        if d == depth_cap {
            // LOD-terminal. Use representative block; preserve the
            // empty sentinel when the subtree is all-empty.
            res.block = child_block_type(packed);
            res.depth = d;
            res.u_lo = child_u_lo;
            res.v_lo = child_v_lo;
            res.r_lo = child_r_lo;
            res.size = child_size;
            res.ratio_u = child_ratio_u;
            res.ratio_v = child_ratio_v;
            res.ratio_r = child_ratio_r;
            res.ratio_depth = d;
            return res;
        }
        node_idx = node_index;
        ratio_u = child_ratio_u;
        ratio_v = child_ratio_v;
        ratio_r = child_ratio_r;
    }
    // Loop exited without a terminal (only possible if depth_cap == 0,
    // which means max_depth == 0 — caller asked for no descent). Return
    // the root-cell sentinel so downstream treats this as empty.
    res.depth = depth_cap;
    res.ratio_depth = depth_cap;
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
//
// PRECISION GUARDS:
// * Ancestor loop `floor(up_u / up_s) * up_s` snaps to the nearest
//   multiple of `up_s`. At O(1) magnitudes this is f32-exact down to
//   `up_s ≈ 1e-7` (ULP of 0.5). Beyond that `up_u` either exceeds
//   1.0 — meaningless on a [0, 1) face — or snaps to zero. Bail when
//   either happens.
// * Descendant loop `uf = (un − dn_u) / dn_s` is the same
//   error-amplifying ratio that broke the walker. `un − dn_u` has
//   f32 precision ~1e-7 absolute (both operands O(1)); divided by
//   `dn_s` gives precision `1e-7 / dn_s`. When that exceeds ~1/3
//   (i.e., `dn_s < 3e-7`), `floor(uf * 3)` jitters by ±1 per pixel
//   and the sub-bevel line renders at drifting sub-cell positions —
//   producing the fine-grained rings the user sees at depth ~10+
//   even after the walker itself is precision-bounded. Bail when
//   `dn_s` would cross that precision threshold.
const BEVEL_DN_MIN_SIZE: f32 = 3e-7;
const BEVEL_UP_MAX_SIZE: f32 = 1.0;

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
        if up_s > BEVEL_UP_MAX_SIZE { break; }
        up_u = floor(up_u / up_s) * up_s;
        up_v = floor(up_v / up_s) * up_s;
        up_px = up_px * 3.0;
        b = b * bevel_level(un, vn, up_u, up_v, up_s, up_px);
    }

    var dn_u = u_lo; var dn_v = v_lo; var dn_s = size; var dn_px = base_px;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let cs = dn_s * (1.0 / 3.0);
        let cpx = dn_px * (1.0 / 3.0);
        // Two independent bail-outs: projected cell below 2 px (nothing
        // visible to draw), OR cell_size below f32 precision threshold
        // (the `(un − dn_u) / cs` ratio jitters → per-pixel noise that
        // reads as spurious grid lines).
        if cpx < 2.0 { break; }
        if cs < BEVEL_DN_MIN_SIZE { break; }
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

    // Per-pixel debug accumulator. Only written when
    // `uniforms.sphere_debug_mode.x != 0`; otherwise the updates below
    // still run (cheap, scalar) but the hit/exit paths skip the
    // override and the normal shading runs. Keeping the accumulator
    // unconditional means the debug hit-path doesn't need a second
    // code path for the "hit on first step" case.
    var dbg: SphereDebug = sphere_debug_init();
    let dbg_mode = uniforms.sphere_debug_mode.x;

    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        dbg.steps = steps;
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

        // Integer-scale the sample coords to `[0, 3^WALK_SCALE_DEPTH)`
        // before handing off to the walker. One f32 multiply, 0.5 ULP
        // error; the walker itself picks slots via integer divide
        // (no further f32 rounding). See `walk_face_subtree` for why
        // this replaces the previous `floor((un - u_lo)/child_size)`
        // — that pick systematically flipped off-by-one at d≥10 near
        // placed-block integer boundaries, and was the hollow-block
        // root cause visible in the F6 mode-4 sphere-debug paint.
        let un_scaled = u32(clamp(walk_un * WALK_SCALE_F32, 0.0, WALK_SCALE_F32 - 1.0));
        let vn_scaled = u32(clamp(walk_vn * WALK_SCALE_F32, 0.0, WALK_SCALE_F32 - 1.0));
        let rn_scaled = u32(clamp(walk_rn * WALK_SCALE_F32, 0.0, WALK_SCALE_F32 - 1.0));

        let walk_depth = face_lod_depth(t, reference_scale);
        let w = walk_face_subtree(face_node_idx, un_scaled, vn_scaled, rn_scaled, walk_depth);

        dbg.face_node_idx = face_node_idx;
        dbg.walker_depth = w.depth;
        dbg.walker_size = w.size;
        dbg.walker_ratio_u = w.ratio_u;
        dbg.walker_ratio_v = w.ratio_v;
        dbg.walker_ratio_depth = w.ratio_depth;

        // 0xFFFEu is REPRESENTATIVE_EMPTY — an empty terminal.
        // Palette index 0 is a real block (STONE), so we can't use
        // zero as the empty sentinel.
        if w.block != FACE_WALK_EMPTY {
            dbg.result_kind = 2u;
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
            // EXPERIMENT: use the smooth radial normal `n` for all
            // sphere hits instead of the discrete cell-wall normal
            // from `last_side`. Rationale: at d≥10 the cell walls
            // are at sub-pixel scale, so `last_side` flips between
            // adjacent rays producing mode-0 stripes. The sphere
            // surface's geometric normal IS radial; per-cell
            // wall-normal shading is a voxel-grid artifact, not a
            // property of the represented surface.
            //
            // If this looks right for terrain but makes placed
            // blocks look like rounded blobs, we'll need to split
            // the shading into "per-cell-wall for block interiors"
            // vs "smooth radial for terrain".
            let hit_normal = n;
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            if dbg_mode != 0u {
                // Debug path: bypass shading (sun, bevel, palette) and
                // paint the accumulator directly. We set the SUN NORMAL
                // here so `shade_pixel` computes full-unity diffuse;
                // combined with the 1e3 flat cell trick, this makes
                // debug colors render WITHOUT directional tint or
                // bevel. Otherwise mode-4 r_lo (0.3, 0.5, 1.0) was
                // rendering as directional-shaded on cube faces and
                // becoming unreadable.
                result.color = sphere_debug_color(dbg_mode, dbg);
                result.normal = normalize(vec3<f32>(0.4, 0.7, 0.3));
                let cs_dbg = max(length(camera.forward), 1.0) * 1e3;
                result.cell_min = camera.pos + ray_dir * t - vec3<f32>(cs_dbg * 0.5);
                result.cell_size = cs_dbg;
                return result;
            }
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

        dbg.result_kind = 1u;

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
        dbg.winning = winning;
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * w.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    // Loop exited without hitting content. In debug mode, force a hit
    // so the pixel renders SOMETHING — paint the debug color based on
    // the last-touched walker cell + step count. The fake hit's t is
    // clamped to `t_exit` so shade_pixel's `cell_min` trick still
    // lands near the ray path.
    if dbg_mode != 0u && dbg.steps > 0u {
        result.hit = true;
        result.t = min(max(t, t_enter + eps_init), t_exit);
        // Sun-aligned normal so shade_pixel's diffuse lighting is
        // unity, letting the debug color pass through untinted.
        result.normal = normalize(vec3<f32>(0.4, 0.7, 0.3));
        result.color = sphere_debug_color(dbg_mode, dbg);
        let cs_dbg = max(length(camera.forward), 1.0) * 1e3;
        result.cell_min = camera.pos + ray_dir * result.t - vec3<f32>(cs_dbg * 0.5);
        result.cell_size = cs_dbg;
    }

    return result;
}
