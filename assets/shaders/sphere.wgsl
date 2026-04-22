#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"

// Cubed-sphere geometry + DDA. One WGSL file with the face-math
// helpers, the face-subtree walker, and the unified sphere march
// (`sphere_dda`). The CPU mirror lives in `src/world/cubesphere.rs`
// + `src/world/raycast/unified.rs`.

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
    // Integer ratio form of the cell corner: `u_lo == f32(ratio_u) *
    // size` up to ~0.5 ULP rounding. Tracked as u32 accumulator
    // `ratio = parent*3 + slot`, so precision is preserved regardless
    // of depth. u32 covers ratio_depth ≤ 20 (3^20 ≈ 3.5e9 < 2^32);
    // for deeper descent the mantissa-cast loses bits but stays more
    // accurate than the additive accumulator `u_lo += us * child_size`.
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

fn walk_face_subtree(
    face_root_idx: u32,
    un_in: f32, vn_in: f32, rn_in: f32,
    max_depth: u32,
) -> FaceWalkResult {
    // Error-bounded walker (mirror of CPU `walk_face_subtree`).
    //
    // The previous implementation iterated `un = un * 3 − us` per
    // level; that recurrence amplifies f32 error by 3× each step, so
    // past depth ~9 adjacent pixels near cell boundaries snap to
    // random neighbors (the ring artifact). Here we keep the sample
    // coords `un_abs / vn_abs / rn_abs` IMMUTABLE and derive each
    // level's slot from the accumulating cell origin:
    //   us = floor((un_abs − u_lo) / child_size)
    // Error stays bounded at ~f32 ULP instead of amplifying. Pushes
    // the precision wall from ~depth 9 to ~depth 14; past that, failure
    // is off-by-one (visual blur) rather than off-by-hundreds (chaos).
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

    let un_abs = clamp(un_in, 0.0, 0.9999999);
    let vn_abs = clamp(vn_in, 0.0, 0.9999999);
    let rn_abs = clamp(rn_in, 0.0, 0.9999999);
    var node_idx = face_root_idx;
    var u_lo: f32 = 0.0;
    var v_lo: f32 = 0.0;
    var r_lo: f32 = 0.0;
    var size: f32 = 1.0;
    // Integer ratio accumulators — exact at every step.
    var ratio_u: u32 = 0u;
    var ratio_v: u32 = 0u;
    var ratio_r: u32 = 0u;

    for (var d: u32 = 1u; d <= max_depth; d = d + 1u) {
        let base = node_offsets[node_idx];
        if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
        let occupancy = tree[base];
        let first_child = tree[base + 1u];
        if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }

        let child_size = size / 3.0;
        // Absolute-coord slot pick. `(abs − lo) / child_size` lives
        // in [0, 3); clamp to [0, 2] to guard against f32 rounding
        // that could push the quotient negative on exact-boundary
        // hits or to 3.0 on the upper edge.
        let us = u32(clamp(floor((un_abs - u_lo) / child_size), 0.0, 2.0));
        let vs = u32(clamp(floor((vn_abs - v_lo) / child_size), 0.0, 2.0));
        let rs = u32(clamp(floor((rn_abs - r_lo) / child_size), 0.0, 2.0));
        let slot = rs * 9u + vs * 3u + us;

        let child_ratio_u = ratio_u * 3u + us;
        let child_ratio_v = ratio_v * 3u + vs;
        let child_ratio_r = ratio_r * 3u + rs;
        // Ratio-derived cell corner — one multiply, ~0.5 ULP. The
        // alternative `u_lo + us*child_size` compounds ~1 ULP per
        // level; by m ≈ 10 the 6 cell-wall plane normals built from
        // `ea_to_cube(u_lo*2-1)` have drifted enough that adjacent
        // walls' normals collapse in f32, the ray marches through
        // solid content without detecting it, and the cell renders
        // hollow.
        let child_u_lo = f32(child_ratio_u) * child_size;
        let child_v_lo = f32(child_ratio_v) * child_size;
        let child_r_lo = f32(child_ratio_r) * child_size;

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
        if d == max_depth {
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
        u_lo = child_u_lo;
        v_lo = child_v_lo;
        r_lo = child_r_lo;
        size = child_size;
        ratio_u = child_ratio_u;
        ratio_v = child_ratio_v;
        ratio_r = child_ratio_r;
        // NOTE: un_abs / vn_abs / rn_abs are NOT updated; they stay
        // as the immutable ray-sample reference for the lifetime of
        // the walk. That's the whole point of this reformulation.
    }
    res.u_lo = u_lo;
    res.v_lo = v_lo;
    res.r_lo = r_lo;
    res.size = size;
    res.ratio_u = ratio_u;
    res.ratio_v = ratio_v;
    res.ratio_r = ratio_r;
    res.ratio_depth = max_depth;
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

// ─────────────────────────── face/body coord conversions (mirror of CPU)

fn face_uv_to_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    let cu = ea_to_cube(u);
    let cv = ea_to_cube(v);
    let n = face_normal(face);
    let ua = face_u_axis(face);
    let va = face_v_axis(face);
    let raw = vec3<f32>(
        n.x + cu * ua.x + cv * va.x,
        n.y + cu * ua.y + cv * va.y,
        n.z + cu * ua.z + cv * va.z,
    );
    return normalize(raw);
}

fn face_space_to_body_point(
    face: u32,
    un: f32, vn: f32, rn: f32,
    inner_r: f32, outer_r: f32,
    body_size: f32,
) -> vec3<f32> {
    let center = vec3<f32>(body_size * 0.5);
    let radius = (inner_r + rn * (outer_r - inner_r)) * body_size;
    let dir = face_uv_to_dir(face, un * 2.0 - 1.0, vn * 2.0 - 1.0);
    return center + dir * radius;
}

struct FacePointShader {
    face: u32,
    un: f32,
    vn: f32,
    rn: f32,
    valid: u32, // 0 = degenerate, 1 = ok
}

fn body_point_to_face_space(
    point_body: vec3<f32>,
    inner_r: f32, outer_r: f32,
    body_size: f32,
) -> FacePointShader {
    var out: FacePointShader;
    out.valid = 0u;
    out.face = 0u;
    out.un = 0.0;
    out.vn = 0.0;
    out.rn = 0.0;
    let center = vec3<f32>(body_size * 0.5);
    let offset = point_body - center;
    let r = length(offset);
    if r <= 1e-12 { return out; }
    let n = offset / r;
    let face = pick_face(n);
    let n_axis = face_normal(face);
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);
    let axis_dot = dot(n, n_axis);
    if abs(axis_dot) <= 1e-12 { return out; }
    let cube_u = dot(n, u_axis) / axis_dot;
    let cube_v = dot(n, v_axis) / axis_dot;
    let inner = inner_r * body_size;
    let outer = outer_r * body_size;
    let shell = outer - inner;
    if shell <= 0.0 { return out; }
    out.face = face;
    out.un = clamp((cube_to_ea(cube_u) + 1.0) * 0.5, 0.0, 0.9999999);
    out.vn = clamp((cube_to_ea(cube_v) + 1.0) * 0.5, 0.0, 0.9999999);
    out.rn = clamp((r - inner) / shell, 0.0, 0.9999999);
    out.valid = 1u;
    return out;
}

// ─────────────────────────────────────────── analytical face Jacobian
//
// Mat3Cols is the column-major form of a 3×3 matrix: each `col_*` is
// one column. `face_jacobian_normalized` returns J such that
//
//     d(body_pos) / d(un, vn, rn)  ≈  J  (at the given un/vn/rn).
//
// For a face-subtree cell of size `frame_size`, the cell's local
// residual ∈ [0, 1)³ relates to body-XYZ via
//
//     body_pos(residual) ≈ corner_body + frame_size · J · residual
//
// where `corner_body = face_space_to_body_point(face, un, vn, rn, …)`
// at the cell's lower corner. The MULTIPLY by `frame_size` is an exact
// scalar factor — it's pulled out of the matrix to keep J's columns
// at O(1) magnitude regardless of cell depth, so the inverse stays
// well-conditioned.
//
// `mat3_inverse_cols` and `mat3_inv_mul_vec` then give the per-axis
// rate of residual change per unit world-t:
//
//     rate = (J_inv · rd_body) / frame_size
//
// (with the `/ frame_size` absorbed into `t_exit = … * frame_size /
// rate_normalized` at the use site so we never form the huge ratio).

struct Mat3Cols {
    col_u: vec3<f32>,
    col_v: vec3<f32>,
    col_r: vec3<f32>,
}

fn face_jacobian_normalized(
    face: u32,
    un: f32, vn: f32, rn: f32,
    inner_r: f32, outer_r: f32, body_size: f32,
) -> Mat3Cols {
    let u = un * 2.0 - 1.0;
    let v = vn * 2.0 - 1.0;
    let n_axis = face_normal(face);
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);
    let cu = ea_to_cube(u);
    let cv = ea_to_cube(v);
    let raw = n_axis + cu * u_axis + cv * v_axis;
    let raw_len = length(raw);
    let dir = raw / raw_len;
    let radius = (inner_r + rn * (outer_r - inner_r)) * body_size;
    // d(cu)/d(un) = sec²(u·π/4) · π/2  (chain rule across u = 2un−1
    // gives factor 2; sec²·π/4 from d(tan)/du; combined π/2).
    let cos_u = cos(u * 0.78539816);
    let cos_v = cos(v * 0.78539816);
    let alpha_u = 1.5707963 / (cos_u * cos_u);
    let alpha_v = 1.5707963 / (cos_v * cos_v);
    let dir_dot_ua = dot(dir, u_axis);
    let dir_dot_va = dot(dir, v_axis);
    let d_dir_du = alpha_u * (u_axis - dir * dir_dot_ua) / raw_len;
    let d_dir_dv = alpha_v * (v_axis - dir * dir_dot_va) / raw_len;
    var out: Mat3Cols;
    out.col_u = d_dir_du * radius;
    out.col_v = d_dir_dv * radius;
    out.col_r = dir * (outer_r - inner_r) * body_size;
    return out;
}

// Inverse of a 3×3 matrix. Returns the inverse in COLUMN-MAJOR form
// — `inv.col_u` is the FIRST COLUMN of the inverse matrix. To
// multiply `inverse · vec`, use `mat3_inv_mul_vec` below.
fn mat3_inverse_cols(m: Mat3Cols) -> Mat3Cols {
    let cross_vr = cross(m.col_v, m.col_r);
    let cross_ru = cross(m.col_r, m.col_u);
    let cross_uv = cross(m.col_u, m.col_v);
    let det = dot(m.col_u, cross_vr);
    let inv_det = select(0.0, 1.0 / det, abs(det) > 1e-30);
    // For inverse(M), row i = cross(M.col_(i+1), M.col_(i+2)) / det.
    // We store as columns of the result struct — but caller uses
    // mat3_inv_mul_vec which knows row vs column.
    var out: Mat3Cols;
    out.col_u = cross_vr * inv_det;
    out.col_v = cross_ru * inv_det;
    out.col_r = cross_uv * inv_det;
    return out;
}

// `inv_rows.col_u` is treated as the first ROW of the inverse matrix
// (mat3_inverse_cols stores rows in `col_*` slots; this is the
// matching consumer).
fn mat3_inv_mul_vec(inv_rows: Mat3Cols, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(inv_rows.col_u, v),
        dot(inv_rows.col_v, v),
        dot(inv_rows.col_r, v),
    );
}

// 3^n for n ≤ 20 (3^20 ≈ 3.5e9 fits in u32). Beyond that, returns 0
// — caller must guard `ratio_depth ≤ 20`.
fn pow_3_u32(n: u32) -> u32 {
    var p: u32 = 1u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        p = p * 3u;
    }
    return p;
}

// ─────────────────────────────────────────── unified sphere DDA

/// **Sphere DDA** (residual + slot-path). Mirrors `unified_raycast`
/// from `src/world/raycast/unified.rs`: per-cell residual ∈ [0, 1)³,
/// integer slot-ratios for the cell's face-normalized corner, and
/// per-cell analytical Jacobian via `face_jacobian_normalized`. No
/// `pos = oc + ray_dir * t` per-step recompute — bounded f32
/// precision regardless of face-subtree depth.
///
/// State per ray:
/// - `face`: which cube face the ray is currently in
/// - `(ratio_u, ratio_v, ratio_r, ratio_depth)`: integer slot path
///   within the face. The cell's lower corner is at face-norm coords
///   `ratio / 3^ratio_depth`. Integer arithmetic; precise to depth 20.
/// - `residual ∈ [0, 1)³`: ray's position within the current cell.
///   Updated incrementally, snapped lossless on the exit axis.
/// - `t_world`: accumulated for hit reporting only — never used to
///   recompute geometry.
/// - `last_axis ∈ {0,1,2,6}`: which axis the previous step exited
///   via, so we can skip that plane this step (would re-detect at
///   t == current).
///
/// Per iteration:
/// 1. Walker descends face_root using `(un, vn, rn) = (ratio +
///    residual) / 3^ratio_depth`, capped at `face_lod_depth(t_world)`.
/// 2. Re-align state to walker's terminal depth (might be coarser).
/// 3. If terminal block != EMPTY: shade and return hit.
/// 4. Else: compute J at terminal cell corner; rate = J_inv · ray_dir.
///    Per-axis t_local = (target − residual[k]) · cell_size /
///    rate_normalized[k]. All operands O(1) — bounded f32 precision
///    regardless of cell depth.
/// 5. Snap residual on min-axis (lossless), advance other axes
///    incrementally; step ratio integer ±1 on min-axis.
/// 6. If ratio overflows: face-seam (u/v) or shell-exit (r) — both
///    terminate for now.
fn sphere_dda(
    body_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
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

    let ray_dir = normalize(ray_dir_in);
    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;
    if shell <= 0.0 { return result; }

    // Ray-sphere intersect with outer shell to find entry t.
    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit_outer = -b + sq;
    if t_exit_outer <= 0.0 { return result; }

    let pixel_density = uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
    let eps_init = max(shell * 1e-5, 1e-7);

    // SINGLE absolute-coord recompute: the entry point. From here on,
    // state is ratio + residual + t_world only — NO `oc + ray_dir * t`
    // per-step recompute.
    var t_world = t_enter + eps_init;
    let entry_local = oc + ray_dir * t_world;
    let entry_r = length(entry_local);
    if entry_r >= cs_outer || entry_r < cs_inner { return result; }
    let entry_n = entry_local / entry_r;
    var face = pick_face(entry_n);

    // Locate face root via body's child at face_slot.
    let body_base = node_offsets[body_idx];
    if ENABLE_STATS { ray_loads_offsets = ray_loads_offsets + 1u; }
    let body_occ = tree[body_base];
    let body_first = tree[body_base + 1u];
    if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 2u; }

    // Lookup face_node_idx for the current face. Re-fetched on
    // face-seam crossings.
    var face_node_idx: u32 = 0u;
    var fslot = face_slot(face);
    if ((body_occ >> fslot) & 1u) == 0u { return result; }
    var frank = countOneBits(body_occ & ((1u << fslot) - 1u));
    face_node_idx = tree[body_first + frank * 2u + 1u];
    if ENABLE_STATS { ray_loads_tree = ray_loads_tree + 1u; }

    // Initial face-norm coords from the entry point. Single absolute
    // computation — O(1) magnitudes.
    var n_axis = face_normal(face);
    var u_axis = face_u_axis(face);
    var v_axis = face_v_axis(face);
    var axis_dot = dot(entry_n, n_axis);
    if abs(axis_dot) < 1e-6 { return result; }
    var cu_init = dot(entry_n, u_axis) / axis_dot;
    var cv_init = dot(entry_n, v_axis) / axis_dot;
    var un_entry = clamp((cube_to_ea(cu_init) + 1.0) * 0.5, 0.0, 0.9999999);
    var vn_entry = clamp((cube_to_ea(cv_init) + 1.0) * 0.5, 0.0, 0.9999999);
    var rn_entry = clamp((entry_r - cs_inner) / shell, 0.0, 0.9999999);

    // Initial state at face_root level (depth 0): single cell covers
    // the whole face. residual is just the entry's face-norm coords.
    var ratio_u: u32 = 0u;
    var ratio_v: u32 = 0u;
    var ratio_r: u32 = 0u;
    var ratio_depth: u32 = 0u;
    var residual: vec3<f32> = vec3<f32>(un_entry, vn_entry, rn_entry);
    var last_axis: u32 = 6u;  // 6 = none / first iteration

    var steps: u32 = 0u;
    loop {
        if t_world >= t_exit_outer || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        // Compute walker input: (un, vn, rn) = (ratio + residual) /
        // cells. PRECISION: ratio + residual fits in f32 mantissa
        // when ratio_depth ≤ 20 (3^20 ≈ 3.5e9 < 2^32, cast to f32
        // preserves ~24 bits). For ratio_depth ≤ 16 (3^16 ≈ 4.3e7),
        // ratio_u as f32 is exact and the divide gives un_abs with
        // ~7 digits of precision — sufficient for walker descent.
        let cells_state = pow_3_u32(ratio_depth);
        let cells_state_f = f32(cells_state);
        let inv_cells = 1.0 / cells_state_f;
        let un_abs = (f32(ratio_u) + residual.x) * inv_cells;
        let vn_abs = (f32(ratio_v) + residual.y) * inv_cells;
        let rn_abs = (f32(ratio_r) + residual.z) * inv_cells;

        // Walker descent. Cap at face_lod_depth based on world-t for
        // screen-LOD termination.
        let walker_max_depth = face_lod_depth(t_world, shell);
        let w = walk_face_subtree(face_node_idx, un_abs, vn_abs, rn_abs, walker_max_depth);

        // Re-align state to walker's terminal depth. The walker's
        // ratio_u/v/r/depth describes the cell it terminated at. If
        // walker descended deeper than our state's ratio_depth, our
        // residual maps into the finer cell. If walker terminated
        // coarser (LOD-cap or empty subtree), our residual covers a
        // larger fraction of the coarse cell.
        if w.ratio_depth > ratio_depth {
            // Walker is deeper — refine state.
            let diff = w.ratio_depth - ratio_depth;
            let factor = pow_3_u32(diff);
            let factor_f = f32(factor);
            let scaled_x = residual.x * factor_f;
            let scaled_y = residual.y * factor_f;
            let scaled_z = residual.z * factor_f;
            let frac_x = floor(scaled_x);
            let frac_y = floor(scaled_y);
            let frac_z = floor(scaled_z);
            ratio_u = ratio_u * factor + u32(frac_x);
            ratio_v = ratio_v * factor + u32(frac_y);
            ratio_r = ratio_r * factor + u32(frac_z);
            residual = vec3<f32>(
                clamp(scaled_x - frac_x, 0.0, 1.0 - 1e-6),
                clamp(scaled_y - frac_y, 0.0, 1.0 - 1e-6),
                clamp(scaled_z - frac_z, 0.0, 1.0 - 1e-6),
            );
            ratio_depth = w.ratio_depth;
        } else if w.ratio_depth < ratio_depth {
            // Walker is coarser — collapse state.
            let diff = ratio_depth - w.ratio_depth;
            let factor = pow_3_u32(diff);
            let factor_f = f32(factor);
            let new_ratio_u = ratio_u / factor;
            let new_ratio_v = ratio_v / factor;
            let new_ratio_r = ratio_r / factor;
            let local_u = ratio_u - new_ratio_u * factor;
            let local_v = ratio_v - new_ratio_v * factor;
            let local_r = ratio_r - new_ratio_r * factor;
            residual = vec3<f32>(
                clamp((f32(local_u) + residual.x) / factor_f, 0.0, 1.0 - 1e-6),
                clamp((f32(local_v) + residual.y) / factor_f, 0.0, 1.0 - 1e-6),
                clamp((f32(local_r) + residual.z) / factor_f, 0.0, 1.0 - 1e-6),
            );
            ratio_u = new_ratio_u;
            ratio_v = new_ratio_v;
            ratio_r = new_ratio_r;
            ratio_depth = w.ratio_depth;
        }
        // Walker's exact corner (from integer ratios in the walker —
        // matches our state).
        let cell_size = w.size;

        if w.block != FACE_WALK_EMPTY {
            // HIT. Shade and return.
            //
            // Hit normal: derive from the LAST exit axis (which face
            // of the cell we entered through). For the initial-cell
            // case (last_axis == 6), use the radial direction.
            // Hit normal: derived from the previous step's exit
            // axis (= current cell's entry axis). For the initial-
            // hit case (last_axis == 6), use radial (face normal).
            // last_axis is 0/1/2 = u/v/r axis; sign comes from r_w
            // sign tracked via `last_axis_sign` if needed (for now,
            // approximate with face axes).
            var hit_normal: vec3<f32>;
            switch last_axis {
                case 0u: { hit_normal =  u_axis; }
                case 1u: { hit_normal =  v_axis; }
                case 2u: { hit_normal =  n_axis; }
                default: { hit_normal =  n_axis; }
            }
            result.hit = true;
            result.t = t_world;
            result.normal = hit_normal;
            let sun = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun), 0.0);
            let axis_tint = abs(hit_normal.y) + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let shape = bevel_layered(
                un_abs, vn_abs, w.u_lo, w.v_lo, w.size,
                shell, t_world, pixel_density,
            );
            let tint = depth_tint(rn_abs);
            result.color = palette[w.block].rgb * (ambient + diffuse * 0.78) * axis_tint * shape * tint;
            // Neutralize shade_pixel's cube_face_bevel: pick a
            // cell_min/cell_size that maps the hit to (0.5, 0.5, 0.5)
            // in cube_face_bevel's local frame so its smoothstep
            // returns 1.0 (no edge darkening) — sphere geometry has
            // its own bevel via `bevel_layered` in face-norm coords.
            let cs = max(length(camera.forward), 1.0) * 1e3;
            result.cell_min = camera.pos + ray_dir * t_world - vec3<f32>(cs * 0.5);
            result.cell_size = cs;
            return result;
        }

        // EMPTY cell. Per-axis residual DDA with analytical Jacobian.
        //
        // J = face_jacobian_normalized at the cell corner. For unit
        // change in (un, vn, rn), J columns give body-XYZ changes.
        // Cell of size `cell_size` covers face-norm range [w.u_lo,
        // w.u_lo + cell_size] × … = residual ∈ [0, 1) for each axis.
        // Local residual derivative: (J_inv · ray_dir) gives
        // (d_un, d_vn, d_rn)/dt — divide by cell_size to convert to
        // residual rate.
        //
        // Per-axis exit time: residual[k] reaches target (0 or 1).
        //   t_local = (target − residual[k]) / rate_residual[k]
        //          = (target − residual[k]) · cell_size / rate_un_per_t[k]
        //
        // The MULTIPLY by cell_size is exact (single FMUL); we never
        // form the huge `1 / cell_size` ratio.
        let j = face_jacobian_normalized(face, w.u_lo, w.v_lo, w.r_lo, inner_r_local, outer_r_local, body_size);
        let j_inv = mat3_inverse_cols(j);
        let rate_un_per_t = mat3_inv_mul_vec(j_inv, ray_dir);

        // Per-axis t_local. For each k: target = 1 if rate_un > 0
        // else 0; t_local = (target − residual[k]) · cell_size /
        // rate_un[k]. If rate_un[k] is near zero, axis is parallel
        // to plane → infinite t → never picked.
        var t_local_axes = vec3<f32>(1e30, 1e30, 1e30);
        for (var k: u32 = 0u; k < 3u; k = k + 1u) {
            let r_k = rate_un_per_t[k];
            if abs(r_k) < 1e-30 { continue; }
            let tgt = select(0.0, 1.0, r_k > 0.0);
            let t_k = (tgt - residual[k]) * cell_size / r_k;
            t_local_axes[k] = select(1e30, t_k, t_k > 0.0);
        }
        // Skip the axis we just exited via (would re-detect t=0).
        if last_axis < 3u {
            t_local_axes[last_axis] = 1e30;
        }
        var winning: u32 = 0u;
        if t_local_axes[1] < t_local_axes[winning] { winning = 1u; }
        if t_local_axes[2] < t_local_axes[winning] { winning = 2u; }
        let t_local = t_local_axes[winning];
        if t_local >= 1e29 { break; }  // No exit found

        // Update residual: snap winning axis (lossless), advance
        // other axes incrementally.
        // Snap convention: if rate is positive, residual was
        // INCREASING toward 1; on cell exit, NEW cell is the
        // +winning neighbor, ENTERED on its low side → snap to eps.
        // If rate is negative, residual was DECREASING toward 0;
        // exit on -winning, NEW cell entered at high side → 1-eps.
        let r_w = rate_un_per_t[winning];
        var new_residual = residual;
        new_residual[winning] = select(1.0 - 1e-6, 1e-6, r_w > 0.0);
        for (var k: u32 = 0u; k < 3u; k = k + 1u) {
            if k == winning { continue; }
            let r_k = rate_un_per_t[k];
            if abs(r_k) < 1e-30 { continue; }
            let advanced = residual[k] + r_k * t_local / cell_size;
            new_residual[k] = clamp(advanced, 0.0, 1.0 - 1e-6);
        }
        residual = new_residual;
        last_axis = winning;
        t_world = t_world + t_local;

        // Step ratio on winning axis. Cells at ratio_depth = 3^d.
        // u/v overflow: face-seam crossing — reproject onto adjacent
        // face via cubesphere geometry. r overflow: shell exit (+r)
        // or inner-shell core entry (-r) — terminate (core-descent
        // is future work).
        let cells_at_depth = pow_3_u32(ratio_depth);
        var seam_crossed = false;
        if winning == 0u {
            if r_w > 0.0 {
                if ratio_u + 1u >= cells_at_depth {
                    seam_crossed = true;
                } else {
                    ratio_u = ratio_u + 1u;
                }
            } else {
                if ratio_u == 0u {
                    seam_crossed = true;
                } else {
                    ratio_u = ratio_u - 1u;
                }
            }
        } else if winning == 1u {
            if r_w > 0.0 {
                if ratio_v + 1u >= cells_at_depth {
                    seam_crossed = true;
                } else {
                    ratio_v = ratio_v + 1u;
                }
            } else {
                if ratio_v == 0u {
                    seam_crossed = true;
                } else {
                    ratio_v = ratio_v - 1u;
                }
            }
        } else {
            // r-axis: shell boundary, terminate.
            if r_w > 0.0 {
                if ratio_r + 1u >= cells_at_depth { break; }
                ratio_r = ratio_r + 1u;
            } else {
                if ratio_r == 0u { break; }
                ratio_r = ratio_r - 1u;
            }
        }

        if seam_crossed {
            // Face-seam crossing. Compute body-XYZ of exit point at
            // current face's edge, then reproject onto adjacent face.
            // The exit point's face-norm coords on the OLD face:
            //   un_exit on overflow axis: 1.0 - eps (if +) or eps (if -)
            //   on other axes: (ratio[k] + new_residual[k]) / cells
            let inv_cells_state = 1.0 / f32(cells_at_depth);
            var un_exit = (f32(ratio_u) + new_residual.x) * inv_cells_state;
            var vn_exit = (f32(ratio_v) + new_residual.y) * inv_cells_state;
            var rn_exit = (f32(ratio_r) + new_residual.z) * inv_cells_state;
            if winning == 0u {
                un_exit = select(1e-6, 1.0 - 1e-6, r_w > 0.0);
            } else if winning == 1u {
                vn_exit = select(1e-6, 1.0 - 1e-6, r_w > 0.0);
            }
            un_exit = clamp(un_exit, 0.0, 0.9999999);
            vn_exit = clamp(vn_exit, 0.0, 0.9999999);
            rn_exit = clamp(rn_exit, 0.0, 0.9999999);

            // Body-local XYZ of the exit point on the OLD face.
            let body_point = face_space_to_body_point(
                face, un_exit, vn_exit, rn_exit,
                inner_r_local, outer_r_local, body_size,
            );
            // Reproject onto adjacent cube face.
            let new_fp = body_point_to_face_space(
                body_point, inner_r_local, outer_r_local, body_size,
            );
            if new_fp.valid == 0u { break; }
            // Switch to the new face. Reset state to face_root level
            // (depth 0, single cell covers whole face).
            face = new_fp.face;
            n_axis = face_normal(face);
            u_axis = face_u_axis(face);
            v_axis = face_v_axis(face);
            // Re-fetch face_node_idx for the new face.
            fslot = face_slot(face);
            if ((body_occ >> fslot) & 1u) == 0u { break; }
            frank = countOneBits(body_occ & ((1u << fslot) - 1u));
            face_node_idx = tree[body_first + frank * 2u + 1u];
            // Reset state: at face_root, single cell, residual = new
            // face's coords.
            ratio_u = 0u;
            ratio_v = 0u;
            ratio_r = 0u;
            ratio_depth = 0u;
            residual = vec3<f32>(new_fp.un, new_fp.vn, new_fp.rn);
            // Reset last_axis since we've teleported to a different
            // face — entry-plane skip from last face is meaningless.
            last_axis = 6u;
        }
    }

    return result;
}
