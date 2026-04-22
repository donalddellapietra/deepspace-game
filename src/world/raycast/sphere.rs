//! Sphere-shell CPU raycast. Cell-boundary stepping that mirrors the
//! GPU `sphere_in_cell` shader exactly — same ray-sphere entry, same
//! per-cell ray-plane / ray-sphere advance, same face LOD depth
//! formula. The result: CPU and GPU terminate at the **same cell**
//! for the same ray, so the broken block, the highlighted AABB, and
//! the rendered pixel all agree.
//!
//! `cs_raycast` reports the terminal cell's face-space bounds in
//! `HitInfo::sphere_cell`, letting `aabb::hit_aabb_body_local` draw
//! the exact rendered cell without re-walking the tree.

use super::{HitInfo, SphereHitCell, MAX_FACE_DEPTH};
use crate::world::cubesphere::{body_point_to_face_space, FACE_SLOTS};
use crate::world::sdf;
use crate::world::tree::{
    slot_index, Child, NodeId, NodeLibrary, EMPTY_NODE, REPRESENTATIVE_EMPTY,
    UNIFORM_EMPTY, UNIFORM_MIXED,
};

/// Sentinel returned by `walk_face_subtree` when the terminal cell
/// is empty. Using a u16 sentinel (not `0`) because palette index 0
/// is a real block type (`block::STONE`).
const EMPTY_CELL: u16 = REPRESENTATIVE_EMPTY;

/// Face-window restriction for sphere raycast (used when the render
/// frame root lives inside a face subtree). Not consumed by the
/// current simplified architecture but reserved as a plug-in seam.
#[derive(Copy, Clone, Debug)]
pub struct FaceWindow {
    pub face: u32,
    pub u_min: f32,
    pub v_min: f32,
    pub r_min: f32,
    pub size: f32,
}

/// Per-ray LOD inputs. Must match the shader's `face_lod_depth`:
/// `pixel_density = screen_height / (2·tan(fov/2))`,
/// `lod_threshold = LOD_PIXEL_THRESHOLD` (1.0 by default).
#[derive(Copy, Clone, Debug)]
pub struct LodParams {
    pub pixel_density: f32,
    pub lod_threshold: f32,
}

impl LodParams {
    /// Equivalent of "always go to max depth" — used when the caller
    /// wants edit-target depth rather than screen-LOD depth.
    pub fn fixed_max() -> Self {
        Self { pixel_density: 1e30, lod_threshold: 1.0 }
    }
}

/// Per-ray walker depth (matches shader `face_lod_depth`).
/// `cell_size = shell * (1/3)^(d-1)`; pick the largest `d` whose
/// projected cell is ≥ `lod_threshold` pixels at ray distance `t`.
fn face_lod_depth(t: f32, shell: f32, lod: LodParams) -> u32 {
    let safe_t = t.max(1e-6);
    let ratio = shell * lod.pixel_density / (safe_t * lod.lod_threshold.max(1e-6));
    if ratio <= 1.0 {
        return 1;
    }
    let log3_ratio = ratio.ln() / 3.0_f32.ln();
    (1.0 + log3_ratio).clamp(1.0, MAX_FACE_DEPTH as f32) as u32
}

/// Walker result — direct mirror of the shader's `FaceWalkResult`
/// plus the path so the CPU hit has a tree-level identity for edit
/// propagation.
///
/// `u_lo / v_lo / r_lo / size` are the f32 cell bounds used internally
/// by the body-march cell-stepper (`ray_plane_t` etc.). `ratio_u /
/// ratio_v / ratio_r / ratio_depth` are the exact rational bounds
/// tracked by integer accumulation (`ratio = parent*3 + slot`); the
/// cell is `[ratio_u / 3^ratio_depth, (ratio_u+1) / 3^ratio_depth)`
/// with zero precision loss at any depth. The hit report passes the
/// ratios through to `SphereHitCell` for f64 AABB derivation.
struct FaceWalk {
    block: u16,
    depth: u32,
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    size: f32,
    ratio_u: i64,
    ratio_v: i64,
    ratio_r: i64,
    ratio_depth: u8,
    path: Vec<(NodeId, usize)>,
}

/// Descend a face subtree from its root along `(un, vn, rn)` to the
/// terminal cell. Mirrors the shader's `walk_face_subtree` exactly:
/// depth-limited descent with LOD-terminal fall-back to the
/// subtree's representative block or the empty sentinel.
///
/// Empty-cell path is padded out to `max_depth` with `EMPTY_NODE`
/// entries so `propagate_edit` can place a block of consistent size
/// even when the tree collapsed the empty region early.
fn walk_face_subtree(
    library: &NodeLibrary,
    face_root_id: NodeId,
    un_in: f32, vn_in: f32, rn_in: f32,
    max_depth: u32,
) -> FaceWalk {
    // Error-bounded walker (approach #1).
    //
    // The previous implementation iterated `un = un * 3 - us` at each
    // descent level. That recurrence AMPLIFIES any f32 error in `un`
    // by a factor of 3 per level — a ~1e-7 initial precision becomes
    // 3^n · 1e-7 by depth n. Past depth ~9 the error exceeds a cell
    // width and pixels near cell boundaries snap to random neighbors
    // (the "concentric rings" artifact the user sees).
    //
    // Here we keep `un_abs`/`vn_abs`/`rn_abs` IMMUTABLE throughout the
    // walk and derive each level's slot from the accumulating cell
    // origin: `us = floor((un_abs − u_lo) / child_size)`. Error stays
    // bounded at ~f32 ULP absolute, not amplified. The precision wall
    // shifts from ~depth 9 to ~depth 14 (where cell_size approaches
    // f32 ULP of the O(1) absolute coords and the subtraction loses
    // significance). When it fails past that, it fails by off-by-one
    // slot — a visually smooth blur, not ring chaos.
    let limit = max_depth.min(MAX_FACE_DEPTH).max(1);
    let un_abs = un_in.clamp(0.0, 0.9999999);
    let vn_abs = vn_in.clamp(0.0, 0.9999999);
    let rn_abs = rn_in.clamp(0.0, 0.9999999);

    // Compute a single level's slot from absolute coords + current
    // cell origin. Shared between the main descent and the empty-cell
    // padding loop so both use the precision-safe formulation.
    #[inline]
    fn slot_at_level(abs_c: f32, lo: f32, child_size: f32) -> usize {
        // (abs_c − lo) is in [0, size); divide by size/3 → [0, 3).
        // Clamp guards against f32 rounding that could push the
        // quotient to a tiny negative (lost slot) or exactly 3.0
        // (overflow).
        let raw = ((abs_c - lo) / child_size).floor();
        raw.clamp(0.0, 2.0) as usize
    }

    let mut node_id = face_root_id;
    let mut u_lo: f32 = 0.0;
    let mut v_lo: f32 = 0.0;
    let mut r_lo: f32 = 0.0;
    let mut size: f32 = 1.0;
    // Integer-ratio mirror of `(u_lo, v_lo, r_lo)`. Invariant:
    // `u_lo == ratio_u as f32 / 3_f32.powi(ratio_depth)` up to f32
    // rounding. Precision-free at any depth. i64 supports ratio_depth
    // up to ~40.
    let mut ratio_u: i64 = 0;
    let mut ratio_v: i64 = 0;
    let mut ratio_r: i64 = 0;
    let mut ratio_depth: u8 = 0;
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(limit as usize);

    for d in 1u32..=limit {
        let Some(node) = library.get(node_id) else {
            return FaceWalk {
                block: EMPTY_CELL, depth: d.saturating_sub(1),
                u_lo, v_lo, r_lo, size,
                ratio_u, ratio_v, ratio_r, ratio_depth,
                path,
            };
        };
        let child_size = size / 3.0;
        let us = slot_at_level(un_abs, u_lo, child_size);
        let vs = slot_at_level(vn_abs, v_lo, child_size);
        let rs = slot_at_level(rn_abs, r_lo, child_size);
        let slot = slot_index(us, vs, rs);
        let child_ratio_u = ratio_u * 3 + us as i64;
        let child_ratio_v = ratio_v * 3 + vs as i64;
        let child_ratio_r = ratio_r * 3 + rs as i64;
        let child_ratio_depth = ratio_depth + 1;
        // Ratio-derived cell corner: one multiplication, ~0.5 ULP
        // error per step. The alternative `u_lo + us * child_size`
        // compounds ~1 ULP per level, so by m=10 the accumulated
        // drift is ~10× worse and the 6 cell-wall plane normals
        // built from `ea_to_cube(u_lo*2-1)` lose enough precision
        // that adjacent walls collapse — the ray steps through
        // solid content and the cell renders hollow. Integer
        // ratios stay exact; a single f32 multiply recovers the
        // corner with minimum precision loss.
        let child_u_lo = child_ratio_u as f32 * child_size;
        let child_v_lo = child_ratio_v as f32 * child_size;
        let child_r_lo = child_ratio_r as f32 * child_size;
        path.push((node_id, slot));

        match node.children[slot] {
            Child::Empty => {
                // Pad path to `limit` using the same absolute-coord
                // slot calc. Each padding level narrows the cell
                // origin by one more step so `propagate_edit` lands
                // at a uniform depth.
                let mut pad_u_lo = child_u_lo;
                let mut pad_v_lo = child_v_lo;
                let mut pad_r_lo = child_r_lo;
                let mut pad_size = child_size;
                for _ in d..limit {
                    let pad_child_size = pad_size / 3.0;
                    let us2 = slot_at_level(un_abs, pad_u_lo, pad_child_size);
                    let vs2 = slot_at_level(vn_abs, pad_v_lo, pad_child_size);
                    let rs2 = slot_at_level(rn_abs, pad_r_lo, pad_child_size);
                    path.push((EMPTY_NODE, slot_index(us2, vs2, rs2)));
                    pad_u_lo += us2 as f32 * pad_child_size;
                    pad_v_lo += vs2 as f32 * pad_child_size;
                    pad_r_lo += rs2 as f32 * pad_child_size;
                    pad_size = pad_child_size;
                }
                return FaceWalk {
                    block: EMPTY_CELL, depth: d,
                    u_lo: child_u_lo, v_lo: child_v_lo,
                    r_lo: child_r_lo, size: child_size,
                    ratio_u: child_ratio_u, ratio_v: child_ratio_v,
                    ratio_r: child_ratio_r, ratio_depth: child_ratio_depth,
                    path,
                };
            }
            Child::Block(b) => {
                return FaceWalk {
                    block: b, depth: d,
                    u_lo: child_u_lo, v_lo: child_v_lo,
                    r_lo: child_r_lo, size: child_size,
                    ratio_u: child_ratio_u, ratio_v: child_ratio_v,
                    ratio_r: child_ratio_r, ratio_depth: child_ratio_depth,
                    path,
                };
            }
            Child::EntityRef(_) => {
                return FaceWalk {
                    block: EMPTY_CELL, depth: d,
                    u_lo: child_u_lo, v_lo: child_v_lo,
                    r_lo: child_r_lo, size: child_size,
                    ratio_u: child_ratio_u, ratio_v: child_ratio_v,
                    ratio_r: child_ratio_r, ratio_depth: child_ratio_depth,
                    path,
                };
            }
            Child::Node(nid) => {
                if d == limit {
                    let Some(child) = library.get(nid) else {
                        return FaceWalk {
                            block: EMPTY_CELL, depth: d,
                            u_lo: child_u_lo, v_lo: child_v_lo,
                            r_lo: child_r_lo, size: child_size,
                            ratio_u: child_ratio_u, ratio_v: child_ratio_v,
                            ratio_r: child_ratio_r, ratio_depth: child_ratio_depth,
                            path,
                        };
                    };
                    let bt = match child.uniform_type {
                        UNIFORM_EMPTY => EMPTY_CELL,
                        UNIFORM_MIXED => {
                            let rep = child.representative_block;
                            if rep == REPRESENTATIVE_EMPTY { EMPTY_CELL } else { rep }
                        }
                        b => b,
                    };
                    return FaceWalk {
                        block: bt, depth: d,
                        u_lo: child_u_lo, v_lo: child_v_lo,
                        r_lo: child_r_lo, size: child_size,
                        ratio_u: child_ratio_u, ratio_v: child_ratio_v,
                        ratio_r: child_ratio_r, ratio_depth: child_ratio_depth,
                        path,
                    };
                }
                node_id = nid;
                u_lo = child_u_lo;
                v_lo = child_v_lo;
                r_lo = child_r_lo;
                size = child_size;
                ratio_u = child_ratio_u;
                ratio_v = child_ratio_v;
                ratio_r = child_ratio_r;
                ratio_depth = child_ratio_depth;
                // NOTE: `un_abs`/`vn_abs`/`rn_abs` are NOT updated —
                // they stay as the immutable ray-sample reference.
                // That's the whole point of this reformulation.
            }
        }
    }
    FaceWalk {
        block: EMPTY_CELL, depth: limit,
        u_lo, v_lo, r_lo, size,
        ratio_u, ratio_v, ratio_r, ratio_depth,
        path,
    }
}

// ──────────────────────────────────────────────── face geometry

/// `(face, u, v, r)` → unit direction from sphere center. Must match
/// `face_uv_to_dir` on the shader side. Used only for the ea-to-cube
/// inverse inside the cell-boundary stepper.
#[inline]
fn ea_to_cube(c: f32) -> f32 {
    (c * std::f32::consts::FRAC_PI_4).tan()
}

fn face_normal(face: u32) -> [f32; 3] {
    match face {
        0 => [ 1.0, 0.0, 0.0],
        1 => [-1.0, 0.0, 0.0],
        2 => [ 0.0, 1.0, 0.0],
        3 => [ 0.0,-1.0, 0.0],
        4 => [ 0.0, 0.0, 1.0],
        _ => [ 0.0, 0.0,-1.0],
    }
}

fn face_u_axis(face: u32) -> [f32; 3] {
    match face {
        0 => [ 0.0, 0.0,-1.0],
        1 => [ 0.0, 0.0, 1.0],
        2 => [ 1.0, 0.0, 0.0],
        3 => [ 1.0, 0.0, 0.0],
        4 => [ 1.0, 0.0, 0.0],
        _ => [-1.0, 0.0, 0.0],
    }
}

fn face_v_axis(face: u32) -> [f32; 3] {
    match face {
        0 => [ 0.0, 1.0, 0.0],
        1 => [ 0.0, 1.0, 0.0],
        2 => [ 0.0, 0.0,-1.0],
        3 => [ 0.0, 0.0, 1.0],
        4 => [ 0.0, 1.0, 0.0],
        _ => [ 0.0, 1.0, 0.0],
    }
}

// ────────────────────────────────────────── ray-primitive helpers

/// Ray → plane t: plane passes through origin (all cell planes in
/// cubemap geometry are center-rooted). Returns a large sentinel for
/// near-parallel rays so the t_next min() safely ignores them.
#[inline]
fn ray_plane_t(origin: [f32; 3], dir: [f32; 3], normal: [f32; 3]) -> f32 {
    let denom = sdf::dot(dir, normal);
    if denom.abs() < 1e-12 { return f32::INFINITY; }
    -sdf::dot(origin, normal) / denom
}

/// Ray → sphere of given radius centered at origin, returning the
/// smallest t > `after` where the ray is on the sphere. Numerical-
/// Recipes stable form; identical to the shader's
/// `ray_sphere_after`.
#[inline]
fn ray_sphere_after(origin: [f32; 3], dir: [f32; 3], radius: f32, after: f32) -> f32 {
    let b = sdf::dot(origin, dir);
    let c = sdf::dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return f32::INFINITY; }
    let sq = disc.sqrt();
    let s = if b >= 0.0 { 1.0 } else { -1.0 };
    let q = -b - s * sq;
    if q.abs() < 1e-30 { return f32::INFINITY; }
    let t0 = q;
    let t1 = c / q;
    let t_lo = t0.min(t1);
    let t_hi = t0.max(t1);
    if t_lo > after { t_lo }
    else if t_hi > after { t_hi }
    else { f32::INFINITY }
}

// ─────────────────────────────────────── unified sphere raycast

/// CPU mirror of the shader's `sphere_in_cell`. Cell-boundary DDA
/// through the sphere shell: at each iteration we find the face +
/// (un, vn, rn) at the current ray position, walk the face subtree
/// to `min(max_face_depth, face_lod_depth(t, shell, lod))` levels,
/// and either report the hit or step to the cell's next boundary.
///
/// `max_face_depth` is the anchor-depth cap (fed by `cs_edit_depth`
/// = `edit_depth` = anchor depth). This is the Cartesian-analog of
/// `max_depth`: the walker resolves to ANCHOR-sized cells, not
/// screen-Nyquist cells. That's why the user's interaction range
/// ("12 anchor cells") always produces on-screen cells much larger
/// than 1 pixel — at 12 anchor-cells distance, the projected
/// anchor cell is ≈ `pixel_density / 12` ≈ 20 px. Sub-pixel edits
/// are geometrically impossible as long as the camera stays within
/// `interaction_range_in_frame`.
///
/// `lod` is a secondary LOD-Nyquist floor — the walker won't
/// descend past the screen-size threshold even if the anchor cap
/// allows it. This mirrors `march_cartesian`'s dual `at_max ||
/// at_lod` termination. Pass `LodParams::fixed_max()` to disable.
#[allow(clippy::too_many_arguments)]
pub(super) fn cs_raycast(
    library: &NodeLibrary,
    body_id: NodeId,
    body_origin: [f32; 3],
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    ancestor_path: &[(NodeId, usize)],
    max_face_depth: u32,
    lod: LodParams,
    _window: Option<FaceWindow>,
) -> Option<HitInfo> {
    let ray_dir = sdf::normalize(ray_dir);
    let cs_center = [
        body_origin[0] + body_size * 0.5,
        body_origin[1] + body_size * 0.5,
        body_origin[2] + body_size * 0.5,
    ];
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;
    if shell <= 0.0 { return None; }

    // Ray–outer-sphere entry. Ray coords are re-centered on the
    // sphere center so the subsequent ray-plane / ray-sphere math
    // can use zero-origin planes (cell planes pass through center).
    let oc = sdf::sub(ray_origin, cs_center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return None; }

    let eps_init = (shell * 1e-5).max(1e-7);
    let mut t = t_enter + eps_init;
    let mut last_side: u32 = 6;
    let mut prev_place_path: Option<Vec<(NodeId, usize)>> = None;
    let body_node = library.get(body_id)?;
    // body_path_len counted inside SphereHitCell is the number of
    // ancestor entries prepended to the returned path. The full hit
    // path looks like `ancestor_path + (body_id, face_slot) + face_descent`,
    // so from the caller's world-root POV the body lives at index
    // `ancestor_path.len()`.
    let body_path_len = ancestor_path.len();

    for _ in 0..4096usize {
        if t >= t_exit { break; }
        let local = sdf::add(oc, sdf::scale(ray_dir, t));
        let r = sdf::length(local);
        if r >= cs_outer || r < cs_inner { break; }

        let n = sdf::scale(local, 1.0 / r);
        let p_body = [
            ray_origin[0] + ray_dir[0] * t - body_origin[0],
            ray_origin[1] + ray_dir[1] * t - body_origin[1],
            ray_origin[2] + ray_dir[2] * t - body_origin[2],
        ];
        let fp = body_point_to_face_space(p_body, inner_r_local, outer_r_local, body_size)?;
        let face = fp.face as u32;
        let face_slot = FACE_SLOTS[face as usize];

        // Resolve the face subtree root. A missing face (body with
        // a hole) is unusual; skip to next cell via ray-sphere.
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => break,
        };

        // Anchor cap first, LOD-Nyquist floor second. Matches
        // Cartesian's `at_max || at_lod` — whichever is reached
        // earlier wins.
        //
        // `max_face_depth` is the anchor depth measured from the
        // WORLD root, but the walker counts descents inside the
        // face subtree alone — the path leading here already ate
        // two levels (root → body, body → face_root), plus the
        // Cartesian descents in `ancestor_path` before the body.
        // Subtract those so a user at world-anchor-depth N gets a
        // cell at face-subtree depth `N − ancestor_prefix − 2`.
        let ancestor_prefix = ancestor_path.len() as u32;
        let body_and_face = 2u32;
        let walker_cap = max_face_depth
            .saturating_sub(ancestor_prefix + body_and_face)
            .max(1);
        let walk_depth = face_lod_depth(t, shell, lod).min(walker_cap);
        let w = walk_face_subtree(library, face_root_id, fp.un, fp.vn, fp.rn, walk_depth);

        if w.block != EMPTY_CELL {
            let mut full_path: Vec<(NodeId, usize)> =
                Vec::with_capacity(ancestor_path.len() + 1 + w.path.len());
            full_path.extend_from_slice(ancestor_path);
            full_path.push((body_id, face_slot));
            full_path.extend(w.path.iter().copied());
            return Some(HitInfo {
                path: full_path,
                // `face` on HitInfo is XYZ-axis face for Cartesian
                // hits; sphere hits carry no xyz-axis semantic, so
                // we reuse slot 4 (+Z) as the sphere sentinel. The
                // place_path and sphere_cell carry the real info.
                face: 4,
                t,
                place_path: prev_place_path,
                sphere_cell: Some(SphereHitCell {
                    face,
                    u_lo: w.u_lo,
                    v_lo: w.v_lo,
                    r_lo: w.r_lo,
                    size: w.size,
                    inner_r: inner_r_local,
                    outer_r: outer_r_local,
                    body_path_len,
                    ratio_u: w.ratio_u,
                    ratio_v: w.ratio_v,
                    ratio_r: w.ratio_r,
                    ratio_depth: w.ratio_depth,
                    sub_c_body: [0.0; 3],
                    sub_j_cols: [[0.0; 3]; 3],
                    sub_local_lo: [0.0; 3],
                    sub_local_size: 0.0,
                }),
            });
        }

        // Empty cell: record full path as the placement target so
        // `place_block` can drop a block of consistent size even
        // when the tree flattened the air region.
        let mut empty_full: Vec<(NodeId, usize)> =
            Vec::with_capacity(ancestor_path.len() + 1 + w.path.len());
        empty_full.extend_from_slice(ancestor_path);
        empty_full.push((body_id, face_slot));
        empty_full.extend(w.path.iter().copied());
        prev_place_path = Some(empty_full);

        // Step to the cell's next boundary. Two paths depending on
        // cell size:
        //
        // SHALLOW (`w.size >= LOCAL_FRAME_STEP_THRESHOLD`): 4 UV
        //     planes + 2 radial spheres, intersected with the ray in
        //     body-CENTERED coords. Curvature-accurate — the cube-
        //     sphere cell's exact boundaries.
        //
        // DEEP (`w.size < LOCAL_FRAME_STEP_THRESHOLD`): the 6-plane
        //     pairs become near-degenerate in f32 (adjacent normals
        //     differ by only `O(w.size)` against the face normal,
        //     below ULP of the individual components). Ray-plane `t`
        //     values are indistinguishable → winning face is
        //     arbitrary → adjacent pixels take divergent paths →
        //     rendered blocks appear HOLLOW because rays skip the
        //     cell interior. Swap in a local-frame mini-sub-frame:
        //     axis-aligned `(boundary − pos) / dir` in `[0, 3)³`
        //     local coords is precision-stable at any depth.
        let (t_next, winning) = if w.size < LOCAL_FRAME_STEP_THRESHOLD {
            local_frame_cell_exit(
                face,
                w.u_lo, w.v_lo, w.r_lo, w.size,
                inner_r_local, outer_r_local, body_size,
                oc, ray_dir, t,
            )
        } else {
            let n_axis = face_normal(face);
            let u_axis = face_u_axis(face);
            let v_axis = face_v_axis(face);
            let u_lo_ea = w.u_lo * 2.0 - 1.0;
            let u_hi_ea = (w.u_lo + w.size) * 2.0 - 1.0;
            let v_lo_ea = w.v_lo * 2.0 - 1.0;
            let v_hi_ea = (w.v_lo + w.size) * 2.0 - 1.0;
            let n_u_lo = sub_scaled(u_axis, n_axis, ea_to_cube(u_lo_ea));
            let n_u_hi = sub_scaled(u_axis, n_axis, ea_to_cube(u_hi_ea));
            let n_v_lo = sub_scaled(v_axis, n_axis, ea_to_cube(v_lo_ea));
            let n_v_hi = sub_scaled(v_axis, n_axis, ea_to_cube(v_hi_ea));
            let r_lo = cs_inner + w.r_lo * shell;
            let r_hi = cs_inner + (w.r_lo + w.size) * shell;

            let t_u_lo = ray_plane_t(oc, ray_dir, n_u_lo);
            let t_u_hi = ray_plane_t(oc, ray_dir, n_u_hi);
            let t_v_lo = ray_plane_t(oc, ray_dir, n_v_lo);
            let t_v_hi = ray_plane_t(oc, ray_dir, n_v_hi);
            let t_r_lo = ray_sphere_after(oc, ray_dir, r_lo, t);
            let t_r_hi = ray_sphere_after(oc, ray_dir, r_hi, t);

            pick_min_after(
                t,
                &[(t_u_lo, 0), (t_u_hi, 1), (t_v_lo, 2), (t_v_hi, 3),
                  (t_r_lo, 4), (t_r_hi, 5)],
            )
        };
        if t_next >= t_exit { break; }
        last_side = winning;
        let _ = last_side; // currently consumed only by shader
        // Nudge past the exit boundary. Scale to the **LOD-depth**
        // cell size (1 / 3^walk_depth), NOT the walker's terminal
        // `w.size`.
        //
        // The old `shell * w.size * 1e-3` scaled to whatever depth
        // THIS pixel's walker happened to terminate at. With the
        // packer's tag=1 flattening of uniform subtrees, adjacent
        // pixels around an edit can terminate at wildly different
        // depths — tag=1 sibling at face-depth 1 (w.size=1/3), edit
        // chain at face-depth 10 (w.size=1.7e-5). That's a 20000×
        // disparity in `cell_eps` across a single screen pixel
        // boundary. When a ray crosses from a shallow-sibling region
        // into the deep-edit region, the big nudge pushes `t` PAST
        // the deep cells entirely — the ray skips solid cells and
        // renders streaks along the face-grid boundary between
        // tag=1 flattened and tag=2 rebuilt siblings.
        //
        // LOD-depth cell size is a property of pixel distance, not
        // tree structure — it's the SAME for every adjacent pixel
        // at a given `t`. Nudging by that scale is safely smaller
        // than any cell the walker could return at this pixel, so
        // the ray never overshoots. Still guarded by the `t_ulp * 4`
        // floor for f32 boundary-drift safety.
        let lod_cell_size = (1.0_f32 / 3.0_f32).powi(walk_depth as i32);
        let t_ulp = (t.abs() * 1.2e-7).max(1e-30);
        let cell_eps = (shell * lod_cell_size * 1e-3).max(t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    None
}

#[inline]
fn sub_scaled(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] - b[0] * s, a[1] - b[1] * s, a[2] - b[2] * s]
}

/// Cell size in FACE-NORMALIZED units below which the 6-plane body-
/// coord cell-stepper's plane-normal pairs (`n_u_lo` vs `n_u_hi`,
/// `n_v_lo` vs `n_v_hi`) become near-degenerate in f32 — the pair
/// separation is `O(w.size)` against the face-normal, and once that
/// is within a few ULPs of the individual normal components the
/// ray-plane `t` for adjacent walls becomes indistinguishable.
/// Adjacent pixels then pick arbitrary winning walls and the ray
/// "skips" solid cells → HOLLOW BLOCKS at deep depth.
///
/// Below this threshold we swap the 6-plane stepper for a local-
/// frame axis-aligned DDA: build a mini-sub-frame Jacobian at the
/// cell corner, transform the ray into `[0, 3)³` local coords, and
/// find the exit wall via trivial `(boundary − pos) / dir` — which
/// is precision-stable at any depth because all operands are O(1).
const LOCAL_FRAME_STEP_THRESHOLD: f32 = 5e-5;

/// Local-frame cell-exit: returns `(t_next, winning_face)` for the
/// ray exiting the walker cell at `(u_lo, v_lo, r_lo, size)` in
/// face-normalized coords. Drops into the same axis-aligned DDA
/// primitive as `sphere_in_sub_frame` — precision-safe where the
/// 6-plane body-coord math fails.
///
/// `oc` is the ray origin in body-CENTERED coords (body center at
/// origin). `ray_dir` is unit. `t_body` is the current DDA t. The
/// returned `t_next` is in the same body-distance units — the outer
/// loop advances with `t = t_next + cell_eps` unchanged.
///
/// `winning_face` uses the same encoding as the 6-plane stepper:
/// 0=u_lo, 1=u_hi, 2=v_lo, 3=v_hi, 4=r_lo, 5=r_hi.
#[allow(clippy::too_many_arguments)]
fn local_frame_cell_exit(
    face: u32,
    u_lo: f32, v_lo: f32, r_lo: f32, size: f32,
    inner_r_local: f32, outer_r_local: f32, body_size: f32,
    oc: [f32; 3], ray_dir: [f32; 3], t_body: f32,
) -> (f32, u32) {
    use crate::world::cubesphere::{face_frame_jacobian, mat3_inv, mat3_mul_vec, Face};

    // Build the cell's Jacobian at its corner. `face_frame_jacobian`
    // returns `c_body` in body-LOCAL coords (origin at body's
    // `[0, 0, 0]`); `oc` is body-CENTERED, so the offset between
    // ray body-pos and `c_body` is `(body_center + oc) - c_body`.
    let face_enum = Face::from_index(face as u8);
    let (c_body_local, j) = face_frame_jacobian(
        face_enum, u_lo, v_lo, r_lo, size,
        inner_r_local, outer_r_local, body_size,
    );
    let body_center = [body_size * 0.5; 3];
    let c_body_centered = [
        c_body_local[0] - body_center[0],
        c_body_local[1] - body_center[1],
        c_body_local[2] - body_center[2],
    ];
    let j_inv = mat3_inv(&j);

    // Ray at current body t, in body-centered coords.
    let body_pt_at_t = [
        oc[0] + ray_dir[0] * t_body,
        oc[1] + ray_dir[1] * t_body,
        oc[2] + ray_dir[2] * t_body,
    ];
    // Offset from cell corner — this is the subtraction that loses
    // absolute precision but stays stable ENOUGH at a single cell
    // entry (one-off precision cost, not compounding).
    let offset = [
        body_pt_at_t[0] - c_body_centered[0],
        body_pt_at_t[1] - c_body_centered[1],
        body_pt_at_t[2] - c_body_centered[2],
    ];
    let ro_local = mat3_mul_vec(&j_inv, offset);
    let rd_local = mat3_mul_vec(&j_inv, ray_dir);

    // Axis-exit in local `[0, 3)³`. Pick smallest positive t_local.
    // `rd_body · t_local = body displacement from t_body`, so
    // `t_next_body = t_body + t_local`.
    let mut t_local_min = f32::INFINITY;
    let mut winning: u32 = 6;
    for axis in 0..3 {
        let p = ro_local[axis];
        let d = rd_local[axis];
        if d.abs() < 1e-30 { continue; }
        let (boundary, face_id_lo, face_id_hi) = match axis {
            0 => (if d > 0.0 { 3.0 } else { 0.0 }, 0u32, 1u32),
            1 => (if d > 0.0 { 3.0 } else { 0.0 }, 2u32, 3u32),
            _ => (if d > 0.0 { 3.0 } else { 0.0 }, 4u32, 5u32),
        };
        let t_cand = (boundary - p) / d;
        if t_cand > 0.0 && t_cand < t_local_min {
            t_local_min = t_cand;
            winning = if d > 0.0 { face_id_hi } else { face_id_lo };
        }
    }
    if !t_local_min.is_finite() {
        return (f32::INFINITY, 6);
    }
    (t_body + t_local_min, winning)
}

#[inline]
fn pick_min_after(after: f32, candidates: &[(f32, u32)]) -> (f32, u32) {
    let mut best_t = f32::INFINITY;
    let mut best_face = 6u32;
    for &(cand, face) in candidates {
        if cand > after && cand < best_t {
            best_t = cand;
            best_face = face;
        }
    }
    (best_t, best_face)
}
