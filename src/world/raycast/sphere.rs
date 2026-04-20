//! CPU sphere-body raycast. Byte-for-byte mirror of the shader's
//! `march_sphere_body` (in `assets/shaders/sphere.wgsl`). One
//! function, one walker, one cell convention — and one per-ray LOD
//! cap, so the CPU lands on the SAME cell the shader renders.
//!
//! Before the rewrite, the CPU walked face subtrees to full
//! `cs_edit_depth` (anchor_depth) while the shader stopped at its
//! Nyquist-LOD cap. The result: the CPU's hit cell was a tiny
//! sub-cell buried inside the chunky LOD-terminal cell the shader
//! drew, so the highlight AABB and the break target pointed at
//! different voxels than the one under the crosshair.
//!
//! This rewrite drives the walker with the same per-ray LOD depth
//! the shader uses and steps cell-to-cell via the same u/v/r
//! boundary crossings the shader picks. The returned `HitInfo.path`
//! ends exactly where the shader's face-subtree walk terminated.

use super::{HitInfo, MAX_FACE_DEPTH};
use crate::world::cubesphere::{cube_to_ea, ea_to_cube, pick_face, Face, FACE_SLOTS};
use crate::world::sdf;
use crate::world::tree::{
    slot_index, Child, NodeId, NodeLibrary, EMPTY_NODE, REPRESENTATIVE_EMPTY, UNIFORM_EMPTY,
    UNIFORM_MIXED,
};

// Shader-side LOD constant. Must match `LOD_PIXEL_THRESHOLD` in
// `bindings.wgsl` (default 1.0 Nyquist floor). CPU raycast uses
// this to pick the same face-subtree terminal depth the shader
// will draw — without it the cursor highlight lands on a finer
// cell than the rendered voxel.
const LOD_PIXEL_THRESHOLD: f32 = 1.0;

/// Effective pixel density in units of `screen_pixels / render_frame_unit`
/// at ray distance 1. Callers derive it from `screen_height / (2 *
/// tan(fov/2))` so that `cell_size * pixel_density_at_1 / ray_dist`
/// gives the on-screen pixel size of a cell. Same formula the shader
/// uses.
fn lod_depth_cap(ray_dist: f32, shell_size_rf: f32, pixel_density: f32) -> u32 {
    let safe_dist = ray_dist.max(1e-6);
    let ratio = shell_size_rf * pixel_density
        / (safe_dist * LOD_PIXEL_THRESHOLD.max(1e-6));
    if ratio <= 1.0 { return 1; }
    // log3(x) = log2(x) / log2(3)
    let log3_ratio = ratio.ln() * (1.0 / 1.0986123); // 1/ln(3)
    let d_f = 1.0 + log3_ratio;
    d_f.clamp(1.0, MAX_FACE_DEPTH as f32) as u32
}

/// Result of walking the face subtree at a point: (block_id,
/// term_depth, path, Kahan-compensated cell bounds). Mirror of the
/// shader's `FaceWalkResult`.
struct FaceWalk {
    block: u16,
    depth: u32,
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    size: f32,
    path: Vec<(NodeId, usize)>,
}

/// CPU sphere raycast inside one `CubedSphereBody` cell. Exact
/// mirror of the shader's `march_sphere_body` — walks the ray
/// cell-by-cell through the face subtrees, using the same per-ray
/// LOD cap, same face-axis plane normals, same Kahan-compensated
/// walker, and returns the exact terminal cell the shader would
/// draw.
///
/// `body_origin`, `body_size` are in render-frame-local coords —
/// never body-absolute. `pixel_density` is the rendered framebuffer's
/// pixel density at ray distance 1 (`screen_height / (2 tan(fov/2))`).
/// `ancestor_path` is the slot chain from the outer frame root down
/// to the body cell; the returned `HitInfo.path` extends it with
/// `(body_id, face_slot)` followed by the walker's terminal path.
pub fn cs_raycast_in_body(
    library: &NodeLibrary,
    body_id: NodeId,
    body_origin: [f32; 3],
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    ancestor_path: &[(NodeId, usize)],
    pixel_density: f32,
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

    // Ray-outer-shell entry. Standard quadratic — oc is bounded
    // by body_size so precision is fine.
    let oc = sdf::sub(ray_origin, cs_center);
    let b = sdf::dot(oc, ray_dir);
    let c_outer = sdf::dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return None; }

    let eps_init = (shell * 1e-5).max(1e-7);
    let mut t = t_enter + eps_init;
    let mut last_face_axis: u32 = 6;
    let mut prev_empty_path: Option<Vec<(NodeId, usize)>> = None;

    for _ in 0..4096u32 {
        if t >= t_exit { break; }

        // Point on ray. oc + ray_dir * t; local relative to body center.
        let local = [
            oc[0] + ray_dir[0] * t,
            oc[1] + ray_dir[1] * t,
            oc[2] + ray_dir[2] * t,
        ];
        let r = sdf::length(local);
        if r >= cs_outer || r < cs_inner { break; }

        let n = [local[0] / r, local[1] / r, local[2] / r];
        let face = pick_face(n);
        let n_axis = face.normal();
        let (u_axis, v_axis) = face.tangents();

        let axis_dot = sdf::dot(n, n_axis);
        let cube_u = sdf::dot(n, u_axis) / axis_dot;
        let cube_v = sdf::dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);

        let un = ((u_ea + 1.0) * 0.5).clamp(0.0, 0.9999999);
        let vn = ((v_ea + 1.0) * 0.5).clamp(0.0, 0.9999999);
        let rn = ((r - cs_inner) / shell).clamp(0.0, 0.9999999);

        // Per-ray LOD cap — same formula the shader uses so we stop
        // at the same depth.
        let walk_depth = lod_depth_cap(t, shell, pixel_density);

        let face_slot = FACE_SLOTS[face as usize];
        let body_node = library.get(body_id)?;
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => {
                // Face subtree missing (degenerate body). Advance by a
                // conservative cell-sized step.
                t += (shell * 0.33).max(eps_init * 4.0);
                continue;
            }
        };

        let walk = walk_face_subtree(library, face_root_id, un, vn, rn, walk_depth);

        if walk.block != 0 {
            // Block hit. Build the path and return.
            let mut full = ancestor_path.to_vec();
            full.push((body_id, face_slot));
            full.extend(walk.path);
            return Some(HitInfo {
                path: full,
                face: match last_face_axis {
                    // 0/1 u, 2/3 v, 4/5 r; 6 = initial (face normal).
                    0 | 1 | 2 | 3 => 4, // sphere placements use place_path
                    _ => 4,
                },
                t,
                place_path: prev_empty_path,
            });
        }

        // Empty cell — remember its full path as a placement target.
        let mut empty_full = ancestor_path.to_vec();
        empty_full.push((body_id, face_slot));
        empty_full.extend(walk.path.clone());
        prev_empty_path = Some(empty_full);

        // Step to next cell boundary. Candidates: the four u/v
        // radial planes and two r spherical shells bounding the
        // walker cell. Same geometry the shader uses.
        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = sdf::sub(u_axis, sdf::scale(n_axis, ea_to_cube(u_lo_ea)));
        let n_u_hi = sdf::sub(u_axis, sdf::scale(n_axis, ea_to_cube(u_hi_ea)));

        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = sdf::sub(v_axis, sdf::scale(n_axis, ea_to_cube(v_lo_ea)));
        let n_v_hi = sdf::sub(v_axis, sdf::scale(n_axis, ea_to_cube(v_hi_ea)));

        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        let mut t_next = t_exit + 1.0;
        let mut winning_axis: u32 = 6;

        for (normal, axis_id) in [
            (n_u_lo, 0u32),
            (n_u_hi, 1u32),
            (n_v_lo, 2u32),
            (n_v_hi, 3u32),
        ] {
            let cand = ray_plane_t_through_origin(oc, ray_dir, normal);
            if cand > t && cand < t_next {
                t_next = cand;
                winning_axis = axis_id;
            }
        }
        for (radius, axis_id) in [(r_lo, 4u32), (r_hi, 5u32)] {
            let cand = ray_sphere_after(oc, ray_dir, radius, t);
            if cand > t && cand < t_next {
                t_next = cand;
                winning_axis = axis_id;
            }
        }

        if t_next >= t_exit { break; }
        last_face_axis = winning_axis;
        // Floor the step: at least 1/1000 of the cell's radial
        // extent, and at least 4 ULPs at this t. Prevents stall
        // when boundary arithmetic rounds to t itself.
        let t_ulp = (t.abs() * 1.2e-7).max(1e-30);
        let cell_eps = (shell * walk.size * 1e-3).max(t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    None
}

/// Kahan-compensated face-subtree walker. Mirror of the shader's
/// `walk_face_subtree` in `face_walk.wgsl`.
fn walk_face_subtree(
    library: &NodeLibrary,
    face_root_id: NodeId,
    un_in: f32,
    vn_in: f32,
    rn_in: f32,
    depth_limit: u32,
) -> FaceWalk {
    let mut out = FaceWalk {
        block: 0,
        depth: 1,
        u_lo: 0.0,
        v_lo: 0.0,
        r_lo: 0.0,
        size: 1.0,
        path: Vec::new(),
    };

    let Some(root) = library.get(face_root_id) else { return out; };
    let mut un = un_in.clamp(0.0, 0.9999999);
    let mut vn = vn_in.clamp(0.0, 0.9999999);
    let mut rn = rn_in.clamp(0.0, 0.9999999);

    // Kahan-compensated accumulators.
    let mut u_sum = 0.0f32; let mut u_comp = 0.0f32;
    let mut v_sum = 0.0f32; let mut v_comp = 0.0f32;
    let mut r_sum = 0.0f32; let mut r_comp = 0.0f32;
    let mut size = 1.0f32;

    let limit = depth_limit.min(MAX_FACE_DEPTH);
    if limit == 0 {
        out.block = 0;
        out.depth = 0;
        return out;
    }

    let mut node_id = face_root_id;
    let _ = root; // suppress unused warning; re-fetched in loop

    for d in 1u32..=limit {
        let Some(node) = library.get(node_id) else {
            out.depth = d;
            return out;
        };

        let us = ((un * 3.0) as usize).min(2);
        let vs = ((vn * 3.0) as usize).min(2);
        let rs = ((rn * 3.0) as usize).min(2);
        let slot = slot_index(us, vs, rs);
        out.path.push((node_id, slot));

        let step_size = size * (1.0 / 3.0);
        // Kahan step.
        kahan_add(&mut u_sum, &mut u_comp, step_size * us as f32);
        kahan_add(&mut v_sum, &mut v_comp, step_size * vs as f32);
        kahan_add(&mut r_sum, &mut r_comp, step_size * rs as f32);
        size = step_size;

        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => {
                // Pad remaining depth with EMPTY_NODE tags so
                // `place_path` depth is uniform at `limit`.
                let mut sun = un * 3.0 - us as f32;
                let mut svn = vn * 3.0 - vs as f32;
                let mut srn = rn * 3.0 - rs as f32;
                for _ in d..limit {
                    let us2 = ((sun * 3.0) as usize).min(2);
                    let vs2 = ((svn * 3.0) as usize).min(2);
                    let rs2 = ((srn * 3.0) as usize).min(2);
                    out.path.push((EMPTY_NODE, slot_index(us2, vs2, rs2)));
                    sun = sun * 3.0 - us2 as f32;
                    svn = svn * 3.0 - vs2 as f32;
                    srn = srn * 3.0 - rs2 as f32;
                }
                out.block = 0;
                out.depth = limit;
                out.u_lo = u_sum + u_comp;
                out.v_lo = v_sum + v_comp;
                out.r_lo = r_sum + r_comp;
                out.size = size;
                return out;
            }
            Child::Block(b) => {
                out.block = b;
                out.depth = d;
                out.u_lo = u_sum + u_comp;
                out.v_lo = v_sum + v_comp;
                out.r_lo = r_sum + r_comp;
                out.size = size;
                return out;
            }
            Child::Node(nid) => {
                if d >= limit {
                    // At LOD cap — materialize the child's effective
                    // block for the LOD-terminal splat.
                    let block = library.get(nid)
                        .map(|c| match c.uniform_type {
                            UNIFORM_MIXED => {
                                if c.representative_block != REPRESENTATIVE_EMPTY {
                                    c.representative_block
                                } else { 0 }
                            }
                            UNIFORM_EMPTY => 0,
                            b => b,
                        })
                        .unwrap_or(0);
                    out.block = block;
                    out.depth = d;
                    out.u_lo = u_sum + u_comp;
                    out.v_lo = v_sum + v_comp;
                    out.r_lo = r_sum + r_comp;
                    out.size = size;
                    return out;
                }
                node_id = nid;
                un = un * 3.0 - us as f32;
                vn = vn * 3.0 - vs as f32;
                rn = rn * 3.0 - rs as f32;
            }
        }
    }

    out.depth = limit;
    out.u_lo = u_sum + u_comp;
    out.v_lo = v_sum + v_comp;
    out.r_lo = r_sum + r_comp;
    out.size = size;
    out
}

#[inline]
fn kahan_add(sum: &mut f32, comp: &mut f32, add: f32) {
    let y = add - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

/// Ray vs. plane through origin with normal `n`. Returns the `t`
/// where the ray crosses the plane, or `-1` if parallel.
#[inline]
fn ray_plane_t_through_origin(origin: [f32; 3], dir: [f32; 3], n: [f32; 3]) -> f32 {
    let denom = sdf::dot(dir, n);
    if denom.abs() < 1e-12 { return -1.0; }
    -sdf::dot(origin, n) / denom
}

/// Ray vs. sphere-through-origin; returns the first `t > after`
/// where the ray crosses radius `radius`, or `-1` if none.
#[inline]
fn ray_sphere_after(origin: [f32; 3], dir: [f32; 3], radius: f32, after: f32) -> f32 {
    let b = sdf::dot(origin, dir);
    let c = sdf::dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return -1.0; }
    let sq = disc.sqrt();
    let s = if b >= 0.0 { 1.0 } else { -1.0 };
    let q = -b - s * sq;
    if q.abs() < 1e-30 { return -1.0; }
    let t0 = q;
    let t1 = c / q;
    let t_lo = t0.min(t1);
    let t_hi = t0.max(t1);
    if t_lo > after { return t_lo; }
    if t_hi > after { return t_hi; }
    -1.0
}

// ───────────────────────────────────────────────────── compat API

/// Back-compat wrapper kept for the cartesian-DDA dispatch site
/// (`src/world/raycast/cartesian.rs`) which passed `max_face_depth`
/// directly. Converts that cap into a pixel-density derived LOD
/// scale by assuming the worst-case (depth cap honored regardless
/// of actual pixel density).
pub(super) fn cs_raycast_in_body_depth_capped(
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
) -> Option<HitInfo> {
    // Synthesize a pixel_density that makes `lod_depth_cap` hit
    // `max_face_depth` independent of t. The cap clamps at
    // `MAX_FACE_DEPTH`, so passing a very large pixel_density
    // produces `min(derived, max_face_depth)` only if we also
    // bound it — simpler to just call the inner walker directly.
    //
    // For the cartesian-DDA dispatch site, LOD is driven by the
    // cartesian parent's LOD anyway; we only need the sphere walker
    // to not descend past max_face_depth.
    cs_raycast_in_body_with_fixed_depth(
        library, body_id, body_origin, body_size,
        inner_r_local, outer_r_local,
        ray_origin, ray_dir, ancestor_path,
        max_face_depth,
    )
}

/// Internal variant of `cs_raycast_in_body` that walks the face
/// subtree to a FIXED depth cap instead of a per-ray LOD cap. Used
/// by the cartesian-DDA dispatch site which has already applied its
/// own LOD and just wants the walker to respect `max_face_depth`.
fn cs_raycast_in_body_with_fixed_depth(
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

    let oc = sdf::sub(ray_origin, cs_center);
    let b = sdf::dot(oc, ray_dir);
    let c_outer = sdf::dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return None; }

    let eps_init = (shell * 1e-5).max(1e-7);
    let mut t = t_enter + eps_init;
    let mut prev_empty_path: Option<Vec<(NodeId, usize)>> = None;

    for _ in 0..4096u32 {
        if t >= t_exit { break; }
        let local = [oc[0] + ray_dir[0]*t, oc[1] + ray_dir[1]*t, oc[2] + ray_dir[2]*t];
        let r = sdf::length(local);
        if r >= cs_outer || r < cs_inner { break; }

        let n = [local[0]/r, local[1]/r, local[2]/r];
        let face = pick_face(n);
        let n_axis = face.normal();
        let (u_axis, v_axis) = face.tangents();
        let axis_dot = sdf::dot(n, n_axis);
        let cube_u = sdf::dot(n, u_axis) / axis_dot;
        let cube_v = sdf::dot(n, v_axis) / axis_dot;
        let un = ((cube_to_ea(cube_u) + 1.0) * 0.5).clamp(0.0, 0.9999999);
        let vn = ((cube_to_ea(cube_v) + 1.0) * 0.5).clamp(0.0, 0.9999999);
        let rn = ((r - cs_inner) / shell).clamp(0.0, 0.9999999);

        let face_slot = FACE_SLOTS[face as usize];
        let body_node = library.get(body_id)?;
        let face_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => { t += shell * 0.33; continue; }
        };

        let walk = walk_face_subtree(library, face_root_id, un, vn, rn, max_face_depth);
        if walk.block != 0 {
            let mut full = ancestor_path.to_vec();
            full.push((body_id, face_slot));
            full.extend(walk.path);
            return Some(HitInfo { path: full, face: 4, t, place_path: prev_empty_path });
        }
        let mut empty_full = ancestor_path.to_vec();
        empty_full.push((body_id, face_slot));
        empty_full.extend(walk.path.clone());
        prev_empty_path = Some(empty_full);

        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = sdf::sub(u_axis, sdf::scale(n_axis, ea_to_cube(u_lo_ea)));
        let n_u_hi = sdf::sub(u_axis, sdf::scale(n_axis, ea_to_cube(u_hi_ea)));
        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = sdf::sub(v_axis, sdf::scale(n_axis, ea_to_cube(v_lo_ea)));
        let n_v_hi = sdf::sub(v_axis, sdf::scale(n_axis, ea_to_cube(v_hi_ea)));
        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        let mut t_next = t_exit + 1.0;
        for normal in [n_u_lo, n_u_hi, n_v_lo, n_v_hi] {
            let cand = ray_plane_t_through_origin(oc, ray_dir, normal);
            if cand > t && cand < t_next { t_next = cand; }
        }
        for radius in [r_lo, r_hi] {
            let cand = ray_sphere_after(oc, ray_dir, radius, t);
            if cand > t && cand < t_next { t_next = cand; }
        }
        if t_next >= t_exit { break; }

        let t_ulp = (t.abs() * 1.2e-7).max(1e-30);
        let cell_eps = (shell * walk.size * 1e-3).max(t_ulp * 4.0);
        t = t_next + cell_eps;
    }
    None
}

// ──────────────────────────────────────────────── deprecated API

// Kept so existing callers (`src/world/aabb.rs`, tests) that used
// the prior walker can continue compiling. Internally routes into
// the unified walker with the caller-supplied fixed depth.
pub fn walk_face_subtree_with_path(
    library: &NodeLibrary,
    face_root_id: NodeId,
    un_in: f32, vn_in: f32, rn_in: f32,
    max_depth: u32,
) -> Option<(u16, u32, Vec<(NodeId, usize)>)> {
    let w = walk_face_subtree(library, face_root_id, un_in, vn_in, rn_in, max_depth);
    Some((w.block, w.depth, w.path))
}
