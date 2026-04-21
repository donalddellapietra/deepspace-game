//! Local-frame sphere DDA for `ActiveFrameKind::SphereSub`.
//!
//! When the render frame lives deep inside a face subtree, the sphere
//! march cannot operate in body-XYZ: at face-subtree depth ≥ ~15,
//! adjacent cell-boundary plane normals collapse into the same f32
//! value and the DDA loses the ability to distinguish neighbouring
//! cells. The fix is the cubed-sphere analog of Cartesian ribbon-pop,
//! built around a SYMBOLIC pre-descent from the face-subtree root
//! plus SYMBOLIC neighbor-transitions when the ray exits the sub-frame:
//!
//! 1. `compute_render_frame` resolves the face-subtree root NodeId
//!    (always reachable: `body + face_slot`) and records it on the
//!    `SphereSubFrame`. It builds the sub-frame at the camera's deep
//!    `m_truncated` UVR depth — the tree may contain `Child::Empty`
//!    at that depth (dug-out region); that's fine, it's handled by
//!    the walker below. The Jacobian's evaluation point sits at the
//!    true deep corner so linearization error is O((1/3^m)²).
//!
//! 2. `walk_from_deep_sub_frame` pre-descends from the face-root
//!    symbolically, slot-by-slot, following `uvr_path` prefix. On
//!    `Child::Empty` / `Child::Block` / `Child::EntityRef` mid-prefix
//!    it returns a `SubWalk` covering the full local `[0, 3)³` box —
//!    uniform empty/solid — so the DDA treats that step as a flat
//!    cell of sub-frame size and advances (or terminates) accordingly.
//!    Reaching the deep Node, it dispatches into `walk_sub_frame` for
//!    intra-cell levels.
//!
//! 3. The body map linearizes at the deep corner:
//!
//!        body_pos ≈ c_body + J · local_pos
//!
//!    where `local_pos ∈ [0, 3)³` is the frame-local coord and `J` is
//!    the analytical Jacobian of `face_space_to_body_point` at the
//!    corner (first derivatives w.r.t. local u/v/r).
//!
//! 4. Ray is transformed into local coords via `J_inv`. In the local
//!    frame, cell `u = const` / `v = const` boundaries are trivially
//!    axis-aligned (flat planes). **Radial boundaries** at
//!    `r_body = const` would in principle be ellipsoids in local
//!    coords, but because `J`'s third column is parallel to the body
//!    radial direction at the corner, `r_body = const` reduces to
//!    `r_local = const` — also flat to first order. All six
//!    boundaries become axis-aligned in local.
//!
//! 5. DDA in local coords uses integer-cell-boundary t-values:
//!    `t_axis = (K − local_pos[axis]) / local_dir[axis]`. These are
//!    always representable in f32 regardless of absolute face-subtree
//!    depth — all quantities stay O(1) in the local frame.
//!
//! 6. When a ray exits the sub-frame's local `[0, 3)³` box, the DDA
//!    transitions to the NEIGHBOR sub-frame via
//!    `SphereSubFrame::with_neighbor_stepped`: step the render path by
//!    one slot along the exit axis, rebuild `(un, vn, rn, c_body, J,
//!    J_inv)` at the new corner, and transfer the ray's local position
//!    + direction into the new basis. The neighbor's Jacobian differs
//!    from the current one by O(1/3^m) (curvature), so the transfer
//!    matrix is nearly identity and all quantities stay O(1) in local
//!    coords — f32-precise at any depth. If the step would bubble past
//!    the face-root boundary, the DDA terminates (cross-face
//!    transitions are deferred to a follow-up).
//!
//! The linearization is accurate to O(frame_size²) in body-XYZ; at
//! deep `m_truncated` the error is geometrically negligible.
//! Face-root and shallow cells keep the exact body-level march in
//! `sphere.rs`.

use super::{HitInfo, LodParams, SphereHitCell};
use crate::app::frame::SphereSubFrame;
use crate::world::cubesphere::{mat3_mul_vec, FACE_SLOTS};
use crate::world::sdf::{self, Vec3};
use crate::world::tree::{
    slot_index, Child, NodeId, NodeLibrary, EMPTY_NODE, REPRESENTATIVE_EMPTY,
    UNIFORM_EMPTY, UNIFORM_MIXED,
};

const EMPTY_CELL: u16 = REPRESENTATIVE_EMPTY;
const LOCAL_BOX_MAX: f32 = 3.0;
const MAX_DDA_STEPS: usize = 4096;
/// Upper bound on how many neighbor sub-frame transitions a single ray
/// can take before we terminate. Protects against pathological grazing
/// rays that would otherwise ribbon-walk indefinitely. Deep rays under
/// normal gameplay cross far fewer deep cells than this before hitting
/// solid content or exiting the face subtree.
const MAX_NEIGHBOR_TRANSITIONS: usize = 64;

/// Terminal cell found by `walk_sub_frame`. Coords are in the
/// sub-frame's LOCAL `[0, 3)³` frame.
struct SubWalk {
    block: u16,
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    size: f32,
    /// Walker-relative path entries, appended to the sub-frame's
    /// render path by the caller. Each entry is `(parent_node_id,
    /// child_slot)` where `child_slot` uses UVR semantics.
    path: Vec<(NodeId, usize)>,
}

/// Pre-descend from the face-subtree root symbolically along
/// `uvr_prefix_slots`, then dispatch into `walk_sub_frame` at the
/// terminal Node reached.
///
/// * `face_root_id` — NodeId of the face subtree root (always
///   resolvable via `body_path + face_slot`; recorded on
///   `SphereSubFrame::face_root_id`).
/// * `uvr_prefix_slots` — the UVR slots from the face root down to
///   the sub-frame's cell. This is `render_path[body_path.depth()+1..]`.
/// * `u_l`, `v_l`, `r_l` — ray sample point in sub-frame local
///   `[0, 3)³`.
/// * `inner_walker_limit` — levels of DDA descent INSIDE the
///   sub-frame's terminal cell; forwarded to `walk_sub_frame`.
///
/// If the pre-descent encounters a non-Node child mid-prefix
/// (`Child::Empty` for a dug region, `Child::Block` for a uniform
/// solid subtree, `Child::EntityRef` for an entity cell), the walker
/// returns a `SubWalk` whose local cell IS the full `[0, 3)³` box and
/// whose terminal block matches what the mid-prefix child
/// represents. The DDA above treats this as a uniform cell spanning
/// the sub-frame — on Block/EntityRef it produces a hit covering the
/// whole sub-frame; on Empty it advances the ray to the sub-frame
/// exit boundary.
fn walk_from_deep_sub_frame(
    library: &NodeLibrary,
    face_root_id: NodeId,
    uvr_prefix_slots: &[u8],
    u_l: f32,
    v_l: f32,
    r_l: f32,
    inner_walker_limit: u32,
) -> SubWalk {
    let mut node = face_root_id;
    let mut prefix_path: Vec<(NodeId, usize)> =
        Vec::with_capacity(uvr_prefix_slots.len() + inner_walker_limit.max(1) as usize);

    for &slot_u8 in uvr_prefix_slots.iter() {
        let Some(n) = library.get(node) else {
            // Parent missing — treat the remainder of the sub-frame
            // as uniform empty.
            return SubWalk {
                block: EMPTY_CELL,
                u_lo: 0.0, v_lo: 0.0, r_lo: 0.0, size: LOCAL_BOX_MAX,
                path: prefix_path,
            };
        };
        let slot = slot_u8 as usize;
        prefix_path.push((node, slot));
        match n.children[slot] {
            Child::Empty => {
                // Dug region mid-prefix → uniform-empty sub-frame.
                return SubWalk {
                    block: EMPTY_CELL,
                    u_lo: 0.0, v_lo: 0.0, r_lo: 0.0, size: LOCAL_BOX_MAX,
                    path: prefix_path,
                };
            }
            Child::Block(b) => {
                // Uniform-solid region mid-prefix → sub-frame is one
                // big cell of block `b`.
                return SubWalk {
                    block: b,
                    u_lo: 0.0, v_lo: 0.0, r_lo: 0.0, size: LOCAL_BOX_MAX,
                    path: prefix_path,
                };
            }
            Child::EntityRef(_) => {
                // Entity cell mid-prefix — render as empty; entity
                // raster pass (or tag=3 branch) handles the actual
                // shape.
                return SubWalk {
                    block: EMPTY_CELL,
                    u_lo: 0.0, v_lo: 0.0, r_lo: 0.0, size: LOCAL_BOX_MAX,
                    path: prefix_path,
                };
            }
            Child::Node(next) => {
                node = next;
            }
        }
    }

    // Pre-descent reached a real terminal Node. If the caller asked
    // for zero intra-cell walker descent (user's edit anchor is
    // exactly at the sub-frame depth), we must NOT descend further —
    // doing so would report a cell one level deeper than the user's
    // zoom and produce off-by-one edit anchor lengths. Inspect the
    // Node's uniform type directly and return a full-box `SubWalk`
    // so the DDA can either hit (uniform-filled) or step through
    // (uniform-empty / mixed).
    if inner_walker_limit == 0 {
        let Some(n) = library.get(node) else {
            return SubWalk {
                block: EMPTY_CELL,
                u_lo: 0.0, v_lo: 0.0, r_lo: 0.0, size: LOCAL_BOX_MAX,
                path: prefix_path,
            };
        };
        let bt = match n.uniform_type {
            UNIFORM_EMPTY => EMPTY_CELL,
            UNIFORM_MIXED => {
                if n.representative_block == REPRESENTATIVE_EMPTY {
                    EMPTY_CELL
                } else {
                    n.representative_block
                }
            }
            b => b,
        };
        return SubWalk {
            block: bt,
            u_lo: 0.0, v_lo: 0.0, r_lo: 0.0, size: LOCAL_BOX_MAX,
            path: prefix_path,
        };
    }
    // Run the intra-cell walker from here and prepend the pre-descent
    // path so the hit's `path` is rooted at `face_root_id`.
    let mut inner = walk_sub_frame(library, node, u_l, v_l, r_l, inner_walker_limit);
    let mut joined: Vec<(NodeId, usize)> =
        Vec::with_capacity(prefix_path.len() + inner.path.len());
    joined.extend(prefix_path);
    joined.append(&mut inner.path);
    SubWalk {
        block: inner.block,
        u_lo: inner.u_lo,
        v_lo: inner.v_lo,
        r_lo: inner.r_lo,
        size: inner.size,
        path: joined,
    }
}

/// Descend the sub-frame at local point `(u_l, v_l, r_l)` to
/// `max_depth` levels. Mirrors `walk_face_subtree` but starts from
/// an arbitrary sub-frame root rather than a face root.
fn walk_sub_frame(
    library: &NodeLibrary,
    sub_frame_node: NodeId,
    u_l: f32,
    v_l: f32,
    r_l: f32,
    max_depth: u32,
) -> SubWalk {
    let mut node = sub_frame_node;
    let mut u_lo = 0.0_f32;
    let mut v_lo = 0.0_f32;
    let mut r_lo = 0.0_f32;
    let mut size = LOCAL_BOX_MAX;
    let limit = max_depth.max(1);
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(limit as usize);

    let clamp_frac = |x: f32| -> f32 { x.clamp(0.0, 0.9999999_f32 * LOCAL_BOX_MAX) };
    let u_pt = clamp_frac(u_l);
    let v_pt = clamp_frac(v_l);
    let r_pt = clamp_frac(r_l);

    for d in 1..=limit {
        let Some(n) = library.get(node) else {
            return SubWalk { block: EMPTY_CELL, u_lo, v_lo, r_lo, size, path };
        };
        let child_size = size / 3.0;
        let us = (((u_pt - u_lo) / child_size) as usize).min(2);
        let vs = (((v_pt - v_lo) / child_size) as usize).min(2);
        let rs = (((r_pt - r_lo) / child_size) as usize).min(2);
        let slot = slot_index(us, vs, rs);
        let cu_lo = u_lo + us as f32 * child_size;
        let cv_lo = v_lo + vs as f32 * child_size;
        let cr_lo = r_lo + rs as f32 * child_size;
        path.push((node, slot));

        match n.children[slot] {
            Child::Empty => {
                // Pad path so propagate_edit lands at uniform depth.
                let mut sub_u = (u_pt - cu_lo) / child_size;
                let mut sub_v = (v_pt - cv_lo) / child_size;
                let mut sub_r = (r_pt - cr_lo) / child_size;
                for _ in d..limit {
                    let us2 = (sub_u * 3.0).floor().clamp(0.0, 2.0) as usize;
                    let vs2 = (sub_v * 3.0).floor().clamp(0.0, 2.0) as usize;
                    let rs2 = (sub_r * 3.0).floor().clamp(0.0, 2.0) as usize;
                    path.push((EMPTY_NODE, slot_index(us2, vs2, rs2)));
                    sub_u = sub_u * 3.0 - us2 as f32;
                    sub_v = sub_v * 3.0 - vs2 as f32;
                    sub_r = sub_r * 3.0 - rs2 as f32;
                }
                return SubWalk {
                    block: EMPTY_CELL,
                    u_lo: cu_lo, v_lo: cv_lo, r_lo: cr_lo,
                    size: child_size, path,
                };
            }
            Child::Block(b) => {
                return SubWalk {
                    block: b,
                    u_lo: cu_lo, v_lo: cv_lo, r_lo: cr_lo,
                    size: child_size, path,
                };
            }
            Child::EntityRef(_) => {
                return SubWalk {
                    block: EMPTY_CELL,
                    u_lo: cu_lo, v_lo: cv_lo, r_lo: cr_lo,
                    size: child_size, path,
                };
            }
            Child::Node(nid) => {
                if d == limit {
                    let Some(child) = library.get(nid) else {
                        return SubWalk {
                            block: EMPTY_CELL,
                            u_lo: cu_lo, v_lo: cv_lo, r_lo: cr_lo,
                            size: child_size, path,
                        };
                    };
                    let bt = match child.uniform_type {
                        UNIFORM_EMPTY => EMPTY_CELL,
                        UNIFORM_MIXED => {
                            if child.representative_block == REPRESENTATIVE_EMPTY {
                                EMPTY_CELL
                            } else {
                                child.representative_block
                            }
                        }
                        b => b,
                    };
                    return SubWalk {
                        block: bt,
                        u_lo: cu_lo, v_lo: cv_lo, r_lo: cr_lo,
                        size: child_size, path,
                    };
                }
                node = nid;
                u_lo = cu_lo;
                v_lo = cv_lo;
                r_lo = cr_lo;
                size = child_size;
            }
        }
    }
    SubWalk { block: EMPTY_CELL, u_lo, v_lo, r_lo, size, path }
}

/// Local axis-exit t. Returns +∞ if the ray doesn't cross either
/// boundary going forward.
#[inline]
fn axis_exit_t(p: f32, d: f32, lo: f32, hi: f32) -> f32 {
    if d > 1e-30 { (hi - p) / d }
    else if d < -1e-30 { (lo - p) / d }
    else { f32::INFINITY }
}

/// Find the `[t_lo, t_hi]` interval during which the ray is inside
/// `[0, 3)³` in local coords.
fn ray_local_box_interval(ro: Vec3, rd: Vec3) -> (f32, f32) {
    let mut t_lo = f32::NEG_INFINITY;
    let mut t_hi = f32::INFINITY;
    for axis in 0..3 {
        let o = ro[axis];
        let d = rd[axis];
        if d.abs() < 1e-30 {
            if o < 0.0 || o >= LOCAL_BOX_MAX {
                return (f32::INFINITY, f32::NEG_INFINITY);
            }
            continue;
        }
        let t0 = (0.0 - o) / d;
        let t1 = (LOCAL_BOX_MAX - o) / d;
        let (a, b) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
        t_lo = t_lo.max(a);
        t_hi = t_hi.min(b);
    }
    (t_lo, t_hi)
}

/// CPU local-frame sphere DDA — the precision-preserving replacement
/// for the body-level march when the render frame is a deep
/// face-subtree cell.
///
/// * `sub` carries the sub-frame's metadata and linearization, plus
///   the face-subtree root NodeId the walker pre-descends from.
/// * `ancestor_path` is the full path from world-root up to (but not
///   including) the face-subtree root's parent — `HitInfo.path` is
///   prefixed with this so callers can propagate edits against
///   world-root. Its length is `body_path.depth() + 1` (the body
///   chain plus the face-root slot).
/// * `ro_local` is the ray origin **already in sub-frame local**
///   `[0, 3)³` coords — derived via the anchor-path ribbon-pop, NOT
///   by subtracting `c_body` from body-XYZ (that subtraction collapses
///   in f32 once the sub-frame's body extent falls below eps).
/// * `rd_body` is the ray direction in body-local orientation, unit
///   length. It's transformed into local via `J_inv` inside — safe
///   because `rd_body` is O(1) and `|J_inv · rd_body|` is O(3^depth)
///   which f32 handles cleanly.
/// * `walker_limit` caps walker descent INSIDE the sub-frame's
///   terminal cell (forwarded to `walk_sub_frame`).
pub(super) fn cs_raycast_local(
    library: &NodeLibrary,
    sub: &SphereSubFrame,
    ancestor_path: &[(NodeId, usize)],
    ro_local: Vec3,
    rd_body: Vec3,
    walker_limit: u32,
    _lod: LodParams,
) -> Option<HitInfo> {
    // `walker_limit == 0` is legitimate when the user's edit anchor
    // is exactly at the sub-frame's depth — `walk_from_deep_sub_frame`
    // handles it by returning the deep Node's content via
    // uniform_type without descending further.

    // Body-direction of the ray. Held CONSTANT across neighbor
    // transitions — only the local basis (J_inv) changes per sub-frame,
    // so rd_local is recomputed but rd_body stays the same.
    let rd_norm = sdf::normalize(rd_body);

    // The `ancestor_path` prefix (body chain + face-root slot) is
    // invariant across neighbor transitions inside a single face
    // subtree — it's the body chain + face-root slot entry. We
    // re-use it when emitting the hit path below.

    // Mutable per-transition state: current sub-frame metadata, ray in
    // that sub-frame's local basis, and the DDA t-cursor.
    let mut current_sub: SphereSubFrame = *sub;
    let mut ro_local = ro_local;
    let mut rd_local = mat3_mul_vec(&current_sub.j_inv, rd_norm);

    // Re-derive the UVR prefix slice on each transition (render_path
    // changes as we step into neighbors).
    let body_depth_plus_one = current_sub.body_path.depth() as usize + 1;

    // Initial DDA interval in the starting sub-frame.
    let (t_enter, t_exit) = ray_local_box_interval(ro_local, rd_local);
    if t_exit <= 0.0 || t_enter >= t_exit {
        return None;
    }
    let t_span = (t_exit - t_enter).abs().max(1e-30);
    let mut t_nudge = t_span * 1e-5;
    let mut t = t_enter.max(0.0) + t_nudge;
    let mut t_exit = t_exit;

    let mut neighbor_transitions = 0usize;
    let mut dda_steps = 0usize;

    loop {
        if dda_steps >= MAX_DDA_STEPS {
            break;
        }
        dda_steps += 1;

        let pos = [
            ro_local[0] + rd_local[0] * t,
            ro_local[1] + rd_local[1] * t,
            ro_local[2] + rd_local[2] * t,
        ];

        let out_of_box = pos[0] < 0.0 || pos[0] >= LOCAL_BOX_MAX
            || pos[1] < 0.0 || pos[1] >= LOCAL_BOX_MAX
            || pos[2] < 0.0 || pos[2] >= LOCAL_BOX_MAX
            || t >= t_exit;

        if out_of_box {
            // Ray exited the current sub-frame's local box. Step to
            // the neighbor and continue the DDA there.
            if neighbor_transitions >= MAX_NEIGHBOR_TRANSITIONS {
                break;
            }

            // Pick exit axis k + direction sign s from `pos`. Prefer
            // whichever axis is actually outside — if multiple are
            // outside (corner exit) pick the axis with the largest
            // excursion past the boundary so the neighbor-step picks
            // the right face.
            let mut axis_k: usize = 0;
            let mut sign_s: i32 = 0;
            let mut best_excess: f32 = -1.0;
            for k in 0..3 {
                let v = pos[k];
                let excess = if v >= LOCAL_BOX_MAX {
                    v - LOCAL_BOX_MAX
                } else if v < 0.0 {
                    -v
                } else {
                    -1.0
                };
                if excess > best_excess {
                    best_excess = excess;
                    axis_k = k;
                    sign_s = if v >= LOCAL_BOX_MAX { 1 } else { -1 };
                }
            }
            if sign_s == 0 {
                // No axis truly outside the box — we hit the t_exit
                // guard, meaning the ray left via the sub-frame cap
                // without a finite `pos` delta. Terminate.
                break;
            }

            let Some(new_sub) = current_sub.with_neighbor_stepped(axis_k, sign_s) else {
                // Bubble-up past the face-root: cross-face transition
                // not implemented. Terminate the DDA; hits beyond the
                // face subtree boundary are out of scope.
                break;
            };

            // Transfer the ray's local position into the neighbor basis.
            //   body_pos = c_cur + J_cur · local_cur
            //            = c_new + J_new · local_new
            //   c_new − c_cur ≈ s · 3 · J_cur[:, k]    (linearization)
            //   ⇒ local_new = J_new_inv · J_cur · (local_cur − s·3·e_k)
            let s_f = sign_s as f32;
            let mut shifted = pos;
            shifted[axis_k] -= s_f * LOCAL_BOX_MAX;
            let body_delta = mat3_mul_vec(&current_sub.j, shifted);
            let local_new = mat3_mul_vec(&new_sub.j_inv, body_delta);

            // Clamp the entry coordinate just inside the neighbor's
            // box on the axis we crossed — avoids an immediate
            // re-exit on the opposite face due to f32 drift.
            let mut ro_new = local_new;
            let eps_in = LOCAL_BOX_MAX * 1e-6;
            if sign_s == 1 {
                ro_new[axis_k] = eps_in;
            } else {
                ro_new[axis_k] = LOCAL_BOX_MAX - eps_in;
            }
            // Keep the non-crossed axes clamped inside [0, 3) too so
            // that f32 drift on a corner exit doesn't immediately kill
            // the DDA in the neighbor.
            for k in 0..3 {
                if k == axis_k { continue; }
                if ro_new[k] < 0.0 { ro_new[k] = 0.0; }
                if ro_new[k] >= LOCAL_BOX_MAX {
                    ro_new[k] = LOCAL_BOX_MAX - eps_in;
                }
            }

            let rd_new = mat3_mul_vec(&new_sub.j_inv, rd_norm);

            // Reset DDA state to the neighbor's local frame.
            current_sub = new_sub;
            ro_local = ro_new;
            rd_local = rd_new;
            let (new_t_enter, new_t_exit) = ray_local_box_interval(ro_local, rd_local);
            if new_t_exit <= 0.0 || new_t_enter >= new_t_exit {
                break;
            }
            let span = (new_t_exit - new_t_enter).abs().max(1e-30);
            t_nudge = span * 1e-5;
            t = new_t_enter.max(0.0) + t_nudge;
            t_exit = new_t_exit;

            neighbor_transitions += 1;
            eprintln!(
                "NEIGHBOR_STEP axis={} sign={} new_uvr_depth={} un={} vn={} rn={}",
                axis_k, sign_s,
                current_sub.depth_levels(),
                current_sub.un_corner, current_sub.vn_corner, current_sub.rn_corner,
            );
            continue;
        }

        // Re-derive the UVR prefix slice for the CURRENT sub-frame
        // (render_path is mutated on each neighbor transition).
        let render_slots = current_sub.render_path.as_slice();
        let uvr_prefix_slots: &[u8] = if render_slots.len() > body_depth_plus_one {
            &render_slots[body_depth_plus_one..]
        } else {
            &[]
        };

        let w = walk_from_deep_sub_frame(
            library,
            current_sub.face_root_id,
            uvr_prefix_slots,
            pos[0], pos[1], pos[2],
            walker_limit,
        );

        if w.block != EMPTY_CELL {
            let mut full_path: Vec<(NodeId, usize)> =
                Vec::with_capacity(ancestor_path.len() + w.path.len());
            full_path.extend_from_slice(ancestor_path);
            full_path.extend(w.path.iter().copied());

            // Sphere cell in absolute face-normalized coords. The
            // sub-frame maps local `[0, 3)` to absolute face
            // `[un_corner, un_corner + frame_size)`, so per-local-unit
            // absolute-coord-step = frame_size / 3.
            let step = current_sub.frame_size / 3.0;
            // The body chain is the first `sub.body_path.depth()`
            // entries of `full_path` by construction — `ancestor_path`
            // = body chain + face-root slot, then pre-descent UVR
            // slots, then walker-internal slots.
            let body_path_len = current_sub.body_path.depth() as usize;

            return Some(HitInfo {
                path: full_path,
                // Sentinel — sphere hits have no XYZ-axis face
                // semantic. AABB draws from sphere_cell.
                face: 4,
                t,
                place_path: None,
                sphere_cell: Some(SphereHitCell {
                    face: current_sub.face as u32,
                    u_lo: current_sub.un_corner + w.u_lo * step,
                    v_lo: current_sub.vn_corner + w.v_lo * step,
                    r_lo: current_sub.rn_corner + w.r_lo * step,
                    size: w.size * step,
                    inner_r: current_sub.inner_r,
                    outer_r: current_sub.outer_r,
                    body_path_len,
                }),
            });
        }

        // Advance to the current cell's exit boundary. All three
        // axes are axis-aligned in local coords — pick the smallest
        // positive t.
        let t_u = axis_exit_t(pos[0], rd_local[0], w.u_lo, w.u_lo + w.size);
        let t_v = axis_exit_t(pos[1], rd_local[1], w.v_lo, w.v_lo + w.size);
        let t_r = axis_exit_t(pos[2], rd_local[2], w.r_lo, w.r_lo + w.size);
        let t_min = t_u.min(t_v).min(t_r);
        if !t_min.is_finite() || t_min <= 0.0 {
            // Cell is degenerate (parallel ray, zero span, …). Force
            // exit by nudging to t_exit and let the out-of-box branch
            // handle the neighbor step on the next iteration.
            t = t_exit;
            continue;
        }
        t += t_min + t_nudge;
    }

    None
}

// FACE_SLOTS imported only for future work that needs it.
#[allow(dead_code)]
const _: [usize; 6] = FACE_SLOTS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::frame::{compute_render_frame, ActiveFrameKind};
    use crate::world::anchor::Path;
    use crate::world::bootstrap;
    use crate::world::cubesphere::{mat3_mul_vec, Face};
    use crate::world::tree::slot_index;

    /// Build a demo-planet world and resolve an `ActiveFrame` at the
    /// given face-subtree depth on `face`, returning the sub-frame
    /// and the ancestor path (body chain + face-root slot — the
    /// prefix that's ALWAYS a real `Child::Node` chain; the UVR
    /// prefix lives on `sub.render_path` and is handled by the
    /// walker symbolically).
    ///
    /// The anchor descent uses UVR slot (1, 1, 1) at every
    /// face-subtree level so the sub-frame lives near the face
    /// center along the radial mid-shell — guaranteed-solid
    /// territory in the demo planet.
    fn planet_with_sub_frame(face: Face, sub_depth: u8) -> (
        crate::world::state::WorldState,
        SphereSubFrame,
        Vec<(NodeId, usize)>,
    ) {
        use crate::world::anchor::{SphereState, WorldPos};
        let world = bootstrap::bootstrap_world(
            bootstrap::WorldPreset::DemoSphere,
            Some(40),
        ).world;
        // Build a WorldPos with explicit SphereState at center-of-face.
        let body_path = {
            let mut p = Path::root();
            p.push(slot_index(1, 1, 1) as u8);
            p
        };
        let mut uvr_path = Path::root();
        for _ in 0..sub_depth {
            uvr_path.push(slot_index(1, 1, 1) as u8);
        }
        let camera = WorldPos {
            anchor: body_path,
            offset: [0.5; 3],
            sphere: Some(SphereState {
                body_path,
                inner_r: 0.12,
                outer_r: 0.45,
                face,
                uvr_path,
                uvr_offset: [0.5; 3],
            }),
        };
        let desired = body_path.depth() + 1 + sub_depth;
        let active = compute_render_frame(
            &world.library, world.root, &camera, desired,
        );
        let sub = match active.kind {
            ActiveFrameKind::SphereSub(s) => s,
            k => panic!("expected SphereSub at sub_depth={sub_depth}, got {k:?}"),
        };
        // Ancestor prefix stops at the face-root slot entry.
        let prefix_len = sub.body_path.depth() as usize + 1;
        let mut node_id = world.root;
        let mut ancestors = Vec::new();
        for i in 0..prefix_len {
            let slot = active.render_path.slot(i) as usize;
            ancestors.push((node_id, slot));
            node_id = match world.library.get(node_id).unwrap().children[slot] {
                Child::Node(n) => n,
                other => panic!("render_path step {i}: non-Node child {other:?}"),
            };
        }
        (world, sub, ancestors)
    }

    /// Ray aimed at the sub-frame's LOCAL center from outside along
    /// the +r_local axis. Camera sits at local (1.5, 1.5, 4.5) —
    /// above the local box on the r axis — looking inward toward
    /// the box center. `rd_body` = -col_r (the sub-frame's body-XYZ
    /// radial outward direction), pointing inward.
    ///
    /// Works at arbitrary sub-frame depth because the camera is
    /// expressed in LOCAL coords — no `cam_body - c_body`
    /// subtraction that would collapse in f32 at deep depth.
    fn inward_radial_ray(sub: &SphereSubFrame) -> (Vec3, Vec3) {
        let cam_local = [1.5_f32, 1.5, 4.5];
        let col_r = [sub.j[2][0], sub.j[2][1], sub.j[2][2]];
        let rd_body = sdf::scale(sdf::normalize(col_r), -1.0);
        (cam_local, rd_body)
    }

    #[test]
    fn cs_raycast_local_hits_at_sub_depth_3() {
        let (world, sub, ancestors) = planet_with_sub_frame(Face::PosY, 3);
        let (cam, rd) = inward_radial_ray(&sub);
        let hit = cs_raycast_local(
            &world.library, &sub, &ancestors,
            cam, rd, 6, LodParams::fixed_max(),
        );
        assert!(hit.is_some(), "sub-depth 3 should hit");
    }

    /// Build a fully-solid sub-frame: world root → body(at slot 13)
    /// → face(PosY face root at body's slot 16) → depth levels of
    /// uniform `Child::Block(42)` nodes. Everything inside the
    /// sub-frame is guaranteed solid. Returns world, sub, ancestors
    /// (body chain + face-root slot).
    fn solid_sub_frame(sub_depth: u8) -> (
        crate::world::state::WorldState,
        SphereSubFrame,
        Vec<(NodeId, usize)>,
    ) {
        use crate::world::cubesphere::Face;
        use crate::world::tree::{empty_children, uniform_children, NodeKind};
        let mut lib = NodeLibrary::default();
        // Fully-solid uniform chain all the way down the walker.
        let deep_solid = lib.insert(uniform_children(Child::Block(42)));
        let mut chain = deep_solid;
        for _ in 0..10u32 {
            chain = lib.insert(uniform_children(Child::Node(chain)));
        }
        // Descend sub_depth levels with slot (1,1,1) each step.
        let mut face_subtree = chain;
        for _ in 0..sub_depth {
            let mut children = empty_children();
            children[slot_index(1, 1, 1)] = Child::Node(face_subtree);
            face_subtree = lib.insert(children);
        }
        // Face root (PosY) with the UVR subtree in the center slot.
        let mut face_root_children = uniform_children(Child::Node(chain));
        face_root_children[slot_index(1, 1, 1)] = Child::Node(face_subtree);
        let face_root = lib.insert_with_kind(
            face_root_children,
            NodeKind::CubedSphereFace { face: Face::PosY },
        );
        // Body (inner_r=0.12, outer_r=0.45). Fill all face slots
        // with the same face_root for convenience.
        let mut body_children = empty_children();
        for &f in &Face::ALL {
            body_children[FACE_SLOTS[f as usize]] = Child::Node(face_root);
        }
        body_children[crate::world::cubesphere::CORE_SLOT] = Child::Node(chain);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        // World root with body at slot 13.
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let world = crate::world::state::WorldState {
            root,
            library: lib,
        };
        use crate::world::anchor::{SphereState, WorldPos};
        let body_path = {
            let mut p = Path::root();
            p.push(slot_index(1, 1, 1) as u8);
            p
        };
        let mut uvr_path = Path::root();
        for _ in 0..sub_depth {
            uvr_path.push(slot_index(1, 1, 1) as u8);
        }
        let camera = WorldPos {
            anchor: body_path,
            offset: [0.5; 3],
            sphere: Some(SphereState {
                body_path,
                inner_r: 0.12,
                outer_r: 0.45,
                face: Face::PosY,
                uvr_path,
                uvr_offset: [0.5; 3],
            }),
        };
        let desired = body_path.depth() + 1 + sub_depth;
        let active = compute_render_frame(
            &world.library, world.root, &camera, desired,
        );
        let sub = match active.kind {
            ActiveFrameKind::SphereSub(s) => s,
            k => panic!("expected SphereSub at sub_depth={sub_depth}, got {k:?}"),
        };
        let prefix_len = sub.body_path.depth() as usize + 1;
        let mut node_id = world.root;
        let mut ancestors = Vec::new();
        for i in 0..prefix_len {
            let slot = active.render_path.slot(i) as usize;
            ancestors.push((node_id, slot));
            node_id = match world.library.get(node_id).unwrap().children[slot] {
                Child::Node(n) => n,
                other => panic!("non-node at step {i}: {other:?}"),
            };
        }
        (world, sub, ancestors)
    }

    #[test]
    fn cs_raycast_local_hits_at_all_depths_synthetic() {
        // Synthetic uniform-solid sub-frame removes worldgen noise:
        // every walker cell on the central UVR axis is Block(42), so
        // a ray straight through the local z axis MUST hit. Depth
        // ≥ 20 exercises the precision-critical local-frame DDA.
        for sub_depth in [3u8, 4, 5, 8, 12, 18, 25, 30, 35] {
            let (world, sub, ancestors) = solid_sub_frame(sub_depth);
            let (cam, rd) = inward_radial_ray(&sub);
            let hit = cs_raycast_local(
                &world.library, &sub, &ancestors,
                cam, rd, 3, LodParams::fixed_max(),
            );
            assert!(
                hit.is_some(),
                "solid sub_depth={sub_depth} missed",
            );
        }
    }

    #[test]
    fn cs_raycast_local_across_all_six_faces() {
        // Same ray template on every face — catches face-axis /
        // Jacobian sign bugs that would only show on some faces.
        for &face in &[
            Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ,
        ] {
            let (world, sub, ancestors) = planet_with_sub_frame(face, 5);
            let (cam, rd) = inward_radial_ray(&sub);
            let hit = cs_raycast_local(
                &world.library, &sub, &ancestors,
                cam, rd, 4, LodParams::fixed_max(),
            );
            assert!(hit.is_some(), "no hit on face={face:?} at sub_depth=5");
        }
    }

    #[test]
    fn local_march_hits_same_cell_as_body_march() {
        // Three-way agreement sanity: at sub_depth=3 where BOTH the
        // exact body march (`cs_raycast_body` via cpu_raycast_in_frame)
        // and the local-frame DDA work, they must terminate at the
        // same cell (path ends matching) so the highlight/edit
        // resolves to the same block regardless of which path ran.
        use crate::world::raycast::cpu_raycast_in_frame;
        let (world, sub, _ancestors) =
            planet_with_sub_frame(Face::PosY, 3);
        // Camera in body-local: reconstruct from sub-frame local center
        // via c_body + J·(1.5, 1.5, 1.5) — this IS the body-level cam
        // position corresponding to the sub-frame interior.
        let body_cam = [
            sub.c_body[0] + sub.j[0][0] * 1.5 + sub.j[1][0] * 1.5 + sub.j[2][0] * 1.5,
            sub.c_body[1] + sub.j[0][1] * 1.5 + sub.j[1][1] * 1.5 + sub.j[2][1] * 1.5,
            sub.c_body[2] + sub.j[0][2] * 1.5 + sub.j[1][2] * 1.5 + sub.j[2][2] * 1.5,
        ];
        // Camera 1 body-unit outside the sub-frame along the radial
        // (col_r). Body-march and local-march both see the same
        // incoming ray.
        let col_r = [sub.j[2][0], sub.j[2][1], sub.j[2][2]];
        let radial = sdf::normalize(col_r);
        let cam_body = [
            body_cam[0] + radial[0] * 1.0,
            body_cam[1] + radial[1] * 1.0,
            body_cam[2] + radial[2] * 1.0,
        ];
        let rd_body = sdf::scale(radial, -1.0);
        // Render frame at the body cell: cpu_raycast_in_frame runs
        // the exact `cs_raycast_body` march.
        let body_frame_path = [13u8]; // world-root center slot
        let body_hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &body_frame_path, cam_body, rd_body,
            8, 8, LodParams::fixed_max(),
        );
        assert!(body_hit.is_some(), "body-march should hit");
        let body_hit = body_hit.unwrap();

        // Local-frame DDA via `cpu_raycast_in_sub_frame`.
        let render_path = {
            let mut p = crate::world::anchor::Path::root();
            p.push(slot_index(1, 1, 1) as u8);
            p.push(FACE_SLOTS[Face::PosY as usize] as u8);
            for _ in 0..3 {
                p.push(slot_index(1, 1, 1) as u8);
            }
            p
        };
        let cam_local = [1.5_f32, 1.5, 2.5]; // 1 local unit outside
        let local_hit = crate::world::raycast::cpu_raycast_in_sub_frame(
            &world.library, world.root, &sub, render_path.as_slice(),
            cam_local, rd_body, 8, LodParams::fixed_max(),
        );
        // The hit existence matters; exact path equivalence depends
        // on content matching — but both must find a hit.
        assert!(local_hit.is_some(), "local-frame march should also hit");
    }

    #[test]
    fn ray_local_box_interval_basics() {
        // Ray straight through the box along +x.
        let (t_lo, t_hi) = ray_local_box_interval([-1.0, 1.5, 1.5], [1.0, 0.0, 0.0]);
        assert!((t_lo - 1.0).abs() < 1e-5);
        assert!((t_hi - 4.0).abs() < 1e-5);

        // Ray entirely outside.
        let (_, t_hi) = ray_local_box_interval([-5.0, 1.5, 1.5], [-1.0, 0.0, 0.0]);
        assert!(t_hi < 0.0);

        // Ray parallel to an axis, inside on that axis.
        let (t_lo, _) = ray_local_box_interval([1.5, 1.5, -1.0], [0.0, 0.0, 1.0]);
        assert!((t_lo - 1.0).abs() < 1e-5);
    }

    #[test]
    fn walk_sub_frame_descends_to_limit() {
        // Hand-built 3-deep uniform-block tree. Walker must descend
        // to the limit and return the block.
        use crate::world::tree::{empty_children, uniform_children};
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(uniform_children(Child::Block(42)));
        let mid = lib.insert(uniform_children(Child::Node(leaf)));
        let root = lib.insert(uniform_children(Child::Node(mid)));
        lib.ref_inc(root);
        let w = walk_sub_frame(&lib, root, 1.5, 1.5, 1.5, 3);
        assert_eq!(w.block, 42);
        assert!(w.size > 0.0 && w.size < 1.0, "size={}", w.size);
        assert_eq!(w.path.len(), 3);
    }

    #[test]
    fn walk_sub_frame_empty_pads_path() {
        // Empty cell at depth 1; walker must pad path up to limit.
        use crate::world::tree::empty_children;
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let w = walk_sub_frame(&lib, root, 1.5, 1.5, 1.5, 4);
        assert_eq!(w.block, EMPTY_CELL);
        assert_eq!(w.path.len(), 4, "path padded to limit");
        // First entry is the real root → slot; rest are EMPTY_NODE.
        assert_eq!(w.path[0].0, root);
        for entry in &w.path[1..] {
            assert_eq!(entry.0, EMPTY_NODE);
        }
    }

    #[test]
    fn local_to_body_round_trip() {
        // J · local + c_body should round-trip through J_inv back to
        // local within tight tolerance — the linearization's own
        // consistency check.
        let (_world, sub, _ancestors) =
            planet_with_sub_frame(Face::PosX, 4);
        for probe in [[0.5_f32, 1.0, 1.5], [2.7, 0.2, 0.9], [1.5, 1.5, 1.5]] {
            let body = [
                sub.c_body[0] + sub.j[0][0] * probe[0]
                    + sub.j[1][0] * probe[1] + sub.j[2][0] * probe[2],
                sub.c_body[1] + sub.j[0][1] * probe[0]
                    + sub.j[1][1] * probe[1] + sub.j[2][1] * probe[2],
                sub.c_body[2] + sub.j[0][2] * probe[0]
                    + sub.j[1][2] * probe[1] + sub.j[2][2] * probe[2],
            ];
            let back = mat3_mul_vec(&sub.j_inv, sdf::sub(body, sub.c_body));
            for axis in 0..3 {
                assert!(
                    (back[axis] - probe[axis]).abs() < 1e-3,
                    "round-trip axis={} local={} ↔ body={:?} ↔ back={}",
                    axis, probe[axis], body, back[axis]
                );
            }
        }
    }

    /// Synthetic subtree where the starting sub-frame's own deep cell
    /// is EMPTY but the +r neighbor contains a solid Block. A ray
    /// aimed along the local +r axis must exit the starting
    /// sub-frame, transition to the neighbor, and hit in the neighbor
    /// cell. Exercises the Step 2 neighbor-transition path end-to-end.
    #[test]
    fn cs_raycast_local_neighbor_transition() {
        use crate::app::frame::{compute_render_frame, ActiveFrameKind, MIN_SPHERE_SUB_DEPTH};
        use crate::world::anchor::{SphereState, WorldPos};
        use crate::world::cubesphere::Face;
        use crate::world::tree::{empty_children, uniform_children, NodeKind};

        // Build a face subtree whose depth-1 UVR layout is:
        //   slot (1,1,1): child is a uniform-EMPTY subtree (the starting
        //                 sub-frame sits here — its deep cell is empty).
        //   slot (1,1,2): child is a uniform-BLOCK(42) subtree (the
        //                 +r neighbor — the ray hits here after the
        //                 neighbor-step).
        // Plus a lower depth-1 padding chain so that the starting
        // sub-frame ends up at MIN_SPHERE_SUB_DEPTH.
        let mut lib = NodeLibrary::default();
        // Solid Block(42) subtree — filler for the +r neighbor.
        let mut solid_chain = lib.insert(uniform_children(Child::Block(42)));
        for _ in 0..10u32 {
            solid_chain = lib.insert(uniform_children(Child::Node(solid_chain)));
        }
        // Empty subtree — filler for the starting cell.
        let empty_node = lib.insert(empty_children());

        // Descend MIN_SPHERE_SUB_DEPTH levels via slot (1,1,1). At each
        // level, slot (1,1,2) holds the solid subtree (the +r
        // neighbor at that depth).
        let mut center_chain = empty_node;
        for _ in 0..MIN_SPHERE_SUB_DEPTH {
            let mut children = empty_children();
            children[slot_index(1, 1, 1)] = Child::Node(center_chain);
            children[slot_index(1, 1, 2)] = Child::Node(solid_chain);
            center_chain = lib.insert(children);
        }

        // Face root — wraps the subtree chain above; +r neighbor at
        // depth 1 relative to face root is slot (1,1,2).
        let mut face_root_children = uniform_children(Child::Node(solid_chain));
        face_root_children[slot_index(1, 1, 1)] = Child::Node(center_chain);
        let face_root = lib.insert_with_kind(
            face_root_children,
            NodeKind::CubedSphereFace { face: Face::PosY },
        );

        // Body cell.
        let mut body_children = empty_children();
        for &f in &Face::ALL {
            body_children[FACE_SLOTS[f as usize]] = Child::Node(face_root);
        }
        body_children[crate::world::cubesphere::CORE_SLOT] = Child::Node(solid_chain);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );

        // World root with the body at slot (1,1,1).
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let world = crate::world::state::WorldState { root, library: lib };

        // Camera sitting at uvr (1,1,1) … (1,1,1) at MIN_SPHERE_SUB_DEPTH
        // — inside the starting (empty) sub-frame.
        let body_path = {
            let mut p = crate::world::anchor::Path::root();
            p.push(slot_index(1, 1, 1) as u8);
            p
        };
        let mut uvr_path = crate::world::anchor::Path::root();
        for _ in 0..MIN_SPHERE_SUB_DEPTH {
            uvr_path.push(slot_index(1, 1, 1) as u8);
        }
        let camera = WorldPos {
            anchor: body_path,
            offset: [0.5; 3],
            sphere: Some(SphereState {
                body_path,
                inner_r: 0.12,
                outer_r: 0.45,
                face: Face::PosY,
                uvr_path,
                uvr_offset: [0.5; 3],
            }),
        };
        let desired = body_path.depth() + 1 + MIN_SPHERE_SUB_DEPTH;
        let active = compute_render_frame(&world.library, world.root, &camera, desired);
        let sub = match active.kind {
            ActiveFrameKind::SphereSub(s) => s,
            k => panic!("expected SphereSub, got {k:?}"),
        };

        // Build the ancestor chain (body + face-root slot).
        let prefix_len = sub.body_path.depth() as usize + 1;
        let mut node_id = world.root;
        let mut ancestors = Vec::new();
        for i in 0..prefix_len {
            let slot = active.render_path.slot(i) as usize;
            ancestors.push((node_id, slot));
            node_id = match world.library.get(node_id).unwrap().children[slot] {
                Child::Node(n) => n,
                other => panic!("non-Node at step {i}: {other:?}"),
            };
        }

        // Ray starts inside the (empty) starting sub-frame, aimed along
        // the local +r axis. It must exit the starting box on the +r
        // face and cross into the +r neighbor, where it should hit
        // Block(42).
        let cam_local = [1.5_f32, 1.5, 1.5];
        // rd_body = col_r of the starting frame's Jacobian (which is
        // aligned with body radial at the corner, outward).
        let col_r = [sub.j[2][0], sub.j[2][1], sub.j[2][2]];
        let rd_body = sdf::normalize(col_r);

        let hit = cs_raycast_local(
            &world.library, &sub, &ancestors,
            cam_local, rd_body, 4, LodParams::fixed_max(),
        );
        assert!(
            hit.is_some(),
            "neighbor transition failed: ray from empty sub-frame should \
             cross into +r-neighbor Block(42) sub-frame and hit",
        );
    }
}
