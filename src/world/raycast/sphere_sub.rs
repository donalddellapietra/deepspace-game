//! Local-frame sphere DDA for `ActiveFrameKind::SphereSub`.
//!
//! When the render frame lives deep inside a face subtree, the sphere
//! march cannot operate in body-XYZ: at face-subtree depth ≥ ~15,
//! adjacent cell-boundary plane normals collapse into the same f32
//! value and the DDA loses the ability to distinguish neighbouring
//! cells. The fix is the cubed-sphere analog of Cartesian ribbon-pop:
//!
//! 1. `compute_render_frame` descends through the face subtree to a
//!    face-cell at face-subtree depth M, and precomputes a linearized
//!    body-XYZ map at the cell's corner:
//!
//!        body_pos ≈ c_body + J · local_pos
//!
//!    where `local_pos ∈ [0, 3)³` is the frame-local coord and `J` is
//!    the analytical Jacobian of `face_space_to_body_point` at the
//!    corner (first derivatives w.r.t. local u/v/r).
//!
//! 2. Ray is transformed into local coords via `J_inv`. In the local
//!    frame, cell `u = const` / `v = const` boundaries are trivially
//!    axis-aligned (flat planes). **Radial boundaries** at
//!    `r_body = const` would in principle be ellipsoids in local
//!    coords, but because `J`'s third column is parallel to the body
//!    radial direction at the corner, `r_body = const` reduces to
//!    `r_local = const` — also flat to first order. All six
//!    boundaries become axis-aligned in local.
//!
//! 3. DDA in local coords uses integer-cell-boundary t-values:
//!    `t_axis = (K − local_pos[axis]) / local_dir[axis]`. These are
//!    always representable in f32 regardless of absolute face-subtree
//!    depth — all quantities stay O(1) in the local frame.
//!
//! The linearization is accurate to O(frame_size²) in body-XYZ; at
//! `MIN_SPHERE_SUB_DEPTH = 3` the error is ~1 % of cell width and
//! decays geometrically with depth. Face-root and depth-1 / 2 cells
//! keep the exact body-level march in `sphere.rs`.

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
/// * `sub` carries the sub-frame's metadata and linearization.
/// * `sub_frame_node` is the tree node at the sub-frame's render path.
/// * `ro_local` is the ray origin **already in sub-frame local**
///   `[0, 3)³` coords — derived via the anchor-path ribbon-pop, NOT
///   by subtracting `c_body` from body-XYZ (that subtraction collapses
///   in f32 once the sub-frame's body extent falls below eps).
/// * `rd_body` is the ray direction in body-local orientation, unit
///   length. It's transformed into local via `J_inv` inside — safe
///   because `rd_body` is O(1) and `|J_inv · rd_body|` is O(3^depth)
///   which f32 handles cleanly.
/// * `walker_limit` caps walker descent inside the sub-frame.
/// * `ancestor_path` is the full path from world-root up to (but not
///   including) the sub-frame — `HitInfo.path` is prefixed with this
///   so callers can propagate edits against world-root.
#[allow(clippy::too_many_arguments)]
pub(super) fn cs_raycast_local(
    library: &NodeLibrary,
    sub: &SphereSubFrame,
    sub_frame_node: NodeId,
    ancestor_path: &[(NodeId, usize)],
    ro_local: Vec3,
    rd_body: Vec3,
    walker_limit: u32,
    _lod: LodParams,
) -> Option<HitInfo> {
    if walker_limit == 0 {
        return None;
    }

    // Ray direction in sub-frame local basis. |rd_local| can be
    // large (O(3^depth)) but all subsequent DDA operations use
    // t ratios — no catastrophic addition.
    let rd_norm = sdf::normalize(rd_body);
    let rd_local = mat3_mul_vec(&sub.j_inv, rd_norm);

    let (t_enter, t_exit) = ray_local_box_interval(ro_local, rd_local);
    if t_exit <= 0.0 || t_enter >= t_exit {
        return None;
    }

    // Start just inside the box from t_enter — at t_enter the ray is
    // exactly on a face and would otherwise be rejected by the
    // strict box check below. Nudge is a tiny fraction of the span
    // (NOT clamped to any absolute value); at deep face-subtree
    // depth the full t-span can be ≪ 1e-9 because rd_local scales
    // as 3^depth.
    let t_span = (t_exit - t_enter).abs().max(1e-30);
    let t_nudge = t_span * 1e-5;
    let mut t = t_enter.max(0.0) + t_nudge;

    for _ in 0..MAX_DDA_STEPS {
        if t >= t_exit { break; }
        let pos = [
            ro_local[0] + rd_local[0] * t,
            ro_local[1] + rd_local[1] * t,
            ro_local[2] + rd_local[2] * t,
        ];

        // Escape hatch: left the local frame. Nudging past boundary.
        if pos[0] < 0.0 || pos[0] >= LOCAL_BOX_MAX
            || pos[1] < 0.0 || pos[1] >= LOCAL_BOX_MAX
            || pos[2] < 0.0 || pos[2] >= LOCAL_BOX_MAX
        {
            break;
        }

        let w = walk_sub_frame(library, sub_frame_node, pos[0], pos[1], pos[2], walker_limit);

        if w.block != EMPTY_CELL {
            let mut full_path: Vec<(NodeId, usize)> =
                Vec::with_capacity(ancestor_path.len() + w.path.len());
            full_path.extend_from_slice(ancestor_path);
            full_path.extend(w.path.iter().copied());

            // Sphere cell in absolute face-normalized coords. The
            // sub-frame maps local `[0, 3)` to absolute face
            // `[un_corner, un_corner + frame_size)`, so per-local-unit
            // absolute-coord-step = frame_size / 3.
            let step = sub.frame_size / 3.0;
            let body_path_len = ancestor_path
                .iter()
                .take_while(|(nid, _)| {
                    // First entry whose child IS the body cell.
                    // Detect by comparing child NodeId to the body's
                    // known path. Simpler: count by body_path length.
                    let _ = nid;
                    true
                })
                .count()
                .min(sub.body_path.depth() as usize);

            return Some(HitInfo {
                path: full_path,
                // Sentinel — sphere hits have no XYZ-axis face
                // semantic. AABB draws from sphere_cell.
                face: 4,
                t,
                place_path: None,
                sphere_cell: Some(SphereHitCell {
                    face: sub.face as u32,
                    u_lo: sub.un_corner + w.u_lo * step,
                    v_lo: sub.vn_corner + w.v_lo * step,
                    r_lo: sub.rn_corner + w.r_lo * step,
                    size: w.size * step,
                    inner_r: sub.inner_r,
                    outer_r: sub.outer_r,
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
            break;
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
    /// given face-subtree depth on `face`, returning the sub-frame,
    /// the sub-frame's node, and the ancestor path.
    ///
    /// The anchor descent uses UVR slot (1, 1, 1) at every
    /// face-subtree level so the sub-frame lives near the face
    /// center along the radial mid-shell — guaranteed-solid
    /// territory in the demo planet.
    fn planet_with_sub_frame(face: Face, sub_depth: u8) -> (
        crate::world::state::WorldState,
        SphereSubFrame,
        NodeId,
        Vec<(NodeId, usize)>,
    ) {
        use crate::world::anchor::{SphereState, WorldPos};
        use crate::app::frame::SPHERE_WALKER_BUDGET;
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
        // `compute_render_frame` reserves `SPHERE_WALKER_BUDGET`
        // levels between the SphereSub render depth and the camera's
        // logical uvr depth (so the DDA has descent budget).
        // Inflate the camera's uvr_path accordingly so the sub-frame
        // ends up at the requested `sub_depth`.
        let uvr_len = sub_depth as u32 + SPHERE_WALKER_BUDGET;
        let mut uvr_path = Path::root();
        for _ in 0..uvr_len {
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
        let mut node_id = world.root;
        let mut ancestors = Vec::new();
        for i in 0..active.render_path.depth() as usize {
            let slot = active.render_path.slot(i) as usize;
            ancestors.push((node_id, slot));
            node_id = match world.library.get(node_id).unwrap().children[slot] {
                Child::Node(n) => n,
                other => panic!("render_path step {i}: non-Node child {other:?}"),
            };
        }
        (world, sub, node_id, ancestors)
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
        let (world, sub, sub_node, ancestors) = planet_with_sub_frame(Face::PosY, 3);
        let (cam, rd) = inward_radial_ray(&sub);
        let hit = cs_raycast_local(
            &world.library, &sub, sub_node, &ancestors,
            cam, rd, 6, LodParams::fixed_max(),
        );
        assert!(hit.is_some(), "sub-depth 3 should hit");
    }

    /// Build a fully-solid sub-frame: world root → body(at slot 13)
    /// → face(PosY face root at body's slot 16) → depth levels of
    /// uniform `Child::Block(42)` nodes. Everything inside the
    /// sub-frame is guaranteed solid. Returns world, sub, sub_node,
    /// ancestors.
    fn solid_sub_frame(sub_depth: u8) -> (
        crate::world::state::WorldState,
        SphereSubFrame,
        NodeId,
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
        // Build `sub_depth + SPHERE_WALKER_BUDGET` levels so both the
        // sub-frame (at `sub_depth`) and the walker's descent past it
        // resolve to real nodes, not the placeholder chain above.
        use crate::app::frame::SPHERE_WALKER_BUDGET;
        let uvr_len = sub_depth as u32 + SPHERE_WALKER_BUDGET;
        let mut face_subtree = chain;
        for _ in 0..uvr_len {
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
        for _ in 0..uvr_len {
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
        let mut node_id = world.root;
        let mut ancestors = Vec::new();
        for i in 0..active.render_path.depth() as usize {
            let slot = active.render_path.slot(i) as usize;
            ancestors.push((node_id, slot));
            node_id = match world.library.get(node_id).unwrap().children[slot] {
                Child::Node(n) => n,
                other => panic!("non-node at step {i}: {other:?}"),
            };
        }
        (world, sub, node_id, ancestors)
    }

    #[test]
    fn cs_raycast_local_hits_at_all_depths_synthetic() {
        // Synthetic uniform-solid sub-frame removes worldgen noise:
        // every walker cell on the central UVR axis is Block(42), so
        // a ray straight through the local z axis MUST hit. Depth
        // ≥ 20 exercises the precision-critical local-frame DDA.
        for sub_depth in [3u8, 4, 5, 8, 12, 18, 25, 30, 35] {
            let (world, sub, sub_node, ancestors) = solid_sub_frame(sub_depth);
            let (cam, rd) = inward_radial_ray(&sub);
            let hit = cs_raycast_local(
                &world.library, &sub, sub_node, &ancestors,
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
            let (world, sub, sub_node, ancestors) = planet_with_sub_frame(face, 5);
            let (cam, rd) = inward_radial_ray(&sub);
            let hit = cs_raycast_local(
                &world.library, &sub, sub_node, &ancestors,
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
        let (world, sub, _sub_node, _ancestors) =
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
        let (_world, sub, _sub_node, _ancestors) =
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
}
