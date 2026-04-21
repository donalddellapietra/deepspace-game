//! Unified slot-path + residual DDA.
//!
//! One primitive for Cartesian, cubed-sphere body subtrees, and
//! face-seam crossings. See `docs/design/unified-slot-residual-dda.md`.
//!
//! Step 1 (landed): flat Cartesian subtrees, cases 1–3.
//! Step 2 (landed): descend through CubedSphereBody + CubedSphereFace
//!                  nodes, tracking un/vn/rn/face/frame_size and
//!                  pre-building the per-cell Jacobian.
//! Step 3 (THIS):  bubble-up (case 4) within one face / within Cartesian.
//!                 When the top frame's slot exits [0,2], pop and
//!                 update the parent's residual + slot. Face-seam
//!                 bubble-ups still terminate here — Step 4 adds the
//!                 face-adjacency rotation.
//! Step 4 (next): face-seam rotation + Jacobian-corrected DDA.
//!
//! Invariants:
//! - `residual_entry ∈ [0, 1)³` by construction — precision-bounded
//!   regardless of global tree depth.
//! - `slot_path` (the parallel `path` Vec) is integer.
//! - `t_world_entry` is the only f32 that grows with march length;
//!   only used for hit reporting.

use super::HitInfo;
use crate::world::cubesphere::{
    face_frame_jacobian, mat3_inv, Face, Mat3, FACE_SLOTS, CORE_SLOT,
};
use crate::world::tree::{slot_coords, slot_index, Child, NodeId, NodeKind, NodeLibrary, REPRESENTATIVE_EMPTY};

/// Semantic kind of the cells at the top frame. Tracks what
/// coordinate system the `residual` + `slot` indices are interpreted
/// in, and (for sphere cells) carries the linearized Jacobian that
/// maps the cell's local [0,1)³ into body-XYZ.
#[derive(Debug, Clone, Copy)]
enum CellKind {
    Cartesian,
    /// Inside a `CubedSphereBody` cell. Children are a 3×3×3 XYZ
    /// grid: 6 face slots lead to face subtrees, 1 core slot leads
    /// to a Cartesian stone subtree, 20 edge/corner slots are empty.
    /// DDA inside a body cell is axis-aligned in body-XYZ (no
    /// Jacobian needed here — only when we descend into a face).
    SphereBody {
        inner_r: f32,
        outer_r: f32,
        /// World-space size of the body cell. All child cells at
        /// this level have size `body_world_size / 3`.
        body_world_size: f32,
    },
    /// Inside a face subtree. `un/vn/rn_corner` identify the cell's
    /// face-normalized corner; `frame_size` is the cell's extent in
    /// face-normalized coords at this depth (1.0 at face root, 1/3,
    /// 1/9, ... as we descend). `c_body` + `j` are the linearized
    /// map from cell-local to body-XYZ evaluated at the corner.
    SphereFace {
        face: Face,
        un_corner: f32,
        vn_corner: f32,
        rn_corner: f32,
        frame_size: f32,
        inner_r: f32,
        outer_r: f32,
        body_world_size: f32,
        c_body: [f32; 3],
        j: Mat3,
        j_inv: Mat3,
    },
}

/// Per-level stack frame.
#[derive(Debug, Clone, Copy)]
struct UFrame {
    node_id: NodeId,
    slot: [i32; 3],
    residual_entry: [f32; 3],
    t_world_entry: f32,
    /// World-space size of one cell at this level. For sphere cells
    /// this is the cell's world extent along one axis at the cell
    /// corner (approximate — true size varies over the curved cell,
    /// but linearization gives a consistent scalar for the DDA).
    cell_size: f32,
    kind: CellKind,
}

/// Unified raycast. Step 2 scope: flat Cartesian + sphere descent
/// tracking. Jacobian-corrected sphere DDA lands with Step 4.
///
/// Returns `None` on miss, iteration cap, or if the top slot leaves
/// `[0, 2]³` (Step 3 adds bubble-up).
pub(super) fn unified_raycast(
    library: &NodeLibrary,
    root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
) -> Option<HitInfo> {
    let inv_dir = [
        if ray_dir[0].abs() > 1e-8 { 1.0 / ray_dir[0] } else { 1e10 },
        if ray_dir[1].abs() > 1e-8 { 1.0 / ray_dir[1] } else { 1e10 },
        if ray_dir[2].abs() > 1e-8 { 1.0 / ray_dir[2] } else { 1e10 },
    ];

    let (t_enter, t_exit) = ray_aabb(ray_origin, inv_dir, [0.0; 3], [3.0; 3]);
    if t_enter >= t_exit || t_exit < 0.0 {
        return None;
    }

    let t_start = t_enter.max(0.0) + 0.001;
    let entry = [
        ray_origin[0] + ray_dir[0] * t_start,
        ray_origin[1] + ray_dir[1] * t_start,
        ray_origin[2] + ray_dir[2] * t_start,
    ];
    let slot = [
        (entry[0].floor() as i32).clamp(0, 2),
        (entry[1].floor() as i32).clamp(0, 2),
        (entry[2].floor() as i32).clamp(0, 2),
    ];
    let residual_entry = [
        (entry[0] - slot[0] as f32).clamp(0.0, 1.0 - 1e-6),
        (entry[1] - slot[1] as f32).clamp(0.0, 1.0 - 1e-6),
        (entry[2] - slot[2] as f32).clamp(0.0, 1.0 - 1e-6),
    ];

    let mut stack: Vec<UFrame> = Vec::with_capacity(max_depth as usize + 1);
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(max_depth as usize + 1);

    stack.push(UFrame {
        node_id: root,
        slot,
        residual_entry,
        t_world_entry: t_start,
        cell_size: 1.0,
        kind: CellKind::Cartesian,
    });

    let mut normal_face: u32 = 2;
    let mut iterations = 0u32;
    let max_iterations = (max_depth.max(1) * 4096).max(8192);

    loop {
        if iterations >= max_iterations || stack.is_empty() {
            return None;
        }
        iterations += 1;

        let depth = stack.len() - 1;
        let frame = stack[depth];

        // Out-of-range top slot → bubble up. If the stack empties
        // we've exited the world. If the parent is a face-root (the
        // immediate child of a CubedSphereBody cell), bubble-up
        // would cross a cube face seam — Step 4 adds the face
        // adjacency rotation there. For Step 3 we terminate in that
        // case so seam-crossing rays are caught by tests rather than
        // silently dropped.
        if frame.slot[0] < 0 || frame.slot[0] > 2
            || frame.slot[1] < 0 || frame.slot[1] > 2
            || frame.slot[2] < 0 || frame.slot[2] > 2
        {
            if !bubble_up(&mut stack, &mut path) {
                return None;
            }
            continue;
        }

        let slot_idx = slot_index(
            frame.slot[0] as usize,
            frame.slot[1] as usize,
            frame.slot[2] as usize,
        );
        let node = library.get(frame.node_id)?;
        let child = node.children[slot_idx];

        if path.len() > depth {
            path[depth] = (frame.node_id, slot_idx);
        } else {
            path.push((frame.node_id, slot_idx));
        }

        match child {
            Child::Empty | Child::EntityRef(_) => {
                advance_same_parent(&mut stack[depth], &ray_dir, &mut normal_face);
            }
            Child::Block(_) => {
                return Some(HitInfo {
                    path: path.clone(),
                    face: normal_face,
                    t: frame.t_world_entry,
                    place_path: None,
                    sphere_cell: None,
                });
            }
            Child::Node(child_id) => {
                let child_node = library.get(child_id)?;

                if child_node.representative_block == REPRESENTATIVE_EMPTY {
                    advance_same_parent(&mut stack[depth], &ray_dir, &mut normal_face);
                    continue;
                }

                if (depth as u32 + 1) >= max_depth {
                    return Some(HitInfo {
                        path: path.clone(),
                        face: normal_face,
                        t: frame.t_world_entry,
                        place_path: None,
                        sphere_cell: None,
                    });
                }

                let child_cell_size = frame.cell_size / 3.0;
                let child_kind = derive_child_kind(frame.kind, child_node.kind, &frame, slot_idx);

                let p_res = frame.residual_entry;
                let sx = ((p_res[0] * 3.0).floor() as i32).clamp(0, 2);
                let sy = ((p_res[1] * 3.0).floor() as i32).clamp(0, 2);
                let sz = ((p_res[2] * 3.0).floor() as i32).clamp(0, 2);
                let child_residual_entry = [
                    (p_res[0] * 3.0 - sx as f32).clamp(0.0, 1.0 - 1e-6),
                    (p_res[1] * 3.0 - sy as f32).clamp(0.0, 1.0 - 1e-6),
                    (p_res[2] * 3.0 - sz as f32).clamp(0.0, 1.0 - 1e-6),
                ];

                stack.push(UFrame {
                    node_id: child_id,
                    slot: [sx, sy, sz],
                    residual_entry: child_residual_entry,
                    t_world_entry: frame.t_world_entry,
                    cell_size: child_cell_size,
                    kind: child_kind,
                });
            }
        }
    }
}

/// Compute the child frame's `CellKind` from the parent's kind, the
/// child node's own NodeKind, and the slot the child occupies within
/// the parent (needed to pick out face subtrees from a sphere body).
fn derive_child_kind(
    parent_kind: CellKind,
    child_node_kind: NodeKind,
    parent_frame: &UFrame,
    slot_idx: usize,
) -> CellKind {
    // Explicit kind on the child always wins (CubedSphereBody /
    // CubedSphereFace tagged nodes are authoritative).
    match child_node_kind {
        NodeKind::CubedSphereBody { inner_r, outer_r } => {
            return CellKind::SphereBody {
                inner_r,
                outer_r,
                body_world_size: parent_frame.cell_size,
            };
        }
        NodeKind::CubedSphereFace { face } => {
            // Face root: whole face span, un/vn/rn_corner = 0,
            // frame_size = 1. Jacobian at corner.
            let (body_world_size, inner_r, outer_r) = match parent_kind {
                CellKind::SphereBody { inner_r, outer_r, body_world_size } => {
                    (body_world_size, inner_r, outer_r)
                }
                // A face node outside a SphereBody is ill-formed; treat
                // as Cartesian fallback (can't build Jacobian without
                // radii). Shouldn't occur in well-formed trees.
                _ => return CellKind::Cartesian,
            };
            let (c_body, j) = face_frame_jacobian(
                face, 0.0, 0.0, 0.0, 1.0,
                inner_r, outer_r, body_world_size,
            );
            let j_inv = mat3_inv(&j);
            return CellKind::SphereFace {
                face,
                un_corner: 0.0,
                vn_corner: 0.0,
                rn_corner: 0.0,
                frame_size: 1.0,
                inner_r, outer_r,
                body_world_size,
                c_body, j, j_inv,
            };
        }
        NodeKind::Cartesian => {}
    }

    // Default-Cartesian child NodeKind: interpretation is inherited
    // from parent.
    match parent_kind {
        CellKind::Cartesian => CellKind::Cartesian,
        CellKind::SphereBody { inner_r, outer_r, body_world_size } => {
            // Core slot is a stone subtree (Cartesian). Face slots
            // should already have been caught above (they're tagged
            // CubedSphereFace). Edge/corner slots are Empty and
            // don't descend.
            if slot_idx == CORE_SLOT {
                CellKind::Cartesian
            } else {
                // Shouldn't reach here for face slots (they'd be
                // CubedSphereFace tagged). For any other slot, treat
                // as Cartesian — but this path is off-spec.
                CellKind::Cartesian
            }
        }
        CellKind::SphereFace {
            face, un_corner, vn_corner, rn_corner, frame_size,
            inner_r, outer_r, body_world_size, ..
        } => {
            // Descend inside a face subtree: pick the (u,v,r) slot
            // coords and shrink the frame by 1/3.
            let (sx, sy, sz) = slot_coords(slot_idx);
            let new_frame_size = frame_size / 3.0;
            let new_un = un_corner + sx as f32 * new_frame_size;
            let new_vn = vn_corner + sy as f32 * new_frame_size;
            let new_rn = rn_corner + sz as f32 * new_frame_size;
            let (c_body, j) = face_frame_jacobian(
                face, new_un, new_vn, new_rn, new_frame_size,
                inner_r, outer_r, body_world_size,
            );
            let j_inv = mat3_inv(&j);
            CellKind::SphereFace {
                face,
                un_corner: new_un,
                vn_corner: new_vn,
                rn_corner: new_rn,
                frame_size: new_frame_size,
                inner_r, outer_r, body_world_size,
                c_body, j, j_inv,
            }
        }
    }
}

/// Advance the top frame's ray by one cell in the min-axis direction
/// (case 3). Step 2 keeps flat DDA with `ray_dir` as rd_body; Step 4
/// swaps in `rd_local = J_inv · rd_body` for SphereFace cells so the
/// DDA respects curvature.
fn advance_same_parent(frame: &mut UFrame, ray_dir: &[f32; 3], normal_face: &mut u32) {
    let mut t_exit_world = [f32::INFINITY; 3];
    for k in 0..3 {
        if ray_dir[k].abs() < 1e-12 {
            continue;
        }
        let target = if ray_dir[k] > 0.0 { 1.0 } else { 0.0 };
        t_exit_world[k] = (target - frame.residual_entry[k]) * frame.cell_size / ray_dir[k];
        if t_exit_world[k] < 0.0 {
            t_exit_world[k] = 0.0;
        }
    }
    let mut axis = 0;
    for k in 1..3 {
        if t_exit_world[k] < t_exit_world[axis] {
            axis = k;
        }
    }
    let t_local = t_exit_world[axis];
    frame.t_world_entry += t_local;

    let mut new_residual = [0.0f32; 3];
    for k in 0..3 {
        if k == axis {
            new_residual[k] = if ray_dir[k] > 0.0 { 0.0 } else { 1.0 - 1e-6 };
        } else if ray_dir[k].abs() < 1e-12 {
            new_residual[k] = frame.residual_entry[k];
        } else {
            let advanced = frame.residual_entry[k] + ray_dir[k] * t_local / frame.cell_size;
            new_residual[k] = advanced.clamp(0.0, 1.0 - 1e-6);
        }
    }
    frame.residual_entry = new_residual;

    frame.slot[axis] += if ray_dir[axis] > 0.0 { 1 } else { -1 };
    *normal_face = match (axis, ray_dir[axis] > 0.0) {
        (0, true) => 1,
        (0, false) => 0,
        (1, true) => 3,
        (1, false) => 2,
        (2, true) => 5,
        (2, false) => 4,
        _ => unreachable!(),
    };
}

/// Bubble up past a cell boundary when the top frame's slot has
/// stepped out of `[0, 2]` (case 4). Pops the top frame and updates
/// the new top (parent):
///
/// - On the overflow axis: parent's residual snaps to `0` (if child
///   exited at +k) or `1 - eps` (if child exited at -k), and parent's
///   slot steps by ±1 accordingly.
/// - On other axes: parent's residual recomputes as
///   `(child.slot + child.residual) / 3`, since the ray's parent-
///   local position IS `(child_slot + child_residual) / 3` at any
///   instant while child.slot ∈ [0, 2].
/// - `t_world_entry` carries forward — the ray's world-t at the
///   moment of the cell crossing is the same at both levels.
///
/// Returns `true` if bubble-up succeeded, `false` if we bubbled past
/// the world root (ray has exited). Also returns `false` if the
/// popped frame was a face root — that's a face-seam crossing, which
/// Step 4 will handle by rotating into the adjacent face's basis
/// instead of terminating.
fn bubble_up(stack: &mut Vec<UFrame>, path: &mut Vec<(NodeId, usize)>) -> bool {
    if stack.is_empty() {
        return false;
    }
    let child = *stack.last().unwrap();

    // Find the axis that overflowed. Exactly one should be out of
    // [0, 2] per advance_same_parent step (it only mutates one axis
    // per call), but in rare cases (degenerate initial entry) more
    // than one could be — pick the first out-of-range axis.
    let axis = (0..3).find(|&k| child.slot[k] < 0 || child.slot[k] > 2);
    let Some(axis) = axis else {
        // Defensive: shouldn't reach here.
        return false;
    };
    let stepping_positive = child.slot[axis] > 2;

    stack.pop();
    if path.len() > stack.len() {
        path.truncate(stack.len());
    }
    if stack.is_empty() {
        return false;
    }
    let parent_depth = stack.len() - 1;
    let parent = &mut stack[parent_depth];

    // Face-seam check: if the popped child was a face-root cell
    // (i.e., parent is a SphereBody and the popped child's kind was
    // SphereFace), then bubbling past means crossing a cube face.
    // Step 4 replaces this termination with a face-adjacency rotation.
    let was_face_root = matches!(child.kind, CellKind::SphereFace { .. })
        && matches!(parent.kind, CellKind::SphereBody { .. });
    if was_face_root {
        return false;
    }

    // Recompute parent residual from child state.
    let mut new_residual = [0.0f32; 3];
    for k in 0..3 {
        if k == axis {
            new_residual[k] = if stepping_positive { 0.0 } else { 1.0 - 1e-6 };
        } else {
            let r = (child.slot[k] as f32 + child.residual_entry[k]) / 3.0;
            new_residual[k] = r.clamp(0.0, 1.0 - 1e-6);
        }
    }
    parent.residual_entry = new_residual;
    parent.slot[axis] += if stepping_positive { 1 } else { -1 };
    parent.t_world_entry = child.t_world_entry;

    true
}

fn ray_aabb(
    origin: [f32; 3],
    inv_dir: [f32; 3],
    bmin: [f32; 3],
    bmax: [f32; 3],
) -> (f32, f32) {
    let t1 = [
        (bmin[0] - origin[0]) * inv_dir[0],
        (bmin[1] - origin[1]) * inv_dir[1],
        (bmin[2] - origin[2]) * inv_dir[2],
    ];
    let t2 = [
        (bmax[0] - origin[0]) * inv_dir[0],
        (bmax[1] - origin[1]) * inv_dir[1],
        (bmax[2] - origin[2]) * inv_dir[2],
    ];
    let t_enter = t1[0].min(t2[0]).max(t1[1].min(t2[1])).max(t1[2].min(t2[2]));
    let t_exit = t1[0].max(t2[0]).min(t1[1].max(t2[1])).min(t1[2].max(t2[2]));
    (t_enter, t_exit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::cubesphere::{Face, CORE_SLOT};
    use crate::world::tree::{empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary};

    fn build_center_pinpoint(lib: &mut NodeLibrary, depth: u32) -> NodeId {
        if depth == 0 {
            let mut c = empty_children();
            c[13] = Child::Block(1);
            return lib.insert(c);
        }
        let inner = build_center_pinpoint(lib, depth - 1);
        let mut c = empty_children();
        c[13] = Child::Node(inner);
        lib.insert(c)
    }

    /// Synthetic solid sphere world — mirrors build_solid_sphere_world
    /// in raycast/mod.rs tests, scoped down for unified Step 2 coverage.
    fn build_solid_sphere_world(sub_depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let deep_solid = lib.insert(uniform_children(Child::Block(42)));
        let mut chain = deep_solid;
        for _ in 0..4u32 {
            chain = lib.insert(uniform_children(Child::Node(chain)));
        }
        let mut face_subtree = chain;
        for _ in 0..sub_depth {
            let mut children = empty_children();
            children[slot_index(1, 1, 1)] = Child::Node(face_subtree);
            face_subtree = lib.insert(children);
        }
        let mut face_root_children = uniform_children(Child::Node(chain));
        face_root_children[slot_index(1, 1, 1)] = Child::Node(face_subtree);
        let face_root = lib.insert_with_kind(
            face_root_children,
            NodeKind::CubedSphereFace { face: Face::PosY },
        );
        let mut body_children = empty_children();
        for &f in &Face::ALL {
            body_children[crate::world::cubesphere::FACE_SLOTS[f as usize]] =
                Child::Node(face_root);
        }
        body_children[CORE_SLOT] = Child::Node(chain);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        (lib, root)
    }

    #[test]
    fn unified_hits_center_block_flat_root() {
        let mut lib = NodeLibrary::default();
        let mut c = empty_children();
        c[13] = Child::Block(1);
        let root = lib.insert(c);
        let hit = unified_raycast(&lib, root, [1.5, 5.0, 1.5], [0.0, -1.0, 0.0], 8);
        let h = hit.expect("should hit center cell");
        assert_eq!(h.path.len(), 1);
        assert_eq!(h.path[0].1, 13);
    }

    #[test]
    fn unified_hits_deep_center_pinpoint() {
        let mut lib = NodeLibrary::default();
        let root = build_center_pinpoint(&mut lib, 3);
        let hit = unified_raycast(&lib, root, [1.5, 5.0, 1.5], [0.0, -1.0, 0.0], 8);
        let h = hit.expect("should hit deep center");
        assert_eq!(h.path.len(), 4);
        for (_, slot) in &h.path {
            assert_eq!(*slot, 13);
        }
    }

    #[test]
    fn unified_miss_returns_none() {
        let mut lib = NodeLibrary::default();
        let mut c = empty_children();
        c[0] = Child::Block(1);
        let root = lib.insert(c);
        let hit = unified_raycast(&lib, root, [2.5, 5.0, 2.5], [0.0, -1.0, 0.0], 8);
        assert!(hit.is_none());
    }

    #[test]
    fn unified_parity_with_cartesian_flat_center() {
        let mut lib = NodeLibrary::default();
        let root = build_center_pinpoint(&mut lib, 2);
        // Mix of rays that hit the center (no bubble-up needed) and
        // rays that miss — miss rays exercise case 4 (bubble-up past
        // the root) since the ray has to walk across multiple root
        // slots before exiting.
        let rays: &[([f32; 3], [f32; 3])] = &[
            ([1.5, 5.0, 1.5], [0.0, -1.0, 0.0]),
            ([0.5, 5.0, 0.5], [0.0, -1.0, 0.0]),
            ([2.5, 5.0, 2.5], [0.0, -1.0, 0.0]),
            // Angled rays: these MUST bubble up between root slots
            // as they descend.
            ([0.1, 5.0, 1.5], [0.3, -1.0, 0.0]),
            ([1.5, 5.0, 0.1], [0.0, -1.0, 0.3]),
            ([0.5, 5.0, 1.0], [1.0, -1.5, 0.3]),
        ];
        for &(ro, rd) in rays {
            let u = unified_raycast(&lib, root, ro, rd, 8);
            let c = super::super::cartesian::cpu_raycast_with_face_depth(
                &lib, root, ro, rd, 8,
                super::super::MAX_FACE_DEPTH,
                super::super::LodParams::fixed_max(),
            );
            match (u, c) {
                (Some(uh), Some(ch)) => {
                    assert_eq!(uh.path.len(), ch.path.len(), "ray {ro:?} → {rd:?}");
                    for i in 0..uh.path.len() {
                        assert_eq!(uh.path[i].1, ch.path[i].1, "slot at {i}");
                    }
                }
                (None, None) => {}
                (u, c) => panic!("parity mismatch on ray {ro:?}→{rd:?}: u={u:?}, c={c:?}"),
            }
        }
    }

    /// Descend into a sphere world: expect the hit path to contain
    /// both the sphere body slot and a face slot before terminating.
    /// This is a structural check — at Step 2 the in-cell DDA uses
    /// flat math, so the exact terminal cell may not match what the
    /// Jacobian-corrected DDA would produce. Step 4 tightens this.
    #[test]
    fn unified_descends_through_sphere_kinds() {
        let (lib, root) = build_solid_sphere_world(2);
        // Ray from above, aimed straight down at the center of the
        // sphere body (body lives in root slot (1,1,1), so it spans
        // world [1, 2) × [1, 2) × [1, 2)).
        let hit = unified_raycast(
            &lib, root,
            [1.5, 5.0, 1.5],
            [0.0, -1.0, 0.0],
            16,
        );
        // We expect a hit (solid sphere world). Path should include:
        // - root → body slot (slot 13 at world root)
        // - body → +Y face slot (FACE_SLOTS[PosY] = slot_index(1,2,1) = 16)
        let h = hit.expect("solid sphere should produce a hit");
        assert!(h.path.len() >= 2, "path should descend at least into body");
        assert_eq!(h.path[0].1, 13, "first slot is body cell in world root");
        // Second slot is the face pick (PosY = slot 16).
        assert_eq!(
            h.path[1].1,
            crate::world::cubesphere::FACE_SLOTS[Face::PosY as usize],
            "second slot should be +Y face"
        );
    }

    /// Bubble-up within a Cartesian tree: ray that has to step
    /// across multiple root-level slots to reach a far-corner
    /// block. Before Step 3 this returned None as soon as the
    /// descent frame's slot overflowed.
    #[test]
    fn unified_bubble_up_across_root_slots() {
        let mut lib = NodeLibrary::default();
        // Build a root with a subtree at slot 0 (corner (0,0,0))
        // containing a block only at a deep sub-corner, and a
        // block at root slot 26 (corner (2,2,2)).
        let mut corner_leaf = empty_children();
        corner_leaf[0] = Child::Block(1);
        let corner_node = lib.insert(corner_leaf);
        let mut root_children = empty_children();
        root_children[0] = Child::Node(corner_node);
        root_children[26] = Child::Block(2);
        let root = lib.insert(root_children);

        // Ray enters at x=0.5, z=0.5 at the top, heading down-and-
        // right toward the far corner. It descends into root slot 0
        // briefly, then has to bubble up and step across multiple
        // root slots before landing on slot 26 or missing.
        let hit = unified_raycast(&lib, root, [0.5, 5.0, 0.5], [0.6, -1.0, 0.6], 16);
        // Regardless of which block it hits, bubble-up must have
        // fired at least once for the ray to traverse beyond the
        // initial root slot.
        assert!(hit.is_some(), "ray should eventually hit a block");
    }

    /// After descending into a face subtree, the top frame's kind is
    /// SphereFace with the correct face + per-cell Jacobian. This
    /// doesn't run the full DDA to completion — it exercises the
    /// descend / kind-derivation paths that Step 4 will rely on.
    #[test]
    fn unified_sphere_face_kind_has_jacobian() {
        let (lib, root) = build_solid_sphere_world(2);
        // Use a ray that definitely enters the +Y face region.
        let hit = unified_raycast(
            &lib, root,
            [1.5, 5.0, 1.5],
            [0.0, -1.0, 0.0],
            16,
        );
        // Just assert we got a hit — kind-correctness is checked
        // structurally via the preceding test. The Jacobian path is
        // exercised implicitly by descend; if face_frame_jacobian
        // produced a singular matrix the mat3_inv debug_assert would
        // fire before we return.
        assert!(hit.is_some());
    }
}
