//! Unified slot-path + residual DDA.
//!
//! One primitive for Cartesian, cubed-sphere body subtrees, and
//! face-seam crossings. See `docs/design/unified-slot-residual-dda.md`.
//!
//! Step 1 scope (THIS FILE): flat Cartesian subtrees only. Cases
//! 1–3 (descend, advance-in-cell, same-parent neighbor). Bubble-up
//! (case 4), sphere face-subtree, and face-seam rotation land in
//! later steps.
//!
//! Invariants (verified per iteration):
//!
//! - `slot` at the top frame stays in `[0, 2]`³ while the DDA is
//!   active. Exit of this range means we'd need case 4 (bubble-up);
//!   Step 1 terminates the march instead.
//! - `residual_entry ∈ [0, 1)³` by construction. All DDA arithmetic
//!   inside a cell operates on residuals, so f32 precision stays
//!   bounded regardless of global tree depth.
//! - `slot_path` (the parallel `path` Vec here) is integer; there is
//!   no f32 coordinate that accumulates across tree depth.
//! - `t_world_entry` accumulates via `t_world += t_local * cell_size`
//!   per step. This is the one f32 that grows with march length;
//!   it's used only for hit reporting, never for geometry.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary, REPRESENTATIVE_EMPTY};

/// Per-level stack frame. Mirrors `cartesian::Frame` but replaces
/// `(side_dist, node_origin)` with `(residual_entry, t_world_entry)`.
/// The invariant here is precision-bounded: `residual_entry` stays
/// in `[0, 1)³`, so cell-level DDA arithmetic doesn't accumulate
/// error with tree depth.
#[derive(Debug, Clone, Copy)]
struct UFrame {
    node_id: NodeId,
    /// Which child of `node_id` we're currently at. Each component
    /// stays in `[0, 2]` while the march is active at this frame.
    slot: [i32; 3],
    /// Where the ray entered the CURRENT cell, in cell-local coords.
    /// Each component ∈ `[0, 1)` — bounded by construction.
    residual_entry: [f32; 3],
    /// World-t value at the moment the ray entered this cell.
    /// Only used for hit reporting (sort/compare), never for geometry.
    t_world_entry: f32,
    /// World-space size of one cell at this level. Equals
    /// `parent.cell_size / 3` for a child frame.
    cell_size: f32,
}

/// Unified Cartesian raycast. Parity target: `cartesian::cpu_raycast_with_face_depth`
/// on flat Cartesian trees (no `CubedSphereBody` nodes).
///
/// Returns `None` if:
/// - The ray misses the root `[0, 3)³` box, or
/// - It reaches the iteration cap, or
/// - The top slot goes out of `[0, 2]³` (Step 1 limitation — Step 3
///   adds bubble-up), or
/// - It encounters a non-Cartesian `NodeKind` (Step 2 adds sphere).
pub(super) fn unified_raycast_cartesian(
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
    // Residual entry into root-level 1×1×1 child cell.
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

        // Case 4 (bubble-up) is Step 3. For Step 1 we terminate on
        // out-of-range top slot. No shim — we let the march fail so
        // tests catch rays that need case 4.
        if frame.slot[0] < 0 || frame.slot[0] > 2
            || frame.slot[1] < 0 || frame.slot[1] > 2
            || frame.slot[2] < 0 || frame.slot[2] > 2
        {
            return None;
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

                // Step 2 will handle CubedSphereBody/Face. For now
                // reject — no shim, per "no shortcuts" memory.
                match child_node.kind {
                    NodeKind::Cartesian => {}
                    _ => return None,
                }

                // Empty-subtree skip (mirror cartesian.rs:161).
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

                // Descend. Child's entry residual comes from scaling
                // the parent's current residual by 3× and picking the
                // integer slot.
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
                    cell_size: frame.cell_size / 3.0,
                });
            }
        }
    }
}

/// Advance the top frame's ray by one cell in the min-axis direction
/// (same-parent neighbor step, case 3). Does not handle bubble-up —
/// caller (the main loop) checks `frame.slot` for out-of-range.
fn advance_same_parent(frame: &mut UFrame, ray_dir: &[f32; 3], normal_face: &mut u32) {
    // t-exit per axis in world-t units, computed from cell-local
    // residual. All operands are [0,1)-bounded; product with
    // cell_size / ray_dir is a single scalar multiply — f32-safe at
    // any tree depth.
    let mut t_exit_world = [f32::INFINITY; 3];
    for k in 0..3 {
        if ray_dir[k].abs() < 1e-12 {
            continue;
        }
        let target = if ray_dir[k] > 0.0 { 1.0 } else { 0.0 };
        t_exit_world[k] = (target - frame.residual_entry[k]) * frame.cell_size / ray_dir[k];
        if t_exit_world[k] < 0.0 {
            // Residual was already on the boundary going the wrong
            // way; treat as zero-advance crossing.
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

    // Update residual: snap exit axis, advance others.
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
    use crate::world::tree::{empty_children, Child, NodeKind, NodeLibrary};

    /// Build a 3-deep Cartesian tree with a single Block at the
    /// center child (slot 13) at every level. Any ray into [0,3)³
    /// from outside that touches the center 1×1×1 at root, then
    /// center 1/3×1/3×1/3 at depth 1, etc., should hit.
    fn build_center_pinpoint(lib: &mut NodeLibrary, depth: u32) -> NodeId {
        if depth == 0 {
            // leaf: a node whose center slot is a Block.
            let mut c = empty_children();
            c[13] = Child::Block(1);
            return lib.insert(c);
        }
        let inner = build_center_pinpoint(lib, depth - 1);
        let mut c = empty_children();
        c[13] = Child::Node(inner);
        lib.insert(c)
    }

    #[test]
    fn unified_hits_center_block_flat_root() {
        let mut lib = NodeLibrary::default();
        // Single root node: center slot is a Block.
        let mut c = empty_children();
        c[13] = Child::Block(1);
        let root = lib.insert(c);

        // Ray from (1.5, 5.0, 1.5) aiming straight down — should
        // hit the center 1×1×1 cell at root.
        let hit = unified_raycast_cartesian(
            &lib,
            root,
            [1.5, 5.0, 1.5],
            [0.0, -1.0, 0.0],
            8,
        );
        let h = hit.expect("should hit center cell");
        assert_eq!(h.path.len(), 1);
        assert_eq!(h.path[0].1, 13); // slot 13 = (1,1,1)
    }

    #[test]
    fn unified_hits_deep_center_pinpoint() {
        let mut lib = NodeLibrary::default();
        let root = build_center_pinpoint(&mut lib, 3);

        // Ray through the center column: enters at y=3, exits at y=0.
        // At every level the center slot (1,1,1) carries the column,
        // so a straight-down ray at (1.5, 5, 1.5) should reach the
        // deepest block.
        let hit = unified_raycast_cartesian(
            &lib,
            root,
            [1.5, 5.0, 1.5],
            [0.0, -1.0, 0.0],
            8,
        );
        let h = hit.expect("should hit deep center");
        assert_eq!(h.path.len(), 4, "should descend 4 levels");
        for (_, slot) in &h.path {
            assert_eq!(*slot, 13);
        }
    }

    #[test]
    fn unified_miss_returns_none() {
        let mut lib = NodeLibrary::default();
        // Root has one Block at corner (0,0,0).
        let mut c = empty_children();
        c[0] = Child::Block(1);
        let root = lib.insert(c);

        // Ray aimed at the opposite corner.
        let hit = unified_raycast_cartesian(
            &lib,
            root,
            [2.5, 5.0, 2.5],
            [0.0, -1.0, 0.0],
            8,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn unified_parity_with_cartesian_flat_center() {
        // Parity against cartesian::cpu_raycast_with_face_depth on
        // a small flat tree. Sample several rays, compare hit path.
        let mut lib = NodeLibrary::default();
        let root = build_center_pinpoint(&mut lib, 2);

        let rays = [
            ([1.5f32, 5.0, 1.5], [0.0f32, -1.0, 0.0]),
            ([0.5, 5.0, 0.5], [0.0, -1.0, 0.0]),
            ([2.5, 5.0, 2.5], [0.0, -1.0, 0.0]),
        ];
        for (ro, rd) in rays {
            let u = unified_raycast_cartesian(&lib, root, ro, rd, 8);
            let c = super::super::cartesian::cpu_raycast_with_face_depth(
                &lib,
                root,
                ro,
                rd,
                8,
                super::super::MAX_FACE_DEPTH,
                super::super::LodParams::fixed_max(),
            );
            match (u, c) {
                (Some(uh), Some(ch)) => {
                    assert_eq!(uh.path.len(), ch.path.len(), "ray {ro:?} → {rd:?}");
                    for i in 0..uh.path.len() {
                        assert_eq!(uh.path[i].1, ch.path[i].1, "ray {ro:?} → {rd:?} slot at {i}");
                    }
                }
                (None, None) => {}
                (u, c) => panic!("parity mismatch on ray {ro:?}→{rd:?}: unified={u:?}, cartesian={c:?}"),
            }
        }
    }
}
