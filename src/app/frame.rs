//! Render-frame helpers: walking the camera path to find the
//! frame root, transforming AABBs between world and frame coords.
//!
//! The "render frame" is the GPU's view of the world. The shader
//! starts ray marching at the frame root, with the camera in
//! frame-local `[0, 3)³` coordinates. The `WorldPos.in_frame`
//! method gives the camera those coords; this module gives the
//! tree walker that picks the frame and the AABB transforms used
//! by the highlight overlay.
//!
//! All functions here are **pure** — no `App` state — for direct
//! unit testing.

use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

/// World-space origin and side length of the cell at `path`.
/// Origin is the world XYZ of the cell's `(0, 0, 0)` corner; size
/// is the cell's side length in world units.
///
/// Used to transform between world coords and frame-local shader
/// coords (which map the cell to `[0, WORLD_SIZE)³`).
pub fn frame_origin_size_world(path: &Path) -> ([f32; 3], f32) {
    let mut origin = [0.0f32; 3];
    let mut size = WORLD_SIZE;
    for k in 0..path.depth() as usize {
        let (sx, sy, sz) = slot_coords(path.slot(k) as usize);
        let child = size / 3.0;
        origin[0] += sx as f32 * child;
        origin[1] += sy as f32 * child;
        origin[2] += sz as f32 * child;
        size = child;
    }
    (origin, size)
}

/// Transform an axis-aligned bounding box from world coordinates
/// into the shader-frame coords for `frame_path`. The frame's cell
/// maps to the shader's `[0, 3)³`, so the scale factor is
/// `WORLD_SIZE / frame_cell_size_world`.
pub fn aabb_world_to_frame(
    frame_path: &Path,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let (frame_origin, frame_size) = frame_origin_size_world(frame_path);
    let scale = WORLD_SIZE / frame_size;
    (
        [
            (aabb_min[0] - frame_origin[0]) * scale,
            (aabb_min[1] - frame_origin[1]) * scale,
            (aabb_min[2] - frame_origin[2]) * scale,
        ],
        [
            (aabb_max[0] - frame_origin[0]) * scale,
            (aabb_max[1] - frame_origin[1]) * scale,
            (aabb_max[2] - frame_origin[2]) * scale,
        ],
    )
}

/// Build a `Path` from the slot prefix the GPU ribbon walker
/// actually reached. This is the renderer's effective frame.
pub fn frame_from_slots(slots: &[u8]) -> Path {
    let mut frame = Path::root();
    for &slot in slots {
        frame.push(slot);
    }
    frame
}

/// Render-frame walker: descends the camera's path from
/// `world_root`, accepting Cartesian or CubedSphereBody nodes and
/// stopping at face cells (the shader's face-cell-as-frame
/// dispatch is still being wired up). Truncates at `desired_depth`.
///
/// Returns `(frame_path, frame_node_id)`.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_anchor: &Path,
    desired_depth: u8,
) -> (Path, NodeId) {
    let mut frame = *camera_anchor;
    frame.truncate(desired_depth);
    let mut node_id = world_root;
    let mut reached = 0u8;
    for k in 0..frame.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = frame.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                let Some(child) = library.get(child_id) else { break };
                match child.kind {
                    NodeKind::Cartesian | NodeKind::CubedSphereBody { .. } => {
                        node_id = child_id;
                        reached = (k as u8) + 1;
                    }
                    NodeKind::CubedSphereFace { .. } => break,
                }
            }
            Child::Block(_) | Child::Empty => break,
        }
    }
    frame.truncate(reached);
    (frame, node_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, slot_index, uniform_children};

    fn cartesian_chain(depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    // --------- compute_render_frame ---------

    #[test]
    fn render_frame_root_when_desired_depth_zero() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..3 { anchor.push(13); }
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 0);
        assert_eq!(frame.depth(), 0);
        assert_eq!(node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 { anchor.push(13); }
        let (frame, _node_id) = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(frame.depth(), 3);
    }

    #[test]
    fn render_frame_stops_at_face_cell() {
        let mut lib = NodeLibrary::default();
        let face = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereFace {
                face: crate::world::cubesphere::Face::PosX,
            },
        );
        let mut body_children = empty_children();
        body_children[14] = Child::Node(face);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        anchor.push(14);
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.depth(), 1, "stopped before face cell");
        assert_eq!(node_id, body);
    }

    #[test]
    fn render_frame_descends_into_body() {
        let mut lib = NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 1);
        assert_eq!(frame.depth(), 1);
        assert_eq!(node_id, body);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        anchor.push(13);
        let (frame, _node_id) = compute_render_frame(&lib, root, &anchor, 5);
        assert!(frame.depth() <= 1);
    }

    #[test]
    fn render_frame_stops_when_path_misses_node() {
        // Build root with a Block at slot 5 (not a Node).
        let mut lib = NodeLibrary::default();
        let mut root_children = empty_children();
        root_children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let mut anchor = Path::root();
        anchor.push(5);
        anchor.push(0);
        let (frame, _) = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.depth(), 0, "Block child terminates descent");
    }

    // --------- frame_origin_size_world ---------

    #[test]
    fn origin_size_root() {
        let p = Path::root();
        let (origin, size) = frame_origin_size_world(&p);
        assert_eq!(origin, [0.0, 0.0, 0.0]);
        assert!((size - WORLD_SIZE).abs() < 1e-6);
    }

    #[test]
    fn origin_size_center_slot() {
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        let (origin, size) = frame_origin_size_world(&p);
        assert!((origin[0] - 1.0).abs() < 1e-6);
        assert!((origin[1] - 1.0).abs() < 1e-6);
        assert!((origin[2] - 1.0).abs() < 1e-6);
        assert!((size - 1.0).abs() < 1e-6);
    }

    #[test]
    fn origin_size_corner_slot() {
        let mut p = Path::root();
        p.push(slot_index(0, 0, 0) as u8);
        let (origin, size) = frame_origin_size_world(&p);
        assert_eq!(origin, [0.0, 0.0, 0.0]);
        assert!((size - 1.0).abs() < 1e-6);
    }

    #[test]
    fn origin_size_two_levels_deep() {
        let mut p = Path::root();
        p.push(slot_index(2, 2, 2) as u8);
        p.push(slot_index(1, 1, 1) as u8);
        let (_origin, size) = frame_origin_size_world(&p);
        assert!((size - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn origin_size_decreases_geometrically() {
        // Each level shrinks size by factor of 3.
        let mut p = Path::root();
        for k in 0..10u8 {
            let (_, size) = frame_origin_size_world(&p);
            let expected = WORLD_SIZE / 3f32.powi(k as i32);
            assert!((size - expected).abs() < 1e-6 * expected.max(1e-9));
            p.push(13);
        }
    }

    // --------- aabb_world_to_frame ---------

    #[test]
    fn aabb_root_frame_is_identity() {
        let p = Path::root();
        let (mn, mx) = aabb_world_to_frame(&p, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        assert!((mn[0] - 0.5).abs() < 1e-6);
        assert!((mx[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn aabb_body_frame_scales_3x() {
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        let (mn, mx) = aabb_world_to_frame(&p, [1.4, 1.4, 1.4], [1.6, 1.6, 1.6]);
        assert!((mn[0] - 1.2).abs() < 1e-5);
        assert!((mx[0] - 1.8).abs() < 1e-5);
    }

    #[test]
    fn aabb_round_trip_at_arbitrary_depth() {
        let mut p = Path::root();
        p.push(slot_index(2, 2, 2) as u8);
        let world_min = [2.5, 2.5, 2.5];
        let world_max = [2.7, 2.7, 2.7];
        let (mn, mx) = aabb_world_to_frame(&p, world_min, world_max);
        let (frame_origin, frame_size) = frame_origin_size_world(&p);
        let back_min = [
            mn[0] * frame_size / WORLD_SIZE + frame_origin[0],
            mn[1] * frame_size / WORLD_SIZE + frame_origin[1],
            mn[2] * frame_size / WORLD_SIZE + frame_origin[2],
        ];
        let back_max = [
            mx[0] * frame_size / WORLD_SIZE + frame_origin[0],
            mx[1] * frame_size / WORLD_SIZE + frame_origin[1],
            mx[2] * frame_size / WORLD_SIZE + frame_origin[2],
        ];
        for i in 0..3 {
            assert!((back_min[i] - world_min[i]).abs() < 1e-5);
            assert!((back_max[i] - world_max[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn aabb_deep_frame_inflates_proportionally() {
        // A frame at depth 3 has cell_size 1/9. AABB of size 0.01
        // in world maps to size 0.01 * 9 = 0.09 in frame.
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        p.push(slot_index(1, 1, 1) as u8);
        p.push(slot_index(1, 1, 1) as u8);
        let (frame_origin, _) = frame_origin_size_world(&p);
        let world_min = frame_origin;
        let world_max = [
            frame_origin[0] + 0.01,
            frame_origin[1] + 0.01,
            frame_origin[2] + 0.01,
        ];
        let (mn, mx) = aabb_world_to_frame(&p, world_min, world_max);
        for i in 0..3 {
            assert!(mn[i].abs() < 1e-5);
            assert!((mx[i] - (0.01 * 27.0)).abs() < 1e-4);
        }
    }

    #[test]
    fn frame_from_slots_builds_exact_prefix() {
        let slots = [13u8, 16u8, 4u8];
        let p = frame_from_slots(&slots);
        assert_eq!(p.depth(), slots.len() as u8);
        assert_eq!(p.as_slice(), &slots);
    }
}
