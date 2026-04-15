//! BFS packing of the world tree into the GPU buffer layout.
//!
//! Two pack functions:
//!
//! - `pack_tree`: full BFS, no LOD. Used by tests and for sanity
//!   debugging. Every reachable node ends up in the buffer at full
//!   detail.
//! - `pack_tree_lod`: distance-aware. Cartesian subtrees that
//!   subtend less than `LOD_THRESHOLD` pixels at the camera get
//!   flattened into a single Block leaf (their representative
//!   block type). Sphere bodies and face cells are exempt from
//!   flattening — their geometry semantics need the full subtree.

use std::collections::HashMap;

use crate::world::tree::{
    slot_coords, Child, NodeId, NodeKind, NodeLibrary,
    CHILDREN_PER_NODE, UNIFORM_EMPTY, UNIFORM_MIXED,
};

use super::types::{GpuChild, GpuNodeKind, GPU_NODE_SIZE};

/// Pack the visible portion of the tree into flat GPU buffers.
/// Returns `(tree_data, node_kinds, root_buffer_index)`.
pub fn pack_tree(
    library: &NodeLibrary,
    root: NodeId,
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut head = 0;
    visited.insert(root, 0);
    ordered.push(root);
    while head < ordered.len() {
        let nid = ordered[head];
        head += 1;
        if let Some(node) = library.get(nid) {
            for child in &node.children {
                if let Child::Node(child_id) = child {
                    if !visited.contains_key(child_id) {
                        let idx = ordered.len() as u32;
                        visited.insert(*child_id, idx);
                        ordered.push(*child_id);
                    }
                }
            }
        }
    }

    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(ordered.len());
    for &nid in &ordered {
        let node = library.get(nid).expect("node in ordered list must exist");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        for child in &node.children {
            data.push(match child {
                Child::Empty => GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 },
                Child::Block(bt) => GpuChild { tag: 1, block_type: *bt, _pad: 0, node_index: 0 },
                Child::Node(child_id) => {
                    let repr = library.get(*child_id)
                        .map(|n| n.representative_block).unwrap_or(0);
                    GpuChild {
                        tag: 2, block_type: repr, _pad: 0,
                        node_index: *visited.get(child_id).expect("child must be visited"),
                    }
                },
            });
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, kinds, root_idx)
}

/// LOD-aware tree packing: only uploads nodes large enough to see.
///
/// Same dual-buffer output as `pack_tree`, plus distance-aware
/// flattening of subtrees that cover less than `LOD_THRESHOLD`
/// pixels on screen. The shader walks whatever ends up in the
/// buffer; flattened cells appear as Block/Empty leaves.
///
/// `camera_pos` is in the same coord system as the BFS uses
/// internally — for `root = world.root` that's world XYZ; the
/// caller is responsible for matching units.
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera_pos: [f32; 3],
    screen_height: f32,
    fov: f32,
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    let half_fov_recip = screen_height / (2.0 * (fov * 0.5).tan());
    const LOD_THRESHOLD: f32 = 0.5;

    struct QueueEntry {
        node_id: NodeId,
        origin: [f32; 3],
        cell_size: f32,
    }

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut overrides: Vec<[Option<GpuChild>; CHILDREN_PER_NODE]> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    overrides.push([None; CHILDREN_PER_NODE]);
    queue.push(QueueEntry {
        node_id: root,
        origin: [0.0; 3],
        cell_size: 1.0,
    });
    let mut head = 0;

    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_origin = entry.origin;
        let cell_size = entry.cell_size;
        let ordered_idx = head;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };

        // Sphere-body and face nodes do NOT participate in
        // distance-LOD flattening — their children are interpreted
        // by the shader's NodeKind dispatch. Flattening would lose
        // the geometric semantics. Only Cartesian nodes apply LOD.
        let lod_active = matches!(node.kind, NodeKind::Cartesian);

        for (slot, child) in node.children.iter().enumerate() {
            if let Child::Node(child_id) = child {
                let child_node = match library.get(*child_id) {
                    Some(n) => n,
                    None => continue,
                };

                // Uniform-content collapse — only safe for Cartesian
                // nodes (face/body subtrees have geometry that the
                // shader needs to walk).
                let child_is_cartesian = matches!(child_node.kind, NodeKind::Cartesian);
                if lod_active && child_is_cartesian
                    && child_node.uniform_type != UNIFORM_MIXED {
                    let gpu = if child_node.uniform_type == UNIFORM_EMPTY {
                        GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                    } else {
                        GpuChild { tag: 1, block_type: child_node.uniform_type, _pad: 0, node_index: 0 }
                    };
                    overrides[ordered_idx][slot] = Some(gpu);
                    continue;
                }

                if lod_active && child_is_cartesian {
                    let (cx, cy, cz) = slot_coords(slot);
                    let child_center = [
                        node_origin[0] + (cx as f32 + 0.5) * cell_size,
                        node_origin[1] + (cy as f32 + 0.5) * cell_size,
                        node_origin[2] + (cz as f32 + 0.5) * cell_size,
                    ];
                    let dx = child_center[0] - camera_pos[0];
                    let dy = child_center[1] - camera_pos[1];
                    let dz = child_center[2] - camera_pos[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
                    let screen_pixels = cell_size / dist * half_fov_recip;
                    if screen_pixels < LOD_THRESHOLD {
                        let gpu = if child_node.representative_block < 255 {
                            GpuChild { tag: 1, block_type: child_node.representative_block, _pad: 0, node_index: 0 }
                        } else {
                            GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                        };
                        overrides[ordered_idx][slot] = Some(gpu);
                        continue;
                    }
                }

                if !visited.contains_key(child_id) {
                    let idx = ordered.len() as u32;
                    visited.insert(*child_id, idx);
                    ordered.push(*child_id);
                    overrides.push([None; CHILDREN_PER_NODE]);
                    let (cx, cy, cz) = slot_coords(slot);
                    let child_origin = [
                        node_origin[0] + cx as f32 * cell_size,
                        node_origin[1] + cy as f32 * cell_size,
                        node_origin[2] + cz as f32 * cell_size,
                    ];
                    queue.push(QueueEntry {
                        node_id: *child_id,
                        origin: child_origin,
                        cell_size: cell_size / 3.0,
                    });
                }
            }
        }
    }

    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(ordered.len());
    for (oi, &nid) in ordered.iter().enumerate() {
        let node = library.get(nid).expect("node in ordered list must exist");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        for (slot, child) in node.children.iter().enumerate() {
            if let Some(gpu) = overrides[oi][slot] {
                data.push(gpu);
            } else {
                data.push(match child {
                    Child::Empty => GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 },
                    Child::Block(bt) => GpuChild { tag: 1, block_type: *bt, _pad: 0, node_index: 0 },
                    Child::Node(child_id) => {
                        let repr = library.get(*child_id).map(|n| n.representative_block).unwrap_or(0);
                        let idx = visited.get(child_id).copied().unwrap_or(0);
                        GpuChild { tag: 2, block_type: repr, _pad: 0, node_index: idx }
                    },
                });
            }
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, kinds, root_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, uniform_children, CENTER_SLOT};

    #[test]
    fn pack_test_world() {
        let world = crate::world::state::WorldState::test_world();
        let (data, kinds, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(data.len() % 27, 0);
        assert_eq!(root_idx, 0);
        assert_eq!(data.len() / 27, world.library.len());
        assert_eq!(kinds.len(), world.library.len());
        for k in &kinds {
            assert_eq!(k.kind, 0);
        }
    }

    fn planet_world() -> crate::world::state::WorldState {
        let mut lib = NodeLibrary::default();
        let leaf_air = lib.insert(empty_children());
        let mut root_children = uniform_children(Child::Node(leaf_air));
        let body_id = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        root_children[CENTER_SLOT] = Child::Node(body_id);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        crate::world::state::WorldState { root, library: lib }
    }

    #[test]
    fn pack_includes_body_kind_and_radii() {
        let world = planet_world();
        let camera_pos = [1.5, 2.0, 1.5];
        let (_data, kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, camera_pos, 1080.0, 1.2,
        );
        let body = kinds.iter().find(|k| k.kind == 1).expect("body kind in buffer");
        assert!((body.inner_r - 0.12).abs() < 1e-6);
        assert!((body.outer_r - 0.45).abs() < 1e-6);
    }

    #[test]
    fn pack_lod_flattens_far_uniform_cartesian() {
        let world = planet_world();
        let camera_pos = [1.5, 2.0, 1.5];
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, camera_pos, 1080.0, 1.2,
        );
        // Slot 0 (corner, far from camera, uniform empty) → tag=0.
        assert_eq!(data[0].tag, 0);
        // Slot 13 = body, must be a Node tag.
        assert_eq!(data[13].tag, 2);
        assert!(data[13].node_index > 0);
    }

    #[test]
    fn pack_planet_body_present() {
        let world = planet_world();
        let camera_pos = [1.5, 2.0, 1.5];
        let (_data, kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, camera_pos, 1080.0, 1.2,
        );
        assert!(kinds.iter().any(|k| k.kind == 1), "body kind present");
    }

    #[test]
    fn pack_lod_keeps_near_subtrees_full() {
        // A subtree with mixed content close to the camera should
        // descend (not flatten). Build a root with a non-uniform
        // child near camera; verify the child is still a Node tag.
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let mut mixed = empty_children();
        mixed[0] = Child::Block(crate::world::palette::block::STONE);
        let mixed_node = lib.insert(mixed);
        let mut root_children = uniform_children(Child::Node(air));
        root_children[CENTER_SLOT] = Child::Node(mixed_node);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        // Camera VERY close so the center cell subtends many pixels.
        let camera_pos = [1.5, 1.5, 1.6];
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &lib, root, camera_pos, 1080.0, 1.2,
        );
        // Center slot 13 should be a Node tag (not flattened).
        assert_eq!(data[CENTER_SLOT].tag, 2);
    }
}
