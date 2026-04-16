//! BFS packing of the world tree into the GPU buffer layout.
//!
//! Two pack functions:
//!
//! - `pack_tree`: full BFS, no LOD. Used by tests and for sanity
//!   debugging. Every reachable node ends up in the buffer at full
//!   detail.
//! - `pack_tree_lod`: path-local LOD. Cartesian subtrees that
//!   subtend less than `LOD_THRESHOLD` pixels in the current node's
//!   local frame get flattened into a single Block leaf (their
//!   representative block type). Sphere bodies and face cells are
//!   exempt from flattening — their geometry semantics need the
//!   full subtree.

use std::collections::{HashMap, HashSet};

use crate::world::anchor::{Path, WorldPos};
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
    let mut head = 0usize;
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
                    let repr = library
                        .get(*child_id)
                        .map(|n| n.representative_block)
                        .unwrap_or(0);
                    GpuChild {
                        tag: 2,
                        block_type: repr,
                        _pad: 0,
                        node_index: *visited.get(child_id).expect("child must be visited"),
                    }
                }
            });
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, kinds, root_idx)
}

fn child_screen_pixels(
    camera: &WorldPos,
    node_path: &Path,
    slot: usize,
    screen_height: f32,
    fov: f32,
) -> f32 {
    let camera_local = camera.in_frame(node_path);
    let (cx, cy, cz) = slot_coords(slot);
    let child_center = [cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5];
    let dx = child_center[0] - camera_local[0];
    let dy = child_center[1] - camera_local[1];
    let dz = child_center[2] - camera_local[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
    let half_fov_recip = screen_height / (2.0 * (fov * 0.5).tan());
    half_fov_recip / dist
}

/// LOD-aware tree packing: only uploads nodes large enough to see.
///
/// Unlike the legacy packer, the LOD decision is made entirely in
/// the node's own local frame. `camera` is a path-anchored
/// `WorldPos`; for each queued node we project that position into
/// the node frame with `WorldPos::in_frame(node_path)` and compare it
/// against the candidate child center in that same local metric.
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    pack_tree_lod_selective(library, root, camera, screen_height, fov, &[], &[])
}

/// Like `pack_tree_lod`, but with one or more `preserve_paths`.
///
/// Slots on a preserve path are never uniform-collapsed or
/// distance-LOD flattened. This guarantees the renderer can rebuild
/// the ribbon and descend to the requested active frame even when the
/// surrounding Cartesian region is visually coarse.
pub fn pack_tree_lod_preserving(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    pack_tree_lod_selective(library, root, camera, screen_height, fov, preserve_paths, &[])
}

/// Like `pack_tree_lod_preserving`, but also supports bounded preserve regions.
///
/// Each preserve region is `(path, extra_depth)`: nodes whose path
/// starts with `path` and whose depth ≤ `path.depth() + extra_depth`
/// are never uniform-collapsed or distance-LOD flattened. This keeps
/// the near field around a local render frame detailed even when the
/// surrounding Cartesian region is visually coarse.
pub fn pack_tree_lod_selective(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
    preserve_regions: &[(Path, u8)],
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    const LOD_THRESHOLD: f32 = 0.5;

    let mut preserve_pairs: HashSet<(NodeId, u8)> = HashSet::new();
    for preserve_path in preserve_paths {
        let mut current = root;
        for &slot in *preserve_path {
            preserve_pairs.insert((current, slot));
            let Some(node) = library.get(current) else { break };
            match node.children[slot as usize] {
                Child::Node(child_id) => current = child_id,
                _ => break,
            }
        }
    }

    let preserve_regions: Vec<(Path, u8)> = preserve_regions.to_vec();

    fn in_preserve_region(path: &Path, preserve_regions: &[(Path, u8)]) -> bool {
        preserve_regions.iter().any(|(region_root, extra_depth)| {
            let required_depth = region_root.depth().saturating_add(*extra_depth);
            path.depth() <= required_depth
                && path.common_prefix_len(region_root) == region_root.depth()
        })
    }

    struct QueueEntry {
        node_id: NodeId,
        path: Path,
    }

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut overrides: Vec<[Option<GpuChild>; CHILDREN_PER_NODE]> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    overrides.push([None; CHILDREN_PER_NODE]);
    queue.push(QueueEntry { node_id: root, path: Path::root() });

    let mut head = 0usize;
    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_path = entry.path;
        let ordered_idx = head;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };
        let lod_active = matches!(node.kind, NodeKind::Cartesian);

        for (slot, child) in node.children.iter().enumerate() {
            let Child::Node(child_id) = child else { continue };
            let Some(child_node) = library.get(*child_id) else { continue };

            let mut child_path = node_path;
            child_path.push(slot as u8);
            let on_preserve = preserve_pairs.contains(&(node_id, slot as u8))
                || in_preserve_region(&child_path, &preserve_regions);
            let child_is_cartesian = matches!(child_node.kind, NodeKind::Cartesian);

            if !on_preserve && lod_active && child_is_cartesian
                && child_node.uniform_type != UNIFORM_MIXED
            {
                overrides[ordered_idx][slot] = Some(if child_node.uniform_type == UNIFORM_EMPTY {
                    GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                } else {
                    GpuChild {
                        tag: 1,
                        block_type: child_node.uniform_type,
                        _pad: 0,
                        node_index: 0,
                    }
                });
                continue;
            }

            if !on_preserve && lod_active && child_is_cartesian {
                let screen_pixels =
                    child_screen_pixels(camera, &node_path, slot, screen_height, fov);
                if screen_pixels < LOD_THRESHOLD {
                    overrides[ordered_idx][slot] = Some(if child_node.representative_block < 255 {
                        GpuChild {
                            tag: 1,
                            block_type: child_node.representative_block,
                            _pad: 0,
                            node_index: 0,
                        }
                    } else {
                        GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                    });
                    continue;
                }
            }

            if !visited.contains_key(child_id) {
                let idx = ordered.len() as u32;
                visited.insert(*child_id, idx);
                ordered.push(*child_id);
                overrides.push([None; CHILDREN_PER_NODE]);
                queue.push(QueueEntry {
                    node_id: *child_id,
                    path: child_path,
                });
            }
        }
    }

    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(ordered.len());
    for (ordered_idx, &node_id) in ordered.iter().enumerate() {
        let node = library.get(node_id).expect("node in ordered list must exist");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        for (slot, child) in node.children.iter().enumerate() {
            if let Some(gpu) = overrides[ordered_idx][slot] {
                data.push(gpu);
                continue;
            }
            data.push(match child {
                Child::Empty => GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 },
                Child::Block(bt) => GpuChild { tag: 1, block_type: *bt, _pad: 0, node_index: 0 },
                Child::Node(child_id) => {
                    let repr = library
                        .get(*child_id)
                        .map(|n| n.representative_block)
                        .unwrap_or(0);
                    let idx = visited.get(child_id).copied().unwrap_or(0);
                    GpuChild { tag: 2, block_type: repr, _pad: 0, node_index: idx }
                }
            });
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, kinds, root_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::anchor::WorldPos;
    use crate::world::bootstrap::plain_test_world;
    use crate::world::tree::{empty_children, uniform_children, CENTER_SLOT};

    #[test]
    fn pack_test_world() {
        let world = plain_test_world();
        let (data, kinds, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(data.len() % 27, 0);
        assert_eq!(root_idx, 0);
        assert_eq!(data.len() / 27, world.library.len());
        assert_eq!(kinds.len(), world.library.len());
        for kind in &kinds {
            assert_eq!(kind.kind, 0);
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

    fn camera_at(xyz: [f32; 3]) -> WorldPos {
        // Root-frame-local coords at shallow depth, then deepened.
        WorldPos::from_frame_local(&crate::world::anchor::Path::root(), xyz, 2)
            .deepened_to(10)
    }

    #[test]
    fn pack_includes_body_kind_and_radii() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (_data, kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        let body = kinds.iter().find(|kind| kind.kind == 1).expect("body kind in buffer");
        assert!((body.inner_r - 0.12).abs() < 1e-6);
        assert!((body.outer_r - 0.45).abs() < 1e-6);
    }

    #[test]
    fn pack_lod_flattens_far_uniform_cartesian() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        assert_eq!(data[0].tag, 0);
        assert_eq!(data[13].tag, 2);
        assert!(data[13].node_index > 0);
    }

    #[test]
    fn pack_planet_body_present() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (_data, kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        assert!(kinds.iter().any(|kind| kind.kind == 1), "body kind present");
    }

    #[test]
    fn preserve_path_prevents_uniform_collapse() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera = camera_at([1.5, 2.0, 1.5]);

        let (no_preserve, _, _) = pack_tree_lod(
            &lib, root, &camera, 1080.0, 1.2,
        );
        assert_eq!(no_preserve[16].tag, 0);

        let (with_preserve, _, _) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&[16u8]],
        );
        assert_eq!(with_preserve[16].tag, 2);
        assert!(with_preserve[16].node_index > 0);
    }

    #[test]
    fn preserve_path_chain_lets_ribbon_descend_to_depth_n() {
        use super::super::ribbon::build_ribbon;

        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..10 {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        let root = node;
        lib.ref_inc(root);
        let camera = camera_at([1.5, 1.5, 1.5]);
        let path = [13u8; 9];

        let (data, _kinds, _root_idx) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&path],
        );
        let ribbon = build_ribbon(&data, &path);
        assert_eq!(ribbon.reached_slots.len(), 9);
        assert_eq!(ribbon.ribbon.len(), 9);
    }

    #[test]
    fn preserve_path_only_affects_chain_slots() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera = camera_at([1.5, 2.0, 1.5]);

        let (data, _, _) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&[16u8]],
        );
        assert_eq!(data[16].tag, 2);
        for sibling in [0, 5, 13, 26] {
            assert_eq!(data[sibling].tag, 0, "sibling slot {sibling} should be flattened");
        }
    }

    #[test]
    fn pack_lod_keeps_near_subtrees_full() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let mut mixed = empty_children();
        mixed[0] = Child::Block(crate::world::palette::block::STONE);
        let mixed_node = lib.insert(mixed);
        let mut root_children = uniform_children(Child::Node(air));
        root_children[CENTER_SLOT] = Child::Node(mixed_node);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let camera = camera_at([1.5, 1.5, 1.6]);
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &lib, root, &camera, 1080.0, 1.2,
        );
        assert_eq!(data[CENTER_SLOT].tag, 2);
    }

    /// Verify that breaking a block at various depths actually changes
    /// the packed GPU tree data. This catches dedup bugs where the
    /// library returns the same node after a break, producing an
    /// identical packed buffer.
    #[test]
    fn break_at_every_depth_changes_packed_data() {
        use crate::world::anchor::Path;
        use crate::world::bootstrap;
        use crate::world::edit;

        for spawn_depth in [4u8, 8, 11, 15, 20, 25, 30, 33, 38] {
            let boot = bootstrap::bootstrap_world(
                bootstrap::WorldPreset::PlainTest,
                Some(40),
            );
            let mut world = boot.world;
            let pos = bootstrap::plain_surface_spawn(spawn_depth);
            bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

            let camera = pos;
            let frame_depth = spawn_depth.saturating_sub(3);
            let mut frame_path = camera.anchor;
            frame_path.truncate(frame_depth);

            let preserve_paths: Vec<&[u8]> = vec![frame_path.as_slice()];
            let frame_path_owned = frame_path;
            let preserve_regions = vec![(frame_path_owned, 3u8)];

            let (data_before, _, _) = pack_tree_lod_selective(
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
            );

            // CPU raycast to find a block to break.
            // cpu_raycast works from root, so ray_origin must be
            // root-local [0,3)^3, not frame-local.
            let ray_origin = camera.in_frame(&Path::root());
            let ray_dir = [0.0f32, -0.4, -0.9]; // forward-down
            let hit = edit::cpu_raycast(
                &world.library,
                world.root,
                ray_origin,
                ray_dir,
                spawn_depth as u32,
            );
            let Some(hit) = hit else {
                panic!("no raycast hit at spawn_depth={spawn_depth}");
            };
            assert!(
                edit::break_block(&mut world, &hit),
                "break_block returned false at spawn_depth={spawn_depth}",
            );

            let (data_after, _, _) = pack_tree_lod_selective(
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
            );

            assert_ne!(
                data_before, data_after,
                "packed GPU data unchanged after break at spawn_depth={spawn_depth} \
                 (library dedup may be collapsing the edit)"
            );
        }
    }
}
