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
//!   block type). Sphere bodies are exempt from distance-based
//!   flattening. Face cells still keep their geometry-aware walk,
//!   but uniform face children may collapse to Block/Empty.

use std::collections::HashMap;

use crate::world::cubesphere::{self, Face, FACE_SLOTS};
use crate::world::tree::{
    slot_coords, Child, NodeId, NodeKind, NodeLibrary,
    CHILDREN_PER_NODE, UNIFORM_EMPTY, UNIFORM_MIXED,
};

use super::types::{GpuChild, GpuNodeKind, GPU_NODE_SIZE};

#[derive(Clone, Copy)]
enum QueueGeom {
    Cartesian {
        origin: [f32; 3],
        cell_size: f32,
    },
    Body {
        origin: [f32; 3],
        body_size: f32,
        inner_r: f32,
        outer_r: f32,
    },
    Face {
        body_origin: [f32; 3],
        body_size: f32,
        inner_r: f32,
        outer_r: f32,
        face: Face,
        u_lo: f32,
        v_lo: f32,
        r_lo: f32,
        size: f32,
    },
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn face_child_bounds(
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    size: f32,
    slot: usize,
) -> (f32, f32, f32, f32) {
    let (us, vs, rs) = slot_coords(slot);
    let child_size = size / 3.0;
    (
        u_lo + us as f32 * child_size,
        v_lo + vs as f32 * child_size,
        r_lo + rs as f32 * child_size,
        child_size,
    )
}

fn child_geom(parent: QueueGeom, slot: usize, child_kind: NodeKind) -> QueueGeom {
    match parent {
        QueueGeom::Cartesian { origin, cell_size } => {
            let (cx, cy, cz) = slot_coords(slot);
            let child_origin = [
                origin[0] + cx as f32 * cell_size,
                origin[1] + cy as f32 * cell_size,
                origin[2] + cz as f32 * cell_size,
            ];
            match child_kind {
                NodeKind::CubedSphereBody { inner_r, outer_r, .. } => QueueGeom::Body {
                    origin: child_origin,
                    body_size: cell_size,
                    inner_r,
                    outer_r,
                },
                _ => QueueGeom::Cartesian {
                    origin: child_origin,
                    cell_size: cell_size / 3.0,
                },
            }
        }
        QueueGeom::Body { origin, body_size, inner_r, outer_r } => {
            if let Some(face_idx) = FACE_SLOTS.iter().position(|&s| s == slot) {
                QueueGeom::Face {
                    body_origin: origin,
                    body_size,
                    inner_r,
                    outer_r,
                    face: Face::from_index(face_idx as u8),
                    u_lo: 0.0,
                    v_lo: 0.0,
                    r_lo: 0.0,
                    size: 1.0,
                }
            } else {
                let child_span = body_size / 3.0;
                let (cx, cy, cz) = slot_coords(slot);
                let child_origin = [
                    origin[0] + cx as f32 * child_span,
                    origin[1] + cy as f32 * child_span,
                    origin[2] + cz as f32 * child_span,
                ];
                QueueGeom::Cartesian {
                    origin: child_origin,
                    cell_size: child_span / 3.0,
                }
            }
        }
        QueueGeom::Face {
            body_origin,
            body_size,
            inner_r,
            outer_r,
            face,
            u_lo,
            v_lo,
            r_lo,
            size,
        } => {
            let (child_u_lo, child_v_lo, child_r_lo, child_size) =
                face_child_bounds(u_lo, v_lo, r_lo, size, slot);
            QueueGeom::Face {
                body_origin,
                body_size,
                inner_r,
                outer_r,
                face,
                u_lo: child_u_lo,
                v_lo: child_v_lo,
                r_lo: child_r_lo,
                size: child_size,
            }
        }
    }
}

fn screen_pixels_for_geom(
    geom: QueueGeom,
    camera_pos: [f32; 3],
    half_fov_recip: f32,
) -> Option<f32> {
    match geom {
        QueueGeom::Cartesian { origin, cell_size } => {
            let child_center = [
                origin[0] + cell_size * 1.5,
                origin[1] + cell_size * 1.5,
                origin[2] + cell_size * 1.5,
            ];
            let dist = vec3_distance(child_center, camera_pos).max(0.001);
            Some((cell_size * 3.0) / dist * half_fov_recip)
        }
        QueueGeom::Body { .. } => None,
        QueueGeom::Face {
            body_origin,
            body_size,
            inner_r,
            outer_r,
            face,
            u_lo,
            v_lo,
            r_lo,
            size,
        } => {
            let body_center = vec3_add(body_origin, [body_size * 0.5; 3]);
            let shell = (outer_r - inner_r) * body_size;
            let r_world = inner_r * body_size + r_lo * shell;
            let corners = cubesphere::block_corners(
                body_center,
                face,
                u_lo * 2.0 - 1.0,
                v_lo * 2.0 - 1.0,
                r_world,
                size * 2.0,
                size * 2.0,
                size * shell,
            );
            let center = corners
                .iter()
                .fold([0.0; 3], |acc, &p| vec3_add(acc, p));
            let center = vec3_scale(center, 1.0 / 8.0);
            let radius = corners
                .iter()
                .map(|&p| vec3_distance(p, center))
                .fold(0.0, f32::max);
            let dist = (vec3_distance(center, camera_pos) - radius).max(0.001);
            Some((radius * 2.0) / dist * half_fov_recip)
        }
    }
}

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
    pack_tree_lod_preserving(library, root, camera_pos, screen_height, fov, &[])
}

/// Like `pack_tree_lod`, but with a `preserve_path`: the slots
/// on the camera's anchor from `root`. Slots on the preserve
/// path are NEVER LOD-flattened or uniform-collapsed — they're
/// always emitted as Node children so `build_ribbon` can walk
/// the full chain and the shader can lift the camera frame
/// arbitrarily deep.
///
/// This is what unlocks layer-1 descent: with `preserve_path`
/// passed, the frame can sit at any depth in the camera's
/// anchor chain regardless of how distant or uniform the
/// surrounding cells are.
pub fn pack_tree_lod_preserving(
    library: &NodeLibrary,
    root: NodeId,
    camera_pos: [f32; 3],
    screen_height: f32,
    fov: f32,
    preserve_path: &[u8],
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    use std::collections::HashSet;

    // Build the set of (parent_node_id, slot) pairs on the camera
    // path that must NOT be flattened. Walk from root following
    // `preserve_path` slot-by-slot, recording the (current,
    // slot) pair at each step.
    let mut preserve_pairs: HashSet<(NodeId, u8)> = HashSet::new();
    {
        let mut current = root;
        for &slot in preserve_path {
            preserve_pairs.insert((current, slot));
            let Some(node) = library.get(current) else { break };
            match node.children[slot as usize] {
                Child::Node(child_id) => { current = child_id; }
                _ => break,
            }
        }
    }

    let half_fov_recip = screen_height / (2.0 * (fov * 0.5).tan());
    const LOD_THRESHOLD: f32 = 0.5;

    struct QueueEntry {
        node_id: NodeId,
        geom: QueueGeom,
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
        geom: QueueGeom::Cartesian {
            origin: [0.0; 3],
            cell_size: 1.0,
        },
    });
    let mut head = 0;

    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_geom = entry.geom;
        let ordered_idx = head;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };
        let uniform_collapse_active = matches!(
            node.kind,
            NodeKind::Cartesian
                | NodeKind::CubedSphereFace { .. }
                | NodeKind::CubedSphereProceduralFace { .. }
        );

        for (slot, child) in node.children.iter().enumerate() {
            if let Child::Node(child_id) = child {
                let child_node = match library.get(*child_id) {
                    Some(n) => n,
                    None => continue,
                };

                // Uniform-content collapse is safe for Cartesian and
                // face subtrees: a terminal Block/Empty child still
                // occupies the same parent cell, just without paying
                // to walk deeper into a uniform subtree. Skipped for
                // slots on the camera's preserve path so build_ribbon
                // can descend.
                let child_is_collapseable = matches!(
                    child_node.kind,
                    NodeKind::Cartesian
                        | NodeKind::CubedSphereFace { .. }
                        | NodeKind::CubedSphereProceduralFace { .. }
                );
                let on_preserve = preserve_pairs.contains(&(node_id, slot as u8));
                let child_geom = child_geom(node_geom, slot, child_node.kind);
                if !on_preserve && uniform_collapse_active && child_is_collapseable
                    && child_node.uniform_type != UNIFORM_MIXED {
                    let gpu = if child_node.uniform_type == UNIFORM_EMPTY {
                        GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                    } else {
                        GpuChild { tag: 1, block_type: child_node.uniform_type, _pad: 0, node_index: 0 }
                    };
                    overrides[ordered_idx][slot] = Some(gpu);
                    continue;
                }

                if !on_preserve && !matches!(child_geom, QueueGeom::Body { .. }) {
                    let screen_pixels = screen_pixels_for_geom(
                        child_geom,
                        camera_pos,
                        half_fov_recip,
                    ).unwrap_or(f32::INFINITY);
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
                    queue.push(QueueEntry {
                        node_id: *child_id,
                        geom: child_geom,
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
            NodeKind::CubedSphereBody {
                inner_r: 0.12,
                outer_r: 0.45,
                surface_r: 0.30,
                noise_scale: 0.0,
                noise_freq: 1.0,
                noise_seed: 0,
                surface_block: 1,
                core_block: 2,
            },
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
    fn preserve_path_prevents_uniform_collapse() {
        // World: root with uniform-empty Cartesian Node at every
        // slot. Without preserve_path, all slots get tag=0
        // (flattened). With preserve_path = [16], slot 16 stays
        // as a Node so the ribbon can descend.
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera_pos = [1.5, 2.0, 1.5];

        let (no_preserve, _, _) = pack_tree_lod(
            &lib, root, camera_pos, 1080.0, 1.2,
        );
        assert_eq!(no_preserve[16].tag, 0,
            "without preserve, slot 16 collapses to tag=0");

        let (with_preserve, _, _) = pack_tree_lod_preserving(
            &lib, root, camera_pos, 1080.0, 1.2,
            &[16],
        );
        assert_eq!(with_preserve[16].tag, 2,
            "with preserve_path=[16], slot 16 stays as a Node");
        assert!(with_preserve[16].node_index > 0,
            "preserved slot points to a real buffer entry");
    }

    #[test]
    fn preserve_path_chain_lets_ribbon_descend_to_depth_n() {
        use super::super::ribbon::build_ribbon;
        // Build a chain of empty Cartesian nodes 10 deep. With
        // preserve_path = [13;10], build_ribbon should walk all
        // 10 levels.
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..10 {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        let root = node;
        lib.ref_inc(root);
        let camera_pos = [1.5, 1.5, 1.5];
        let path = [13u8; 9];  // 9 descents = depth 9 frame

        let (data, _kinds, _root_idx) = pack_tree_lod_preserving(
            &lib, root, camera_pos, 1080.0, 1.2,
            &path,
        );
        let r = build_ribbon(&data, &path);
        assert_eq!(r.reached_slots.len(), 9,
            "preserve_path enables descent through 9 levels");
        assert_eq!(r.ribbon.len(), 9);
    }

    #[test]
    fn preserve_path_only_affects_chain_slots() {
        // Verify that OTHER slots (not on preserve_path) still
        // get LOD-flattened.
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera_pos = [1.5, 2.0, 1.5];

        let (data, _, _) = pack_tree_lod_preserving(
            &lib, root, camera_pos, 1080.0, 1.2,
            &[16],
        );
        // Slot 16 preserved (Node). Slots 0, 5, 13, 26 should
        // still be flattened (tag=0).
        assert_eq!(data[16].tag, 2);
        for sib in [0, 5, 13, 26] {
            assert_eq!(data[sib].tag, 0,
                "sibling slot {sib} should be flattened");
        }
    }

    #[test]
    fn pack_lod_collapses_uniform_face_child_off_preserve_path() {
        let mut lib = NodeLibrary::default();
        let uniform_leaf_face = lib.insert_with_kind(
            uniform_children(Child::Block(crate::world::palette::block::STONE)),
            NodeKind::CubedSphereFace { face: crate::world::cubesphere::Face::PosX },
        );
        let mut root_children = empty_children();
        root_children[0] = Child::Node(uniform_leaf_face);
        let root = lib.insert_with_kind(
            root_children,
            NodeKind::CubedSphereFace { face: crate::world::cubesphere::Face::PosX },
        );
        lib.ref_inc(root);

        let camera_pos = [1.5, 2.0, 1.5];
        let (data, _kinds, _root_idx) = pack_tree_lod_preserving(
            &lib, root, camera_pos, 1080.0, 1.2, &[],
        );

        assert_eq!(data[0].tag, 1, "uniform face subtree should collapse to Block");
        assert_eq!(data[0].block_type, crate::world::palette::block::STONE);
    }

    #[test]
    fn pack_lod_flattens_far_face_root() {
        let mut lib = NodeLibrary::default();
        let mut face_children = empty_children();
        face_children[0] = Child::Block(crate::world::palette::block::STONE);
        let face_root = lib.insert_with_kind(
            face_children,
            NodeKind::CubedSphereFace { face: crate::world::cubesphere::Face::PosX },
        );
        let mut body_children = empty_children();
        body_children[FACE_SLOTS[0]] = Child::Node(face_root);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody {
                inner_r: 0.12,
                outer_r: 0.45,
                surface_r: 0.30,
                noise_scale: 0.0,
                noise_freq: 1.0,
                noise_seed: 0,
                surface_block: 1,
                core_block: 2,
            },
        );
        let mut root_children = empty_children();
        root_children[CENTER_SLOT] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let (data, kinds, _) = pack_tree_lod(
            &lib, root, [1000.0, 1000.0, 1000.0], 1080.0, 1.2,
        );
        let body_idx = kinds.iter().position(|k| k.kind == 1).expect("body present");
        assert_ne!(data[body_idx * 27 + FACE_SLOTS[0]].tag, 2,
            "far face root should LOD-flatten instead of forcing a node");
    }

    #[test]
    fn pack_lod_flattens_far_face_descendant_off_preserve_path() {
        let mut lib = NodeLibrary::default();
        let mut inner_children = empty_children();
        inner_children[0] = Child::Block(crate::world::palette::block::STONE);
        let inner = lib.insert(inner_children);

        let mut face_children = empty_children();
        face_children[0] = Child::Node(inner);
        let face_root = lib.insert_with_kind(
            face_children,
            NodeKind::CubedSphereFace { face: crate::world::cubesphere::Face::PosX },
        );

        let mut body_children = empty_children();
        body_children[FACE_SLOTS[0]] = Child::Node(face_root);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody {
                inner_r: 0.12,
                outer_r: 0.45,
                surface_r: 0.30,
                noise_scale: 0.0,
                noise_freq: 1.0,
                noise_seed: 0,
                surface_block: 1,
                core_block: 2,
            },
        );

        let mut root_children = empty_children();
        root_children[CENTER_SLOT] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let (data, kinds, _) = pack_tree_lod_preserving(
            &lib,
            root,
            [1000.0, 1000.0, 1000.0],
            1080.0,
            1.2,
            &[CENTER_SLOT as u8, FACE_SLOTS[0] as u8],
        );
        let body_idx = kinds.iter().position(|k| k.kind == 1).expect("body present");
        let face_idx = data[body_idx * 27 + FACE_SLOTS[0]].node_index as usize;
        assert_ne!(data[face_idx * 27].tag, 2,
            "face descendants should use face-geometry LOD even when stored as Cartesian nodes");
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
