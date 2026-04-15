//! GPU data packing: convert tree nodes into flat buffers for the
//! ray march shader.
//!
//! Two parallel buffers are produced per pack:
//!
//! - `tree: Vec<GpuChild>` — 27 children per node, BFS-ordered. Each
//!   child has a tag (Empty / Block / Node) and either a block_type
//!   or a buffer-local node index.
//! - `node_kinds: Vec<GpuNodeKind>` — one entry per packed node,
//!   carrying its `NodeKind` discriminant + per-kind data (sphere
//!   body radii, cube face index). The shader looks this up when
//!   it walks into a Node child to decide whether to descend with
//!   the standard Cartesian DDA or switch to the cubed-sphere DDA.

use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

use super::tree::*;

// Each child in the GPU buffer is 8 bytes:
//   tag (u8): 0=Empty, 1=Block, 2=Node
//   block_type (u8): valid when tag==1
//   _pad (u16)
//   node_index (u32): buffer-local index, valid when tag==2
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuChild {
    pub tag: u8,
    pub block_type: u8,
    pub _pad: u16,
    pub node_index: u32,
}

/// One node in the GPU buffer = 27 GpuChild = 216 bytes.
pub const GPU_NODE_SIZE: usize = 27;

/// Per-packed-node metadata: which `NodeKind` this node is, plus the
/// per-kind data the shader needs to render its content. Indexed by
/// the same buffer index used in `GpuChild::node_index`.
///
/// 16 bytes per node so the WGSL `array<NodeKindGpu>` aligns
/// cleanly. `kind` discriminant: 0 = Cartesian, 1 = CubedSphereBody,
/// 2 = CubedSphereFace.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuNodeKind {
    pub kind: u32,
    pub face: u32,
    pub inner_r: f32,
    pub outer_r: f32,
}

impl GpuNodeKind {
    pub fn from_node_kind(k: NodeKind) -> Self {
        match k {
            NodeKind::Cartesian => Self { kind: 0, face: 0, inner_r: 0.0, outer_r: 0.0 },
            NodeKind::CubedSphereBody { inner_r, outer_r } => Self {
                kind: 1, face: 0, inner_r, outer_r,
            },
            NodeKind::CubedSphereFace { face } => Self {
                kind: 2, face: face as u32, inner_r: 0.0, outer_r: 0.0,
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCamera {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub forward: [f32; 3],
    pub _pad1: f32,
    pub right: [f32; 3],
    pub _pad2: f32,
    pub up: [f32; 3],
    pub fov: f32,
}

/// Block color palette — up to 256 RGBA colors indexed by block type.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPalette {
    pub colors: [[f32; 4]; 256],
}

/// One entry in the ancestor ribbon. The shader pops from the
/// frame upward; `ribbon[0]` is the frame's direct parent, then
/// `ribbon[1]` the grandparent, etc., up to the absolute root.
///
/// `node_idx` is the buffer index of the ancestor's node. `slot`
/// is the slot in the ancestor that contained the level the ray
/// is popping FROM — the shader uses `slot_coords(slot)` to add
/// the integer offset when remapping the ray into the ancestor's
/// frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct GpuRibbonEntry {
    pub node_idx: u32,
    pub slot: u32,
}

/// Walk the GPU buffer from index 0 (world root) along `frame_slots`,
/// following Node-tagged children, returning:
///
/// - `frame_root_idx` — the buffer index of the deepest reached node
/// - `ribbon` — pop-ordered ancestors from frame's direct parent up
///   to (and including) the world root. Empty when `frame_slots`
///   is empty (frame == world root).
///
/// Stops early if a slot points at a non-Node child (e.g. uniform
/// LOD-flattened); the frame_root then sits at that depth.
pub fn build_ribbon(tree: &[GpuChild], frame_slots: &[u8]) -> (u32, Vec<GpuRibbonEntry>) {
    let mut walk: Vec<u32> = Vec::with_capacity(frame_slots.len() + 1);
    walk.push(0);
    let mut reached_slots: Vec<u8> = Vec::with_capacity(frame_slots.len());
    let mut current = 0u32;
    for &slot in frame_slots {
        let idx = (current as usize) * GPU_NODE_SIZE + slot as usize;
        if idx >= tree.len() { break; }
        let child = tree[idx];
        if child.tag != 2 { break; }
        current = child.node_index;
        walk.push(current);
        reached_slots.push(slot);
    }
    let frame_root_idx = *walk.last().unwrap();
    let depth = reached_slots.len();
    let mut ribbon = Vec::with_capacity(depth);
    for pop in 0..depth {
        let ancestor_idx = walk[depth - 1 - pop];
        let slot = reached_slots[depth - 1 - pop];
        ribbon.push(GpuRibbonEntry {
            node_idx: ancestor_idx,
            slot: slot as u32,
        });
    }
    (frame_root_idx, ribbon)
}

impl Default for GpuPalette {
    fn default() -> Self {
        let mut colors = [[0.0f32; 4]; 256];
        for &(idx, _, color) in super::palette::BUILTINS {
            colors[idx as usize] = color;
        }
        Self { colors }
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
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera_pos: [f32; 3],
    screen_height: f32,
    fov: f32,
) -> (Vec<GpuChild>, Vec<GpuNodeKind>, u32) {
    use super::tree::{UNIFORM_EMPTY, UNIFORM_MIXED, slot_coords};

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
    use super::super::palette::block;

    #[test]
    fn pack_test_world() {
        let world = super::super::state::WorldState::test_world();
        let (data, kinds, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(data.len() % 27, 0);
        assert_eq!(root_idx, 0);
        assert_eq!(data.len() / 27, world.library.len());
        assert_eq!(kinds.len(), world.library.len());
        // All test_world nodes are Cartesian.
        for k in &kinds {
            assert_eq!(k.kind, 0);
        }
    }

    #[test]
    fn gpu_child_size() {
        assert_eq!(std::mem::size_of::<GpuChild>(), 8);
    }

    #[test]
    fn gpu_node_kind_size() {
        assert_eq!(std::mem::size_of::<GpuNodeKind>(), 16);
    }

    #[test]
    fn gpu_ribbon_entry_size() {
        // Must match WGSL `RibbonEntry { node_idx: u32, slot: u32 }`.
        assert_eq!(std::mem::size_of::<GpuRibbonEntry>(), 8);
    }

    // --------- build_ribbon ---------

    fn one_node_tree() -> Vec<GpuChild> {
        // 27 children, all Empty
        vec![GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }; 27]
    }

    fn two_node_tree(parent_slot: u8) -> Vec<GpuChild> {
        // Two nodes in buffer: index 0 (parent) has Node child at
        // `parent_slot` pointing to index 1.
        let mut data = vec![GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }; 54];
        data[parent_slot as usize] = GpuChild { tag: 2, block_type: 0, _pad: 0, node_index: 1 };
        data
    }

    #[test]
    fn build_ribbon_empty_path_gives_empty_ribbon() {
        let tree = one_node_tree();
        let (frame_idx, ribbon) = build_ribbon(&tree, &[]);
        assert_eq!(frame_idx, 0);
        assert!(ribbon.is_empty());
    }

    #[test]
    fn build_ribbon_single_step() {
        let tree = two_node_tree(13);
        let (frame_idx, ribbon) = build_ribbon(&tree, &[13]);
        assert_eq!(frame_idx, 1, "frame should be at child node");
        assert_eq!(ribbon.len(), 1);
        assert_eq!(ribbon[0], GpuRibbonEntry { node_idx: 0, slot: 13 });
    }

    #[test]
    fn build_ribbon_stops_at_non_node_child() {
        // Path requests slot 13 → Node, then slot 5 → ... but child
        // at idx 1 has only Empty children. Walker stops at frame=1.
        let tree = two_node_tree(13);
        let (frame_idx, ribbon) = build_ribbon(&tree, &[13, 5]);
        assert_eq!(frame_idx, 1, "ran out of Node children at depth 1");
        assert_eq!(ribbon.len(), 1);
    }

    #[test]
    fn build_ribbon_multi_step_pop_order() {
        // Three nodes: 0 → slot 16 → 1 → slot 8 → 2.
        let mut data = vec![GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }; 81];
        data[16] = GpuChild { tag: 2, block_type: 0, _pad: 0, node_index: 1 };
        data[27 + 8] = GpuChild { tag: 2, block_type: 0, _pad: 0, node_index: 2 };

        let (frame_idx, ribbon) = build_ribbon(&data, &[16, 8]);
        assert_eq!(frame_idx, 2);
        assert_eq!(ribbon.len(), 2);
        // Pop order: ribbon[0] = direct parent (idx 1, came from
        // slot 8); ribbon[1] = grandparent (idx 0, came from slot 16).
        assert_eq!(ribbon[0], GpuRibbonEntry { node_idx: 1, slot: 8 });
        assert_eq!(ribbon[1], GpuRibbonEntry { node_idx: 0, slot: 16 });
    }

    // --------- pack_tree_lod from world root with body sibling ---------

    fn planet_world() -> super::super::state::WorldState {
        let mut lib = NodeLibrary::default();
        let leaf_air = lib.insert(empty_children());
        let mut root_children = uniform_children(Child::Node(leaf_air));
        // Body at slot 13 — kind = CubedSphereBody.
        let body_id = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        root_children[CENTER_SLOT] = Child::Node(body_id);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        super::super::state::WorldState { root, library: lib }
    }

    #[test]
    fn pack_includes_body_kind_and_radii() {
        let world = planet_world();
        let camera_pos = [1.5, 2.0, 1.5];
        let (_data, kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, camera_pos, 1080.0, 1.2,
        );
        // Find the body kind entry (kind == 1).
        let body = kinds.iter().find(|k| k.kind == 1).expect("body kind in buffer");
        assert!((body.inner_r - 0.12).abs() < 1e-6);
        assert!((body.outer_r - 0.45).abs() < 1e-6);
    }

    #[test]
    fn pack_lod_flattens_far_uniform_cartesian() {
        // Camera far from world; uniform empty Cartesian siblings
        // should be tag=0 (empty) overrides, not full descents.
        let world = planet_world();
        let camera_pos = [1.5, 2.0, 1.5];
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, camera_pos, 1080.0, 1.2,
        );
        // Root's slot 0 (corner, far from camera, uniform empty)
        // should be tag=0 (empty leaf).
        assert_eq!(data[0].tag, 0);
        // Slot 13 = body, must be a Node tag with non-zero index.
        assert_eq!(data[13].tag, 2);
        assert!(data[13].node_index > 0);
    }

    #[test]
    fn pack_planet_face_subtrees_present() {
        let world = planet_world();
        let camera_pos = [1.5, 2.0, 1.5];
        let (_data, kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, camera_pos, 1080.0, 1.2,
        );
        // No face nodes in our minimal planet_world (body has empty
        // children). Sanity check: only the body kind exists.
        let has_body = kinds.iter().any(|k| k.kind == 1);
        assert!(has_body);
    }

    // --------- build_ribbon on a real packed planet world ---------

    #[test]
    fn ribbon_for_path_into_body() {
        let world = planet_world();
        let camera_pos = [1.5, 2.0, 1.5];
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, camera_pos, 1080.0, 1.2,
        );
        // Frame path = [13] (down to body).
        let (frame_idx, ribbon) = build_ribbon(&data, &[13]);
        assert!(frame_idx > 0, "body was packed at non-zero index");
        assert_eq!(ribbon.len(), 1);
        assert_eq!(ribbon[0].node_idx, 0, "world root is at index 0");
        assert_eq!(ribbon[0].slot, 13);
    }
}
