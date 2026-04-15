//! GPU data packing: convert tree nodes into a flat buffer for the
//! ray march shader.

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

/// Per-node metadata: the `NodeKind` exposed to the shader so its
/// tree walk can dispatch on body / face branches without any
/// separate uniform state. One entry per node in the packed tree
/// buffer, indexed by `node_idx`.
///
/// Layout (16 bytes):
/// - `kind_tag`: 0 = Cartesian, 1 = CubedSphereBody, 2 = CubedSphereFace.
/// - `face_index`: 0..6 for `CubedSphereFace`, 0 otherwise.
/// - `inner_r`, `outer_r`: body-local radii `[0, 0.5]`, only valid for
///   `CubedSphereBody`. Zero for other kinds.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuNodeMeta {
    pub kind_tag: u32,
    pub face_index: u32,
    pub inner_r: f32,
    pub outer_r: f32,
}

impl Default for GpuNodeMeta {
    fn default() -> Self {
        Self { kind_tag: 0, face_index: 0, inner_r: 0.0, outer_r: 0.0 }
    }
}

impl GpuNodeMeta {
    pub fn from_kind(kind: &super::tree::NodeKind) -> Self {
        use super::tree::NodeKind;
        match kind {
            NodeKind::Cartesian => Self::default(),
            NodeKind::CubedSphereBody { inner_r, outer_r } => Self {
                kind_tag: 1,
                face_index: 0,
                inner_r: *inner_r,
                outer_r: *outer_r,
            },
            NodeKind::CubedSphereFace { face } => Self {
                kind_tag: 2,
                face_index: *face as u32,
                inner_r: 0.0,
                outer_r: 0.0,
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
/// Built from a `ColorRegistry` via `to_gpu_palette()`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPalette {
    pub colors: [[f32; 4]; 256],
}

impl Default for GpuPalette {
    fn default() -> Self {
        // Populate from the builtin palette entries.
        let mut colors = [[0.0f32; 4]; 256];
        for &(idx, _, color) in super::palette::BUILTINS {
            colors[idx as usize] = color;
        }
        Self { colors }
    }
}

/// Pack the visible portion of the tree into a flat GPU buffer.
/// Returns `(node_data, node_metas, root_buffer_index)`.
pub fn pack_tree(
    library: &NodeLibrary,
    root: NodeId,
) -> (Vec<GpuChild>, Vec<GpuNodeMeta>, u32) {
    // BFS to collect all reachable nodes. `ordered` doubles as the
    // queue (head advances through it) and the result (insertion order
    // = buffer order).
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

    // Pack into flat buffer.
    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    for &nid in &ordered {
        let Some(node) = library.get(nid) else {
            // Shouldn't happen on a consistent library — a node was
            // enqueued via `library.get(...)` succeeding but is now
            // missing. Emit empty children so packing stays in lock-
            // step with `ordered` (one node slot per entry).
            for _ in 0..CHILDREN_PER_NODE {
                data.push(GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 });
            }
            continue;
        };
        for child in &node.children {
            data.push(match child {
                Child::Empty => GpuChild {
                    tag: 0,
                    block_type: 0,
                    _pad: 0,
                    node_index: 0,
                },
                Child::Block(bt) => GpuChild {
                    tag: 1,
                    block_type: *bt,
                    _pad: 0,
                    node_index: 0,
                },
                Child::Node(child_id) => {
                    // Store the child node's representative block type so
                    // the shader can use a meaningful color at LOD cutoff.
                    let repr = library.get(*child_id)
                        .map(|n| n.representative_block)
                        .unwrap_or(0);
                    GpuChild {
                        tag: 2,
                        block_type: repr,
                        _pad: 0,
                        node_index: *visited.get(child_id).expect("child must be visited"),
                    }
                },
            });
        }
    }

    let metas: Vec<GpuNodeMeta> = ordered
        .iter()
        .map(|&nid| library.get(nid).map(|n| GpuNodeMeta::from_kind(&n.kind)).unwrap_or_default())
        .collect();

    let root_idx = *visited.get(&root).unwrap();
    (data, metas, root_idx)
}

/// LOD-aware tree packing: only uploads nodes large enough to see.
///
/// - **Uniform flattening:** Nodes where the entire subtree is one
///   block type (or all empty) are packed as Block/Empty — the shader
///   never descends into solid mountains or air volumes.
/// - **Distance culling:** Nodes whose screen-space size is below a
///   threshold are packed as Block(representative_block). Nearby terrain
///   gets full depth, distant terrain gets 1-2 levels.
/// - **Presence-preserving:** Nodes with representative_block=255 (all
///   empty subtree) are flattened to Empty, not solid grey.
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera_pos: [f32; 3],
    screen_height: f32,
    fov: f32,
) -> (Vec<GpuChild>, u32) {
    use super::tree::{UNIFORM_EMPTY, UNIFORM_MIXED, slot_coords};

    let half_fov_recip = screen_height / (2.0 * (fov * 0.5).tan());
    const LOD_THRESHOLD: f32 = 1.5; // pixels — stop descending below this

    // BFS with position tracking.
    struct QueueEntry {
        node_id: NodeId,
        origin: [f32; 3],  // world-space min corner of this node
        cell_size: f32,     // size of one child cell within this node
    }

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    // Track which children should be flattened (by ordered index + slot).
    // For each ordered node, store 27 overrides: None = use real child,
    // Some(GpuChild) = use this flattened value.
    let mut overrides: Vec<[Option<GpuChild>; CHILDREN_PER_NODE]> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    overrides.push([None; CHILDREN_PER_NODE]);
    queue.push(QueueEntry {
        node_id: root,
        origin: [0.0; 3],
        cell_size: 1.0, // root cells are 1.0 wide, node spans [0,3)
    });
    let mut head = 0;

    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_origin = entry.origin;
        let cell_size = entry.cell_size;
        let ordered_idx = head; // queue and ordered are aligned
        head += 1;

        let Some(node) = library.get(node_id) else { continue };

        for (slot, child) in node.children.iter().enumerate() {
            if let Child::Node(child_id) = child {
                let child_node = match library.get(*child_id) {
                    Some(n) => n,
                    None => continue,
                };

                // (B) Uniform flattening: entire subtree is one type.
                if child_node.uniform_type != UNIFORM_MIXED {
                    let gpu = if child_node.uniform_type == UNIFORM_EMPTY {
                        GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                    } else {
                        GpuChild { tag: 1, block_type: child_node.uniform_type, _pad: 0, node_index: 0 }
                    };
                    overrides[ordered_idx][slot] = Some(gpu);
                    continue;
                }

                // (C) Distance-aware culling: is this cell large enough on screen?
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
                    // Too small to see detail — flatten to representative color.
                    // Presence-preserving: if representative_block is 255
                    // (all-empty subtree), flatten to Empty so the ray
                    // passes through instead of hitting a grey wall.
                    let gpu = if child_node.representative_block < 255 {
                        GpuChild { tag: 1, block_type: child_node.representative_block, _pad: 0, node_index: 0 }
                    } else {
                        GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                    };
                    overrides[ordered_idx][slot] = Some(gpu);
                    continue;
                }

                // This node is visible — add to BFS if not already visited.
                if !visited.contains_key(child_id) {
                    let idx = ordered.len() as u32;
                    visited.insert(*child_id, idx);
                    ordered.push(*child_id);
                    overrides.push([None; CHILDREN_PER_NODE]);
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

    // Pack into flat buffer, applying overrides.
    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    for (oi, &nid) in ordered.iter().enumerate() {
        let Some(node) = library.get(nid) else {
            // Shouldn't happen on a consistent library — a node was
            // enqueued via `library.get(...)` succeeding but is now
            // missing. Emit empty children so packing stays in lock-
            // step with `ordered` (one node slot per entry).
            for _ in 0..CHILDREN_PER_NODE {
                data.push(GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 });
            }
            continue;
        };
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
    (data, root_idx)
}

/// Like `pack_tree_lod`, but packs multiple root subtrees into the
/// same flat buffer. Used when the engine needs several independent
/// trees on the GPU — the Cartesian space tree *and* each of a
/// spherical planet's 6 face subtrees, for instance. Returns the
/// buffer index of each root in the same order given. Distance-LOD
/// is applied to the first root only (the Cartesian tree has a
/// meaningful world-space origin); face subtrees always pack full
/// depth because their axes don't live in world space.
pub fn pack_tree_lod_multi(
    library: &NodeLibrary,
    roots: &[NodeId],
    camera_pos: [f32; 3],
    screen_height: f32,
    fov: f32,
) -> (Vec<GpuChild>, Vec<GpuNodeMeta>, Vec<u32>) {
    pack_tree_lod_multi_with_frame(
        library, roots, camera_pos, screen_height, fov,
        [0.0, 0.0, 0.0], 1.0,
    )
}

/// Variant of `pack_tree_lod_multi` that declares where the first
/// root sits in world space. `root_origin` is the world-space min
/// corner; `root_cell_size` is the world width of one root-cell (so
/// the root node spans `[root_origin, root_origin + 3·root_cell_size)`).
///
/// LOD culling uses the actual world-space cell size at each level,
/// so rendering a sub-root frame correctly treats its cells as larger
/// (or smaller) than the world root. Secondary roots (i.e., the six
/// cubed-sphere face subtrees) still use the legacy `[0, 1)` cell
/// size because their axes don't live in world space — LOD is
/// disabled for them, as before.
pub fn pack_tree_lod_multi_with_frame(
    library: &NodeLibrary,
    roots: &[NodeId],
    camera_pos: [f32; 3],
    screen_height: f32,
    fov: f32,
    root_origin: [f32; 3],
    root_cell_size: f32,
) -> (Vec<GpuChild>, Vec<GpuNodeMeta>, Vec<u32>) {
    use super::tree::{UNIFORM_EMPTY, UNIFORM_MIXED, slot_coords};

    let half_fov_recip = screen_height / (2.0 * (fov * 0.5).tan());
    const LOD_THRESHOLD: f32 = 1.5;

    struct QueueEntry {
        node_id: NodeId,
        origin: [f32; 3],
        cell_size: f32,
        use_lod: bool,
    }

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut overrides: Vec<[Option<GpuChild>; CHILDREN_PER_NODE]> = Vec::new();

    for (ri, &root) in roots.iter().enumerate() {
        if visited.contains_key(&root) { continue; }
        let idx = ordered.len() as u32;
        visited.insert(root, idx);
        ordered.push(root);
        overrides.push([None; CHILDREN_PER_NODE]);
        let (origin, cell_size) = if ri == 0 {
            (root_origin, root_cell_size)
        } else {
            ([0.0; 3], 1.0)
        };
        queue.push(QueueEntry {
            node_id: root,
            origin,
            cell_size,
            use_lod: ri == 0,
        });
    }
    let mut head = 0;
    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_origin = entry.origin;
        let cell_size = entry.cell_size;
        let use_lod = entry.use_lod;
        let ordered_idx = head;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };

        for (slot, child) in node.children.iter().enumerate() {
            if let Child::Node(child_id) = child {
                let child_node = match library.get(*child_id) {
                    Some(n) => n,
                    None => continue,
                };
                // Sphere body / face nodes must reach the GPU as
                // full Node children — the cubed-sphere DDA reads
                // them via `child_node_index`. Skip uniform / LOD
                // flattening that would convert them to a Block.
                let preserve_node = !child_node.kind.is_cartesian();
                if !preserve_node && child_node.uniform_type != UNIFORM_MIXED {
                    let gpu = if child_node.uniform_type == UNIFORM_EMPTY {
                        GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }
                    } else {
                        GpuChild { tag: 1, block_type: child_node.uniform_type, _pad: 0, node_index: 0 }
                    };
                    overrides[ordered_idx][slot] = Some(gpu);
                    continue;
                }
                if use_lod && !preserve_node {
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
                        use_lod,
                    });
                }
            }
        }
    }

    let mut data: Vec<GpuChild> = Vec::with_capacity(ordered.len() * GPU_NODE_SIZE);
    for (oi, &nid) in ordered.iter().enumerate() {
        let Some(node) = library.get(nid) else {
            // Shouldn't happen on a consistent library — a node was
            // enqueued via `library.get(...)` succeeding but is now
            // missing. Emit empty children so packing stays in lock-
            // step with `ordered` (one node slot per entry).
            for _ in 0..CHILDREN_PER_NODE {
                data.push(GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 });
            }
            continue;
        };
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

    // Parallel per-node metadata buffer. One entry per packed node
    // in the same BFS order the tree buffer uses.
    let metas: Vec<GpuNodeMeta> = ordered
        .iter()
        .map(|&nid| {
            library.get(nid)
                .map(|n| GpuNodeMeta::from_kind(&n.kind))
                .unwrap_or_default()
        })
        .collect();

    let root_indices: Vec<u32> = roots
        .iter()
        .map(|r| *visited.get(r).expect("every root must be visited"))
        .collect();
    (data, metas, root_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_test_world() {
        let world = super::super::state::WorldState::test_world();
        let (data, _metas, root_idx) = pack_tree(&world.library, world.root);
        // Verify data is a multiple of 27 (each node is 27 children).
        assert_eq!(data.len() % 27, 0);
        // Root is always first in BFS order.
        assert_eq!(root_idx, 0);
        // Should have all reachable nodes packed.
        assert_eq!(data.len() / 27, world.library.len());
    }

    #[test]
    fn gpu_child_size() {
        assert_eq!(std::mem::size_of::<GpuChild>(), 8);
    }
}
