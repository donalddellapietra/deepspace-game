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

/// Block color palette — 10 RGBA colors, one per BlockType.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPalette {
    pub colors: [[f32; 4]; 16], // 10 used + 6 padding for alignment
}

impl Default for GpuPalette {
    fn default() -> Self {
        let mut colors = [[0.0; 4]; 16];
        colors[0]  = [0.5, 0.5, 0.5, 1.0];    // Stone
        colors[1]  = [0.45, 0.3, 0.15, 1.0];   // Dirt
        colors[2]  = [0.3, 0.6, 0.2, 1.0];     // Grass
        colors[3]  = [0.55, 0.35, 0.15, 1.0];  // Wood
        colors[4]  = [0.2, 0.5, 0.1, 1.0];     // Leaf
        colors[5]  = [0.85, 0.8, 0.55, 1.0];   // Sand
        colors[6]  = [0.2, 0.4, 0.8, 1.0];     // Water
        colors[7]  = [0.7, 0.3, 0.2, 1.0];     // Brick
        colors[8]  = [0.75, 0.75, 0.8, 1.0];   // Metal
        colors[9]  = [0.85, 0.9, 1.0, 1.0];    // Glass
        Self { colors }
    }
}

/// Pack the visible portion of the tree into a flat GPU buffer.
/// Returns (node_data, root_buffer_index).
pub fn pack_tree(
    library: &NodeLibrary,
    root: NodeId,
) -> (Vec<GpuChild>, u32) {
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
        let node = library.get(nid).expect("node in ordered list must exist");
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
                    block_type: *bt as u8,
                    _pad: 0,
                    node_index: 0,
                },
                Child::Node(child_id) => {
                    // Store the child node's dominant block type so the
                    // shader can use a meaningful color at LOD cutoff.
                    let dominant = library.get(*child_id)
                        .map(|n| n.dominant_block)
                        .unwrap_or(0);
                    GpuChild {
                        tag: 2,
                        block_type: dominant,
                        _pad: 0,
                        node_index: *visited.get(child_id).expect("child must be visited"),
                    }
                },
            });
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, root_idx)
}

/// LOD-aware tree packing: only uploads nodes large enough to see.
///
/// - **Uniform flattening (B):** Nodes where the entire subtree is one
///   block type (or all empty) are packed as Block/Empty — the shader
///   never descends into solid mountains or air volumes.
/// - **Distance culling (C):** Nodes whose screen-space size is below
///   a threshold are packed as Block(dominant_color). Nearby terrain
///   gets full depth, distant terrain gets 1-2 levels.
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
                    // Too small to see detail — flatten to dominant color.
                    let gpu = if child_node.dominant_block < 255 {
                        GpuChild { tag: 1, block_type: child_node.dominant_block, _pad: 0, node_index: 0 }
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
        let node = library.get(nid).expect("node in ordered list must exist");
        for (slot, child) in node.children.iter().enumerate() {
            if let Some(gpu) = overrides[oi][slot] {
                data.push(gpu);
            } else {
                data.push(match child {
                    Child::Empty => GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 },
                    Child::Block(bt) => GpuChild { tag: 1, block_type: *bt as u8, _pad: 0, node_index: 0 },
                    Child::Node(child_id) => {
                        let dominant = library.get(*child_id).map(|n| n.dominant_block).unwrap_or(0);
                        let idx = visited.get(child_id).copied().unwrap_or(0);
                        GpuChild { tag: 2, block_type: dominant, _pad: 0, node_index: idx }
                    },
                });
            }
        }
    }

    let root_idx = *visited.get(&root).unwrap();
    (data, root_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_test_world() {
        let world = super::super::state::WorldState::test_world();
        let (data, root_idx) = pack_tree(&world.library, world.root);
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
