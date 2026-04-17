//! BFS packing of the world tree into the interleaved GPU layout.
//!
//! A single `tree: Vec<u32>` buffer carries per-node headers and
//! payloads inline. Three node flavours share the buffer:
//!
//! ## Sparse (Cartesian / Sphere / Face)
//!
//! ```text
//! tree[base + 0]                         = occupancy mask (27 bits)
//! tree[base + 1]                         = first_child_offset
//! tree[first_child + rank*2]             = packed (tag | block_type | pad)
//! tree[first_child + rank*2 + 1]         = child.node_index (BFS idx,
//!                                           valid when tag == 2)
//! ```
//!
//! Total: `2 + 2 * popcount(occupancy)` u32s. `first_child_offset` is
//! absolute u32-offset into `tree[]`. BFS lays children contiguously
//! after the header so header + first child share one 64 B cache line.
//!
//! ## Brick (NodeKind::Brick)
//!
//! ```text
//! tree[base + 0]                     = BRICK_FLAG_BIT
//!                                        | (side_code << SIDE_SHIFT)
//! tree[base + 1]                     = brick_data_offset (= base + 2)
//! tree[base + 2 .. base + 2 + N]     = side³ u8 cells, 4 per u32,
//!                                       little-endian, slot =
//!                                       x + y*side + z*side²
//! ```
//!
//! where `side ∈ {3, 9, 27}` is chosen per-brick at library insertion
//! time, `side_code` is `{0: 3, 1: 9, 2: 27}` stored in bits
//! 28-29, and `N = ceil(side³ / 4)`. Empty cells use the 255 sentinel.
//!
//! Bricks replace 1-3 levels of recursion with a flat byte array. The
//! shader dispatches on `NodeKindGpu::kind == GPU_NODE_KIND_BRICK` and
//! walks the grid directly via `march_brick`.
//!
//! ## Why content-defined bricks
//!
//! Force-collapse at a fixed pack-time depth (previous approach) had
//! to use `representative_block` to summarize mixed subtrees, which
//! destroyed detail. Library-materialized bricks carry **real voxel
//! bytes** so they are lossless. Bricks only appear where the library
//! promises terminal content, so the packer's only job is to emit
//! them in the right format.
//!
//! ## Two side buffers
//!
//! - `node_kinds: Vec<GpuNodeKind>` — one per BFS node. Carries the
//!   dispatch discriminant (Cartesian / Sphere body / Sphere face /
//!   Brick) plus per-kind data (radii, face id). Touched only at
//!   descent/pop; inner DDA loops never read this.
//! - `node_offsets: Vec<u32>` — BFS index → header u32-offset. Also
//!   cold-path only.
//!
//! ## Public entry points
//!
//! - `pack_tree`: full BFS, no LOD. Used by tests and ribbon setup.
//! - `pack_tree_lod`: per-pixel-LOD packing with no preserve paths.
//! - `pack_tree_lod_preserving`: with ribbon preserve paths.
//! - `pack_tree_lod_selective`: with preserve paths + wide preserve
//!   regions (the renderer's hot call).

use std::collections::{HashMap, HashSet};

use crate::world::anchor::{Path, WorldPos};
use crate::world::tree::{
    slot_coords, Child, NodeId, NodeKind, NodeLibrary, BRICK_SIDES, CHILDREN_PER_NODE,
    UNIFORM_EMPTY, UNIFORM_MIXED,
};

use super::types::{GpuChild, GpuNodeKind};

// ---------------------------------------------------------- brick header

/// Bit 27 of the brick header word marks a brick node. Kept in sync
/// with `BRICK_FLAG_BIT` in `assets/shaders/bindings.wgsl`.
const BRICK_FLAG_BIT: u32 = 1 << 27;

/// Bits 28-29 of the brick header carry the side code. Code → side:
/// 0 → 3, 1 → 9, 2 → 27.
const BRICK_SIDE_SHIFT: u32 = 28;
const BRICK_SIDE_MASK: u32 = 0x3 << BRICK_SIDE_SHIFT;

/// Empty-cell sentinel inside a brick's byte array. Matches
/// `BRICK_EMPTY_BT` in `bindings.wgsl`.
const BRICK_EMPTY_BT: u8 = 255;

/// Map brick side → 2-bit side code stored in the header. Must be
/// invertible via `side_from_code`.
pub fn side_to_code(side: u8) -> u32 {
    match side {
        3 => 0,
        9 => 1,
        27 => 2,
        _ => panic!("unsupported brick side {side}"),
    }
}

/// Inverse of `side_to_code`. Used by GPU-buffer inspection tests.
pub fn side_from_code(code: u32) -> u8 {
    match code {
        0 => 3,
        1 => 9,
        2 => 27,
        _ => panic!("unsupported brick side code {code}"),
    }
}

/// u32s needed to carry `side³` bytes, 4 bytes per u32 (last u32
/// may carry fewer used bytes).
#[inline]
fn brick_data_u32s(side: u8) -> u32 {
    let cells = (side as u32).pow(3);
    (cells + 3) / 4
}

/// Total u32 footprint of a brick node: 2-word header + data words.
#[inline]
fn brick_total_u32s(side: u8) -> u32 {
    2 + brick_data_u32s(side)
}

/// Pack a slice of `side³` per-cell block-type bytes into
/// `brick_data_u32s(side)` u32s, little-endian within each word.
fn pack_brick_words(cells: &[u8], side: u8) -> Vec<u32> {
    let expected = (side as usize).pow(3);
    debug_assert_eq!(
        cells.len(),
        expected,
        "brick side {} wants {} cells, got {}",
        side,
        expected,
        cells.len(),
    );
    let n_words = brick_data_u32s(side) as usize;
    let mut words = vec![0u32; n_words];
    for (i, &bt) in cells.iter().enumerate() {
        words[i / 4] |= (bt as u32) << ((i % 4) * 8);
    }
    words
}

/// Encode a sparse child's first u32: `tag | (block_type << 8) | (pad << 16)`.
fn pack_child_first(c: GpuChild) -> u32 {
    (c.tag as u32) | ((c.block_type as u32) << 8) | ((c._pad as u32) << 16)
}

// --------------------------------------------------------------- output

/// Result of packing: `(tree, node_kinds, node_offsets, root_bfs_index)`.
pub type PackedTree = (Vec<u32>, Vec<GpuNodeKind>, Vec<u32>, u32);

// ---------------------------------------------------------- pack_tree

/// Emit the full tree rooted at `root` without any LOD. Every
/// Cartesian / Sphere / Face node is packed sparse; every
/// `NodeKind::Brick` node is packed as a brick of its stored side.
pub fn pack_tree(library: &NodeLibrary, root: NodeId) -> PackedTree {
    // Phase 1: BFS-visit every node reachable from root. Assign BFS
    // index on first visit; brick nodes have no children so they end
    // the walk on their branch.
    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    visited.insert(root, 0);
    ordered.push(root);
    let mut head = 0usize;
    while head < ordered.len() {
        let nid = ordered[head];
        head += 1;
        let Some(node) = library.get(nid) else { continue };
        if node.is_brick() {
            continue;
        }
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

    // Phase 2: compute occupancies + per-node offsets.
    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    for &nid in &ordered {
        let node = library.get(nid).expect("node in ordered list must exist");
        if node.is_brick() {
            occupancies.push(0);
            continue;
        }
        let mut occ: u32 = 0;
        for (slot, child) in node.children.iter().enumerate() {
            if !matches!(child, Child::Empty) {
                occ |= 1u32 << slot;
            }
        }
        occupancies.push(occ);
    }

    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for i in 0..n_nodes {
        node_offsets.push(running);
        let node = library
            .get(ordered[i])
            .expect("node in ordered list must exist");
        running += if node.is_brick() {
            brick_total_u32s(node.brick_side)
        } else {
            2 + 2 * occupancies[i].count_ones()
        };
    }
    let total_u32s = running as usize;

    // Phase 3: emit interleaved tree[].
    let mut tree: Vec<u32> = Vec::with_capacity(total_u32s);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(n_nodes);
    for (i, &nid) in ordered.iter().enumerate() {
        let node = library.get(nid).expect("node in ordered list must exist");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        let header_off = node_offsets[i];
        debug_assert_eq!(tree.len() as u32, header_off);

        if node.is_brick() {
            emit_brick(&mut tree, header_off, node.brick_side, &node.brick_cells);
            continue;
        }

        let occupancy = occupancies[i];
        let first_child_off = header_off + 2;
        tree.push(occupancy);
        tree.push(first_child_off);
        for (slot, child) in node.children.iter().enumerate() {
            let entry = match child {
                Child::Empty => continue,
                Child::Block(bt) => GpuChild {
                    tag: 1, block_type: *bt, _pad: 0, node_index: 0,
                },
                Child::Node(child_id) => {
                    let repr = library
                        .get(*child_id)
                        .map(|n| n.representative_block)
                        .unwrap_or(0);
                    GpuChild {
                        tag: 2,
                        block_type: repr,
                        _pad: 0,
                        node_index: *visited.get(child_id).expect("child visited"),
                    }
                }
            };
            debug_assert_ne!(occupancy & (1u32 << slot), 0);
            tree.push(pack_child_first(entry));
            tree.push(entry.node_index);
        }
    }
    debug_assert_eq!(tree.len(), total_u32s);

    let root_idx = *visited.get(&root).unwrap();
    (tree, kinds, node_offsets, root_idx)
}

/// Write a brick's header + data into `tree` at `header_off`. Caller
/// must have reserved `brick_total_u32s(side)` u32s there.
fn emit_brick(tree: &mut Vec<u32>, header_off: u32, side: u8, cells: &[u8]) {
    let header = BRICK_FLAG_BIT | (side_to_code(side) << BRICK_SIDE_SHIFT);
    let data_off = header_off + 2;
    tree.push(header);
    tree.push(data_off);
    let words = pack_brick_words(cells, side);
    for w in words {
        tree.push(w);
    }
}

// ---------------------------------------------------- screen-pixel LOD

/// Per-child screen-pixel estimate in the parent node's local frame.
/// Matches the shader's `lod_pixels = cell_size / ray_dist *
/// screen_height / (2 tan(fov/2))`, sampled at pack time using the
/// child cell's centre instead of the per-ray DDA position.
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

/// Legacy parameter name retained for API compat. Previously was the
/// force-collapse depth; now unused (all pack-time LOD is per-pixel).
pub const DEFAULT_LOD_LEAF_DEPTH: u32 = 4;

// ------------------------------------------------------- pack_tree_lod

/// LOD-aware tree packing with per-pixel truncation.
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
) -> PackedTree {
    pack_tree_lod_selective(
        library, root, camera, screen_height, fov, &[], &[], DEFAULT_LOD_LEAF_DEPTH,
    )
}

/// `pack_tree_lod` with one or more `preserve_paths` (ribbon chains
/// that must remain sparse so the CPU ribbon builder can descend).
pub fn pack_tree_lod_preserving(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
) -> PackedTree {
    pack_tree_lod_selective(
        library, root, camera, screen_height, fov,
        preserve_paths, &[], DEFAULT_LOD_LEAF_DEPTH,
    )
}

/// Full-featured packer used by the renderer.
///
/// - `preserve_paths`: exact NodeId-slot chains (ribbon ancestors)
///   that skip per-pixel collapse so the ribbon builder can walk them.
/// - `preserve_regions`: wide near-camera (frame-path, extra-depth)
///   regions that also skip per-pixel collapse — content inside these
///   regions retains full detail regardless of projected size.
/// - `_lod_leaf_depth`: legacy parameter, ignored.
///
/// LOD policy: the only LOD is per-pixel — Cartesian subtree children
/// whose projected size is below `LOD_PIXEL_THRESHOLD` get collapsed
/// to a tag=1 terminal with `representative_block`. Bricks are ALWAYS
/// packed as-is (they're terminal by construction). Sphere/face
/// children are never collapsed (they carry content-specific data).
pub fn pack_tree_lod_selective(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
    preserve_regions: &[(Path, u8)],
    _lod_leaf_depth: u32,
) -> PackedTree {
    const LOD_PIXEL_THRESHOLD: f32 = 1.0;

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

    /// Per-slot pack result for a single packed node. `None` = empty
    /// (not in occupancy); `Some(child)` with `child.tag in {1, 2}`
    /// = non-empty.
    type SlotOverride = [Option<GpuChild>; CHILDREN_PER_NODE];

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut per_node: Vec<SlotOverride> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    per_node.push([None; CHILDREN_PER_NODE]);
    queue.push(QueueEntry { node_id: root, path: Path::root() });

    let mut head = 0usize;
    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_path = entry.path;
        let ordered_idx = head;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };

        // Brick nodes have no recursive children — nothing to enqueue.
        if node.is_brick() {
            continue;
        }

        let lod_active = matches!(node.kind, NodeKind::Cartesian);

        for (slot, child) in node.children.iter().enumerate() {
            let stored = match child {
                Child::Empty => None,
                Child::Block(bt) => Some(GpuChild {
                    tag: 1, block_type: *bt, _pad: 0, node_index: 0,
                }),
                Child::Node(_) => None,
            };
            if let Some(c) = stored {
                per_node[ordered_idx][slot] = Some(c);
            }

            let Child::Node(child_id) = child else { continue };
            let Some(child_node) = library.get(*child_id) else { continue };

            let mut child_path = node_path;
            child_path.push(slot as u8);
            let on_preserve = preserve_pairs.contains(&(node_id, slot as u8))
                || in_preserve_region(&child_path, &preserve_regions);
            let child_is_cartesian = matches!(child_node.kind, NodeKind::Cartesian);

            // Uniform subtree collapse (LOSSLESS). A Cartesian
            // subtree that's entirely one block type (or entirely
            // empty) gets flattened to a single tag=1 terminal. No
            // detail is lost — the subtree IS that value everywhere.
            if !on_preserve && lod_active && child_is_cartesian
                && child_node.uniform_type != UNIFORM_MIXED
            {
                per_node[ordered_idx][slot] = if child_node.uniform_type == UNIFORM_EMPTY {
                    None
                } else {
                    Some(GpuChild {
                        tag: 1,
                        block_type: child_node.uniform_type,
                        _pad: 0,
                        node_index: 0,
                    })
                };
                continue;
            }

            // Per-pixel LOD. Bricks are never collapsed (they're
            // terminal content — the whole point is to descend into
            // them cheaply). Non-Cartesian recursive kinds aren't
            // collapsed either (sphere/face need their own DDA).
            if !on_preserve && lod_active && child_is_cartesian {
                let pixels = child_screen_pixels(
                    camera, &node_path, slot, screen_height, fov,
                );
                if pixels < LOD_PIXEL_THRESHOLD {
                    per_node[ordered_idx][slot] = if child_node.representative_block < 255 {
                        Some(GpuChild {
                            tag: 1,
                            block_type: child_node.representative_block,
                            _pad: 0,
                            node_index: 0,
                        })
                    } else {
                        None
                    };
                    continue;
                }
            }

            if !visited.contains_key(child_id) {
                let idx = ordered.len() as u32;
                visited.insert(*child_id, idx);
                ordered.push(*child_id);
                per_node.push([None; CHILDREN_PER_NODE]);
                queue.push(QueueEntry {
                    node_id: *child_id,
                    path: child_path,
                });
            }
            let child_idx = *visited.get(child_id).expect("child just visited");
            per_node[ordered_idx][slot] = Some(GpuChild {
                tag: 2,
                block_type: child_node.representative_block,
                _pad: 0,
                node_index: child_idx,
            });
        }
    }

    // Phase 2: compute header offsets. Each node's footprint depends
    // on whether it's a brick (fixed by brick_side) or sparse (by
    // occupancy popcount).
    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    for (i, slots) in per_node.iter().enumerate() {
        let node = library.get(ordered[i]).expect("node in ordered list");
        if node.is_brick() {
            occupancies.push(0);
            continue;
        }
        let mut occ: u32 = 0;
        for (slot, entry) in slots.iter().enumerate() {
            if entry.is_some() {
                occ |= 1u32 << slot;
            }
        }
        occupancies.push(occ);
    }

    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for i in 0..n_nodes {
        node_offsets.push(running);
        let node = library.get(ordered[i]).expect("node in ordered list");
        running += if node.is_brick() {
            brick_total_u32s(node.brick_side)
        } else {
            2 + 2 * occupancies[i].count_ones()
        };
    }
    let total_u32s = running as usize;

    // Phase 3: emit.
    let mut tree: Vec<u32> = Vec::with_capacity(total_u32s);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(n_nodes);
    for (i, &node_id) in ordered.iter().enumerate() {
        let node = library.get(node_id).expect("node in ordered list");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        let header_off = node_offsets[i];
        debug_assert_eq!(tree.len() as u32, header_off);

        if node.is_brick() {
            emit_brick(&mut tree, header_off, node.brick_side, &node.brick_cells);
            continue;
        }

        let occupancy = occupancies[i];
        let first_child_off = header_off + 2;
        tree.push(occupancy);
        tree.push(first_child_off);
        for slot in 0..CHILDREN_PER_NODE {
            if let Some(entry) = per_node[i][slot] {
                tree.push(pack_child_first(entry));
                tree.push(entry.node_index);
            }
        }
    }
    debug_assert_eq!(tree.len(), total_u32s);

    let root_idx = *visited.get(&root).unwrap();
    (tree, kinds, node_offsets, root_idx)
}

// ---------------------------------------------------------- test helpers

/// Valid brick sides, re-exported for tests.
pub const VALID_BRICK_SIDES: [u8; 3] = BRICK_SIDES;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::anchor::WorldPos;
    use crate::world::bootstrap::{menger_world, plain_test_world, plain_world};
    use crate::world::tree::{empty_children, uniform_children, CENTER_SLOT};

    /// Read a sparse node's slot. Returns a synthesized tag=0 stub
    /// when the slot is empty. Panics on brick nodes — tests that
    /// want brick data should use `brick_cell`.
    pub(super) fn sparse_child(
        tree: &[u32],
        node_offsets: &[u32],
        bfs_idx: u32,
        slot: u8,
    ) -> GpuChild {
        let header_off = node_offsets[bfs_idx as usize] as usize;
        let occupancy = tree[header_off];
        assert!(
            (occupancy & BRICK_FLAG_BIT) == 0,
            "sparse_child() called on brick node bfs_idx={}",
            bfs_idx,
        );
        let first_child = tree[header_off + 1] as usize;
        let bit = 1u32 << slot;
        if occupancy & bit == 0 {
            return GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 };
        }
        let rank = (occupancy & (bit - 1)).count_ones() as usize;
        let off = first_child + rank * 2;
        let packed = tree[off];
        let tag = (packed & 0xFF) as u8;
        let block_type = ((packed >> 8) & 0xFF) as u8;
        let _pad = ((packed >> 16) & 0xFFFF) as u16;
        let node_index = tree[off + 1];
        GpuChild { tag, block_type, _pad, node_index }
    }

    /// Read a brick node's (x, y, z) cell byte.
    pub(super) fn brick_cell(
        tree: &[u32],
        node_offsets: &[u32],
        bfs_idx: u32,
        x: usize,
        y: usize,
        z: usize,
    ) -> u8 {
        let header_off = node_offsets[bfs_idx as usize] as usize;
        let header = tree[header_off];
        assert!(
            (header & BRICK_FLAG_BIT) != 0,
            "brick_cell() called on sparse node bfs_idx={}",
            bfs_idx,
        );
        let side = side_from_code((header & BRICK_SIDE_MASK) >> BRICK_SIDE_SHIFT) as usize;
        let data_off = tree[header_off + 1] as usize;
        let idx = z * side * side + y * side + x;
        let word = tree[data_off + idx / 4];
        ((word >> ((idx % 4) * 8)) & 0xFF) as u8
    }

    #[test]
    fn pack_test_world() {
        let world = plain_test_world();
        let (tree, kinds, node_offsets, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(root_idx, 0);
        assert_eq!(kinds.len(), world.library.len());
        assert_eq!(node_offsets.len(), world.library.len());
        assert!(!tree.is_empty());
        assert_eq!(node_offsets[0], 0);
        for kind in &kinds {
            // No bricks in the plain-test preset.
            assert_ne!(kind.kind, super::super::types::GPU_NODE_KIND_BRICK);
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
        WorldPos::from_frame_local(&crate::world::anchor::Path::root(), xyz, 2)
            .deepened_to(10)
    }

    #[test]
    fn pack_includes_body_kind_and_radii() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (_tree, kinds, _offsets, _root_idx) =
            pack_tree_lod(&world.library, world.root, &camera, 1080.0, 1.2);
        let body = kinds.iter().find(|kind| kind.kind == 1).expect("body kind in buffer");
        assert!((body.inner_r - 0.12).abs() < 1e-6);
        assert!((body.outer_r - 0.45).abs() < 1e-6);
    }

    #[test]
    fn pack_lod_flattens_far_uniform_cartesian() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (tree, _kinds, offsets, _root_idx) =
            pack_tree_lod(&world.library, world.root, &camera, 1080.0, 1.2);
        assert_eq!(sparse_child(&tree, &offsets, 0, 0).tag, 0);
        let body_entry = sparse_child(&tree, &offsets, 0, 13);
        assert_eq!(body_entry.tag, 2);
        assert!(body_entry.node_index > 0);
    }

    #[test]
    fn preserve_path_prevents_collapse() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera = camera_at([1.5, 2.0, 1.5]);

        let (tree, _, offsets, _) = pack_tree_lod(
            &lib, root, &camera, 1080.0, 1.2,
        );
        assert_eq!(sparse_child(&tree, &offsets, 0, 16).tag, 0);

        let (tree2, _, offsets2, _) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&[16u8]],
        );
        let preserved = sparse_child(&tree2, &offsets2, 0, 16);
        assert_eq!(preserved.tag, 2);
        assert!(preserved.node_index > 0);
    }

    #[test]
    fn preserve_path_chain_lets_ribbon_descend() {
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

        let (tree, _kinds, offsets, _root_idx) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&path],
        );
        let ribbon = build_ribbon(&tree, &offsets, &path);
        assert_eq!(ribbon.reached_slots.len(), 9);
        assert_eq!(ribbon.ribbon.len(), 9);
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
        let (tree, _kinds, offsets, _root_idx) =
            pack_tree_lod(&lib, root, &camera, 1080.0, 1.2);
        assert_eq!(sparse_child(&tree, &offsets, 0, CENTER_SLOT as u8).tag, 2);
    }

    #[test]
    fn brick_is_packed_inline() {
        let mut lib = NodeLibrary::default();
        let mut cells = vec![BRICK_EMPTY_BT; 27];
        cells[0] = crate::world::palette::block::STONE;
        cells[13] = crate::world::palette::block::GRASS;
        let brick_id = lib.insert_brick(cells, 3);
        let mut root_children = empty_children();
        root_children[CENTER_SLOT] = Child::Node(brick_id);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let (tree, kinds, offsets, _root_idx) = pack_tree(&lib, root);
        // Root is Cartesian sparse; CENTER slot points to brick.
        let center = sparse_child(&tree, &offsets, 0, CENTER_SLOT as u8);
        assert_eq!(center.tag, 2);
        let brick_bfs_idx = center.node_index;
        assert_eq!(
            kinds[brick_bfs_idx as usize].kind,
            super::super::types::GPU_NODE_KIND_BRICK,
        );
        // Probe brick cells directly from the buffer.
        assert_eq!(brick_cell(&tree, &offsets, brick_bfs_idx, 0, 0, 0), crate::world::palette::block::STONE);
        assert_eq!(brick_cell(&tree, &offsets, brick_bfs_idx, 1, 1, 1), crate::world::palette::block::GRASS);
        assert_eq!(brick_cell(&tree, &offsets, brick_bfs_idx, 2, 2, 2), BRICK_EMPTY_BT);
    }

    /// Write a distinctive voxel pattern into a side-9 brick and verify
    /// every cell round-trips through the packed GPU buffer. Catches
    /// slot-encoding / byte-ordering mismatches between pack.rs and
    /// march_brick's `slot = x + y*side + z*side²`.
    #[test]
    fn brick_side9_full_roundtrip() {
        let side: u8 = 9;
        let s = side as usize;
        // Distinctive: cell value = 1 + (x + y*2 + z*3) mod 200 so
        // every position has a unique (and non-255) byte.
        let mut cells = vec![0u8; s * s * s];
        for z in 0..s { for y in 0..s { for x in 0..s {
            cells[z * s * s + y * s + x] = 1 + ((x + y * 2 + z * 3) % 200) as u8;
        }}}

        let mut lib = NodeLibrary::default();
        let brick_id = lib.insert_brick(cells.clone(), side);
        let mut root_children = empty_children();
        root_children[CENTER_SLOT] = Child::Node(brick_id);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let (tree, _kinds, offsets, _root_idx) = pack_tree(&lib, root);
        let brick_bfs_idx = sparse_child(&tree, &offsets, 0, CENTER_SLOT as u8).node_index;

        // Probe every cell.
        for z in 0..s { for y in 0..s { for x in 0..s {
            let expected = cells[z * s * s + y * s + x];
            let actual = brick_cell(&tree, &offsets, brick_bfs_idx, x, y, z);
            assert_eq!(
                actual, expected,
                "cell ({x}, {y}, {z}) mismatch: expected {expected}, got {actual}",
            );
        }}}
    }

    #[test]
    fn brick_side_encoding_round_trip() {
        for &side in &VALID_BRICK_SIDES {
            let mut cells = vec![BRICK_EMPTY_BT; (side as usize).pow(3)];
            cells[0] = crate::world::palette::block::STONE;
            let mut lib = NodeLibrary::default();
            let brick_id = lib.insert_brick(cells, side);
            let mut root_children = empty_children();
            root_children[CENTER_SLOT] = Child::Node(brick_id);
            let root = lib.insert(root_children);
            lib.ref_inc(root);

            let (tree, _kinds, offsets, _root_idx) = pack_tree(&lib, root);
            let brick_bfs_idx = sparse_child(
                &tree, &offsets, 0, CENTER_SLOT as u8,
            ).node_index;
            let header = tree[offsets[brick_bfs_idx as usize] as usize];
            assert!(header & BRICK_FLAG_BIT != 0);
            let code = (header & BRICK_SIDE_MASK) >> BRICK_SIDE_SHIFT;
            assert_eq!(side_from_code(code), side);
            assert_eq!(brick_cell(&tree, &offsets, brick_bfs_idx, 0, 0, 0), crate::world::palette::block::STONE);
        }
    }

    /// Baseline size check.
    #[test]
    fn menger_pack_size_regression() {
        let world = menger_world(5);
        let (tree, _kinds, _offsets, _root) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!("menger depth=5 pack size: {u32s} u32s ({} bytes)", u32s * 4);
        const MENGER_D5_MAX_U32S: usize = 320;
        assert!(u32s < MENGER_D5_MAX_U32S, "menger depth=5 pack size regressed: {u32s}");
    }

    #[test]
    fn plain_pack_size_regression() {
        let world = plain_world(5);
        let (tree, _kinds, _offsets, _root) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!("plain layers=5 pack size: {u32s} u32s ({} bytes)", u32s * 4);
        const PLAIN_L5_MAX_U32S: usize = 1400;
        assert!(u32s < PLAIN_L5_MAX_U32S, "plain layers=5 pack size regressed: {u32s}");
    }

    #[test]
    fn break_at_every_depth_changes_packed_data() {
        use crate::world::anchor::Path;
        use crate::world::bootstrap;
        use crate::world::edit;
        use crate::world::raycast;

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

            let (tree_before, _, offsets_before, _) = pack_tree_lod_selective(
                &world.library, world.root, &camera,
                720.0, 1.2,
                &preserve_paths, &preserve_regions,
                DEFAULT_LOD_LEAF_DEPTH,
            );

            let ray_origin = camera.in_frame(&Path::root());
            let ray_dir = [0.0f32, -0.4, -0.9];
            let hit = raycast::cpu_raycast(
                &world.library, world.root, ray_origin, ray_dir, spawn_depth as u32,
            );
            let Some(hit) = hit else {
                panic!("no raycast hit at spawn_depth={spawn_depth}");
            };
            assert!(edit::break_block(&mut world, &hit));

            let (tree_after, _, offsets_after, _) = pack_tree_lod_selective(
                &world.library, world.root, &camera,
                720.0, 1.2,
                &preserve_paths, &preserve_regions,
                DEFAULT_LOD_LEAF_DEPTH,
            );

            assert!(
                tree_before != tree_after || offsets_before != offsets_after,
                "packed GPU data unchanged after break at spawn_depth={spawn_depth}",
            );
        }
    }
}
