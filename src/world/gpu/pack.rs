//! BFS packing of the world tree into the interleaved sparse GPU
//! layout.
//!
//! A single storage buffer `tree: Vec<u32>` carries headers and
//! children interleaved per node. Each node occupies
//! `2 + 2*popcount(occupancy)` contiguous u32s:
//!
//! ```text
//! tree[base + 0]                     = occupancy mask (27 bits)
//! tree[base + 1]                     = first_child_offset
//! tree[first_child_offset + rank*2]     = packed (tag|block_type|pad)
//! tree[first_child_offset + rank*2 + 1] = child.node_index (BFS idx,
//!                                         valid when tag==2)
//! ```
//!
//! `first_child_offset` is an ABSOLUTE u32 offset into this same
//! `tree` buffer. In BFS layout, every node's children are packed
//! immediately after its 2-u32 header, so
//! `first_child_offset == base + 2` — the header + first child
//! entry share a 64-byte cache line.
//!
//! ## Brick nodes
//!
//! A Cartesian node whose children are ALL terminal (every slot is
//! either `Child::Empty` or `Child::Block(bt)` — no `Child::Node`)
//! is packed as a BRICK instead of sparse. The shader detects this
//! via `BRICK_FLAG_BIT` (bit 27) set in the occupancy mask, and
//! walks the node's 3×3×3 cells with a flat DDA — no popcount,
//! no per-cell rank indexing, no tree-recursion. Just
//! `block_type = brick[slot/4] >> ((slot%4)*8) & 0xFF` per cell.
//!
//! Brick layout (9 u32s total):
//!
//! ```text
//! tree[base + 0]      = occupancy_mask | BRICK_FLAG_BIT
//! tree[base + 1]      = first_child_offset (= base + 2; unused by
//!                        the shader's brick path but kept for layout
//!                        uniformity so node_offsets still works)
//! tree[base + 2..8]  = 27 u8 block types, 4 per u32, little-endian.
//!                        slot = x + y*3 + z*9, stored at
//!                        brick[slot/4] byte (slot%4).
//!                        block_type = 255 = "empty" sentinel.
//! ```
//!
//! Dense nodes save a lot of space as bricks: 27 fully-populated
//! terminal slots would cost 2 + 27*2 = 56 u32s as sparse but only
//! 9 u32s as brick. Symmetric with layer semantics — a node is
//! brickable iff its content is brickable, irrespective of depth.
//!
//! Two parallel side-buffers:
//!
//! - `node_kinds: Vec<GpuNodeKind>` — indexed by BFS position.
//!   Carries `NodeKind` discriminant + per-kind data (sphere body
//!   radii, cube face index). Touched only on descent / ribbon pop.
//! - `node_offsets: Vec<u32>` — indexed by BFS position. Maps a
//!   node's BFS index to its header offset in `tree[]`. Also touched
//!   only on descent / ribbon pop, so the DDA inner loop only hits
//!   `tree[]`.
//!
//! Pack functions:
//!
//! - `pack_tree`: full BFS, no LOD. Used by tests and sanity
//!   debugging.
//! - `pack_tree_lod` / `pack_tree_lod_selective`: path-local LOD.
//!   Cartesian subtrees that subtend less than `LOD_THRESHOLD`
//!   pixels in the current node's local frame get flattened into a
//!   single Block leaf (their representative block type). Sphere
//!   bodies and face cells are exempt from flattening.

use std::collections::{HashMap, HashSet};

use crate::world::anchor::{Path, WorldPos};
use crate::world::tree::{
    slot_coords, Child, Node, NodeId, NodeKind, NodeLibrary, CHILDREN_PER_NODE,
};

use super::types::{GpuChild, GpuNodeKind};

/// Occupancy-mask flag bit marking a brick-packed node. Must stay in
/// sync with `BRICK_FLAG_BIT` in `assets/shaders/bindings.wgsl`.
/// Lives at bit 27 since the occupancy mask itself only uses 0..26.
const BRICK_FLAG_BIT: u32 = 1 << 27;

/// Empty-slot sentinel for brick entries. Must match `BRICK_EMPTY_BT`
/// in `bindings.wgsl`.
const BRICK_EMPTY_BT: u8 = 255;

/// A Cartesian node is brick-eligible iff every child is a terminal
/// (Empty or Block); no recursive Node children. We only brick
/// Cartesian nodes — sphere/face nodes have per-kind traversal paths
/// that don't match the flat-DDA brick layout.
fn is_node_brickable(node: &Node) -> bool {
    matches!(node.kind, NodeKind::Cartesian)
        && node.children.iter().all(|child| !matches!(child, Child::Node(_)))
}

/// Byte layout of a brick's children block: 27 u8 block_types packed
/// into 7 u32s (4 per word, little-endian within each word).
const BRICK_DATA_U32S: u32 = 7;
const BRICK_TOTAL_U32S: u32 = 2 + BRICK_DATA_U32S;

/// Pack 27 per-slot block types into 7 contiguous u32s.
fn pack_brick_words(slots: impl Iterator<Item = u8>) -> [u32; BRICK_DATA_U32S as usize] {
    let mut words = [0u32; BRICK_DATA_U32S as usize];
    for (slot, bt) in slots.enumerate() {
        if slot >= CHILDREN_PER_NODE { break; }
        words[slot / 4] |= (bt as u32) << ((slot % 4) * 8);
    }
    words
}

/// Result of packing: (tree, node_kinds, node_offsets, root_bfs_index).
///
/// - `tree`: single interleaved u32 buffer holding headers +
///   children inline. See module docs.
/// - `node_kinds`: per-BFS-node kind metadata.
/// - `node_offsets`: BFS index → tree[] u32-offset of that node's
///   header.
/// - `root_bfs_index`: BFS index of the root node. The renderer
///   converts this to a tree-offset via `node_offsets[root]`.
pub type PackedTree = (Vec<u32>, Vec<GpuNodeKind>, Vec<u32>, u32);

/// Pack the visible portion of the tree into the interleaved GPU
/// buffer. Returns `(tree, node_kinds, node_offsets, root_bfs_idx)`.
pub fn pack_tree(
    library: &NodeLibrary,
    root: NodeId,
) -> PackedTree {
    // Phase 1: BFS-visit every node; assign a BFS index and compute
    // each node's occupancy (so we know how many u32s its block
    // occupies).
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

    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut brickable: Vec<bool> = Vec::with_capacity(n_nodes);
    for &nid in &ordered {
        let node = library.get(nid).expect("node in ordered list must exist");
        let mut occ: u32 = 0;
        for (slot, child) in node.children.iter().enumerate() {
            if !matches!(child, Child::Empty) {
                occ |= 1u32 << slot;
            }
        }
        occupancies.push(occ);
        brickable.push(is_node_brickable(node));
    }

    // Phase 2: compute each node's header offset in tree[].
    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for i in 0..n_nodes {
        node_offsets.push(running);
        running += if brickable[i] {
            BRICK_TOTAL_U32S
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
        let occupancy = occupancies[i];
        let header_off = node_offsets[i];
        let first_child_off = header_off + 2;
        debug_assert_eq!(tree.len() as u32, header_off);

        if brickable[i] {
            tree.push(occupancy | BRICK_FLAG_BIT);
            tree.push(first_child_off);
            let words = pack_brick_words((0..CHILDREN_PER_NODE).map(|slot| {
                match node.children[slot] {
                    Child::Empty => BRICK_EMPTY_BT,
                    Child::Block(bt) => bt,
                    Child::Node(_) => unreachable!("brickable node has no Node children"),
                }
            }));
            for w in words { tree.push(w); }
            continue;
        }

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
                        node_index: *visited.get(child_id).expect("child must be visited"),
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

/// Encode a child's first u32: tag | (block_type << 8) | (_pad << 16).
fn pack_child_first(c: GpuChild) -> u32 {
    (c.tag as u32) | ((c.block_type as u32) << 8) | ((c._pad as u32) << 16)
}

/// Per-child screen-pixel estimate in the parent node's local frame.
/// Matches the shader's `lod_pixels = cell_size / ray_dist *
/// screen_height / (2 tan(fov/2))` but sampled at pack time using
/// the cell center instead of the per-ray DDA position.
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

/// Legacy parameter, retained for API compat. Previously controlled
/// force-collapse depth; now unused (the only pack-time LOD is
/// per-pixel, applied uniformly at every depth).
pub const DEFAULT_LOD_LEAF_DEPTH: u32 = 4;

/// LOD-aware tree packing.
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
) -> PackedTree {
    pack_tree_lod_selective(library, root, camera, screen_height, fov, &[], &[], DEFAULT_LOD_LEAF_DEPTH)
}

/// Like `pack_tree_lod`, but with one or more `preserve_paths`.
pub fn pack_tree_lod_preserving(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
) -> PackedTree {
    pack_tree_lod_selective(library, root, camera, screen_height, fov, preserve_paths, &[], DEFAULT_LOD_LEAF_DEPTH)
}

/// Like `pack_tree_lod_preserving`, but also supports bounded preserve regions.
///
/// `lod_leaf_depth` is the packed-tree depth at which the renderer
/// stops descending (= shader's `BASE_DETAIL_DEPTH`). Nodes at this
/// depth force-collapse their tag=2 subtree children into tag=1
/// (using the child's representative_block), so the node becomes
/// brickable — the shader descends into it via `march_brick` and
/// does a flat-byte lookup instead of per-cell tag dispatch. Visually
/// equivalent to the shader's at_max LOD-terminal behavior, just
/// baked into the GPU data layout.
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
    // The only pack-time LOD is per-pixel: a Cartesian child whose
    // projected screen size is below `LOD_PIXEL_THRESHOLD` is stored
    // as a tag=1 terminal with `representative_block` instead of
    // descending. This matches the shader's `at_lod` check exactly.
    // No force-collapse, no uniform-type collapse, no depth cap —
    // bricks are used only where a node is NATURALLY brickable
    // (every child is Block or Empty), which preserves real voxel
    // data losslessly.
    //
    // Preserve-paths (exact ribbon ancestors) and preserve-regions
    // (wide near-camera area) skip per-pixel collapse so the CPU
    // ribbon builder can descend and so near-camera content keeps
    // full detail regardless of projected size.
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

    /// Per-slot pack result for a single packed node. `None` =
    /// empty (will be absent from the interleaved tree); `Some(child)`
    /// with `child.tag in {1, 2}` = non-empty entry.
    type SlotOverride = [Option<GpuChild>; CHILDREN_PER_NODE];

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut per_node: Vec<SlotOverride> = Vec::new();
    let mut node_depths: Vec<u8> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    per_node.push([None; CHILDREN_PER_NODE]);
    node_depths.push(0);
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

            // Per-pixel LOD: if the child cell projects below
            // LOD_PIXEL_THRESHOLD pixels in the node's local frame,
            // store it as a tag=1 terminal with representative_block.
            // This matches the shader's runtime `at_lod` check so far
            // content doesn't expand the packed tree unnecessarily.
            if !on_preserve && lod_active && child_is_cartesian {
                let pixels =
                    child_screen_pixels(camera, &node_path, slot, screen_height, fov);
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
                node_depths.push(child_path.depth() as u8);
                queue.push(QueueEntry {
                    node_id: *child_id,
                    path: child_path,
                });
            }
            let child_idx = *visited.get(child_id).expect("child just visited");
            let repr = child_node.representative_block;
            per_node[ordered_idx][slot] = Some(GpuChild {
                tag: 2, block_type: repr, _pad: 0, node_index: child_idx,
            });
        }
    }

    // Phase 2: compute per-node header offsets.
    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut brickable: Vec<bool> = Vec::with_capacity(n_nodes);
    for (i, slots) in per_node.iter().enumerate() {
        let mut occ: u32 = 0;
        for (slot, entry) in slots.iter().enumerate() {
            if entry.is_some() {
                occ |= 1u32 << slot;
            }
        }
        occupancies.push(occ);
        // A packed node is brickable iff it's Cartesian AND every
        // non-empty slot is a terminal Block entry (tag == 1). Any
        // tag == 2 forces the sparse layout (recursion lives
        // through the child's node_index).
        //
        // A/B switch: DEEPSPACE_NO_BRICK disables the brick format
        // entirely (natural-brickable nodes get packed as sparse
        // instead). Used for perf comparison; shouldn't change
        // visible output.
        let bricks_disabled = std::env::var("DEEPSPACE_NO_BRICK").is_ok();
        let node = library.get(ordered[i])
            .expect("node in ordered list must exist");
        let is_cart = matches!(node.kind, NodeKind::Cartesian);
        let all_terminal = !bricks_disabled && slots.iter().all(|e| match e {
            None => true,
            Some(c) => c.tag == 1,
        });
        brickable.push(is_cart && all_terminal);
    }

    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for i in 0..n_nodes {
        node_offsets.push(running);
        running += if brickable[i] {
            BRICK_TOTAL_U32S
        } else {
            2 + 2 * occupancies[i].count_ones()
        };
    }
    let total_u32s = running as usize;

    // Phase 3: emit interleaved tree[].
    let mut tree: Vec<u32> = Vec::with_capacity(total_u32s);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(n_nodes);
    for (i, &node_id) in ordered.iter().enumerate() {
        let node = library.get(node_id).expect("node in ordered list must exist");
        kinds.push(GpuNodeKind::from_node_kind(node.kind));
        let occupancy = occupancies[i];
        let header_off = node_offsets[i];
        let first_child_off = header_off + 2;
        debug_assert_eq!(tree.len() as u32, header_off);

        if brickable[i] {
            tree.push(occupancy | BRICK_FLAG_BIT);
            tree.push(first_child_off);
            let words = pack_brick_words((0..CHILDREN_PER_NODE).map(|slot| {
                match per_node[i][slot] {
                    None => BRICK_EMPTY_BT,
                    Some(c) => c.block_type,
                }
            }));
            for w in words { tree.push(w); }
            continue;
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::anchor::WorldPos;
    use crate::world::bootstrap::{menger_world, plain_test_world, plain_world};
    use crate::world::tree::{empty_children, uniform_children, CENTER_SLOT};

    /// Read the child at (bfs_idx, slot) from a packed tree. Returns
    /// a synthesized `tag=0` GpuChild when the slot is empty.
    pub(super) fn sparse_child(
        tree: &[u32],
        node_offsets: &[u32],
        bfs_idx: u32,
        slot: u8,
    ) -> GpuChild {
        let header_off = node_offsets[bfs_idx as usize] as usize;
        let occupancy = tree[header_off];
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

    #[test]
    fn pack_test_world() {
        let world = plain_test_world();
        let (tree, kinds, node_offsets, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(root_idx, 0);
        assert_eq!(kinds.len(), world.library.len());
        assert_eq!(node_offsets.len(), world.library.len());
        assert!(!tree.is_empty());
        // Root at offset 0.
        assert_eq!(node_offsets[0], 0);
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
        let (_tree, kinds, _offsets, _root_idx) = pack_tree_lod(
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
        let (tree, _kinds, offsets, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        // Slot 0 (uniform-empty) has no bit in the root's occupancy.
        assert_eq!(sparse_child(&tree, &offsets, 0, 0).tag, 0);
        // Slot 13 (body) is present as a Node.
        let body_entry = sparse_child(&tree, &offsets, 0, 13);
        assert_eq!(body_entry.tag, 2);
        assert!(body_entry.node_index > 0);
    }

    #[test]
    fn pack_planet_body_present() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (_tree, kinds, _offsets, _root_idx) = pack_tree_lod(
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

        let (tree, _kinds, offsets, _root_idx) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&path],
        );
        let ribbon = build_ribbon(&tree, &offsets, &path);
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

        let (tree, _, offsets, _) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&[16u8]],
        );
        assert_eq!(sparse_child(&tree, &offsets, 0, 16).tag, 2);
        for sibling in [0u8, 5, 13, 26] {
            assert_eq!(
                sparse_child(&tree, &offsets, 0, sibling).tag, 0,
                "sibling slot {sibling} should be flattened",
            );
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
        let (tree, _kinds, offsets, _root_idx) = pack_tree_lod(
            &lib, root, &camera, 1080.0, 1.2,
        );
        assert_eq!(sparse_child(&tree, &offsets, 0, CENTER_SLOT as u8).tag, 2);
    }

    /// Baseline-capture test: builds a Menger sponge at depth 5 and
    /// packs it without LOD. Asserts the interleaved tree buffer stays
    /// below a hardcoded cap so silent pack-size drift (e.g. the 3.7%
    /// regression on fully-occupied nodes under sparse) trips CI.
    #[test]
    fn menger_pack_size_regression() {
        let world = menger_world(5);
        let (tree, _kinds, _offsets, _root) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!(
            "menger depth=5 pack size: {} u32s ({} bytes)",
            u32s, u32s * 4
        );
        // Measured baseline (sparse interleaved layout): 210 u32s.
        // Threshold is ~1.5x the measured baseline so normal churn
        // doesn't trip it but material regressions (>50%) do.
        const MENGER_D5_MAX_U32S: usize = 320;
        assert!(
            u32s < MENGER_D5_MAX_U32S,
            "menger depth=5 pack size regressed: {} u32s (> {} u32s)",
            u32s, MENGER_D5_MAX_U32S,
        );
    }

    /// Baseline-capture test for the plain-world preset.
    #[test]
    fn plain_pack_size_regression() {
        let world = plain_world(5);
        let (tree, _kinds, _offsets, _root) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!(
            "plain layers=5 pack size: {} u32s ({} bytes)",
            u32s, u32s * 4
        );
        // Measured baseline (sparse interleaved layout): 898 u32s.
        const PLAIN_L5_MAX_U32S: usize = 1400;
        assert!(
            u32s < PLAIN_L5_MAX_U32S,
            "plain layers=5 pack size regressed: {} u32s (> {} u32s)",
            u32s, PLAIN_L5_MAX_U32S,
        );
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
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
                DEFAULT_LOD_LEAF_DEPTH,
            );

            // CPU raycast to find a block to break.
            let ray_origin = camera.in_frame(&Path::root());
            let ray_dir = [0.0f32, -0.4, -0.9];
            let hit = raycast::cpu_raycast(
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

            let (tree_after, _, offsets_after, _) = pack_tree_lod_selective(
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
                DEFAULT_LOD_LEAF_DEPTH,
            );

            assert!(
                tree_before != tree_after || offsets_before != offsets_after,
                "packed GPU data unchanged after break at spawn_depth={spawn_depth} \
                 (library dedup may be collapsing the edit)",
            );
        }
    }
}
