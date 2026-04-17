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
    slot_coords, Child, NodeId, NodeKind, NodeLibrary,
    CHILDREN_PER_NODE, UNIFORM_EMPTY, UNIFORM_MIXED,
};

use super::types::{GpuChild, GpuNodeKind, GpuParentInfo};

/// Result of packing: (tree, node_kinds, node_offsets, parent_info, root_bfs_index).
///
/// - `tree`: single interleaved u32 buffer holding headers +
///   children inline. See module docs.
/// - `node_kinds`: per-BFS-node kind metadata.
/// - `node_offsets`: BFS index → tree[] u32-offset of that node's
///   header.
/// - `parent_info`: BFS index → `(parent_node_idx, slot_in_parent,
///   siblings_all_empty)`. The shader uses this to pop upward
///   without consulting a side ribbon. Root entry is the sentinel
///   `GpuParentInfo::root()`.
/// - `root_bfs_index`: BFS index of the root node. The renderer
///   converts this to a tree-offset via `node_offsets[root]`.
pub type PackedTree = (Vec<u32>, Vec<GpuNodeKind>, Vec<u32>, Vec<GpuParentInfo>, u32);

/// Pack the visible portion of the tree into the interleaved GPU
/// buffer. Returns `(tree, node_kinds, node_offsets, parent_info, root_bfs_idx)`.
pub fn pack_tree(
    library: &NodeLibrary,
    root: NodeId,
) -> PackedTree {
    // Phase 1: BFS-visit every node; assign a BFS index and compute
    // each node's occupancy (so we know how many u32s its block
    // occupies). Track each child's parent (BFS idx + slot) for
    // parent_info.
    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    // parent_link[child_bfs] = (parent_bfs, slot_in_parent). Root
    // entry is filled with the sentinel.
    let mut parent_link: Vec<(u32, u8)> = Vec::new();
    let mut head = 0usize;
    visited.insert(root, 0);
    ordered.push(root);
    parent_link.push((u32::MAX, 0));
    while head < ordered.len() {
        let nid = ordered[head];
        let parent_bfs = head as u32;
        head += 1;
        if let Some(node) = library.get(nid) {
            for (slot, child) in node.children.iter().enumerate() {
                if let Child::Node(child_id) = child {
                    if !visited.contains_key(child_id) {
                        let idx = ordered.len() as u32;
                        visited.insert(*child_id, idx);
                        ordered.push(*child_id);
                        parent_link.push((parent_bfs, slot as u8));
                    }
                }
            }
        }
    }

    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    for &nid in &ordered {
        let node = library.get(nid).expect("node in ordered list must exist");
        let mut occ: u32 = 0;
        for (slot, child) in node.children.iter().enumerate() {
            if !matches!(child, Child::Empty) {
                occ |= 1u32 << slot;
            }
        }
        occupancies.push(occ);
    }

    // Phase 2: compute each node's header offset in tree[].
    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for &occ in &occupancies {
        node_offsets.push(running);
        running = running + 2 + 2 * occ.count_ones();
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
        tree.push(occupancy);
        tree.push(first_child_off);
        for (slot, child) in node.children.iter().enumerate() {
            let entry = match child {
                Child::Empty => continue,
                Child::Block(bt) => GpuChild {
                    tag: 1, block_type: *bt, _pad: 0, node_index: 0,
                },
                Child::Node(child_id) => {
                    let child_bfs = *visited.get(child_id).expect("child must be visited");
                    let child_aabb = content_aabb(occupancies[child_bfs as usize]);
                    let repr = library
                        .get(*child_id)
                        .map(|n| n.representative_block)
                        .unwrap_or(0);
                    GpuChild {
                        tag: 2,
                        block_type: repr,
                        _pad: child_aabb,
                        node_index: child_bfs,
                    }
                }
            };
            debug_assert_ne!(occupancy & (1u32 << slot), 0);
            tree.push(pack_child_first(entry));
            tree.push(entry.node_index);
        }
    }
    debug_assert_eq!(tree.len(), total_u32s);

    let parent_info = build_parent_info(&parent_link, &occupancies);
    let root_idx = *visited.get(&root).unwrap();
    (tree, kinds, node_offsets, parent_info, root_idx)
}

/// Build `parent_info[]` from the BFS parent links and per-node
/// occupancies. Root entry is the sentinel; each non-root entry
/// records its parent's BFS idx, the slot it occupies in its
/// parent, and the `siblings_all_empty` flag (parent has exactly
/// one occupied slot — the one we descended through).
fn build_parent_info(
    parent_link: &[(u32, u8)],
    occupancies: &[u32],
) -> Vec<GpuParentInfo> {
    let mut out = Vec::with_capacity(parent_link.len());
    for (i, &(parent_bfs, slot)) in parent_link.iter().enumerate() {
        if i == 0 {
            out.push(GpuParentInfo::root());
            continue;
        }
        let siblings_all_empty = occupancies
            .get(parent_bfs as usize)
            .map(|occ| occ.count_ones() == 1)
            .unwrap_or(false);
        out.push(GpuParentInfo::new(parent_bfs, slot, siblings_all_empty));
    }
    out
}

/// Encode a child's first u32: tag | (block_type << 8) | (_pad << 16).
fn pack_child_first(c: GpuChild) -> u32 {
    (c.tag as u32) | ((c.block_type as u32) << 8) | ((c._pad as u32) << 16)
}

/// Compute a tight axis-aligned bounding box (slot-granular) of the
/// occupied slots in a 3×3×3 node. Returns a 12-bit packed value:
///
/// ```text
/// bits  0-1: min_x (0..=2, inclusive)
/// bits  2-3: min_y
/// bits  4-5: min_z
/// bits  6-7: max_x (0..=2, inclusive — shader treats as max+1 exclusive)
/// bits  8-9: max_y
/// bits 10-11: max_z
/// ```
///
/// Used by the ray-march shader to cull descents: on a non-empty
/// child, ray-box-test against the child's content AABB before
/// committing to the child DDA. If the ray misses the AABB, skip
/// the descent entirely. See `assets/shaders/march.wgsl`.
///
/// For an empty occupancy mask (shouldn't happen for a referenced
/// child), returns `0` — the shader treats an all-zero AABB as a
/// degenerate "no content" box that never hits, so a miss falls
/// through to the usual DDA path.
pub(crate) fn content_aabb(occupancy: u32) -> u16 {
    if occupancy == 0 {
        return 0;
    }
    let mut min_x = 3u32;
    let mut max_x = 0u32;
    let mut min_y = 3u32;
    let mut max_y = 0u32;
    let mut min_z = 3u32;
    let mut max_z = 0u32;
    for slot in 0..27u32 {
        if (occupancy >> slot) & 1 == 0 {
            continue;
        }
        let x = slot % 3;
        let y = (slot / 3) % 3;
        let z = slot / 9;
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
        if z < min_z { min_z = z; }
        if z > max_z { max_z = z; }
    }
    ((min_x << 0) | (min_y << 2) | (min_z << 4)
        | (max_x << 6) | (max_y << 8) | (max_z << 10)) as u16
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
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
) -> PackedTree {
    pack_tree_lod_selective(library, root, camera, screen_height, fov, &[], &[])
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
    pack_tree_lod_selective(library, root, camera, screen_height, fov, preserve_paths, &[])
}

/// Like `pack_tree_lod_preserving`, but also supports bounded preserve regions.
pub fn pack_tree_lod_selective(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
    preserve_regions: &[(Path, u8)],
) -> PackedTree {
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

    /// Per-slot pack result for a single packed node. `None` =
    /// empty (will be absent from the interleaved tree); `Some(child)`
    /// with `child.tag in {1, 2}` = non-empty entry.
    type SlotOverride = [Option<GpuChild>; CHILDREN_PER_NODE];

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut per_node: Vec<SlotOverride> = Vec::new();
    // parent_link[child_bfs] = (parent_bfs, slot_in_parent). Root
    // entry is filled with the sentinel.
    let mut parent_link: Vec<(u32, u8)> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    per_node.push([None; CHILDREN_PER_NODE]);
    parent_link.push((u32::MAX, 0));
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

            if !on_preserve && lod_active && child_is_cartesian {
                let screen_pixels =
                    child_screen_pixels(camera, &node_path, slot, screen_height, fov);
                if screen_pixels < LOD_THRESHOLD {
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
                parent_link.push((ordered_idx as u32, slot as u8));
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
    for slots in &per_node {
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
    for &occ in &occupancies {
        node_offsets.push(running);
        running = running + 2 + 2 * occ.count_ones();
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
        tree.push(occupancy);
        tree.push(first_child_off);
        for slot in 0..CHILDREN_PER_NODE {
            if let Some(mut entry) = per_node[i][slot] {
                // For tag=2 node children, stash the child's content
                // AABB in `_pad` so the shader can ray-box-cull the
                // descent before committing. tag=1 block leaves have
                // no subtree; leave `_pad` as 0.
                if entry.tag == 2 {
                    entry._pad = content_aabb(occupancies[entry.node_index as usize]);
                }
                tree.push(pack_child_first(entry));
                tree.push(entry.node_index);
            }
        }
    }
    debug_assert_eq!(tree.len(), total_u32s);

    let parent_info = build_parent_info(&parent_link, &occupancies);
    let root_idx = *visited.get(&root).unwrap();
    (tree, kinds, node_offsets, parent_info, root_idx)
}

/// Walk the packed tree from BFS root along `frame_slots`, following
/// only Node-tagged children. Stops when a slot is empty / the child
/// is a Block / a slot-OOB step is requested. Returns the deepest
/// reached BFS index and the actual prefix of `frame_slots` walked.
///
/// Replaces `build_ribbon`'s frame-walk side-effect: the renderer
/// uses the returned `(frame_root_idx, reached_slots)` to set the
/// shader's starting node and reproject the camera. Pop-up itself
/// is now driven by `parent_info[current_idx]` inside the shader,
/// so we no longer accumulate the ancestor chain here.
pub fn walk_to_frame_root(
    tree: &[u32],
    node_offsets: &[u32],
    frame_slots: &[u8],
) -> (u32, Vec<u8>) {
    let mut reached: Vec<u8> = Vec::with_capacity(frame_slots.len());
    let mut current: u32 = 0;
    for &slot in frame_slots {
        let Some(entry) = sparse_lookup(tree, node_offsets, current, slot) else { break };
        if entry.0 != 2 { break; }
        current = entry.1;
        reached.push(slot);
    }
    (current, reached)
}

/// Decode (tag, node_index) for `slot` at the BFS-indexed node,
/// returning `None` if the slot is OOB or empty. Used by
/// `walk_to_frame_root`.
fn sparse_lookup(
    tree: &[u32],
    node_offsets: &[u32],
    bfs_idx: u32,
    slot: u8,
) -> Option<(u8, u32)> {
    let header_off = *node_offsets.get(bfs_idx as usize)? as usize;
    let occupancy = *tree.get(header_off)?;
    let bit = 1u32 << slot;
    if occupancy & bit == 0 {
        return None;
    }
    let first_child = *tree.get(header_off + 1)? as usize;
    let rank = (occupancy & (bit - 1)).count_ones() as usize;
    let off = first_child + rank * 2;
    let packed = *tree.get(off)?;
    let tag = (packed & 0xFF) as u8;
    let node_index = *tree.get(off + 1)?;
    Some((tag, node_index))
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
        let (tree, kinds, node_offsets, _parent_info, root_idx) = pack_tree(&world.library, world.root);
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
        let (_tree, kinds, _offsets, _parent_info, _root_idx) = pack_tree_lod(
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
        let (tree, _kinds, offsets, _parent_info, _root_idx) = pack_tree_lod(
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
        let (_tree, kinds, _offsets, _parent_info, _root_idx) = pack_tree_lod(
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

        let (tree, _, offsets, _, _) = pack_tree_lod(
            &lib, root, &camera, 1080.0, 1.2,
        );
        assert_eq!(sparse_child(&tree, &offsets, 0, 16).tag, 0);

        let (tree2, _, offsets2, _, _) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&[16u8]],
        );
        let preserved = sparse_child(&tree2, &offsets2, 0, 16);
        assert_eq!(preserved.tag, 2);
        assert!(preserved.node_index > 0);
    }

    #[test]
    fn preserve_path_chain_lets_walker_descend_to_depth_n() {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..10 {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        let root = node;
        lib.ref_inc(root);
        let camera = camera_at([1.5, 1.5, 1.5]);
        let path = [13u8; 9];

        let (tree, _kinds, offsets, parent_info, _root_idx) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&path],
        );
        let (frame_root_idx, reached) = walk_to_frame_root(&tree, &offsets, &path);
        assert_eq!(reached.len(), 9);
        assert!(frame_root_idx > 0);
        // Walking parent_info from the frame root back to root must
        // visit exactly 9 ancestors before hitting the sentinel.
        let mut hops = 0;
        let mut cur = frame_root_idx;
        while !parent_info[cur as usize].is_root() {
            cur = parent_info[cur as usize].parent_node_idx;
            hops += 1;
            assert!(hops <= 32, "pop chain looped");
        }
        assert_eq!(hops, 9);
    }

    #[test]
    fn preserve_path_only_affects_chain_slots() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera = camera_at([1.5, 2.0, 1.5]);

        let (tree, _, offsets, _, _) = pack_tree_lod_preserving(
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
        let (tree, _kinds, offsets, _parent_info, _root_idx) = pack_tree_lod(
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
        let (tree, _kinds, _offsets, _parent_info, _root) = pack_tree(&world.library, world.root);
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
        let (tree, _kinds, _offsets, _parent_info, _root) = pack_tree(&world.library, world.root);
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

            let (tree_before, _, offsets_before, _, _) = pack_tree_lod_selective(
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
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

            let (tree_after, _, offsets_after, _, _) = pack_tree_lod_selective(
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
            );

            assert!(
                tree_before != tree_after || offsets_before != offsets_after,
                "packed GPU data unchanged after break at spawn_depth={spawn_depth} \
                 (library dedup may be collapsing the edit)",
            );
        }
    }
}
