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
//! Pack is a pure function of `(library, root)`. It does NOT depend
//! on camera position, frame, or any viewing parameter — so motion
//! never invalidates the packed buffer. Only tree edits do.
//!
//! Runtime LOD (screen-pixel sub-sample termination, max-depth cap)
//! lives in the shader. The pack only applies a content-driven
//! optimization: uniform subtrees (every leaf is the same block
//! type) are flattened to `Child::Block(uniform_type)` in their
//! parent's slab. Uniform-empty subtrees are dropped entirely. This
//! is safe regardless of view because a uniform subtree's contents
//! are invariant to descent depth.

use std::collections::HashMap;

use crate::world::tree::{
    Child, NodeId, NodeKind, NodeLibrary, UNIFORM_EMPTY, UNIFORM_MIXED,
};

use super::types::{GpuChild, GpuNodeKind};

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

/// Per-slot entry a packed node carries in its children slab.
/// `None` = empty (cleared in occupancy mask); `Some(child)` with
/// `tag in {1, 2}` = non-empty entry.
type SlotEntry = Option<GpuChild>;

/// Pack the world tree into the interleaved GPU buffer. Returns
/// `(tree, node_kinds, node_offsets, root_bfs_idx)`.
///
/// Pure function of `(library, root)`. Uniform subtrees get flattened
/// to their single-block representation in the parent's slab; the
/// subtree's nodes are not emitted (BFS does not recurse into them).
/// All other reachable nodes are emitted in BFS order.
pub fn pack_tree(library: &NodeLibrary, root: NodeId) -> PackedTree {
    // Phase 1: BFS. At each node, decide per-slot whether to emit a
    // tag=1 Block (for Child::Block or uniform-flattened Child::Node)
    // or a tag=2 Node that the BFS recurses into.
    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut ordered: Vec<NodeId> = Vec::new();
    let mut per_node_slots: Vec<[SlotEntry; 27]> = Vec::new();

    visited.insert(root, 0);
    ordered.push(root);
    per_node_slots.push([None; 27]);

    let mut head = 0usize;
    while head < ordered.len() {
        let node_id = ordered[head];
        let ordered_idx = head;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };

        for (slot, child) in node.children.iter().enumerate() {
            let entry = match child {
                Child::Empty => None,
                Child::Block(bt) => Some(GpuChild {
                    tag: 1, block_type: *bt, _pad: 0, node_index: 0,
                }),
                Child::Node(child_id) => {
                    // Content-driven flatten: Cartesian uniform subtrees
                    // collapse to a single Block in the parent's slab.
                    // Sphere body / face nodes carry geometry that the
                    // shader dispatches on — they must stay as Node
                    // children regardless of children uniformity.
                    let child_node = match library.get(*child_id) {
                        Some(n) => n,
                        None => continue,
                    };
                    let child_is_cartesian = matches!(child_node.kind, NodeKind::Cartesian);
                    if child_is_cartesian && child_node.uniform_type == UNIFORM_EMPTY {
                        None
                    } else if child_is_cartesian && child_node.uniform_type != UNIFORM_MIXED {
                        Some(GpuChild {
                            tag: 1,
                            block_type: child_node.uniform_type,
                            _pad: 0,
                            node_index: 0,
                        })
                    } else {
                        // Non-uniform subtree: emit as Node, recurse.
                        let child_bfs = if let Some(&idx) = visited.get(child_id) {
                            idx
                        } else {
                            let idx = ordered.len() as u32;
                            visited.insert(*child_id, idx);
                            ordered.push(*child_id);
                            per_node_slots.push([None; 27]);
                            idx
                        };
                        Some(GpuChild {
                            tag: 2,
                            block_type: child_node.representative_block,
                            _pad: 0,
                            node_index: child_bfs,
                        })
                    }
                }
            };
            per_node_slots[ordered_idx][slot] = entry;
        }
    }

    // Phase 2: compute occupancies and per-node header offsets.
    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    for slots in &per_node_slots {
        let mut occ: u32 = 0;
        for (slot, e) in slots.iter().enumerate() {
            if e.is_some() {
                occ |= 1u32 << slot;
            }
        }
        occupancies.push(occ);
    }
    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for &occ in &occupancies {
        node_offsets.push(running);
        running += 2 + 2 * occ.count_ones();
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
        for slot in 0..27usize {
            if let Some(mut entry) = per_node_slots[i][slot] {
                if entry.tag == 2 {
                    // Stash child's content AABB in _pad (bits 0-11)
                    // so the shader can ray-box-cull descent before
                    // committing.
                    entry._pad = content_aabb(occupancies[entry.node_index as usize]);
                }
                tree.push(pack_child_first(entry));
                tree.push(entry.node_index);
            }
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
/// the descent entirely.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::{menger_world, plain_test_world, plain_world};
    use crate::world::tree::{empty_children, uniform_children, Child, NodeKind, NodeLibrary, CENTER_SLOT};

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
    fn pack_test_world_root_at_bfs_zero() {
        let world = plain_test_world();
        let (tree, _kinds, node_offsets, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(root_idx, 0);
        assert_eq!(node_offsets[0], 0);
        assert!(!tree.is_empty());
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
        let (_tree, kinds, _offsets, _root_idx) = pack_tree(&world.library, world.root);
        let body = kinds.iter().find(|kind| kind.kind == 1).expect("body kind in buffer");
        assert!((body.inner_r - 0.12).abs() < 1e-6);
        assert!((body.outer_r - 0.45).abs() < 1e-6);
    }

    #[test]
    fn pack_flattens_uniform_empty_siblings() {
        // Planet world: 26 of 27 root slots are Node(leaf_air), leaf_air
        // is uniform-empty. Uniform-flatten drops them from the buffer.
        let world = planet_world();
        let (tree, _kinds, offsets, _root_idx) = pack_tree(&world.library, world.root);
        // Slot 0 (uniform-empty) should be absent.
        assert_eq!(sparse_child(&tree, &offsets, 0, 0).tag, 0);
        // CENTER_SLOT has the body — should be present as Node.
        let body_entry = sparse_child(&tree, &offsets, 0, CENTER_SLOT as u8);
        assert_eq!(body_entry.tag, 2);
        assert!(body_entry.node_index > 0);
    }

    #[test]
    fn pack_flattens_uniform_nonempty_subtree_to_block() {
        // Root has 1 Node child that's a uniform-stone subtree.
        let mut lib = NodeLibrary::default();
        let stone_leaf = lib.insert(uniform_children(Child::Block(
            crate::world::palette::block::STONE,
        )));
        // stone_leaf's uniform_type == STONE.
        let mut root_children = empty_children();
        root_children[13] = Child::Node(stone_leaf);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let (tree, kinds, offsets, _) = pack_tree(&lib, root);
        // Root should have slot 13 as Block (tag=1), NOT as Node.
        let entry = sparse_child(&tree, &offsets, 0, 13);
        assert_eq!(entry.tag, 1, "uniform-nonempty subtree flattened to Block");
        assert_eq!(entry.block_type, crate::world::palette::block::STONE);
        // stone_leaf's subtree should NOT be in the packed buffer.
        assert_eq!(kinds.len(), 1, "only root emitted; uniform subtree pruned");
    }

    /// Baseline-capture test: builds a Menger sponge at depth 5 and
    /// packs it. Asserts the interleaved tree buffer stays below a
    /// hardcoded cap so silent pack-size drift trips CI.
    #[test]
    fn menger_pack_size_regression() {
        let world = menger_world(5);
        let (tree, _kinds, _offsets, _root) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!(
            "menger depth=5 pack size: {} u32s ({} bytes)",
            u32s, u32s * 4
        );
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
        const PLAIN_L5_MAX_U32S: usize = 1400;
        assert!(
            u32s < PLAIN_L5_MAX_U32S,
            "plain layers=5 pack size regressed: {} u32s (> {} u32s)",
            u32s, PLAIN_L5_MAX_U32S,
        );
    }

    /// Verify that breaking a block at various depths actually changes
    /// the packed GPU tree data. Guards against dedup bugs where the
    /// library returns the same node after a break.
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
            let (tree_before, _, offsets_before, _) = pack_tree(&world.library, world.root);

            let ray_origin = camera.in_frame(&Path::root());
            let ray_dir = [0.0f32, -0.4, -0.9];
            let hit = raycast::cpu_raycast(
                &world.library, world.root, ray_origin, ray_dir, spawn_depth as u32,
            );
            let Some(hit) = hit else {
                panic!("no raycast hit at spawn_depth={spawn_depth}");
            };
            assert!(
                edit::break_block(&mut world, &hit),
                "break_block returned false at spawn_depth={spawn_depth}",
            );

            let (tree_after, _, offsets_after, _) = pack_tree(&world.library, world.root);
            assert!(
                tree_before != tree_after || offsets_before != offsets_after,
                "packed GPU data unchanged after break at spawn_depth={spawn_depth}",
            );
        }
    }
}
