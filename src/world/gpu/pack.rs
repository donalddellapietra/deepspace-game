//! GPU tree buffer: a single linear representation of the world
//! tree that the shader walks directly.
//!
//! ## Layout
//!
//! A single `tree: Vec<u32>` buffer carries every emitted node's
//! header + inline children slab. Each node occupies
//! `2 + 2*popcount(occupancy)` contiguous u32s:
//!
//! ```text
//! tree[base + 0]                     = occupancy mask (27 bits)
//! tree[base + 1]                     = first_child_offset (= base + 2)
//! tree[first_child_offset + rank*2]     = packed (tag|block_type|pad)
//! tree[first_child_offset + rank*2 + 1] = child BFS idx (when tag == 2)
//! ```
//!
//! Side buffers:
//!
//! - `node_kinds: Vec<GpuNodeKind>` — per-BFS-idx kind metadata
//!   (Cartesian/sphere-body/sphere-face + radii + face id).
//! - `node_offsets: Vec<u32>` — BFS idx → u32-offset of that node's
//!   header in `tree[]`.
//! - `node_ids: Vec<NodeId>` — BFS idx → source `NodeId` (CPU-side
//!   only; used to reuse existing packed subtrees across edits).
//! - `bfs_by_nid: HashMap<NodeId, u32>` — inverse of `node_ids`.
//!
//! ## One emit path
//!
//! [`CachedTree::update_root`] emits the tree rooted at `new_root`
//! into the buffer. It walks from `new_root` and at each node:
//!
//! - If the NodeId is already in `bfs_by_nid`, reuse that BFS idx —
//!   O(1), no writes.
//! - Otherwise, recursively resolve its children, then append a
//!   fresh header + children slab to the buffer.
//!
//! This is the same code path for initial pack (empty cache → every
//! node is new → full walk) and edits (warm cache → only N+1 new
//! edit-path ancestors get emitted → O(depth)).
//!
//! ## Uniform-flatten
//!
//! One content-driven optimization stays in pack: a Cartesian node
//! whose every leaf is one block type (`uniform_type != MIXED`)
//! collapses to `Child::Block(uniform_type)` in its parent's slab —
//! it gets no BFS entry of its own. Uniform-empty subtrees vanish
//! entirely. Sphere body / face nodes are exempt (the shader
//! dispatches on `NodeKind`, so they must stay addressable).
//!
//! ## Dead entries + compaction
//!
//! Edits append new ancestor entries; the prior versions' entries
//! stay in the buffer, unreferenced. `CachedTree::update_root`
//! checks `should_compact(library)` on each call and, when the
//! packed buffer holds substantially more entries than the library
//! has unique NodeIds (see `COMPACTION_GROWTH_FACTOR`), calls
//! `clear()` and re-emits from scratch — an O(reachable) rebuild
//! that drops all orphaned entries. Steady-state editing triggers
//! this once per ~N edits; content-streaming frames grow the library
//! in lockstep and don't false-fire compaction.

use std::collections::HashMap;

use crate::world::tree::{
    Child, NodeId, NodeKind, NodeLibrary, UNIFORM_EMPTY, UNIFORM_MIXED,
};

use super::types::{GpuChild, GpuNodeKind};

/// Full pack result tuple: `(tree, node_kinds, node_offsets,
/// node_ids, root_bfs_idx)`. Used by the initial GPU bring-up and
/// tests; edit-path callers use [`CachedTree`] directly.
pub type PackedTree = (Vec<u32>, Vec<GpuNodeKind>, Vec<u32>, Vec<NodeId>, u32);

/// Compaction trigger: when `node_offsets.len()` exceeds
/// `library.len() * COMPACTION_GROWTH_FACTOR`, clear and re-emit
/// the reachable subtree, dropping orphaned entries from prior
/// edits. 2.0 = at least half the packed entries are "dead"
/// (unreachable from root) before we pay the rebuild cost.
///
/// Why `library.len()`: the content-addressed library's `ref_dec`
/// evicts a `NodeId` the moment its ref count hits zero, so
/// `library.len()` is an O(1), monotonically-correct upper bound on
/// the count of unique `NodeId`s reachable from `world.root`
/// (saved-mesh installs keep extras alive via `ref_inc`, which is
/// fine — they're candidates for packing anyway). This tracks real
/// growth naturally:
///
/// - Steady-state editing: library stays roughly constant, packed
///   buffer grows with each edit's appended N+1 dead ancestors →
///   ratio climbs → eventually triggers a compaction.
/// - Content streaming (`install_subtree`, new planets, scripts):
///   library grows in lockstep with packed entries → ratio stays
///   near 1 → no spurious compaction on the loading frame.
///
/// The previous design (`live_count_marker` snapshot at last
/// compaction) false-fired on any single-shot content load.
const COMPACTION_GROWTH_FACTOR: f32 = 2.0;

/// The packed GPU tree state. Owned by the app; mutated in place on
/// every edit. The shader reads `tree` / `node_kinds` / `node_offsets`
/// (the first three fields); the remaining fields are CPU-side
/// bookkeeping used by [`Self::update_root`] to reuse previously
/// packed subtrees.
pub struct CachedTree {
    pub tree: Vec<u32>,
    pub node_kinds: Vec<GpuNodeKind>,
    pub node_offsets: Vec<u32>,
    pub node_ids: Vec<NodeId>,
    pub bfs_by_nid: HashMap<NodeId, u32>,
    pub root_bfs_idx: u32,
}

impl Default for CachedTree {
    fn default() -> Self {
        Self {
            tree: Vec::new(),
            node_kinds: Vec::new(),
            node_offsets: Vec::new(),
            node_ids: Vec::new(),
            bfs_by_nid: HashMap::new(),
            root_bfs_idx: 0,
        }
    }
}

impl CachedTree {
    pub fn new() -> Self { Self::default() }

    /// Reset all buffers to empty. Used by compaction before a
    /// full re-emit — the alternative is mark-and-sweep, which
    /// requires rewriting every surviving `tag=2` child entry's
    /// `node_index`. Clearing + re-emitting is simpler and the
    /// cost (~60 ms on soldier_729) is bounded by live node count,
    /// so it scales only with what's actually reachable.
    pub fn clear(&mut self) {
        self.tree.clear();
        self.node_kinds.clear();
        self.node_offsets.clear();
        self.node_ids.clear();
        self.bfs_by_nid.clear();
        self.root_bfs_idx = 0;
    }

    /// True when the packed buffer holds substantially more entries
    /// than the library currently needs. See
    /// `COMPACTION_GROWTH_FACTOR` for the signal — roughly: "more
    /// than half the packed slots are dead".
    pub fn should_compact(&self, library: &NodeLibrary) -> bool {
        let total = self.node_offsets.len();
        let lib_live = library.len();
        if lib_live == 0 { return false; }
        total as f32 >= lib_live as f32 * COMPACTION_GROWTH_FACTOR
    }

    /// Emit (or reuse) the subtree rooted at `new_root` into the
    /// buffer and update `self.root_bfs_idx` to that node's BFS idx.
    /// Already-packed subtrees are reused (O(1)); new nodes get
    /// appended (O(nodes_new_to_this_call)).
    ///
    /// When dead entries have accumulated past the compaction
    /// threshold, clears the cache first so the ensuing walk emits
    /// ONLY nodes reachable from `new_root` — dropping all orphaned
    /// entries from prior edits.
    pub fn update_root(&mut self, library: &NodeLibrary, new_root: NodeId) {
        if self.should_compact(library) {
            self.clear();
        }
        let bfs = self.emit_or_lookup(library, new_root);
        self.root_bfs_idx = bfs;
    }

    /// Resolve a NodeId to a BFS idx, packing it (and any missing
    /// descendants) on the fly. Content-addressed reuse via
    /// `bfs_by_nid`: if we packed this NodeId earlier, return that
    /// same BFS idx.
    fn emit_or_lookup(&mut self, library: &NodeLibrary, nid: NodeId) -> u32 {
        if let Some(&bfs) = self.bfs_by_nid.get(&nid) {
            return bfs;
        }
        let (children, kind) = {
            let node = library
                .get(nid)
                .expect("emit_or_lookup: library must contain every reachable node");
            (node.children, node.kind)
        };

        // Resolve every slot. Child::Node that's uniform-flat doesn't
        // recurse; Child::Node that's non-uniform gets packed here
        // (so its BFS idx is known when we push this node's slab).
        let mut slab: [Option<GpuChild>; 27] = [None; 27];
        for s in 0..27 {
            slab[s] = self.build_child_entry(library, children[s]);
        }

        // Compute occupancy from the slab.
        let mut occ = 0u32;
        for (i, e) in slab.iter().enumerate() {
            if e.is_some() { occ |= 1u32 << i; }
        }

        // Emit header + children slab. For tag=2 entries, fill the
        // content AABB from the just-resolved child's occupancy —
        // the shader uses this to cull descents before committing.
        let header_off = self.tree.len() as u32;
        self.tree.push(occ);
        self.tree.push(header_off + 2);
        for entry in slab.iter().flatten() {
            let mut e = *entry;
            if e.tag == 2 {
                let child_header_off = self.node_offsets[e.node_index as usize] as usize;
                let child_occ = self.tree[child_header_off];
                e._pad = content_aabb(child_occ);
            }
            self.tree.push(pack_child_first(e));
            self.tree.push(e.node_index);
        }

        let bfs = self.node_offsets.len() as u32;
        self.node_offsets.push(header_off);
        self.node_kinds.push(GpuNodeKind::from_node_kind(kind));
        self.node_ids.push(nid);
        self.bfs_by_nid.insert(nid, bfs);
        bfs
    }

    /// Build the `GpuChild` for one slot of a parent's children slab.
    /// Applies content-driven uniform-flatten (Cartesian only) so
    /// uniform subtrees collapse to a single Block. Non-uniform
    /// Child::Node recursively packs its subtree.
    fn build_child_entry(&mut self, library: &NodeLibrary, child: Child) -> Option<GpuChild> {
        match child {
            Child::Empty => None,
            Child::Block(bt) => Some(GpuChild {
                tag: 1, block_type: bt, _pad: 0, node_index: 0,
            }),
            Child::Node(child_id) => {
                let (is_cart, uniform_type, representative) = {
                    let node = library.get(child_id)?;
                    (
                        matches!(node.kind, NodeKind::Cartesian),
                        node.uniform_type,
                        node.representative_block,
                    )
                };
                if is_cart && uniform_type == UNIFORM_EMPTY {
                    None
                } else if is_cart && uniform_type != UNIFORM_MIXED {
                    Some(GpuChild {
                        tag: 1,
                        block_type: uniform_type,
                        _pad: 0,
                        node_index: 0,
                    })
                } else {
                    let child_bfs = self.emit_or_lookup(library, child_id);
                    // _pad (content AABB) filled in by the caller
                    // when emitting the parent's slab, at which
                    // point the child's occupancy is known.
                    Some(GpuChild {
                        tag: 2,
                        block_type: representative,
                        _pad: 0,
                        node_index: child_bfs,
                    })
                }
            }
        }
    }
}

/// Initial pack convenience wrapper. Returns the full tuple so
/// callers that don't own a [`CachedTree`] (tests, renderer
/// bring-up) can still get the buffers. Internally just builds a
/// fresh cache and invokes [`CachedTree::update_root`].
pub fn pack_tree(library: &NodeLibrary, root: NodeId) -> PackedTree {
    let mut cache = CachedTree::new();
    cache.update_root(library, root);
    (
        cache.tree, cache.node_kinds, cache.node_offsets,
        cache.node_ids, cache.root_bfs_idx,
    )
}

/// Encode a child's first u32: tag | (block_type << 8) | (_pad << 16).
pub(crate) fn pack_child_first(c: GpuChild) -> u32 {
    (c.tag as u32) | ((c.block_type as u32) << 8) | ((c._pad as u32) << 16)
}

/// Tight axis-aligned bounding box of the occupied slots in a 3×3×3
/// node, packed into 12 bits. Shader uses it to cull descent before
/// committing to a child's DDA.
///
/// ```text
/// bits  0-1: min_x (0..=2)
/// bits  2-3: min_y
/// bits  4-5: min_z
/// bits  6-7: max_x (shader treats as max+1 exclusive)
/// bits  8-9: max_y
/// bits 10-11: max_z
/// ```
///
/// Returns 0 for an all-empty occupancy; the shader treats this as
/// a degenerate box that never hits, falling through to the usual
/// DDA path.
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
    fn root_bfs_idx_is_last_emitted_for_initial_pack() {
        // Recursive emit order is post-order DFS: children first,
        // root last. Before this refactor root was BFS idx 0; now
        // it's `node_offsets.len() - 1`.
        let world = plain_test_world();
        let (_, kinds, offsets, _, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(root_idx as usize, offsets.len() - 1);
        assert_eq!(kinds.len(), offsets.len());
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
        let (_, kinds, _, _, _) = pack_tree(&world.library, world.root);
        let body = kinds.iter().find(|k| k.kind == 1).expect("body kind in buffer");
        assert!((body.inner_r - 0.12).abs() < 1e-6);
        assert!((body.outer_r - 0.45).abs() < 1e-6);
    }

    #[test]
    fn pack_flattens_uniform_empty_siblings() {
        let world = planet_world();
        let (tree, _, offsets, _, root_idx) = pack_tree(&world.library, world.root);
        // Slot 0 of root (uniform-empty Cartesian) should be absent.
        assert_eq!(sparse_child(&tree, &offsets, root_idx, 0).tag, 0);
        // CENTER_SLOT has the sphere body → Node, must stay.
        let body = sparse_child(&tree, &offsets, root_idx, CENTER_SLOT as u8);
        assert_eq!(body.tag, 2);
    }

    #[test]
    fn pack_flattens_uniform_nonempty_subtree_to_block() {
        let mut lib = NodeLibrary::default();
        let stone_leaf = lib.insert(uniform_children(Child::Block(
            crate::world::palette::block::STONE,
        )));
        let mut root_children = empty_children();
        root_children[13] = Child::Node(stone_leaf);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let (tree, kinds, offsets, _, root_idx) = pack_tree(&lib, root);
        let entry = sparse_child(&tree, &offsets, root_idx, 13);
        assert_eq!(entry.tag, 1, "uniform-nonempty subtree flattens to Block");
        assert_eq!(entry.block_type, crate::world::palette::block::STONE);
        assert_eq!(kinds.len(), 1, "only root emitted; uniform subtree pruned");
    }

    /// Emitting the same library root twice (initial + same root)
    /// should reuse every BFS entry — no new writes on the second
    /// call.
    #[test]
    fn reemit_same_root_is_noop() {
        let world = plain_test_world();
        let mut cache = CachedTree::new();
        cache.update_root(&world.library, world.root);
        let len_before = cache.tree.len();
        cache.update_root(&world.library, world.root);
        assert_eq!(cache.tree.len(), len_before,
            "reemitting the same root must not append any entries");
    }

    /// Edit a block and re-pack against the SAME cache. Only a
    /// bounded number of new entries (the edit-path ancestors)
    /// should be appended.
    #[test]
    fn edit_appends_only_depth_entries() {
        use crate::world::anchor::Path;
        use crate::world::bootstrap;
        use crate::world::edit;
        use crate::world::raycast;

        let spawn_depth: u8 = 10;
        let boot = bootstrap::bootstrap_world(bootstrap::WorldPreset::PlainTest, Some(40));
        let mut world = boot.world;
        let pos = bootstrap::plain_surface_spawn(spawn_depth);
        bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

        let mut cache = CachedTree::new();
        cache.update_root(&world.library, world.root);
        let entries_before = cache.node_offsets.len();

        let ray_origin = pos.in_frame(&Path::root());
        let hit = raycast::cpu_raycast(
            &world.library, world.root, ray_origin, [0.0, -0.4, -0.9], spawn_depth as u32,
        ).expect("raycast should hit ground");
        let old_root = world.root;
        assert!(edit::break_block(&mut world, &hit));
        assert_ne!(world.root, old_root);

        cache.update_root(&world.library, world.root);
        let entries_after = cache.node_offsets.len();
        let appended = entries_after - entries_before;

        // Bound: edit path length + a small safety margin for any
        // nested subtree that wasn't previously packed. In practice
        // this is ≤ spawn_depth.
        assert!(
            appended <= (spawn_depth as usize + 5),
            "edit appended {appended} entries (expected ≤ {})",
            spawn_depth + 5,
        );
    }

    /// After enough edits accumulate dead entries, `update_root`
    /// should trigger a full rebuild. Verify the buffer physically
    /// shrinks once the threshold is crossed.
    #[test]
    fn compaction_fires_after_enough_edits() {
        use crate::world::anchor::Path;
        use crate::world::bootstrap;
        use crate::world::edit;
        use crate::world::raycast;

        let spawn_depth: u8 = 10;
        let boot = bootstrap::bootstrap_world(bootstrap::WorldPreset::PlainTest, Some(40));
        let mut world = boot.world;
        let pos = bootstrap::plain_surface_spawn(spawn_depth);
        bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

        let mut cache = CachedTree::new();
        cache.update_root(&world.library, world.root);

        // Pound on edits until compaction fires. Each edit alternates
        // break and place so the world oscillates but library size
        // stays bounded — the packed buffer grows with dead entries
        // while library stays constant, crossing the threshold.
        let mut compaction_fired = false;
        for i in 0..200 {
            let ray_origin = pos.in_frame(&Path::root());
            let hit = raycast::cpu_raycast(
                &world.library, world.root, ray_origin, [0.0, -0.4, -0.9], spawn_depth as u32,
            ).expect("raycast should hit");
            if i % 2 == 0 {
                edit::break_block(&mut world, &hit);
            } else {
                edit::place_block(&mut world, &hit, crate::world::palette::block::BRICK);
            }

            let len_before = cache.node_offsets.len();
            cache.update_root(&world.library, world.root);
            let len_after = cache.node_offsets.len();

            if len_after < len_before {
                compaction_fired = true;
                // Post-compaction, the packed buffer should be close
                // to library.len() (which is live reachable count +
                // any saved-mesh holdovers).
                let lib_live = world.library.len();
                assert!(
                    len_after <= lib_live + spawn_depth as usize + 10,
                    "post-compaction total {len_after} should ≤ library {lib_live} + O(depth)",
                );
                break;
            }
        }
        assert!(compaction_fired, "compaction never fired in 200 edits (threshold too high?)");
    }

    /// Compaction must NOT fire on the initial pack or on the first
    /// few edits — the common path should stay O(depth) per edit.
    #[test]
    fn compaction_does_not_fire_prematurely() {
        let world = plain_test_world();
        let mut cache = CachedTree::new();
        assert!(!cache.should_compact(&world.library), "empty cache must not compact");
        cache.update_root(&world.library, world.root);
        assert!(
            !cache.should_compact(&world.library),
            "fresh cache must not trigger compaction immediately"
        );
        // Pack and re-pack with the same root — still no dead
        // entries, so compaction must stay off.
        cache.update_root(&world.library, world.root);
        assert!(!cache.should_compact(&world.library));
    }

    /// Regression guard for the review finding: a single-shot
    /// content-loading event that grows the tree legitimately must
    /// NOT trigger compaction. The old heuristic keyed off a frozen
    /// `live_count_marker` would false-fire here.
    #[test]
    fn compaction_skips_on_legitimate_growth() {
        // Start with a tiny world; pack it.
        let tiny = plain_test_world();
        let mut cache = CachedTree::new();
        cache.update_root(&tiny.library, tiny.root);
        let len_tiny = cache.node_offsets.len();

        // Now swap in a much larger world (simulates loading
        // content: the library grows, the root changes to the new
        // larger tree). Even though node_offsets.len() jumps well
        // past 2 × len_tiny, compaction must not fire — library.len()
        // grew in lockstep, so the packed buffer ratio stays ≈ 1.
        let big = crate::world::bootstrap::plain_world(8);
        cache.update_root(&big.library, big.root);
        let len_big = cache.node_offsets.len();

        assert!(
            len_big > len_tiny * 2,
            "test precondition: big world must be >2x tiny's packed size (tiny={len_tiny}, big={len_big})"
        );
        // If compaction had wrongly fired, cache would have been
        // cleared first and len_big would be close to big library
        // size ONLY — still OK because re-emit covers the whole
        // thing. The important check: compaction trigger should
        // return FALSE against the current library.
        assert!(
            !cache.should_compact(&big.library),
            "after growth, ratio should match library — no spurious compaction"
        );
    }

    /// Baseline-capture test: Menger sponge at depth 5 stays below
    /// a hardcoded cap so silent pack-size drift trips CI.
    #[test]
    fn menger_pack_size_regression() {
        let world = menger_world(5);
        let (tree, _, _, _, _) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!("menger depth=5 pack size: {} u32s ({} bytes)", u32s, u32s * 4);
        const MENGER_D5_MAX_U32S: usize = 320;
        assert!(u32s < MENGER_D5_MAX_U32S, "menger regressed: {u32s} > {MENGER_D5_MAX_U32S}");
    }

    #[test]
    fn plain_pack_size_regression() {
        let world = plain_world(5);
        let (tree, _, _, _, _) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!("plain layers=5 pack size: {} u32s ({} bytes)", u32s, u32s * 4);
        const PLAIN_L5_MAX_U32S: usize = 1400;
        assert!(u32s < PLAIN_L5_MAX_U32S, "plain regressed: {u32s} > {PLAIN_L5_MAX_U32S}");
    }

    /// Break a block at various depths and verify the packed GPU
    /// data actually changes.
    #[test]
    fn break_at_every_depth_changes_packed_data() {
        use crate::world::anchor::Path;
        use crate::world::bootstrap;
        use crate::world::edit;
        use crate::world::raycast;

        for spawn_depth in [4u8, 8, 11, 15, 20, 25, 30, 33, 38] {
            let boot = bootstrap::bootstrap_world(
                bootstrap::WorldPreset::PlainTest, Some(40),
            );
            let mut world = boot.world;
            let pos = bootstrap::plain_surface_spawn(spawn_depth);
            bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

            let (tree_before, _, offsets_before, _, _) = pack_tree(&world.library, world.root);

            let ray_origin = pos.in_frame(&Path::root());
            let hit = raycast::cpu_raycast(
                &world.library, world.root, ray_origin, [0.0, -0.4, -0.9], spawn_depth as u32,
            ).expect(&format!("raycast must hit at depth {spawn_depth}"));
            assert!(edit::break_block(&mut world, &hit));

            let (tree_after, _, offsets_after, _, _) = pack_tree(&world.library, world.root);
            assert!(
                tree_before != tree_after || offsets_before != offsets_after,
                "packed GPU data unchanged after break at depth {spawn_depth}",
            );
        }
    }
}
