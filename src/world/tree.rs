//! Content-addressed voxel tree nodes and library.
//!
//! See `docs/architecture/voxels.md`, `editing.md`, and `rendering.md`
//! for the design. This module is pure Rust with no Bevy dependencies —
//! Bevy wiring lives in the render / edit / state modules that build on
//! top of it.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;

use crate::block::BlockType;

// ------------------------------------------------------------- constants

pub const BRANCH_FACTOR: usize = 5;
pub const CHILDREN_PER_NODE: usize = 125; // 5³
pub const NODE_VOXELS_PER_AXIS: usize = 25;
pub const NODE_VOXELS: usize = 15_625; // 25³
pub const MAX_LAYER: u8 = 12;

// ---------------------------------------------------------------- voxel

/// Packed voxel content. `0` = empty. `1..=BlockType::ALL.len()` encodes
/// a `BlockType` by its discriminant + 1.
pub type Voxel = u8;
pub const EMPTY_VOXEL: Voxel = 0;

/// 25³ voxel grid for one node, heap-allocated.
pub type VoxelGrid = Box<[Voxel; NODE_VOXELS]>;

pub fn voxel_from_block(block: Option<BlockType>) -> Voxel {
    match block {
        None => EMPTY_VOXEL,
        Some(bt) => (bt as u8) + 1,
    }
}

pub fn block_from_voxel(v: Voxel) -> Option<BlockType> {
    if v == EMPTY_VOXEL {
        None
    } else {
        BlockType::from_index(v - 1)
    }
}

/// Allocate a zero-initialised voxel grid directly on the heap.
pub fn empty_voxel_grid() -> VoxelGrid {
    let v = vec![EMPTY_VOXEL; NODE_VOXELS].into_boxed_slice();
    v.try_into().unwrap_or_else(|_| unreachable!("size constant"))
}

/// Allocate a voxel grid filled with `fill`.
pub fn filled_voxel_grid(fill: Voxel) -> VoxelGrid {
    let v = vec![fill; NODE_VOXELS].into_boxed_slice();
    v.try_into().unwrap_or_else(|_| unreachable!("size constant"))
}

/// Row-major index into a 25³ voxel grid. `x` varies fastest, then `y`,
/// then `z`.
#[inline]
pub const fn voxel_idx(x: usize, y: usize, z: usize) -> usize {
    (z * NODE_VOXELS_PER_AXIS + y) * NODE_VOXELS_PER_AXIS + x
}

// ---------------------------------------------------------- slot encoding
//
// See `docs/architecture/coordinates.md`. This is the one canonical
// encoding used everywhere in the world module.

#[inline]
pub const fn slot_index(x: usize, y: usize, z: usize) -> usize {
    (z * BRANCH_FACTOR + y) * BRANCH_FACTOR + x
}

#[inline]
pub const fn slot_coords(slot: usize) -> (usize, usize, usize) {
    (
        slot % BRANCH_FACTOR,
        (slot / BRANCH_FACTOR) % BRANCH_FACTOR,
        slot / (BRANCH_FACTOR * BRANCH_FACTOR),
    )
}

// --------------------------------------------------------------- node id

/// Library-assigned content id. `0` is reserved for "no node / empty
/// slot" — see `EMPTY_NODE`.
pub type NodeId = u64;
pub const EMPTY_NODE: NodeId = 0;

/// 125 child ids, one per slot in the 5³ layout.
pub type Children = Box<[NodeId; CHILDREN_PER_NODE]>;

pub fn empty_children() -> Children {
    let v = vec![EMPTY_NODE; CHILDREN_PER_NODE].into_boxed_slice();
    v.try_into().unwrap_or_else(|_| unreachable!("size constant"))
}

pub fn uniform_children(id: NodeId) -> Children {
    let v = vec![id; CHILDREN_PER_NODE].into_boxed_slice();
    v.try_into().unwrap_or_else(|_| unreachable!("size constant"))
}

// ------------------------------------------------------------------ node

pub struct Node {
    /// 25³ voxel grid. Authoritative at leaves; cached downsample at
    /// non-leaves (used for mesh baking and rendering).
    pub voxels: VoxelGrid,
    /// `None` at leaves. `Some([id; 125])` at non-leaves.
    pub children: Option<Children>,
    /// Refcount for library eviction.
    pub ref_count: u32,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

// ------------------------------------------------------------- library

/// Content-addressed library of unique nodes.
///
/// Leaves are deduped by their voxel grid content. Non-leaves are
/// deduped by their 125-element children array. The two live in
/// separate hash tables so they can never collide.
///
/// Inserting a non-leaf increments each of its children's refcounts
/// once per slot (so an all-same-child non-leaf contributes 125 refs
/// to that child). Evicting a non-leaf decrements the same refs; when
/// a child's refcount hits zero, it is evicted recursively.
///
/// Inserting a leaf does NOT increment anything — leaves have no
/// children. The caller decides whether to `ref_inc` the returned id
/// to keep the leaf alive.
pub struct NodeLibrary {
    nodes: HashMap<NodeId, Node>,
    leaf_by_hash: HashMap<u64, Vec<NodeId>>,
    non_leaf_by_hash: HashMap<u64, Vec<NodeId>>,
    next_id: u64,
}

impl Default for NodeLibrary {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            leaf_by_hash: HashMap::new(),
            non_leaf_by_hash: HashMap::new(),
            // 0 reserved for EMPTY_NODE
            next_id: 1,
        }
    }
}

impl NodeLibrary {
    /// Insert a leaf. If an existing leaf has identical voxel content,
    /// its id is returned and the input `voxels` is dropped.
    pub fn insert_leaf(&mut self, voxels: VoxelGrid) -> NodeId {
        let h = hash_voxels(&voxels);
        if let Some(candidates) = self.leaf_by_hash.get(&h) {
            for &id in candidates {
                if let Some(node) = self.nodes.get(&id) {
                    if node.voxels.as_ref() == voxels.as_ref() {
                        return id;
                    }
                }
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            Node {
                voxels,
                children: None,
                ref_count: 0,
            },
        );
        self.leaf_by_hash.entry(h).or_default().push(id);
        id
    }

    /// Insert a non-leaf. Dedup is by children array. On a fresh
    /// insertion, refcounts of all 125 children are incremented.
    pub fn insert_non_leaf(
        &mut self,
        voxels: VoxelGrid,
        children: Children,
    ) -> NodeId {
        let h = hash_children(&children);
        if let Some(candidates) = self.non_leaf_by_hash.get(&h) {
            for &id in candidates {
                if let Some(node) = self.nodes.get(&id) {
                    if let Some(existing) = &node.children {
                        if existing.as_ref() == children.as_ref() {
                            return id;
                        }
                    }
                }
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        // Copy child ids so we can ref_inc after the insert without a
        // borrow conflict on `self.nodes`.
        let child_ids: [NodeId; CHILDREN_PER_NODE] = *children;
        self.nodes.insert(
            id,
            Node {
                voxels,
                children: Some(children),
                ref_count: 0,
            },
        );
        self.non_leaf_by_hash.entry(h).or_default().push(id);
        for child_id in child_ids {
            self.ref_inc(child_id);
        }
        id
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn ref_inc(&mut self, id: NodeId) {
        if id == EMPTY_NODE {
            return;
        }
        if let Some(node) = self.nodes.get_mut(&id) {
            node.ref_count = node.ref_count.saturating_add(1);
        }
    }

    pub fn ref_dec(&mut self, id: NodeId) {
        if id == EMPTY_NODE {
            return;
        }
        let should_evict = {
            let Some(node) = self.nodes.get_mut(&id) else {
                return;
            };
            node.ref_count = node.ref_count.saturating_sub(1);
            node.ref_count == 0
        };
        if should_evict {
            self.evict(id);
        }
    }

    fn evict(&mut self, id: NodeId) {
        let Some(node) = self.nodes.remove(&id) else {
            return;
        };
        match node.children {
            None => {
                let h = hash_voxels(&node.voxels);
                if let Some(v) = self.leaf_by_hash.get_mut(&h) {
                    v.retain(|&i| i != id);
                    if v.is_empty() {
                        self.leaf_by_hash.remove(&h);
                    }
                }
            }
            Some(children) => {
                let h = hash_children(&children);
                if let Some(v) = self.non_leaf_by_hash.get_mut(&h) {
                    v.retain(|&i| i != id);
                    if v.is_empty() {
                        self.non_leaf_by_hash.remove(&h);
                    }
                }
                let child_ids: [NodeId; CHILDREN_PER_NODE] = *children;
                for child_id in child_ids {
                    self.ref_dec(child_id);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

// ------------------------------------------------------------- hashing

fn hash_voxels(voxels: &[Voxel; NODE_VOXELS]) -> u64 {
    let mut h = DefaultHasher::new();
    h.write_u8(0x01); // leaf domain separator
    h.write(voxels);
    h.finish()
}

fn hash_children(children: &[NodeId; CHILDREN_PER_NODE]) -> u64 {
    let mut h = DefaultHasher::new();
    h.write_u8(0x02); // non-leaf domain separator
    for &id in children.iter() {
        h.write_u64(id);
    }
    h.finish()
}

// ----------------------------------------------------------- downsample

/// Convenience wrapper: look up 125 children by `NodeId` and
/// downsample them. Panics if any child id is missing from the
/// library (caller's responsibility to ensure the tree is
/// consistent). `EMPTY_NODE` slots are treated as all-empty voxels.
pub fn downsample_from_library(
    library: &NodeLibrary,
    children: &[NodeId; CHILDREN_PER_NODE],
) -> VoxelGrid {
    let empty = filled_voxel_grid(EMPTY_VOXEL);
    let refs: [&VoxelGrid; CHILDREN_PER_NODE] = std::array::from_fn(|i| {
        let id = children[i];
        if id == EMPTY_NODE {
            &empty
        } else {
            &library
                .get(id)
                .expect("downsample_from_library: missing child")
                .voxels
        }
    });
    downsample(refs)
}

/// Compress 125 children (each a 25³ voxel grid) into a parent 25³
/// grid. Each parent voxel majority-votes the 5³ child-voxel block it
/// represents. See `docs/architecture/voxels.md` for the math.
pub fn downsample(children: [&VoxelGrid; CHILDREN_PER_NODE]) -> VoxelGrid {
    let mut out = empty_voxel_grid();
    for pz in 0..NODE_VOXELS_PER_AXIS {
        for py in 0..NODE_VOXELS_PER_AXIS {
            for px in 0..NODE_VOXELS_PER_AXIS {
                let cx = px / BRANCH_FACTOR;
                let cy = py / BRANCH_FACTOR;
                let cz = pz / BRANCH_FACTOR;
                let child = children[slot_index(cx, cy, cz)];

                let bx = (px % BRANCH_FACTOR) * BRANCH_FACTOR;
                let by = (py % BRANCH_FACTOR) * BRANCH_FACTOR;
                let bz = (pz % BRANCH_FACTOR) * BRANCH_FACTOR;

                let mut counts = [0u16; 256];
                for dz in 0..BRANCH_FACTOR {
                    for dy in 0..BRANCH_FACTOR {
                        for dx in 0..BRANCH_FACTOR {
                            let v = child[voxel_idx(bx + dx, by + dy, bz + dz)];
                            counts[v as usize] += 1;
                        }
                    }
                }

                let mut best_v = 0u8;
                let mut best_count = counts[0];
                for v in 1..256usize {
                    if counts[v] > best_count {
                        best_v = v as u8;
                        best_count = counts[v];
                    }
                }
                out[voxel_idx(px, py, pz)] = best_v;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn grass_voxel() -> Voxel {
        voxel_from_block(Some(BlockType::Grass))
    }
    fn stone_voxel() -> Voxel {
        voxel_from_block(Some(BlockType::Stone))
    }

    #[test]
    fn voxel_block_round_trip() {
        for &bt in BlockType::ALL.iter() {
            let v = voxel_from_block(Some(bt));
            assert_ne!(v, EMPTY_VOXEL);
            assert_eq!(block_from_voxel(v), Some(bt));
        }
        assert_eq!(voxel_from_block(None), EMPTY_VOXEL);
        assert_eq!(block_from_voxel(EMPTY_VOXEL), None);
    }

    #[test]
    fn slot_encoding_round_trip() {
        for i in 0..CHILDREN_PER_NODE {
            let (x, y, z) = slot_coords(i);
            assert_eq!(slot_index(x, y, z), i);
        }
    }

    #[test]
    fn voxel_idx_bounds() {
        let last = voxel_idx(
            NODE_VOXELS_PER_AXIS - 1,
            NODE_VOXELS_PER_AXIS - 1,
            NODE_VOXELS_PER_AXIS - 1,
        );
        assert_eq!(last, NODE_VOXELS - 1);
    }

    #[test]
    fn downsample_all_same() {
        let grass = grass_voxel();
        let child = filled_voxel_grid(grass);
        let refs: [&VoxelGrid; CHILDREN_PER_NODE] =
            std::array::from_fn(|_| &child);
        let out = downsample(refs);
        assert!(out.iter().all(|&v| v == grass));
    }

    #[test]
    fn downsample_one_child_differs() {
        let grass = grass_voxel();
        let stone = stone_voxel();
        let grass_child = filled_voxel_grid(grass);
        let stone_child = filled_voxel_grid(stone);
        let mut refs: [&VoxelGrid; CHILDREN_PER_NODE] =
            std::array::from_fn(|_| &grass_child);
        // Replace only slot (0, 0, 0).
        refs[slot_index(0, 0, 0)] = &stone_child;
        let out = downsample(refs);
        // Child (0,0,0) occupies parent voxels (0..5, 0..5, 0..5).
        for z in 0..BRANCH_FACTOR {
            for y in 0..BRANCH_FACTOR {
                for x in 0..BRANCH_FACTOR {
                    assert_eq!(out[voxel_idx(x, y, z)], stone);
                }
            }
        }
        // Everything else is grass.
        for z in 0..NODE_VOXELS_PER_AXIS {
            for y in 0..NODE_VOXELS_PER_AXIS {
                for x in 0..NODE_VOXELS_PER_AXIS {
                    if x < BRANCH_FACTOR && y < BRANCH_FACTOR && z < BRANCH_FACTOR {
                        continue;
                    }
                    assert_eq!(out[voxel_idx(x, y, z)], grass);
                }
            }
        }
    }

    #[test]
    fn leaf_dedup() {
        let mut lib = NodeLibrary::default();
        let id1 = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        let id2 = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        assert_eq!(id1, id2);
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn leaf_distinct() {
        let mut lib = NodeLibrary::default();
        let id1 = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        let id2 = lib.insert_leaf(filled_voxel_grid(stone_voxel()));
        assert_ne!(id1, id2);
        assert_eq!(lib.len(), 2);
    }

    #[test]
    fn non_leaf_dedup_by_children() {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        let id1 = lib.insert_non_leaf(empty_voxel_grid(), uniform_children(leaf));
        let id2 = lib.insert_non_leaf(empty_voxel_grid(), uniform_children(leaf));
        assert_eq!(id1, id2);
    }

    #[test]
    fn non_leaf_distinct_by_children() {
        let mut lib = NodeLibrary::default();
        let leaf_g = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        let leaf_s = lib.insert_leaf(filled_voxel_grid(stone_voxel()));
        let mut children = uniform_children(leaf_g);
        children[0] = leaf_s;
        let id1 = lib.insert_non_leaf(empty_voxel_grid(), uniform_children(leaf_g));
        let id2 = lib.insert_non_leaf(empty_voxel_grid(), children);
        assert_ne!(id1, id2);
    }

    #[test]
    fn refcount_evict_leaf() {
        let mut lib = NodeLibrary::default();
        let id = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        lib.ref_inc(id);
        assert!(lib.get(id).is_some());
        lib.ref_dec(id);
        assert!(lib.get(id).is_none());
    }

    #[test]
    fn non_leaf_insertion_refs_children() {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        lib.insert_non_leaf(empty_voxel_grid(), uniform_children(leaf));
        assert_eq!(
            lib.get(leaf).unwrap().ref_count,
            CHILDREN_PER_NODE as u32
        );
    }

    #[test]
    fn refcount_cascades_through_non_leaf() {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert_leaf(filled_voxel_grid(grass_voxel()));
        let non_leaf =
            lib.insert_non_leaf(empty_voxel_grid(), uniform_children(leaf));

        lib.ref_inc(non_leaf);
        lib.ref_dec(non_leaf);
        assert!(lib.get(non_leaf).is_none());
        // Cascade: non_leaf's 125 refs on the leaf were released, and
        // the leaf had no other owners, so it's evicted too.
        assert!(lib.get(leaf).is_none());
        assert_eq!(lib.len(), 0);
    }
}
