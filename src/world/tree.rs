//! Content-addressed base-3 recursive tree.
//!
//! Every node has exactly 27 children (3x3x3). Each child is either
//! Empty, a Block type, or a reference to another node. Nodes store
//! nothing else — no voxel grid, no mesh, no metadata.
//!
//! See `docs/experimental-architecture/node.md`.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::cubesphere::Face;

// ------------------------------------------------------------- constants

pub const BRANCH: usize = 3;
pub const CHILDREN_PER_NODE: usize = 27; // 3³
pub const MAX_DEPTH: usize = 63;

// --------------------------------------------------------------- child

/// One child slot in a node's 3x3x3 grid.
///
/// `Block(u16)` holds a palette index. Builtin block indices live in
/// `palette::block` (0..=9); imported model colors occupy 10..=65_534.
/// Emptiness is represented exclusively by `Child::Empty` — palette
/// index 0 is a real color (Stone), not a sentinel.
///
/// `EntityRef(idx)` is a per-frame scene overlay: the shader treats
/// the cell as a portal into an entity's voxel subtree, looked up in
/// the `entities[idx]` GPU buffer. EntityRef children only appear in
/// ephemeral scene-root nodes built by `world::scene`; terrain nodes
/// never carry them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Child {
    Empty,
    Block(u16),
    Node(NodeId),
    EntityRef(u32),
}

/// Sentinel `representative_block` for entity-bearing ephemeral
/// subtrees. Picked above the valid palette range (0..=65_534) and
/// distinct from `REPRESENTATIVE_EMPTY` so the shader's empty-rep
/// fast path never skips a scene ancestor chain.
pub const ENTITY_REPRESENTATIVE: u16 = 0xFFFD;

impl Child {
    #[inline]
    pub fn is_empty(self) -> bool {
        matches!(self, Child::Empty)
    }

    #[inline]
    pub fn is_solid(self) -> bool {
        !self.is_empty()
    }
}

// --------------------------------------------------------------- node id

pub type NodeId = u64;
pub const EMPTY_NODE: NodeId = 0;

// ------------------------------------------------------------------ node

pub type Children = [Child; CHILDREN_PER_NODE];

pub fn empty_children() -> Children {
    [Child::Empty; CHILDREN_PER_NODE]
}

pub fn uniform_children(child: Child) -> Children {
    [child; CHILDREN_PER_NODE]
}

/// Semantic kind of a node. Determines how the 27 children's
/// positions are interpreted and how the coordinate primitives
/// dispatch when the anchor is at this depth.
///
/// Part of the content-addressed hash: two nodes with identical
/// children but different kinds do NOT dedup into one.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// Standard Cartesian subdivision. Default for every node.
    Cartesian,
    /// Root of a cubed-sphere body. 6 children at face-center
    /// slots are `CubedSphereFace` subtrees. Radii are in the
    /// body cell's local [0, 1) frame: 0 < inner_r < outer_r <= 0.5.
    CubedSphereBody { inner_r: f32, outer_r: f32 },
    /// One face of a cubed-sphere body. Children are interpreted
    /// on (u_slot, v_slot, r_slot) axes.
    CubedSphereFace { face: Face },
    /// Cartesian subdivision rendered rotated 45° about the world
    /// Y axis and stretched by √2 in XZ so the rotated block's
    /// footprint fits the cell AABB exactly (corners touch cell-
    /// edge midpoints in XZ; Y fills the full cell). Internally
    /// the 3×3×3 children are regular cartesian — the shader
    /// applies the rotation when the ray enters the subtree.
    Rotated45Y,
}

impl Default for NodeKind {
    fn default() -> Self {
        NodeKind::Cartesian
    }
}

impl Eq for NodeKind {}

impl Hash for NodeKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            NodeKind::Cartesian => {}
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                inner_r.to_bits().hash(state);
                outer_r.to_bits().hash(state);
            }
            NodeKind::CubedSphereFace { face } => {
                face.hash(state);
            }
            NodeKind::Rotated45Y => {}
        }
    }
}

pub struct Node {
    pub children: Children,
    pub kind: NodeKind,
    pub ref_count: u32,
    /// Presence-preserving representative block type for this subtree.
    /// The most common NON-EMPTY block type among all terminals in the
    /// subtree. Used by the renderer at LOD cutoff — when a cell is too
    /// small to descend into, it renders as this color.
    /// `REPRESENTATIVE_EMPTY` = all empty (no solid content).
    ///
    /// Presence-preserving means: if ANY child is non-empty, the
    /// representative is non-empty. A tree trunk (1 voxel of wood in
    /// 26 air voxels) gets representative = Wood, not Air. Thin
    /// features survive cascaded LOD across arbitrary depth.
    pub representative_block: u16,
    /// If the entire subtree is one type: 0..=`MAX_BLOCK_TYPE` = that
    /// BlockType, `UNIFORM_EMPTY` = all empty, `UNIFORM_MIXED` = not
    /// uniform. Uniform nodes can be flattened to a single Block
    /// during GPU packing.
    pub uniform_type: u16,
    /// Depth of this subtree: 1 for a leaf node (children are all
    /// Block/Empty), 1 + max(child.depth) for nodes with Child::Node
    /// children. Cached at insert time so `WorldState::tree_depth()`
    /// is O(1) instead of a full DFS on every edit.
    pub depth: u32,
}

/// Sentinel placed in `Node::uniform_type` / `representative_block`
/// to denote "subtree is wholly empty" (no real palette color).
pub const UNIFORM_EMPTY: u16 = 0xFFFE;
/// Sentinel placed in `Node::uniform_type` to denote "subtree
/// contains multiple block types" (not flattenable).
pub const UNIFORM_MIXED: u16 = 0xFFFF;
/// Same sentinel as `UNIFORM_EMPTY`, but for `representative_block`
/// readers that want a semantically-named constant.
pub const REPRESENTATIVE_EMPTY: u16 = UNIFORM_EMPTY;

// ---------------------------------------------------------- slot encoding

/// Row-major index into a 3x3x3 grid: x varies fastest, then y, then z.
#[inline]
pub const fn slot_index(x: usize, y: usize, z: usize) -> usize {
    z * 9 + y * 3 + x
}

#[inline]
pub const fn slot_coords(slot: usize) -> (usize, usize, usize) {
    (slot % 3, (slot / 3) % 3, slot / 9)
}

/// The center child: (1, 1, 1) = slot 13.
pub const CENTER_SLOT: usize = slot_index(1, 1, 1);

// ------------------------------------------------------------- library

pub struct NodeLibrary {
    nodes: HashMap<NodeId, Node>,
    by_hash: HashMap<u64, Vec<NodeId>>,
    next_id: u64,
}

impl Default for NodeLibrary {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            by_hash: HashMap::new(),
            next_id: 1, // 0 reserved for EMPTY_NODE
        }
    }
}

impl NodeLibrary {
    /// Insert a Cartesian node. Shorthand for `insert_with_kind`
    /// with `NodeKind::Cartesian`.
    pub fn insert(&mut self, children: Children) -> NodeId {
        self.insert_with_kind(children, NodeKind::Cartesian)
    }

    /// Insert a node with an explicit `NodeKind`. If an existing
    /// node has identical children AND kind, return its id
    /// (content-addressed dedup).
    pub fn insert_with_kind(&mut self, children: Children, kind: NodeKind) -> NodeId {
        let h = hash_node_content(&children, &kind);
        if let Some(candidates) = self.by_hash.get(&h) {
            for &id in candidates {
                if let Some(node) = self.nodes.get(&id) {
                    if node.children == children && node.kind == kind {
                        return id;
                    }
                }
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        let child_node_ids: Vec<NodeId> = children
            .iter()
            .filter_map(|c| match c {
                Child::Node(nid) => Some(*nid),
                _ => None,
            })
            .collect();
        // Compute representative block (presence-preserving):
        // Most common NON-EMPTY block type among children. Empty children
        // are ignored so that thin features (a single wood voxel in air)
        // survive cascaded LOD. For Node children, inherit their
        // representative_block recursively.
        // Sparse tally: with u16 palette indices we can't preallocate
        // a `[u32; 65536]` on the stack. A `HashMap` keyed by block id
        // handles both narrow (builtins-only) and wide (scene-palette)
        // tally cases cleanly, at the cost of a bit of alloc per node.
        let mut counts: std::collections::HashMap<u16, u32> =
            std::collections::HashMap::new();
        for c in &children {
            match c {
                Child::Block(bt) => *counts.entry(*bt).or_insert(0) += 1,
                Child::Node(nid) => {
                    if let Some(child_node) = self.nodes.get(nid) {
                        if child_node.representative_block != REPRESENTATIVE_EMPTY {
                            *counts
                                .entry(child_node.representative_block)
                                .or_insert(0) += 1;
                        }
                    }
                }
                Child::EntityRef(_) => {
                    *counts.entry(ENTITY_REPRESENTATIVE).or_insert(0) += 1;
                }
                Child::Empty => {}
            }
        }
        let representative_block = counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(bt, _)| *bt)
            .unwrap_or(REPRESENTATIVE_EMPTY);
        // Compute uniform_type: is the entire subtree one block type?
        let uniform_type = {
            let mut first: Option<u16> = None;
            let mut uniform = true;
            for c in &children {
                let ct = match c {
                    Child::Empty => UNIFORM_EMPTY,
                    Child::Block(bt) => *bt,
                    Child::Node(nid) => self
                        .nodes
                        .get(nid)
                        .map(|n| n.uniform_type)
                        .unwrap_or(UNIFORM_MIXED),
                    // EntityRef children force MIXED so pack.rs never
                    // tries to uniform-flatten a scene ancestor into
                    // a single Block.
                    Child::EntityRef(_) => UNIFORM_MIXED,
                };
                if ct == UNIFORM_MIXED {
                    uniform = false;
                    break;
                }
                match first {
                    None => first = Some(ct),
                    Some(f) if f == ct => {}
                    _ => {
                        uniform = false;
                        break;
                    }
                }
            }
            if uniform {
                first.unwrap_or(UNIFORM_EMPTY)
            } else {
                UNIFORM_MIXED
            }
        };
        // Cache depth: 1 for leaves (all children are Block/Empty),
        // 1 + max(child.depth) when any child is Child::Node. O(1)
        // per insert thanks to child caching.
        let mut max_child_depth = 0u32;
        for c in &children {
            if let Child::Node(nid) = c {
                if let Some(child_node) = self.nodes.get(nid) {
                    if child_node.depth > max_child_depth {
                        max_child_depth = child_node.depth;
                    }
                }
            }
        }
        let depth = 1 + max_child_depth;
        self.nodes.insert(id, Node { children, kind, ref_count: 0, representative_block, uniform_type, depth });
        self.by_hash.entry(h).or_default().push(id);
        for nid in child_node_ids {
            self.ref_inc(nid);
        }
        id
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Build a uniform subtree of `depth` levels filled with `block_type`.
    /// depth=0 returns `Child::Block(block_type)`.
    /// depth=1 returns a node whose 27 children are all `Block(block_type)`.
    /// depth=N returns a node of uniform nodes, N levels deep.
    /// Content-addressed dedup means this creates at most N unique nodes.
    /// Override an existing node's `representative_block`. Used by
    /// the scene overlay to make ephemeral ancestors inherit the
    /// terrain's representative at their position — otherwise the
    /// ephemeral chain's rep would be dominated by a single
    /// `Child::EntityRef` sentinel and the shader's LOD-terminal
    /// splat would show entity-sentinel instead of terrain color.
    ///
    /// ## Safety
    ///
    /// The node's `uniform_type` is NOT recomputed; callers must
    /// ensure the override is consistent with that cached field.
    pub fn set_representative(&mut self, id: NodeId, rep: u16) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.representative_block = rep;
        }
    }

    pub fn build_uniform_subtree(&mut self, block_type: u16, depth: u32) -> Child {
        if depth == 0 {
            return Child::Block(block_type);
        }
        let mut child = Child::Block(block_type);
        for _ in 0..depth {
            let id = self.insert(uniform_children(child));
            child = Child::Node(id);
        }
        child
    }

    pub fn ref_inc(&mut self, id: NodeId) {
        if id == EMPTY_NODE { return; }
        if let Some(node) = self.nodes.get_mut(&id) {
            node.ref_count = node.ref_count.saturating_add(1);
        }
    }

    pub fn ref_dec(&mut self, id: NodeId) {
        if id == EMPTY_NODE { return; }
        let should_evict = {
            let Some(node) = self.nodes.get_mut(&id) else { return };
            node.ref_count = node.ref_count.saturating_sub(1);
            node.ref_count == 0
        };
        if should_evict {
            self.evict(id);
        }
    }

    fn evict(&mut self, id: NodeId) {
        let Some(node) = self.nodes.remove(&id) else { return };
        let h = hash_node_content(&node.children, &node.kind);
        if let Some(v) = self.by_hash.get_mut(&h) {
            v.retain(|&i| i != id);
            if v.is_empty() { self.by_hash.remove(&h); }
        }
        for child in &node.children {
            if let Child::Node(nid) = child {
                self.ref_dec(*nid);
            }
        }
    }
}

// ------------------------------------------------------------- hashing

fn hash_node_content(children: &Children, kind: &NodeKind) -> u64 {
    let mut h = DefaultHasher::new();
    children.hash(&mut h);
    kind.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;

    #[test]
    fn slot_round_trip() {
        for i in 0..CHILDREN_PER_NODE {
            let (x, y, z) = slot_coords(i);
            assert_eq!(slot_index(x, y, z), i);
        }
    }

    #[test]
    fn center_slot_is_1_1_1() {
        assert_eq!(slot_coords(CENTER_SLOT), (1, 1, 1));
    }

    #[test]
    fn dedup() {
        let mut lib = NodeLibrary::default();
        let c = uniform_children(Child::Block(block::STONE));
        let id1 = lib.insert(c);
        let id2 = lib.insert(c);
        assert_eq!(id1, id2);
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn distinct() {
        let mut lib = NodeLibrary::default();
        let id1 = lib.insert(uniform_children(Child::Block(block::STONE)));
        let id2 = lib.insert(uniform_children(Child::Block(block::GRASS)));
        assert_ne!(id1, id2);
        assert_eq!(lib.len(), 2);
    }

    #[test]
    fn refcount_eviction() {
        let mut lib = NodeLibrary::default();
        let id = lib.insert(empty_children());
        lib.ref_inc(id);
        assert!(lib.get(id).is_some());
        lib.ref_dec(id);
        assert!(lib.get(id).is_none());
    }

    #[test]
    fn dedup_respects_node_kind() {
        let mut lib = NodeLibrary::default();
        let children = uniform_children(Child::Block(block::STONE));
        let a = lib.insert(children);
        let b = lib.insert_with_kind(children, NodeKind::Cartesian);
        assert_eq!(a, b, "identical Cartesian kind should dedup");
        let c = lib.insert_with_kind(
            children,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        assert_ne!(a, c, "different kind must not dedup");
        assert_eq!(lib.len(), 2);
    }

    #[test]
    fn cascade_eviction() {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(uniform_children(Child::Block(block::GRASS)));
        let mut parent_children = empty_children();
        parent_children[0] = Child::Node(leaf);
        let parent = lib.insert(parent_children);
        lib.ref_inc(parent);
        assert_eq!(lib.get(leaf).unwrap().ref_count, 1);
        lib.ref_dec(parent);
        assert!(lib.get(parent).is_none());
        assert!(lib.get(leaf).is_none());
        assert_eq!(lib.len(), 0);
    }
}
