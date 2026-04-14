//! Content-addressed base-3 recursive tree.
//!
//! Every node has exactly 27 children (3x3x3). Each child is either
//! Empty, a Block type, or a reference to another node. Nodes store
//! nothing else — no voxel grid, no mesh, no metadata.
//!
//! See `docs/experimental-architecture/node.md`.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

// ------------------------------------------------------------- constants

pub const BRANCH: usize = 3;
pub const CHILDREN_PER_NODE: usize = 27; // 3³
pub const MAX_DEPTH: usize = 63;

// --------------------------------------------------------------- child

/// One child slot in a node's 3x3x3 grid.
///
/// `Block(u8)` holds a palette index. Builtin block indices live in
/// `palette::block`; imported model colors use indices 10-254.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Child {
    Empty,
    Block(u8),
    Node(NodeId),
}

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

// ------------------------------------------------------------- node kind

/// Coordinate semantics of a node's children.
///
/// All nodes have 27 children regardless of kind — the 27-slot layout
/// is invariant. Kind determines how the children's slot coordinates
/// map to physical space:
///
/// - `Cartesian`: children tile the parent's `[0, 1)³` cell as a
///   regular 3×3×3 grid.
/// - `CubedSphereBody`: same geometric layout as Cartesian, but this
///   node is the root of a spherical body. 6 of its 27 children (the
///   face-center slots) are `CubedSphereFace` subtrees; the body's
///   solid shell occupies `[inner_r, outer_r]` in the parent's local
///   frame (`0 < inner_r < outer_r ≤ 0.5`, so a body fits inside one
///   cell of its Cartesian parent).
/// - `CubedSphereFace`: children are indexed by `(u_slot, v_slot,
///   r_slot)` rather than `(x, y, z)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeKind {
    Cartesian,
    CubedSphereBody { inner_r: f32, outer_r: f32 },
    CubedSphereFace { face: u8 },
}

impl NodeKind {
    /// Stable byte tag used in the content-address hash. Changes to
    /// this mapping break dedup across versions — add new variants at
    /// the end rather than renumbering.
    #[inline]
    pub fn tag(self) -> u8 {
        match self {
            NodeKind::Cartesian => 0,
            NodeKind::CubedSphereBody { .. } => 1,
            NodeKind::CubedSphereFace { .. } => 2,
        }
    }

    #[inline]
    pub fn is_cartesian(self) -> bool {
        matches!(self, NodeKind::Cartesian)
    }
}

impl Eq for NodeKind {}

impl std::hash::Hash for NodeKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the tag plus any payload as raw bits. f32 payloads are
        // treated as their bit pattern (NaN-sensitive, which is fine
        // — worldgen never produces NaN radii, and distinct bit
        // patterns should dedup distinctly).
        self.tag().hash(state);
        match self {
            NodeKind::Cartesian => {}
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                inner_r.to_bits().hash(state);
                outer_r.to_bits().hash(state);
            }
            NodeKind::CubedSphereFace { face } => {
                face.hash(state);
            }
        }
    }
}

impl Default for NodeKind {
    fn default() -> Self { NodeKind::Cartesian }
}

// ------------------------------------------------------------------ node

pub type Children = [Child; CHILDREN_PER_NODE];

pub fn empty_children() -> Children {
    [Child::Empty; CHILDREN_PER_NODE]
}

pub fn uniform_children(child: Child) -> Children {
    [child; CHILDREN_PER_NODE]
}

pub struct Node {
    pub children: Children,
    pub kind: NodeKind,
    pub ref_count: u32,
    /// Presence-preserving representative block type for this subtree.
    /// The most common NON-EMPTY block type among all terminals in the
    /// subtree. Used by the renderer at LOD cutoff — when a cell is too
    /// small to descend into, it renders as this color.
    /// 255 = all empty (no solid content in this subtree).
    ///
    /// Presence-preserving means: if ANY child is non-empty, the
    /// representative is non-empty. A tree trunk (1 voxel of wood in
    /// 26 air voxels) gets representative = Wood, not Air. Thin
    /// features survive cascaded LOD across arbitrary depth.
    pub representative_block: u8,
    /// If the entire subtree is one type: 0-253 = that BlockType,
    /// 254 = all empty, 255 = mixed (not uniform).
    /// Uniform nodes can be flattened to a single Block during GPU packing.
    pub uniform_type: u8,
}

pub const UNIFORM_EMPTY: u8 = 254;
pub const UNIFORM_MIXED: u8 = 255;

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
    /// Insert a Cartesian node. Convenience wrapper around
    /// `insert_with_kind` for the common case.
    pub fn insert(&mut self, children: Children) -> NodeId {
        self.insert_with_kind(children, NodeKind::Cartesian)
    }

    /// Insert a node with the given kind. If an existing node has
    /// identical `(children, kind)`, return its id (content-addressed
    /// dedup). Two subtrees with the same children but different
    /// kinds are distinct.
    pub fn insert_with_kind(&mut self, children: Children, kind: NodeKind) -> NodeId {
        let h = hash_node(&children, &kind);
        if let Some(candidates) = self.by_hash.get(&h) {
            for &id in candidates {
                if let Some(node) = self.nodes.get(&id) {
                    if node.kind == kind && node.children == children {
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
        let mut counts = [0u32; 256];
        for c in &children {
            match c {
                Child::Block(bt) => counts[*bt as usize] += 1,
                Child::Node(nid) => {
                    if let Some(child_node) = self.nodes.get(nid) {
                        if child_node.representative_block < 255 {
                            counts[child_node.representative_block as usize] += 1;
                        }
                    }
                }
                Child::Empty => {}
            }
        }
        let representative_block = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| *count)
            .filter(|&(_, count)| *count > 0)
            .map(|(i, _)| i as u8)
            .unwrap_or(255);
        // Compute uniform_type: is the entire subtree one block type?
        let uniform_type = {
            let mut first: Option<u8> = None;
            let mut uniform = true;
            for c in &children {
                let ct = match c {
                    Child::Empty => UNIFORM_EMPTY,
                    Child::Block(bt) => *bt,
                    Child::Node(nid) => {
                        self.nodes.get(nid).map(|n| n.uniform_type).unwrap_or(UNIFORM_MIXED)
                    }
                };
                if ct == UNIFORM_MIXED { uniform = false; break; }
                match first {
                    None => first = Some(ct),
                    Some(f) if f == ct => {}
                    _ => { uniform = false; break; }
                }
            }
            if uniform { first.unwrap_or(UNIFORM_EMPTY) } else { UNIFORM_MIXED }
        };
        self.nodes.insert(id, Node { children, kind, ref_count: 0, representative_block, uniform_type });
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
    pub fn build_uniform_subtree(&mut self, block_type: u8, depth: u32) -> Child {
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
        let h = hash_node(&node.children, &node.kind);
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

fn hash_node(children: &Children, kind: &NodeKind) -> u64 {
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
    fn dedup_is_kind_sensitive() {
        // Identical children with different NodeKind must produce
        // distinct ids — NodeKind is part of the content address.
        let mut lib = NodeLibrary::default();
        let c = uniform_children(Child::Block(block::STONE));
        let a = lib.insert_with_kind(c, NodeKind::Cartesian);
        let b = lib.insert_with_kind(c, NodeKind::CubedSphereFace { face: 0 });
        assert_ne!(a, b);
        assert_eq!(lib.len(), 2);
    }

    #[test]
    fn default_insert_is_cartesian() {
        let mut lib = NodeLibrary::default();
        let id = lib.insert(empty_children());
        assert!(lib.get(id).unwrap().kind.is_cartesian());
    }

    #[test]
    fn cubed_sphere_body_dedup_by_radii() {
        let mut lib = NodeLibrary::default();
        let c = empty_children();
        let a = lib.insert_with_kind(c, NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 });
        let a2 = lib.insert_with_kind(c, NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 });
        let b = lib.insert_with_kind(c, NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.5 });
        assert_eq!(a, a2, "identical body radii must dedup");
        assert_ne!(a, b, "differing outer_r must not dedup");
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
