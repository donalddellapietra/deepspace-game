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

// ------------------------------------------------------------ node kind

/// What kind of axes a node's 27 children branch on.
///
/// All existing Cartesian-tree content uses [`NodeKind::Cartesian`].
/// Step 2 introduces the enum itself; the body/face variants are
/// constructed by worldgen in step 5+ when planets move into the
/// tree. Dispatching on `kind` during `step_neighbor` (§2c of
/// refactor-decisions.md) arrives in step 8.
///
/// `CubedSphereBody` has no `center` field: a body always occupies
/// its parent cell, with the sphere centered at `(0.5, 0.5, 0.5)` in
/// that cell's local frame. Only `inner_r` and `outer_r` (also in
/// the parent's local `[0, 1)` frame) are per-node.
#[derive(Copy, Clone, Debug)]
pub enum NodeKind {
    /// Standard Cartesian branching. Children indexed by
    /// `(x_slot, y_slot, z_slot)` via [`slot_index`].
    Cartesian,
    /// A cubed-sphere body occupying its parent's cell. Six specific
    /// child slots (the face-center slots of the 27-grid) hold the
    /// per-face subtrees; other slots are the sphere's interior
    /// filler or empty.
    CubedSphereBody { inner_r: f32, outer_r: f32 },
    /// One face of a cubed-sphere body. Children indexed by
    /// `(u_slot, v_slot, r_slot)`. `face` is `0..=5` matching the
    /// indexing used by `cubesphere::Face`.
    CubedSphereFace { face: u8 },
}

impl Default for NodeKind {
    fn default() -> Self {
        NodeKind::Cartesian
    }
}

// f32 fields mean we can't derive PartialEq/Eq/Hash cleanly — we
// compare/hash bit patterns so two bodies with identical radii dedup
// correctly in the library.
impl PartialEq for NodeKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NodeKind::Cartesian, NodeKind::Cartesian) => true,
            (
                NodeKind::CubedSphereBody { inner_r: a, outer_r: b },
                NodeKind::CubedSphereBody { inner_r: c, outer_r: d },
            ) => a.to_bits() == c.to_bits() && b.to_bits() == d.to_bits(),
            (
                NodeKind::CubedSphereFace { face: a },
                NodeKind::CubedSphereFace { face: b },
            ) => a == b,
            _ => false,
        }
    }
}
impl Eq for NodeKind {}

impl Hash for NodeKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            NodeKind::Cartesian => 0u8.hash(state),
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                1u8.hash(state);
                inner_r.to_bits().hash(state);
                outer_r.to_bits().hash(state);
            }
            NodeKind::CubedSphereFace { face } => {
                2u8.hash(state);
                face.hash(state);
            }
        }
    }
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
    /// Axis semantics of this node's 27-slot branching. Defaults to
    /// `Cartesian`; `CubedSphereBody` / `CubedSphereFace` are used
    /// when a planet lives in the tree (step 5+).
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
    /// Insert a Cartesian node. If an existing node has identical
    /// children and kind, return its id (content-addressed dedup).
    pub fn insert(&mut self, children: Children) -> NodeId {
        self.insert_with_kind(children, NodeKind::Cartesian)
    }

    /// Insert a node with an explicit [`NodeKind`]. Dedup matches on
    /// `(children, kind)` together — two nodes with identical children
    /// but different kinds are distinct (e.g. a Cartesian subtree and
    /// a CubedSphereFace subtree with the same child array would be
    /// interpreted differently by `step_neighbor` and the shader).
    pub fn insert_with_kind(&mut self, children: Children, kind: NodeKind) -> NodeId {
        let h = hash_node_identity(&children, &kind);
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
        self.nodes.insert(
            id,
            Node { children, kind, ref_count: 0, representative_block, uniform_type },
        );
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
        let h = hash_node_identity(&node.children, &node.kind);
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

/// Hash identifying a unique node in the library. Includes kind
/// because two nodes with identical children but different kinds are
/// distinct (e.g. face subtree vs. Cartesian subtree with same child
/// arrangement).
fn hash_node_identity(children: &Children, kind: &NodeKind) -> u64 {
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
    fn default_kind_is_cartesian() {
        let mut lib = NodeLibrary::default();
        let id = lib.insert(empty_children());
        matches!(lib.get(id).unwrap().kind, NodeKind::Cartesian);
    }

    #[test]
    fn dedup_distinguishes_kind() {
        // Same children, different kinds → different library entries.
        let mut lib = NodeLibrary::default();
        let c = uniform_children(Child::Block(block::STONE));
        let cart = lib.insert_with_kind(c, NodeKind::Cartesian);
        let body = lib.insert_with_kind(
            c,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let face = lib.insert_with_kind(c, NodeKind::CubedSphereFace { face: 0 });
        assert_ne!(cart, body);
        assert_ne!(body, face);
        assert_ne!(cart, face);
        assert_eq!(lib.len(), 3);
    }

    #[test]
    fn dedup_same_kind_merges() {
        let mut lib = NodeLibrary::default();
        let c = uniform_children(Child::Block(block::STONE));
        let a = lib.insert_with_kind(
            c,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let b = lib.insert_with_kind(
            c,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        assert_eq!(a, b);
    }

    #[test]
    fn dedup_respects_radii() {
        // Different radii → different nodes.
        let mut lib = NodeLibrary::default();
        let c = uniform_children(Child::Block(block::STONE));
        let a = lib.insert_with_kind(
            c,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let b = lib.insert_with_kind(
            c,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.5 },
        );
        assert_ne!(a, b);
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
