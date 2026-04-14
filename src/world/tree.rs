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

// --------------------------------------------------------------- block

/// Block types. Simple enum — colors are defined in the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlockType {
    Stone = 0,
    Dirt = 1,
    Grass = 2,
    Wood = 3,
    Leaf = 4,
    Sand = 5,
    Water = 6,
    Brick = 7,
    Metal = 8,
    Glass = 9,
}

impl BlockType {
    pub const ALL: [Self; 10] = [
        Self::Stone, Self::Dirt, Self::Grass, Self::Wood, Self::Leaf,
        Self::Sand, Self::Water, Self::Brick, Self::Metal, Self::Glass,
    ];

    pub fn from_index(i: u8) -> Option<Self> {
        Self::ALL.get(i as usize).copied()
    }
}

// --------------------------------------------------------------- child

/// One child slot in a node's 3x3x3 grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Child {
    Empty,
    Block(BlockType),
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
    pub ref_count: u32,
}

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
    /// Insert a node. If an existing node has identical children,
    /// return its id (content-addressed dedup).
    pub fn insert(&mut self, children: Children) -> NodeId {
        let h = hash_children(&children);
        if let Some(candidates) = self.by_hash.get(&h) {
            for &id in candidates {
                if let Some(node) = self.nodes.get(&id) {
                    if node.children == children {
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
        self.nodes.insert(id, Node { children, ref_count: 0 });
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
        let h = hash_children(&node.children);
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

fn hash_children(children: &Children) -> u64 {
    let mut h = DefaultHasher::new();
    children.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

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
        let c = uniform_children(Child::Block(BlockType::Stone));
        let id1 = lib.insert(c);
        let id2 = lib.insert(c);
        assert_eq!(id1, id2);
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn distinct() {
        let mut lib = NodeLibrary::default();
        let id1 = lib.insert(uniform_children(Child::Block(BlockType::Stone)));
        let id2 = lib.insert(uniform_children(Child::Block(BlockType::Grass)));
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
    fn cascade_eviction() {
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(uniform_children(Child::Block(BlockType::Grass)));
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
