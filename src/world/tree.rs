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
#[derive(Copy, Clone, Debug)]
pub enum NodeKind {
    /// Standard Cartesian subdivision. Default for every node.
    Cartesian,
    /// Wrapped-Cartesian planet root.
    WrappedPlane {
        dims: [u32; 3],
        slab_depth: u8,
    },
    /// A Cartesian subtree whose local `[0, 3)³` frame is rotated
    /// relative to its parent's slot frame. The rotation is applied
    /// at descent around the cube's geometric center `(1.5, 1.5, 1.5)`
    /// — purely frame-local, NO world-space coordinates anywhere.
    ///
    /// `rotation` is column-major: `rotation[col][row]`. It maps
    /// LOCAL cube coords to PARENT-frame coords (`parent = R · local`).
    /// At descent the renderer applies `R^T` to the ray; on hit the
    /// normal is rotated back via `R · local_normal`.
    ///
    /// `cell_offset` is the cell's displacement from its natural slot
    /// centre, expressed in parent-frame `[0, 3)³` units. For ordinary
    /// TBs (rotated_cube_test, dodecahedron_test) this is `[0, 0, 0]`
    /// — the cell sits at its slot. For `SphericalWrappedPlane`
    /// children, this is `sphere_position − slot_centre`, repositioning
    /// the cell onto the sphere surface while keeping the parent's
    /// flat slab as ground-truth storage.
    ///
    /// Two TangentBlocks with bit-distinct `(rotation, cell_offset)`
    /// pairs DO NOT dedup; bit-identical pairs dedup as usual.
    TangentBlock {
        rotation: [[f32; 3]; 3],
        cell_offset: [f32; 3],
    },
}

/// Identity rotation (column-major). A `TangentBlock` carrying this
/// rotation renders identically to a `Cartesian` node modulo the dedup
/// distinction.
pub const IDENTITY_ROTATION: [[f32; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// Rotation about the +Y axis by `radians`, column-major.
/// `R · v` rotates `v` counter-clockwise looking down +Y.
pub fn rotation_y(radians: f32) -> [[f32; 3]; 3] {
    let (s, c) = radians.sin_cos();
    [
        [c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [s, 0.0, c],
    ]
}

#[inline]
fn rotation_bits(r: &[[f32; 3]; 3]) -> [[u32; 3]; 3] {
    let mut out = [[0u32; 3]; 3];
    for c in 0..3 {
        for r_ in 0..3 {
            out[c][r_] = r[c][r_].to_bits();
        }
    }
    out
}

/// Bit-pattern of a `[f32; 3]` for dedup hashing — same trick as
/// `rotation_bits`. Two `cell_offset`s with identical bit patterns
/// dedup; `0.0` and `-0.0` are treated as distinct (the bit pattern
/// differs), but no current call site constructs `-0.0`.
#[inline]
fn f32_3_bits(v: &[f32; 3]) -> [u32; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

impl PartialEq for NodeKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NodeKind::Cartesian, NodeKind::Cartesian) => true,
            (
                NodeKind::WrappedPlane { dims: a, slab_depth: ad },
                NodeKind::WrappedPlane { dims: b, slab_depth: bd },
            ) => a == b && ad == bd,
            (
                NodeKind::TangentBlock { rotation: a, cell_offset: ao },
                NodeKind::TangentBlock { rotation: b, cell_offset: bo },
            ) => rotation_bits(a) == rotation_bits(b) && f32_3_bits(ao) == f32_3_bits(bo),
            _ => false,
        }
    }
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
            NodeKind::WrappedPlane { dims, slab_depth } => {
                dims.hash(state);
                slab_depth.hash(state);
            }
            NodeKind::TangentBlock { rotation, cell_offset } => {
                rotation_bits(rotation).hash(state);
                f32_3_bits(cell_offset).hash(state);
            }
        }
    }
}

impl NodeKind {
    /// Whether the GPU packer is allowed to flatten a uniform
    /// subtree of this kind into a single Block at its parent's
    /// slot. Cartesian nodes flatten freely (the standard
    /// content-addressed shrink); kinds that carry metadata the
    /// shader needs to dispatch on (WrappedPlane, TangentBlock)
    /// must NOT flatten — otherwise the metadata vanishes the moment
    /// the subtree happens to be uniformly empty / uniformly air.
    #[inline]
    pub fn allows_uniform_flatten(self) -> bool {
        matches!(self, NodeKind::Cartesian)
    }

    /// True iff this kind is `TangentBlock`, regardless of rotation.
    #[inline]
    pub fn is_tangent_block(self) -> bool {
        matches!(self, NodeKind::TangentBlock { .. })
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
    fn tangent_block_kind_blocks_flatten_and_dedups() {
        let mut lib = NodeLibrary::default();
        let stone = uniform_children(Child::Block(block::STONE));
        let cart = lib.insert_with_kind(stone, NodeKind::Cartesian);
        let tan1 = lib.insert_with_kind(
            stone, NodeKind::TangentBlock { rotation: IDENTITY_ROTATION, cell_offset: [0.0; 3] });
        let tan2 = lib.insert_with_kind(
            stone, NodeKind::TangentBlock { rotation: IDENTITY_ROTATION, cell_offset: [0.0; 3] });
        // Different kinds are distinct nodes even with identical children.
        assert_ne!(cart, tan1);
        // Identical kind + children dedup (same rotation).
        assert_eq!(tan1, tan2);
        // TangentBlock is not allowed to flatten.
        let tb_id = NodeKind::TangentBlock { rotation: IDENTITY_ROTATION, cell_offset: [0.0; 3] };
        assert!(!tb_id.allows_uniform_flatten());
        // Bit-distinct rotations DO NOT dedup.
        let tan_rot = lib.insert_with_kind(
            stone, NodeKind::TangentBlock { rotation: rotation_y(0.5), cell_offset: [0.0; 3] });
        assert_ne!(tan1, tan_rot);
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
