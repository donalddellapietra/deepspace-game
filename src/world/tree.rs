//! Content-addressed base-3 recursive tree with first-class brick
//! leaves.
//!
//! Two node flavours, both stored in one `NodeLibrary`:
//!
//! * **Recursive** (`NodeKind::Cartesian`, `CubedSphereBody`,
//!   `CubedSphereFace`) — a `[Child; 27]` subdivision, where each
//!   slot is `Empty`, `Block(bt)`, or a `Node(child_id)` pointer.
//!   This is the classic sparse-octree-with-branch-3 representation.
//!
//! * **Brick** (`NodeKind::Brick`) — a dense `side³` voxel grid held
//!   inline in the node. `side ∈ {3, 9, 27}` so one brick collapses
//!   1, 2, or 3 levels of recursion into a flat array. Each cell is
//!   a `u8` block type (255 = empty). No sub-nodes, no pointers —
//!   the shader's brick DDA walks the grid with a flat byte read.
//!
//! Why both? The recursive form compresses sparse / repeated content
//! (air pockets, repeated patterns dedup to one node), while the
//! brick form compresses **dense terminal content** (a chunk of
//! mixed voxels is 1 brick vs 1 + 27 + 729 sparse nodes). The packer
//! emits them with different GPU layouts; the shader dispatches on
//! kind.
//!
//! Both kinds go through the same dedup path: content + kind + brick
//! bytes are hashed, identical payloads return the same `NodeId`.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::cubesphere::Face;

// ------------------------------------------------------------- constants

pub const BRANCH: usize = 3;
pub const CHILDREN_PER_NODE: usize = 27; // 3³
pub const MAX_DEPTH: usize = 63;

/// Allowed brick side lengths. `side` is the cells-per-axis; total
/// cells = `side³`. Values correspond to collapsing 1, 2, or 3
/// levels of the base-3 recursion into a flat array.
pub const BRICK_SIDES: [u8; 3] = [3, 9, 27];

// --------------------------------------------------------------- child

/// One child slot in a recursive node's 3×3×3 grid.
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

// ------------------------------------------------------------------ node

pub type Children = [Child; CHILDREN_PER_NODE];

pub fn empty_children() -> Children {
    [Child::Empty; CHILDREN_PER_NODE]
}

pub fn uniform_children(child: Child) -> Children {
    [child; CHILDREN_PER_NODE]
}

/// Empty-cell sentinel inside a brick's byte array. Must match the
/// shader's `BRICK_EMPTY_BT` (255) in `bindings.wgsl`.
pub const BRICK_EMPTY: u8 = 255;

/// Semantic kind of a node. Determines how the node's payload is
/// interpreted and how coordinate primitives + raycast dispatch
/// when the anchor is at this depth.
///
/// Part of the content-addressed hash: two nodes with identical
/// payload but different kinds do NOT dedup.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// Standard Cartesian subdivision via 27 `Child` slots.
    Cartesian,
    /// Root of a cubed-sphere body. 6 children at face-center
    /// slots are `CubedSphereFace` subtrees. Radii in body-local
    /// [0, 1): 0 < inner_r < outer_r ≤ 0.5.
    CubedSphereBody { inner_r: f32, outer_r: f32 },
    /// One face of a cubed-sphere body. Children interpreted on
    /// (u_slot, v_slot, r_slot) axes.
    CubedSphereFace { face: Face },
    /// Dense leaf node: `side³` voxels packed in `brick_cells`.
    /// Replaces 1-3 levels of recursion with a flat byte grid.
    Brick,
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
            NodeKind::Brick => {}
        }
    }
}

/// A node in the world tree. Holds either recursive children (27
/// slots) or a dense brick payload (`side³` bytes), discriminated by
/// `kind`.
///
/// The unused field for each variant is kept at a trivial default
/// (all-`Empty` children, empty `brick_cells`, `brick_side = 0`) so
/// the struct layout is uniform across kinds. The waste is one empty
/// `Vec` + 432 bytes per brick node — acceptable since bricks are
/// materialized at most once per deduped subtree and the
/// alternative (enum `Node`) would rewrite every call site that
/// accesses `children`.
pub struct Node {
    pub kind: NodeKind,
    /// Recursive payload. All `Empty` when `kind == Brick`.
    pub children: Children,
    /// Brick payload. Length `brick_side³` bytes when
    /// `kind == Brick`, empty otherwise.
    pub brick_cells: Vec<u8>,
    /// Cells per axis for a brick node. 3/9/27 when `kind == Brick`,
    /// 0 otherwise.
    pub brick_side: u8,
    pub ref_count: u32,
    /// Presence-preserving representative block: most common
    /// non-empty block type across the node's payload. `255` =
    /// entire subtree is empty. Used at LOD cutoff (small cells
    /// render as this single block).
    pub representative_block: u8,
    /// Single-block-type summary: 0-253 = subtree is entirely that
    /// block, 254 = entirely empty, 255 = mixed. Enables lossless
    /// collapse of uniform subtrees during packing.
    pub uniform_type: u8,
}

pub const UNIFORM_EMPTY: u8 = 254;
pub const UNIFORM_MIXED: u8 = 255;

impl Node {
    #[inline]
    pub fn is_brick(&self) -> bool {
        matches!(self.kind, NodeKind::Brick)
    }

    /// Total cells in the brick payload (`brick_side³`). 0 for
    /// non-brick nodes.
    #[inline]
    pub fn brick_cell_count(&self) -> usize {
        (self.brick_side as usize).pow(3)
    }

    /// Read a single voxel from a brick by local coordinates. `x`,
    /// `y`, `z` are in `[0, brick_side)`. Panics if called on a
    /// non-brick node or with out-of-range coords.
    #[inline]
    pub fn brick_voxel(&self, x: usize, y: usize, z: usize) -> u8 {
        debug_assert!(self.is_brick(), "brick_voxel on non-Brick node");
        let side = self.brick_side as usize;
        debug_assert!(x < side && y < side && z < side);
        self.brick_cells[z * side * side + y * side + x]
    }
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

/// Row-major index into a side×side×side brick.
#[inline]
pub const fn brick_index(x: usize, y: usize, z: usize, side: usize) -> usize {
    z * side * side + y * side + x
}

/// Inverse of `brick_index`.
#[inline]
pub const fn brick_coords(idx: usize, side: usize) -> (usize, usize, usize) {
    (idx % side, (idx / side) % side, idx / (side * side))
}

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
    /// Insert a Cartesian node. Shorthand for `insert_with_kind`.
    pub fn insert(&mut self, children: Children) -> NodeId {
        self.insert_with_kind(children, NodeKind::Cartesian)
    }

    /// Insert a recursive-kind node (Cartesian, sphere body, or
    /// face) with explicit children. If an identical node already
    /// exists, returns its id.
    pub fn insert_with_kind(&mut self, children: Children, kind: NodeKind) -> NodeId {
        assert!(
            !matches!(kind, NodeKind::Brick),
            "use insert_brick() for NodeKind::Brick"
        );
        let h = hash_recursive(&children, &kind);
        if let Some(candidates) = self.by_hash.get(&h) {
            for &id in candidates {
                if let Some(node) = self.nodes.get(&id) {
                    if node.kind == kind
                        && node.brick_side == 0
                        && node.children == children
                    {
                        return id;
                    }
                }
            }
        }

        let (representative_block, uniform_type) =
            compute_recursive_metadata(&children, &self.nodes);

        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            Node {
                kind,
                children,
                brick_cells: Vec::new(),
                brick_side: 0,
                ref_count: 0,
                representative_block,
                uniform_type,
            },
        );
        self.by_hash.entry(h).or_default().push(id);

        // Ref-inc Node children. Compute AFTER inserting so that
        // self-dedup on repeated calls does not double-count.
        let child_ids: Vec<NodeId> = children
            .iter()
            .filter_map(|c| match c {
                Child::Node(nid) => Some(*nid),
                _ => None,
            })
            .collect();
        for nid in child_ids {
            self.ref_inc(nid);
        }
        id
    }

    /// Insert a Brick node carrying `side³` voxel bytes. `side`
    /// must be one of `BRICK_SIDES` (3, 9, or 27). Cells are indexed
    /// row-major via `brick_index(x, y, z, side)`. A byte of 255
    /// marks an empty cell.
    ///
    /// Content-addressed dedup: two bricks with identical bytes +
    /// side return the same id.
    pub fn insert_brick(&mut self, cells: Vec<u8>, side: u8) -> NodeId {
        assert!(
            BRICK_SIDES.contains(&side),
            "invalid brick side {}; must be one of {:?}",
            side,
            BRICK_SIDES,
        );
        let expected = (side as usize).pow(3);
        assert_eq!(
            cells.len(),
            expected,
            "brick side {} expects {} cells, got {}",
            side,
            expected,
            cells.len(),
        );

        let h = hash_brick(&cells, side);
        if let Some(candidates) = self.by_hash.get(&h) {
            for &id in candidates {
                if let Some(node) = self.nodes.get(&id) {
                    if node.kind == NodeKind::Brick
                        && node.brick_side == side
                        && node.brick_cells == cells
                    {
                        return id;
                    }
                }
            }
        }

        let (representative_block, uniform_type) = compute_brick_metadata(&cells);

        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            Node {
                kind: NodeKind::Brick,
                children: empty_children(),
                brick_cells: cells,
                brick_side: side,
                ref_count: 0,
                representative_block,
                uniform_type,
            },
        );
        self.by_hash.entry(h).or_default().push(id);
        id
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Iterate over every node in the library in arbitrary order.
    /// Used by tests and diagnostics; hot paths go via `get()`.
    pub fn nodes_iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Build a uniform subtree of `depth` levels filled with
    /// `block_type`. depth=0 returns `Child::Block(block_type)`.
    /// depth=1 returns a node whose 27 children are all
    /// `Block(block_type)`. depth=N returns a node of uniform nodes,
    /// N levels deep. Content-addressed dedup means this creates at
    /// most N unique nodes.
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
        let h = node_hash(&node);
        if let Some(v) = self.by_hash.get_mut(&h) {
            v.retain(|&i| i != id);
            if v.is_empty() { self.by_hash.remove(&h); }
        }
        // Bricks have no Node children — only recursive kinds cascade.
        if node.brick_side == 0 {
            for child in &node.children {
                if let Child::Node(nid) = child {
                    self.ref_dec(*nid);
                }
            }
        }
    }
}

// --------------------------------------------------------------- metadata

/// Compute (representative_block, uniform_type) for a recursive
/// node. `representative_block` is the most common non-empty block
/// type across children (recursing through Node children's own
/// representative). `uniform_type` is 0..=253 if the entire subtree
/// is one block type, `UNIFORM_EMPTY` if all empty, `UNIFORM_MIXED`
/// otherwise.
fn compute_recursive_metadata(
    children: &Children,
    nodes: &HashMap<NodeId, Node>,
) -> (u8, u8) {
    let mut counts = [0u32; 256];
    for c in children {
        match c {
            Child::Block(bt) => counts[*bt as usize] += 1,
            Child::Node(nid) => {
                if let Some(child_node) = nodes.get(nid) {
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

    let uniform_type = {
        let mut first: Option<u8> = None;
        let mut uniform = true;
        for c in children {
            let ct = match c {
                Child::Empty => UNIFORM_EMPTY,
                Child::Block(bt) => *bt,
                Child::Node(nid) => {
                    nodes.get(nid).map(|n| n.uniform_type).unwrap_or(UNIFORM_MIXED)
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

    (representative_block, uniform_type)
}

/// Compute (representative_block, uniform_type) for a brick. All
/// payload is explicit bytes so we just scan.
fn compute_brick_metadata(cells: &[u8]) -> (u8, u8) {
    let mut counts = [0u32; 256];
    let mut first_solid: Option<u8> = None;
    let mut any_solid = false;
    let mut any_empty = false;
    let mut all_same_solid = true;
    for &c in cells {
        if c == BRICK_EMPTY {
            any_empty = true;
        } else {
            counts[c as usize] += 1;
            any_solid = true;
            match first_solid {
                None => first_solid = Some(c),
                Some(f) if f == c => {}
                _ => { all_same_solid = false; }
            }
        }
    }
    let representative_block = if any_solid {
        counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| *count)
            .map(|(i, _)| i as u8)
            .unwrap_or(255)
    } else {
        255
    };
    let uniform_type = if !any_solid {
        UNIFORM_EMPTY
    } else if all_same_solid && !any_empty {
        first_solid.unwrap()
    } else {
        UNIFORM_MIXED
    };
    (representative_block, uniform_type)
}

// ------------------------------------------------------------- hashing

fn hash_recursive(children: &Children, kind: &NodeKind) -> u64 {
    let mut h = DefaultHasher::new();
    0u8.hash(&mut h); // tag byte: recursive
    children.hash(&mut h);
    kind.hash(&mut h);
    h.finish()
}

fn hash_brick(cells: &[u8], side: u8) -> u64 {
    let mut h = DefaultHasher::new();
    1u8.hash(&mut h); // tag byte: brick
    side.hash(&mut h);
    cells.hash(&mut h);
    h.finish()
}

fn node_hash(node: &Node) -> u64 {
    if node.brick_side > 0 {
        hash_brick(&node.brick_cells, node.brick_side)
    } else {
        hash_recursive(&node.children, &node.kind)
    }
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
    fn brick_index_round_trip() {
        for &side in &BRICK_SIDES {
            let s = side as usize;
            for i in 0..(s * s * s) {
                let (x, y, z) = brick_coords(i, s);
                assert_eq!(brick_index(x, y, z, s), i);
            }
        }
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

    #[test]
    fn brick_insert_and_dedup() {
        let mut lib = NodeLibrary::default();
        let cells = vec![block::STONE; 27];
        let a = lib.insert_brick(cells.clone(), 3);
        let b = lib.insert_brick(cells.clone(), 3);
        assert_eq!(a, b, "identical brick should dedup");
        assert_eq!(lib.len(), 1);
        let node = lib.get(a).unwrap();
        assert!(node.is_brick());
        assert_eq!(node.brick_side, 3);
        assert_eq!(node.brick_cell_count(), 27);
        assert_eq!(node.representative_block, block::STONE);
        assert_eq!(node.uniform_type, block::STONE);
    }

    #[test]
    fn brick_distinct_by_side() {
        let mut lib = NodeLibrary::default();
        let small = lib.insert_brick(vec![block::STONE; 27], 3);
        let medium = lib.insert_brick(vec![block::STONE; 729], 9);
        assert_ne!(small, medium);
        assert_eq!(lib.len(), 2);
    }

    #[test]
    fn brick_metadata_mixed() {
        let mut lib = NodeLibrary::default();
        let mut cells = vec![BRICK_EMPTY; 27];
        cells[0] = block::STONE;
        cells[1] = block::STONE;
        cells[2] = block::GRASS;
        let id = lib.insert_brick(cells, 3);
        let node = lib.get(id).unwrap();
        assert_eq!(node.representative_block, block::STONE);
        assert_eq!(node.uniform_type, UNIFORM_MIXED);
    }

    #[test]
    fn brick_metadata_fully_empty() {
        let mut lib = NodeLibrary::default();
        let id = lib.insert_brick(vec![BRICK_EMPTY; 27], 3);
        let node = lib.get(id).unwrap();
        assert_eq!(node.representative_block, 255);
        assert_eq!(node.uniform_type, UNIFORM_EMPTY);
    }

    #[test]
    #[should_panic]
    fn brick_wrong_side_panics() {
        let mut lib = NodeLibrary::default();
        lib.insert_brick(vec![0u8; 27], 5);
    }

    #[test]
    #[should_panic]
    fn brick_wrong_cell_count_panics() {
        let mut lib = NodeLibrary::default();
        lib.insert_brick(vec![0u8; 10], 3);
    }
}
