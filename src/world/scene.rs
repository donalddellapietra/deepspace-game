//! Per-frame scene root: the terrain tree with entity cells
//! overlaid as `Child::EntityRef` leaves.
//!
//! Each entity has an anchor path (from the render frame down to the
//! entity's anchor depth). The scene root is a fresh Cartesian
//! subtree that shares every terrain subtree NOT on any entity's
//! path, and ephemerally forks only the ancestor chain needed to
//! carry the `EntityRef(idx)` leaves.
//!
//! ## Why overlay, not merge
//!
//! Entities don't replace terrain — a soldier standing on grass
//! still has grass under its feet. The scene root is a **sparse
//! overlay**: at any slot where no entity descends, it keeps the
//! terrain child verbatim (reused NodeId via content-addressing in
//! `NodeLibrary`).
//!
//! ## Trie build
//!
//! A simple iterative trie over the entity relative paths (slots
//! from render-frame depth to anchor depth). Inserting N entities
//! is O(N * avg_path_depth). Walking the trie bottom-up builds
//! scene nodes via `NodeLibrary::insert_with_kind`:
//!
//! - At the deepest level reached by any entity: replace the
//!   terrain child at that slot with `Child::EntityRef(idx)`.
//! - At intermediate levels: replace the terrain child with
//!   `Child::Node(new_ephemeral_id)` pointing at the child scene
//!   node.
//! - Siblings of the overlay path: keep whatever was in the terrain
//!   node (typically a `Child::Node(...)` that now gets shared
//!   between terrain and scene via library dedup).
//!
//! ## Ref counting
//!
//! The caller is expected to `ref_inc(scene_root)` after build and
//! `ref_dec(prev_scene_root)` after the next build — ephemeral
//! ancestors will cascade-evict when their parent scene_root dies,
//! while terrain subtrees reused by both live nodes stay alive via
//! their own primary reference (typically `world.root`).

use crate::world::tree::{
    empty_children, Child, NodeId, NodeKind, NodeLibrary, EMPTY_NODE,
};

/// One node in the per-frame entity-path trie.
///
/// `slot_children[s]` is `Some(subtrie)` when at least one entity
/// passes through slot `s`. Leaf slots carry `leaf_entity = Some(idx)`.
///
/// Sparse representation: most slots are `None` because entities
/// cluster in a small part of the world. The children are stored in
/// a flat `[Option<Box<...>>; 27]` for simple indexing.
struct TrieNode {
    slot_children: [Option<Box<TrieNode>>; 27],
    /// `Some(entity_idx)` when an entity's anchor bottoms out at
    /// this node — the parent's slot that points here becomes
    /// `Child::EntityRef(entity_idx)` in the scene tree. Multiple
    /// entities may share an anchor cell; we pick one (the first
    /// inserted) per slot — overlaps are rare in practice and the
    /// scene overlay is visual, not authoritative for game logic.
    leaf_entity: Option<u32>,
}

impl TrieNode {
    fn empty() -> Self {
        Self { slot_children: Default::default(), leaf_entity: None }
    }
}

/// Input description of one entity's anchor path, relative to the
/// render frame. `path_slots.len()` equals the number of tree levels
/// below the frame that the entity is anchored at. Empty path means
/// the entity's anchor IS the render frame (scene-root slot replaces
/// directly) — unusual but handled.
pub struct EntityPath {
    pub entity_idx: u32,
    pub path_slots: Vec<u8>,
}

/// Result of a scene build: the ephemeral scene root NodeId that
/// callers should ref-hold (and later ref-release when the next
/// frame's scene root arrives), and the depth at which entity cells
/// sit (used by the shader for cell-size derivation).
pub struct SceneRoot {
    pub node_id: NodeId,
}

/// Build the per-frame scene root by overlaying `entity_paths` onto
/// the terrain subtree rooted at `terrain_root`. All paths are
/// relative to the same render frame (slot 0 of a path = child slot
/// under `terrain_root`).
///
/// Idempotent under dedup: if the same entity set + layout + terrain
/// was built last frame, the returned NodeId equals last frame's —
/// the pack cache will hit its memo.
pub fn build_scene_root(
    library: &mut NodeLibrary,
    terrain_root: NodeId,
    entity_paths: &[EntityPath],
) -> SceneRoot {
    if entity_paths.is_empty() {
        return SceneRoot { node_id: terrain_root };
    }
    let mut trie = TrieNode::empty();
    for e in entity_paths {
        let mut node = &mut trie;
        for &slot in &e.path_slots {
            let idx = slot as usize;
            if node.slot_children[idx].is_none() {
                node.slot_children[idx] = Some(Box::new(TrieNode::empty()));
            }
            node = node.slot_children[idx].as_mut().unwrap();
        }
        // Path-empty entities are installed at trie root directly.
        // Otherwise the entity sits at this trie leaf.
        if node.leaf_entity.is_none() {
            node.leaf_entity = Some(e.entity_idx);
        }
    }
    let node_id = build_scene_node(library, terrain_root, &trie);
    SceneRoot { node_id }
}

/// Recursive build from trie bottom-up. Produces a NodeId whose
/// children mirror the terrain node at `terrain_id`, EXCEPT that
/// slots with trie coverage carry entity overlays or recurse into a
/// scene sub-node.
fn build_scene_node(
    library: &mut NodeLibrary,
    terrain_id: NodeId,
    trie: &TrieNode,
) -> NodeId {
    // Seed from terrain's children + kind (or empty Cartesian when
    // the terrain slot had no Node — the trie may dive into air).
    let (mut base_children, kind) = match library.get(terrain_id) {
        Some(node) => (node.children, node.kind),
        None => (empty_children(), NodeKind::Cartesian),
    };
    for s in 0..27 {
        let Some(sub) = trie.slot_children[s].as_deref() else { continue };
        // Terrain-side child currently occupying this slot. If it's
        // a Node, pass it down so the scene child inherits the
        // terrain subtree under it. Otherwise treat as empty —
        // entities are spawning into what was previously Block or
        // Empty, so the scene subtree starts from air below.
        let terrain_child_id = match base_children[s] {
            Child::Node(id) => id,
            _ => EMPTY_NODE,
        };
        base_children[s] = slot_overlay(library, terrain_child_id, sub);
    }
    library.insert_with_kind(base_children, kind)
}

/// Build one slot's overlay. If the trie sub-node has a leaf entity
/// (and no further descent), return `Child::EntityRef(idx)`.
/// Otherwise recurse to produce a scene child Node.
fn slot_overlay(
    library: &mut NodeLibrary,
    terrain_child_id: NodeId,
    trie: &TrieNode,
) -> Child {
    let has_descent = trie.slot_children.iter().any(|c| c.is_some());
    if let Some(eidx) = trie.leaf_entity {
        if !has_descent {
            // Pure leaf: entity's anchor bottoms out here.
            return Child::EntityRef(eidx);
        }
        // Mixed: entity at this level AND descendants below. The
        // overlay at this level is an EntityRef-leaf; descendants
        // are dropped because `Child` can only hold one thing per
        // slot. Game-layer spawn already guarantees unique anchor
        // cells via the spawn grid, so this path shouldn't fire.
        return Child::EntityRef(eidx);
    }
    // Pure descent: build the scene sub-node recursively.
    let id = build_scene_node(library, terrain_child_id, trie);
    Child::Node(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, uniform_children, CENTER_SLOT};

    /// One entity at [CENTER_SLOT, 5] — two levels below an empty
    /// terrain root. Scene root should have an ephemeral chain
    /// ending in EntityRef(0).
    #[test]
    fn single_entity_into_empty_terrain_creates_overlay_chain() {
        let mut lib = NodeLibrary::default();
        let terrain = lib.insert(empty_children());
        lib.ref_inc(terrain);

        let scene = build_scene_root(
            &mut lib,
            terrain,
            &[EntityPath { entity_idx: 0, path_slots: vec![CENTER_SLOT as u8, 5] }],
        );
        assert_ne!(scene.node_id, terrain, "scene root must differ from terrain when overlay added");

        let root = lib.get(scene.node_id).expect("scene root in library");
        let Child::Node(mid_id) = root.children[CENTER_SLOT] else {
            panic!("scene root's center slot must be Node pointing at the overlay chain");
        };
        let mid = lib.get(mid_id).expect("mid in library");
        match mid.children[5] {
            Child::EntityRef(0) => {}
            other => panic!("expected EntityRef(0) at slot 5 of overlay mid, got {other:?}"),
        }
    }

    /// Two entities under the same ancestor slot at different leaf
    /// slots should share the intermediate ephemeral node.
    #[test]
    fn siblings_share_intermediate_scene_node() {
        let mut lib = NodeLibrary::default();
        let terrain = lib.insert(empty_children());
        lib.ref_inc(terrain);

        let scene = build_scene_root(&mut lib, terrain, &[
            EntityPath { entity_idx: 0, path_slots: vec![CENTER_SLOT as u8, 5] },
            EntityPath { entity_idx: 1, path_slots: vec![CENTER_SLOT as u8, 7] },
        ]);
        let root = lib.get(scene.node_id).unwrap();
        let Child::Node(mid_id) = root.children[CENTER_SLOT] else {
            panic!("expected ephemeral mid node");
        };
        let mid = lib.get(mid_id).unwrap();
        assert!(matches!(mid.children[5], Child::EntityRef(0)));
        assert!(matches!(mid.children[7], Child::EntityRef(1)));
    }

    /// When terrain already has content at the scene's path, the
    /// scene's overlay should PRESERVE the non-path siblings.
    #[test]
    fn overlay_preserves_nonpath_terrain_siblings() {
        let mut lib = NodeLibrary::default();
        let leaf_grass = lib.insert(uniform_children(Child::Block(5)));
        let mut terrain_children = empty_children();
        terrain_children[CENTER_SLOT] = Child::Node(leaf_grass);
        terrain_children[0] = Child::Block(9); // unrelated sibling
        let terrain = lib.insert(terrain_children);
        lib.ref_inc(terrain);

        let scene = build_scene_root(&mut lib, terrain, &[
            EntityPath { entity_idx: 0, path_slots: vec![CENTER_SLOT as u8, 3] },
        ]);
        let root = lib.get(scene.node_id).unwrap();
        assert!(
            matches!(root.children[0], Child::Block(9)),
            "non-overlay sibling must be preserved verbatim",
        );
        let Child::Node(mid_id) = root.children[CENTER_SLOT] else {
            panic!("overlay path must descend through a Node");
        };
        let mid = lib.get(mid_id).unwrap();
        assert!(matches!(mid.children[3], Child::EntityRef(0)));
        // Siblings of the leaf slot inside the ephemeral mid should
        // match terrain's grass leaf subtree (all Block(5)).
        assert!(matches!(mid.children[4], Child::Block(5)));
        assert!(matches!(mid.children[0], Child::Block(5)));
    }

    /// Empty entity list returns terrain unchanged (no ephemeral).
    #[test]
    fn empty_entity_list_returns_terrain_root() {
        let mut lib = NodeLibrary::default();
        let terrain = lib.insert(empty_children());
        let scene = build_scene_root(&mut lib, terrain, &[]);
        assert_eq!(scene.node_id, terrain);
    }
}
