//! Force-extend the world tree along the camera anchor so the
//! render-frame walker reaches the anchor's depth.
//!
//! The render frame stops at the first non-Node child along the
//! anchor (`compute_render_frame` in `app/frame.rs`). When the anchor
//! extends past where the world tree has Nodes — i.e. the camera is
//! positioned with sub-cell precision inside an `Empty` or solid
//! `Block` leaf — the gap between `anchor.depth()` and
//! `render_path.depth()` can exceed the shader's `MAX_STACK_DEPTH=8`
//! and the deepest cells become invisible.
//!
//! The fix: where the walk terminates at `Child::Empty` or
//! `Child::Block(b)`, install a uniform subtree of the same content
//! reaching anchor depth. The content-addressed library dedups
//! uniform-empty and uniform-block chains across the world, so the
//! cost is bounded by `O(anchor_depth)` distinct nodes per (content,
//! depth) pair regardless of how many anchors land in empty space.
//!
//! Idempotent: subsequent calls along the same anchor walk the
//! already-installed uniform chain and return `false`. The path
//! through uniform Nodes works for any sub-slot indices because
//! every slot at every depth is the same uniform child.

use super::state::WorldState;
use super::tree::{
    uniform_children, Child, NodeId, NodeKind, NodeLibrary,
};

impl WorldState {
    /// Ensure the world tree has Cartesian Nodes along `anchor_slots`
    /// down to `anchor_slots.len()` levels. Returns `true` if the
    /// root changed.
    ///
    /// Stops at WrappedPlane / TangentBlock / EntityRef parents —
    /// those carry shader-side dispatch semantics this routine must
    /// not disturb. The user's deep-anchor empty-space case is pure
    /// Cartesian; tangent-block extension can be added if needed.
    pub fn ensure_anchor_extended(&mut self, anchor_slots: &[u8]) -> bool {
        let mut walked: Vec<(NodeId, usize)> =
            Vec::with_capacity(anchor_slots.len());
        let mut current_id = self.root;

        for (level, &slot) in anchor_slots.iter().enumerate() {
            let slot = slot as usize;
            let node = match self.library.get(current_id) {
                Some(n) => n,
                None => return false,
            };
            if !matches!(node.kind, NodeKind::Cartesian) {
                return false;
            }
            let child = node.children[slot];
            walked.push((current_id, slot));
            match child {
                Child::Node(child_id) => {
                    current_id = child_id;
                }
                Child::EntityRef(_) => return false,
                Child::Empty | Child::Block(_) => {
                    let levels = anchor_slots.len() - level;
                    let new_child = uniform_chain(
                        &mut self.library, child, levels,
                    );
                    return rebuild_ancestors(self, &walked, new_child);
                }
            }
        }
        false
    }
}

/// Wrap `seed` in `levels` uniform Cartesian Nodes. `levels > 0`,
/// result is always `Child::Node(_)`.
fn uniform_chain(
    library: &mut NodeLibrary,
    seed: Child,
    levels: usize,
) -> Child {
    debug_assert!(levels > 0);
    let mut child = seed;
    for _ in 0..levels {
        let id = library.insert(uniform_children(child));
        child = Child::Node(id);
    }
    child
}

/// Clone-on-write rebuild from `walked`'s deepest `(parent, slot)` up
/// to a fresh root, installing `replacement` at the deepest slot.
fn rebuild_ancestors(
    world: &mut WorldState,
    walked: &[(NodeId, usize)],
    replacement: Child,
) -> bool {
    let mut child = replacement;
    for &(parent_id, slot) in walked.iter().rev() {
        let node = match world.library.get(parent_id) {
            Some(n) => n,
            None => return false,
        };
        let kind = node.kind;
        let mut new_children = node.children;
        new_children[slot] = child;
        let new_id = world.library.insert_with_kind(new_children, kind);
        child = Child::Node(new_id);
    }
    if let Child::Node(new_root) = child {
        world.swap_root(new_root);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, slot_index, Child, NodeLibrary};

    fn slot_path(slots: &[usize]) -> Vec<u8> {
        slots.iter().map(|&s| s as u8).collect()
    }

    #[test]
    fn extends_through_empty_to_anchor_depth() {
        // Tree: root has Empty at slot 13. Anchor goes 5 levels deep.
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let anchor = slot_path(&[13, 0, 1, 2, 13]);
        let changed = world.ensure_anchor_extended(&anchor);
        assert!(changed, "extending through Empty must change root");

        // Walk the anchor — must reach depth 5 through Nodes.
        let mut current = world.root;
        for (level, &slot) in anchor.iter().enumerate() {
            let node = world.library.get(current).expect("node");
            match node.children[slot as usize] {
                Child::Node(child_id) => current = child_id,
                other => panic!(
                    "anchor descent broke at level {} slot {}: {:?}",
                    level, slot, other
                ),
            }
        }
    }

    #[test]
    fn idempotent_second_call_no_op() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let anchor = slot_path(&[13, 0, 1, 2, 13]);
        assert!(world.ensure_anchor_extended(&anchor));
        let root_after_first = world.root;
        assert!(!world.ensure_anchor_extended(&anchor));
        assert_eq!(world.root, root_after_first, "root must not change");
    }

    #[test]
    fn extends_through_block_with_uniform_block() {
        // Root has a Block at slot 5; extension must subdivide.
        let mut lib = NodeLibrary::default();
        let mut children = empty_children();
        children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(children);
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let anchor = slot_path(&[5, 13, 13]);
        assert!(world.ensure_anchor_extended(&anchor));

        let mut current = world.root;
        for (level, &slot) in anchor.iter().enumerate() {
            let node = world.library.get(current).expect("node");
            match node.children[slot as usize] {
                Child::Node(child_id) => current = child_id,
                Child::Block(_) if level + 1 == anchor.len() => {
                    // Deepest leaf may be Block; deeper steps already
                    // succeeded as Node.
                }
                other => panic!("unexpected child at level {level}: {other:?}"),
            }
        }
    }

    #[test]
    fn second_lateral_anchor_reuses_uniform_chain() {
        // After extending [13, 0, 1, 2, 13], walking [13, 1, 1, 2, 13]
        // (different slot at depth 1) must succeed without further
        // extension — uniform Nodes have all slots Identical by dedup.
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        world.ensure_anchor_extended(&slot_path(&[13, 0, 1, 2, 13]));
        let root_after = world.root;
        // Different middle slots: still inside the uniform-empty
        // chain, no rebuild needed.
        assert!(!world.ensure_anchor_extended(&slot_path(&[13, 1, 1, 2, 13])));
        assert!(!world.ensure_anchor_extended(&slot_path(&[13, 5, 7, 19, 0])));
        assert_eq!(world.root, root_after);
    }

    #[test]
    fn empty_anchor_is_no_op() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };
        assert!(!world.ensure_anchor_extended(&[]));
    }

    #[test]
    fn does_not_overwrite_existing_node() {
        // Tree already has a Node at slot 13 of root. Extension along
        // [13, ...] must walk through it, not replace it.
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(uniform_children(Child::Block(
            crate::world::palette::block::STONE,
        )));
        let mut children = empty_children();
        children[slot_index(1, 1, 1)] = Child::Node(leaf);
        let root = lib.insert(children);
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let anchor = slot_path(&[slot_index(1, 1, 1), 7, 13]);
        assert!(world.ensure_anchor_extended(&anchor));
        // The leaf node still exists with its block content.
        let leaf_node = world.library.get(leaf).expect("leaf preserved");
        assert!(matches!(
            leaf_node.children[7],
            Child::Block(_)
        ));
    }
}
