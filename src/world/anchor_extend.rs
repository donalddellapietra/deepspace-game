//! Force-extend the world tree along the camera anchor so the
//! render-frame walker reaches anchor depth via real Nodes (not bare
//! `Child::Empty` or `Child::Block(b)` sparse leaves).
//!
//! ## Why
//!
//! `compute_render_frame` (`src/app/frame.rs`) descends `world.root`
//! along the anchor's slot path and stops at the first non-Node
//! child. When the anchor extends past where the tree has Nodes
//! (camera floating in air at deep zoom; sparse slabs in
//! wrapped-planet worlds), the gap between `anchor.depth()` and
//! `render_path.depth()` exceeds the shader's `MAX_STACK_DEPTH=8`
//! and the deepest cells become unreachable for editing /
//! highlighting. Worse: tiny camera moves that flip an intermediate
//! slot make the render frame oscillate between depths, producing
//! visible LOD popping.
//!
//! ## Fix
//!
//! Where the walk would terminate at `Child::Empty` or
//! `Child::Block(b)`, install a uniform subtree of matching content
//! reaching anchor depth. Content-addressed dedup (`NodeLibrary`)
//! collapses uniform-empty / uniform-block chains globally to a
//! single chain of NodeIds — so the cost is `O(anchor_depth)`
//! distinct nodes per (content, depth) pair, regardless of how many
//! anchor positions land in empty space.
//!
//! Idempotent: subsequent calls along the same anchor walk the
//! existing uniform chain and return without modifying the tree.
//! Uniform Nodes have all 27 slots identical, so any sub-slot path
//! the camera takes through a previously-extended region is also a
//! no-op.
//!
//! ## Scope
//!
//! Stops at `WrappedPlane` / `TangentBlock` / `EntityRef` parents —
//! those carry shader-side dispatch semantics this routine must not
//! disturb. The deep-anchor empty-space case is pure Cartesian.

use super::state::WorldState;
use super::tree::{
    uniform_children, Child, NodeId, NodeKind, NodeLibrary,
};

impl WorldState {
    /// Ensure Cartesian Nodes exist along `anchor_slots` down to its
    /// full depth. Returns `true` if `world.root` changed.
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
            // Cartesian-only extension; non-Cartesian kinds carry
            // shader-side semantics this routine must not disturb.
            if !matches!(node.kind, NodeKind::Cartesian) {
                return false;
            }
            let child = node.children[slot];
            walked.push((current_id, slot));
            match child {
                Child::Node(child_id) => current_id = child_id,
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
/// result is always `Child::Node(_)`. Content-addressed dedup means
/// repeated calls with the same `(seed, levels)` return the same
/// NodeId chain.
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
/// Mirrors `edit::propagate_edit`'s ascent pattern.
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

    fn slots(s: &[usize]) -> Vec<u8> { s.iter().map(|&v| v as u8).collect() }

    #[test]
    fn extends_through_empty_to_anchor_depth() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let anchor = slots(&[13, 0, 1, 2, 13]);
        assert!(world.ensure_anchor_extended(&anchor));

        let mut current = world.root;
        for (level, &slot) in anchor.iter().enumerate() {
            let node = world.library.get(current).expect("node");
            match node.children[slot as usize] {
                Child::Node(id) => current = id,
                other => panic!(
                    "anchor descent broke at level {level} slot {slot}: {other:?}"
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

        let anchor = slots(&[13, 0, 1, 2, 13]);
        assert!(world.ensure_anchor_extended(&anchor));
        let after_first = world.root;
        assert!(!world.ensure_anchor_extended(&anchor));
        assert_eq!(world.root, after_first);
    }

    #[test]
    fn extends_through_block_with_uniform_block() {
        let mut lib = NodeLibrary::default();
        let mut children = empty_children();
        children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(children);
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let anchor = slots(&[5, 13, 13]);
        assert!(world.ensure_anchor_extended(&anchor));

        let mut current = world.root;
        for (level, &slot) in anchor.iter().enumerate() {
            let node = world.library.get(current).expect("node");
            match node.children[slot as usize] {
                Child::Node(id) => current = id,
                Child::Block(_) if level + 1 == anchor.len() => {}
                other => panic!("unexpected at level {level}: {other:?}"),
            }
        }
    }

    #[test]
    fn second_lateral_anchor_reuses_uniform_chain() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        world.ensure_anchor_extended(&slots(&[13, 0, 1, 2, 13]));
        let after = world.root;
        assert!(!world.ensure_anchor_extended(&slots(&[13, 1, 1, 2, 13])));
        assert!(!world.ensure_anchor_extended(&slots(&[13, 5, 7, 19, 0])));
        assert_eq!(world.root, after);
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
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(uniform_children(Child::Block(
            crate::world::palette::block::STONE,
        )));
        let mut children = empty_children();
        children[slot_index(1, 1, 1)] = Child::Node(leaf);
        let root = lib.insert(children);
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let anchor = slots(&[slot_index(1, 1, 1), 7, 13]);
        assert!(world.ensure_anchor_extended(&anchor));
        let leaf_node = world.library.get(leaf).expect("leaf preserved");
        assert!(matches!(leaf_node.children[7], Child::Block(_)));
    }
}
