//! Worldgen: produces an empty space tree.
//!
//! The tree nests `TARGET_DEPTH` levels deep so zoom levels, edit
//! depths, and LOD heuristics work unchanged — we just don't fill
//! any of it with content at worldgen time. Edits placed in empty
//! space materialize correctly.

use super::state::WorldState;
use super::tree::*;

const TARGET_DEPTH: usize = 21;

pub fn generate_world() -> WorldState {
    let mut lib = NodeLibrary::default();

    // Build the deepest uniform-empty subtree and nest it all the way
    // up to the root. Every level gets dedup'd to a single node, so
    // the whole world is O(TARGET_DEPTH) unique entries.
    // Each `insert(uniform_children(Child::Node(_)))` adds one level
    // of nesting. We want total depth = TARGET_DEPTH, which is
    // 1 (the empty leaf) + (TARGET_DEPTH - 1) wraps.
    let mut root = lib.insert(empty_children());
    for _ in 1..TARGET_DEPTH {
        root = lib.insert(uniform_children(Child::Node(root)));
    }
    lib.ref_inc(root);

    let world = WorldState { root, library: lib };
    eprintln!(
        "Generated empty space tree: {} unique nodes, depth {}",
        world.library.len(), world.tree_depth(),
    );
    world
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::raycast;

    #[test]
    fn generated_world_is_empty() {
        let w = generate_world();
        let depth = w.tree_depth();
        assert_eq!(depth as usize, TARGET_DEPTH);
        // Every point in the world box should read as empty.
        for &p in &[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]] {
            assert!(!raycast::is_solid_at(&w.library, w.root, p, depth),
                "expected empty at {:?}", p);
        }
    }
}
