//! Worldgen: produces an empty space tree.
//!
//! Planets are no longer baked into the voxel tree — the cubed-sphere
//! demo planet is rendered directly in the shader from its own data
//! buffer (see `cubesphere.rs`). The voxel tree is kept as a pure
//! "space / void" background so the raymarcher has something to hit
//! (nothing) around the sphere.
//!
//! The tree still nests `TARGET_DEPTH` levels deep so zoom levels,
//! edit depths, and LOD heuristics continue to work unchanged — we
//! just don't fill any of it with content at worldgen time. Edits
//! placed in empty space will still materialize correctly.
//!
//! This file used to contain SDF-into-voxel-tree planet generation.
//! That model has been removed in favor of the cubed-sphere approach
//! which handles large-scale voxel gameplay natively (blocks that
//! bulge outward on a sphere). If you need the old behavior, look at
//! the git history for `build_space_subtree`.

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
    use crate::world::edit;

    #[test]
    fn generated_world_is_empty() {
        let w = generate_world();
        let depth = w.tree_depth();
        assert_eq!(depth as usize, TARGET_DEPTH);
        // Every point in the world box should read as empty.
        for &p in &[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]] {
            assert!(!edit::is_solid_at(&w.library, w.root, p, depth),
                "expected empty at {:?}", p);
        }
    }
}
