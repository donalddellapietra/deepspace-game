//! Single-rotated-block test world.
//!
//! Tree depth 31 (root + TangentBlock + 29-level patterned interior).
//! The world is a flat 3×3×3 cartesian root; slot (1,1,1) holds a
//! `NodeKind::TangentBlock` whose 29-level interior is a self-
//! similar 3-color patterned subtree. Adjacent slots
//! (0,1,1) / (2,1,1) carry control stone cubes so the rotation is
//! visually framed against axis-aligned siblings.
//!
//! Why 30 levels: depth 12 was the goal, but a 12-deep test world
//! lets a broken implementation cheat with absolute f32 coords
//! (3^-12 ≈ 5e-6 still fits in mantissa). At depth 30, 3^-30 ≈ 1e-14
//! is well past f32 precision — only a frame-local descent that
//! never touches absolute world coords below the rotation boundary
//! can render the deep cells correctly.
//!
//! Self-similar interior: every level inside the rotated subtree
//! references the SAME `patterned_node` of one shallower depth, so
//! at any zoom level the 3-color pattern is visible at the leaf
//! frontier. Library size is `O(depth)` — ~30 nodes — because
//! content-addressed dedup collapses identical references.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, BRANCH, Child, NodeId, NodeKind, NodeLibrary,
};

/// Tree depth of the patterned subtree under the TangentBlock node.
/// Total tree depth = root (1) + TangentBlock (1) +
/// `ROTATED_INTERIOR_DEPTH` = 31.
pub const ROTATED_INTERIOR_DEPTH: u32 = 29;

/// Build a self-similar 3-color patterned subtree of the given depth.
///
/// At depth 1 the 27 children are `Block` leaves colored by
/// `(x+y+z) % 3`. At depth K > 1 all 27 children are `Child::Node`
/// references to the depth-(K-1) patterned subtree. Result: at any
/// zoom level the rotated cube exposes the same 3-color stripe
/// signature one level finer — every level has visible non-uniform
/// content for rotation verification.
fn patterned_node(library: &mut NodeLibrary, depth: u32) -> NodeId {
    debug_assert!(depth >= 1);
    if depth == 1 {
        let mut children = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let color = match (x + y + z) % 3 {
                        0 => block::BRICK,
                        1 => block::STONE,
                        _ => block::WOOD,
                    };
                    children[slot_index(x, y, z)] = Child::Block(color);
                }
            }
        }
        library.insert(children)
    } else {
        let sub = patterned_node(library, depth - 1);
        let mut children = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] = Child::Node(sub);
                }
            }
        }
        library.insert(children)
    }
}

pub(crate) fn bootstrap_rotated_test_world() -> WorldBootstrap {
    let mut library = NodeLibrary::default();

    // Build the rotated subtree's interior (29 levels) as a
    // self-similar patterned chain, then wrap it in a TangentBlock
    // node to mark the rotation boundary for the renderer + walker.
    let interior_root = patterned_node(&mut library, ROTATED_INTERIOR_DEPTH);
    // The TangentBlock node is structurally a cartesian 3x3x3 with
    // 27 children pointing at the patterned subtree (so its FIRST
    // level of cells already shows the 3-color signature). The kind
    // tag tells the walker to apply a rotation when descending into
    // this node from any frame.
    let mut tb_children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                tb_children[slot_index(x, y, z)] = Child::Node(interior_root);
            }
        }
    }
    let rotated_id = library.insert_with_kind(tb_children, NodeKind::TangentBlock);

    // Root: 3x3x3 cartesian. Stone-cube controls flank the rotated
    // cell on either side along X so the rotation is visually
    // grounded against axis-aligned reference geometry.
    let mut root_children = empty_children();
    root_children[slot_index(0, 1, 1)] = Child::Block(block::STONE);
    root_children[slot_index(1, 1, 1)] = Child::Node(rotated_id);
    root_children[slot_index(2, 1, 1)] = Child::Block(block::STONE);
    let root = library.insert(root_children);
    library.ref_inc(root);

    let world = WorldState { root, library };

    // Spawn outside the rotated cell, looking down/in from +Y/+Z.
    // Anchor at depth 4 (shallow) so the active frame at startup
    // contains the whole world; in-game zoom (edit_depth) descends
    // the anchor as needed for fine-detail viewing.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.5, 2.4, 2.85],
        4,
    )
    .deepened_to(8);

    let tree_depth = world.tree_depth() as u8;

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.45,
        plain_layers: tree_depth,
        color_registry: ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_depth_is_at_least_30() {
        let boot = bootstrap_rotated_test_world();
        let depth = boot.world.tree_depth();
        // 30 is the architectural correctness bar — at depth 18+,
        // f32 absolute coords lose precision, so anything that
        // renders correctly here must be using frame-local descent.
        assert!(
            depth >= 30,
            "tree depth must be >= 30 to defeat absolute-coord cheats, got {}",
            depth
        );
    }

    #[test]
    fn rotated_node_has_kind_tangent_block() {
        let boot = bootstrap_rotated_test_world();
        let root_node = boot.world.library.get(boot.world.root).expect("root");
        let center_child = root_node.children[slot_index(1, 1, 1)];
        let rotated_id = match center_child {
            Child::Node(id) => id,
            other => panic!("expected center slot to be a Node, got {:?}", other),
        };
        let rotated = boot
            .world
            .library
            .get(rotated_id)
            .expect("rotated subtree present");
        assert_eq!(rotated.kind, NodeKind::TangentBlock);
    }

    /// Self-similarity: every level inside the rotated subtree refers
    /// to the same patterned_node one level shallower. Library size
    /// should be `ROTATED_INTERIOR_DEPTH + small overhead`, NOT
    /// O(3^depth).
    #[test]
    fn library_is_compact() {
        let boot = bootstrap_rotated_test_world();
        // 29 patterned-node levels + 1 TangentBlock + 1 root ≈ 31
        // nodes. Allow a modest fudge factor for the dedup machinery.
        let lib_count = boot.world.library.len();
        assert!(
            lib_count <= 64,
            "library should be O(depth), got {} entries",
            lib_count
        );
    }

    /// Adjacent stone cubes are present so visual tests have axis-
    /// aligned reference geometry to compare rotation against.
    #[test]
    fn stone_controls_flank_rotated_cell() {
        let boot = bootstrap_rotated_test_world();
        let root_node = boot.world.library.get(boot.world.root).expect("root");
        for slot in [slot_index(0, 1, 1), slot_index(2, 1, 1)] {
            match root_node.children[slot] {
                Child::Block(b) => assert_eq!(b, block::STONE),
                other => panic!("expected stone control at slot {}, got {:?}", slot, other),
            }
        }
    }
}
