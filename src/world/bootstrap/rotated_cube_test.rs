//! Step-1 unit primitive: a row of three sibling cubes, each itself
//! a depth-29 recursive subtree of uniform stone. Total tree depth =
//! 30. The centre cube's outermost node is a `TangentBlock` rotated
//! 45° around Y; the outer two are plain `Cartesian`. Side-by-side
//! comparison makes the rotation visible by contrast at any zoom
//! level.
//!
//! Tree shape:
//!
//! ```text
//! root (Cartesian, depth 30)
//!   slot 12 (0,1,1) -> Cartesian uniform-stone subtree    (depth 29)  axis-aligned A
//!   slot 7 (1,1,1) -> TangentBlock { rot_y(π/4) }        (depth 29)  rotated
//!                        wraps a 28-deep uniform-stone Cartesian subtree
//!   slot 14 (2,1,1) -> Cartesian uniform-stone subtree    (depth 29)  axis-aligned B
//! ```
//!
//! Each cube has 29 levels of recursive subdivision below it; zoom
//! reveals progressively finer cells. Because every cell at every
//! layer is just dedup'd uniform stone, the library stays small
//! (~30 entries total) regardless of the depth.
//!
//! Camera spawn: anchor depth 1, slot 16 (= (1, 2, 1) above the row),
//! offset (0.5, 0.5, 0.5), pitch −π/2 looking straight down.
//!
//! Visual expectation:
//!
//! - Axis-aligned dispatch only → three squares in a row
//! - Correct rotation dispatch    → square, diamond, square in a row
//!
//! The rotation must persist as the player zooms in and the render
//! frame descends past the TangentBlock — that's what the camera-
//! direction frame-chain rotation fix protects.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, rotation_y, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

/// Rotation angle for the TangentBlock, radians. 45° is the most
/// visually distinctive: from above the cube reads as a diamond.
const ROTATION_ANGLE_RAD: f32 = std::f32::consts::FRAC_PI_4;

/// Depth of each per-cube subtree (slots 5/7/6 of the root). The
/// total tree depth is `CUBE_SUBTREE_DEPTH + 1` (root + each
/// subtree). For the precision-pressure test, 29 puts the deepest
/// stone leaves at root tree depth 30.
pub const CUBE_SUBTREE_DEPTH: u8 = 29;

/// Total tree depth produced by this preset.
pub const ROTATED_CUBE_TEST_TREE_DEPTH: u32 = CUBE_SUBTREE_DEPTH as u32 + 1;

/// Build a uniform-block recursive Cartesian subtree of `depth`
/// levels. Returns a `Child` referencing the root of the subtree
/// (Block at depth 0, Node otherwise). Dedup means a depth-N tree
/// contributes at most N library entries — the same uniform node
/// is shared at every level.
fn build_uniform_cartesian_subtree(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
) -> Child {
    if depth == 0 {
        return Child::Block(block_id);
    }
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert(uniform_children(inner)))
}

/// Build a recursive subtree of `depth` levels whose OUTERMOST node
/// is a `TangentBlock` carrying `rotation`. All internal nodes
/// below the TB are plain Cartesian uniform stone; the rotation
/// only attaches to the TB itself. Asserts `depth >= 1` because a
/// TB with `depth == 0` would be a Block and couldn't carry a
/// NodeKind.
fn build_tangent_block_subtree(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
    rotation: [[f32; 3]; 3],
) -> Child {
    assert!(depth >= 1, "TangentBlock subtree depth must be >= 1");
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert_with_kind(
        uniform_children(inner),
        NodeKind::TangentBlock { rotation },
    ))
}

pub fn rotated_cube_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // Three sibling subtrees forming a horizontal row at root y=1, z=1.
    // Each is a depth-29 uniform-stone tree; the centre's outermost
    // node is a TangentBlock with the test rotation.
    let cube_a = build_uniform_cartesian_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH,
    );
    let cube_centre = build_tangent_block_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH, rotation_y(ROTATION_ANGLE_RAD),
    );
    let cube_b = build_uniform_cartesian_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH,
    );

    // Root: Cartesian, populated only at slots 5/7/6 (the row).
    let mut root_children = empty_children();
    root_children[slot_index(0, 1, 1)] = cube_a;
    root_children[slot_index(1, 1, 1)] = cube_centre;
    let root = library.insert_with_kind(root_children, NodeKind::Cartesian);
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "rotated_cube_test world: tree_depth={}, library_entries={}",
        world.tree_depth(),
        world.library.len(),
    );
    world
}

/// Camera spawn: one cell directly above the row, looking down.
/// Anchor depth 1 (slot 7 = (1, 1, 1) of root); offset (0.5, 0.5,
/// 0.5) puts world position at (1.0, 1.0, 1.0) in root frame coords.
pub fn rotated_cube_test_spawn() -> WorldPos {
    WorldPos::uniform_column(slot_index(1, 1, 1) as u8, 1, [0.5, 0.5, 0.5])
}

pub(super) fn bootstrap_rotated_cube_test_world() -> WorldBootstrap {
    let world = rotated_cube_test_world();
    let spawn_pos = rotated_cube_test_spawn();
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -std::f32::consts::FRAC_PI_2,
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_has_expected_depth() {
        let world = rotated_cube_test_world();
        assert_eq!(world.tree_depth(), ROTATED_CUBE_TEST_TREE_DEPTH);
    }

    #[test]
    fn library_dedups_uniform_subtrees() {
        let world = rotated_cube_test_world();
        // Depth-29 uniform Cartesian + 1 TB head = 30 library entries.
        // (Every level of uniform stone dedups across cubes A and B
        // with the inside of cube_centre.)
        assert!(
            world.library.len() <= 32,
            "library should dedup uniform subtrees, got {} entries",
            world.library.len(),
        );
    }

    #[test]
    fn root_slots_match_layout() {
        let world = rotated_cube_test_world();
        let root_node = world.library.get(world.root).expect("root exists");
        assert_eq!(root_node.kind, NodeKind::Cartesian);

        // Outer slots are uniform-stone Cartesian subtrees (Node, not
        // Block — they're recursive subdivisions, not leaves).
        for slot_xyz in [(0u8, 1u8, 1u8), (2, 1, 1)] {
            let slot = slot_index(slot_xyz.0 as usize, slot_xyz.1 as usize, slot_xyz.2 as usize);
            let child = root_node.children[slot];
            match child {
                Child::Node(id) => {
                    let n = world.library.get(id).expect("subtree exists");
                    assert_eq!(n.kind, NodeKind::Cartesian);
                }
                other => panic!("expected Node at slot {slot_xyz:?}, got {other:?}"),
            }
        }

        // Centre slot is a TangentBlock subtree.
        match root_node.children[slot_index(1, 1, 1)] {
            Child::Node(id) => {
                let n = world.library.get(id).expect("centre subtree exists");
                assert!(n.kind.is_tangent_block(), "expected TB, got {:?}", n.kind);
            }
            other => panic!("expected Node(TB) at centre slot, got {other:?}"),
        }
    }
}
