//! Step-1 unit primitive: a row of three sibling cubes at the
//! deepest layer of a 30-deep tree. The centre cube is a
//! `TangentBlock` rotated 45° around Y; the outer two are
//! axis-aligned `Block(stone)` leaves. Side-by-side comparison
//! makes the rotation visible by contrast.
//!
//! Tree shape (depth 30):
//!
//! ```text
//! root (Cartesian, depth 30)
//!   slot 13 -> Cartesian wrapper-29 (depth 29)
//!     slot 13 -> Cartesian wrapper-28 (depth 28)
//!       ...
//!         Cartesian row-wrapper (depth 2)
//!           slot 12 = (0,1,1) -> Block(stone)                          (axis-aligned A)
//!           slot 13 = (1,1,1) -> TangentBlock { rot_y(π/4) } (depth 1) (rotated)
//!                                  27 stone children
//!           slot 14 = (2,1,1) -> Block(stone)                          (axis-aligned B)
//! ```
//!
//! Camera spawn: 27 levels of centre-column (slot 13) descent, then
//! one slot 16 = (1, 2, 1) — the cell DIRECTLY ABOVE the row. Offset
//! (0.5, 0.5, 0.5) centres the camera in that cell, with pitch -π/2
//! looking straight down. Anchor depth = 28; render frame depth =
//! anchor − render_frame_k = 25 (well within f32 precision since
//! every step stays in `[0, 3)³` local coords — there are no
//! world-absolute coordinates anywhere in the descent).
//!
//! Visual expectation (same as the depth-3 test, just at maximum
//! precision pressure):
//!
//! - Axis-aligned dispatch only → three squares in a row
//! - Correct rotation dispatch    → square, diamond, square in a row
//!
//! If the diamond renders cleanly at this depth, the architecture
//! is precision-stable: the rotated-cube primitive holds at the
//! deepest layer the tree can reach.

use super::WorldBootstrap;
use crate::world::anchor::{Path, WorldPos};
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, rotation_y, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

/// Rotation angle for the TangentBlock, radians. 45° is the most
/// visually distinctive: from above the cube reads as a diamond.
const ROTATION_ANGLE_RAD: f32 = std::f32::consts::FRAC_PI_4;

/// Total tree depth produced by this preset. The 3 sibling cubes
/// live at the leaf layer; the camera spawns at depth
/// `TREE_DEPTH − 2 = 28`.
pub const ROTATED_CUBE_TEST_TREE_DEPTH: u32 = 30;

pub fn rotated_cube_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // Inner: rotated TangentBlock with uniform stone interior.
    let stone = uniform_children(Child::Block(block::STONE));
    let tangent_block_id = library.insert_with_kind(
        stone,
        NodeKind::TangentBlock {
            rotation: rotation_y(ROTATION_ANGLE_RAD),
        },
    );

    // Row wrapper: Cartesian with three populated slots forming a
    // horizontal X-axis row at (y=1, z=1):
    //   slot 12 (0,1,1): axis-aligned stone Block (left)
    //   slot 13 (1,1,1): rotated TangentBlock     (centre)
    //   slot 14 (2,1,1): axis-aligned stone Block (right)
    let mut row_children = empty_children();
    row_children[slot_index(0, 1, 1)] = Child::Block(block::STONE);
    row_children[slot_index(1, 1, 1)] = Child::Node(tangent_block_id);
    row_children[slot_index(2, 1, 1)] = Child::Block(block::STONE);
    let row_wrapper_id = library.insert_with_kind(row_children, NodeKind::Cartesian);

    // Wrap in centre-column Cartesian wrappers until the root reaches
    // tree depth `ROTATED_CUBE_TEST_TREE_DEPTH`. The row wrapper has
    // `Node.depth = 2` (its tallest child is the depth-1 TangentBlock),
    // so we need (TREE_DEPTH − 2) more wrappers to make the root
    // reach the target depth.
    let wrappers_needed = ROTATED_CUBE_TEST_TREE_DEPTH as usize - 2;
    let mut current = Child::Node(row_wrapper_id);
    for _ in 0..wrappers_needed {
        let mut children = empty_children();
        children[slot_index(1, 1, 1)] = current;
        current = Child::Node(library.insert_with_kind(children, NodeKind::Cartesian));
    }
    let root = match current {
        Child::Node(id) => id,
        _ => unreachable!("wrappers always produce Child::Node"),
    };
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "rotated_cube_test world: tree_depth={}, library_entries={}",
        world.tree_depth(),
        world.library.len(),
    );
    world
}

/// Camera spawn: 27 levels of centre-column descent + one slot 16
/// (the cell directly above the 3-sibling row). Anchor depth = 28;
/// camera offset (0.5, 0.5, 0.5) inside that cell. Pitch -π/2 looks
/// straight down so all three cubes are framed below.
pub fn rotated_cube_test_spawn() -> WorldPos {
    let mut anchor = Path::root();
    let centre = slot_index(1, 1, 1) as u8;
    // 27 × centre-column descent — drops us into the row wrapper's
    // own slot in its parent. One more slot (16 = above the row)
    // puts the camera in empty space directly above the cubes.
    for _ in 0..(ROTATED_CUBE_TEST_TREE_DEPTH as usize - 3) {
        anchor.push(centre);
    }
    anchor.push(slot_index(1, 2, 1) as u8); // (1, 2, 1) = above-row
    WorldPos::new(anchor, [0.5, 0.5, 0.5])
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
    fn centre_slot_is_tangent_block_outer_slots_are_block() {
        let world = rotated_cube_test_world();
        // Walk the centre-column chain down to the row wrapper.
        let mut node_id = world.root;
        let centre = slot_index(1, 1, 1);
        for _ in 0..(ROTATED_CUBE_TEST_TREE_DEPTH as usize - 2) {
            let n = world.library.get(node_id).expect("wrapper exists");
            assert_eq!(n.kind, NodeKind::Cartesian);
            match n.children[centre] {
                Child::Node(child) => node_id = child,
                other => panic!("expected Cartesian wrapper, got {other:?}"),
            }
        }
        let row = world.library.get(node_id).expect("row wrapper exists");
        assert_eq!(row.kind, NodeKind::Cartesian);
        // Centre slot 13 = TangentBlock
        match row.children[slot_index(1, 1, 1)] {
            Child::Node(tb_id) => {
                let tb = world.library.get(tb_id).unwrap();
                assert!(tb.kind.is_tangent_block(), "expected TB, got {:?}", tb.kind);
            }
            other => panic!("expected Node(TangentBlock), got {other:?}"),
        }
        // Outer slots = axis-aligned Block(stone)
        assert!(matches!(row.children[slot_index(0, 1, 1)], Child::Block(_)));
        assert!(matches!(row.children[slot_index(2, 1, 1)], Child::Block(_)));
    }
}
