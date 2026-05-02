//! Step-1 unit primitive: a row of three sibling cubes at tree
//! depth 3 — the centre is a `TangentBlock` rotated 45° around Y,
//! the two outer cubes are axis-aligned `Block(stone)` leaves.
//!
//! Side-by-side comparison in a single render makes the rotation
//! visible by contrast: if the centre is a diamond and the outer
//! two are squares, the dispatch is correct.
//!
//! Tree shape:
//!
//! ```text
//! root (Cartesian, depth 3)
//!   slot 13 -> Cartesian wrapper (depth 2)
//!     slot 12 = (0,1,1) -> Block(stone)                            (axis-aligned A)
//!     slot 13 = (1,1,1) -> TangentBlock { rot_y(π/4) } (depth 1)   (rotated)
//!                            27 stone children
//!     slot 14 = (2,1,1) -> Block(stone)                            (axis-aligned B)
//! ```
//!
//! In root's `[0, 3)³` frame the row spans X=[1, 2], Y=[1.333, 1.667],
//! Z=[1.333, 1.667] — three 1/3-sided sub-cubes in a horizontal line.
//! Camera spawns one cell directly above (root frame ≈ (1.5, 2.5, 1.5))
//! looking straight down. From above:
//!
//! - Axis-aligned dispatch only → three squares in a row
//! - Correct rotation dispatch    → square, diamond, square in a row

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

    // Wrapper: Cartesian with three populated slots forming a
    // horizontal X-axis row at (y=1, z=1):
    //   slot 12 (0,1,1): axis-aligned stone Block (left)
    //   slot 13 (1,1,1): rotated TangentBlock     (centre)
    //   slot 14 (2,1,1): axis-aligned stone Block (right)
    let mut wrapper_children = empty_children();
    wrapper_children[slot_index(0, 1, 1)] = Child::Block(block::STONE);
    wrapper_children[slot_index(1, 1, 1)] = Child::Node(tangent_block_id);
    wrapper_children[slot_index(2, 1, 1)] = Child::Block(block::STONE);
    let wrapper_id = library.insert_with_kind(wrapper_children, NodeKind::Cartesian);

    // Root: Cartesian, only slot 13 populated with the wrapper.
    let mut root_children = empty_children();
    root_children[slot_index(1, 1, 1)] = Child::Node(wrapper_id);
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

/// Camera spawn: one cell directly above the cube, looking down.
/// Anchor depth 1 (slot 22 = (1, 2, 1) of root); offset centres
/// the camera inside that slot, putting world position at root
/// `[0, 3)³` frame coords (1.5, 2.5, 1.5).
pub fn rotated_cube_test_spawn() -> WorldPos {
    WorldPos::uniform_column(
        slot_index(1, 2, 1) as u8,
        1,
        [0.5, 0.5, 0.5],
    )
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
        assert_eq!(world.tree_depth(), 3);
    }

    #[test]
    fn centre_slot_is_tangent_block_outer_slots_are_block() {
        let world = rotated_cube_test_world();
        // root → slot 13 = wrapper
        let root_node = world.library.get(world.root).unwrap();
        let wrapper_id = match root_node.children[slot_index(1, 1, 1)] {
            Child::Node(id) => id,
            other => panic!("expected Node at root slot 13, got {other:?}"),
        };
        let wrapper = world.library.get(wrapper_id).unwrap();
        assert_eq!(wrapper.kind, NodeKind::Cartesian);
        // Centre slot 13 = TangentBlock
        match wrapper.children[slot_index(1, 1, 1)] {
            Child::Node(tb_id) => {
                let tb = world.library.get(tb_id).unwrap();
                assert!(tb.kind.is_tangent_block(), "expected TB, got {:?}", tb.kind);
            }
            other => panic!("expected Node(TangentBlock), got {other:?}"),
        }
        // Outer slots = axis-aligned Block(stone)
        assert!(matches!(wrapper.children[slot_index(0, 1, 1)], Child::Block(_)));
        assert!(matches!(wrapper.children[slot_index(2, 1, 1)], Child::Block(_)));
    }
}
