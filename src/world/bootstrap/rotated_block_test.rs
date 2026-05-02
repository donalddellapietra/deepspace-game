//! Step-1 unit primitive: a single rotated `TangentBlock` cube at
//! tree depth 30, flanked by two axis-aligned cartesian Block siblings
//! as a sanity baseline.
//!
//! Tree shape (depth 30):
//!
//! ```text
//! root (Cartesian, depth 30)
//!   slot 13 -> Cartesian (depth 29)
//!     ... 27 more centre-slot Cartesian wrappers ...
//!       Cartesian (depth 1, the deepest internal node)
//!         slot 12 (=(0,1,1)) -> Block(stone)            // axis-aligned A
//!         slot 13 (=(1,1,1)) -> TangentBlock { rot_y(30°) }
//!                                with 27 stone-Block children
//!         slot 14 (=(2,1,1)) -> Block(stone)            // axis-aligned B
//! ```
//!
//! All three siblings live at the same depth — the rotation must
//! affect ONLY the centre cube. The two cartesian siblings are the
//! reference: if they render differently between pre/post change, the
//! cartesian descent is broken; if the centre renders unrotated, the
//! TangentBlock dispatch is broken; if the centre renders rotated and
//! the siblings stay axis-aligned, Step 1 is correct.
//!
//! Precision claim: every level walks `[0, 3)³` frame-local — there
//! are no world-space absolute coordinates. The tree depth (30) is
//! deeper than f32's 23-bit absolute-coord wall by design; if the
//! cube renders cleanly at this depth, the architecture passes the
//! precision gate.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, rotation_y, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

/// Rotation angle for the centre TangentBlock, radians.
const ROTATION_ANGLE_RAD: f32 = std::f32::consts::FRAC_PI_6; // 30°

/// Total tree depth produced by this preset.
pub const ROTATED_BLOCK_TEST_TREE_DEPTH: u32 = 30;

/// Camera anchor depth for default spawn. The path of 28 slot-13s
/// terminates at the cell containing the row-parent node; the row
/// parent's 3 sibling cells (slots 12/13/14 = the y=1, z=1 row)
/// occupy the central sub-cells. With the camera offset above that
/// row (y=0.85 ≈ row-parent local y ≈ 2.55), the render frame
/// (anchor - render_frame_k) puts all 3 cells in the visible
/// `[0, 3)³` at moderate cell size.
pub const ROTATED_BLOCK_TEST_SPAWN_ANCHOR_DEPTH: u8 = 28;

/// Build the test world.
///
/// Tree-construction strategy: bottom-up. We synthesize the deepest
/// 3-sibling node (the "row" parent) first, then wrap it in the
/// centre-slot Cartesian chain up to the root.
pub fn rotated_block_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // The TangentBlock cube: 27 stone children, rotated 30° around Y.
    let stone = Child::Block(block::STONE);
    let tangent_block_id = library.insert_with_kind(
        uniform_children(stone),
        NodeKind::TangentBlock {
            rotation: rotation_y(ROTATION_ANGLE_RAD),
        },
    );

    // The "row" parent at the deepest internal level: 3 populated
    // slots in a horizontal line (cell_y = 1, cell_z = 1).
    let mut row_children = empty_children();
    row_children[slot_index(0, 1, 1)] = Child::Block(block::STONE); // axis-aligned A
    row_children[slot_index(1, 1, 1)] = Child::Node(tangent_block_id); // rotated C
    row_children[slot_index(2, 1, 1)] = Child::Block(block::STONE); // axis-aligned B
    let mut current = Child::Node(library.insert_with_kind(row_children, NodeKind::Cartesian));

    // Wrap in centre-slot Cartesian nodes until the root reaches
    // tree depth 30. The row parent has Node.depth = 2 (it has
    // a Node child, the TangentBlock leaf with Node.depth = 1), so
    // we need (30 - 2) = 28 more wrappers to make the root depth 30.
    let wrappers_needed = ROTATED_BLOCK_TEST_TREE_DEPTH as usize - 2;
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
        "rotated_block_test world: tree_depth={}, library_entries={}",
        world.tree_depth(),
        world.library.len(),
    );
    world
}

/// Camera spawn: anchored at the depth-29 row-parent, positioned
/// just above the 3-sibling row (y=1) looking down at it.
///
/// Local offset within the row-parent's `[0, 3)³`:
/// - x = 1.5 (centred over the row)
/// - y = 2.5 (above the row's y=1 cells; ample sky margin)
/// - z = 1.5 (centred along z)
pub fn rotated_block_test_spawn() -> WorldPos {
    WorldPos::uniform_column(
        slot_index(1, 1, 1) as u8,
        ROTATED_BLOCK_TEST_SPAWN_ANCHOR_DEPTH,
        [1.5, 2.5, 1.5],
    )
}

pub(super) fn bootstrap_rotated_block_test_world() -> WorldBootstrap {
    let world = rotated_block_test_world();
    let spawn_pos = rotated_block_test_spawn();
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
    fn rotated_block_test_tree_has_expected_depth() {
        let world = rotated_block_test_world();
        assert_eq!(world.tree_depth(), ROTATED_BLOCK_TEST_TREE_DEPTH);
    }

    #[test]
    fn rotated_block_test_centre_slot_is_tangent_block() {
        let world = rotated_block_test_world();
        // Walk the centre column for 28 levels, then dive into slot
        // (1, 1, 1) of the row parent — that's the rotated TB.
        let mut node_id = world.root;
        for _ in 0..(ROTATED_BLOCK_TEST_TREE_DEPTH as usize - 2) {
            let node = world.library.get(node_id).expect("wrapper node exists");
            assert_eq!(node.kind, NodeKind::Cartesian);
            match node.children[slot_index(1, 1, 1)] {
                Child::Node(child) => node_id = child,
                other => panic!("expected Cartesian wrapper, got {other:?}"),
            }
        }
        // Row parent
        let row_node = world.library.get(node_id).expect("row parent exists");
        assert_eq!(row_node.kind, NodeKind::Cartesian);
        // Centre slot is TangentBlock
        match row_node.children[slot_index(1, 1, 1)] {
            Child::Node(tb_id) => {
                let tb = world.library.get(tb_id).expect("TB exists");
                assert!(tb.kind.is_tangent_block(), "centre cell is TangentBlock");
            }
            other => panic!("expected Node(TangentBlock), got {other:?}"),
        }
        // Sibling slots are axis-aligned Block
        assert!(matches!(row_node.children[slot_index(0, 1, 1)], Child::Block(_)));
        assert!(matches!(row_node.children[slot_index(2, 1, 1)], Child::Block(_)));
    }
}
