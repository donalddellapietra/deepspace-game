//! Minimal test world for the `Rotated45Y` node kind.
//!
//! Root is a plain cartesian 3×3×3 of depth 1 with:
//! - y=0 plane: 9 stone cells (ground)
//! - y=1 slot (0,1,1): a stone cube (axis-aligned — control)
//! - y=1 slot (1,1,1): a Rotated45Y subtree filled with brick (diamond)
//! - y=1 slot (2,1,1): a stone cube (axis-aligned — control)
//!
//! Everything else empty. Camera spawns above and a bit behind, so the
//! three mid-row cells sit in view and the rotated one is clearly a
//! diamond prism between two normal cubes.
//!
//! The rotated subtree carries `ROTATED_INTERIOR_DEPTH` cartesian
//! layers inside — so any depth the player chooses to edit at (up to
//! that depth) has real cell structure underneath, not a single block
//! leaf. Uniform brick fill means content-addressed dedup collapses
//! the subtree to `ROTATED_INTERIOR_DEPTH` library nodes regardless
//! of width, and the GPU packer flattens the uniform cartesian chain
//! to a single Brick leaf per rotated slot — no shader cost for the
//! extra layers until the interior becomes non-uniform.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, Child, NodeKind, NodeLibrary, BRANCH,
};

/// Tree depth of the rotated subtree (i.e. levels inside the
/// rotated node). Matches the "30 layers" target for parity with
/// normal cartesian subtrees — so the rotated block can be dug
/// into at fine granularity the same as a regular block.
pub const ROTATED_INTERIOR_DEPTH: u32 = 30;

pub(crate) fn bootstrap_rotated_test_world() -> WorldBootstrap {
    let mut lib = NodeLibrary::default();

    // 29 cartesian layers of uniform brick, so wrapping them with
    // 27 slots in the rotated node yields tree_depth = 30 inside.
    let interior = lib.build_uniform_subtree(
        block::BRICK,
        ROTATED_INTERIOR_DEPTH - 1,
    );
    let mut rot_children = empty_children();
    for s in 0..27 {
        rot_children[s] = interior;
    }
    let rotated_id = lib.insert_with_kind(rot_children, NodeKind::Rotated45Y);

    // Root node — one level deep, 27 direct child slots.
    let mut root_children = empty_children();

    // Ground: y=0 plane filled with stone cubes.
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            root_children[slot_index(x, 0, z)] = Child::Block(block::STONE);
        }
    }

    // Middle row at (x, 1, 1): stone | rotated | stone.
    root_children[slot_index(0, 1, 1)] = Child::Block(block::STONE);
    root_children[slot_index(1, 1, 1)] = Child::Node(rotated_id);
    root_children[slot_index(2, 1, 1)] = Child::Block(block::STONE);

    let root = lib.insert(root_children);
    lib.ref_inc(root);
    let world = WorldState { root, library: lib };

    // Camera above and toward +Z, looking back along -Z at a downward
    // angle. Same construct-shallow-then-deepen pattern used by every
    // other fractal preset for f32-precise spawn at any anchor depth.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.5, 2.2, 2.85],
        2,
    )
    .deepened_to(8);

    // Tree depth is 1 (root) + ROTATED_INTERIOR_DEPTH (inside the
    // rotated node). plain_layers tracks the full depth so downstream
    // LOD / ribbon budgets don't cap below the rotated subtree's leaf
    // level.
    let tree_depth = world.tree_depth() as u8;

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.35,
        plain_layers: tree_depth,
        color_registry: ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::raycast::cpu_raycast;

    #[test]
    fn has_rotated45y_node() {
        let boot = bootstrap_rotated_test_world();
        let rot_id = match boot.world.library.get(boot.world.root).unwrap().children
            [slot_index(1, 1, 1)]
        {
            Child::Node(id) => id,
            other => panic!("expected rotated Node, got {other:?}"),
        };
        let rot = boot.world.library.get(rot_id).expect("rotated node present");
        assert_eq!(rot.kind, NodeKind::Rotated45Y);
        // Subtree depth should equal ROTATED_INTERIOR_DEPTH so edits
        // have full cell structure to navigate into.
        assert_eq!(rot.depth, ROTATED_INTERIOR_DEPTH);
    }

    /// Ray straight down through the centre of the rotated cell hits
    /// brick — the diamond's top face is a square in XZ covering the
    /// cell-centre column.
    #[test]
    fn raycast_center_hits_rotated_interior() {
        let boot = bootstrap_rotated_test_world();
        let hit = cpu_raycast(
            &boot.world.library,
            boot.world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            ROTATED_INTERIOR_DEPTH + 1,
        )
        .expect("ray straight down through diamond centre must hit");
        // Path entry 0 is (world.root, slot_of_rotated_in_root) =
        // slot_index(1, 1, 1) = 13.
        assert_eq!(
            hit.path[0].1,
            slot_index(1, 1, 1),
            "first path entry must be the rotated node's root slot"
        );
        // Deep path — if raycast walked the full 30-layer subtree it
        // will have ~30 entries; a shallow (1-entry) path means the
        // rotated dispatch treated the subtree as a single cell.
        assert!(
            hit.path.len() >= 2,
            "raycast must descend into the rotated subtree, got path.len={}",
            hit.path.len()
        );
    }

    /// Ray down through the −X−Z corner of the rotated cell lies
    /// outside the inscribed diamond, so it must pass through the
    /// empty corner and land on the stone ground cell beneath.
    #[test]
    fn raycast_diamond_corner_gap_reaches_ground() {
        let boot = bootstrap_rotated_test_world();
        let hit = cpu_raycast(
            &boot.world.library,
            boot.world.root,
            [1.1, 2.5, 1.1],
            [0.0, -1.0, 0.0],
            ROTATED_INTERIOR_DEPTH + 1,
        )
        .expect("ray down through diamond gap must hit ground");
        // First path entry is (world.root, slot_of_ground_below_cell).
        // (1.1, ?, 1.1) sits in root slot (1, 0, 1) = slot_index 10,
        // which is stone ground — NOT the rotated node at slot 13.
        assert_eq!(
            hit.path[0].1,
            slot_index(1, 0, 1),
            "corner-gap ray must hit the stone ground, not the rotated subtree"
        );
    }
}
