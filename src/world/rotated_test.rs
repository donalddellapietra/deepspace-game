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

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, Child, NodeKind, NodeLibrary, BRANCH,
};

pub(crate) fn bootstrap_rotated_test_world() -> WorldBootstrap {
    let mut lib = NodeLibrary::default();

    // Interior of the rotated subtree — 27 brick cells. After
    // rotation + inscribed stretch, this renders as a solid brick
    // diamond-prism filling the parent cell's AABB.
    let mut rot_children = empty_children();
    for s in 0..27 {
        rot_children[s] = Child::Block(block::BRICK);
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

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.35,
        plain_layers: 2,
        color_registry: ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_rotated45y_node() {
        let boot = bootstrap_rotated_test_world();
        let rot = boot.world.library.get(
            match boot.world.library.get(boot.world.root).unwrap().children
                [slot_index(1, 1, 1)]
            {
                Child::Node(id) => id,
                other => panic!("expected rotated Node, got {other:?}"),
            },
        );
        let rot = rot.expect("rotated node present");
        assert_eq!(rot.kind, NodeKind::Rotated45Y);
    }
}
