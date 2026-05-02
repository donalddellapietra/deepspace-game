//! Single rotated cube — minimal world for the TangentBlock-from-
//! Cartesian dispatch prototype.
//!
//! Pure Cartesian world tree. One slot of the root holds a
//! `NodeKind::TangentBlock` whose interior is uniform grass. The
//! shader's `march_cartesian` recognises the kind on descent,
//! transforms the ray into the cube's local frame using a rotation
//! computed at descent time, runs `march_in_tangent_cube` inside,
//! and rotates the returned normal back to world.
//!
//! Precision discipline: nothing about the world data is rotated.
//! Cube position and size are exact (tree-derived). The rotation
//! lives in the shader, applied on entry to one cell — never
//! propagated through descent. Below the cube's interior, descent
//! is shallow Cartesian (one level of grass), so f32 is plenty.
//!
//! This is the single-primitive proof that gates the larger
//! "voxel sphere of rotated tiles" work.

use super::WorldBootstrap;
use crate::world::anchor::{Path, WorldPos};
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

pub fn rotated_cube_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // The cube: a TangentBlock node whose 27 children are all grass.
    // TangentBlock is not allowed to uniform-flatten (per the
    // NodeKind invariant in tree.rs), so we keep the explicit node
    // here — the shader uses its presence to dispatch the rotation.
    let cube_node = library.insert_with_kind(
        uniform_children(Child::Block(block::GRASS)),
        NodeKind::TangentBlock,
    );

    // Place the cube two levels deep so plenty of empty space surrounds
    // it in the camera view. Path: root.slot13 → empty-cartesian.slot13
    // → cube. Cube spans world [1+1/3, 1+2/3)³ — a 1/3-side cube at
    // world centre.
    let mut depth1_children = empty_children();
    depth1_children[slot_index(1, 1, 1)] = Child::Node(cube_node);
    let depth1_node = library.insert(depth1_children);

    let mut root_children = empty_children();
    root_children[slot_index(1, 1, 1)] = Child::Node(depth1_node);
    let root = library.insert(root_children);
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "Rotated cube world: {} library entries, depth={}",
        world.library.len(),
        world.tree_depth(),
    );
    world
}

pub(crate) fn bootstrap_rotated_cube_world() -> WorldBootstrap {
    let world = rotated_cube_world();

    // Cube spans world [4/3, 5/3)³ ≈ [1.333, 1.667)³ (centre at 1.5,
    // side 1/3). Place camera at (1.5, 1.5, 2.5) — ~0.83 world units in
    // front of the cube's near face, plenty of empty space to either
    // side, well inside the world root [0, 3)³.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.5, 1.5, 2.5],
        2,
    )
    .deepened_to(8);

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // yaw=0 is -Z (looking into the screen) per camera.rs basis.
        default_spawn_yaw: 0.0,
        default_spawn_pitch: 0.0,
        plain_layers: 1,
        color_registry: ColorRegistry::new(),
    }
}
