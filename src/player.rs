//! Player entity. Gravity + WASD + jump, with [`WorldPosition`] as
//! the authoritative location and `Transform` purely derived.
//!
//! ## Data flow each frame
//!
//! 1. [`move_player`] reads input, `WorldPosition`, `Velocity`, and
//!    calls `collision::move_and_collide`, which mutates the
//!    `WorldPosition` in place via exact `i64` path math.
//! 2. [`sync_anchor_to_player`] copies the player's new integer
//!    leaf coord into the global [`WorldAnchor`] resource, so
//!    every subsequent system (rendering, raycasting, highlight
//!    gizmos, other entities' `Transform` derivations) sees a
//!    consistent "Bevy `(0, 0, 0)` is here" anchor.
//! 3. [`derive_transforms`] iterates every entity with both
//!    [`WorldPosition`] and `Transform` and writes the `Transform`
//!    from `bevy_from_position(pos, anchor)`. The player's
//!    `Transform` ends up at its sub-voxel offset — always tiny.
//!
//! The player's `Vec3` `Transform` is never a source of truth and
//! never accumulates a long-range coordinate. Every future entity
//! that adds a `WorldPosition` gets the same large-world behaviour
//! for free.

use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::inventory::InventoryState;
use crate::world::collision::{self, PLAYER_H};
use crate::world::position::Position;
use crate::world::generator::MAX_TERRAIN_AMPLITUDE;
use crate::world::state::{sphere_center, SPHERE_RADIUS};
use crate::world::view::{
    bevy_from_position, cell_size_at_layer, position_from_leaf_coord,
    position_to_leaf_coord, norm_for_layer, scale_for_layer, target_layer_for, WorldAnchor,
};
use crate::world::{CameraZoom, WorldPosition, WorldState};

pub const PLAYER_HEIGHT: f32 = PLAYER_H;

// Movement constants are expressed in CELLS per second / cells per
// second². At runtime they're multiplied by `cell_size_at_layer(zoom.layer)`
// to convert to leaves per second at the current zoom. The player
// crosses one cell in the same wall-clock time at every view layer —
// pressing Q to zoom out makes the player effectively bigger and
// faster, matching the 2D prototype's "cells subtend a constant
// visual angle" behaviour.
const WALK_SPEED_CELLS: f32 = 8.0;
const SPRINT_SPEED_CELLS: f32 = 16.0;
const JUMP_IMPULSE_CELLS: f32 = 8.0;
const GRAVITY_CELLS: f32 = 20.0;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player).add_systems(
            Update,
            (move_player, sync_anchor_to_player, derive_transforms).chain(),
        );
    }
}

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Velocity(pub Vec3);

/// Spawn 2 voxels above the north pole of the sphere, at the
/// world's x/z centre. The floating [`WorldAnchor`] tracks the
/// player's integer leaf coord, so precision is perfect regardless
/// of where in the 6-billion-leaf root this lands.
pub fn spawn_position() -> Position {
    let center = sphere_center();
    let max_h = SPHERE_RADIUS + MAX_TERRAIN_AMPLITUDE as i64;
    let coord = [center[0], center[1] + max_h + 2, center[2]];
    let mut pos = position_from_leaf_coord(coord)
        .expect("sphere spawn position inside world bounds");
    pos.offset = [0.5, 0.0, 0.5];
    pos
}

/// The [`WorldAnchor`] that would place the spawn `Position` at
/// Bevy `(0, 0, 0)`. Used by [`spawn_player`] to initialise the
/// resource at startup.
pub fn spawn_anchor() -> WorldAnchor {
    WorldAnchor {
        leaf_coord: position_to_leaf_coord(&spawn_position()),
        // Start at leaf layer (MAX_LAYER), target = MAX_LAYER, norm = 1.0
        norm: norm_for_layer(crate::world::tree::MAX_LAYER),
    }
}

/// Startup system: insert the `WorldAnchor` matching the spawn, and
/// spawn the player entity with a `WorldPosition` pointing at the
/// centre of the root. The `Transform` starts at the sub-voxel
/// offset; the `Update`-phase `derive_transforms` recomputes it on
/// the first frame.
fn spawn_player(mut commands: Commands) {
    let pos = spawn_position();
    let anchor = spawn_anchor();
    let translation = bevy_from_position(&pos, &anchor);
    commands.insert_resource(anchor);
    commands.spawn((
        Player,
        WorldPosition(pos),
        Velocity(Vec3::ZERO),
        Transform::from_translation(translation),
        Visibility::Hidden,
    ));
}

pub fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    world: Res<WorldState>,
    inv: Res<InventoryState>,
    zoom: Res<CameraZoom>,
    mut player_q: Query<(&mut WorldPosition, &mut Velocity), With<Player>>,
    camera_q: Query<&FpsCam>,
    mut timings: ResMut<crate::world::render::RenderTimings>,
) {
    if inv.open {
        return;
    }

    let Ok((mut world_pos, mut vel)) = player_q.single_mut() else {
        return;
    };
    let Ok(cam) = camera_q.single() else {
        return;
    };
    // Clamp dt to prevent runaway physics on slow frames (e.g.
    // mesh baking burst after a zoom change). Without this, a
    // 2-second bake frame causes enormous gravity velocity →
    // huge collision sweep → even slower frame → feedback loop.
    let dt = time.delta_secs().min(0.1);

    // Convert cell-rate constants into leaves-per-second values for
    // the current zoom level. At view L=12 (leaves) cell_size = 1,
    // so these match the original numbers; as you zoom out, every
    // speed and gravity scales linearly so the player still crosses
    // one cell per ~0.125s and jumps ~1.6 cells high.
    let cell = cell_size_at_layer(zoom.layer);
    let walk_speed = WALK_SPEED_CELLS * cell;
    let sprint_speed = SPRINT_SPEED_CELLS * cell;
    let jump_impulse = JUMP_IMPULSE_CELLS * cell;
    let gravity = GRAVITY_CELLS * cell;

    // Camera-relative horizontal basis.
    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let right = Vec3::new(-forward.z, 0.0, forward.x);

    let mut input = Vec2::ZERO;
    if keyboard.pressed(KeyCode::KeyW) {
        input.y += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        input.y -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        input.x += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        input.x -= 1.0;
    }

    // Jump (must be on ground before applying gravity).
    if keyboard.just_pressed(KeyCode::Space)
        && collision::on_ground(&world_pos.0, &world, zoom.layer)
    {
        vel.0.y = jump_impulse;
    }

    // Gravity.
    vel.0.y -= gravity * dt;

    // Horizontal movement delta in leaves.
    let speed = if keyboard.pressed(KeyCode::ShiftLeft) {
        sprint_speed
    } else {
        walk_speed
    };
    let h_delta = if input.length_squared() > 0.0 {
        let dir = (forward * input.y + right * input.x).normalize();
        Vec2::new(dir.x * speed * dt, dir.z * speed * dt)
    } else {
        Vec2::ZERO
    };

    let col_start = bevy::platform::time::Instant::now();
    collision::move_and_collide(
        &mut world_pos.0,
        &mut vel.0,
        h_delta,
        dt,
        &world,
        zoom.layer,
    );
    timings.collision_us = col_start.elapsed().as_micros() as u64;
}

/// After physics, copy the player's new integer leaf coord into the
/// global [`WorldAnchor`] resource. Subsequent systems this frame
/// (camera, highlight gizmo, rendering, raycasting) then share
/// one consistent frame whose Bevy `(0, 0, 0)` is the player's
/// current leaf.
pub fn sync_anchor_to_player(
    mut anchor: ResMut<WorldAnchor>,
    zoom: Res<CameraZoom>,
    player_q: Query<&WorldPosition, With<Player>>,
) {
    if let Ok(pos) = player_q.single() {
        anchor.leaf_coord = position_to_leaf_coord(&pos.0);
    }
    // Normalization disabled — raw leaf-voxel coordinates avoid tile
    // boundary seams. Re-enable when LOD composition is implemented.
    anchor.norm = 1.0;
}

/// Derive every entity's `Transform.translation` from its
/// [`WorldPosition`] and the current [`WorldAnchor`]. The player's
/// `Transform` always lands at its sub-voxel offset (tiny), and
/// any future entity with a `WorldPosition` gets anchor-relative
/// placement for free.
pub fn derive_transforms(
    anchor: Res<WorldAnchor>,
    mut q: Query<(&WorldPosition, &mut Transform)>,
) {
    for (world_pos, mut tf) in &mut q {
        tf.translation = bevy_from_position(&world_pos.0, &anchor);
    }
}
