//! Player entity. Gravity + WASD + jump driven by
//! [`crate::world::collision::move_and_collide`].
//!
//! The player's `Transform.translation` is deliberately kept near
//! Bevy `(0, 0, 0)` by the floating [`WorldAnchor`] — every frame,
//! after physics, [`recenter_anchor`] moves the anchor's integer
//! leaf coord to track the player's current leaf position and
//! subtracts the integer part back out of the `Transform`. Only
//! the sub-leaf fractional drift is left in the `Transform`, so the
//! `f32` ever resolves a step size smaller than a leaf, no matter
//! how deep the player wanders into the 6-billion-leaf root.

use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::inventory::InventoryState;
use crate::world::collision::{self, PLAYER_H};
use crate::world::position::{Position, NODE_PATH_LEN};
use crate::world::tree::{slot_index, NODE_VOXELS_PER_AXIS};
use crate::world::view::{
    bevy_from_position, cell_size_at_layer, position_to_leaf_coord, WorldAnchor,
};
use crate::world::{CameraZoom, WorldState};

pub const PLAYER_HEIGHT: f32 = PLAYER_H;

// Movement constants are expressed in CELLS per second / cells per
// second². At runtime they're multiplied by `cell_size_at_layer(zoom.layer)`
// to convert to Bevy units. This way the player crosses one cell in
// the same wall-clock time at every view layer — pressing Q to zoom
// out makes the player effectively bigger and faster, matching the
// 2D prototype's "cells subtend a constant visual angle" behaviour.
const WALK_SPEED_CELLS: f32 = 8.0;
const SPRINT_SPEED_CELLS: f32 = 16.0;
const JUMP_IMPULSE_CELLS: f32 = 8.0;
const GRAVITY_CELLS: f32 = 20.0;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, (move_player, recenter_anchor).chain());
    }
}

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Velocity(pub Vec3);

/// Path-based spawn position. Returns a [`Position`] pointing at the
/// arithmetic centre of the `25³ × 5^MAX_LAYER`-leaf root node, one
/// row above the grass surface. Every path slot is `(2, _, 2)` (the
/// middle of the `5³` child array on `x`/`z`), and the in-leaf voxel
/// is centred too — so the spawn is as close to the root's
/// geometric centre as the tree structure permits.
///
/// Why this is only possible with the floating anchor: under the
/// old constant `ROOT_ORIGIN` the centre of the root was at Bevy
/// `x ≈ 3e9` where `f32` step size is hundreds of leaves — leaf-
/// level collision and picking would silently collapse. With
/// `WorldAnchor` tracking the player's integer leaf coord, the
/// player's `Transform` is tiny regardless of where they are, so
/// leaf precision is preserved all the way to the centre.
///
/// Bottom-row slot `(2, 0, 2)` at depth `MAX_LAYER - 1` puts the
/// leaf flush against the grass's top face; depth `MAX_LAYER - 2`
/// uses `sy = 1` so the layer-`(MAX_LAYER - 1)` node containing
/// that leaf sits just above the world floor.
fn spawn_position() -> Position {
    let mut path = [0u8; NODE_PATH_LEN];
    // Every level above (MAX_LAYER - 2): centre slot on x/z, floor on y.
    for depth in 0..(NODE_PATH_LEN - 2) {
        path[depth] = slot_index(2, 0, 2) as u8;
    }
    // Grandparent of the leaf: centre on x/z, one row above the floor
    // (so the leaf's y face lines up with the top of the grass).
    path[NODE_PATH_LEN - 2] = slot_index(2, 1, 2) as u8;
    // Leaf parent: centre on x/z, bottom of its parent on y.
    path[NODE_PATH_LEN - 1] = slot_index(2, 0, 2) as u8;
    let mid = (NODE_VOXELS_PER_AXIS / 2) as u8;
    Position {
        path,
        voxel: [mid, 2, mid],
        offset: [0.5, 0.0, 0.5],
    }
}

/// The [`WorldAnchor`] that places the spawn `Position` at Bevy
/// `(0, 0, 0)`. Used by [`spawn_player`] to initialise the
/// resource, and by [`spawn_translation`] so the reset-to-spawn
/// translation is consistently zero.
pub fn spawn_anchor() -> WorldAnchor {
    WorldAnchor {
        leaf_coord: position_to_leaf_coord(&spawn_position()),
    }
}

/// Bevy translation of the spawn point in the player's current
/// `anchor` frame. Used by `editor::tools::reset_player` to teleport
/// the player — passing the current anchor keeps the teleport
/// small-f32 regardless of how far the player has wandered since
/// startup.
pub fn spawn_translation(anchor: &WorldAnchor) -> Vec3 {
    bevy_from_position(&spawn_position(), anchor)
}

/// Startup system: insert the [`WorldAnchor`] resource so that the
/// player's spawn position sits at Bevy `(0, 0, 0)`, then spawn the
/// player entity with a zero translation. Anything Bevy-shaped that
/// runs after this reads a tiny `Transform` regardless of where the
/// spawn is in the world.
fn spawn_player(mut commands: Commands) {
    let anchor = spawn_anchor();
    // `bevy_from_position(spawn_position(), spawn_anchor())` evaluates
    // to `(0, 0, 0) + spawn_position().offset`, which lives in the
    // same leaf — we just store the offset as the initial
    // translation so the first frame's render/collision already
    // sees the correct sub-voxel drift.
    let translation = bevy_from_position(&spawn_position(), &anchor);
    commands.insert_resource(anchor);
    commands.spawn((
        Player,
        Velocity(Vec3::ZERO),
        Transform::from_translation(translation),
        Visibility::Hidden,
    ));
}

fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    world: Res<WorldState>,
    inv: Res<InventoryState>,
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
    camera_q: Query<&FpsCam>,
) {
    if inv.open {
        return;
    }

    let Ok((mut tf, mut vel)) = player_q.single_mut() else {
        return;
    };
    let Ok(cam) = camera_q.single() else {
        return;
    };
    let dt = time.delta_secs();

    // Convert cell-rate constants into Bevy-unit/second values for
    // the current zoom level. At view L=12 (leaves) cell_size = 1, so
    // these match the original numbers; as you zoom out, every speed
    // and gravity scales linearly so the player still crosses one
    // cell per ~0.125s and jumps ~1.6 cells high.
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
        && collision::on_ground(tf.translation, &world, zoom.layer, &anchor)
    {
        vel.0.y = jump_impulse;
    }

    // Gravity.
    vel.0.y -= gravity * dt;

    // Horizontal movement delta.
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

    collision::move_and_collide(
        &mut tf.translation,
        &mut vel.0,
        h_delta,
        dt,
        &world,
        zoom.layer,
        &anchor,
    );
}

/// Run after [`move_player`]: take the integer leaf part of the
/// player's drift in this frame and roll it into the anchor, leaving
/// only the sub-leaf fractional part in `Transform.translation`.
///
/// This is the mechanism that keeps f32 precision perfect even
/// though the player is conceptually traversing a 6-billion-leaf
/// world: the Bevy `Transform` never accumulates large magnitudes,
/// because every whole-leaf chunk of drift is paid out to the
/// anchor as an exact `i64` delta.
fn recenter_anchor(
    mut anchor: ResMut<WorldAnchor>,
    mut player_q: Query<&mut Transform, With<Player>>,
) {
    let Ok(mut tf) = player_q.single_mut() else {
        return;
    };
    // Integer leaf drift since the last recenter.
    let shift: [i64; 3] = [
        tf.translation.x.floor() as i64,
        tf.translation.y.floor() as i64,
        tf.translation.z.floor() as i64,
    ];
    if shift == [0, 0, 0] {
        return;
    }
    anchor.leaf_coord[0] += shift[0];
    anchor.leaf_coord[1] += shift[1];
    anchor.leaf_coord[2] += shift[2];
    tf.translation.x -= shift[0] as f32;
    tf.translation.y -= shift[1] as f32;
    tf.translation.z -= shift[2] as f32;
}
