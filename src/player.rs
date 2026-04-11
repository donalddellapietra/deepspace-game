//! Player entity. Gravity + WASD + jump driven by
//! [`crate::world::collision::move_and_collide`]. The player's
//! `Transform` lives in Bevy leaf-voxel space (1 unit = 1 voxel).

use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::inventory::InventoryState;
use crate::world::collision::{self, PLAYER_H};
use crate::world::render::cell_size_at_layer;
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
            .add_systems(Update, move_player);
    }
}

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Velocity(pub Vec3);

fn spawn_player(mut commands: Commands) {
    // Spawn well clear of the world's `-x, -z` corner. The all-zero
    // path leaf sits at the world's negative corner (`ROOT_ORIGIN`),
    // so spawning at Bevy `(0, 2, 0)` puts the player ~13 units from
    // the world edge in `-x` and `-z` — close enough that the
    // 400-unit render radius clips against the edge and the player
    // can fall off after a few steps. Pulling the spawn out to
    // `(500, 2, 500)` puts the entire 400-unit visible sphere
    // comfortably inside the world in every direction.
    commands.spawn((
        Player,
        Velocity(Vec3::ZERO),
        Transform::from_xyz(500.0, 2.0, 500.0),
        Visibility::Hidden,
    ));
}

fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    world: Res<WorldState>,
    inv: Res<InventoryState>,
    zoom: Res<CameraZoom>,
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
        && collision::on_ground(tf.translation, &world)
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

    collision::move_and_collide(&mut tf.translation, &mut vel.0, h_delta, dt, &world);
}
