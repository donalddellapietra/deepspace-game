//! Player entity. Gravity + WASD + jump driven by
//! [`crate::world::collision::move_and_collide`]. The player's
//! `Transform` lives in Bevy leaf-voxel space (1 unit = 1 voxel).

use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::inventory::InventoryState;
use crate::world::collision::{self, PLAYER_H};
use crate::world::WorldState;

pub const PLAYER_HEIGHT: f32 = PLAYER_H;

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const JUMP_IMPULSE: f32 = 8.0;
const GRAVITY: f32 = 20.0;

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
    commands.spawn((
        Player,
        Velocity(Vec3::ZERO),
        Transform::from_xyz(0.0, 2.0, 0.0),
        Visibility::Hidden,
    ));
}

fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    world: Res<WorldState>,
    inv: Res<InventoryState>,
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
        vel.0.y = JUMP_IMPULSE;
    }

    // Gravity.
    vel.0.y -= GRAVITY * dt;

    // Horizontal movement delta.
    let speed = if keyboard.pressed(KeyCode::ShiftLeft) {
        SPRINT_SPEED
    } else {
        WALK_SPEED
    };
    let h_delta = if input.length_squared() > 0.0 {
        let dir = (forward * input.y + right * input.x).normalize();
        Vec2::new(dir.x * speed * dt, dir.z * speed * dt)
    } else {
        Vec2::ZERO
    };

    collision::move_and_collide(&mut tf.translation, &mut vel.0, h_delta, dt, &world);
}
