use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::inventory::InventoryState;
use crate::world::collision::{self, PLAYER_H};
use crate::world::WorldState;

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const JUMP_IMPULSE: f32 = 8.0;
const GRAVITY: f32 = 20.0;
pub const PLAYER_HEIGHT: f32 = PLAYER_H;

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
    // We start at depth 0, where 1 bevy unit = 1 super-chunk. The ground
    // super-chunk row at super_y=0 is fully populated (5 chunk layers), so its
    // top is at bevy y=1. Spawn just above and let gravity drop the player on.
    commands.spawn((Player, Velocity(Vec3::ZERO), Transform::from_xyz(0.5, 2.0, 0.5), Visibility::Hidden));
}

fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    state: Res<WorldState>,
    inv: Res<InventoryState>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
    camera_q: Query<&FpsCam>,
) {
    if inv.open { return }

    let Ok((mut tf, mut vel)) = player_q.single_mut() else { return };
    let Ok(cam) = camera_q.single() else { return };
    let dt = time.delta_secs();

    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let right = Vec3::new(-forward.z, 0.0, forward.x);

    let mut input = Vec2::ZERO;
    if keyboard.pressed(KeyCode::KeyW) { input.y += 1.0; }
    if keyboard.pressed(KeyCode::KeyS) { input.y -= 1.0; }
    if keyboard.pressed(KeyCode::KeyD) { input.x += 1.0; }
    if keyboard.pressed(KeyCode::KeyA) { input.x -= 1.0; }

    if keyboard.just_pressed(KeyCode::Space) && collision::on_ground(tf.translation, &*state) {
        vel.0.y = JUMP_IMPULSE;
    }

    vel.0.y -= GRAVITY * dt;

    let speed = if keyboard.pressed(KeyCode::ShiftLeft) { SPRINT_SPEED } else { WALK_SPEED };
    let h_delta = if input.length_squared() > 0.0 {
        let dir = (forward * input.y + right * input.x).normalize();
        Vec2::new(dir.x * speed * dt, dir.z * speed * dt)
    } else {
        Vec2::ZERO
    };

    collision::move_and_collide(&mut tf.translation, &mut vel.0, h_delta, dt, &*state);
}
