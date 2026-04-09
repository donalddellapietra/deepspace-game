use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::camera::FpsCam;
use crate::layer::ActiveLayer;
use crate::world::{self, VoxelWorld};

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const JUMP_IMPULSE: f32 = 8.0;
const GRAVITY: f32 = 20.0;
/// Camera eye height above player.translation.y (which is feet).
pub const PLAYER_HEIGHT: f32 = 1.7;

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
    // Feet at y=1.0 = top of ground cells at top layer
    commands.spawn((Player, Velocity(Vec3::ZERO), Transform::from_xyz(0.5, 1.0, 0.5), Visibility::Hidden));
}

fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    active: Res<ActiveLayer>,
    world: Res<VoxelWorld>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
    camera_q: Query<&FpsCam>,
) {
    let Ok((mut tf, mut vel)) = player_q.single_mut() else { return };
    let Ok(cam) = camera_q.single() else { return };
    let dt = time.delta_secs();

    // Movement directions from camera yaw
    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let right = Vec3::new(-forward.z, 0.0, forward.x);
    let input = wasd(&keyboard);

    // --- Horizontal movement ---
    if input.length_squared() > 0.0 {
        let dir = (forward * input.y + right * input.x).normalize();
        let speed = if keyboard.pressed(KeyCode::ShiftLeft) { SPRINT_SPEED } else { WALK_SPEED };
        tf.translation.x += dir.x * speed * dt;
        tf.translation.z += dir.z * speed * dt;
    }

    // --- Gravity ---
    vel.0.y -= GRAVITY * dt;
    tf.translation.y += vel.0.y * dt;

    // --- Floor collision (computed AFTER all movement) ---
    let floor = if active.is_top_layer() {
        world::floor_top_layer(&world.cells, tf.translation)
    } else {
        world::floor_inner(&world, &active.nav_stack, tf.translation)
    };

    if tf.translation.y < floor {
        tf.translation.y = floor;
        vel.0.y = 0.0;
    }

    // --- Jump (after floor snap, so on_ground is accurate) ---
    let on_ground = (tf.translation.y - floor).abs() < 0.05;
    if keyboard.just_pressed(KeyCode::Space) && on_ground {
        vel.0.y = JUMP_IMPULSE;
    }

    // --- Clamp inside grid when drilled in ---
    if !active.is_top_layer() {
        let margin = 1.0 + NEIGHBOR_RANGE_F;
        let limit = MODEL_SIZE as f32 + margin;
        tf.translation.x = tf.translation.x.clamp(-margin, limit);
        tf.translation.z = tf.translation.z.clamp(-margin, limit);
    }
}

/// How far the player can walk into neighboring cells (must match NEIGHBOR_RANGE in world)
const NEIGHBOR_RANGE_F: f32 = 4.0 * MODEL_SIZE as f32;

fn wasd(kb: &ButtonInput<KeyCode>) -> Vec2 {
    let mut v = Vec2::ZERO;
    if kb.pressed(KeyCode::KeyW) { v.y += 1.0; }
    if kb.pressed(KeyCode::KeyS) { v.y -= 1.0; }
    if kb.pressed(KeyCode::KeyD) { v.x += 1.0; }
    if kb.pressed(KeyCode::KeyA) { v.x -= 1.0; }
    v
}
