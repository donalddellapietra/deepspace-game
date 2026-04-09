use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::camera::FpsCam;
use crate::layer::ActiveLayer;
use crate::world::{self, VoxelWorld};

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const JUMP_IMPULSE: f32 = 8.0;
const GRAVITY: f32 = 20.0;
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

    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let right = Vec3::new(-forward.z, 0.0, forward.x);
    let input = wasd(&keyboard);
    let speed = if keyboard.pressed(KeyCode::ShiftLeft) { SPRINT_SPEED } else { WALK_SPEED };

    // --- Horizontal movement with collision ---
    if input.length_squared() > 0.0 {
        let dir = (forward * input.y + right * input.x).normalize();

        // Try X and Z separately so you slide along walls
        let new_x = tf.translation.x + dir.x * speed * dt;
        if !is_blocked(&world, &active, Vec3::new(new_x, tf.translation.y, tf.translation.z)) {
            tf.translation.x = new_x;
        }
        let new_z = tf.translation.z + dir.z * speed * dt;
        if !is_blocked(&world, &active, Vec3::new(tf.translation.x, tf.translation.y, new_z)) {
            tf.translation.z = new_z;
        }
    }

    // --- Gravity ---
    vel.0.y -= GRAVITY * dt;
    tf.translation.y += vel.0.y * dt;

    // --- Floor collision (AFTER all movement) ---
    let floor = if active.is_top_layer() {
        world::floor_top_layer(&world.cells, tf.translation)
    } else {
        world::floor_inner(&world, &active.nav_stack, tf.translation)
    };

    if tf.translation.y < floor {
        tf.translation.y = floor;
        vel.0.y = 0.0;
    }

    // --- Jump ---
    if keyboard.just_pressed(KeyCode::Space) && (tf.translation.y - floor).abs() < 0.05 {
        vel.0.y = JUMP_IMPULSE;
    }

    // --- Clamp inside navigable area ---
    if !active.is_top_layer() {
        let margin = NEIGHBOR_RANGE_BLOCKS;
        let limit = MODEL_SIZE as f32 + margin;
        tf.translation.x = tf.translation.x.clamp(-margin, limit);
        tf.translation.z = tf.translation.z.clamp(-margin, limit);
    }
}

const NEIGHBOR_RANGE_BLOCKS: f32 = 4.0 * MODEL_SIZE as f32;

/// Check if feet or torso would be inside a solid block.
fn is_blocked(world: &VoxelWorld, active: &ActiveLayer, pos: Vec3) -> bool {
    // Check at feet+0.1 and mid-body
    for dy in [0.1, PLAYER_HEIGHT * 0.5] {
        if world::is_solid_at(world, &active.nav_stack, Vec3::new(pos.x, pos.y + dy, pos.z)) {
            return true;
        }
    }
    false
}

fn wasd(kb: &ButtonInput<KeyCode>) -> Vec2 {
    let mut v = Vec2::ZERO;
    if kb.pressed(KeyCode::KeyW) { v.y += 1.0; }
    if kb.pressed(KeyCode::KeyS) { v.y -= 1.0; }
    if kb.pressed(KeyCode::KeyD) { v.x += 1.0; }
    if kb.pressed(KeyCode::KeyA) { v.x -= 1.0; }
    v
}
