use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::world::terrain::TerrainGenerator;

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const JUMP_IMPULSE: f32 = 8.0;
const GRAVITY: f32 = 20.0;
const PLAYER_HEIGHT: f32 = 1.8;

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

fn spawn_player(mut commands: Commands, terrain: Res<TerrainGenerator>) {
    let spawn_height = terrain.height_at(0.0, 0.0) + PLAYER_HEIGHT;

    commands.spawn((
        Player,
        Velocity(Vec3::ZERO),
        Transform::from_xyz(0.0, spawn_height, 0.0),
        Visibility::Hidden,
    ));
}

fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    terrain: Res<TerrainGenerator>,
    mut player_query: Query<(&mut Transform, &mut Velocity), With<Player>>,
    camera_query: Query<&FpsCam>,
) {
    let Ok((mut tf, mut vel)) = player_query.single_mut() else {
        return;
    };
    let Ok(cam) = camera_query.single() else {
        return;
    };

    let dt = time.delta_secs();
    let ground = terrain.height_at(tf.translation.x, tf.translation.z) + PLAYER_HEIGHT;
    let on_ground = tf.translation.y <= ground + 0.05;

    // Horizontal movement from WASD, relative to camera yaw
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

    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let right = Vec3::new(forward.z, 0.0, -forward.x);

    if input.length_squared() > 0.0 {
        input = input.normalize();
        let speed = if keyboard.pressed(KeyCode::ShiftLeft) {
            SPRINT_SPEED
        } else {
            WALK_SPEED
        };
        let move_dir = forward * input.y + right * input.x;
        tf.translation.x += move_dir.x * speed * dt;
        tf.translation.z += move_dir.z * speed * dt;
    }

    // Jump
    if keyboard.just_pressed(KeyCode::Space) && on_ground {
        vel.0.y = JUMP_IMPULSE;
    }

    // Gravity
    vel.0.y -= GRAVITY * dt;
    tf.translation.y += vel.0.y * dt;

    // Ground collision
    if tf.translation.y <= ground {
        tf.translation.y = ground;
        vel.0.y = 0.0;
    }
}
