use bevy::prelude::*;

use crate::camera::OrbitCamera;
use crate::world::terrain::TerrainGenerator;

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const ROTATION_SMOOTHING: f32 = 10.0;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, move_player);
    }
}

#[derive(Component)]
pub struct Player;

fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    terrain: Res<TerrainGenerator>,
) {
    let spawn_height = terrain.height_at(0.0, 0.0) + 1.0;

    commands.spawn((
        Player,
        Mesh3d(meshes.add(Capsule3d::new(0.4, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.3, 0.2),
            ..default()
        })),
        Transform::from_xyz(0.0, spawn_height, 0.0),
    ));
}

fn move_player(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    terrain: Res<TerrainGenerator>,
    mut player_query: Query<&mut Transform, With<Player>>,
    camera_query: Query<&OrbitCamera>,
) {
    let Ok(mut tf) = player_query.single_mut() else {
        return;
    };
    let Ok(orbit) = camera_query.single() else {
        return;
    };

    // Gather WASD input
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

    if input.length_squared() > 0.0 {
        input = input.normalize();

        // Movement directions derived from camera yaw (projected onto XZ plane)
        let forward = Vec3::new(-orbit.yaw.sin(), 0.0, -orbit.yaw.cos());
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let movement = forward * input.y + right * input.x;

        let speed = if keyboard.pressed(KeyCode::ShiftLeft) {
            SPRINT_SPEED
        } else {
            WALK_SPEED
        };
        tf.translation += movement * speed * time.delta_secs();

        // Smoothly rotate player to face movement direction
        let target_yaw = movement.x.atan2(movement.z);
        let target_rot = Quat::from_rotation_y(target_yaw);
        tf.rotation = tf.rotation.slerp(target_rot, ROTATION_SMOOTHING * time.delta_secs());
    }

    // Stick player to terrain surface
    let ground = terrain.height_at(tf.translation.x, tf.translation.z);
    tf.translation.y = ground + 1.0;
}
