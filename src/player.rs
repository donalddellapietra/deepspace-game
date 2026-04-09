use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::camera::FpsCam;
use crate::layer::GameLayer;
use crate::world::Layer1World;

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const JUMP_IMPULSE: f32 = 8.0;
const GRAVITY: f32 = 20.0;
const PLAYER_HEIGHT: f32 = 1.7;
const CELL_SIZE: f32 = MODEL_SIZE as f32;

// Editor fly mode
const FLY_SPEED: f32 = 5.0;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, move_world.run_if(in_state(GameLayer::World)))
            .add_systems(Update, move_editor.run_if(in_state(GameLayer::Editing)));
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
        Transform::from_xyz(0.0, 20.0, 0.0),
        Visibility::Hidden,
    ));
}

/// World mode: walk on ground with gravity and jumping.
fn move_world(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    world: Res<Layer1World>,
    mut player_query: Query<(&mut Transform, &mut Velocity), With<Player>>,
    camera_query: Query<&FpsCam>,
) {
    let Ok((mut tf, mut vel)) = player_query.single_mut() else { return };
    let Ok(cam) = camera_query.single() else { return };

    let dt = time.delta_secs();
    let ground = ground_height(&world, tf.translation);
    let on_ground = tf.translation.y <= ground + PLAYER_HEIGHT + 0.05;

    let (forward, right) = cam_directions(cam);
    let input = gather_wasd(&keyboard);

    if input.length_squared() > 0.0 {
        let input = input.normalize();
        let speed = if keyboard.pressed(KeyCode::ShiftLeft) { SPRINT_SPEED } else { WALK_SPEED };
        let move_dir = forward * input.y + right * input.x;
        tf.translation.x += move_dir.x * speed * dt;
        tf.translation.z += move_dir.z * speed * dt;
    }

    if keyboard.just_pressed(KeyCode::Space) && on_ground {
        vel.0.y = JUMP_IMPULSE;
    }

    vel.0.y -= GRAVITY * dt;
    tf.translation.y += vel.0.y * dt;

    let floor = ground + PLAYER_HEIGHT;
    if tf.translation.y <= floor {
        tf.translation.y = floor;
        vel.0.y = 0.0;
    }
}

/// Editor mode: free fly (no gravity).
fn move_editor(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<&mut Transform, With<Player>>,
    camera_query: Query<&FpsCam>,
) {
    let Ok(mut tf) = player_query.single_mut() else { return };
    let Ok(cam) = camera_query.single() else { return };

    let dt = time.delta_secs();
    let (forward, right) = cam_directions(cam);
    let input = gather_wasd(&keyboard);

    let speed = if keyboard.pressed(KeyCode::ShiftLeft) { FLY_SPEED * 2.0 } else { FLY_SPEED };

    if input.length_squared() > 0.0 {
        let input = input.normalize();
        let move_dir = forward * input.y + right * input.x;
        tf.translation += move_dir * speed * dt;
    }

    // Vertical
    if keyboard.pressed(KeyCode::Space) {
        tf.translation.y += speed * dt;
    }
    if keyboard.pressed(KeyCode::ControlLeft) {
        tf.translation.y -= speed * dt;
    }
}

fn cam_directions(cam: &FpsCam) -> (Vec3, Vec3) {
    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let right = Vec3::new(forward.z, 0.0, -forward.x);
    (forward, right)
}

fn gather_wasd(keyboard: &ButtonInput<KeyCode>) -> Vec2 {
    let mut input = Vec2::ZERO;
    if keyboard.pressed(KeyCode::KeyW) { input.y += 1.0; }
    if keyboard.pressed(KeyCode::KeyS) { input.y -= 1.0; }
    if keyboard.pressed(KeyCode::KeyD) { input.x += 1.0; }
    if keyboard.pressed(KeyCode::KeyA) { input.x -= 1.0; }
    input
}

fn ground_height(world: &Layer1World, pos: Vec3) -> f32 {
    let cx = (pos.x / CELL_SIZE).floor() as i32;
    let cz = (pos.z / CELL_SIZE).floor() as i32;

    let mut best_y = f32::NEG_INFINITY;
    for y in -2..20 {
        let coord = IVec3::new(cx, y, cz);
        if world.cells.contains_key(&coord) {
            let cell_top = (y as f32 + 1.0) * CELL_SIZE;
            if cell_top <= pos.y + 0.5 && cell_top > best_y {
                best_y = cell_top;
            }
        }
    }
    best_y
}
