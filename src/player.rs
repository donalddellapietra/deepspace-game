use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::camera::FpsCam;
use crate::layer::{EditingContext, GameLayer};
use crate::world::Layer1World;

const WALK_SPEED: f32 = 8.0;
const SPRINT_SPEED: f32 = 16.0;
const JUMP_IMPULSE: f32 = 8.0;
const GRAVITY: f32 = 20.0;
const PLAYER_HEIGHT: f32 = 1.7;
const CELL_SIZE: f32 = MODEL_SIZE as f32;

const FLY_SPEED: f32 = 5.0;
const EDIT_MARGIN: f32 = 2.0; // how far outside the cell you can go

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

    let (forward, right) = cam_directions(cam);
    let input = gather_wasd(&keyboard);

    if input.length_squared() > 0.0 {
        let input = input.normalize();
        let speed = if keyboard.pressed(KeyCode::ShiftLeft) { SPRINT_SPEED } else { WALK_SPEED };
        let move_dir = forward * input.y + right * input.x;
        tf.translation.x += move_dir.x * speed * dt;
        tf.translation.z += move_dir.z * speed * dt;
    }

    // Ground collision — find highest solid cell top below player
    let ground = ground_height(&world, tf.translation);
    let on_ground = tf.translation.y <= ground + PLAYER_HEIGHT + 0.1;

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

/// Editor mode: fly, clamped to the cell being edited.
fn move_editor(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    context: Option<Res<EditingContext>>,
    mut player_query: Query<&mut Transform, With<Player>>,
    camera_query: Query<&FpsCam>,
) {
    let Ok(mut tf) = player_query.single_mut() else { return };
    let Ok(cam) = camera_query.single() else { return };
    let Some(ctx) = context else { return };

    let dt = time.delta_secs();
    let (forward, right) = cam_directions(cam);
    let input = gather_wasd(&keyboard);

    let speed = if keyboard.pressed(KeyCode::ShiftLeft) { FLY_SPEED * 2.0 } else { FLY_SPEED };

    if input.length_squared() > 0.0 {
        let input = input.normalize();
        let move_dir = forward * input.y + right * input.x;
        tf.translation += move_dir * speed * dt;
    }

    if keyboard.pressed(KeyCode::Space) {
        tf.translation.y += speed * dt;
    }
    if keyboard.pressed(KeyCode::ControlLeft) {
        tf.translation.y -= speed * dt;
    }

    // Clamp to cell bounds + margin
    let cell_min = Vec3::new(
        ctx.cell_coord.x as f32 * CELL_SIZE - EDIT_MARGIN,
        ctx.cell_coord.y as f32 * CELL_SIZE - EDIT_MARGIN,
        ctx.cell_coord.z as f32 * CELL_SIZE - EDIT_MARGIN,
    );
    let cell_max = Vec3::new(
        (ctx.cell_coord.x as f32 + 1.0) * CELL_SIZE + EDIT_MARGIN,
        (ctx.cell_coord.y as f32 + 1.0) * CELL_SIZE + EDIT_MARGIN + 3.0,
        (ctx.cell_coord.z as f32 + 1.0) * CELL_SIZE + EDIT_MARGIN,
    );

    tf.translation = tf.translation.clamp(cell_min, cell_max);
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
            // Top of this cell's content (ground model fills y=0,1,2 so top is at 3 blocks)
            let cell_top = y as f32 * CELL_SIZE + 3.0; // 3 blocks of ground model
            if cell_top <= pos.y + 0.5 && cell_top > best_y {
                best_y = cell_top;
            }
        }
    }
    best_y
}
