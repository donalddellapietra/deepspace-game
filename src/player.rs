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
const EDIT_MARGIN: f32 = 2.0;

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

    let ground = ground_height(&world, tf.translation);
    let floor = ground + PLAYER_HEIGHT;
    let on_ground = tf.translation.y <= floor + 0.1;

    if keyboard.just_pressed(KeyCode::Space) && on_ground {
        vel.0.y = JUMP_IMPULSE;
    }

    vel.0.y -= GRAVITY * dt;
    tf.translation.y += vel.0.y * dt;

    if tf.translation.y <= floor {
        tf.translation.y = floor;
        vel.0.y = 0.0;
    }
}

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

    if keyboard.pressed(KeyCode::Space) { tf.translation.y += speed * dt; }
    if keyboard.pressed(KeyCode::ControlLeft) { tf.translation.y -= speed * dt; }

    // Clamp to cell bounds + margin
    let cell_min = ctx.cell_coord.as_vec3() * CELL_SIZE - Vec3::splat(EDIT_MARGIN);
    let cell_max = (ctx.cell_coord.as_vec3() + Vec3::ONE) * CELL_SIZE + Vec3::splat(EDIT_MARGIN) + Vec3::Y * 3.0;
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

/// Sample actual voxel data to find the highest solid block beneath the player.
fn ground_height(world: &Layer1World, pos: Vec3) -> f32 {
    let cs = CELL_SIZE;
    let s = MODEL_SIZE as i32;

    // Check a few sample points around the player's footprint
    let offsets = [
        Vec2::ZERO,
        Vec2::new(0.3, 0.0), Vec2::new(-0.3, 0.0),
        Vec2::new(0.0, 0.3), Vec2::new(0.0, -0.3),
    ];

    let mut best_y = f32::NEG_INFINITY;

    for offset in &offsets {
        let sx = pos.x + offset.x;
        let sz = pos.z + offset.y;

        let cx = (sx / cs).floor() as i32;
        let cz = (sz / cs).floor() as i32;

        let local_x = ((sx - cx as f32 * cs).floor() as i32).clamp(0, s - 1) as usize;
        let local_z = ((sz - cz as f32 * cs).floor() as i32).clamp(0, s - 1) as usize;

        for cy in -2..10 {
            let coord = IVec3::new(cx, cy, cz);
            let Some(cell_data) = world.cells.get(&coord) else { continue };
            let cell_base_y = cy as f32 * cs;

            for local_y in (0..MODEL_SIZE).rev() {
                if cell_data.blocks[local_y][local_z][local_x].is_some() {
                    let block_top = cell_base_y + local_y as f32 + 1.0;
                    if block_top <= pos.y + 0.5 && block_top > best_y {
                        best_y = block_top;
                    }
                }
            }
        }
    }

    best_y
}
