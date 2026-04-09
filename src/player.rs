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
const PLAYER_RADIUS: f32 = 0.3;

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
        let dx = dir.x * speed * dt;
        let dz = dir.z * speed * dt;

        // Try X movement
        let test_x = tf.translation + Vec3::new(dx, 0.0, 0.0);
        if !is_blocked(&world, &active, test_x) {
            tf.translation.x = test_x.x;
        }
        // Try Z movement
        let test_z = tf.translation + Vec3::new(0.0, 0.0, dz);
        if !is_blocked(&world, &active, test_z) {
            tf.translation.z = test_z.z;
        }
    }

    // --- Gravity ---
    vel.0.y -= GRAVITY * dt;
    tf.translation.y += vel.0.y * dt;

    // --- Floor collision ---
    let floor = get_floor(&world, &active, tf.translation);

    if tf.translation.y < floor {
        tf.translation.y = floor;
        vel.0.y = 0.0;
    }

    // --- Ceiling collision (head hitting block above) ---
    let head_y = tf.translation.y + PLAYER_HEIGHT;
    let ceil = get_ceiling(&world, &active, tf.translation);
    if head_y > ceil && vel.0.y > 0.0 {
        tf.translation.y = ceil - PLAYER_HEIGHT;
        vel.0.y = 0.0;
    }

    // --- Jump ---
    let on_ground = (tf.translation.y - floor).abs() < 0.05;
    if keyboard.just_pressed(KeyCode::Space) && on_ground {
        vel.0.y = JUMP_IMPULSE;
    }

    // --- Clamp inside grid bounds when drilled in ---
    if !active.is_top_layer() {
        let margin = 4.0 * MODEL_SIZE as f32;
        let limit = MODEL_SIZE as f32 + margin;
        tf.translation.x = tf.translation.x.clamp(-margin, limit);
        tf.translation.z = tf.translation.z.clamp(-margin, limit);
    }
}

/// Check if the player at `pos` would be inside a solid block (for horizontal collision).
fn is_blocked(world: &VoxelWorld, active: &ActiveLayer, pos: Vec3) -> bool {
    // Check two heights: feet and mid-body
    for check_y in [pos.y + 0.1, pos.y + PLAYER_HEIGHT * 0.5] {
        let check = Vec3::new(pos.x, check_y, pos.z);
        if is_solid_at(world, active, check) {
            return true;
        }
    }
    false
}

/// Is there a solid block at this exact world position?
fn is_solid_at(world: &VoxelWorld, active: &ActiveLayer, pos: Vec3) -> bool {
    if active.is_top_layer() {
        let coord = IVec3::new(pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
        world.cells.contains_key(&coord)
    } else {
        // Inside a grid: check current cell and neighbors
        let sf = MODEL_SIZE as f32;
        let s = MODEL_SIZE as i32;
        let current = active.nav_stack.last().unwrap().cell_coord;

        let pdx = pos.x.div_euclid(sf) as i32;
        let pdy = pos.y.div_euclid(sf) as i32;
        let pdz = pos.z.div_euclid(sf) as i32;

        let lx = (pos.x.rem_euclid(sf).floor() as i32).clamp(0, s - 1) as usize;
        let ly = (pos.y.rem_euclid(sf).floor() as i32).clamp(0, s - 1) as usize;
        let lz = (pos.z.rem_euclid(sf).floor() as i32).clamp(0, s - 1) as usize;

        let grid = if pdx == 0 && pdy == 0 && pdz == 0 {
            world.get_grid(&active.nav_stack)
        } else if active.nav_stack.len() == 1 {
            let nc = current + IVec3::new(pdx, pdy, pdz);
            world.cells.get(&nc)
        } else {
            None
        };

        match grid {
            Some(g) => g.slots[ly][lz][lx].is_solid(),
            None => false,
        }
    }
}

/// Find the floor height at the given position (highest solid surface below feet).
fn get_floor(world: &VoxelWorld, active: &ActiveLayer, pos: Vec3) -> f32 {
    if active.is_top_layer() {
        world::floor_top_layer(&world.cells, pos)
    } else {
        world::floor_inner(world, &active.nav_stack, pos)
    }
}

/// Find the ceiling height at the given position (lowest solid block bottom above head).
fn get_ceiling(world: &VoxelWorld, active: &ActiveLayer, pos: Vec3) -> f32 {
    if active.is_top_layer() {
        // At top layer, ceiling = bottom of the cell above
        let gx = pos.x.floor() as i32;
        let gz = pos.z.floor() as i32;
        let head_cell = (pos.y + PLAYER_HEIGHT).ceil() as i32;
        for cy in head_cell..head_cell + 5 {
            if world.cells.contains_key(&IVec3::new(gx, cy, gz)) {
                return cy as f32;
            }
        }
        f32::INFINITY
    } else {
        // Inside grid: find lowest solid block above player head
        let sf = MODEL_SIZE as f32;
        let s = MODEL_SIZE as i32;
        let current = active.nav_stack.last().unwrap().cell_coord;

        let pdx = pos.x.div_euclid(sf) as i32;
        let pdz = pos.z.div_euclid(sf) as i32;
        let lx = (pos.x.rem_euclid(sf).floor() as i32).clamp(0, s - 1) as usize;
        let lz = (pos.z.rem_euclid(sf).floor() as i32).clamp(0, s - 1) as usize;

        let grid = if pdx == 0 && pdz == 0 {
            world.get_grid(&active.nav_stack)
        } else if active.nav_stack.len() == 1 {
            world.cells.get(&(current + IVec3::new(pdx, 0, pdz)))
        } else {
            None
        };

        let Some(grid) = grid else { return f32::INFINITY; };
        let head_y = pos.y + PLAYER_HEIGHT;
        let start_y = head_y.ceil() as i32;
        let base_y = if pdx == 0 && pdz == 0 { 0.0 } else { 0.0 }; // same base for neighbors at same dy

        for y in 0..MODEL_SIZE {
            if grid.slots[y][lz][lx].is_solid() {
                let block_bottom = base_y + y as f32;
                if block_bottom >= head_y {
                    return block_bottom;
                }
            }
        }
        f32::INFINITY
    }
}

fn wasd(kb: &ButtonInput<KeyCode>) -> Vec2 {
    let mut v = Vec2::ZERO;
    if kb.pressed(KeyCode::KeyW) { v.y += 1.0; }
    if kb.pressed(KeyCode::KeyS) { v.y -= 1.0; }
    if kb.pressed(KeyCode::KeyD) { v.x += 1.0; }
    if kb.pressed(KeyCode::KeyA) { v.x -= 1.0; }
    v
}
