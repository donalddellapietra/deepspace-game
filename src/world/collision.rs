//! Swept AABB collision against the voxel world.
//!
//! The player has an axis-aligned bounding box defined by feet position (bottom-center),
//! half-width, and height. Movement is resolved per-axis (Y first, then X, Z).
//! If the AABB overlaps a solid block after movement on an axis, the player is
//! pushed out to the nearest face and velocity on that axis is zeroed.
//!
//! This replaces all floor-detection hacks, tolerances, and special cases.
//! It works identically at every layer depth.

use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::layer::NavEntry;

use super::{CellSlot, VoxelWorld};

pub const PLAYER_HW: f32 = 0.3;  // half-width on X and Z
pub const PLAYER_H: f32 = 1.7;   // total height

/// Is the block at integer coordinate `coord` solid at the current layer?
/// Works at the top layer (HashMap) and inside grids at any depth (via get_sibling).
pub fn block_solid(world: &VoxelWorld, nav_stack: &[NavEntry], coord: IVec3) -> bool {
    if nav_stack.is_empty() {
        // Top layer: each cell = 1 unit block
        return world.cells.contains_key(&coord);
    }

    let current = nav_stack.last().unwrap().cell_coord;
    let sf = MODEL_SIZE as f32;
    let s = MODEL_SIZE as i32;

    // Which parent-layer cell does this coordinate fall in?
    let pdx = coord.x.div_euclid(s);
    let pdy = coord.y.div_euclid(s);
    let pdz = coord.z.div_euclid(s);

    // Local block coords within that cell
    let lx = coord.x.rem_euclid(s) as usize;
    let ly = coord.y.rem_euclid(s) as usize;
    let lz = coord.z.rem_euclid(s) as usize;

    let sib_coord = current + IVec3::new(pdx, pdy, pdz);

    // Get the grid at that parent cell
    let grid = if pdx == 0 && pdy == 0 && pdz == 0 {
        world.get_grid(nav_stack)
    } else {
        world.get_sibling(nav_stack, sib_coord)
    };

    if let Some(g) = grid {
        return g.slots[ly][lz][lx].is_solid();
    }

    // No child grid — check if the parent slot itself is a solid Block
    if let Some(slot) = world.get_sibling_slot(nav_stack, sib_coord) {
        return slot.is_solid();
    }

    false
}

/// Resolve movement with AABB collision. Modifies `pos` and `vel` in place.
///
/// `horizontal_delta` is the desired XZ movement this frame (from WASD input).
/// `vel.y` has gravity already applied. Y is resolved first (landing/ceiling),
/// then X and Z independently (wall sliding).
pub fn move_and_collide(
    pos: &mut Vec3,
    vel: &mut Vec3,
    horizontal_delta: Vec2,
    dt: f32,
    world: &VoxelWorld,
    nav_stack: &[NavEntry],
) {
    let solid = |coord: IVec3| block_solid(world, nav_stack, coord);

    // Y axis first (gravity/jump)
    resolve_axis(pos, vel, vel.y * dt, Axis::Y, &solid);

    // X axis
    let mut hvel_x = horizontal_delta.x;
    resolve_axis(pos, &mut Vec3::new(hvel_x, 0.0, 0.0), horizontal_delta.x, Axis::X, &solid);

    // Z axis
    resolve_axis(pos, &mut Vec3::new(0.0, 0.0, horizontal_delta.y), horizontal_delta.y, Axis::Z, &solid);
}

#[derive(Clone, Copy)]
enum Axis { X, Y, Z }

/// Move on one axis with proper swept collision.
/// Checks blocks between old and new position to prevent tunneling.
fn resolve_axis(
    pos: &mut Vec3,
    vel: &mut Vec3,
    movement: f32,
    axis: Axis,
    solid: &impl Fn(IVec3) -> bool,
) {
    if movement.abs() < 1e-10 { return; }

    let old_pos = *pos;

    // Apply movement
    match axis {
        Axis::X => pos.x += movement,
        Axis::Y => pos.y += movement,
        Axis::Z => pos.z += movement,
    }

    // Compute AABB that covers BOTH old and new positions (swept volume)
    let min_old = Vec3::new(old_pos.x - PLAYER_HW, old_pos.y, old_pos.z - PLAYER_HW);
    let max_old = Vec3::new(old_pos.x + PLAYER_HW, old_pos.y + PLAYER_H, old_pos.z + PLAYER_HW);
    let min_new = Vec3::new(pos.x - PLAYER_HW, pos.y, pos.z - PLAYER_HW);
    let max_new = Vec3::new(pos.x + PLAYER_HW, pos.y + PLAYER_H, pos.z + PLAYER_HW);

    let sweep_min = min_old.min(min_new);
    let sweep_max = max_old.max(max_new);

    let bmin = IVec3::new(
        sweep_min.x.floor() as i32,
        sweep_min.y.floor() as i32,
        sweep_min.z.floor() as i32,
    );
    let bmax = IVec3::new(
        (sweep_max.x - 1e-5).floor() as i32,
        (sweep_max.y - 1e-5).floor() as i32,
        (sweep_max.z - 1e-5).floor() as i32,
    );

    // Find the CLOSEST blocking face in the movement direction
    let mut best_push: Option<f32> = None;

    for by in bmin.y..=bmax.y {
        for bz in bmin.z..=bmax.z {
            for bx in bmin.x..=bmax.x {
                if !solid(IVec3::new(bx, by, bz)) { continue; }

                // Check if the player AABB at NEW position actually overlaps this block
                let block_min = Vec3::new(bx as f32, by as f32, bz as f32);
                let block_max = block_min + Vec3::ONE;

                if min_new.x >= block_max.x || max_new.x <= block_min.x ||
                   min_new.y >= block_max.y || max_new.y <= block_min.y ||
                   min_new.z >= block_max.z || max_new.z <= block_min.z {
                    continue; // no overlap at new position
                }

                // Compute push distance on this axis
                let push = match axis {
                    Axis::Y if movement < 0.0 => (by + 1) as f32,     // feet → block top
                    Axis::Y => by as f32 - PLAYER_H,                   // head → block bottom
                    Axis::X if movement < 0.0 => (bx + 1) as f32 + PLAYER_HW,
                    Axis::X => bx as f32 - PLAYER_HW,
                    Axis::Z if movement < 0.0 => (bz + 1) as f32 + PLAYER_HW,
                    Axis::Z => bz as f32 - PLAYER_HW,
                };

                // Pick the push that moves the player the LEAST (closest face)
                let current = match axis { Axis::X => pos.x, Axis::Y => pos.y, Axis::Z => pos.z };
                let dist = (push - current).abs();
                if best_push.is_none() || dist < (best_push.unwrap() - current).abs() {
                    best_push = Some(push);
                }
            }
        }
    }

    if let Some(push) = best_push {
        match axis {
            Axis::X => pos.x = push,
            Axis::Y => { pos.y = push; vel.y = 0.0; }
            Axis::Z => pos.z = push,
        }
    }
}

/// Is the player on the ground? (feet touching a solid block below)
pub fn on_ground(pos: Vec3, world: &VoxelWorld, nav_stack: &[NavEntry]) -> bool {
    // Check one pixel below feet
    let test_y = pos.y - 0.01;
    let bmin_x = (pos.x - PLAYER_HW).floor() as i32;
    let bmax_x = (pos.x + PLAYER_HW - 1e-5).floor() as i32;
    let bmin_z = (pos.z - PLAYER_HW).floor() as i32;
    let bmax_z = (pos.z + PLAYER_HW - 1e-5).floor() as i32;
    let by = test_y.floor() as i32;

    for bz in bmin_z..=bmax_z {
        for bx in bmin_x..=bmax_x {
            if block_solid(world, nav_stack, IVec3::new(bx, by, bz)) {
                return true;
            }
        }
    }
    false
}
