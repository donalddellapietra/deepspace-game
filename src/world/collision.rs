//! AABB clipping collision against the voxel world.
//!
//! Standard Minecraft/Quake algorithm: for each axis, find the maximum safe
//! movement distance by clipping against all nearby solid blocks. The player
//! never penetrates a block — movement is clamped BEFORE it's applied.
//!
//! Works identically at every layer depth.

use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::layer::NavEntry;

use super::{CellSlot, VoxelWorld};

pub const PLAYER_HW: f32 = 0.3;
pub const PLAYER_H: f32 = 1.7;

/// Is the block at integer coordinate `coord` solid?
///
/// Walks the ENTIRE ancestor chain. If `coord` is outside the current grid,
/// it transforms to parent coordinates. If outside the parent, transforms to
/// grandparent, and so on up to the top-layer HashMap.
///
/// This means collision works correctly no matter how far the player moves
/// from the current cell — ground from any ancestor layer is detected.
pub fn block_solid(world: &VoxelWorld, nav_stack: &[NavEntry], coord: IVec3) -> bool {
    if nav_stack.is_empty() {
        return world.cells.contains_key(&coord);
    }

    let s = MODEL_SIZE as i32;

    // Transform coord through each level of the nav stack from deepest to shallowest.
    // At each level, if the coordinate maps to the current cell (offset 0,0,0),
    // check the grid. If it maps to a sibling, check the sibling. If it maps
    // outside the grid entirely, go up one more level.
    let mut check_coord = coord;

    for depth in (0..nav_stack.len()).rev() {
        let cell_coord = nav_stack[depth].cell_coord;

        // What's the offset from the current cell at this level?
        let pdx = check_coord.x.div_euclid(s);
        let pdy = check_coord.y.div_euclid(s);
        let pdz = check_coord.z.div_euclid(s);

        let lx = check_coord.x.rem_euclid(s) as usize;
        let ly = check_coord.y.rem_euclid(s) as usize;
        let lz = check_coord.z.rem_euclid(s) as usize;

        let abs_coord = cell_coord + IVec3::new(pdx, pdy, pdz);

        if depth == 0 {
            // This level's parent is the top-layer HashMap.
            // abs_coord is a top-layer cell coordinate.
            if let Some(grid) = world.cells.get(&abs_coord) {
                return grid.slots[ly][lz][lx].is_solid();
            }
            // No cell at this position in the top layer
            return false;
        }

        // This level's parent is the grid at nav_stack[..depth].
        // Check if abs_coord is within the parent grid bounds (0..MODEL_SIZE).
        if abs_coord.x >= 0 && abs_coord.x < s
            && abs_coord.y >= 0 && abs_coord.y < s
            && abs_coord.z >= 0 && abs_coord.z < s
        {
            // In bounds — check the parent grid's slot
            let parent_nav = &nav_stack[..depth];
            if let Some(parent_grid) = world.get_grid(parent_nav) {
                let slot = &parent_grid.slots[abs_coord.y as usize][abs_coord.z as usize][abs_coord.x as usize];
                match slot {
                    CellSlot::Child(child) => {
                        return child.slots[ly][lz][lx].is_solid();
                    }
                    CellSlot::Block(_) => return true, // solid parent block fills everything
                    CellSlot::Empty => return false,
                }
            }
            return false;
        }

        // Out of bounds at this level — transform to the next level up.
        // Convert check_coord to the parent's coordinate system:
        // parent_coord = cell_coord + offset, then that becomes the new check_coord
        check_coord = abs_coord;
    }

    // Should not reach here (depth=0 case handles the top layer)
    false
}

/// The player's axis-aligned bounding box, defined by feet position.
/// min = (x - hw, y, z - hw), max = (x + hw, y + h, z + hw)
#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn from_feet(pos: Vec3) -> Self {
        Self {
            min: Vec3::new(pos.x - PLAYER_HW, pos.y, pos.z - PLAYER_HW),
            max: Vec3::new(pos.x + PLAYER_HW, pos.y + PLAYER_H, pos.z + PLAYER_HW),
        }
    }

    /// Expand the AABB by `delta` on one axis to cover the swept path.
    fn expanded(&self, axis: usize, delta: f32) -> Self {
        let mut r = *self;
        if delta > 0.0 {
            r.max[axis] += delta;
        } else {
            r.min[axis] += delta;
        }
        r
    }

    /// Does this AABB overlap a unit block at integer position (bx, by, bz)?
    fn overlaps_block(&self, bx: i32, by: i32, bz: i32) -> bool {
        self.min.x < (bx + 1) as f32 && self.max.x > bx as f32 &&
        self.min.y < (by + 1) as f32 && self.max.y > by as f32 &&
        self.min.z < (bz + 1) as f32 && self.max.z > bz as f32
    }

    /// Block range that this AABB could overlap.
    fn block_range(&self) -> (IVec3, IVec3) {
        (
            IVec3::new(self.min.x.floor() as i32, self.min.y.floor() as i32, self.min.z.floor() as i32),
            IVec3::new(
                (self.max.x - 1e-5).floor() as i32,
                (self.max.y - 1e-5).floor() as i32,
                (self.max.z - 1e-5).floor() as i32,
            ),
        )
    }
}

/// Clip a movement delta on one axis against a single block AABB.
/// Returns the clipped delta (smaller magnitude if blocked).
fn clip_axis(player: &Aabb, delta: f32, axis: usize, bx: i32, by: i32, bz: i32) -> f32 {
    // Only clip if the player overlaps the block on the OTHER two axes.
    let (a1, a2) = match axis { 0 => (1, 2), 1 => (0, 2), _ => (0, 1) };

    let b_min = [bx as f32, by as f32, bz as f32];
    let b_max = [(bx + 1) as f32, (by + 1) as f32, (bz + 1) as f32];

    // Check overlap on the two non-movement axes
    if player.max[a1] <= b_min[a1] || player.min[a1] >= b_max[a1] { return delta; }
    if player.max[a2] <= b_min[a2] || player.min[a2] >= b_max[a2] { return delta; }

    if delta < 0.0 {
        // Moving in negative direction. Block's max face could stop us.
        let face = b_max[axis]; // e.g., by + 1 for Y axis
        let gap = face - player.min[axis]; // distance from player min to block max
        if gap <= 0.0 && gap > delta {
            return gap; // clip: can only move this far
        }
    } else if delta > 0.0 {
        // Moving in positive direction. Block's min face could stop us.
        let face = b_min[axis]; // e.g., by for Y axis
        let gap = face - player.max[axis]; // distance from player max to block min
        if gap >= 0.0 && gap < delta {
            return gap;
        }
    }

    delta
}

/// Move the player with proper AABB clipping collision.
/// Resolves Y first (gravity), then X, then Z.
pub fn move_and_collide(
    pos: &mut Vec3,
    vel: &mut Vec3,
    horizontal_delta: Vec2,
    dt: f32,
    world: &VoxelWorld,
    nav_stack: &[NavEntry],
) {
    let solid = |coord: IVec3| block_solid(world, nav_stack, coord);

    let mut dy = vel.y * dt;
    let dx = horizontal_delta.x;
    let dz = horizontal_delta.y;

    // Collect all solid blocks near the player's swept path.
    // Expand the AABB to cover the full movement in all axes.
    let player = Aabb::from_feet(*pos);
    let expanded = Aabb {
        min: Vec3::new(
            player.min.x + dx.min(0.0) - 1.0,
            player.min.y + dy.min(0.0) - 1.0,
            player.min.z + dz.min(0.0) - 1.0,
        ),
        max: Vec3::new(
            player.max.x + dx.max(0.0) + 1.0,
            player.max.y + dy.max(0.0) + 1.0,
            player.max.z + dz.max(0.0) + 1.0,
        ),
    };

    let (bmin, bmax) = expanded.block_range();

    // Collect solid blocks in range
    let mut blocks: Vec<(i32, i32, i32)> = Vec::new();
    for by in bmin.y..=bmax.y {
        for bz in bmin.z..=bmax.z {
            for bx in bmin.x..=bmax.x {
                if solid(IVec3::new(bx, by, bz)) {
                    blocks.push((bx, by, bz));
                }
            }
        }
    }

    // --- Y axis first ---
    let mut player_aabb = Aabb::from_feet(*pos);
    for &(bx, by, bz) in &blocks {
        dy = clip_axis(&player_aabb, dy, 1, bx, by, bz);
    }
    pos.y += dy;
    if (dy - vel.y * dt).abs() > 1e-6 {
        vel.y = 0.0; // hit floor or ceiling
    }

    // --- X axis ---
    player_aabb = Aabb::from_feet(*pos);
    let mut clipped_dx = dx;
    for &(bx, by, bz) in &blocks {
        clipped_dx = clip_axis(&player_aabb, clipped_dx, 0, bx, by, bz);
    }
    pos.x += clipped_dx;

    // --- Z axis ---
    player_aabb = Aabb::from_feet(*pos);
    let mut clipped_dz = dz;
    for &(bx, by, bz) in &blocks {
        clipped_dz = clip_axis(&player_aabb, clipped_dz, 2, bx, by, bz);
    }
    pos.z += clipped_dz;
}

/// Is the player on the ground? (feet touching a solid block directly below)
pub fn on_ground(pos: Vec3, world: &VoxelWorld, nav_stack: &[NavEntry]) -> bool {
    let player = Aabb::from_feet(pos);
    // Try to move 0.01 downward — if it gets clipped to 0, we're on ground
    let mut test_dy = -0.05f32;
    let (bmin, bmax) = Aabb::from_feet(pos).expanded(1, -0.1).block_range();
    for by in bmin.y..=bmax.y {
        for bz in bmin.z..=bmax.z {
            for bx in bmin.x..=bmax.x {
                if block_solid(world, nav_stack, IVec3::new(bx, by, bz)) {
                    test_dy = clip_axis(&player, test_dy, 1, bx, by, bz);
                }
            }
        }
    }
    test_dy.abs() < 0.04 // movement was clipped → ground is right below
}
