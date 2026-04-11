//! Swept-AABB collision for the player.
//!
//! The player's `Transform` lives in Bevy `Vec3` float space. This
//! module turns "I want to move the player by (dx, dy, dz)" into a
//! per-axis clip against the world's block grid at the current view
//! layer:
//!
//! * [`solid_at_integer`] asks "is the leaf voxel at Bevy integer
//!   coord (x, y, z) solid?" — a legacy leaf-only probe kept for the
//!   remaining leaf-level tests.
//! * [`move_and_collide`] does per-axis swept-AABB clipping for the
//!   player, sampling blocks at
//!   [`target_layer_for`](super::view::target_layer_for) so the
//!   collision lattice matches what the renderer draws.
//! * [`on_ground`] tests whether the player is resting on something
//!   directly beneath their feet.
//!
//! The Bevy ↔ tree coordinate helpers ([`layer_pos_from_bevy`],
//! [`is_layer_pos_solid`], [`position_from_bevy`], …) used to live
//! here; they now live in [`super::view`] and this module imports
//! them.

use bevy::prelude::*;

use super::edit::get_voxel;
use super::position::Position;
use super::state::WorldState;
use super::tree::EMPTY_VOXEL;
use super::view::{
    bevy_from_position, cell_size_at_layer, is_layer_pos_solid, layer_pos_from_bevy,
    position_from_bevy, target_layer_for, ROOT_ORIGIN,
};

// ------------------------------------------------------------ player AABB

/// Half-width on X and Z of the player's AABB, in leaf voxels.
pub const PLAYER_HW: f32 = 0.3;
/// Total height of the player's AABB, in leaf voxels.
pub const PLAYER_H: f32 = 1.7;

// --------------------------------------------------------------- solidity

/// Is the leaf voxel at `position` non-empty?
pub fn is_voxel_solid(world: &WorldState, position: &Position) -> bool {
    get_voxel(world, position) != EMPTY_VOXEL
}

/// Is the leaf voxel at the Bevy integer coordinate `coord` solid?
///
/// `coord` names the unit-cube `[coord, coord+1]` in Bevy space.
/// Anything outside the materialised root (`Position` conversion
/// fails) is treated as empty space — the player can fall into the
/// void, but physics doesn't panic.
pub fn solid_at_integer(world: &WorldState, coord: IVec3) -> bool {
    let center = Vec3::new(
        coord.x as f32 + 0.5,
        coord.y as f32 + 0.5,
        coord.z as f32 + 0.5,
    );
    match position_from_bevy(center) {
        Some(p) => is_voxel_solid(world, &p),
        None => false,
    }
}

// ----------------------------------------------------------- collision
//
// Player AABB scales with the view layer's cell size: at view L the
// player is `PLAYER_HW` cells wide and `PLAYER_H` cells tall (= the
// camera's eye height). Collision blocks are sampled at `target_layer
// = (L + 2).min(MAX_LAYER)` — the same layer the renderer reads from
// — so the visible mesh and the collision lattice always agree, at
// every zoom level.
//
// At L = 10/11/12, target = MAX_LAYER (the leaves), block_size = 1
// Bevy unit, and this code is functionally identical to the
// previous leaf-voxel collision. At lower L the block grid grows
// (`5^(MAX_LAYER - target)` Bevy units per side) and the player AABB
// grows in lockstep, so the per-frame cell-iteration count stays
// roughly constant (~`PLAYER_H * 25` cells in y, similar in x/z).

#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

fn player_aabb(pos: Vec3, view_cell: f32) -> Aabb {
    let hw = PLAYER_HW * view_cell;
    let h = PLAYER_H * view_cell;
    Aabb {
        min: Vec3::new(pos.x - hw, pos.y, pos.z - hw),
        max: Vec3::new(pos.x + hw, pos.y + h, pos.z + hw),
    }
}

/// Clip a one-axis movement `delta` against a single block of side
/// `block_size`, anchored to the cell lattice rooted at `cell_origin`.
fn clip_axis(
    player: &Aabb,
    delta: f32,
    axis: usize,
    bx: i32,
    by: i32,
    bz: i32,
    block_size: f32,
    cell_origin: Vec3,
) -> f32 {
    let (a1, a2) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let b_min = [
        cell_origin.x + bx as f32 * block_size,
        cell_origin.y + by as f32 * block_size,
        cell_origin.z + bz as f32 * block_size,
    ];
    let b_max = [
        b_min[0] + block_size,
        b_min[1] + block_size,
        b_min[2] + block_size,
    ];

    if player.max[a1] <= b_min[a1] || player.min[a1] >= b_max[a1] {
        return delta;
    }
    if player.max[a2] <= b_min[a2] || player.min[a2] >= b_max[a2] {
        return delta;
    }

    if delta < 0.0 {
        let face = b_max[axis];
        let gap = face - player.min[axis];
        if gap <= 0.0 && gap > delta {
            return gap;
        }
    } else if delta > 0.0 {
        let face = b_min[axis];
        let gap = face - player.max[axis];
        if gap >= 0.0 && gap < delta {
            return gap;
        }
    }
    delta
}

/// Sample the layer-`target_layer` cell at integer coord `(bx, by, bz)`
/// (relative to `cell_origin` in `block_size` strides).
fn is_target_block_solid(
    world: &WorldState,
    target_layer: u8,
    bx: i32,
    by: i32,
    bz: i32,
    block_size: f32,
    cell_origin: Vec3,
) -> bool {
    let center = cell_origin
        + Vec3::new(
            (bx as f32 + 0.5) * block_size,
            (by as f32 + 0.5) * block_size,
            (bz as f32 + 0.5) * block_size,
        );
    match layer_pos_from_bevy(center, target_layer) {
        Some(lp) => is_layer_pos_solid(world, &lp),
        None => false,
    }
}

/// Convert an AABB into the half-open cell range that overlaps it,
/// expressed in cell-units relative to `cell_origin` and `block_size`.
fn aabb_block_range(aabb: Aabb, block_size: f32, cell_origin: Vec3) -> (IVec3, IVec3) {
    (
        IVec3::new(
            ((aabb.min.x - cell_origin.x) / block_size).floor() as i32,
            ((aabb.min.y - cell_origin.y) / block_size).floor() as i32,
            ((aabb.min.z - cell_origin.z) / block_size).floor() as i32,
        ),
        IVec3::new(
            ((aabb.max.x - cell_origin.x - 1e-5) / block_size).floor() as i32,
            ((aabb.max.y - cell_origin.y - 1e-5) / block_size).floor() as i32,
            ((aabb.max.z - cell_origin.z - 1e-5) / block_size).floor() as i32,
        ),
    )
}

/// Resolve player movement with per-axis clipping. The player AABB
/// scales with the view layer's cell size, and the collision blocks
/// are sampled at the same layer the renderer reads from
/// (`target = (view_layer + 2).min(MAX_LAYER)`).
pub fn move_and_collide(
    pos: &mut Vec3,
    vel: &mut Vec3,
    horizontal_delta: Vec2,
    dt: f32,
    world: &WorldState,
    view_layer: u8,
) {
    let view_cell = cell_size_at_layer(view_layer);
    let target_layer = target_layer_for(view_layer);
    let block_size = cell_size_at_layer(target_layer);
    let cell_origin = ROOT_ORIGIN;

    let mut dy = vel.y * dt;
    let dx = horizontal_delta.x;
    let dz = horizontal_delta.y;

    // Expanded AABB covers the player's range plus the move delta
    // plus one block's margin on each side.
    let player = player_aabb(*pos, view_cell);
    let expanded = Aabb {
        min: Vec3::new(
            player.min.x + dx.min(0.0) - block_size,
            player.min.y + dy.min(0.0) - block_size,
            player.min.z + dz.min(0.0) - block_size,
        ),
        max: Vec3::new(
            player.max.x + dx.max(0.0) + block_size,
            player.max.y + dy.max(0.0) + block_size,
            player.max.z + dz.max(0.0) + block_size,
        ),
    };
    let (bmin, bmax) = aabb_block_range(expanded, block_size, cell_origin);

    let mut blocks: Vec<(i32, i32, i32)> = Vec::new();
    for by in bmin.y..=bmax.y {
        for bz in bmin.z..=bmax.z {
            for bx in bmin.x..=bmax.x {
                if is_target_block_solid(
                    world,
                    target_layer,
                    bx,
                    by,
                    bz,
                    block_size,
                    cell_origin,
                ) {
                    blocks.push((bx, by, bz));
                }
            }
        }
    }

    // --- Y first ---
    let pa = player_aabb(*pos, view_cell);
    let original_dy = dy;
    for &(bx, by, bz) in &blocks {
        dy = clip_axis(&pa, dy, 1, bx, by, bz, block_size, cell_origin);
    }
    pos.y += dy;
    if (dy - original_dy).abs() > 1e-6 {
        vel.y = 0.0;
    }

    // --- X ---
    let pa = player_aabb(*pos, view_cell);
    let mut clipped_dx = dx;
    for &(bx, by, bz) in &blocks {
        clipped_dx = clip_axis(&pa, clipped_dx, 0, bx, by, bz, block_size, cell_origin);
    }
    pos.x += clipped_dx;

    // --- Z ---
    let pa = player_aabb(*pos, view_cell);
    let mut clipped_dz = dz;
    for &(bx, by, bz) in &blocks {
        clipped_dz = clip_axis(&pa, clipped_dz, 2, bx, by, bz, block_size, cell_origin);
    }
    pos.z += clipped_dz;
}

/// Is the player standing on something? Tests a tiny downward nudge
/// (one tenth of a block at the current layer) against all nearby
/// blocks at `target_layer`. If the nudge gets clipped ~to zero we
/// were on the floor.
pub fn on_ground(pos: Vec3, world: &WorldState, view_layer: u8) -> bool {
    let view_cell = cell_size_at_layer(view_layer);
    let target_layer = target_layer_for(view_layer);
    let block_size = cell_size_at_layer(target_layer);
    let cell_origin = ROOT_ORIGIN;

    let player = player_aabb(pos, view_cell);
    let probe = Aabb {
        min: Vec3::new(player.min.x, player.min.y - 0.1 * block_size, player.min.z),
        max: player.max,
    };
    let (bmin, bmax) = aabb_block_range(probe, block_size, cell_origin);
    let mut test_dy = -0.05 * block_size;
    for by in bmin.y..=bmax.y {
        for bz in bmin.z..=bmax.z {
            for bx in bmin.x..=bmax.x {
                if is_target_block_solid(
                    world,
                    target_layer,
                    bx,
                    by,
                    bz,
                    block_size,
                    cell_origin,
                ) {
                    test_dy = clip_axis(
                        &player,
                        test_dy,
                        1,
                        bx,
                        by,
                        bz,
                        block_size,
                        cell_origin,
                    );
                }
            }
        }
    }
    test_dy.abs() < 0.04 * block_size
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_at_integer_grassland_below_surface_is_solid() {
        let world = WorldState::new_grassland();
        // Below ground (Bevy y < 0) — the grass region.
        assert!(solid_at_integer(&world, IVec3::new(0, -1, 0)));
        assert!(solid_at_integer(&world, IVec3::new(5, -10, -3)));
        assert!(solid_at_integer(&world, IVec3::new(0, -50, 0)));
        // The deepest solid voxel sits at Bevy y = ROOT_ORIGIN.y,
        // which is -GROUND_Y_VOXELS.
        assert!(solid_at_integer(
            &world,
            IVec3::new(0, ROOT_ORIGIN.y as i32, 0)
        ));
    }

    #[test]
    fn solid_at_integer_grassland_above_surface_is_empty() {
        let world = WorldState::new_grassland();
        // Above ground (Bevy y >= 0) — all air.
        assert!(!solid_at_integer(&world, IVec3::new(0, 0, 0)));
        assert!(!solid_at_integer(&world, IVec3::new(0, 5, 0)));
        assert!(!solid_at_integer(&world, IVec3::new(100, 100, 100)));
    }

    #[test]
    fn solid_at_integer_outside_root_is_empty() {
        let world = WorldState::new_grassland();
        // Strictly below ROOT_ORIGIN.y is outside the materialised
        // world and should be air, not solid.
        assert!(!solid_at_integer(
            &world,
            IVec3::new(0, ROOT_ORIGIN.y as i32 - 5, 0)
        ));
        // Left of ROOT_ORIGIN.x.
        assert!(!solid_at_integer(
            &world,
            IVec3::new(ROOT_ORIGIN.x as i32 - 5, -1, 0)
        ));
    }

    #[test]
    fn on_ground_just_above_surface() {
        use super::super::tree::MAX_LAYER;
        let world = WorldState::new_grassland();
        // Feet at y = 0.001 are a hair above the grass top face.
        // Use the leaf view layer so block_size = 1 (= the legacy
        // collision behaviour).
        assert!(on_ground(Vec3::new(0.0, 0.001, 0.0), &world, MAX_LAYER));
        // Feet a metre up — still walking, not on ground.
        assert!(!on_ground(Vec3::new(0.0, 2.0, 0.0), &world, MAX_LAYER));
        // Feet out in the void, way below ROOT_ORIGIN.
        assert!(!on_ground(
            Vec3::new(0.0, ROOT_ORIGIN.y - 50.0, 0.0),
            &world,
            MAX_LAYER,
        ));
    }

}
