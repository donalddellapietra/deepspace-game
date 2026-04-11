//! Solidity, collision, and Bevy↔`Position` conversion.
//!
//! The world is addressed by `Position` (see
//! `docs/architecture/coordinates.md`), but the player's `Transform`
//! lives in Bevy `Vec3` float space where `1 unit = 1 leaf voxel` and
//! the all-zero-path leaf's min corner is at [`ROOT_ORIGIN`]. The
//! helpers here bridge the two worlds:
//!
//! * [`position_from_bevy`] / [`bevy_from_position`] convert between
//!   the integer-ish `Position` and the float `Vec3`.
//! * [`solid_at_integer`] asks "is the leaf voxel at Bevy integer
//!   coord (x, y, z) solid?".
//! * [`move_and_collide`] does per-axis swept-AABB clipping for the
//!   player.
//! * [`on_ground`] tests whether the player is resting on something
//!   directly beneath their feet.
//!
//! Movement and raycasting don't need to touch `Position` directly —
//! they work in Bevy `Vec3` / `IVec3` space and only go through the
//! tree via `solid_at_integer`.

use bevy::prelude::*;

use super::edit::get_voxel;
use super::position::{LayerPos, Position, NODE_PATH_LEN};
use super::render::{cell_size_at_layer, target_layer_for, ROOT_ORIGIN};
use super::state::{world_extent_voxels, WorldState};
use super::tree::{
    slot_coords, slot_index, voxel_idx, EMPTY_NODE, EMPTY_VOXEL, NODE_VOXELS_PER_AXIS,
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

// ----------------------------------------------------------- conversions

/// Convert a Bevy `Vec3` into a tree `Position`. Returns `None` when
/// the point is outside the world.
///
/// The all-zero `Position` has its min corner at `ROOT_ORIGIN`, so
/// `local = pos - ROOT_ORIGIN` is a point inside the world in
/// leaf-voxel units (1 Bevy unit = 1 leaf voxel).
///
/// At descent step `K` (0-indexed), the current extent starts at
/// `25 * 5^(MAX_LAYER - K)` leaf voxels (the axis size of the node
/// we're inside). The child's extent is one fifth of that — we
/// divide the running remainder by the child extent to find which
/// of the 5 child slots on each axis we're in, then take the
/// remainder for the next step. After `MAX_LAYER` steps the extent
/// is exactly `25` and the remainder is the leaf-local voxel coord
/// in `0..25`. The floored fractional part of `local` is the
/// sub-voxel offset.
pub fn position_from_bevy(pos: Vec3) -> Option<Position> {
    let local = pos - ROOT_ORIGIN;
    if !local.x.is_finite() || !local.y.is_finite() || !local.z.is_finite() {
        return None;
    }
    if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
        return None;
    }

    let max = world_extent_voxels() as f32;
    if local.x >= max || local.y >= max || local.z >= max {
        return None;
    }

    let vx = local.x.floor() as i64;
    let vy = local.y.floor() as i64;
    let vz = local.z.floor() as i64;

    let mut rem: [i64; 3] = [vx, vy, vz];
    let mut path = [0u8; NODE_PATH_LEN];
    // `extent` is the current node's axis size in leaf voxels.
    // Starts at `25 * 5^MAX_LAYER` (root).
    let mut extent: i64 = world_extent_voxels();
    for depth in 0..NODE_PATH_LEN {
        let child_extent = extent / 5;
        // `rem` is `0 <= rem < extent`, so slot is `0..5`.
        let sx = (rem[0] / child_extent) as usize;
        let sy = (rem[1] / child_extent) as usize;
        let sz = (rem[2] / child_extent) as usize;
        debug_assert!(sx < 5 && sy < 5 && sz < 5);
        path[depth] = slot_index(sx, sy, sz) as u8;
        rem[0] -= (sx as i64) * child_extent;
        rem[1] -= (sy as i64) * child_extent;
        rem[2] -= (sz as i64) * child_extent;
        extent = child_extent;
    }
    // `extent == 25` now, and each `rem[i]` is `0..25`.
    debug_assert_eq!(extent, NODE_VOXELS_PER_AXIS as i64);
    let voxel = [rem[0] as u8, rem[1] as u8, rem[2] as u8];
    let offset = [
        local.x - local.x.floor(),
        local.y - local.y.floor(),
        local.z - local.z.floor(),
    ];
    Some(Position { path, voxel, offset })
}

// ----------------------------------------------------- LayerPos helpers

/// Project a Bevy `Vec3` straight to a [`LayerPos`] at view layer
/// `layer`. Equivalent to
/// `LayerPos::from_leaf(position_from_bevy(pos)?, layer)` but skips
/// the intermediate leaf-level conversion when the caller doesn't
/// need it.
pub fn layer_pos_from_bevy(pos: Vec3, layer: u8) -> Option<LayerPos> {
    let leaf = position_from_bevy(pos)?;
    Some(LayerPos::from_leaf(&leaf, layer))
}

/// Bevy-space min corner of the cell at `lp`. Walks `lp.path` once,
/// accumulating slot offsets in `i64` leaf-voxel units (overflow-free
/// for any `MAX_LAYER`), then adds the in-node `cell` scaled by the
/// view layer's cell size.
pub fn bevy_origin_of_layer_pos(lp: &LayerPos) -> Vec3 {
    let mut origin = ROOT_ORIGIN;
    let mut extent: i64 = world_extent_voxels();
    for depth in 0..lp.path.len() {
        let child_extent = extent / 5;
        let (sx, sy, sz) = slot_coords(lp.path[depth] as usize);
        origin.x += (sx as i64 * child_extent) as f32;
        origin.y += (sy as i64 * child_extent) as f32;
        origin.z += (sz as i64 * child_extent) as f32;
        extent = child_extent;
    }
    // After descending `lp.layer` slots, `extent` is the layer-`L`
    // node's axis size in leaf voxels. The node has 25 cells per
    // axis, so one cell = `extent / 25` Bevy units.
    let cell_size = (extent / 25) as f32;
    origin
        + Vec3::new(
            lp.cell[0] as f32 * cell_size,
            lp.cell[1] as f32 * cell_size,
            lp.cell[2] as f32 * cell_size,
        )
}

/// Center of the cell at `lp` in Bevy space.
pub fn bevy_center_of_layer_pos(lp: &LayerPos) -> Vec3 {
    let cell = cell_size_at_layer(lp.layer);
    bevy_origin_of_layer_pos(lp) + Vec3::splat(cell * 0.5)
}

/// Walk the tree from the root to the layer-`lp.layer` node and read
/// the cell `lp.cell` from its `25³` voxel grid. Used by the
/// cell-level raycast.
pub fn is_layer_pos_solid(world: &WorldState, lp: &LayerPos) -> bool {
    let mut id = world.root;
    for &slot in &lp.path {
        let Some(node) = world.library.get(id) else {
            return false;
        };
        let Some(children) = node.children.as_ref() else {
            return false;
        };
        id = children[slot as usize];
        if id == EMPTY_NODE {
            return false;
        }
    }
    let Some(node) = world.library.get(id) else {
        return false;
    };
    let v = node.voxels[voxel_idx(
        lp.cell[0] as usize,
        lp.cell[1] as usize,
        lp.cell[2] as usize,
    )];
    v != EMPTY_VOXEL
}

/// Inverse of [`position_from_bevy`]. Accumulates the origin as we
/// descend, then adds the leaf voxel and sub-voxel offset at the end.
pub fn bevy_from_position(pos: &Position) -> Vec3 {
    let mut origin = ROOT_ORIGIN;
    let mut extent: i64 = world_extent_voxels();
    for depth in 0..NODE_PATH_LEN {
        let child_extent = extent / 5;
        let (sx, sy, sz) = slot_coords(pos.path[depth] as usize);
        origin.x += (sx as i64 * child_extent) as f32;
        origin.y += (sy as i64 * child_extent) as f32;
        origin.z += (sz as i64 * child_extent) as f32;
        extent = child_extent;
    }
    origin
        + Vec3::new(
            pos.voxel[0] as f32 + pos.offset[0],
            pos.voxel[1] as f32 + pos.offset[1],
            pos.voxel[2] as f32 + pos.offset[2],
        )
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

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn round_trip_origin() {
        let p = Position::origin();
        let v = bevy_from_position(&p);
        assert!(approx_eq(v.x, ROOT_ORIGIN.x));
        assert!(approx_eq(v.y, ROOT_ORIGIN.y));
        assert!(approx_eq(v.z, ROOT_ORIGIN.z));
        let back = position_from_bevy(v).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn concrete_leaf_voxel_example() {
        // Position in the all-zero-path leaf with voxel (5, 10, 0)
        // and offset (0.5, 0, 0) should map to
        //   ROOT_ORIGIN + (5.5, 10, 0).
        let mut p = Position::origin();
        p.voxel = [5, 10, 0];
        p.offset = [0.5, 0.0, 0.0];
        let v = bevy_from_position(&p);
        assert!(approx_eq(v.x, ROOT_ORIGIN.x + 5.5), "x={}", v.x);
        assert!(approx_eq(v.y, ROOT_ORIGIN.y + 10.0), "y={}", v.y);
        assert!(approx_eq(v.z, ROOT_ORIGIN.z), "z={}", v.z);
    }

    #[test]
    fn round_trip_various_points() {
        // Points picked inside the grass region (y < 0) and just
        // above the grass (y in [0, 25)). All should survive a
        // Bevy→Position→Bevy round trip.
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(-5.0, -10.5, -11.25),
            Vec3::new(11.999, -0.001, 11.999),
            Vec3::new(0.0, 20.0, 0.0),
            Vec3::new(-7.5, -100.0, 4.25),
        ];
        for &pt in &points {
            let p = position_from_bevy(pt).expect("in-range");
            let back = bevy_from_position(&p);
            assert!(approx_eq(back.x, pt.x), "x: {} vs {}", back.x, pt.x);
            assert!(approx_eq(back.y, pt.y), "y: {} vs {}", back.y, pt.y);
            assert!(approx_eq(back.z, pt.z), "z: {} vs {}", back.z, pt.z);
        }
    }

    #[test]
    fn position_from_bevy_rejects_below_root() {
        // ROOT_ORIGIN.y = -GROUND_Y_VOXELS (= -125). Anything strictly
        // below that is outside the world.
        assert!(
            position_from_bevy(Vec3::new(0.0, ROOT_ORIGIN.y - 0.5, 0.0)).is_none()
        );
    }

    #[test]
    fn position_from_bevy_boundary_at_ground_surface() {
        // Exactly at y = 0 (the top face of the grass surface):
        // local.y = GROUND_Y_VOXELS, which is the boundary between
        // the top grass leaf and the first air leaf above it. The
        // converter should land in the air leaf with voxel.y = 0
        // and offset.y = 0.
        let p = position_from_bevy(Vec3::new(0.0, 0.0, 0.0)).unwrap();
        assert_eq!(p.voxel[1], 0);
        assert!(p.offset[1].abs() < 1e-6);
        let v = bevy_from_position(&p);
        assert!(approx_eq(v.y, 0.0));
    }

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

    /// The grassland's ground surface must be reachable as a solid
    /// [`LayerPos`] at every view layer. Regression against a
    /// majority-vote downsample that washed out thin features — at
    /// view L ≤ 8 the 125-leaf-deep ground used to collapse to air
    /// in the cascaded downsample, so the crosshair clicked through
    /// ground it could see. The presence-preserving downsample in
    /// `tree::downsample` is what makes this hold.
    #[test]
    fn ground_is_solid_at_every_view_layer() {
        use super::super::position::LayerPos;
        use super::super::render::{target_layer_for, MIN_ZOOM, MAX_ZOOM};
        let world = WorldState::new_grassland();
        // A point guaranteed to sit inside the grass region: one
        // voxel below the surface at the world's all-zero-path.
        let probe = Vec3::new(0.0, -1.0, 0.0);
        for view_layer in MIN_ZOOM..=MAX_ZOOM {
            let target = target_layer_for(view_layer);
            let lp_view = layer_pos_from_bevy(probe, view_layer)
                .expect("probe inside world at view layer");
            let lp_target = layer_pos_from_bevy(probe, target)
                .expect("probe inside world at target layer");
            assert!(
                is_layer_pos_solid(&world, &lp_view),
                "view layer {view_layer}: ground reads as air (downsample loss)"
            );
            assert!(
                is_layer_pos_solid(&world, &lp_target),
                "target layer {target} (from view {view_layer}): ground reads as air"
            );
            // Silence an unused-import warning on some configurations.
            let _: LayerPos = lp_view;
            let _: LayerPos = lp_target;
        }
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
