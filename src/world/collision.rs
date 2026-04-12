//! Swept-AABB collision that operates on [`Position`] directly.
//!
//! The public API takes `&(mut) Position` and never exposes a
//! long-range `Vec3` — no caller has to worry about `f32` precision
//! regardless of where in the 6-billion-leaf world the entity sits.
//! Internally, each function builds a **local** [`WorldAnchor`]
//! equal to the entity's own integer leaf coord, runs the existing
//! AABB algorithm on a small `Vec3` scratch (the entity's sub-voxel
//! offset is the starting point; deltas of at most a view cell are
//! added each frame), and then folds the updated `Vec3` back into
//! the `Position` via an exact `i64` leaf addition. The `Vec3`
//! never accumulates a large value and the `Position` is the
//! single source of truth.
//!
//! * [`move_and_collide`] runs per-axis swept clipping against the
//!   target-layer block grid and updates `pos` in place.
//! * [`on_ground`] tests whether the entity is resting on something
//!   at the current view layer.
//! * [`snap_to_ground`] re-places the entity on top of the apparent
//!   ground after a zoom change (presence-preserving downsample can
//!   inflate thin features vertically, so the collision grid grows
//!   upward when zooming out).

use bevy::prelude::*;

use super::position::Position;
use super::state::WorldState;
use super::view::{
    bevy_from_position, cell_origin_for_anchor, cell_size_at_layer,
    is_layer_pos_solid, layer_pos_from_bevy, position_from_leaf_coord,
    position_to_leaf_coord, target_layer_for, WorldAnchor,
};

// ------------------------------------------------------------ player AABB

/// Half-width on X and Z of the player's AABB, in view-layer cells.
pub const PLAYER_HW: f32 = 0.3;
/// Total height of the player's AABB, in view-layer cells.
pub const PLAYER_H: f32 = 1.7;

// ----------------------------------------------------- local-frame helpers

/// Build the local anchor that places `pos` at (approximately)
/// Bevy `(0, 0, 0)` — specifically, at `Vec3::from(pos.offset)`.
/// This lets every collision function run the existing AABB
/// algorithm on a tiny scratch `Vec3` without any `f32` precision
/// concerns, no matter how deep `pos` is in the tree.
#[inline]
fn local_anchor(pos: &Position) -> WorldAnchor {
    WorldAnchor {
        leaf_coord: position_to_leaf_coord(pos),
    }
}

/// The entity's Bevy position in its own local anchor frame. Always
/// equal to the sub-voxel offset `(pos.offset[0], pos.offset[1],
/// pos.offset[2])`, which lives in `[0, 1)³`.
#[inline]
fn local_bevy(pos: &Position, anchor: &WorldAnchor) -> Vec3 {
    bevy_from_position(pos, anchor)
}

/// Fold a local Bevy `Vec3` (in the frame of `anchor`) back into a
/// [`Position`], splitting it into an exact `i64` leaf delta and a
/// `[0, 1)` sub-voxel offset. Returns `None` if the result leaves
/// the world.
fn position_from_local(local: Vec3, anchor: &WorldAnchor) -> Option<Position> {
    let int_delta: [i64; 3] = [
        local.x.floor() as i64,
        local.y.floor() as i64,
        local.z.floor() as i64,
    ];
    let new_leaf = [
        anchor.leaf_coord[0] + int_delta[0],
        anchor.leaf_coord[1] + int_delta[1],
        anchor.leaf_coord[2] + int_delta[2],
    ];
    let mut new_pos = position_from_leaf_coord(new_leaf)?;
    new_pos.offset = [
        local.x - int_delta[0] as f32,
        local.y - int_delta[1] as f32,
        local.z - int_delta[2] as f32,
    ];
    Some(new_pos)
}

// ----------------------------------------------------------- block grid
//
// The block grid at `target_layer` is anchored at the root's
// `-x,-y,-z` corner in the world's integer leaf frame. Viewed from
// the local anchor (= the entity's own leaf coord), block index
// `rbx = 0` is the block that contains the entity, `rbx = 1` is
// its +x neighbour, and so on. `cell_origin` is the Bevy offset
// from the local anchor to that `rbx = 0` block's corner, which
// is always `-(anchor.leaf_coord % block_size)` — a small value
// in `(-block_size, 0]`.

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

/// Sample the layer-`target_layer` cell at the anchor-local block
/// index `(bx, by, bz)`. Goes through [`layer_pos_from_bevy`] which
/// handles the `i64` absolute recovery internally.
fn is_target_block_solid(
    world: &WorldState,
    target_layer: u8,
    bx: i32,
    by: i32,
    bz: i32,
    block_size: f32,
    cell_origin: Vec3,
    anchor: &WorldAnchor,
) -> bool {
    let center = cell_origin
        + Vec3::new(
            (bx as f32 + 0.5) * block_size,
            (by as f32 + 0.5) * block_size,
            (bz as f32 + 0.5) * block_size,
        );
    match layer_pos_from_bevy(center, target_layer, anchor) {
        Some(lp) => is_layer_pos_solid(world, &lp),
        None => false,
    }
}

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

// --------------------------------------------------------- public API

/// Resolve movement for an entity at `pos` with velocity `vel` and a
/// horizontal input delta, against the target-layer block grid. On
/// return `pos` has been updated in place via exact `i64` path math;
/// `vel.y` is zeroed if a vertical clip occurred.
pub fn move_and_collide(
    pos: &mut Position,
    vel: &mut Vec3,
    horizontal_delta: Vec2,
    dt: f32,
    world: &WorldState,
    view_layer: u8,
) {
    let view_cell = cell_size_at_layer(view_layer);
    let target_layer = target_layer_for(view_layer);
    let block_size = cell_size_at_layer(target_layer);
    let block_size_i64 = block_size as i64;
    let anchor = local_anchor(pos);
    let cell_origin = cell_origin_for_anchor(&anchor, block_size_i64);

    // `local_pos` is the entity's Bevy position in its own local
    // anchor frame — always equal to the sub-voxel offset, so
    // always in `[0, 1)³`. The AABB algorithm runs entirely on
    // this small scratch `Vec3`; nothing here ever touches a
    // long-range coordinate.
    let mut local_pos = local_bevy(pos, &anchor);

    let mut dy = vel.y * dt;
    let dx = horizontal_delta.x;
    let dz = horizontal_delta.y;

    let player = player_aabb(local_pos, view_cell);
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
                    &anchor,
                ) {
                    blocks.push((bx, by, bz));
                }
            }
        }
    }

    // --- Y first ---
    let pa = player_aabb(local_pos, view_cell);
    let original_dy = dy;
    for &(bx, by, bz) in &blocks {
        dy = clip_axis(&pa, dy, 1, bx, by, bz, block_size, cell_origin);
    }
    local_pos.y += dy;
    if (dy - original_dy).abs() > 1e-6 {
        vel.y = 0.0;
    }

    // --- X ---
    let pa = player_aabb(local_pos, view_cell);
    let mut clipped_dx = dx;
    for &(bx, by, bz) in &blocks {
        clipped_dx = clip_axis(&pa, clipped_dx, 0, bx, by, bz, block_size, cell_origin);
    }
    local_pos.x += clipped_dx;

    // --- Z ---
    let pa = player_aabb(local_pos, view_cell);
    let mut clipped_dz = dz;
    for &(bx, by, bz) in &blocks {
        clipped_dz = clip_axis(&pa, clipped_dz, 2, bx, by, bz, block_size, cell_origin);
    }
    local_pos.z += clipped_dz;

    // Fold the new local Vec3 back into the `Position`. The delta
    // from the anchor (an `i64` leaf coord) is small — at most a
    // fraction of a view cell — so the `f32`-to-`i64` cast is
    // exact.
    if let Some(updated) = position_from_local(local_pos, &anchor) {
        *pos = updated;
    }
}

/// Is the entity at `pos` standing on something at `view_layer`?
/// Probes a small downward nudge against the target-layer block
/// grid. Purely a read — does not mutate `pos`.
pub fn on_ground(pos: &Position, world: &WorldState, view_layer: u8) -> bool {
    let view_cell = cell_size_at_layer(view_layer);
    let target_layer = target_layer_for(view_layer);
    let block_size = cell_size_at_layer(target_layer);
    let block_size_i64 = block_size as i64;
    let anchor = local_anchor(pos);
    let cell_origin = cell_origin_for_anchor(&anchor, block_size_i64);
    let local_pos = local_bevy(pos, &anchor);

    let player = player_aabb(local_pos, view_cell);
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
                    &anchor,
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

/// Snap `pos` onto the top of the apparent ground at the current
/// view layer. Used after a zoom change, because the target-layer
/// collision grid can inflate thin features when the block size
/// grows past the feature's actual thickness.
///
/// The grassland ground is 125 leaves deep; the presence-preserving
/// downsample in `tree::downsample` propagates "any non-empty child
/// → solid parent". So at view layer 6 (block size 625 leaves) the
/// one-row ground becomes a full 625-leaf-tall solid block. An
/// entity at `pos.y = 2` after spawning would be inside that
/// inflated block, and `move_and_collide`'s clip only stops motion
/// heading into a block from outside — gravity would otherwise
/// pull the entity straight through.
///
/// Walks `pos`'s column at the new target layer. If the feet block
/// is solid, walk upward until reaching empty space. Otherwise walk
/// downward until reaching solid (the mirror case — zoom in shrinks
/// the apparent ground, entity has to drop). Both walks are capped
/// at 256 steps — more than enough to cross the 125-leaf grassland
/// ground even at `target_layer = MAX_LAYER` (`block_size = 1`), so
/// the walk converges at every view layer without hand-tuning per
/// target.
pub fn snap_to_ground(pos: &mut Position, world: &WorldState, view_layer: u8) {
    let target_layer = target_layer_for(view_layer);
    let block_size = cell_size_at_layer(target_layer);
    let block_size_i64 = block_size as i64;
    let anchor = local_anchor(pos);
    let cell_origin = cell_origin_for_anchor(&anchor, block_size_i64);
    let mut local_pos = local_bevy(pos, &anchor);

    let bx = ((local_pos.x - cell_origin.x) / block_size).floor() as i32;
    let bz = ((local_pos.z - cell_origin.z) / block_size).floor() as i32;
    let mut by = ((local_pos.y - cell_origin.y) / block_size).floor() as i32;

    let solid_at = |by: i32| {
        is_target_block_solid(
            world,
            target_layer,
            bx,
            by,
            bz,
            block_size,
            cell_origin,
            &anchor,
        )
    };

    let mut new_y: Option<f32> = None;
    if solid_at(by) {
        for _ in 0..256 {
            by += 1;
            if !solid_at(by) {
                new_y = Some(cell_origin.y + by as f32 * block_size);
                break;
            }
        }
    } else {
        for _ in 0..256 {
            by -= 1;
            if solid_at(by) {
                new_y = Some(cell_origin.y + (by + 1) as f32 * block_size);
                break;
            }
        }
    }

    if let Some(y) = new_y {
        local_pos.y = y;
        if let Some(updated) = position_from_local(local_pos, &anchor) {
            *pos = updated;
        }
    }
    // If we didn't converge in 64 steps, leave `pos` as-is rather
    // than teleport somewhere surprising.
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tree::{slot_index, MAX_LAYER, NODE_VOXELS_PER_AXIS};
    use super::super::position::NODE_PATH_LEN;

    /// Leaf-level position in the all-zero path at voxel `(12, vy, 12)`
    /// — the all-zero leaf is at Bevy (~0, ~-125..-100, ~0) under the
    /// legacy constant anchor. With the floating anchor we express
    /// tests against the Position directly.
    fn position_at_voxel(vy: u8) -> Position {
        let mut p = Position::origin();
        p.voxel = [12, vy, 12];
        p.offset = [0.5, 0.0, 0.5];
        p
    }

    /// Position resting directly on the grass surface: leaf-coord
    /// y = `GROUND_Y_VOXELS` (the first air voxel), voxel y = 0,
    /// fractional offset y = 0. In the position's own local anchor
    /// frame that's Bevy `(offset.x, 0.0, offset.z)` with the top
    /// face of the grass at Bevy `y = 0`.
    fn position_on_ground() -> Position {
        let mut path = [0u8; NODE_PATH_LEN];
        // At depth `MAX_LAYER - 2` the layer-10 node at slot
        // `(0, 1, 0)` starts at root-local leaf y = 125 =
        // `GROUND_Y_VOXELS` — the first air voxel above the grass.
        path[NODE_PATH_LEN - 2] = slot_index(0, 1, 0) as u8;
        let mid = (NODE_VOXELS_PER_AXIS / 2) as u8;
        Position {
            path,
            voxel: [mid, 0, mid],
            offset: [0.5, 0.0, 0.5],
        }
    }

    /// Position 2 leaves above the grass top face. Drops under
    /// gravity for a couple of frames before landing.
    fn position_airborne() -> Position {
        let mut path = [0u8; NODE_PATH_LEN];
        path[NODE_PATH_LEN - 2] = slot_index(0, 1, 0) as u8;
        let mid = (NODE_VOXELS_PER_AXIS / 2) as u8;
        Position {
            path,
            voxel: [mid, 2, mid],
            offset: [0.5, 0.0, 0.5],
        }
    }

    /// At every view layer, a position placed "just above the grass"
    /// must end up standing on top of the apparent ground after a
    /// [`snap_to_ground`] call — feet in an empty block, block
    /// directly below them solid. Regression against the bug where
    /// pressing Q put the player inside an inflated ground block
    /// and gravity then pulled them through.
    #[test]
    fn snap_to_ground_leaves_position_standing_at_every_view_layer() {
        use super::super::render::{MAX_ZOOM, MIN_ZOOM};
        let world = WorldState::new_grassland();
        for view_layer in MIN_ZOOM..=MAX_ZOOM {
            let target = target_layer_for(view_layer);
            let block_size = cell_size_at_layer(target);
            let block_size_i64 = block_size as i64;
            let mut pos = position_airborne();
            snap_to_ground(&mut pos, &world, view_layer);

            // Reconstruct the same local-anchor frame the public
            // API uses internally and verify the contract.
            let anchor = local_anchor(&pos);
            let cell_origin = cell_origin_for_anchor(&anchor, block_size_i64);
            let local_pos = local_bevy(&pos, &anchor);
            let bx = ((local_pos.x - cell_origin.x) / block_size).floor() as i32;
            let bz = ((local_pos.z - cell_origin.z) / block_size).floor() as i32;
            // `local_pos.y` lands exactly on a block boundary after
            // the snap, so nudge up by a fraction of a block when
            // computing which block "contains" the feet for the
            // is-empty check.
            let by_feet = ((local_pos.y - cell_origin.y + 0.01 * block_size)
                / block_size)
                .floor() as i32;
            assert!(
                !is_target_block_solid(
                    &world,
                    target,
                    bx,
                    by_feet,
                    bz,
                    block_size,
                    cell_origin,
                    &anchor,
                ),
                "view layer {view_layer}: feet block (by={by_feet}) is solid after snap"
            );
            assert!(
                is_target_block_solid(
                    &world,
                    target,
                    bx,
                    by_feet - 1,
                    bz,
                    block_size,
                    cell_origin,
                    &anchor,
                ),
                "view layer {view_layer}: block under feet is not solid — position is floating"
            );
        }
    }

    #[test]
    fn on_ground_just_above_surface() {
        let world = WorldState::new_grassland();
        // Feet resting on the top face of the grass.
        let standing = position_on_ground();
        assert!(on_ground(&standing, &world, MAX_LAYER));

        // Feet 2 leaves up — airborne, not on the floor.
        let airborne = position_airborne();
        assert!(!on_ground(&airborne, &world, MAX_LAYER));
    }

    /// Ground solidity at every layer — mirrors the test in `view.rs`
    /// but exercises it through the collision pipeline.
    #[test]
    fn ground_is_reachable_via_collision_at_every_view_layer() {
        use super::super::render::{MAX_ZOOM, MIN_ZOOM};
        let world = WorldState::new_grassland();
        // A position inside the grass region — voxel y=0 inside the
        // all-zero leaf puts root-local y = 0 (the top grass voxel).
        let grass_pos = position_at_voxel(0);
        for view_layer in MIN_ZOOM..=MAX_ZOOM {
            // Starting inside solid, `on_ground` won't necessarily
            // be true (we're inside, not resting on), but
            // `snap_to_ground` should move us out and standing.
            let mut p = grass_pos.clone();
            snap_to_ground(&mut p, &world, view_layer);
            assert!(
                on_ground(&p, &world, view_layer),
                "view layer {view_layer}: snapped position is not on_ground",
            );
        }
    }
}
