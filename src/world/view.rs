//! Layer-space: the single home for Bevy ↔ tree coordinate math,
//! per-layer cell sizes, the view → target layer rule, and the
//! view-cell solidity query.
//!
//! Everything that needs to answer "where is this in Bevy space",
//! "how big is a cell at view layer L", or "is there anything
//! visible at this [`LayerPos`]" imports from here. `render.rs`
//! keeps only the renderer itself; `collision.rs` keeps only the
//! collision algorithms; `interaction/`, `editor/`, `player.rs`,
//! and `camera.rs` all route layer-space math through this module.
//!
//! ## Why this module exists
//!
//! These helpers used to live in two places: `render.rs` owned
//! [`ROOT_ORIGIN`], [`cell_size_at_layer`], and [`target_layer_for`];
//! `collision.rs` owned [`position_from_bevy`],
//! [`layer_pos_from_bevy`], and [`is_layer_pos_solid`]. Every other
//! module imported from both. Two classes of bug came out of that:
//!
//! 1. **The `+2` rule drifted.** Collision and render both computed
//!    `(view_layer + 2).min(MAX_LAYER)` inline; it was easy to update
//!    one and forget the other. Centralising it in
//!    [`target_layer_for`] makes that impossible.
//! 2. **`is_layer_pos_solid` lived next to the collision algorithms**,
//!    which made it look collision-specific — so the raycast in
//!    `interaction/` happily re-implemented its own solidity notion
//!    and diverged. Moving it here makes it obvious that it's the
//!    *one* solidity query.
//!
//! ## Contract with [`super::tree::downsample`]
//!
//! [`is_layer_pos_solid`] at a non-leaf layer reads one voxel from
//! the layer's stored `25³` downsample grid. That voxel correctly
//! answers "is there anything visible below me" **only** because
//! `tree::downsample` is presence-preserving — any non-empty child
//! voxel surfaces the most common non-empty value at the parent, and
//! only a fully-empty `5³` block collapses to `EMPTY_VOXEL`. If you
//! change the downsample rule you break this query, and the render
//! and picking invariants with it.

use bevy::prelude::*;

use super::position::{LayerPos, Position, NODE_PATH_LEN};
use super::state::{world_extent_voxels, WorldState};
use super::tree::{
    slot_coords, slot_index, voxel_idx, EMPTY_NODE, EMPTY_VOXEL, MAX_LAYER,
    NODE_VOXELS_PER_AXIS,
};

// ---------------------------------------------------------------- anchor

/// Origin of the root node in Bevy space.
///
/// `y` is chosen so that the top face of the `GROUND_Y_VOXELS`-deep
/// grass surface lines up with Bevy `y = 0`. The root-local y range
/// `(0, GROUND_Y_VOXELS)` maps to Bevy `(-GROUND_Y_VOXELS, 0)`, and
/// the all-zero-path leaf sits at the very bottom of the grass
/// region.
///
/// `x` and `z` are **integer offsets**. It is important that they
/// are integer: the raycast in `src/interaction/mod.rs` steps through
/// Bevy integer cells `[c, c+1]`, and the highlight gizmo centres a
/// unit cube at `c + 0.5`. If `ROOT_ORIGIN.{x,z}` had a fractional
/// part, voxels would sit on a non-integer lattice and the outline
/// would drift half a voxel away from the block it names.
///
/// `-13` puts the all-zero-path leaf at Bevy `(-13..12)` on each of
/// x and z, so Bevy `(0, ·, 0)` is near the leaf's centre (voxel 13
/// out of 25). The player spawns at `(0, 2, 0)`, a couple of voxels
/// above the grass surface, and falls onto it.
pub const ROOT_ORIGIN: Vec3 = Vec3::new(
    -13.0,
    -(super::state::GROUND_Y_VOXELS as f32),
    -13.0,
);

// ----------------------------------------------------- per-layer sizes

/// Scale multiplier for a node at `layer`. A layer-`MAX_LAYER` (leaf)
/// node has scale `1.0`; a layer-`MAX_LAYER - 1` node has scale `5.0`;
/// the root (layer 0) has scale `5^MAX_LAYER`. Values get large near
/// the root but stay finite in f32 — only ~`2.4e8` at layer 0.
#[inline]
pub fn scale_for_layer(layer: u8) -> f32 {
    let up = (MAX_LAYER - layer) as i32;
    // 5^up. Up to 5^12 ≈ 2.4e8, well within f32.
    let mut acc: f32 = 1.0;
    for _ in 0..up {
        acc *= 5.0;
    }
    acc
}

/// Bevy-space extent (per axis) of a full node at `layer` — i.e.
/// `25` cells wide, each [`cell_size_at_layer`] big.
#[inline]
pub fn extent_for_layer(layer: u8) -> f32 {
    scale_for_layer(layer) * (NODE_VOXELS_PER_AXIS as f32)
}

/// Bevy-space edge length of ONE cell inside a layer-`layer` node's
/// `25³` grid. At the leaf layer this is `1.0` (a leaf cell is a
/// unit cube in Bevy space); at `layer = MAX_LAYER - 1` it's `5.0`;
/// at the root it's `5^MAX_LAYER`.
///
/// The editor, the highlight gizmo, and the player's speed / eye
/// height all use this to scale with the current zoom.
#[inline]
pub fn cell_size_at_layer(layer: u8) -> f32 {
    scale_for_layer(layer)
}

/// The layer the renderer emits entities at for a given view layer,
/// and the layer collision samples blocks at. Ported from the 2D
/// prototype's `subtexture_25` rule: at view layer `L`, one visible
/// cell corresponds to one layer-`(L + 2)` node, so each emitted
/// entity's mesh shows the fine `(L + 2)` voxel grid instead of the
/// single layer-`L` voxel. Clamped to [`MAX_LAYER`] because you can't
/// descend past the leaves.
///
/// **Every consumer that needs this rule calls this function.** Do
/// not hardcode `(view + 2).min(MAX_LAYER)` at call sites — past
/// bugs came from having the rule in two places and only updating
/// one.
#[inline]
pub fn target_layer_for(view_layer: u8) -> u8 {
    view_layer.saturating_add(2).min(MAX_LAYER)
}

// ------------------------------------------------------ conversions

/// Convert a Bevy `Vec3` into a tree `Position`. Returns `None` when
/// the point is outside the world.
///
/// The all-zero `Position` has its min corner at [`ROOT_ORIGIN`], so
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

/// Project a Bevy `Vec3` straight to a [`LayerPos`] at view layer
/// `layer`. Equivalent to
/// `LayerPos::from_leaf(&position_from_bevy(pos)?, layer)`.
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

// -------------------------------------------------------- solidity

/// Walk the tree from the root to the layer-`lp.layer` node and read
/// the cell `lp.cell` from its `25³` voxel grid. Returns `true` when
/// the voxel is non-empty.
///
/// At a non-leaf layer the voxel is a presence-preserving downsample
/// (see [`super::tree::downsample`]), so this correctly answers
/// "does the view cell named by `lp` contain anything visible" at
/// every layer — not just leaves. This is the single solidity query
/// used by the raycast (`interaction/`) and the collision block
/// sampler (`collision::is_target_block_solid`).
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

// ----------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn scale_at_leaf_is_one() {
        assert_eq!(scale_for_layer(MAX_LAYER), 1.0);
    }

    #[test]
    fn scale_at_layer_above_leaf_is_five() {
        assert_eq!(scale_for_layer(MAX_LAYER - 1), 5.0);
    }

    #[test]
    fn extent_at_leaf_is_node_voxels() {
        assert_eq!(extent_for_layer(MAX_LAYER), NODE_VOXELS_PER_AXIS as f32);
    }

    #[test]
    fn target_layer_clamps_at_max() {
        assert_eq!(target_layer_for(MAX_LAYER), MAX_LAYER);
        assert_eq!(target_layer_for(MAX_LAYER - 1), MAX_LAYER);
        assert_eq!(target_layer_for(MAX_LAYER - 2), MAX_LAYER);
        assert_eq!(target_layer_for(MAX_LAYER - 3), MAX_LAYER - 1);
        assert_eq!(target_layer_for(0), 2);
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

    /// The grassland's ground surface must be reachable as a solid
    /// [`LayerPos`] at every view layer. Regression against a
    /// majority-vote downsample that washed out thin features — at
    /// view L ≤ 8 the 125-leaf-deep ground used to collapse to air
    /// in the cascaded downsample, so the crosshair clicked through
    /// ground it could see. The presence-preserving downsample in
    /// [`super::tree::downsample`] is what makes this hold.
    #[test]
    fn ground_is_solid_at_every_view_layer() {
        let world = WorldState::new_grassland();
        // A point guaranteed to sit inside the grass region: one
        // voxel below the surface at the world's all-zero-path.
        let probe = Vec3::new(0.0, -1.0, 0.0);
        for view_layer in 2..=MAX_LAYER {
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
        }
    }
}
