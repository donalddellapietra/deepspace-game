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
//! ## Floating anchor (a.k.a. "no absolute Bevy coords")
//!
//! The design doc ([`docs/architecture/coordinates.md`]) is explicit:
//!
//! > Nothing is ever a "big number." Every component is bounded.
//!
//! The authoritative representation of any location in the world is
//! a [`Position`] — a path of 12 slot indices, a `25³` in-leaf voxel
//! coord, and a `[0, 1)` sub-voxel offset. None of those components
//! ever exceed 125. There is no big number anywhere in that
//! representation, and it works equally well at the corner or the
//! centre of the 6-billion-leaf root.
//!
//! The earlier code pinned the root's `(-x, -y, -z)` corner to a
//! constant Bevy `Vec3` called `ROOT_ORIGIN`. Every `bevy_from_*`
//! helper computed "origin + path-walk-offset-in-leaves", which is
//! fine when the path-walk offset is small, but loses `f32`
//! precision catastrophically for paths deep inside the root — at
//! Bevy `x ≈ 3e9` the step size of `f32` is ~180 leaves, so leaf-
//! granularity anything (collision, raycasting, picking) fails.
//! That's why the old spawn was stuck at a corner of the root.
//!
//! Fix: the *anchor* between path-space and Bevy-space is now a
//! [`WorldAnchor`] resource, not a constant, and it holds an
//! **integer leaf coordinate**. Every `bevy_from_*` helper computes
//! its output as `(target_leaf_coord - anchor.leaf_coord)`, an
//! **`i64` subtraction** (exact), and only casts the small delta to
//! `f32`. The anchor is updated each frame to track the player's
//! current integer leaf coord, so the player's Bevy `Transform`
//! always stays within `[-1, 1]`-ish regardless of where they are
//! in the world. The player can spawn anywhere — including the
//! exact centre of the root — with perfect `f32` precision at the
//! leaf level.
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
#[cfg(test)]
use super::state::GROUND_Y_VOXELS;
use super::tree::{
    slot_coords, slot_index, voxel_idx, DETAIL_DEPTH, EMPTY_NODE, EMPTY_VOXEL, MAX_LAYER,
    NODE_VOXELS_PER_AXIS,
};

// ---------------------------------------------------------------- anchor

/// The floating anchor between tree-space and Bevy-space.
///
/// `leaf_coord` is the integer leaf-voxel coordinate in the root's
/// frame that Bevy `(0, 0, 0)` currently represents. The player's
/// Bevy `Transform.translation` is always small because the anchor
/// is kept aligned with the player's integer leaf coord — the whole
/// world shifts around the player rather than the player drifting
/// across a fixed Bevy grid.
///
/// Updating the anchor is a straight assignment; callers never need
/// to compute deltas themselves. See `player::sync_anchor_to_player`
/// for the per-frame update loop.
#[derive(Resource, Copy, Clone, Debug)]
pub struct WorldAnchor {
    pub leaf_coord: [i64; 3],
    /// Normalization divisor: how many leaf voxels equal one Bevy unit.
    /// Set to `scale_for_layer(target_layer_for(zoom.layer))` so that
    /// Bevy-space coordinates stay in a bounded range (~800 units)
    /// regardless of zoom level. This prevents atmosphere LUT banding
    /// and post-processing artifacts caused by huge coordinate ranges
    /// at zoomed-out layers.
    pub norm: f32,
}

impl WorldAnchor {
    /// Size of one cell at `layer` in normalized Bevy units.
    /// At the target layer this is 1.0; at the view layer it's
    /// typically 25.0 (= 5^(target - view) = 5^2).
    #[inline]
    pub fn cell_bevy(&self, layer: u8) -> f32 {
        scale_for_layer(layer) / self.norm
    }
}

impl Default for WorldAnchor {
    fn default() -> Self {
        Self {
            leaf_coord: [0, 0, 0],
            norm: 1.0,
        }
    }
}

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
/// and the layer collision samples blocks at. At view layer `L`, one
/// visible cell corresponds to one layer-`(L + DETAIL_DEPTH)` node.
/// Clamped to [`MAX_LAYER`] because you can't descend past the leaves.
///
/// **Every consumer that needs this rule calls this function.** Do
/// not hardcode the depth offset at call sites — past bugs came from
/// having the rule in two places and only updating one.
#[inline]
pub fn target_layer_for(view_layer: u8) -> u8 {
    view_layer.saturating_add(DETAIL_DEPTH).min(MAX_LAYER)
}

/// Bevy offset from the given `anchor` to the `-x, -y, -z` corner
/// of the `cell_size_leaves`-aligned cell that contains the anchor.
///
/// Used by every module that iterates an **anchor-local cell grid**
/// (collision block-sweep, interaction DDA, highlight placement…).
/// The returned `Vec3` is always in `(-cell_size_leaves, 0]` on
/// each axis — tiny, `f32`-safe — because it's just the negated
/// `leaf_coord.rem_euclid(cell_size_leaves)`. Call sites that
/// iterate relative block indices `(rbx, rby, rbz)` then compute
/// each block's Bevy corner as
/// `cell_origin + (rbx, rby, rbz) as f32 * cell_size_leaves as f32`,
/// which stays small-f32 too so long as the iteration range is
/// bounded.
///
/// `cell_size_leaves` is the cell edge length in leaves, not in
/// Bevy units — for a target-layer collision grid at layer `L`
/// that's `cell_size_at_layer(L) as i64`, and for a view-cell
/// grid at view layer `L` it's the same.
#[inline]
pub fn cell_origin_for_anchor(anchor: &WorldAnchor, cell_size_leaves: i64) -> Vec3 {
    let n = anchor.norm;
    Vec3::new(
        -(anchor.leaf_coord[0].rem_euclid(cell_size_leaves) as f32) / n,
        -(anchor.leaf_coord[1].rem_euclid(cell_size_leaves) as f32) / n,
        -(anchor.leaf_coord[2].rem_euclid(cell_size_leaves) as f32) / n,
    )
}

// ------------------------------------------------------ leaf coord math

/// Integer leaf coordinate of a [`Position`] in the root's frame.
///
/// Walks the path top-down, accumulating each slot's contribution
/// in `i64` leaf units, then adds the in-leaf voxel. The offset is
/// ignored — this is the integer part. Total output is in
/// `0..world_extent_voxels()`.
///
/// Used as the common ground between Bevy space and path space:
/// two positions in the world can be subtracted in this
/// representation and the delta is exact, regardless of how far
/// apart they are.
pub fn position_to_leaf_coord(pos: &Position) -> [i64; 3] {
    let mut coord: [i64; 3] = [0; 3];
    let mut extent: i64 = world_extent_voxels();
    for depth in 0..NODE_PATH_LEN {
        let child_extent = extent / 5;
        let (sx, sy, sz) = slot_coords(pos.path[depth] as usize);
        coord[0] += (sx as i64) * child_extent;
        coord[1] += (sy as i64) * child_extent;
        coord[2] += (sz as i64) * child_extent;
        extent = child_extent;
    }
    coord[0] += pos.voxel[0] as i64;
    coord[1] += pos.voxel[1] as i64;
    coord[2] += pos.voxel[2] as i64;
    coord
}

/// Inverse of [`position_to_leaf_coord`]. Walks the tree from the
/// root, picking slot indices from the magnitude of `coord` at each
/// descent step. Returns `None` if `coord` is outside the world
/// bounds. Offset is initialised to zero; callers holding a
/// sub-leaf fractional part set it themselves.
pub fn position_from_leaf_coord(coord: [i64; 3]) -> Option<Position> {
    let world_max = world_extent_voxels();
    if coord.iter().any(|&c| c < 0 || c >= world_max) {
        return None;
    }
    let mut rem = coord;
    let mut path = [0u8; NODE_PATH_LEN];
    let mut extent: i64 = world_max;
    for depth in 0..NODE_PATH_LEN {
        let child_extent = extent / 5;
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
    debug_assert_eq!(extent, NODE_VOXELS_PER_AXIS as i64);
    let voxel = [rem[0] as u8, rem[1] as u8, rem[2] as u8];
    Some(Position { path, voxel, offset: [0.0; 3] })
}

/// `i64` leaf-coord delta, normalized to Bevy units by dividing by
/// `anchor.norm`. This is the one place we cross from the exact-
/// integer world frame into the approximate-`f32` Bevy frame.
/// With normalization, the output stays in a bounded range (~800)
/// regardless of zoom level, preserving `f32` precision and
/// preventing atmosphere/post-processing artifacts.
#[inline]
fn delta_as_vec3(target: [i64; 3], anchor: &WorldAnchor) -> Vec3 {
    let n = anchor.norm;
    Vec3::new(
        (target[0] - anchor.leaf_coord[0]) as f32 / n,
        (target[1] - anchor.leaf_coord[1]) as f32 / n,
        (target[2] - anchor.leaf_coord[2]) as f32 / n,
    )
}

// ------------------------------------------------------ conversions

/// Convert a [`Position`] into its Bevy coordinate in the frame of
/// `anchor`. `anchor.leaf_coord` is the integer leaf coord that the
/// Bevy origin currently represents, so the output is
/// `(pos - anchor)` in f32 leaves, plus the sub-voxel fractional
/// offset from `pos.offset`.
pub fn bevy_from_position(pos: &Position, anchor: &WorldAnchor) -> Vec3 {
    let coord = position_to_leaf_coord(pos);
    delta_as_vec3(coord, anchor)
        + Vec3::new(pos.offset[0], pos.offset[1], pos.offset[2]) / anchor.norm
}

/// Convert a Bevy `Vec3` in the `anchor` frame back to a
/// [`Position`]. Returns `None` when the resulting leaf coord is
/// outside the world.
///
/// Interprets `bevy` as `(pos.leaf - anchor.leaf_coord) + offset`,
/// where `offset ∈ [0, 1)` is the sub-voxel fractional part. The
/// integer leaf coord recovery is exact (`i64` addition); only the
/// sub-voxel offset lives in `f32`.
pub fn position_from_bevy(bevy: Vec3, anchor: &WorldAnchor) -> Option<Position> {
    if !bevy.x.is_finite() || !bevy.y.is_finite() || !bevy.z.is_finite() {
        return None;
    }
    // Scale back from normalized Bevy units to leaf units.
    let leaf = bevy * anchor.norm;
    // Split into integer leaf delta and sub-voxel fractional part.
    let fx = leaf.x.floor();
    let fy = leaf.y.floor();
    let fz = leaf.z.floor();
    let int_delta: [i64; 3] = [fx as i64, fy as i64, fz as i64];
    let coord: [i64; 3] = [
        anchor.leaf_coord[0] + int_delta[0],
        anchor.leaf_coord[1] + int_delta[1],
        anchor.leaf_coord[2] + int_delta[2],
    ];
    let mut pos = position_from_leaf_coord(coord)?;
    pos.offset = [leaf.x - fx, leaf.y - fy, leaf.z - fz];
    pos.debug_check_offset();
    Some(pos)
}

/// Project a Bevy `Vec3` straight to a [`LayerPos`] at view layer
/// `layer`, in the `anchor` frame. Equivalent to
/// `LayerPos::from_leaf(&position_from_bevy(bevy, anchor)?, layer)`.
pub fn layer_pos_from_bevy(
    bevy: Vec3,
    layer: u8,
    anchor: &WorldAnchor,
) -> Option<LayerPos> {
    let leaf = position_from_bevy(bevy, anchor)?;
    Some(LayerPos::from_leaf(&leaf, layer))
}

/// Integer leaf coord of the min corner of the cell at `lp` —
/// i.e. the `(leaf x, leaf y, leaf z)` of the cell's `-x, -y, -z`
/// face in the root's frame.
pub fn layer_pos_min_leaf_coord(lp: &LayerPos) -> [i64; 3] {
    let mut coord: [i64; 3] = [0; 3];
    let mut extent: i64 = world_extent_voxels();
    let path = lp.path();
    for depth in 0..path.len() {
        let child_extent = extent / 5;
        let (sx, sy, sz) = slot_coords(path[depth] as usize);
        coord[0] += (sx as i64) * child_extent;
        coord[1] += (sy as i64) * child_extent;
        coord[2] += (sz as i64) * child_extent;
        extent = child_extent;
    }
    // After descending `lp.layer` slots, `extent` is the layer-`L`
    // node's axis size in leaf voxels. A cell is `extent / 25`
    // leaves wide, and `lp.cell` picks one of the 25 cells on each
    // axis inside that node.
    let cell_size_leaves = extent / (NODE_VOXELS_PER_AXIS as i64);
    coord[0] += (lp.cell[0] as i64) * cell_size_leaves;
    coord[1] += (lp.cell[1] as i64) * cell_size_leaves;
    coord[2] += (lp.cell[2] as i64) * cell_size_leaves;
    coord
}

/// Bevy-space min corner of the cell at `lp`, in the `anchor`
/// frame. Works at any view layer — `lp.layer = 0` returns the
/// root's corner offset from the anchor.
pub fn bevy_origin_of_layer_pos(lp: &LayerPos, anchor: &WorldAnchor) -> Vec3 {
    delta_as_vec3(layer_pos_min_leaf_coord(lp), anchor)
}

/// Bevy-space centre of the cell at `lp`, in the `anchor` frame.
pub fn bevy_center_of_layer_pos(lp: &LayerPos, anchor: &WorldAnchor) -> Vec3 {
    let cell = anchor.cell_bevy(lp.layer);
    bevy_origin_of_layer_pos(lp, anchor) + Vec3::splat(cell * 0.5)
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
    for &slot in lp.path() {
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

// -------------------------------------------- direct leaf-coord → LayerPos

/// Convert an absolute leaf coordinate directly to a [`LayerPos`] at
/// `layer`, without round-tripping through float or a full leaf
/// `Position`. Decomposes the coordinate into slot indices via
/// integer division, stopping after `layer` steps instead of
/// descending all the way to `MAX_LAYER`.
///
/// Returns `None` if the coordinate is outside the world.
pub fn layer_pos_from_leaf_coord_direct(
    coord: [i64; 3],
    layer: u8,
) -> Option<LayerPos> {
    let world_max = world_extent_voxels();
    if coord.iter().any(|&c| c < 0 || c >= world_max) {
        return None;
    }
    let mut rem = coord;
    let mut path = [0u8; NODE_PATH_LEN];
    let mut extent: i64 = world_max;
    for depth in 0..(layer as usize) {
        let child_extent = extent / 5;
        let sx = (rem[0] / child_extent) as usize;
        let sy = (rem[1] / child_extent) as usize;
        let sz = (rem[2] / child_extent) as usize;
        path[depth] = slot_index(sx, sy, sz) as u8;
        rem[0] -= (sx as i64) * child_extent;
        rem[1] -= (sy as i64) * child_extent;
        rem[2] -= (sz as i64) * child_extent;
        extent = child_extent;
    }
    // `extent` is now the node extent in leaf voxels at `layer`.
    // Each cell is `extent / 25` leaves wide.
    let cell_leaves = extent / (NODE_VOXELS_PER_AXIS as i64);
    let cx = (rem[0] / cell_leaves) as u8;
    let cy = (rem[1] / cell_leaves) as u8;
    let cz = (rem[2] / cell_leaves) as u8;
    Some(LayerPos::from_path_and_cell(path, [cx, cy, cz], layer))
}

// ----------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn anchor_origin() -> WorldAnchor {
        // "Origin anchor" — Bevy (0, 0, 0) represents the root's
        // all-zero leaf corner. Matches the historical constant
        // `ROOT_ORIGIN + (13, 125, 13)` after the coordinate shift.
        WorldAnchor { leaf_coord: [0, 0, 0], norm: 1.0 }
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
    fn leaf_coord_round_trip_origin() {
        let p = Position::origin();
        assert_eq!(position_to_leaf_coord(&p), [0, 0, 0]);
        let back = position_from_leaf_coord([0, 0, 0]).unwrap();
        assert_eq!(back.path, p.path);
        assert_eq!(back.voxel, p.voxel);
    }

    #[test]
    fn leaf_coord_round_trip_arbitrary() {
        // Pick a point well inside the world and round-trip via
        // `position_to_leaf_coord` and `position_from_leaf_coord`.
        // This proves the two inversions are consistent.
        let mut p = Position::origin();
        // Layer-11 slot (3, 2, 4), voxel (7, 12, 19).
        p.path[NODE_PATH_LEN - 1] = slot_index(3, 2, 4) as u8;
        p.voxel = [7, 12, 19];

        let coord = position_to_leaf_coord(&p);
        let back = position_from_leaf_coord(coord).unwrap();
        assert_eq!(back.path, p.path);
        assert_eq!(back.voxel, p.voxel);
    }

    #[test]
    fn bevy_round_trip_at_origin_anchor() {
        // At the origin anchor, positions round-trip through Bevy
        // space for points whose leaf coords fit in `f32` without
        // loss. We stay well below the `2^24`-leaf precision limit.
        let anchor = anchor_origin();
        let mut p = Position::origin();
        p.voxel = [5, 10, 0];
        p.offset = [0.5, 0.0, 0.0];
        let v = bevy_from_position(&p, &anchor);
        assert!(approx_eq(v.x, 5.5), "x={}", v.x);
        assert!(approx_eq(v.y, 10.0), "y={}", v.y);
        assert!(approx_eq(v.z, 0.0), "z={}", v.z);
        let back = position_from_bevy(v, &anchor).unwrap();
        assert_eq!(back.voxel, p.voxel);
        assert!(approx_eq(back.offset[0], 0.5));
    }

    #[test]
    fn bevy_from_position_is_zero_when_anchor_is_player() {
        // Crucial invariant for the floating-anchor scheme: if the
        // anchor's leaf coord matches a position's leaf coord, the
        // Bevy projection is `(0, 0, 0)` modulo sub-voxel offset.
        // This is what keeps the player's `Transform.translation`
        // tiny no matter where in the 6-billion-leaf world they
        // are.
        let mut p = Position::origin();
        // Push the path deep into the +x, +y, +z quadrant of the
        // root — the position whose absolute Bevy coord would be
        // catastrophically imprecise under the old constant anchor.
        for depth in 0..NODE_PATH_LEN {
            p.path[depth] = slot_index(2, 2, 2) as u8;
        }
        p.voxel = [12, 12, 12];
        p.offset = [0.3, 0.7, 0.1];

        let anchor = WorldAnchor {
            leaf_coord: position_to_leaf_coord(&p),
            norm: 1.0,
        };
        let v = bevy_from_position(&p, &anchor);
        // Integer delta is exactly zero; only the sub-voxel offset
        // survives.
        assert!(approx_eq(v.x, 0.3));
        assert!(approx_eq(v.y, 0.7));
        assert!(approx_eq(v.z, 0.1));
    }

    #[test]
    fn bevy_round_trip_from_root_centre() {
        // The reason the refactor exists: a position at the
        // arithmetic centre of the root must round-trip with
        // perfect precision, provided the anchor tracks it. This is
        // the spawn-at-centre case the old absolute-`ROOT_ORIGIN`
        // scheme physically couldn't support.
        let mut p = Position::origin();
        for depth in 0..NODE_PATH_LEN {
            p.path[depth] = slot_index(2, 2, 2) as u8;
        }
        p.voxel = [12, 12, 12];
        let anchor = WorldAnchor {
            leaf_coord: position_to_leaf_coord(&p),
            norm: 1.0,
        };
        let v = bevy_from_position(&p, &anchor);
        let back = position_from_bevy(v, &anchor).unwrap();
        assert_eq!(back.path, p.path);
        assert_eq!(back.voxel, p.voxel);
    }

    #[test]
    fn position_from_bevy_rejects_outside_world() {
        let anchor = anchor_origin();
        // One leaf west of the root's `-x` edge.
        assert!(position_from_bevy(Vec3::new(-0.5, 0.0, 0.0), &anchor).is_none());
        // One leaf below the root's `-y` edge.
        assert!(position_from_bevy(Vec3::new(0.0, -0.5, 0.0), &anchor).is_none());
    }

    #[test]
    fn bevy_origin_of_layer_pos_matches_leaf_coord() {
        let anchor = anchor_origin();
        let lp = LayerPos::from_parts(
            &vec![slot_index(0, 0, 0) as u8; (MAX_LAYER - 2) as usize],
            [3, 0, 3],
            MAX_LAYER - 2,
        );
        let bevy = bevy_origin_of_layer_pos(&lp, &anchor);
        let leaf = layer_pos_min_leaf_coord(&lp);
        assert!(approx_eq(bevy.x, leaf[0] as f32));
        assert!(approx_eq(bevy.y, leaf[1] as f32));
        assert!(approx_eq(bevy.z, leaf[2] as f32));
    }

    /// The grassland's ground surface must be reachable as a solid
    /// [`LayerPos`] at every view layer. Regression against a
    /// majority-vote downsample that washed out thin features — the
    /// ground used to collapse to air in the cascaded downsample, so
    /// the crosshair clicked through ground it could see. The
    /// presence-preserving downsample in [`super::tree::downsample`]
    /// is what makes this hold.
    #[test]
    fn ground_is_solid_at_every_view_layer() {
        let world = WorldState::new_grassland();
        let anchor = anchor_origin();
        // A point guaranteed to sit inside the grass region: the
        // first leaf below the surface at the world's all-zero path.
        // In the origin-anchor frame, the grass top face is at
        // Bevy y = `GROUND_Y_VOXELS`, so `y - 1` sits one leaf
        // inside the grass.
        let probe = Vec3::new(0.0, (GROUND_Y_VOXELS - 1) as f32, 0.0);
        for view_layer in 2..=MAX_LAYER {
            let target = target_layer_for(view_layer);
            let lp_view = layer_pos_from_bevy(probe, view_layer, &anchor)
                .expect("probe inside world at view layer");
            let lp_target = layer_pos_from_bevy(probe, target, &anchor)
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
