//! Positions, paths, and walking operations.
//!
//! See `docs/architecture/coordinates.md` for the design. In Model A
//! every live position sits at the leaf layer of the tree, so a
//! `Position` always has `path.len() == MAX_LAYER` slots.

use super::tree::{
    slot_coords, slot_index, BRANCH_FACTOR, MAX_LAYER, NODE_VOXELS_PER_AXIS,
};

/// Length of a leaf `NodePath` — equals `MAX_LAYER`.
pub const NODE_PATH_LEN: usize = MAX_LAYER as usize;

/// Slots from the root to a leaf. `path[0]` is the slot at the root;
/// `path[NODE_PATH_LEN - 1]` is the leaf's slot in its immediate parent.
pub type NodePath = [u8; NODE_PATH_LEN];

/// Origin-aligned path (`[0; 12]`).
pub const fn zero_path() -> NodePath {
    [0; NODE_PATH_LEN]
}

/// A continuous position anywhere inside the world. Always leaf-layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub path: NodePath,
    /// Voxel inside the leaf. Each component is `0..NODE_VOXELS_PER_AXIS`.
    pub voxel: [u8; 3],
    /// Sub-voxel offset. Each component is `0.0..1.0`.
    pub offset: [f32; 3],
}

/// A bounded position at an arbitrary view layer.
///
/// The valid path prefix has `path().len() == layer`, plus an in-node
/// `cell` in `0..NODE_VOXELS_PER_AXIS`. This is the type the renderer
/// and input layer hand back for clicks at a zoomed-out view —
/// identifying exactly one cell within one layer-`layer` node without
/// ever computing a global coordinate.
///
/// This type is fully stack-allocated and `Copy` — every component is
/// bounded in size (see `docs/architecture/coordinates.md`: "nothing
/// is ever a big number"). The `path_slots` buffer is `NODE_PATH_LEN`
/// bytes long, and only the first `layer` slots are semantically
/// valid; the remainder is zero padding. Access the valid prefix via
/// [`LayerPos::path`].
///
/// See `docs/architecture/editing.md` and the 2D prototype's `LayerPos`:
/// a click on a cell at view layer `L` names the layer-`(L + 2)` subtree
/// reached by decomposing `(cx, cy, cz)` into `slot_a = (c / 5)` and
/// `slot_b = (c % 5)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerPos {
    /// First `layer` slots are valid; slots `[layer..]` are zero
    /// padding. Private so the "valid prefix" invariant is enforced
    /// — the only readers go through [`LayerPos::path`].
    path_slots: NodePath,
    /// Cell coordinates inside the node's `25³` grid. Each `0..25`.
    pub cell: [u8; 3],
    /// The layer the node lives at. `0..=MAX_LAYER`.
    pub layer: u8,
}

impl LayerPos {
    /// Project a leaf `Position` down to the layer-`layer` cell that
    /// contains it. Walks up the leaf's path, applying the downsample's
    /// inverse at each step: the leaf's contribution to the parent cell
    /// at `(cx, cy, cz)` where `cx = 5 * slot_x + child_cx / 5`.
    ///
    /// Every intermediate cell stays in `0..NODE_VOXELS_PER_AXIS`.
    pub fn from_leaf(leaf: &Position, layer: u8) -> Self {
        assert!(layer <= MAX_LAYER);
        let mut cx = leaf.voxel[0];
        let mut cy = leaf.voxel[1];
        let mut cz = leaf.voxel[2];
        // Walk up from the leaf-parent slot (index MAX_LAYER - 1) down
        // to `layer`, applying the parent-cell projection once per step.
        for i in (layer as usize..NODE_PATH_LEN).rev() {
            let (sx, sy, sz) = slot_coords(leaf.path[i] as usize);
            cx = (BRANCH_FACTOR as u8) * (sx as u8) + cx / (BRANCH_FACTOR as u8);
            cy = (BRANCH_FACTOR as u8) * (sy as u8) + cy / (BRANCH_FACTOR as u8);
            cz = (BRANCH_FACTOR as u8) * (sz as u8) + cz / (BRANCH_FACTOR as u8);
        }
        let mut path_slots: NodePath = zero_path();
        path_slots[..layer as usize]
            .copy_from_slice(&leaf.path[..layer as usize]);
        Self {
            path_slots,
            cell: [cx, cy, cz],
            layer,
        }
    }

    /// Build a `LayerPos` from a full `NodePath` array, a cell, and a
    /// layer. Only the first `layer` slots of `path_slots` are
    /// semantically valid; the rest must be zero. This is the fast
    /// path used by `layer_pos_from_leaf_coord_direct` which already
    /// builds a zero-initialised array.
    #[inline]
    pub fn from_path_and_cell(
        path_slots: NodePath,
        cell: [u8; 3],
        layer: u8,
    ) -> Self {
        debug_assert!(layer <= MAX_LAYER);
        Self {
            path_slots,
            cell,
            layer,
        }
    }

    /// Build a `LayerPos` from an explicit slot slice, a cell, and a
    /// layer. Panics if `path.len() != layer as usize` or the layer
    /// exceeds `MAX_LAYER`. Used by tests and callers that need to
    /// synthesise a `LayerPos` directly rather than project one from a
    /// leaf.
    #[cfg(test)]
    pub fn from_parts(path: &[u8], cell: [u8; 3], layer: u8) -> Self {
        assert!(layer <= MAX_LAYER);
        assert_eq!(
            path.len(),
            layer as usize,
            "LayerPos::from_parts: path.len() must equal layer"
        );
        let mut path_slots: NodePath = zero_path();
        path_slots[..layer as usize].copy_from_slice(path);
        Self {
            path_slots,
            cell,
            layer,
        }
    }

    /// Read the valid path prefix — the `lp.layer`-long list of slot
    /// indices from the root to the node this position sits in.
    /// `layer == 0` returns an empty slice (the position is in the
    /// root itself).
    #[inline]
    pub fn path(&self) -> &[u8] {
        &self.path_slots[..self.layer as usize]
    }
}

impl Position {
    /// Debug-assert the [`Position::offset`] invariant: every
    /// component is in `[0.0, 1.0)`. Called at every non-trivial
    /// construction site in the codebase so a floor/fract mismatch
    /// on a boundary trips a clear assertion instead of silently
    /// desyncing the integer and fractional parts.
    #[inline]
    pub fn debug_check_offset(&self) {
        debug_assert!(
            (0.0..1.0).contains(&self.offset[0])
                && (0.0..1.0).contains(&self.offset[1])
                && (0.0..1.0).contains(&self.offset[2]),
            "Position::offset out of range: {:?}",
            self.offset
        );
    }

    pub fn origin() -> Self {
        Self {
            path: zero_path(),
            voxel: [0; 3],
            offset: [0.0; 3],
        }
    }

    /// Step by a signed integer number of leaf voxels on one axis.
    /// Returns `false` if the step walked past the root. On failure
    /// the position is restored to its pre-call state — an earlier
    /// version left a partially-walked `path` when a multi-leaf
    /// crossing failed mid-loop.
    pub fn step_voxels(&mut self, axis: usize, delta: i32) -> bool {
        debug_assert!(axis < 3);
        let saved_path = self.path;
        let saved_voxel = self.voxel[axis];
        let mut new_v = self.voxel[axis] as i32 + delta;
        while new_v >= NODE_VOXELS_PER_AXIS as i32 {
            if !self.step_neighbor_leaf(axis, true) {
                self.path = saved_path;
                self.voxel[axis] = saved_voxel;
                return false;
            }
            new_v -= NODE_VOXELS_PER_AXIS as i32;
        }
        while new_v < 0 {
            if !self.step_neighbor_leaf(axis, false) {
                self.path = saved_path;
                self.voxel[axis] = saved_voxel;
                return false;
            }
            new_v += NODE_VOXELS_PER_AXIS as i32;
        }
        self.voxel[axis] = new_v as u8;
        true
    }

    /// Add a sub-voxel offset on one axis, carrying whole voxels into
    /// `voxel` and `path`. Returns `false` on walking off the root.
    ///
    /// On failure, `self` is left untouched: the voxel step is tried
    /// first and the new fractional offset is only committed if the
    /// step succeeded. An earlier version mutated `self.offset`
    /// before calling `step_voxels`, which left a silent 1-leaf
    /// desync (post-delta offset, rolled-back voxel) on failure.
    pub fn add_offset_axis(&mut self, axis: usize, delta: f32) -> bool {
        debug_assert!(axis < 3);
        let new_offset = self.offset[axis] + delta;
        let whole = new_offset.floor();
        if whole != 0.0 && !self.step_voxels(axis, whole as i32) {
            return false;
        }
        self.offset[axis] = new_offset - whole;
        true
    }

    /// Cross one leaf boundary on the given axis. Walks up the path
    /// until it finds an ancestor whose slot can step on that axis,
    /// then walks back down resetting lower slots and `voxel[axis]`
    /// to the opposite face. `O(MAX_LAYER)` worst case.
    fn step_neighbor_leaf(&mut self, axis: usize, positive: bool) -> bool {
        let delta: i32 = if positive { 1 } else { -1 };
        let reset_slot_axis: usize = if positive { 0 } else { BRANCH_FACTOR - 1 };
        let reset_voxel_axis: u8 = if positive {
            0
        } else {
            (NODE_VOXELS_PER_AXIS - 1) as u8
        };

        let mut layer_idx: usize = NODE_PATH_LEN - 1;
        loop {
            let slot = self.path[layer_idx];
            let (sx, sy, sz) = slot_coords(slot as usize);
            let mut ax = [sx, sy, sz];
            let new_a = ax[axis] as i32 + delta;
            if new_a >= 0 && new_a < BRANCH_FACTOR as i32 {
                ax[axis] = new_a as usize;
                self.path[layer_idx] =
                    slot_index(ax[0], ax[1], ax[2]) as u8;
                for lower in (layer_idx + 1)..NODE_PATH_LEN {
                    let ls = self.path[lower];
                    let (lx, ly, lz) = slot_coords(ls as usize);
                    let mut lax = [lx, ly, lz];
                    lax[axis] = reset_slot_axis;
                    self.path[lower] =
                        slot_index(lax[0], lax[1], lax[2]) as u8;
                }
                self.voxel[axis] = reset_voxel_axis;
                return true;
            }
            if layer_idx == 0 {
                return false;
            }
            layer_idx -= 1;
        }
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn origin_at(voxel: [u8; 3]) -> Position {
        let mut p = Position::origin();
        p.voxel = voxel;
        p
    }

    #[test]
    fn walk_within_leaf() {
        let mut p = origin_at([5, 5, 5]);
        assert!(p.step_voxels(0, 3));
        assert_eq!(p.voxel, [8, 5, 5]);
        assert_eq!(p.path, zero_path());
    }

    #[test]
    fn walk_into_negative_voxel() {
        let mut p = origin_at([5, 5, 5]);
        assert!(p.step_voxels(0, -3));
        assert_eq!(p.voxel, [2, 5, 5]);
        assert_eq!(p.path, zero_path());
    }

    #[test]
    fn walk_crosses_one_leaf_boundary_positive() {
        let mut p = origin_at([24, 7, 12]);
        assert!(p.step_voxels(0, 1));
        // We stepped into the +x neighbor leaf. The leaf's slot (at
        // path[NODE_PATH_LEN - 1]) changed from slot (0,0,0) to (1,0,0).
        assert_eq!(p.voxel, [0, 7, 12]);
        assert_eq!(p.path[NODE_PATH_LEN - 1], slot_index(1, 0, 0) as u8);
        for lo in 0..(NODE_PATH_LEN - 1) {
            assert_eq!(p.path[lo], 0);
        }
    }

    #[test]
    fn walk_crosses_one_leaf_boundary_negative() {
        // Start at voxel.x = 0 of a leaf in slot (1, 0, 0). Walking -x
        // by one voxel should cross into slot (0, 0, 0), voxel.x = 24.
        let mut p = Position::origin();
        p.path[NODE_PATH_LEN - 1] = slot_index(1, 0, 0) as u8;
        assert!(p.step_voxels(0, -1));
        assert_eq!(p.voxel, [(NODE_VOXELS_PER_AXIS - 1) as u8, 0, 0]);
        assert_eq!(p.path[NODE_PATH_LEN - 1], 0);
    }

    #[test]
    fn walk_crosses_two_layer_boundary() {
        // Leaf slot (4, 0, 0); parent slot (4, 0, 0); grandparent slot
        // (0, 0, 0). Walking +x crosses all three at once: the leaf and
        // parent slots can't step, but the grandparent can.
        let mut p = Position::origin();
        p.path[NODE_PATH_LEN - 1] = slot_index(4, 0, 0) as u8;
        p.path[NODE_PATH_LEN - 2] = slot_index(4, 0, 0) as u8;
        p.path[NODE_PATH_LEN - 3] = slot_index(0, 0, 0) as u8;
        p.voxel = [(NODE_VOXELS_PER_AXIS - 1) as u8, 7, 12];
        assert!(p.step_voxels(0, 1));
        assert_eq!(p.voxel, [0, 7, 12]);
        assert_eq!(p.path[NODE_PATH_LEN - 3], slot_index(1, 0, 0) as u8);
        assert_eq!(p.path[NODE_PATH_LEN - 2], slot_index(0, 0, 0) as u8);
        assert_eq!(p.path[NODE_PATH_LEN - 1], slot_index(0, 0, 0) as u8);
    }

    #[test]
    fn walk_preserves_other_axes_when_crossing() {
        // Leaf slot has non-zero y, z components. Walking +x across
        // should preserve them at every level.
        let mut p = Position::origin();
        p.path[NODE_PATH_LEN - 1] = slot_index(4, 3, 2) as u8;
        p.path[NODE_PATH_LEN - 2] = slot_index(4, 2, 1) as u8;
        p.path[NODE_PATH_LEN - 3] = slot_index(0, 1, 3) as u8;
        p.voxel = [(NODE_VOXELS_PER_AXIS - 1) as u8, 0, 0];
        assert!(p.step_voxels(0, 1));
        assert_eq!(p.path[NODE_PATH_LEN - 3], slot_index(1, 1, 3) as u8);
        assert_eq!(p.path[NODE_PATH_LEN - 2], slot_index(0, 2, 1) as u8);
        assert_eq!(p.path[NODE_PATH_LEN - 1], slot_index(0, 3, 2) as u8);
        assert_eq!(p.voxel, [0, 0, 0]);
    }

    #[test]
    fn walk_off_root_returns_false() {
        let mut p = Position::origin();
        for i in 0..NODE_PATH_LEN {
            p.path[i] = slot_index(0, 0, 0) as u8;
        }
        p.voxel = [0, 0, 0];
        assert!(!p.step_voxels(0, -1));
    }

    #[test]
    fn walk_off_root_positive() {
        let mut p = Position::origin();
        for i in 0..NODE_PATH_LEN {
            p.path[i] = slot_index(4, 0, 0) as u8;
        }
        p.voxel = [(NODE_VOXELS_PER_AXIS - 1) as u8, 0, 0];
        assert!(!p.step_voxels(0, 1));
    }

    #[test]
    fn round_trip_walk() {
        // Walk +x then -x by the same amount; expect to land where we
        // started.
        let start = origin_at([5, 7, 3]);
        let mut p = start;
        assert!(p.step_voxels(0, 123));
        assert!(p.step_voxels(0, -123));
        assert_eq!(p, start);
    }

    #[test]
    fn round_trip_walk_crossing_multiple_boundaries() {
        let mut start = Position::origin();
        start.path[NODE_PATH_LEN - 1] = slot_index(2, 1, 3) as u8;
        start.voxel = [12, 5, 20];
        let mut p = start;
        assert!(p.step_voxels(0, 100)); // multiple leaf crossings
        assert!(p.step_voxels(0, -100));
        assert_eq!(p, start);
    }

    #[test]
    fn add_offset_carries_into_voxel() {
        let mut p = origin_at([5, 5, 5]);
        p.offset = [0.3, 0.0, 0.0];
        assert!(p.add_offset_axis(0, 0.5));
        assert_eq!(p.voxel, [5, 5, 5]);
        assert!((p.offset[0] - 0.8).abs() < 1e-5);

        // 0.8 + 0.5 = 1.3 — one whole voxel carried, 0.3 leftover.
        assert!(p.add_offset_axis(0, 0.5));
        assert_eq!(p.voxel, [6, 5, 5]);
        assert!((p.offset[0] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn add_offset_negative_borrows_from_voxel() {
        let mut p = origin_at([5, 5, 5]);
        p.offset = [0.3, 0.0, 0.0];
        assert!(p.add_offset_axis(0, -0.5));
        assert_eq!(p.voxel, [4, 5, 5]);
        assert!((p.offset[0] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn add_offset_large_delta_chains_carries() {
        let mut p = origin_at([5, 5, 5]);
        // 30 leaf voxels in +x from voxel 5 crosses one leaf boundary
        // (at voxel 25). We should end up in slot (1, 0, 0), voxel.x = 10.
        assert!(p.add_offset_axis(0, 30.0));
        assert_eq!(p.voxel, [10, 5, 5]);
        assert_eq!(p.path[NODE_PATH_LEN - 1], slot_index(1, 0, 0) as u8);
    }

    // --------------------------------------------------------- LayerPos

    #[test]
    fn layer_pos_from_leaf_at_leaf_is_identity() {
        let mut p = origin_at([7, 11, 3]);
        p.path[NODE_PATH_LEN - 1] = slot_index(2, 1, 4) as u8;
        let lp = LayerPos::from_leaf(&p, MAX_LAYER);
        assert_eq!(lp.layer, MAX_LAYER);
        assert_eq!(lp.path().len(), NODE_PATH_LEN);
        assert_eq!(lp.path(), &p.path[..]);
        assert_eq!(lp.cell, [7, 11, 3]);
    }

    #[test]
    fn layer_pos_from_leaf_projects_one_level() {
        // Leaf's last slot is (sx=3, sy=2, sz=4), voxel is (17, 6, 11).
        // Going up one level: parent cell =
        //   (5*3 + 17/5, 5*2 + 6/5, 5*4 + 11/5) = (18, 11, 22).
        let mut p = Position::origin();
        p.path[NODE_PATH_LEN - 1] = slot_index(3, 2, 4) as u8;
        p.voxel = [17, 6, 11];
        let lp = LayerPos::from_leaf(&p, MAX_LAYER - 1);
        assert_eq!(lp.layer, MAX_LAYER - 1);
        assert_eq!(lp.path().len(), NODE_PATH_LEN - 1);
        assert_eq!(lp.cell, [18, 11, 22]);
    }

    #[test]
    fn layer_pos_from_leaf_stays_bounded_at_every_layer() {
        let mut p = Position::origin();
        for i in 0..NODE_PATH_LEN {
            p.path[i] = slot_index(
                (i as usize) % 5,
                (i as usize + 2) % 5,
                (i as usize + 3) % 5,
            ) as u8;
        }
        p.voxel = [23, 19, 7];
        for layer in 0..=MAX_LAYER {
            let lp = LayerPos::from_leaf(&p, layer);
            assert!(lp.cell[0] < NODE_VOXELS_PER_AXIS as u8);
            assert!(lp.cell[1] < NODE_VOXELS_PER_AXIS as u8);
            assert!(lp.cell[2] < NODE_VOXELS_PER_AXIS as u8);
            assert_eq!(lp.path().len(), layer as usize);
        }
    }
}
