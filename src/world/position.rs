//! Path-based coordinates.
//!
//! A `Position` locates a point in the tree without any absolute-XYZ
//! accumulator. `path[0..depth]` walks from root (index 0) to the
//! deepest resolved node (index `depth - 1`); `offset` is the
//! sub-slot fractional position in `[0, 1)³` inside that node.
//!
//! See `docs/experimental-architecture/refactor-decisions.md` §2.
//!
//! Step 1 of the migration: this module is purely additive — no
//! existing code calls it yet. `step_neighbor` here is the Cartesian
//! dispatch only; `NodeKind`-aware dispatch arrives in step 2.

use crate::world::tree::{slot_coords, slot_index, CHILDREN_PER_NODE, MAX_DEPTH};

// -------------------------------------------------------------- position

/// A point in the recursive tree expressed as (path, offset).
///
/// `path[0]` is the root's chosen child slot; `path[depth - 1]` is
/// the slot of the deepest resolved node. Slots beyond `depth` are
/// unused (kept at 0 for hygiene; never read).
///
/// `offset` is the sub-slot fractional position, each axis in
/// `[0, 1)`. For Cartesian nodes the axes are `(x, y, z)`; for
/// `CubedSphereFace` nodes (step 2+) they're `(u, v, r)`. The struct
/// is agnostic — the deepest node's `NodeKind` determines meaning.
///
/// `PartialEq` compares path, depth, and offset exactly. Use
/// [`Position::same_cell`] to compare only cell identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub path: [u8; MAX_DEPTH],
    pub depth: u8,
    pub offset: [f32; 3],
}

impl Default for Position {
    fn default() -> Self {
        Self::root()
    }
}

impl Position {
    /// The root of the tree, offset `(0, 0, 0)`.
    pub const fn root() -> Self {
        Self { path: [0; MAX_DEPTH], depth: 0, offset: [0.0; 3] }
    }

    /// Construct from an explicit slot slice and sub-slot offset.
    ///
    /// Panics if `slots.len() > MAX_DEPTH`, any slot is ≥ 27, or any
    /// offset axis is outside `[0, 1)` or not finite. Callers should
    /// use this constructor rather than poking at the `path` array
    /// directly.
    pub fn at_slot_path(slots: &[u8], offset: [f32; 3]) -> Self {
        assert!(slots.len() <= MAX_DEPTH, "path exceeds MAX_DEPTH");
        for &s in slots {
            assert!((s as usize) < CHILDREN_PER_NODE, "slot {} out of range", s);
        }
        for (i, v) in offset.iter().enumerate() {
            assert!(v.is_finite(), "offset[{}] not finite", i);
            assert!(*v >= 0.0 && *v < 1.0, "offset[{}] = {} outside [0, 1)", i, v);
        }
        let mut path = [0u8; MAX_DEPTH];
        path[..slots.len()].copy_from_slice(slots);
        Self { path, depth: slots.len() as u8, offset }
    }

    /// The populated portion of the path.
    #[inline]
    pub fn slots(&self) -> &[u8] {
        &self.path[..self.depth as usize]
    }

    /// Cell identity: same path + depth, ignoring offset.
    ///
    /// Two positions "in the same cell" agree on `same_cell` but may
    /// differ in `==` because their sub-slot offsets differ. Use this
    /// whenever you mean "do these refer to the same node?" rather
    /// than "are these bit-identical locations?".
    pub fn same_cell(&self, other: &Self) -> bool {
        self.depth == other.depth && self.slots() == other.slots()
    }

    /// Descend one level: push the slot determined by offset overflow.
    ///
    /// `offset * 3`'s integer part becomes the new slot; the
    /// fractional part is the new offset in the child's frame. No-op
    /// and returns `false` if already at `MAX_DEPTH`.
    pub fn zoom_in(&mut self) -> bool {
        if self.depth as usize >= MAX_DEPTH {
            return false;
        }
        let mut coords = [0usize; 3];
        for axis in 0..3 {
            let v = self.offset[axis] * 3.0;
            let i = (v.floor() as i32).clamp(0, 2) as usize;
            coords[axis] = i;
            self.offset[axis] = v - i as f32;
            // Floating-point can leave offset at exactly 1.0 (e.g.
            // 0.333... * 3.0 rounds to 1.0). Clamp back into [0, 1).
            if self.offset[axis] >= 1.0 {
                self.offset[axis] = 0.0;
                coords[axis] = (coords[axis] + 1).min(2);
            } else if self.offset[axis] < 0.0 {
                self.offset[axis] = 0.0;
            }
        }
        self.path[self.depth as usize] =
            slot_index(coords[0], coords[1], coords[2]) as u8;
        self.depth += 1;
        true
    }

    /// Ascend one level: pop the deepest slot, remapping the offset
    /// into the parent's frame.
    ///
    /// Returns `false` if already at root — zoom-out past root clamps,
    /// per §2b of refactor-decisions.md.
    pub fn zoom_out(&mut self) -> bool {
        if self.depth == 0 {
            return false;
        }
        self.depth -= 1;
        let slot = self.path[self.depth as usize] as usize;
        let (sx, sy, sz) = slot_coords(slot);
        let s = [sx as f32, sy as f32, sz as f32];
        for axis in 0..3 {
            self.offset[axis] = (self.offset[axis] + s[axis]) / 3.0;
        }
        self.path[self.depth as usize] = 0;
        true
    }

    /// Integrate a velocity step: `offset += delta`, then carry across
    /// cell boundaries via [`step_neighbor`]. Clamps at the tree root
    /// if the carry bubbles past it.
    pub fn add_offset(&mut self, delta: [f32; 3]) {
        for axis in 0..3 {
            self.offset[axis] += delta[axis];
            loop {
                if self.offset[axis] >= 1.0 {
                    if !carry_axis(self, axis, 1) {
                        // Past root in +axis; clamp just below 1.
                        self.offset[axis] = 1.0 - f32::EPSILON;
                        break;
                    }
                    self.offset[axis] -= 1.0;
                } else if self.offset[axis] < 0.0 {
                    if !carry_axis(self, axis, -1) {
                        // Past root in -axis; clamp at 0.
                        self.offset[axis] = 0.0;
                        break;
                    }
                    self.offset[axis] += 1.0;
                } else {
                    break;
                }
            }
        }
    }
}

// -------------------------------------------------------- step_neighbor

/// Move one cell along `axis` by `dir` (`+1` or `-1`). Cartesian
/// semantics: increment/decrement the slot coord; carry to parent on
/// wrap.
///
/// Returns `false` and leaves `pos` unchanged if the carry reaches
/// past the root — the step is absorbed by the caller (typically by
/// clamping offset, as in [`Position::add_offset`]).
///
/// Step 2 introduces `NodeKind` and turns this into a dispatcher over
/// body/face kinds (§2c); step 1 is Cartesian-only.
pub fn step_neighbor(pos: &mut Position, axis: usize, dir: i32) -> bool {
    debug_assert!(axis < 3, "axis must be 0, 1, or 2");
    debug_assert!(dir == 1 || dir == -1, "dir must be ±1");
    carry_axis(pos, axis, dir)
}

/// Walk up `pos.path`, adjusting the axis coord by `dir` at each
/// level, carrying through 3-wraps. Returns `false` if we walk past
/// the root without resolving; on failure the path is restored.
fn carry_axis(pos: &mut Position, axis: usize, dir: i32) -> bool {
    let saved = pos.path;
    let mut d = pos.depth as usize;
    while d > 0 {
        let slot = pos.path[d - 1] as usize;
        let (sx, sy, sz) = slot_coords(slot);
        let mut coords = [sx as i32, sy as i32, sz as i32];
        let v = coords[axis] + dir;
        if (0..=2).contains(&v) {
            coords[axis] = v;
            pos.path[d - 1] =
                slot_index(coords[0] as usize, coords[1] as usize, coords[2] as usize) as u8;
            return true;
        }
        // Wrap and continue carrying into the parent.
        coords[axis] = if v < 0 { 2 } else { 0 };
        pos.path[d - 1] =
            slot_index(coords[0] as usize, coords[1] as usize, coords[2] as usize) as u8;
        d -= 1;
    }
    pos.path = saved;
    false
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn pos_at(slots: &[u8], offset: [f32; 3]) -> Position {
        Position::at_slot_path(slots, offset)
    }

    #[test]
    fn root_defaults() {
        let p = Position::root();
        assert_eq!(p.depth, 0);
        assert_eq!(p.offset, [0.0; 3]);
        assert_eq!(p.slots(), &[] as &[u8]);
    }

    #[test]
    fn at_slot_path_round_trip() {
        let slots = [slot_index(1, 0, 2) as u8, slot_index(2, 2, 1) as u8];
        let p = pos_at(&slots, [0.25, 0.5, 0.75]);
        assert_eq!(p.slots(), &slots);
        assert_eq!(p.depth, 2);
        assert_eq!(p.offset, [0.25, 0.5, 0.75]);
    }

    #[test]
    #[should_panic]
    fn at_slot_path_rejects_bad_slot() {
        Position::at_slot_path(&[27], [0.0; 3]);
    }

    #[test]
    #[should_panic]
    fn at_slot_path_rejects_offset_at_one() {
        Position::at_slot_path(&[0], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn same_cell_ignores_offset() {
        let a = pos_at(&[1, 2], [0.1, 0.2, 0.3]);
        let b = pos_at(&[1, 2], [0.9, 0.8, 0.7]);
        assert!(a.same_cell(&b));
        assert_ne!(a, b);
    }

    #[test]
    fn same_cell_respects_path_and_depth() {
        let a = pos_at(&[1, 2], [0.0; 3]);
        let b = pos_at(&[1, 3], [0.0; 3]);
        let c = pos_at(&[1], [0.0; 3]);
        assert!(!a.same_cell(&b));
        assert!(!a.same_cell(&c));
    }

    #[test]
    fn zoom_in_then_out_is_identity() {
        let original = pos_at(&[5, 10], [0.2, 0.6, 0.4]);
        let mut p = original;
        assert!(p.zoom_in());
        assert_eq!(p.depth, 3);
        assert!(p.zoom_out());
        assert_eq!(p.depth, 2);
        // Path prefix matches; offset is close (modulo float roundoff).
        assert_eq!(p.slots(), original.slots());
        for axis in 0..3 {
            assert!(
                (p.offset[axis] - original.offset[axis]).abs() < 1e-5,
                "axis {} drifted: {} vs {}",
                axis,
                p.offset[axis],
                original.offset[axis]
            );
        }
    }

    #[test]
    fn zoom_in_derives_correct_slot() {
        // offset (0.7, 0.2, 0.5) * 3 = (2.1, 0.6, 1.5).
        // slot coords (2, 0, 1); new offset (0.1, 0.6, 0.5).
        let mut p = pos_at(&[], [0.7, 0.2, 0.5]);
        assert!(p.zoom_in());
        assert_eq!(p.slots(), &[slot_index(2, 0, 1) as u8]);
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
        assert!((p.offset[1] - 0.6).abs() < 1e-5);
        assert!((p.offset[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn zoom_out_at_root_clamps() {
        let mut p = Position::root();
        assert!(!p.zoom_out());
        assert_eq!(p, Position::root());
    }

    #[test]
    fn step_neighbor_within_parent() {
        // Start at slot (1, 1, 1) inside one parent; step +x → (2, 1, 1).
        let mut p = pos_at(&[slot_index(1, 1, 1) as u8], [0.5; 3]);
        assert!(step_neighbor(&mut p, 0, 1));
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
    }

    #[test]
    fn step_neighbor_carries_to_parent() {
        // Slot (2, 0, 0) inside slot (0, 0, 0); step +x wraps the
        // deeper slot to (0, 0, 0) and carries +x at the parent.
        let mut p = pos_at(
            &[slot_index(0, 0, 0) as u8, slot_index(2, 0, 0) as u8],
            [0.5; 3],
        );
        assert!(step_neighbor(&mut p, 0, 1));
        assert_eq!(
            p.slots(),
            &[slot_index(1, 0, 0) as u8, slot_index(0, 0, 0) as u8]
        );
    }

    #[test]
    fn step_neighbor_multi_level_carry() {
        // Root slot is middle-x; two descendants sit at the +x edge.
        // Step +x at the deepest level carries through both edges and
        // resolves at the root by incrementing its x from 1 to 2.
        let edge = slot_index(2, 0, 0) as u8;
        let mut p = pos_at(&[slot_index(1, 0, 0) as u8, edge, edge], [0.5; 3]);
        assert!(step_neighbor(&mut p, 0, 1));
        assert_eq!(
            p.slots(),
            &[
                slot_index(2, 0, 0) as u8,
                slot_index(0, 0, 0) as u8,
                slot_index(0, 0, 0) as u8,
            ]
        );
    }

    #[test]
    fn step_neighbor_past_root_fails_and_restores() {
        // Single-level position at the -x edge; step -x should hit
        // root and return false, leaving path unchanged.
        let before = pos_at(&[slot_index(0, 1, 1) as u8], [0.5; 3]);
        let mut p = before;
        assert!(!step_neighbor(&mut p, 0, -1));
        assert_eq!(p.path, before.path);
    }

    #[test]
    fn add_offset_no_carry() {
        let mut p = pos_at(&[slot_index(1, 1, 1) as u8], [0.1, 0.2, 0.3]);
        p.add_offset([0.1, 0.1, 0.1]);
        assert_eq!(p.slots(), &[slot_index(1, 1, 1) as u8]);
        for (i, expected) in [0.2, 0.3, 0.4].iter().enumerate() {
            assert!((p.offset[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn add_offset_crosses_cell() {
        // Offset near +x edge; +0.3 overflows into next cell.
        let mut p = pos_at(&[slot_index(1, 1, 1) as u8], [0.8, 0.0, 0.0]);
        p.add_offset([0.3, 0.0, 0.0]);
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn add_offset_crosses_multiple_cells() {
        // +2.5 on x crosses two cell boundaries: (0, *, *) → (1, *, *)
        // after +1.0, → (2, *, *) after another +1.0, lands at
        // offset 0.5 in the third cell.
        let mut p = pos_at(&[slot_index(0, 1, 1) as u8], [0.0, 0.0, 0.0]);
        p.add_offset([2.5, 0.0, 0.0]);
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
        assert!((p.offset[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn add_offset_clamps_at_root_positive() {
        // At root-level +x edge; push past root in +x should clamp.
        let mut p = pos_at(&[slot_index(2, 1, 1) as u8], [0.9, 0.5, 0.5]);
        p.add_offset([0.5, 0.0, 0.0]);
        // Clamped: still at (2, 1, 1), offset[0] pinned just below 1.
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
        assert!(p.offset[0] < 1.0);
        assert!(p.offset[0] > 0.99);
    }

    #[test]
    fn add_offset_clamps_at_root_negative() {
        let mut p = pos_at(&[slot_index(0, 1, 1) as u8], [0.1, 0.5, 0.5]);
        p.add_offset([-0.5, 0.0, 0.0]);
        assert_eq!(p.slots(), &[slot_index(0, 1, 1) as u8]);
        assert_eq!(p.offset[0], 0.0);
    }
}
