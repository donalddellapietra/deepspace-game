//! Path-based position in the recursive base-3 tree.
//!
//! A `Position` is a pair `(path, offset)`:
//!
//! - `path: [u8; 63]` is a sequence of 27-ary slot indices. `path[0]` is
//!   the slot chosen at the root, `path[depth-1]` is the slot at the
//!   deepest resolved ancestor. Root-at-index-0, leaf-at-index-depth-1.
//! - `depth: u8` says how many entries of `path` are populated
//!   (0 ≤ depth ≤ 63). Entries past `depth` are 0 and never read.
//! - `offset: [f32; 3]` is the sub-slot fractional position inside the
//!   deepest resolved node, always in `[0, 1)³`.
//!
//! No f32 ever accumulates across cell boundaries — offsets live inside
//! one node's local frame, and overflow is absorbed by `cartesian_step`
//! which bubbles carries up the path.
//!
//! See `docs/experimental-architecture/refactor-decisions.md` for the
//! full model this is the foundation of.

use super::tree::{CHILDREN_PER_NODE, MAX_DEPTH, slot_coords, slot_index};

// ----------------------------------------------------------------- Position

#[derive(Clone, Copy, Debug)]
pub struct Position {
    pub path: [u8; MAX_DEPTH],
    pub depth: u8,
    pub offset: [f32; 3],
}

impl Position {
    /// Position at the tree root with a given offset in `[0, 1)³`.
    pub fn root(offset: [f32; 3]) -> Self {
        Self { path: [0; MAX_DEPTH], depth: 0, offset }
    }

    /// Construct a position at an explicit sequence of slots.
    ///
    /// `slots` lists slot indices from root to leaf; `slots[0]` is the
    /// root-level slot. Panics if `slots.len() > MAX_DEPTH` or if any
    /// slot is ≥ 27. `offset` must be in `[0, 1)³`.
    pub fn at_slot_path(slots: &[u8], offset: [f32; 3]) -> Self {
        assert!(slots.len() <= MAX_DEPTH, "path too deep");
        let mut path = [0u8; MAX_DEPTH];
        for (i, &s) in slots.iter().enumerate() {
            assert!((s as usize) < CHILDREN_PER_NODE, "slot out of range");
            path[i] = s;
        }
        Self { path, depth: slots.len() as u8, offset }
    }

    #[inline]
    pub fn slots(&self) -> &[u8] {
        &self.path[..self.depth as usize]
    }

    /// Path up to (but not including) the leaf slot. Empty at root.
    #[inline]
    pub fn parent_slots(&self) -> &[u8] {
        if self.depth == 0 { &[] } else { &self.path[..(self.depth as usize - 1)] }
    }

    /// True if two positions resolve to the same cell (same path + depth),
    /// regardless of offset. Use this when locality matters and sub-cell
    /// offset does not.
    #[inline]
    pub fn same_cell(&self, other: &Self) -> bool {
        self.depth == other.depth && self.slots() == other.slots()
    }
}

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.slots() == other.slots()
            && self.offset == other.offset
    }
}

// ------------------------------------------------------- step_neighbor (Cartesian)

/// Step one cell along `axis` (0=X, 1=Y, 2=Z) in `direction` (±1), under
/// the Cartesian interpretation (every level is a 3×3×3 XYZ grid).
///
/// Returns `true` if the step was absorbed somewhere in the path, `false`
/// if the carry propagated past the root (zoom-out clamp — position is
/// left unchanged).
///
/// When a step overflows the leaf slot along `axis`, the carry bubbles
/// up the path until a level is found where the corresponding slot can
/// be incremented/decremented in range. Every level that was popped has
/// its `axis`-coord rewritten to the wrap position (0 when going +,
/// 2 when going -) — the position re-enters the neighbor cell from the
/// opposite side at each level.
///
/// Offset is NOT modified by this function. Callers that are integrating
/// continuous motion should adjust offset around the step (see
/// `Position::add_offset`).
pub fn cartesian_step(pos: &mut Position, axis: usize, direction: i8) -> bool {
    debug_assert!(axis < 3);
    debug_assert!(direction == 1 || direction == -1);
    if pos.depth == 0 { return false; }

    // Walk from leaf upward, find the highest level where the step is in range.
    let mut absorbed_at: Option<usize> = None;
    for level in (0..(pos.depth as usize)).rev() {
        let (sx, sy, sz) = slot_coords(pos.path[level] as usize);
        let mut coords = [sx, sy, sz];
        let c = coords[axis] as i8 + direction;
        if (0..3).contains(&c) {
            coords[axis] = c as usize;
            pos.path[level] = slot_index(coords[0], coords[1], coords[2]) as u8;
            absorbed_at = Some(level);
            break;
        }
    }
    let Some(level) = absorbed_at else { return false; };

    // Levels that were popped: rewrite axis-coord to the wrap position.
    let wrap = if direction > 0 { 0 } else { 2 };
    for l in (level + 1)..(pos.depth as usize) {
        let (sx, sy, sz) = slot_coords(pos.path[l] as usize);
        let mut coords = [sx, sy, sz];
        coords[axis] = wrap;
        pos.path[l] = slot_index(coords[0], coords[1], coords[2]) as u8;
    }
    true
}

// -------------------------------------------------- offset integration + zoom

impl Position {
    /// Integrate a sub-cell delta into `offset`, bubbling whole-cell
    /// carries into `cartesian_step`. O(1) in the common case (no
    /// overflow), O(depth) when crossing a high-level boundary.
    ///
    /// Each axis independently. If the carry walks past the root, the
    /// offset is clamped into `[0, 1)` on that axis and the step is
    /// dropped (the position cannot leave the root cell).
    pub fn add_offset(&mut self, delta: [f32; 3]) {
        for axis in 0..3 {
            self.offset[axis] += delta[axis];
            while self.offset[axis] >= 1.0 {
                if cartesian_step(self, axis, 1) {
                    self.offset[axis] -= 1.0;
                } else {
                    self.offset[axis] = 1.0 - f32::EPSILON;
                    break;
                }
            }
            while self.offset[axis] < 0.0 {
                if cartesian_step(self, axis, -1) {
                    self.offset[axis] += 1.0;
                } else {
                    self.offset[axis] = 0.0;
                    break;
                }
            }
        }
    }

    /// Descend one level: pick the child slot containing the current
    /// offset, and express the offset in that child's `[0, 1)` frame.
    /// No-op if already at `MAX_DEPTH`.
    pub fn zoom_in(&mut self) {
        if (self.depth as usize) >= MAX_DEPTH { return; }
        let mut coords = [0usize; 3];
        for axis in 0..3 {
            let scaled = self.offset[axis] * 3.0;
            let slot = (scaled.floor() as i32).clamp(0, 2) as usize;
            coords[axis] = slot;
            self.offset[axis] = (scaled - slot as f32).clamp(0.0, 1.0 - f32::EPSILON);
        }
        self.path[self.depth as usize] =
            slot_index(coords[0], coords[1], coords[2]) as u8;
        self.depth += 1;
    }

    /// Ascend one level: merge the leaf slot back into the offset.
    /// No-op at root (we never fabricate virtual ancestor nodes).
    pub fn zoom_out(&mut self) {
        if self.depth == 0 { return; }
        self.depth -= 1;
        let slot = self.path[self.depth as usize] as usize;
        let (sx, sy, sz) = slot_coords(slot);
        let coords = [sx, sy, sz];
        for axis in 0..3 {
            self.offset[axis] = (coords[axis] as f32 + self.offset[axis]) / 3.0;
        }
    }
}

// ----------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_slot_path_round_trip() {
        let slots = [13u8, 0, 26, 5];
        let p = Position::at_slot_path(&slots, [0.25, 0.5, 0.75]);
        assert_eq!(p.depth, 4);
        assert_eq!(p.slots(), &slots);
        assert_eq!(p.offset, [0.25, 0.5, 0.75]);
    }

    #[test]
    fn root_position() {
        let p = Position::root([0.0, 0.0, 0.0]);
        assert_eq!(p.depth, 0);
        assert!(p.slots().is_empty());
        assert!(p.parent_slots().is_empty());
    }

    #[test]
    fn same_cell_ignores_offset() {
        let a = Position::at_slot_path(&[13, 0], [0.1, 0.2, 0.3]);
        let b = Position::at_slot_path(&[13, 0], [0.9, 0.8, 0.7]);
        assert!(a.same_cell(&b));
        assert_ne!(a, b);
    }

    #[test]
    fn same_cell_rejects_different_depth() {
        let a = Position::at_slot_path(&[13], [0.0; 3]);
        let b = Position::at_slot_path(&[13, 0], [0.0; 3]);
        assert!(!a.same_cell(&b));
    }

    #[test]
    fn cartesian_step_x_within_cell() {
        // slot 13 = (1,1,1). +X => (2,1,1) = slot_index(2,1,1).
        let mut p = Position::at_slot_path(&[13], [0.0; 3]);
        assert!(cartesian_step(&mut p, 0, 1));
        assert_eq!(p.path[0] as usize, slot_index(2, 1, 1));
    }

    #[test]
    fn cartesian_step_bubbles_up_one_level() {
        // leaf at (2,1,1): +X overflows leaf, bubbles to parent (1,1,1)→(2,1,1),
        // leaf wraps to (0,1,1).
        let parent = slot_index(1, 1, 1) as u8;
        let leaf = slot_index(2, 1, 1) as u8;
        let mut p = Position::at_slot_path(&[parent, leaf], [0.0; 3]);
        assert!(cartesian_step(&mut p, 0, 1));
        assert_eq!(p.path[0] as usize, slot_index(2, 1, 1));
        assert_eq!(p.path[1] as usize, slot_index(0, 1, 1));
    }

    #[test]
    fn cartesian_step_bubbles_up_multiple_levels() {
        // All three levels at x=2: +X bubbles all the way to level 0.
        let slot = slot_index(2, 1, 1) as u8;
        let mut p = Position::at_slot_path(&[slot_index(0, 1, 1) as u8, slot, slot, slot], [0.0; 3]);
        assert!(cartesian_step(&mut p, 0, 1));
        // Level 0 absorbs: (0,1,1)→(1,1,1).
        assert_eq!(p.path[0] as usize, slot_index(1, 1, 1));
        // Levels 1..3 wrap to x=0, other axes preserved.
        for l in 1..4 {
            let (sx, sy, sz) = slot_coords(p.path[l] as usize);
            assert_eq!(sx, 0);
            assert_eq!(sy, 1);
            assert_eq!(sz, 1);
        }
    }

    #[test]
    fn cartesian_step_clamps_past_root() {
        let slot = slot_index(2, 1, 1) as u8;
        let mut p = Position::at_slot_path(&[slot, slot], [0.25; 3]);
        let before = p;
        // +X all the way out: no level can absorb.
        assert!(!cartesian_step(&mut p, 0, 1));
        assert_eq!(p, before, "clamp leaves position unchanged");
    }

    #[test]
    fn cartesian_step_negative_direction() {
        let slot = slot_index(0, 1, 1) as u8;
        let mut p = Position::at_slot_path(&[slot_index(2, 1, 1) as u8, slot], [0.0; 3]);
        assert!(cartesian_step(&mut p, 0, -1));
        // Parent (2,1,1)→(1,1,1); leaf wraps to x=2.
        assert_eq!(p.path[0] as usize, slot_index(1, 1, 1));
        assert_eq!(p.path[1] as usize, slot_index(2, 1, 1));
    }

    #[test]
    fn cartesian_step_preserves_other_axes_on_bubble() {
        // leaf (2, 2, 0): +X bubbles, wrap x=0 but y=2, z=0 preserved.
        let leaf = slot_index(2, 2, 0) as u8;
        let parent = slot_index(1, 1, 1) as u8;
        let mut p = Position::at_slot_path(&[parent, leaf], [0.0; 3]);
        assert!(cartesian_step(&mut p, 0, 1));
        assert_eq!(p.path[0] as usize, slot_index(2, 1, 1));
        assert_eq!(p.path[1] as usize, slot_index(0, 2, 0));
    }

    #[test]
    fn zoom_in_then_zoom_out_is_identity() {
        let mut p = Position::at_slot_path(&[13, 5], [0.3, 0.7, 0.1]);
        let before = p;
        p.zoom_in();
        assert_eq!(p.depth, 3);
        p.zoom_out();
        assert_eq!(p.depth, 2);
        assert_eq!(p.slots(), before.slots());
        for axis in 0..3 {
            assert!((p.offset[axis] - before.offset[axis]).abs() < 1e-5);
        }
    }

    #[test]
    fn zoom_in_picks_correct_slot() {
        // offset (0.7, 0.1, 0.5) → scaled (2.1, 0.3, 1.5) → slot (2, 0, 1).
        let mut p = Position::at_slot_path(&[13], [0.7, 0.1, 0.5]);
        p.zoom_in();
        assert_eq!(p.depth, 2);
        let (sx, sy, sz) = slot_coords(p.path[1] as usize);
        assert_eq!((sx, sy, sz), (2, 0, 1));
        // Remainder: (0.1, 0.3, 0.5).
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
        assert!((p.offset[1] - 0.3).abs() < 1e-5);
        assert!((p.offset[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn zoom_out_at_root_is_noop() {
        let mut p = Position::root([0.5; 3]);
        p.zoom_out();
        assert_eq!(p.depth, 0);
        assert_eq!(p.offset, [0.5; 3]);
    }

    #[test]
    fn add_offset_within_cell() {
        let mut p = Position::at_slot_path(&[13], [0.2, 0.2, 0.2]);
        p.add_offset([0.3, 0.0, 0.0]);
        assert_eq!(p.slots(), &[13]);
        assert!((p.offset[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn add_offset_crosses_cell_boundary() {
        let parent = slot_index(1, 1, 1) as u8;
        let leaf = slot_index(1, 1, 1) as u8;
        let mut p = Position::at_slot_path(&[parent, leaf], [0.9, 0.5, 0.5]);
        p.add_offset([0.2, 0.0, 0.0]);
        // Overflow: leaf (1,1,1) → (2,1,1); offset x → 0.1.
        assert_eq!(p.path[1] as usize, slot_index(2, 1, 1));
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn add_offset_clamps_at_root() {
        let mut p = Position::at_slot_path(&[slot_index(2, 1, 1) as u8], [0.99, 0.5, 0.5]);
        p.add_offset([10.0, 0.0, 0.0]);
        // Can't leave root: offset clamped, path unchanged.
        assert_eq!(p.path[0] as usize, slot_index(2, 1, 1));
        assert!(p.offset[0] < 1.0);
    }

    #[test]
    fn add_offset_negative_crosses_cell() {
        let parent = slot_index(1, 1, 1) as u8;
        let leaf = slot_index(1, 1, 1) as u8;
        let mut p = Position::at_slot_path(&[parent, leaf], [0.05, 0.5, 0.5]);
        p.add_offset([-0.1, 0.0, 0.0]);
        // leaf (1,1,1) → (0,1,1); offset → 0.95.
        assert_eq!(p.path[1] as usize, slot_index(0, 1, 1));
        assert!((p.offset[0] - 0.95).abs() < 1e-5);
    }
}
