//! Path-anchored world coordinates.
//!
//! See `docs/experimental-architecture/anchor-refactor-decisions.md`.
//!
//! Every position is `(anchor: Path, offset: [f32; 3])` where the
//! offset is kept in `[0, 1)³` of the anchor cell's local frame.
//! f32 never accumulates across cells: as motion overflows a cell,
//! the anchor advances. Zoom changes the anchor's depth.
//!
//! Step 1 of the anchor refactor introduces these types alongside
//! the legacy `[f32; 3]` coordinates without changing runtime
//! behavior. Later steps migrate the camera, renderer, and editing
//! paths onto `WorldPos`.

use std::hash::{Hash, Hasher};

use crate::world::tree::{slot_coords, slot_index, NodeLibrary, MAX_DEPTH};

// --------------------------------------------------------------- Path

/// Symbolic path through the 27-ary tree. Exact at any depth; no
/// f32 precision loss.
#[derive(Clone, Copy)]
pub struct Path {
    slots: [u8; MAX_DEPTH],
    depth: u8,
}

impl Path {
    pub const fn root() -> Self {
        Self { slots: [0u8; MAX_DEPTH], depth: 0 }
    }

    #[inline]
    pub fn depth(&self) -> u8 {
        self.depth
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        self.depth == 0
    }

    #[inline]
    pub fn slot(&self, i: usize) -> u8 {
        debug_assert!(i < self.depth as usize);
        self.slots[i]
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.slots[..self.depth as usize]
    }

    pub fn push(&mut self, slot: u8) {
        debug_assert!((slot as usize) < 27, "slot must be < 27");
        debug_assert!((self.depth as usize) < MAX_DEPTH, "path overflow");
        self.slots[self.depth as usize] = slot;
        self.depth += 1;
    }

    pub fn pop(&mut self) -> Option<u8> {
        if self.depth == 0 {
            None
        } else {
            self.depth -= 1;
            Some(self.slots[self.depth as usize])
        }
    }

    /// Truncate to `new_depth` levels. If `new_depth >= depth`, no-op.
    pub fn truncate(&mut self, new_depth: u8) {
        if new_depth < self.depth {
            self.depth = new_depth;
        }
    }

    pub fn with_truncated(mut self, new_depth: u8) -> Self {
        self.truncate(new_depth);
        self
    }

    /// Length of the longest common prefix between `self` and `other`.
    pub fn common_prefix_len(&self, other: &Self) -> u8 {
        let n = self.depth.min(other.depth) as usize;
        let mut i = 0;
        while i < n && self.slots[i] == other.slots[i] {
            i += 1;
        }
        i as u8
    }

    /// Step one cell along `axis` (0=x, 1=y, 2=z) by `direction` ±1
    /// in the Cartesian interpretation. Bubbles up through parent
    /// cells on overflow.
    ///
    /// At the root, stepping is a no-op (the world does not extend
    /// above the root cell in v1).
    pub fn step_neighbor_cartesian(&mut self, axis: usize, direction: i32) {
        debug_assert!(axis < 3);
        debug_assert!(direction == 1 || direction == -1);
        if self.depth == 0 {
            return;
        }
        let d = self.depth as usize - 1;
        let slot = self.slots[d] as usize;
        let (x, y, z) = slot_coords(slot);
        let mut coords = [x, y, z];
        let v = coords[axis] as i32 + direction;
        if (0..3).contains(&v) {
            coords[axis] = v as usize;
            self.slots[d] = slot_index(coords[0], coords[1], coords[2]) as u8;
        } else {
            // Bubble up: pop, step parent, push the wrapped slot.
            self.depth -= 1;
            self.step_neighbor_cartesian(axis, direction);
            let wrapped = if direction < 0 { 2 } else { 0 };
            coords[axis] = wrapped;
            let new_slot = slot_index(coords[0], coords[1], coords[2]) as u8;
            self.slots[self.depth as usize] = new_slot;
            self.depth += 1;
        }
    }
}

impl Default for Path {
    fn default() -> Self {
        Path::root()
    }
}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth && self.as_slice() == other.as_slice()
    }
}

impl Eq for Path {}

impl Hash for Path {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.depth.hash(state);
        self.as_slice().hash(state);
    }
}

impl std::fmt::Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Path(d={},[", self.depth)?;
        for (i, s) in self.as_slice().iter().enumerate() {
            if i > 0 { write!(f, ",")?; }
            write!(f, "{}", s)?;
        }
        write!(f, "])")
    }
}

// ------------------------------------------------------------ Transition

/// Event fired when a coordinate primitive crosses a meaningful
/// boundary. Game-level handlers react (camera up rotation, UI,
/// etc.); the coordinate math itself is already complete by the
/// time a transition is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    None,
    SphereEntry { body_path: Path },
    SphereExit { body_path: Path },
    FaceEntry { body_path: Path },
    FaceExit { body_path: Path },
    CubeSeam { body_path: Path },
}

// ------------------------------------------------------------ WorldPos

/// `(anchor, offset)` position with the offset held in `[0, 1)³`
/// by invariant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPos {
    pub anchor: Path,
    pub offset: [f32; 3],
}

impl WorldPos {
    pub const fn new_unchecked(anchor: Path, offset: [f32; 3]) -> Self {
        Self { anchor, offset }
    }

    /// Construct and renormalize so the invariant holds.
    pub fn new(anchor: Path, offset: [f32; 3]) -> Self {
        let mut p = Self { anchor, offset };
        p.renormalize_cartesian();
        p
    }

    pub const fn root_origin() -> Self {
        Self { anchor: Path::root(), offset: [0.0, 0.0, 0.0] }
    }

    /// Restore `offset[i] ∈ [0, 1)` by stepping the anchor along
    /// each axis as needed. Cartesian interpretation only (step 1).
    fn renormalize_cartesian(&mut self) {
        for axis in 0..3 {
            // Pull overflow in steps so f32 non-finite inputs don't
            // loop forever — callers are expected to pass finite
            // deltas, but we guard anyway.
            let mut guard: i32 = 0;
            while self.offset[axis] >= 1.0 && guard < 1 << 20 {
                self.offset[axis] -= 1.0;
                self.anchor.step_neighbor_cartesian(axis, 1);
                guard += 1;
            }
            while self.offset[axis] < 0.0 && guard < 1 << 20 {
                self.offset[axis] += 1.0;
                self.anchor.step_neighbor_cartesian(axis, -1);
                guard += 1;
            }
            // Floating-point drift can leave the value at exactly
            // 1.0 after subtraction; clamp back into [0, 1).
            if self.offset[axis] >= 1.0 {
                self.offset[axis] = 1.0 - f32::EPSILON;
            }
            if self.offset[axis] < 0.0 {
                self.offset[axis] = 0.0;
            }
        }
    }

    /// Advance by a local delta (in units of the current cell).
    /// Restores the `[0, 1)` invariant on return. The `_lib`
    /// argument is the future dispatch surface for non-Cartesian
    /// anchors; unused in step 1.
    pub fn add_local(&mut self, delta: [f32; 3], _lib: &NodeLibrary) -> Transition {
        for i in 0..3 {
            self.offset[i] += delta[i];
        }
        self.renormalize_cartesian();
        Transition::None
    }

    /// Push a new slot under the current offset. Offset rescaled so
    /// the world position is unchanged.
    pub fn zoom_in(&mut self) -> Transition {
        let mut coords = [0usize; 3];
        for i in 0..3 {
            let s = (self.offset[i] * 3.0).floor();
            coords[i] = s.clamp(0.0, 2.0) as usize;
            self.offset[i] = (self.offset[i] * 3.0 - s).clamp(0.0, 1.0 - f32::EPSILON);
        }
        let slot = slot_index(coords[0], coords[1], coords[2]) as u8;
        self.anchor.push(slot);
        Transition::None
    }

    /// Pop the deepest slot. Offset rescaled so the world position
    /// is unchanged. Clamps at root.
    pub fn zoom_out(&mut self) -> Transition {
        let Some(slot) = self.anchor.pop() else { return Transition::None; };
        let (sx, sy, sz) = slot_coords(slot as usize);
        self.offset[0] = (self.offset[0] + sx as f32) / 3.0;
        self.offset[1] = (self.offset[1] + sy as f32) / 3.0;
        self.offset[2] = (self.offset[2] + sz as f32) / 3.0;
        // Division by 3 of a value in [0, 1) lands in [0, 1); the
        // only worry is +sx with sx=2 giving (offset+2)/3 which
        // stays < 1 as long as offset < 1. Assert in debug.
        debug_assert!(self.offset.iter().all(|&x| (0.0..1.0).contains(&x)));
        Transition::None
    }
}

// --------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn lib() -> NodeLibrary {
        NodeLibrary::default()
    }

    #[test]
    fn path_root_and_push_pop() {
        let mut p = Path::root();
        assert_eq!(p.depth(), 0);
        assert!(p.is_root());
        p.push(13);
        p.push(5);
        assert_eq!(p.depth(), 2);
        assert_eq!(p.as_slice(), &[13, 5]);
        assert_eq!(p.pop(), Some(5));
        assert_eq!(p.pop(), Some(13));
        assert_eq!(p.pop(), None);
    }

    #[test]
    fn path_eq_and_hash() {
        use std::collections::hash_map::DefaultHasher;
        let mut a = Path::root();
        a.push(1);
        a.push(2);
        let mut b = Path::root();
        b.push(1);
        b.push(2);
        // c shares prefix but differs in depth.
        let mut c = Path::root();
        c.push(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let hash = |p: &Path| -> u64 {
            let mut h = DefaultHasher::new();
            p.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&a), hash(&b));
    }

    #[test]
    fn common_prefix() {
        let mut a = Path::root();
        let mut b = Path::root();
        for s in [1u8, 2, 3, 4] { a.push(s); }
        for s in [1u8, 2, 9, 0] { b.push(s); }
        assert_eq!(a.common_prefix_len(&b), 2);
    }

    #[test]
    fn step_neighbor_within_cell() {
        // At depth 2 starting at slot (1,1,1), step +x -> (2,1,1).
        let mut p = Path::root();
        p.push(0);
        p.push(slot_index(1, 1, 1) as u8);
        p.step_neighbor_cartesian(0, 1);
        assert_eq!(p.slot(1), slot_index(2, 1, 1) as u8);
        p.step_neighbor_cartesian(0, -1);
        assert_eq!(p.slot(1), slot_index(1, 1, 1) as u8);
    }

    #[test]
    fn step_neighbor_bubbles_up() {
        // Depth 2 at (0, 0, 0) within parent (0, 0, 0). Step -x
        // should bubble up; parent is already at x=0 of root, so
        // root step is clamped (no-op), and the child slot should
        // be rewritten as if we crossed the boundary (to x=2).
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        p.push(slot_index(0, 1, 1) as u8);
        p.step_neighbor_cartesian(0, -1);
        // Parent stepped from (1,1,1) to (0,1,1); child wrapped to (2,1,1).
        assert_eq!(p.slot(0), slot_index(0, 1, 1) as u8);
        assert_eq!(p.slot(1), slot_index(2, 1, 1) as u8);
    }

    #[test]
    fn zoom_round_trip() {
        let anchor = {
            let mut p = Path::root();
            p.push(5);
            p
        };
        let mut pos = WorldPos::new(anchor, [0.25, 0.5, 0.75]);
        let before = pos;
        pos.zoom_in();
        assert_eq!(pos.anchor.depth(), 2);
        pos.zoom_out();
        assert_eq!(pos.anchor, before.anchor);
        for i in 0..3 {
            assert!((pos.offset[i] - before.offset[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn zoom_in_preserves_invariant() {
        // Offset at 1-eps corners still ends up in [0, 1).
        let anchor = Path::root();
        let mut pos = WorldPos::new(anchor, [1.0 - f32::EPSILON; 3]);
        pos.zoom_in();
        for v in pos.offset.iter() {
            assert!((0.0..1.0).contains(v), "offset {} out of range", v);
        }
    }

    #[test]
    fn zoom_out_at_root_is_noop() {
        let mut pos = WorldPos::new(Path::root(), [0.1, 0.2, 0.3]);
        let before = pos;
        pos.zoom_out();
        assert_eq!(pos, before);
    }

    #[test]
    fn add_local_small_delta() {
        let l = lib();
        let mut pos = WorldPos::new(Path::root(), [0.5, 0.5, 0.5]);
        let t = pos.add_local([0.1, 0.0, 0.0], &l);
        assert_eq!(t, Transition::None);
        assert!((pos.offset[0] - 0.6).abs() < 1e-5);
        assert_eq!(pos.anchor, Path::root());
    }

    #[test]
    fn add_local_crosses_cell_boundary() {
        let l = lib();
        // At depth 1, slot = (1,1,1), offset near x=1. Step +x.
        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        let mut pos = WorldPos::new(anchor, [0.9, 0.5, 0.5]);
        pos.add_local([0.2, 0.0, 0.0], &l);
        assert_eq!(pos.anchor.slot(0), slot_index(2, 1, 1) as u8);
        assert!((pos.offset[0] - 0.1).abs() < 1e-4);
    }

    #[test]
    fn add_local_bubbles_up_parent() {
        let l = lib();
        // Depth 2; child at (2,1,1) of parent (1,1,1). Step +x
        // overflows child; parent becomes (2,1,1); child becomes (0,1,1).
        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        anchor.push(slot_index(2, 1, 1) as u8);
        let mut pos = WorldPos::new(anchor, [0.9, 0.5, 0.5]);
        pos.add_local([0.2, 0.0, 0.0], &l);
        assert_eq!(pos.anchor.slot(0), slot_index(2, 1, 1) as u8);
        assert_eq!(pos.anchor.slot(1), slot_index(0, 1, 1) as u8);
        assert!((pos.offset[0] - 0.1).abs() < 1e-4);
    }

    #[test]
    fn add_local_large_negative_delta() {
        let l = lib();
        // Step back across two cells.
        let mut anchor = Path::root();
        anchor.push(slot_index(2, 1, 1) as u8);
        let mut pos = WorldPos::new(anchor, [0.1, 0.5, 0.5]);
        pos.add_local([-1.2, 0.0, 0.0], &l);
        // From slot (2,1,1) step back 2 cells -> (0,1,1); offset
        // becomes 0.1 - 1.2 + 2 = 0.9.
        assert_eq!(pos.anchor.slot(0), slot_index(0, 1, 1) as u8);
        assert!((pos.offset[0] - 0.9).abs() < 1e-4);
    }

    #[test]
    fn add_local_offset_is_normalized() {
        let l = lib();
        let mut pos = WorldPos::new(Path::root(), [0.0, 0.0, 0.0]);
        pos.add_local([0.3, 0.7, 0.999], &l);
        for &v in &pos.offset {
            assert!((0.0..1.0).contains(&v));
        }
    }
}
