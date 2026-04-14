//! Anchor-based world coordinates.
//!
//! Every world position is `(anchor, offset)`:
//!   - `anchor: Path` — a sequence of slot indices descending the
//!     27-ary tree. Exact, symbolic, no f32 loss at any depth.
//!   - `offset: [f32; 3]` — a small local coordinate inside the
//!     anchor cell's `[0, 1)³` frame.
//!
//! Invariant: `offset[i] ∈ [0, 1)` for each axis. Primitives preserve
//! this across moves and zooms. f32 never accumulates across cells;
//! rendering always happens in a frame small enough for f32.
//!
//! For the full design see
//! `docs/experimental-architecture/anchor-refactor-decisions.md`.

use super::cubesphere::Face;
use super::tree::{slot_coords, slot_index, MAX_DEPTH, NodeLibrary};

// ---------------------------------------------------------------- Path

/// A sequence of slot indices (0..27) descending from the tree root.
///
/// `depth` is the number of live entries in `slots`. `depth == 0` is
/// the root cell (no descent). Each `push(slot)` descends one level.
///
/// Equality is a depth compare plus a memcmp over `slots[..depth]`, so
/// path compares are fast regardless of capacity.
#[derive(Clone, Copy)]
pub struct Path {
    slots: [u8; MAX_DEPTH],
    depth: u8,
}

impl Path {
    pub const fn root() -> Self {
        Self { slots: [0; MAX_DEPTH], depth: 0 }
    }

    #[inline]
    pub fn depth(&self) -> u8 { self.depth }

    #[inline]
    pub fn is_root(&self) -> bool { self.depth == 0 }

    #[inline]
    pub fn slots(&self) -> &[u8] {
        &self.slots[..self.depth as usize]
    }

    /// Descend one level. Returns false if already at `MAX_DEPTH`.
    pub fn push(&mut self, slot: u8) -> bool {
        debug_assert!((slot as usize) < 27, "slot {} out of range", slot);
        if (self.depth as usize) >= MAX_DEPTH {
            return false;
        }
        self.slots[self.depth as usize] = slot;
        self.depth += 1;
        true
    }

    /// Ascend one level. Returns the popped slot, or `None` at root.
    pub fn pop(&mut self) -> Option<u8> {
        if self.depth == 0 { return None; }
        self.depth -= 1;
        Some(self.slots[self.depth as usize])
    }

    /// The slot at the deepest level, or `None` at root.
    #[inline]
    pub fn last_slot(&self) -> Option<u8> {
        if self.depth == 0 { None } else { Some(self.slots[(self.depth - 1) as usize]) }
    }

    /// Truncate to at most `new_depth` levels (no-op if shorter).
    pub fn truncate(&mut self, new_depth: u8) {
        if new_depth < self.depth {
            self.depth = new_depth;
        }
    }

    /// Return a copy truncated to `new_depth`.
    pub fn ancestor(&self, new_depth: u8) -> Path {
        let mut p = *self;
        p.truncate(new_depth);
        p
    }

    /// Length of the common prefix between `self` and `other`.
    pub fn common_prefix_len(&self, other: &Path) -> u8 {
        let n = self.depth.min(other.depth) as usize;
        for i in 0..n {
            if self.slots[i] != other.slots[i] { return i as u8; }
        }
        n as u8
    }

    /// Move one cell along `axis` (0..3) in `dir` (-1 or +1) at the
    /// current depth. On overflow at this level, bubbles up to the
    /// parent and retries, then pushes the mirror slot on the far
    /// side. Returns `false` if clamped at the root (no further
    /// parent exists to bubble into).
    ///
    /// This is the pure Cartesian neighbor step. Sphere-aware stepping
    /// (face seams, radial r axis) will be layered on top of this by
    /// `WorldPos::add_local` once `NodeKind` is introduced.
    pub fn step_neighbor_cartesian(&mut self, axis: u8, dir: i8) -> bool {
        debug_assert!(axis < 3, "axis must be 0..3");
        debug_assert!(dir == 1 || dir == -1, "dir must be ±1");
        if self.depth == 0 { return false; }
        let slot_pos = (self.depth - 1) as usize;
        let s = self.slots[slot_pos] as usize;
        let (cx, cy, cz) = slot_coords(s);
        let mut coords = [cx, cy, cz];
        let c = coords[axis as usize] as i32 + dir as i32;
        if (0..3).contains(&c) {
            coords[axis as usize] = c as usize;
            self.slots[slot_pos] = slot_index(coords[0], coords[1], coords[2]) as u8;
            true
        } else {
            // Overflow; bubble up to parent.
            let saved = coords;
            self.depth -= 1;
            let ok = self.step_neighbor_cartesian(axis, dir);
            if !ok {
                self.depth += 1;
                return false;
            }
            // Re-enter on the far side along `axis`.
            let mut nc = saved;
            nc[axis as usize] = if c < 0 { 2 } else { 0 };
            self.slots[self.depth as usize] =
                slot_index(nc[0], nc[1], nc[2]) as u8;
            self.depth += 1;
            true
        }
    }
}

impl Default for Path {
    fn default() -> Self { Self::root() }
}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.slots[..self.depth as usize] == other.slots[..other.depth as usize]
    }
}

impl Eq for Path {}

impl std::hash::Hash for Path {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.depth.hash(state);
        self.slots[..self.depth as usize].hash(state);
    }
}

impl std::fmt::Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Path[depth={}, slots={:?}]", self.depth, self.slots())
    }
}

// ------------------------------------------------------------ Transition

/// Semantic event emitted when the anchor crosses a coordinate-meaning
/// boundary. Coordinate math itself is handled inside the primitives;
/// these events let game-level code react (camera up-vector rotation,
/// orientation re-expression, UI hints).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    None,
    SphereEntry { body_path: Path },
    SphereExit  { body_path: Path },
    FaceEntry   { face: Face },
    FaceExit    { face: Face },
    CubeSeam    { from_face: Face, to_face: Face },
}

// ------------------------------------------------------------- WorldPos

/// A position anywhere in the world: an anchor path plus a local
/// offset inside the anchor cell's `[0, 1)³` frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPos {
    pub anchor: Path,
    pub offset: [f32; 3],
}

impl WorldPos {
    pub const fn root() -> Self {
        Self { anchor: Path::root(), offset: [0.0, 0.0, 0.0] }
    }

    pub fn new(anchor: Path, offset: [f32; 3]) -> Self {
        let mut p = Self { anchor, offset };
        p.normalize();
        p
    }

    /// Force `offset` into `[0, 1)` by clamping (no step). Callers
    /// should prefer `add_local` which steps across cell boundaries;
    /// this is a safety net after direct field assignment.
    fn normalize(&mut self) {
        for i in 0..3 {
            if !self.offset[i].is_finite() { self.offset[i] = 0.0; }
            if self.offset[i] < 0.0 { self.offset[i] = 0.0; }
            if self.offset[i] >= 1.0 { self.offset[i] = below_one(); }
        }
    }

    /// Add `delta` to `offset`. If any axis overflows `[0, 1)`, steps
    /// to the neighboring cell at the current depth (bubbling up the
    /// path if needed). Returns a `Transition` describing any
    /// coordinate-meaning boundary that was crossed.
    ///
    /// In the current (Cartesian-only) implementation, this always
    /// returns `Transition::None`. Sphere-aware dispatch lands in a
    /// later step once `NodeKind` is wired through the tree.
    pub fn add_local(&mut self, delta: [f32; 3], _lib: &NodeLibrary) -> Transition {
        for i in 0..3 {
            self.offset[i] += delta[i];
        }
        for axis in 0..3usize {
            // Forward overflow.
            while self.offset[axis] >= 1.0 {
                if !self.anchor.step_neighbor_cartesian(axis as u8, 1) {
                    // Root clamp.
                    self.offset[axis] = below_one();
                    break;
                }
                self.offset[axis] -= 1.0;
            }
            // Backward overflow.
            while self.offset[axis] < 0.0 {
                if !self.anchor.step_neighbor_cartesian(axis as u8, -1) {
                    self.offset[axis] = 0.0;
                    break;
                }
                self.offset[axis] += 1.0;
            }
        }
        Transition::None
    }

    /// Descend the anchor into the child slot currently containing the
    /// offset. Offset is rescaled so the world point is unchanged; the
    /// anchor just expresses it in a finer cell. No-op at `MAX_DEPTH`.
    pub fn zoom_in(&mut self) -> Transition {
        if (self.anchor.depth() as usize) >= MAX_DEPTH {
            return Transition::None;
        }
        let sx = pick_slot(self.offset[0]);
        let sy = pick_slot(self.offset[1]);
        let sz = pick_slot(self.offset[2]);
        self.offset[0] = rescale_down(self.offset[0], sx);
        self.offset[1] = rescale_down(self.offset[1], sy);
        self.offset[2] = rescale_down(self.offset[2], sz);
        self.anchor.push(slot_index(sx, sy, sz) as u8);
        Transition::None
    }

    /// Ascend the anchor to its parent. Offset is rescaled to remain
    /// the same world point. No-op at root.
    pub fn zoom_out(&mut self) -> Transition {
        let Some(popped) = self.anchor.pop() else {
            return Transition::None;
        };
        let (sx, sy, sz) = slot_coords(popped as usize);
        self.offset[0] = (self.offset[0] + sx as f32) / 3.0;
        self.offset[1] = (self.offset[1] + sy as f32) / 3.0;
        self.offset[2] = (self.offset[2] + sz as f32) / 3.0;
        Transition::None
    }

    /// Repeatedly `zoom_out` until `anchor.depth() <= target_depth`.
    pub fn zoom_out_to(&mut self, target_depth: u8) {
        while self.anchor.depth() > target_depth {
            self.zoom_out();
        }
    }
}

// ---------------------------------------------------------------- helpers

/// Largest representable f32 strictly less than 1.0.
#[inline]
fn below_one() -> f32 {
    // 1.0 - 2^-24 is the next representable f32 below 1.0.
    // We use a slightly larger gap so math on it stays safely < 1.0.
    1.0 - f32::EPSILON
}

#[inline]
fn pick_slot(v: f32) -> usize {
    ((v * 3.0).floor() as i32).clamp(0, 2) as usize
}

#[inline]
fn rescale_down(v: f32, slot: usize) -> f32 {
    let r = v * 3.0 - slot as f32;
    if r < 0.0 { 0.0 } else if r >= 1.0 { below_one() } else { r }
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Path ----

    #[test]
    fn path_root_is_empty() {
        let p = Path::root();
        assert_eq!(p.depth(), 0);
        assert!(p.is_root());
        assert!(p.slots().is_empty());
    }

    #[test]
    fn path_push_pop() {
        let mut p = Path::root();
        assert!(p.push(5));
        assert!(p.push(12));
        assert_eq!(p.depth(), 2);
        assert_eq!(p.slots(), &[5, 12]);
        assert_eq!(p.last_slot(), Some(12));
        assert_eq!(p.pop(), Some(12));
        assert_eq!(p.pop(), Some(5));
        assert_eq!(p.pop(), None);
    }

    #[test]
    fn path_full_rejects_push() {
        let mut p = Path::root();
        for _ in 0..MAX_DEPTH {
            assert!(p.push(0));
        }
        assert!(!p.push(0), "push at MAX_DEPTH must fail");
    }

    #[test]
    fn path_equality_memcmp_semantics() {
        let mut a = Path::root();
        let mut b = Path::root();
        a.push(1); a.push(2); a.push(3);
        b.push(1); b.push(2); b.push(3);
        assert_eq!(a, b);
        b.pop();
        assert_ne!(a, b);
    }

    #[test]
    fn path_common_prefix() {
        let mut a = Path::root();
        let mut b = Path::root();
        for s in [1u8, 2, 3, 4] { a.push(s); }
        for s in [1u8, 2, 9] { b.push(s); }
        assert_eq!(a.common_prefix_len(&b), 2);
        assert_eq!(a.common_prefix_len(&a), 4);
    }

    #[test]
    fn step_neighbor_same_level() {
        let mut p = Path::root();
        // Start at slot (1,1,1) = 13 one level down.
        p.push(slot_index(1, 1, 1) as u8);
        assert!(p.step_neighbor_cartesian(0, 1)); // +x
        assert_eq!(p.last_slot(), Some(slot_index(2, 1, 1) as u8));
        assert!(p.step_neighbor_cartesian(1, -1)); // -y
        assert_eq!(p.last_slot(), Some(slot_index(2, 0, 1) as u8));
    }

    #[test]
    fn step_neighbor_bubbles_across_parent() {
        // Start at (x=2, y=1, z=1) within parent (x=0, y=0, z=0).
        // Stepping +x must bubble: parent becomes (1,0,0), child
        // reenters at (0, 1, 1).
        let mut p = Path::root();
        p.push(slot_index(0, 0, 0) as u8);
        p.push(slot_index(2, 1, 1) as u8);
        assert!(p.step_neighbor_cartesian(0, 1));
        assert_eq!(p.depth(), 2);
        assert_eq!(p.slots()[0], slot_index(1, 0, 0) as u8);
        assert_eq!(p.slots()[1], slot_index(0, 1, 1) as u8);
    }

    #[test]
    fn step_neighbor_clamps_at_root() {
        // No-op at root: step_neighbor returns false, path unchanged.
        let mut p = Path::root();
        assert!(!p.step_neighbor_cartesian(0, 1));
        assert_eq!(p.depth(), 0);
    }

    #[test]
    fn step_neighbor_clamps_at_edge_of_world() {
        // Single-level path at the far +x edge of root; any +x step
        // attempts to bubble past root and must fail gracefully,
        // leaving the path unchanged.
        let mut p = Path::root();
        p.push(slot_index(2, 0, 0) as u8);
        let before = p;
        assert!(!p.step_neighbor_cartesian(0, 1));
        assert_eq!(p, before);
    }

    // ---- WorldPos ----

    #[test]
    fn world_pos_default_at_root() {
        let p = WorldPos::root();
        assert_eq!(p.anchor.depth(), 0);
        assert_eq!(p.offset, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn add_local_small_delta() {
        let lib = NodeLibrary::default();
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 1, 1) as u8);
        let t = p.add_local([0.1, 0.2, 0.3], &lib);
        assert_eq!(t, Transition::None);
        assert_eq!(p.anchor.depth(), 1);
        assert!((p.offset[0] - 0.1).abs() < 1e-6);
        assert!((p.offset[1] - 0.2).abs() < 1e-6);
        assert!((p.offset[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn add_local_crosses_cell_boundary() {
        let lib = NodeLibrary::default();
        let mut p = WorldPos::root();
        // Anchor at slot (1,1,1); offset near +x edge.
        p.anchor.push(slot_index(1, 1, 1) as u8);
        p.offset = [0.9, 0.5, 0.5];
        p.add_local([0.2, 0.0, 0.0], &lib);
        // Stepped +x one cell: slot now (2,1,1), offset.x wraps to 0.1.
        assert_eq!(p.anchor.last_slot(), Some(slot_index(2, 1, 1) as u8));
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn add_local_bubbles_up_and_back_down() {
        let lib = NodeLibrary::default();
        // Two-deep path at (0,0,0)→(2,1,1); +x crosses parent seam.
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(0, 0, 0) as u8);
        p.anchor.push(slot_index(2, 1, 1) as u8);
        p.offset = [0.9, 0.5, 0.5];
        p.add_local([0.2, 0.0, 0.0], &lib);
        assert_eq!(p.anchor.depth(), 2);
        assert_eq!(p.anchor.slots()[0], slot_index(1, 0, 0) as u8);
        assert_eq!(p.anchor.slots()[1], slot_index(0, 1, 1) as u8);
    }

    #[test]
    fn add_local_negative_crosses_back() {
        let lib = NodeLibrary::default();
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 1, 1) as u8);
        p.offset = [0.05, 0.5, 0.5];
        p.add_local([-0.1, 0.0, 0.0], &lib);
        assert_eq!(p.anchor.last_slot(), Some(slot_index(0, 1, 1) as u8));
        assert!((p.offset[0] - 0.95).abs() < 1e-4);
    }

    #[test]
    fn zoom_in_then_out_preserves_position() {
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 1, 1) as u8);
        p.offset = [0.37, 0.72, 0.45];
        let before = p;
        p.zoom_in();
        assert_eq!(p.anchor.depth(), before.anchor.depth() + 1);
        p.zoom_out();
        assert_eq!(p.anchor.depth(), before.anchor.depth());
        for i in 0..3 {
            assert!((p.offset[i] - before.offset[i]).abs() < 1e-5,
                "axis {i}: got {}, want {}", p.offset[i], before.offset[i]);
        }
    }

    #[test]
    fn zoom_in_picks_correct_slot() {
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(0, 0, 0) as u8);
        p.offset = [0.8, 0.1, 0.5];
        p.zoom_in();
        // 0.8 * 3 = 2.4 → slot 2 on x; 0.1*3 = 0.3 → 0 on y; 0.5*3=1.5 → 1 on z.
        assert_eq!(p.anchor.last_slot(), Some(slot_index(2, 0, 1) as u8));
        assert!((p.offset[0] - 0.4).abs() < 1e-5);
        assert!((p.offset[1] - 0.3).abs() < 1e-5);
        assert!((p.offset[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn zoom_out_at_root_is_noop() {
        let mut p = WorldPos::root();
        p.offset = [0.5, 0.5, 0.5];
        let t = p.zoom_out();
        assert_eq!(t, Transition::None);
        assert_eq!(p.anchor.depth(), 0);
        assert_eq!(p.offset, [0.5, 0.5, 0.5]);
    }
}
