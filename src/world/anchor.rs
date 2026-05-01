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

use crate::world::tree::{slot_coords, slot_index, NodeId, NodeLibrary, MAX_DEPTH};

/// Local frame convention: every node's children span
/// `[0, WORLD_SIZE)³` because there are 3 children per axis.
/// This is a frame-local coordinate constant, not an absolute
/// world-scale measurement. See `docs/no-absolute-coordinates.md`.
pub const WORLD_SIZE: f32 = 3.0;

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

    /// Kind-aware neighbor step. Currently equivalent to
    /// `step_neighbor_cartesian` (no sphere-aware nodes yet); kept as
    /// the API surface for callers that already pass library/root so
    /// UV-sphere wrap can be hooked in later without churning sites.
    /// Returns `true` iff an in-place wrap fired (always `false` while
    /// every node is Cartesian).
    pub fn step_neighbor_in_world(
        &mut self,
        _library: &NodeLibrary,
        _world_root: NodeId,
        axis: usize,
        direction: i32,
    ) -> bool {
        self.step_neighbor_cartesian(axis, direction);
        false
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

    /// Precise anchor constructed directly as `slot` repeated `depth`
    /// times, with a fixed sub-cell `offset ∈ [0, 1)³`. Bypasses
    /// `from_frame_local + deepened_to` for callers that need an
    /// exact ternary-rational world position at deep anchors.
    ///
    /// # Why this exists
    ///
    /// `from_frame_local` decomposes a world xyz into `(path, offset)`
    /// by walking depth levels and rounding. For a world position
    /// like `1.5` — which has the *infinite* ternary expansion
    /// `0.1̄₃` — the expected decomposition is `path = (1,1,1)^depth,
    /// offset = 0.5`. But the intermediate `(1.5 - 4/3) / (1/3)`
    /// rounds to `≈ 0.5 − 1e-7` in f32, and each `zoom_in` triples
    /// that error (`new = old·3 − 1`), so by about depth 15 the
    /// offset dips below `1/3` and the next slot flips from center
    /// (13) to corner (0). The resulting anchor is wrong.
    ///
    /// This constructor sidesteps the problem: it pushes the same
    /// slot `depth` times — one exact operation, no f32 arithmetic on
    /// the offset across levels.
    pub fn uniform_column(slot: u8, depth: u8, offset: [f32; 3]) -> Self {
        debug_assert!((slot as usize) < 27, "slot must be < 27");
        debug_assert!((depth as usize) <= MAX_DEPTH, "depth exceeds MAX_DEPTH");
        let mut anchor = Path::root();
        for _ in 0..depth {
            anchor.push(slot);
        }
        Self::new(anchor, offset)
    }

    /// Restore `offset[i] ∈ [0, 1)` by stepping the anchor along
    /// each axis as needed. Cartesian interpretation only — used at
    /// construction (where no library/world_root is available) and
    /// as a fallback for kind-agnostic callers.
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

    /// Kind-aware renormalize. Currently delegates to
    /// `renormalize_cartesian` (every node is Cartesian); kept as the
    /// API surface so UV-sphere wrap can hook in later without
    /// churning sites.
    fn renormalize_world(
        &mut self,
        _library: &NodeLibrary,
        _world_root: NodeId,
    ) -> Transition {
        self.renormalize_cartesian();
        Transition::None
    }

    /// Advance by a local delta (in units of the current cell).
    /// Restores the `[0, 1)` invariant via `renormalize_world`.
    pub fn add_local(
        &mut self,
        delta: [f32; 3],
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        for i in 0..3 {
            self.offset[i] += delta[i];
        }
        self.renormalize_world(library, world_root)
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

    /// Vector from `other` to `self`, computed
    /// without ever materializing either position at world scale.
    ///
    /// **Precision-safe.** Both positions are walked in their
    /// common ancestor cell's frame, and the subtraction happens
    /// at that local scale — so the difference's f32 precision is
    /// bounded by the common ancestor cell size, not by the root.
    /// When `self` and `other` share a deep prefix (camera near a
    /// tracked entity), precision improves geometrically with that
    /// shared depth.
    ///
    /// Returns `self - other` in root-frame-equivalent units.
    pub fn offset_from(&self, other: &Self) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(&other.anchor) as usize;
        // Cell size of the common ancestor in world units.
        let mut common_size = WORLD_SIZE;
        for _ in 0..c {
            common_size /= 3.0;
        }
        let walk = |p: &Self| -> [f32; 3] {
            let mut pos = [0.0f32; 3];
            let mut size = common_size;
            for k in c..(p.anchor.depth() as usize) {
                let (sx, sy, sz) = slot_coords(p.anchor.slot(k) as usize);
                let child = size / 3.0;
                pos[0] += sx as f32 * child;
                pos[1] += sy as f32 * child;
                pos[2] += sz as f32 * child;
                size = child;
            }
            pos[0] += p.offset[0] * size;
            pos[1] += p.offset[1] * size;
            pos[2] += p.offset[2] * size;
            pos
        };
        let a = walk(self);
        let b = walk(other);
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Zoom this position in repeatedly until its anchor reaches
    /// `target_depth`. Precision-safe: each step is pure path-slot
    /// arithmetic, no absolute-XYZ accumulation.
    pub fn deepened_to(mut self, target_depth: u8) -> Self {
        while self.anchor.depth() < target_depth {
            self.zoom_in();
        }
        self
    }

    /// Position expressed in `frame`'s local coordinate system,
    /// where the frame's cell spans `[0, WORLD_SIZE)³`.
    ///
    /// Handles any relationship between `self.anchor` and `frame`:
    /// - If `frame` is a prefix of `self.anchor` (the usual render
    ///   case for the camera): the result is precise sub-cell
    ///   precision via pure path-slot composition in the tail —
    ///   f32 magnitudes stay bounded by `WORLD_SIZE` regardless of
    ///   how deep the anchor sits.
    /// - If the two diverge: both positions are computed in their
    ///   common ancestor's frame (each in `[0, WORLD_SIZE)`) and
    ///   the difference is scaled into the frame's local system.
    ///   The result lands outside `[0, WORLD_SIZE)` when `self` is
    ///   outside the frame's cell — correct behavior for shader
    ///   inputs like a distant planet center.
    pub fn in_frame(&self, frame: &Path) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(frame) as usize;

        // Walk [c..self.anchor.depth) + offset → position in the
        // common ancestor's frame (spans [0, WORLD_SIZE)).
        let mut pos_common = [0.0f32; 3];
        let mut size = WORLD_SIZE;
        for k in c..(self.anchor.depth() as usize) {
            let (sx, sy, sz) = slot_coords(self.anchor.slot(k) as usize);
            let child = size / 3.0;
            pos_common[0] += sx as f32 * child;
            pos_common[1] += sy as f32 * child;
            pos_common[2] += sz as f32 * child;
            size = child;
        }
        pos_common[0] += self.offset[0] * size;
        pos_common[1] += self.offset[1] * size;
        pos_common[2] += self.offset[2] * size;

        // Walk [c..frame.depth) → frame's min corner in the common
        // ancestor's frame, plus the frame's cell size there.
        let mut frame_origin = [0.0f32; 3];
        let mut frame_size = WORLD_SIZE;
        for k in c..(frame.depth() as usize) {
            let (sx, sy, sz) = slot_coords(frame.slot(k) as usize);
            let child = frame_size / 3.0;
            frame_origin[0] += sx as f32 * child;
            frame_origin[1] += sy as f32 * child;
            frame_origin[2] += sz as f32 * child;
            frame_size = child;
        }

        // Transform common-ancestor coords → frame-local: the
        // frame's cell is `[frame_origin, frame_origin + frame_size)`
        // in the common ancestor, and `[0, WORLD_SIZE)` in frame
        // local. When `c == frame.depth()` (frame is prefix of
        // anchor) this reduces to the identity, i.e.
        // `pos_common - 0 * 1 = pos_common` — the precise tail walk.
        let scale = WORLD_SIZE / frame_size;
        [
            (pos_common[0] - frame_origin[0]) * scale,
            (pos_common[1] - frame_origin[1]) * scale,
            (pos_common[2] - frame_origin[2]) * scale,
        ]
    }

    /// Build a `WorldPos` at `anchor_depth` whose frame-local
    /// coordinate under `frame` equals `xyz`. Inverse of
    /// `in_frame`. Used when scroll-zoom reconstructs the camera
    /// after a frame-local dolly.
    pub fn from_frame_local(frame: &Path, xyz: [f32; 3], anchor_depth: u8) -> Self {
        debug_assert!(anchor_depth >= frame.depth());
        let clamped = [
            xyz[0].clamp(0.0, WORLD_SIZE - f32::EPSILON),
            xyz[1].clamp(0.0, WORLD_SIZE - f32::EPSILON),
            xyz[2].clamp(0.0, WORLD_SIZE - f32::EPSILON),
        ];
        let mut anchor = *frame;
        let mut origin = [0.0f32; 3];
        let mut size = WORLD_SIZE;
        for _ in frame.depth()..anchor_depth {
            let child = size / 3.0;
            let mut s = [0usize; 3];
            for i in 0..3 {
                let v = ((clamped[i] - origin[i]) / child).floor().clamp(0.0, 2.0) as usize;
                s[i] = v;
                origin[i] += v as f32 * child;
            }
            anchor.push(slot_index(s[0], s[1], s[2]) as u8);
            size = child;
        }
        let offset = [
            ((clamped[0] - origin[0]) / size).clamp(0.0, 1.0 - f32::EPSILON),
            ((clamped[1] - origin[1]) / size).clamp(0.0, 1.0 - f32::EPSILON),
            ((clamped[2] - origin[2]) / size).clamp(0.0, 1.0 - f32::EPSILON),
        ];
        Self { anchor, offset }
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

    /// Sentinel "no real world" root for the kind-agnostic path.
    /// `node_kind_at_depth` returns `None` for missing nodes, so
    /// `add_local` falls through to the Cartesian bubble — equivalent
    /// to the pre-Phase-2 behavior. Used by the existing tests below.
    const NO_ROOT: crate::world::tree::NodeId = 0;

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
        let t = pos.add_local([0.1, 0.0, 0.0], &l, NO_ROOT);
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
        pos.add_local([0.2, 0.0, 0.0], &l, NO_ROOT);
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
        pos.add_local([0.2, 0.0, 0.0], &l, NO_ROOT);
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
        pos.add_local([-1.2, 0.0, 0.0], &l, NO_ROOT);
        // From slot (2,1,1) step back 2 cells -> (0,1,1); offset
        // becomes 0.1 - 1.2 + 2 = 0.9.
        assert_eq!(pos.anchor.slot(0), slot_index(0, 1, 1) as u8);
        assert!((pos.offset[0] - 0.9).abs() < 1e-4);
    }

    // ---- zoom / position preservation tests ----
    // These use in_frame(&Path::root()) to verify position is
    // unchanged — equivalent to the old to_world_xyz() at shallow
    // depths where f32 is precise.

    #[test]
    fn zoom_preserves_position() {
        let mut p = WorldPos::from_frame_local(&Path::root(), [1.23, 2.34, 0.56], 5);
        let before = p.in_frame(&Path::root());
        p.zoom_in();
        let after_in = p.in_frame(&Path::root());
        for i in 0..3 {
            assert!((before[i] - after_in[i]).abs() < 1e-4);
        }
        p.zoom_out();
        let after_out = p.in_frame(&Path::root());
        for i in 0..3 {
            assert!((before[i] - after_out[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn zoom_in_then_zoom_out_preserves_position() {
        let mut p = WorldPos::from_frame_local(&Path::root(), [1.234, 2.345, 0.567], 4);
        let before = p.in_frame(&Path::root());
        for _ in 0..16 { p.zoom_in(); }
        for _ in 0..16 { p.zoom_out(); }
        let after = p.in_frame(&Path::root());
        for i in 0..3 {
            assert!((after[i] - before[i]).abs() < 1e-4,
                "axis {}: {} -> {}", i, before[i], after[i]);
        }
    }

    #[test]
    fn many_zoom_ins_preserve_position() {
        let mut p = WorldPos::from_frame_local(&Path::root(), [1.234, 2.345, 0.567], 4);
        let before = p.in_frame(&Path::root());
        for k in 0..15 {
            p.zoom_in();
            let after = p.in_frame(&Path::root());
            for i in 0..3 {
                assert!((after[i] - before[i]).abs() < 1e-4,
                    "after {} zoom_ins, axis {}: {} -> {}",
                    k + 1, i, before[i], after[i]);
            }
        }
    }

    #[test]
    fn deepened_to_preserves_position() {
        let p = WorldPos::from_frame_local(&Path::root(), [1.234, 2.345, 0.567], 4);
        let before = p.in_frame(&Path::root());
        for d in [4u8, 6, 8, 12] {
            let q = p.deepened_to(d);
            let after = q.in_frame(&Path::root());
            for i in 0..3 {
                assert!((before[i] - after[i]).abs() < 1e-4,
                    "depth {}: axis {}: {} vs {}", d, i, before[i], after[i]);
            }
        }
    }

    // ---- in_frame tests ----

    #[test]
    fn in_frame_at_root_gives_expected_coords() {
        // At shallow depth, root-frame-local coords match the input.
        let p = WorldPos::from_frame_local(&Path::root(), [1.5, 2.25, 0.75], 7);
        let local = p.in_frame(&Path::root());
        assert!((local[0] - 1.5).abs() < 1e-4);
        assert!((local[1] - 2.25).abs() < 1e-4);
        assert!((local[2] - 0.75).abs() < 1e-4);
    }

    #[test]
    fn in_frame_round_trip_via_from_frame_local() {
        let p = WorldPos::from_frame_local(&Path::root(), [1.5, 2.1, 0.9], 12);
        let mut frame = p.anchor;
        frame.truncate(frame.depth() - 3);
        let local = p.in_frame(&frame);
        let q = WorldPos::from_frame_local(&frame, local, p.anchor.depth());
        // Both should project to the same root-frame coords.
        let back = q.in_frame(&Path::root());
        let orig = p.in_frame(&Path::root());
        for i in 0..3 {
            assert!((back[i] - orig[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn in_frame_cross_branch() {
        // Point and frame in different depth-1 branches: the returned
        // local coords fall outside [0, WORLD_SIZE) because the point
        // is outside the frame's cell.
        let point = WorldPos::from_frame_local(&Path::root(), [2.5, 0.25, 1.5], 4);
        let mut frame = Path::root();
        frame.push(slot_index(0, 2, 0) as u8); // depth-1 cell (0, 2, 0)
        frame.push(slot_index(1, 1, 1) as u8); // depth-2 center within it
        let actual = point.in_frame(&frame);
        // Point is at (2.5, 0.25, 1.5) in root frame.
        // Frame cell origin is (0+1/3, 2+1/3, 0+1/3) = (1/3, 7/3, 1/3) in root frame.
        // Frame cell size is 1/9 in root frame.
        // Frame local = (root_pos - frame_origin) / frame_cell_size * WORLD_SIZE
        // The exact values depend on slot arithmetic, but the point should
        // be well outside [0, WORLD_SIZE) on at least one axis.
        assert!(actual[0] > WORLD_SIZE || actual[1] < 0.0,
            "cross-branch point should be outside frame bounds: {:?}", actual);
    }

    #[test]
    fn in_frame_precision_at_deep_anchor() {
        // Construct at shallow depth then deepen — the frame-local
        // coord should stay within [0, WORLD_SIZE) since the anchor
        // shares a deep prefix with the frame.
        let p = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4)
            .deepened_to(18);
        let mut frame = p.anchor;
        frame.truncate(frame.depth() - 3);
        let local = p.in_frame(&frame);
        for &v in &local {
            assert!((0.0..super::WORLD_SIZE).contains(&v), "local {v} out of frame");
        }
    }

    // ---- offset_from tests ----

    #[test]
    fn offset_from_consistent_across_depths() {
        // Construct at shallow depth then deepen via zoom_in (always
        // precise). offset_from should give the same result regardless
        // of anchor depth.
        let planet = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
        let cam_shallow = WorldPos::from_frame_local(&Path::root(), [1.5, 2.32, 1.5], 4);
        let baseline = cam_shallow.offset_from(&planet);
        assert!((baseline[1] - 0.82).abs() < 1e-4);
        for d in [4u8, 8, 12, 16, 20] {
            let cam = cam_shallow.deepened_to(d);
            let oc = cam.offset_from(&planet);
            for i in 0..3 {
                assert!((oc[i] - baseline[i]).abs() < 1e-4,
                    "depth {d}: axis {i}: {} vs baseline {}", oc[i], baseline[i]);
            }
        }
    }

    #[test]
    fn offset_from_after_zoom_chain_matches_baseline() {
        let planet = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
        let cam = WorldPos::from_frame_local(&Path::root(), [1.5, 2.32, 1.5], 4)
            .deepened_to(16);
        let mut zoomed = cam;
        for _ in 0..7 { zoomed.zoom_out(); }
        assert_eq!(zoomed.anchor.depth(), 9);
        let oc_chained = zoomed.offset_from(&planet);
        let oc_deep = cam.offset_from(&planet);
        for i in 0..3 {
            assert!(
                (oc_chained[i] - oc_deep[i]).abs() < 1e-4,
                "axis {}: chained {} vs deep {}",
                i, oc_chained[i], oc_deep[i],
            );
        }
        assert!(oc_chained[1].abs() > 0.5,
            "oc.y collapsed to 0 after zoom chain — sphere would be invisible");
    }

    #[test]
    fn offset_from_self_is_zero() {
        let base = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 4);
        for d in [4u8, 8, 12, 16] {
            let p = base.deepened_to(d);
            let o = p.offset_from(&p);
            for v in o {
                assert!(v.abs() < 1e-6, "depth {}: o = {:?}", d, o);
            }
        }
    }

    #[test]
    fn offset_from_is_antisymmetric() {
        let a = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 8);
        let b = WorldPos::from_frame_local(&Path::root(), [0.5, 1.5, 1.5], 8);
        let ab = a.offset_from(&b);
        let ba = b.offset_from(&a);
        for i in 0..3 {
            assert!((ab[i] + ba[i]).abs() < 1e-5,
                "axis {}: ab={} ba={}", i, ab[i], ba[i]);
        }
    }

    #[test]
    fn offset_from_satisfies_triangle_equality() {
        let a = WorldPos::from_frame_local(&Path::root(), [0.5, 1.5, 1.5], 6);
        let b = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 6);
        let c = WorldPos::from_frame_local(&Path::root(), [2.0, 1.5, 1.5], 6);
        let ac = a.offset_from(&c);
        let ab = a.offset_from(&b);
        let bc = b.offset_from(&c);
        for i in 0..3 {
            let sum = ab[i] + bc[i];
            assert!((ac[i] - sum).abs() < 1e-5,
                "axis {}: ac={} ab+bc={}", i, ac[i], sum);
        }
    }

    #[test]
    fn offset_from_invariant_under_anchor_depth() {
        // Construct at depth 4 then deepen. offset_from should be
        // consistent because deepened_to is pure slot arithmetic.
        let target = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
        let base = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 4);
        let baseline = base.offset_from(&target);
        for depth in [4u8, 6, 8, 12, 16, 20] {
            let p = base.deepened_to(depth);
            let o = p.offset_from(&target);
            for i in 0..3 {
                assert!(
                    (o[i] - baseline[i]).abs() < 1e-5,
                    "depth {}: axis {}: {} vs baseline {}",
                    depth, i, o[i], baseline[i],
                );
            }
        }
    }

    #[test]
    fn deepened_offset_from_matches_base() {
        let target = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
        let base = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 4);
        let base_o = base.offset_from(&target);
        for d in [4u8, 6, 8, 12, 16, 20] {
            let deeper = base.deepened_to(d);
            let o = deeper.offset_from(&target);
            for i in 0..3 {
                assert!((o[i] - base_o[i]).abs() < 1e-5,
                    "depth {}: axis {}: {} vs base {}",
                    d, i, o[i], base_o[i]);
            }
        }
    }

    #[test]
    fn offset_from_matches_in_frame_diff_at_shallow_anchors() {
        // At shallow depth, offset_from(b) should equal the
        // root-frame coordinate difference.
        let a = WorldPos::from_frame_local(&Path::root(), [2.5, 0.25, 1.5], 4);
        let b = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
        let o = a.offset_from(&b);
        let aw = a.in_frame(&Path::root());
        let bw = b.in_frame(&Path::root());
        for i in 0..3 {
            assert!((o[i] - (aw[i] - bw[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn offset_from_precision_at_deep_common_prefix() {
        // Two positions inside the same depth-12 cell — common
        // prefix is 12, so the offset resolves at sub-cell precision
        // even though root-frame coords would lose precision.
        let mut anchor = Path::root();
        for _ in 0..12 { anchor.push(slot_index(1, 1, 1) as u8); }
        let a = WorldPos::new(anchor, [0.30, 0.50, 0.70]);
        let b = WorldPos::new(anchor, [0.20, 0.50, 0.70]);
        let o = a.offset_from(&b);
        let cell = WORLD_SIZE / 3.0f32.powi(12);
        let expected_x = 0.10 * cell;
        assert!((o[0] - expected_x).abs() < cell * 1e-5,
            "diff {} expected {}", o[0], expected_x);
        assert!(o[1].abs() < cell * 1e-5);
        assert!(o[2].abs() < cell * 1e-5);
    }

    #[test]
    fn add_local_offset_is_normalized() {
        let l = lib();
        let mut pos = WorldPos::new(Path::root(), [0.0, 0.0, 0.0]);
        pos.add_local([0.3, 0.7, 0.999], &l, NO_ROOT);
        for &v in &pos.offset {
            assert!((0.0..1.0).contains(&v));
        }
    }
}
