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

/// Root cell spans `[0, WORLD_SIZE)³` in world units.
///
/// Every cell at anchor depth `d` is `WORLD_SIZE / 3^d` wide. The
/// whole engine uses this as the single convention for converting
/// between `WorldPos` and f32 world-space XYZ.
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

    /// World-space XYZ this position represents. Root cell spans
    /// `[0, WORLD_SIZE)³`; each anchor slot narrows that cell by 1/3.
    pub fn to_world_xyz(&self) -> [f32; 3] {
        let mut origin = [0.0f32; 3];
        let mut size = WORLD_SIZE;
        for k in 0..self.anchor.depth() as usize {
            let (sx, sy, sz) = slot_coords(self.anchor.slot(k) as usize);
            let child = size / 3.0;
            origin[0] += sx as f32 * child;
            origin[1] += sy as f32 * child;
            origin[2] += sz as f32 * child;
            size = child;
        }
        [
            origin[0] + self.offset[0] * size,
            origin[1] + self.offset[1] * size,
            origin[2] + self.offset[2] * size,
        ]
    }

    /// Build a `WorldPos` anchored at `anchor_depth` for a world-space
    /// XYZ point. The XYZ is clamped into `[0, WORLD_SIZE)` — positions
    /// outside the root cell are not representable and collapse to the
    /// boundary.
    pub fn from_world_xyz(xyz: [f32; 3], anchor_depth: u8) -> Self {
        let clamped = [
            xyz[0].clamp(0.0, WORLD_SIZE - f32::EPSILON),
            xyz[1].clamp(0.0, WORLD_SIZE - f32::EPSILON),
            xyz[2].clamp(0.0, WORLD_SIZE - f32::EPSILON),
        ];
        let mut anchor = Path::root();
        let mut origin = [0.0f32; 3];
        let mut size = WORLD_SIZE;
        let depth = (anchor_depth as usize).min(MAX_DEPTH);
        for _ in 0..depth {
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

    /// World-space size of the anchor's cell.
    pub fn cell_size(&self) -> f32 {
        WORLD_SIZE / 3.0f32.powi(self.anchor.depth() as i32)
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
    fn world_xyz_round_trip() {
        for depth in [0u8, 1, 3, 7, 12] {
            for xyz in [
                [0.0, 0.0, 0.0],
                [1.5, 2.3, 1.5],
                [2.999, 0.001, 1.0],
                [0.5, 0.5, 0.5],
            ] {
                let p = WorldPos::from_world_xyz(xyz, depth);
                assert_eq!(p.anchor.depth(), depth);
                let back = p.to_world_xyz();
                for i in 0..3 {
                    assert!(
                        (back[i] - xyz[i]).abs() < WORLD_SIZE * 1e-5,
                        "depth {}: xyz {:?} round-tripped to {:?}",
                        depth, xyz, back
                    );
                }
            }
        }
    }

    #[test]
    fn cell_size_matches_depth() {
        let p = WorldPos::from_world_xyz([1.5, 1.5, 1.5], 0);
        assert!((p.cell_size() - WORLD_SIZE).abs() < 1e-5);
        let p = WorldPos::from_world_xyz([1.5, 1.5, 1.5], 1);
        assert!((p.cell_size() - 1.0).abs() < 1e-5);
        let p = WorldPos::from_world_xyz([1.5, 1.5, 1.5], 7);
        assert!((p.cell_size() - (WORLD_SIZE / 3.0f32.powi(7))).abs() < 1e-7);
    }

    #[test]
    fn zoom_preserves_world_xyz() {
        let mut p = WorldPos::from_world_xyz([1.23, 2.34, 0.56], 5);
        let before = p.to_world_xyz();
        p.zoom_in();
        let after_in = p.to_world_xyz();
        for i in 0..3 {
            assert!((before[i] - after_in[i]).abs() < 1e-4);
        }
        p.zoom_out();
        let after_out = p.to_world_xyz();
        for i in 0..3 {
            assert!((before[i] - after_out[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn in_frame_is_identity_at_root_frame() {
        let p = WorldPos::from_world_xyz([1.5, 2.25, 0.75], 7);
        let local = p.in_frame(&Path::root());
        let world = p.to_world_xyz();
        for i in 0..3 {
            assert!((local[i] - world[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn in_frame_round_trip_via_from_frame_local() {
        let p = WorldPos::from_world_xyz([1.5, 2.1, 0.9], 12);
        let mut frame = p.anchor;
        frame.truncate(frame.depth() - 3);
        let local = p.in_frame(&frame);
        let q = WorldPos::from_frame_local(&frame, local, p.anchor.depth());
        let world_back = q.to_world_xyz();
        let world_orig = p.to_world_xyz();
        for i in 0..3 {
            assert!((world_back[i] - world_orig[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn in_frame_cross_branch_matches_world_diff() {
        // Point and frame living in different depth-1 branches: the
        // returned local coords fall outside [0, WORLD_SIZE) and must
        // match (world_pos - frame_world_origin) * (WORLD_SIZE /
        // frame_size) to within f32 precision.
        let point = WorldPos::from_world_xyz([2.5, 0.25, 1.5], 4);
        let mut frame = Path::root();
        frame.push(slot_index(0, 2, 0) as u8); // depth-1 cell (0, 2, 0) = y-top-left
        frame.push(slot_index(1, 1, 1) as u8); // depth-2 center within it
        let frame_size = super::WORLD_SIZE / 3.0f32.powi(frame.depth() as i32);
        let mut frame_origin = [0.0f32; 3];
        let mut size = super::WORLD_SIZE;
        for k in 0..frame.depth() as usize {
            let (sx, sy, sz) = slot_coords(frame.slot(k) as usize);
            let child = size / 3.0;
            frame_origin[0] += sx as f32 * child;
            frame_origin[1] += sy as f32 * child;
            frame_origin[2] += sz as f32 * child;
            size = child;
        }
        let expected = {
            let w = point.to_world_xyz();
            let s = super::WORLD_SIZE / frame_size;
            [
                (w[0] - frame_origin[0]) * s,
                (w[1] - frame_origin[1]) * s,
                (w[2] - frame_origin[2]) * s,
            ]
        };
        let actual = point.in_frame(&frame);
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-3,
                "axis {i}: got {} expected {}", actual[i], expected[i],
            );
        }
    }

    #[test]
    fn in_frame_precision_at_deep_anchor() {
        // At anchor depth 18, absolute world XYZ near 1.5 loses
        // sub-cell precision in f32. The frame-local coord should
        // hit f32-safe magnitudes inside the frame's [0, WORLD_SIZE).
        let p = WorldPos::from_world_xyz([1.5, 1.5, 1.5], 18);
        let mut frame = p.anchor;
        frame.truncate(frame.depth() - 3);
        let local = p.in_frame(&frame);
        for &v in &local {
            assert!((0.0..super::WORLD_SIZE).contains(&v), "local {v} out of frame");
        }
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
