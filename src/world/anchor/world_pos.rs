//! `(anchor, offset)` position with the offset held in `[0, 1)³`
//! by invariant. All zoom / motion / frame-projection arithmetic
//! lives here.

use super::path::{Path, Transition};
use super::WORLD_SIZE;
use crate::world::gpu::inscribed_cube_scale;
use crate::world::tree::{
    slot_coords, slot_index, Child, NodeId, NodeKind, NodeLibrary, IDENTITY_ROTATION, MAX_DEPTH,
};

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
    /// as a fallback for kind-agnostic callers. Runtime motion
    /// (`add_local`) uses `renormalize_world` so wrap fires on
    /// WrappedPlane subtrees.
    fn renormalize_cartesian(&mut self) {
        for axis in 0..3 {
            let mut guard: i32 = 0;
            while self.offset[axis] >= 1.0 && guard < 1 << 20 {
                self.offset[axis] -= 1.0;
                if !self.anchor.step_neighbor_cartesian(axis, 1) {
                    self.offset[axis] = 1.0 - f32::EPSILON;
                    break;
                }
                guard += 1;
            }
            while self.offset[axis] < 0.0 && guard < 1 << 20 {
                self.offset[axis] += 1.0;
                if !self.anchor.step_neighbor_cartesian(axis, -1) {
                    self.offset[axis] = 0.0;
                    break;
                }
                guard += 1;
            }
            if self.offset[axis] >= 1.0 {
                self.offset[axis] = 1.0 - f32::EPSILON;
            }
            if self.offset[axis] < 0.0 {
                self.offset[axis] = 0.0;
            }
        }
    }

    /// Kind-aware renormalize. Cell-local pop + redescend that keeps
    /// the camera's physical position continuous across cell
    /// boundaries, including TangentBlock rotation boundaries and
    /// WrappedPlane wrap boundaries.
    ///
    /// Algorithm:
    /// - Loop: if offset is in `[0, 1)³` and depth meets target, done.
    /// - WrappedPlane wrap: if the deepest cell's parent is a slab
    ///   and offset[0] overflows, increment / decrement the deepest
    ///   slot's X coord (wrapping at slab edges), adjust offset[0],
    ///   stay at same depth.
    /// - Pop: drop the deepest slot. If the cell being popped was a
    ///   `TangentBlock`, apply forward `R` about `(0.5, 0.5, 0.5)`
    ///   to convert offset from TB-storage frame into the parent's
    ///   slot frame (Cartesian world). Then `(slot + offset) / 3`
    ///   to express in the new deepest cell's children frame.
    /// - Redescend: pick slot via `floor(offset * 3)`. If the picked
    ///   child is a `TangentBlock`, apply `R^T` about `(0.5, 0.5, 0.5)`
    ///   to convert the post-floor offset from parent-frame into
    ///   TB-storage so subsequent descents (and the offset's frame)
    ///   are in storage.
    ///
    /// Pure cell-local arithmetic — magnitudes stay ≤ 0.5 in the
    /// rotations regardless of anchor depth, so f32 precision is
    /// bounded by the cell-fraction not the world position.
    fn renormalize_world(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        let target_depth = self.anchor.depth();
        let mut transition = Transition::None;
        let mut guard: u32 = 0;
        let max_guard: u32 = 1 << 20;
        loop {
            guard += 1;
            if guard > max_guard {
                debug_assert!(false, "renormalize_world guard exceeded");
                break;
            }
            let in_range = self.offset[0] >= 0.0 && self.offset[0] < 1.0
                && self.offset[1] >= 0.0 && self.offset[1] < 1.0
                && self.offset[2] >= 0.0 && self.offset[2] < 1.0;
            if in_range && self.anchor.depth() >= target_depth {
                break;
            }
            if !in_range {
                if self.anchor.depth() == 0 {
                    // Hit root, clamp.
                    for i in 0..3 {
                        if self.offset[i] >= 1.0 {
                            self.offset[i] = 1.0 - f32::EPSILON;
                        } else if self.offset[i] < 0.0 {
                            self.offset[i] = 0.0;
                        }
                    }
                    break;
                }
                // Try WrappedPlane wrap on axis 0 if the deepest
                // cell's parent is a slab.
                let depth = self.anchor.depth() as usize;
                let parent_slots_len = depth - 1;
                let parent_kind = super::path::node_kind_at_depth(
                    library,
                    world_root,
                    &self.anchor.as_slice()[..parent_slots_len],
                );
                if let Some(NodeKind::WrappedPlane { .. }) = parent_kind {
                    let last_slot = self.anchor.slot(depth - 1) as usize;
                    let (sx, sy, sz) = slot_coords(last_slot);
                    if self.offset[0] >= 1.0 {
                        self.offset[0] -= 1.0;
                        let new_sx = if sx < 2 { sx + 1 } else { 0 };
                        if new_sx == 0 {
                            transition = Transition::WrappedPlaneWrap { axis: 0 };
                        }
                        self.anchor.pop();
                        self.anchor.push(slot_index(new_sx, sy, sz) as u8);
                        continue;
                    }
                    if self.offset[0] < 0.0 {
                        self.offset[0] += 1.0;
                        let new_sx = if sx > 0 { sx - 1 } else { 2 };
                        if new_sx == 2 {
                            transition = Transition::WrappedPlaneWrap { axis: 0 };
                        }
                        self.anchor.pop();
                        self.anchor.push(slot_index(new_sx, sy, sz) as u8);
                        continue;
                    }
                    // Y or Z overflow: pop normally below.
                }
                // Pop one level (TangentBlock-rotation-aware).
                self.pop_one_level_rot_aware(library, world_root);
            } else {
                // In range but depth < target. Try to redescend.
                // Returns false when the camera physically sits in
                // a TB's parent-slot corner outside the inscribed
                // content — in which case the anchor must stay
                // shallower than `target_depth`.
                if !self.descend_one_level_rot_aware(library, world_root) {
                    break;
                }
            }
        }
        transition
    }

    /// Pop the deepest slot. If the cell being popped was a
    /// `TangentBlock`, apply forward `R · tb_scale` about
    /// `(0.5, 0.5, 0.5)` to convert the offset from TB-storage
    /// (rotated, inscribed-shrunk) frame into the parent-of-TB
    /// Cartesian frame BEFORE the standard
    /// `(slot_offset + offset) / 3` rescaling.
    fn pop_one_level_rot_aware(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) {
        if self.anchor.depth() == 0 {
            return;
        }
        // Kind of the cell currently at the end of the path (the
        // cell being popped).
        let popped_kind = super::path::node_kind_at_depth(
            library, world_root, self.anchor.as_slice(),
        );
        let mut adjusted = self.offset;
        if let Some(NodeKind::TangentBlock { rotation: r }) = popped_kind {
            let tb_scale = inscribed_cube_scale(&r);
            let centred = [adjusted[0] - 0.5, adjusted[1] - 0.5, adjusted[2] - 0.5];
            let rotated = mat3_mul_vec3(&r, &centred);
            // tb_scale ≤ 1 so this stays within [0, 1)³.
            adjusted[0] = rotated[0] * tb_scale + 0.5;
            adjusted[1] = rotated[1] * tb_scale + 0.5;
            adjusted[2] = rotated[2] * tb_scale + 0.5;
        }
        let last_slot = self.anchor.pop().unwrap_or(0) as usize;
        let (sx, sy, sz) = slot_coords(last_slot);
        self.offset[0] = (sx as f32 + adjusted[0]) / 3.0;
        self.offset[1] = (sy as f32 + adjusted[1]) / 3.0;
        self.offset[2] = (sz as f32 + adjusted[2]) / 3.0;
    }

    /// Descend one level: pick the slot containing the camera and
    /// push it onto the anchor. If the picked child is a
    /// `TangentBlock`, apply `R^T / tb_scale` about `(0.5, 0.5, 0.5)`
    /// to the post-floor offset to convert from parent-frame into
    /// TB-storage frame.
    ///
    /// Returns `false` if the would-be new offset falls outside
    /// `[0, 1)³` after the `/tb_scale` expansion (camera physically
    /// in the parent slot's corner outside the inscribed TB content).
    /// On false the descent is undone — slot popped back, offset
    /// untouched — so the caller can stop descending and accept the
    /// shallower anchor.
    fn descend_one_level_rot_aware(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> bool {
        let storage_pos = [
            self.offset[0] * 3.0,
            self.offset[1] * 3.0,
            self.offset[2] * 3.0,
        ];
        let sx = storage_pos[0].floor().clamp(0.0, 2.0) as i32 as usize;
        let sy = storage_pos[1].floor().clamp(0.0, 2.0) as i32 as usize;
        let sz = storage_pos[2].floor().clamp(0.0, 2.0) as i32 as usize;
        let slot = slot_index(sx, sy, sz) as u8;
        // Push first, then look up the descended-into cell's kind.
        self.anchor.push(slot);
        let descend_kind = super::path::node_kind_at_depth(
            library, world_root, self.anchor.as_slice(),
        );
        let mut new_offset = [
            (storage_pos[0] - sx as f32).clamp(0.0, 1.0 - f32::EPSILON),
            (storage_pos[1] - sy as f32).clamp(0.0, 1.0 - f32::EPSILON),
            (storage_pos[2] - sz as f32).clamp(0.0, 1.0 - f32::EPSILON),
        ];
        if let Some(NodeKind::TangentBlock { rotation: r }) = descend_kind {
            let tb_scale = inscribed_cube_scale(&r);
            // R^T · (offset - 0.5) / tb_scale + 0.5
            let centred = [new_offset[0] - 0.5, new_offset[1] - 0.5, new_offset[2] - 0.5];
            let rotated = [
                r[0][0] * centred[0] + r[0][1] * centred[1] + r[0][2] * centred[2],
                r[1][0] * centred[0] + r[1][1] * centred[1] + r[1][2] * centred[2],
                r[2][0] * centred[0] + r[2][1] * centred[1] + r[2][2] * centred[2],
            ];
            let candidate = [
                rotated[0] / tb_scale + 0.5,
                rotated[1] / tb_scale + 0.5,
                rotated[2] / tb_scale + 0.5,
            ];
            // If `/tb_scale` pushed the offset outside [0, 1)³, the
            // camera is physically in the parent slot's corner that
            // sits outside the inscribed TB content. Refuse the
            // descent — the anchor must stay at the parent depth.
            let in_range = candidate.iter().all(|&x| (0.0..1.0).contains(&x));
            if !in_range {
                self.anchor.pop();
                return false;
            }
            new_offset = candidate;
        }
        self.offset = new_offset;
        true
    }

    /// Advance by a local delta (in units of the current cell).
    /// Restores the `[0, 1)` invariant via `renormalize_world` so
    /// motion across a `WrappedPlane` boundary wraps modulo the
    /// slab dims rather than bubbling out of the slab subtree.
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

    /// Tree-aware zoom: like `zoom_in`, but when the offset lies
    /// near a cell boundary AND the geometric pick lands on
    /// Empty/Block while a neighboring cell has a Node, pick the
    /// neighbor. This biases tie-breaks at boundaries toward slots
    /// with carved structure so the anchor follows the tree's
    /// content rather than drifting into adjacent empty space.
    ///
    /// World position shift is bounded by `BOUNDARY_EPS * cell_size`
    /// — imperceptible in gameplay but enough to keep the render
    /// path on the carved branch when the camera is right on a wall.
    pub fn zoom_in_in_world(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        const BOUNDARY_EPS: f32 = 1e-3;
        // Resolve the parent node by walking the anchor in the tree.
        let parent = node_at_path(library, world_root, &self.anchor);
        // Naive geometric pick (in current deepest cell's children frame).
        let mut coords = [0usize; 3];
        let mut new_offset = [0.0f32; 3];
        for i in 0..3 {
            let s = (self.offset[i] * 3.0).floor();
            coords[i] = s.clamp(0.0, 2.0) as usize;
            new_offset[i] = (self.offset[i] * 3.0 - s).clamp(0.0, 1.0 - f32::EPSILON);
        }
        // Tree-aware tie-break: per axis, if offset is near 0 or 1
        // (we just crossed a boundary) AND the geometric slot leads
        // to Empty/Block while the boundary-neighbor on this axis
        // is a Node, snap to the neighbor. Cartesian only — the
        // tie-break operates in the current cell's children frame.
        if let Some(parent_id) = parent {
            for axis in 0..3 {
                let near_low = new_offset[axis] < BOUNDARY_EPS && coords[axis] > 0;
                let near_high = new_offset[axis] > 1.0 - BOUNDARY_EPS && coords[axis] < 2;
                if !near_low && !near_high { continue; }
                let cur_is_node = slot_is_node(library, parent_id, coords, axis, 0);
                if cur_is_node { continue; }
                // Only consider a snap if current pick is Empty/Block.
                let alt_dir = if near_low { -1i32 } else { 1 };
                let alt_is_node = slot_is_node(library, parent_id, coords, axis, alt_dir);
                if alt_is_node {
                    // Snap: move to neighbor. Offset on this axis
                    // wraps to the other end of [0, 1).
                    let new_coord = (coords[axis] as i32 + alt_dir) as usize;
                    coords[axis] = new_coord;
                    new_offset[axis] = if alt_dir < 0 {
                        1.0 - f32::EPSILON
                    } else {
                        0.0
                    };
                }
            }
        }
        let slot = slot_index(coords[0], coords[1], coords[2]) as u8;
        self.anchor.push(slot);
        // If the cell we just descended INTO is a TangentBlock, the
        // offset's frame changes from parent-Cartesian to TB-storage.
        // Apply R^T rotation about (0.5, 0.5, 0.5).
        let descend_kind = super::path::node_kind_at_depth(
            library, world_root, self.anchor.as_slice(),
        );
        if let Some(NodeKind::TangentBlock { rotation: r }) = descend_kind {
            let tb_scale = inscribed_cube_scale(&r);
            let centred = [new_offset[0] - 0.5, new_offset[1] - 0.5, new_offset[2] - 0.5];
            let rotated = [
                r[0][0] * centred[0] + r[0][1] * centred[1] + r[0][2] * centred[2],
                r[1][0] * centred[0] + r[1][1] * centred[1] + r[1][2] * centred[2],
                r[2][0] * centred[0] + r[2][1] * centred[1] + r[2][2] * centred[2],
            ];
            // User-explicit zoom: clamp to [0, 1)³ if `/tb_scale`
            // pushes the offset outside the inscribed content. Causes
            // a small world snap (≤ ~0.21·cell_size in offset units),
            // which is acceptable for an explicit user action — the
            // alternative is rejecting the zoom, which is jarring.
            new_offset[0] = (rotated[0] / tb_scale + 0.5).clamp(0.0, 1.0 - f32::EPSILON);
            new_offset[1] = (rotated[1] / tb_scale + 0.5).clamp(0.0, 1.0 - f32::EPSILON);
            new_offset[2] = (rotated[2] / tb_scale + 0.5).clamp(0.0, 1.0 - f32::EPSILON);
        }
        self.offset = new_offset;
        Transition::None
    }

    /// Pop the deepest slot. World position preserved across the
    /// pop, including TangentBlock rotation when the cell being
    /// popped was a TB. Use this at runtime; `zoom_out` is a
    /// Cartesian fallback for callers without a library.
    pub fn zoom_out_in_world(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        if self.anchor.depth() == 0 {
            return Transition::None;
        }
        // Kind of the cell at the end of the path (the cell being
        // popped). If it's a TangentBlock, the offset is in
        // TB-storage frame and must be rotated by R about (0.5, 0.5, 0.5)
        // to express in the parent-of-TB Cartesian frame BEFORE the
        // standard pop rescale.
        let popped_kind = super::path::node_kind_at_depth(
            library, world_root, self.anchor.as_slice(),
        );
        let mut adjusted = self.offset;
        if let Some(NodeKind::TangentBlock { rotation: r }) = popped_kind {
            let tb_scale = inscribed_cube_scale(&r);
            let centred = [adjusted[0] - 0.5, adjusted[1] - 0.5, adjusted[2] - 0.5];
            let rotated = mat3_mul_vec3(&r, &centred);
            // tb_scale ≤ 1 keeps the result in [0, 1)³.
            adjusted[0] = rotated[0] * tb_scale + 0.5;
            adjusted[1] = rotated[1] * tb_scale + 0.5;
            adjusted[2] = rotated[2] * tb_scale + 0.5;
        }
        let last_slot = match self.anchor.pop() {
            Some(s) => s as usize,
            None => return Transition::None,
        };
        let (sx, sy, sz) = slot_coords(last_slot);
        self.offset[0] = (sx as f32 + adjusted[0]) / 3.0;
        self.offset[1] = (sy as f32 + adjusted[1]) / 3.0;
        self.offset[2] = (sz as f32 + adjusted[2]) / 3.0;
        debug_assert!(self.offset.iter().all(|&x| (0.0..1.0).contains(&x)));
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

    /// Rotation-aware variant of [`in_frame`]. When the anchor path
    /// crosses a `NodeKind::TangentBlock` whose stored rotation is
    /// non-identity, every slot offset *past that node* and the final
    /// `offset` are rotated by the cumulative chain rotation around
    /// each enclosing cube's geometric centre. Frame-local throughout
    /// — never world-absolute. For all-Cartesian paths, the result is
    /// bit-identical to [`in_frame`].
    ///
    /// `library` and `world_root` are needed because the rotation
    /// data lives on `Node` records keyed by `NodeId`. Currently
    /// assumes the `frame` path is Cartesian (i.e. no rotated
    /// ancestors above the camera's anchor) — the renderer guarantees
    /// this for shallow worlds. If a rotated frame becomes possible,
    /// extend with frame-side rotation accumulation.
    pub fn in_frame_rot(
        &self,
        library: &NodeLibrary,
        world_root: NodeId,
        frame: &Path,
    ) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(frame) as usize;

        // Walk from world root to the common ancestor, tracking the
        // cumulative rotation experienced going from root → common.
        // (Frames assumed Cartesian; this is identity in the typical
        // case but preserved for future when frame can be rotated.)
        let mut node = world_root;
        let mut common_rot = IDENTITY_ROTATION;
        for k in 0..c {
            let n = match library.get(node) {
                Some(n) => n,
                None => return self.in_frame(frame),
            };
            match n.children[self.anchor.slot(k) as usize] {
                Child::Node(child) => {
                    if let Some(child_node) = library.get(child) {
                        if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                            common_rot = matmul3x3(&common_rot, &r);
                        }
                    }
                    node = child;
                }
                _ => return self.in_frame(frame),
            }
        }
        let common_node = node;

        // Walk from common ancestor down the anchor's tail,
        // accumulating both the cube *centre* in common-ancestor
        // local coords and the cumulative rotation. At each step we
        // rotate the slot's centred-local offset by the cumulative
        // rotation before adding to the centre.
        let mut cur_centre = [WORLD_SIZE * 0.5; 3];
        let mut cur_size = WORLD_SIZE;
        // `cur_rot` starts as identity: the tail walk produces the
        // position in the common ancestor's unrotated local frame.
        // TangentBlock rotation from `common_rot` is handled by the
        // shader (R^T at frame entry for TB frame roots), not here.
        let mut cur_rot = IDENTITY_ROTATION;
        // `cur_scale` tracks the cumulative inscribed-cube shrink
        // factor accumulated through TBs along the tail. Each TB
        // contributes `inscribed_cube_scale(R)` (≈0.707 for 45° Y).
        // The shader applies the same shrink at TB boundaries
        // (rot_col0.w in `GpuNodeKind`) — multiplying here keeps
        // CPU-side world position consistent with what the shader
        // renders for a given (anchor, offset).
        let mut cur_scale: f32 = 1.0;
        let _ = common_rot;
        let mut have_node = true;
        let mut node = common_node;
        for k in c..(self.anchor.depth() as usize) {
            let slot = self.anchor.slot(k);
            let (sx, sy, sz) = slot_coords(slot as usize);
            let child_size = cur_size / 3.0;
            // Slot's centre offset relative to the current node's
            // centre, in current node's LOCAL frame:
            let centred_local = [
                (sx as f32 - 1.0) * child_size,
                (sy as f32 - 1.0) * child_size,
                (sz as f32 - 1.0) * child_size,
            ];
            // Rotate, scale by cumulative TB shrink, then add to centre.
            let rotated = mat3_mul_vec3(&cur_rot, &centred_local);
            cur_centre = [
                cur_centre[0] + rotated[0] * cur_scale,
                cur_centre[1] + rotated[1] * cur_scale,
                cur_centre[2] + rotated[2] * cur_scale,
            ];

            // Update rotation/scale if descending into a TangentBlock.
            // `have_node` tracks whether the path is still inside the
            // tree; once it falls off (Block / Empty / EntityRef)
            // further iterations only contribute slot-position offsets
            // (rotation chain doesn't extend below the tree's nodes).
            if have_node {
                let n = library.get(node).unwrap();
                match n.children[slot as usize] {
                    Child::Node(child_id) => {
                        if let Some(child_node) = library.get(child_id) {
                            if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                                cur_rot = matmul3x3(&cur_rot, &r);
                                cur_scale *= inscribed_cube_scale(&r);
                            }
                            node = child_id;
                        } else {
                            have_node = false;
                        }
                    }
                    _ => have_node = false,
                }
            }
            cur_size = child_size;
        }

        // Apply the offset, also rotated and scaled by the cumulative
        // chain around the anchor cell's centre.
        let centred_offset_local = [
            (self.offset[0] - 0.5) * cur_size,
            (self.offset[1] - 0.5) * cur_size,
            (self.offset[2] - 0.5) * cur_size,
        ];
        let rotated_offset = mat3_mul_vec3(&cur_rot, &centred_offset_local);
        let pos_common = [
            cur_centre[0] + rotated_offset[0] * cur_scale,
            cur_centre[1] + rotated_offset[1] * cur_scale,
            cur_centre[2] + rotated_offset[2] * cur_scale,
        ];

        // Walk frame's tail from common ancestor to find frame's
        // origin + size in common ancestor coords. Frame is assumed
        // Cartesian here; if it becomes rotated, extend.
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

    /// Build a `WorldPos` at `anchor_depth` whose world-frame
    /// coordinates equal `world_xyz`. Walks the tree from the world
    /// root, applying `R^T` about each `TangentBlock`'s centre to
    /// pick the storage-frame slot at every level. Pure function on
    /// the tree state — no `App`.
    ///
    /// This is the rotation-aware inverse of
    /// `in_frame_rot(library, world_root, &Path::root())`, and is
    /// the canonical way to set the camera's anchor when its world
    /// position changes (movement, teleport, edit-driven snap).
    /// Path-level Cartesian step_neighbor inherits source-cell slot
    /// indices that are world-frame, but a TB's children are indexed
    /// in storage frame — so a Cartesian step that pops up across a
    /// TB ancestor places the camera at the wrong storage cell.
    /// Re-deriving the anchor from world coordinates avoids that.
    pub fn from_world_xyz(
        world_xyz: [f32; 3],
        anchor_depth: u8,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Self {
        let clamped = [
            world_xyz[0].clamp(0.0, WORLD_SIZE - f32::EPSILON),
            world_xyz[1].clamp(0.0, WORLD_SIZE - f32::EPSILON),
            world_xyz[2].clamp(0.0, WORLD_SIZE - f32::EPSILON),
        ];
        let mut anchor = Path::root();
        // `pos` lives in the current node's local `[0, WORLD_SIZE)³`
        // frame; for descendants of a TB this is the storage-frame
        // local coordinate after applying `R^T` about the TB centre
        // when we reach the TB level.
        let mut pos = clamped;
        let mut have_node = true;
        let mut node = world_root;

        for _ in 0..anchor_depth {
            // If the current node is a TB, its children are indexed
            // in storage frame: rotate `pos` by `R^T` about (1.5,
            // 1.5, 1.5) before flooring to a slot.
            if have_node {
                if let Some(n) = library.get(node) {
                    if let NodeKind::TangentBlock { rotation: r } = n.kind {
                        let centered = [pos[0] - 1.5, pos[1] - 1.5, pos[2] - 1.5];
                        // R^T · centered (column-major `r[col][row]`).
                        let rotated = [
                            r[0][0] * centered[0] + r[0][1] * centered[1] + r[0][2] * centered[2],
                            r[1][0] * centered[0] + r[1][1] * centered[1] + r[1][2] * centered[2],
                            r[2][0] * centered[0] + r[2][1] * centered[1] + r[2][2] * centered[2],
                        ];
                        pos = [rotated[0] + 1.5, rotated[1] + 1.5, rotated[2] + 1.5];
                    }
                }
            }
            let sx = (pos[0].floor() as i32).clamp(0, 2) as usize;
            let sy = (pos[1].floor() as i32).clamp(0, 2) as usize;
            let sz = (pos[2].floor() as i32).clamp(0, 2) as usize;
            let slot = slot_index(sx, sy, sz);
            anchor.push(slot as u8);
            // Sub-cell position in child's `[0, WORLD_SIZE)³` frame.
            pos = [
                (pos[0] - sx as f32) * 3.0,
                (pos[1] - sy as f32) * 3.0,
                (pos[2] - sz as f32) * 3.0,
            ];
            // Descend into child node so the next iteration can
            // detect a TB at the new level.
            if have_node {
                if let Some(n) = library.get(node) {
                    match n.children[slot] {
                        Child::Node(child) => node = child,
                        _ => have_node = false,
                    }
                } else {
                    have_node = false;
                }
            }
        }

        let offset = [
            (pos[0] / 3.0).clamp(0.0, 1.0 - f32::EPSILON),
            (pos[1] / 3.0).clamp(0.0, 1.0 - f32::EPSILON),
            (pos[2] / 3.0).clamp(0.0, 1.0 - f32::EPSILON),
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

/// 3×3 matrix multiply, both column-major (`r[col][row]`).
fn matmul3x3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for c in 0..3 {
        for r in 0..3 {
            let mut s = 0.0f32;
            for k in 0..3 {
                // (a · b)[r, c] = sum_k a[r, k] · b[k, c]
                // a[r, k] = a[k][r];  b[k, c] = b[c][k]
                s += a[k][r] * b[c][k];
            }
            out[c][r] = s;
        }
    }
    out
}

/// Apply a column-major 3×3 matrix to a 3-vector: `(m · v).i =
/// sum_j m[j][i] · v.j`.
fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Walk `library` from `world_root` along `path`'s slots; return the
/// final NodeId reached (or None if the path leaves the tree).
fn node_at_path(library: &NodeLibrary, world_root: NodeId, path: &Path) -> Option<NodeId> {
    let mut nid = world_root;
    for k in 0..(path.depth() as usize) {
        let node = library.get(nid)?;
        match node.children[path.slot(k) as usize] {
            Child::Node(child) => nid = child,
            _ => return None,
        }
    }
    Some(nid)
}

/// Is `parent`'s child at `coords + axis_delta` a Node?
/// `delta` is along `axis` only (or 0 for "current").
fn slot_is_node(
    library: &NodeLibrary,
    parent: NodeId,
    coords: [usize; 3],
    axis: usize,
    delta: i32,
) -> bool {
    let mut c = coords;
    let new = c[axis] as i32 + delta;
    if new < 0 || new > 2 { return false; }
    c[axis] = new as usize;
    let slot = slot_index(c[0], c[1], c[2]);
    let Some(node) = library.get(parent) else { return false; };
    matches!(node.children[slot], Child::Node(_))
}
