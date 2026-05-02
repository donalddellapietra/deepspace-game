//! `(anchor, offset)` position with the offset held in `[0, 1)³`
//! by invariant.
//!
//! # Rotation-aware semantics
//!
//! `offset` is a continuous position within the deepest cell on the
//! `anchor` path, expressed in **the deepest cell's parent's internal
//! frame** (the frame in which the deepest slot index is interpreted).
//!
//! For a path that passes through a `NodeKind::TangentBlock` whose
//! `rotation` is non-identity, every cell *below* that TB inherits the
//! TB's content frame for its slot indexing — so the `offset` is in
//! that rotated frame too. Motion deltas (`add_local`) are given in
//! WORLD frame and converted via `R^T_parent` before being added to
//! `offset`. Position queries (`in_frame`, `world_position`) walk the
//! tree rotation-aware so the camera always lands at the
//! rotation-correct world point.
//!
//! For Cartesian-only paths (no TB on path) every transform reduces to
//! the identity and the behavior is bit-identical to a pure-Cartesian
//! position. This is the historical default for fractal worlds.

use super::path::{Path, Transition};
use super::WORLD_SIZE;
use crate::world::tree::{
    slot_coords, slot_index, Child, NodeId, NodeKind, NodeLibrary, IDENTITY_ROTATION, MAX_DEPTH,
};

/// `(anchor, offset)` position. Offset interpretation: see module docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPos {
    pub anchor: Path,
    pub offset: [f32; 3],
}

impl WorldPos {
    pub const fn new_unchecked(anchor: Path, offset: [f32; 3]) -> Self {
        Self { anchor, offset }
    }

    /// Construct without rotation context. Renormalizes via Cartesian
    /// stepping only — safe for callers that build positions in
    /// Cartesian-only worlds (most fractals, harness setup).
    pub fn new(anchor: Path, offset: [f32; 3]) -> Self {
        let mut p = Self { anchor, offset };
        p.renormalize_cartesian();
        p
    }

    pub const fn root_origin() -> Self {
        Self { anchor: Path::root(), offset: [0.0, 0.0, 0.0] }
    }

    /// Anchor of `slot` repeated `depth` times with explicit `offset`.
    /// No rotation interpretation — caller is responsible for ensuring
    /// the offset is consistent with the implied frame.
    pub fn uniform_column(slot: u8, depth: u8, offset: [f32; 3]) -> Self {
        debug_assert!((slot as usize) < 27, "slot must be < 27");
        debug_assert!((depth as usize) <= MAX_DEPTH, "depth exceeds MAX_DEPTH");
        let mut anchor = Path::root();
        for _ in 0..depth {
            anchor.push(slot);
        }
        Self::new(anchor, offset)
    }

    /// Cartesian-only renormalize. Used at construction and as a
    /// fallback when no library/world_root is available.
    fn renormalize_cartesian(&mut self) {
        for axis in 0..3 {
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
            if self.offset[axis] >= 1.0 {
                self.offset[axis] = 1.0 - f32::EPSILON;
            }
            if self.offset[axis] < 0.0 {
                self.offset[axis] = 0.0;
            }
        }
    }

    /// Kind-aware renormalize. Bubbles cell-local overflow through the
    /// tree so wraps (WrappedPlane) fire in place. Cell-local axes are
    /// the deepest cell's parent's internal frame — for paths inside a
    /// TB this is the TB's content frame, but stepping is unchanged
    /// because every level inside the TB shares that frame.
    fn renormalize_world(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        let mut transition = Transition::None;
        for axis in 0..3 {
            let mut guard: i32 = 0;
            while self.offset[axis] >= 1.0 && guard < 1 << 20 {
                self.offset[axis] -= 1.0;
                if self.anchor.step_neighbor_in_world(library, world_root, axis, 1) {
                    transition = Transition::WrappedPlaneWrap { axis: axis as u8 };
                }
                guard += 1;
            }
            while self.offset[axis] < 0.0 && guard < 1 << 20 {
                self.offset[axis] += 1.0;
                if self.anchor.step_neighbor_in_world(library, world_root, axis, -1) {
                    transition = Transition::WrappedPlaneWrap { axis: axis as u8 };
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
        transition
    }

    /// Cumulative rotation of the deepest cell's parent's internal
    /// frame relative to world. For a Cartesian-only path this is
    /// `IDENTITY_ROTATION`. For a path that crosses a `TangentBlock`,
    /// this is the product of `R_TB` for each TB cell on the path
    /// from root down to (but not including) the deepest cell.
    ///
    /// This is the frame in which `self.offset` lives.
    pub fn cumulative_rotation(
        &self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> [[f32; 3]; 3] {
        let mut r_acc = IDENTITY_ROTATION;
        let mut node_id = world_root;
        let depth = self.anchor.depth() as usize;
        if depth == 0 {
            return r_acc;
        }
        // Walk path[0..depth-1]: pick up R_TB at each TB cell traversed,
        // EXCLUDING the deepest cell itself (whose rotation, if any,
        // would apply to its own children, not to its offset).
        for k in 0..(depth - 1) {
            let n = match library.get(node_id) {
                Some(n) => n,
                None => break,
            };
            let slot = self.anchor.slot(k) as usize;
            match n.children[slot] {
                Child::Node(child_id) => {
                    if let Some(child_node) = library.get(child_id) {
                        if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                            r_acc = matmul3x3(&r_acc, &r);
                        }
                    }
                    node_id = child_id;
                }
                _ => break,
            }
        }
        r_acc
    }

    /// Absolute world position. Walks the anchor rotation-aware,
    /// applying TB rotations at each descent into a TB child.
    ///
    /// **Precision note:** the result has f32 absolute precision (≈
    /// `WORLD_SIZE * 1e-7`) regardless of anchor depth. For deep
    /// anchors (≥ ~14 levels) the cell size drops below precision, so
    /// `world_position` cannot resolve sub-cell offsets there. Use
    /// `in_frame` (precision-safe via common ancestor) for shader
    /// inputs whenever possible.
    pub fn world_position(
        &self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> [f32; 3] {
        let mut centre = [WORLD_SIZE * 0.5; 3];
        let mut size = WORLD_SIZE;
        let mut r_internal = IDENTITY_ROTATION;
        let mut deepest_parent_r = IDENTITY_ROTATION;
        let mut node_id = world_root;
        let mut have_node = true;

        for k in 0..self.anchor.depth() as usize {
            deepest_parent_r = r_internal;

            let slot = self.anchor.slot(k) as usize;
            let (sx, sy, sz) = slot_coords(slot);
            let child_size = size / 3.0;
            let centred_local = [
                (sx as f32 - 1.0) * child_size,
                (sy as f32 - 1.0) * child_size,
                (sz as f32 - 1.0) * child_size,
            ];
            let centred_world = mat3_mul_vec3(&r_internal, &centred_local);
            centre[0] += centred_world[0];
            centre[1] += centred_world[1];
            centre[2] += centred_world[2];
            size = child_size;

            if have_node {
                if let Some(n) = library.get(node_id) {
                    match n.children[slot] {
                        Child::Node(child_id) => {
                            if let Some(child_node) = library.get(child_id) {
                                if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                                    r_internal = matmul3x3(&r_internal, &r);
                                }
                            }
                            node_id = child_id;
                        }
                        _ => have_node = false,
                    }
                } else {
                    have_node = false;
                }
            }
        }

        let centred_offset_local = [
            (self.offset[0] - 0.5) * size,
            (self.offset[1] - 0.5) * size,
            (self.offset[2] - 0.5) * size,
        ];
        let centred_offset_world = mat3_mul_vec3(&deepest_parent_r, &centred_offset_local);
        [
            centre[0] + centred_offset_world[0],
            centre[1] + centred_offset_world[1],
            centre[2] + centred_offset_world[2],
        ]
    }

    /// Construct from absolute world position by descending the tree
    /// rotation-aware. At each level picks the slot whose AABB (in the
    /// node's internal frame) contains the world point.
    ///
    /// Same precision caveats as `world_position`.
    pub fn from_world(
        world: [f32; 3],
        depth: u8,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Self {
        let mut anchor = Path::root();
        let mut centre = [WORLD_SIZE * 0.5; 3];
        let mut size = WORLD_SIZE;
        let mut r_internal = IDENTITY_ROTATION;
        let mut deepest_parent_r = IDENTITY_ROTATION;
        let mut node_id = world_root;
        let mut have_node = true;

        for _ in 0..depth {
            deepest_parent_r = r_internal;

            // Which child of the current node contains `world`?
            // Map world to current node's internal frame.
            let centred_world = [
                world[0] - centre[0],
                world[1] - centre[1],
                world[2] - centre[2],
            ];
            let centred_local = mat3_transpose_mul_vec3(&r_internal, &centred_world);
            // centred_local in [-size/2, size/2]³ (if world inside node);
            // slot coord = floor((centred_local + size/2) / (size/3)).
            let cell_step = size / 3.0;
            let mut s = [0usize; 3];
            for i in 0..3 {
                let f = (centred_local[i] + size * 0.5) / cell_step;
                s[i] = f.floor().clamp(0.0, 2.0) as usize;
            }
            let slot = slot_index(s[0], s[1], s[2]) as u8;
            anchor.push(slot);

            // Descend.
            let centred_local_to_centre = [
                (s[0] as f32 - 1.0) * cell_step,
                (s[1] as f32 - 1.0) * cell_step,
                (s[2] as f32 - 1.0) * cell_step,
            ];
            let centred_world_to_centre = mat3_mul_vec3(&r_internal, &centred_local_to_centre);
            centre[0] += centred_world_to_centre[0];
            centre[1] += centred_world_to_centre[1];
            centre[2] += centred_world_to_centre[2];
            size = cell_step;

            if have_node {
                if let Some(n) = library.get(node_id) {
                    match n.children[slot as usize] {
                        Child::Node(child_id) => {
                            if let Some(child_node) = library.get(child_id) {
                                if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                                    r_internal = matmul3x3(&r_internal, &r);
                                }
                            }
                            node_id = child_id;
                        }
                        _ => have_node = false,
                    }
                } else {
                    have_node = false;
                }
            }
        }

        // Compute offset in deepest cell's parent's internal frame.
        let centred_world = [
            world[0] - centre[0],
            world[1] - centre[1],
            world[2] - centre[2],
        ];
        let centred_local = mat3_transpose_mul_vec3(&deepest_parent_r, &centred_world);
        let offset = [
            (centred_local[0] / size + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
            (centred_local[1] / size + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
            (centred_local[2] / size + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
        ];

        Self { anchor, offset }
    }

    /// Position expressed in `frame`'s content frame, where the frame's
    /// cell spans `[0, WORLD_SIZE)³`. **Precision-safe** via
    /// common-ancestor walk: f32 magnitudes stay bounded by the common
    /// ancestor's size, not by `WORLD_SIZE`, so deep anchors with
    /// shared frame prefix retain sub-cell precision.
    ///
    /// Rotation-aware: when either the anchor or the frame crosses a
    /// `TangentBlock`, slot offsets and the final mapping pick up the
    /// TB's rotation around the cube's centre.
    pub fn in_frame(
        &self,
        frame: &Path,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(frame) as usize;

        // Walk root → common ancestor to get common's NodeId.
        let mut common_node_id = world_root;
        let mut have_common = true;
        for k in 0..c {
            if let Some(n) = library.get(common_node_id) {
                let slot = self.anchor.slot(k) as usize;
                match n.children[slot] {
                    Child::Node(child_id) => common_node_id = child_id,
                    _ => {
                        have_common = false;
                        break;
                    }
                }
            } else {
                have_common = false;
                break;
            }
        }

        // Common ancestor's size in world units.
        let mut common_size = WORLD_SIZE;
        for _ in 0..c {
            common_size /= 3.0;
        }

        // Walk anchor[c..] in common's internal frame, tracking centre,
        // size, rotation, and deepest-parent rotation.
        let (anchor_cell_centre, anchor_cell_size, anchor_parent_r) = walk_subpath(
            library,
            common_node_id,
            have_common,
            self.anchor.as_slice(),
            c,
            common_size,
        );

        // Apply offset in deepest cell's parent's frame.
        let centred_offset_local = [
            (self.offset[0] - 0.5) * anchor_cell_size,
            (self.offset[1] - 0.5) * anchor_cell_size,
            (self.offset[2] - 0.5) * anchor_cell_size,
        ];
        let centred_offset_in_common = mat3_mul_vec3(&anchor_parent_r, &centred_offset_local);
        let pos_in_common = [
            anchor_cell_centre[0] + centred_offset_in_common[0],
            anchor_cell_centre[1] + centred_offset_in_common[1],
            anchor_cell_centre[2] + centred_offset_in_common[2],
        ];

        // Walk frame[c..] in common's internal frame to find frame's
        // centre, size, and internal rotation (used for the final map).
        let (frame_centre, frame_size_in_common, frame_internal_r) = walk_frame_subpath(
            library,
            common_node_id,
            have_common,
            frame.as_slice(),
            c,
            common_size,
        );

        // Map pos (in common's frame) into frame's content frame.
        let centred_in_common = [
            pos_in_common[0] - frame_centre[0],
            pos_in_common[1] - frame_centre[1],
            pos_in_common[2] - frame_centre[2],
        ];
        let centred_in_frame = mat3_transpose_mul_vec3(&frame_internal_r, &centred_in_common);
        let scale = WORLD_SIZE / frame_size_in_common;
        [
            centred_in_frame[0] * scale + WORLD_SIZE * 0.5,
            centred_in_frame[1] * scale + WORLD_SIZE * 0.5,
            centred_in_frame[2] * scale + WORLD_SIZE * 0.5,
        ]
    }

    /// Inverse of [`in_frame`]. Build a `WorldPos` at `anchor_depth`
    /// whose frame-local coordinate under `frame` equals `xyz`.
    /// Rotation-aware: descends through TB children correctly.
    pub fn from_frame_local(
        frame: &Path,
        xyz: [f32; 3],
        anchor_depth: u8,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Self {
        debug_assert!(anchor_depth >= frame.depth());
        // Map xyz (in frame's content frame) to world position.
        let (frame_centre, frame_size, frame_r) = walk_path_world(library, world_root, frame);
        let scale = frame_size / WORLD_SIZE;
        let centred_in_frame = [
            (xyz[0] - WORLD_SIZE * 0.5) * scale,
            (xyz[1] - WORLD_SIZE * 0.5) * scale,
            (xyz[2] - WORLD_SIZE * 0.5) * scale,
        ];
        let centred_world = mat3_mul_vec3(&frame_r, &centred_in_frame);
        let world = [
            frame_centre[0] + centred_world[0],
            frame_centre[1] + centred_world[1],
            frame_centre[2] + centred_world[2],
        ];
        Self::from_world(world, anchor_depth, library, world_root)
    }

    /// Push a new slot under the current offset. For Cartesian-only
    /// paths this is precision-safe pure cell-local arithmetic. For
    /// paths where the *new* deepest cell is a TB, applies `R_TB^T` to
    /// the centred offset so it lands in the TB's content frame.
    pub fn zoom_in(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        let mut coords = [0usize; 3];
        let mut new_offset = [0.0f32; 3];
        for i in 0..3 {
            let s = (self.offset[i] * 3.0).floor();
            coords[i] = s.clamp(0.0, 2.0) as usize;
            new_offset[i] = (self.offset[i] * 3.0 - s).clamp(0.0, 1.0 - f32::EPSILON);
        }
        let new_slot = slot_index(coords[0], coords[1], coords[2]) as u8;

        // Inspect the cell we're about to push: if it's a TB, the
        // offset's frame changes from current parent's frame to the
        // new cell's parent's frame. The "new cell's parent" is the
        // current deepest cell — same frame as before — so no
        // transform if we're just adding a Cartesian level.
        //
        // BUT if the *current* deepest cell is itself a TB, the new
        // child cell's parent (= the TB) has a rotated internal frame,
        // so the new offset (which we just computed in the TB's PARENT
        // frame) needs to be rotated into the TB's content frame.
        let new_anchor = {
            let mut a = self.anchor;
            a.push(new_slot);
            a
        };
        let new_parent_r =
            (Self { anchor: new_anchor, offset: new_offset }).cumulative_rotation(library, world_root);
        let old_parent_r = self.cumulative_rotation(library, world_root);

        if !rotations_equal(&new_parent_r, &old_parent_r) {
            // Rotation context changed: transform offset.
            let centred = [
                new_offset[0] - 0.5,
                new_offset[1] - 0.5,
                new_offset[2] - 0.5,
            ];
            // old → new transform: world stays the same, so
            //   R_old · centred_old = R_new · centred_new
            //   centred_new = R_new^T · R_old · centred_old
            let in_world = mat3_mul_vec3(&old_parent_r, &centred);
            let in_new = mat3_transpose_mul_vec3(&new_parent_r, &in_world);
            new_offset = [
                (in_new[0] + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
                (in_new[1] + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
                (in_new[2] + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
            ];
        }

        self.anchor.push(new_slot);
        self.offset = new_offset;
        Transition::None
    }

    /// Pop the deepest slot. For Cartesian-only paths this is pure
    /// cell-local arithmetic. For TB-crossing pops, applies `R_TB` to
    /// the centred offset so it lands in the new (shallower) parent's
    /// frame.
    pub fn zoom_out(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        let Some(slot) = self.anchor.pop() else { return Transition::None; };
        let (sx, sy, sz) = slot_coords(slot as usize);

        // Compute fresh cumulative rotation for the now-shallower path.
        let new_parent_r = self.cumulative_rotation(library, world_root);
        // Old parent rotation: same path + the popped slot. We can
        // reconstruct by re-pushing temporarily.
        let old_parent_r = {
            let mut tmp = *self;
            tmp.anchor.push(slot);
            tmp.cumulative_rotation(library, world_root)
        };

        let centred_old = [
            self.offset[0] - 0.5,
            self.offset[1] - 0.5,
            self.offset[2] - 0.5,
        ];
        // First map into the popped cell's parent frame at cell-size 1
        // by combining with the slot offset (slot - 1 ∈ {-1, 0, 1}).
        let combined_old = [
            (centred_old[0] + (sx as f32 - 1.0)) / 3.0,
            (centred_old[1] + (sy as f32 - 1.0)) / 3.0,
            (centred_old[2] + (sz as f32 - 1.0)) / 3.0,
        ];
        // If the rotation context changed (popped cell was a TB),
        // re-express in the new parent frame.
        let centred_new = if rotations_equal(&new_parent_r, &old_parent_r) {
            combined_old
        } else {
            let in_world = mat3_mul_vec3(&old_parent_r, &combined_old);
            mat3_transpose_mul_vec3(&new_parent_r, &in_world)
        };

        self.offset = [
            (centred_new[0] + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
            (centred_new[1] + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
            (centred_new[2] + 0.5).clamp(0.0, 1.0 - f32::EPSILON),
        ];
        Transition::None
    }

    /// Advance by a delta in WORLD frame. Internally converted to the
    /// deepest cell's parent frame via `R^T_parent`, added to offset,
    /// then renormalized.
    ///
    /// If the renormalize bubble crosses a TB level upward (i.e., the
    /// path's cumulative rotation context changes), falls back to
    /// world-canonical: compute world_pre, add delta, `from_world` at
    /// the original depth. Acceptable precision loss because TB-cross
    /// is rare.
    pub fn add_local(
        &mut self,
        delta_world: [f32; 3],
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Transition {
        let pre = *self;
        let r = pre.cumulative_rotation(library, world_root);
        let delta_local = mat3_transpose_mul_vec3(&r, &delta_world);
        for i in 0..3 {
            self.offset[i] += delta_local[i];
        }
        let transition = self.renormalize_world(library, world_root);

        // TB-cross detection: if any slot above the first TB on path
        // changed, we crossed.
        let crossed = match first_tb_path_index(library, world_root, &pre.anchor) {
            Some(tb_idx) => (0..=tb_idx)
                .any(|i| self.anchor.slot(i) != pre.anchor.slot(i)),
            None => false,
        };

        if crossed {
            let world_pre = pre.world_position(library, world_root);
            let world_post = [
                world_pre[0] + delta_world[0],
                world_pre[1] + delta_world[1],
                world_pre[2] + delta_world[2],
            ];
            *self = Self::from_world(world_post, pre.anchor.depth(), library, world_root);
        }

        transition
    }

    /// Vector from `other` to `self` in WORLD-frame units. **Precision-
    /// safe** via common-ancestor walk.
    pub fn offset_from(
        &self,
        other: &Self,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(&other.anchor) as usize;

        let mut common_node_id = world_root;
        let mut have_common = true;
        for k in 0..c {
            if let Some(n) = library.get(common_node_id) {
                match n.children[self.anchor.slot(k) as usize] {
                    Child::Node(child_id) => common_node_id = child_id,
                    _ => {
                        have_common = false;
                        break;
                    }
                }
            } else {
                have_common = false;
                break;
            }
        }

        let mut common_size = WORLD_SIZE;
        for _ in 0..c {
            common_size /= 3.0;
        }

        // Common's internal frame relative to world.
        let mut common_internal_r = IDENTITY_ROTATION;
        let mut node_walk = world_root;
        for k in 0..c {
            let n = match library.get(node_walk) {
                Some(n) => n,
                None => break,
            };
            let slot = self.anchor.slot(k) as usize;
            match n.children[slot] {
                Child::Node(child_id) => {
                    if let Some(child_node) = library.get(child_id) {
                        if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                            common_internal_r = matmul3x3(&common_internal_r, &r);
                        }
                    }
                    node_walk = child_id;
                }
                _ => break,
            }
        }

        let pos_self = position_in_subpath(
            library,
            common_node_id,
            have_common,
            &self.anchor,
            self.offset,
            c,
            common_size,
        );
        let pos_other = position_in_subpath(
            library,
            common_node_id,
            have_common,
            &other.anchor,
            other.offset,
            c,
            common_size,
        );

        let centred_in_common = [
            pos_self[0] - pos_other[0],
            pos_self[1] - pos_other[1],
            pos_self[2] - pos_other[2],
        ];
        // Rotate from common's internal frame to world frame.
        mat3_mul_vec3(&common_internal_r, &centred_in_common)
    }

    /// Zoom in repeatedly until anchor reaches `target_depth`.
    pub fn deepened_to(
        mut self,
        target_depth: u8,
        library: &NodeLibrary,
        world_root: NodeId,
    ) -> Self {
        while self.anchor.depth() < target_depth {
            self.zoom_in(library, world_root);
        }
        self
    }
}

/// Walk a sub-path starting at `start_node` (which sits at depth `c`
/// in the world tree) along `slots[c..]`, returning:
/// - centre of the deepest cell reached, in `start_node`'s internal frame,
/// - size of the deepest cell in world units,
/// - rotation of the deepest cell's PARENT internal frame relative to
///   `start_node`'s internal frame (the frame in which the offset of a
///   `WorldPos` with this anchor would live).
fn walk_subpath(
    library: &NodeLibrary,
    start_node: NodeId,
    have_node_init: bool,
    slots: &[u8],
    start_depth: usize,
    start_size: f32,
) -> ([f32; 3], f32, [[f32; 3]; 3]) {
    let mut centre = [0.0f32; 3];
    let mut size = start_size;
    let mut r_internal = IDENTITY_ROTATION;
    let mut deepest_parent_r = IDENTITY_ROTATION;
    let mut node_id = start_node;
    let mut have_node = have_node_init;

    for k in start_depth..slots.len() {
        deepest_parent_r = r_internal;

        let slot = slots[k] as usize;
        let (sx, sy, sz) = slot_coords(slot);
        let child_size = size / 3.0;
        let centred_local = [
            (sx as f32 - 1.0) * child_size,
            (sy as f32 - 1.0) * child_size,
            (sz as f32 - 1.0) * child_size,
        ];
        let centred_world = mat3_mul_vec3(&r_internal, &centred_local);
        centre[0] += centred_world[0];
        centre[1] += centred_world[1];
        centre[2] += centred_world[2];
        size = child_size;

        if have_node {
            if let Some(n) = library.get(node_id) {
                match n.children[slot] {
                    Child::Node(child_id) => {
                        if let Some(child_node) = library.get(child_id) {
                            if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                                r_internal = matmul3x3(&r_internal, &r);
                            }
                        }
                        node_id = child_id;
                    }
                    _ => have_node = false,
                }
            } else {
                have_node = false;
            }
        }
    }

    (centre, size, deepest_parent_r)
}

/// Walk a sub-path treated as a *frame*: returns centre, size, and
/// **internal** rotation (the frame's content rotation, used to map
/// world points into the frame's `[0, WORLD_SIZE)³` content space).
fn walk_frame_subpath(
    library: &NodeLibrary,
    start_node: NodeId,
    have_node_init: bool,
    slots: &[u8],
    start_depth: usize,
    start_size: f32,
) -> ([f32; 3], f32, [[f32; 3]; 3]) {
    let mut centre = [0.0f32; 3];
    let mut size = start_size;
    let mut r_internal = IDENTITY_ROTATION;
    let mut node_id = start_node;
    let mut have_node = have_node_init;

    for k in start_depth..slots.len() {
        let slot = slots[k] as usize;
        let (sx, sy, sz) = slot_coords(slot);
        let child_size = size / 3.0;
        let centred_local = [
            (sx as f32 - 1.0) * child_size,
            (sy as f32 - 1.0) * child_size,
            (sz as f32 - 1.0) * child_size,
        ];
        let centred_world = mat3_mul_vec3(&r_internal, &centred_local);
        centre[0] += centred_world[0];
        centre[1] += centred_world[1];
        centre[2] += centred_world[2];
        size = child_size;

        if have_node {
            if let Some(n) = library.get(node_id) {
                match n.children[slot] {
                    Child::Node(child_id) => {
                        if let Some(child_node) = library.get(child_id) {
                            if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                                r_internal = matmul3x3(&r_internal, &r);
                            }
                        }
                        node_id = child_id;
                    }
                    _ => have_node = false,
                }
            } else {
                have_node = false;
            }
        }
    }

    (centre, size, r_internal)
}

/// Walk a path from world root, returning (centre_world, size_world,
/// internal_rotation). Used by `from_frame_local`.
fn walk_path_world(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
) -> ([f32; 3], f32, [[f32; 3]; 3]) {
    let (centre_local, size, r) = walk_frame_subpath(
        library,
        world_root,
        true,
        path.as_slice(),
        0,
        WORLD_SIZE,
    );
    (
        [
            centre_local[0] + WORLD_SIZE * 0.5,
            centre_local[1] + WORLD_SIZE * 0.5,
            centre_local[2] + WORLD_SIZE * 0.5,
        ],
        size,
        r,
    )
}

/// Helper for `offset_from`: compute a position in common's internal
/// frame, where the path is given as full anchor + offset (subset
/// `[c..]` is what we walk).
fn position_in_subpath(
    library: &NodeLibrary,
    common_node_id: NodeId,
    have_common: bool,
    anchor: &Path,
    offset: [f32; 3],
    c: usize,
    common_size: f32,
) -> [f32; 3] {
    let (cell_centre, cell_size, parent_r) = walk_subpath(
        library,
        common_node_id,
        have_common,
        anchor.as_slice(),
        c,
        common_size,
    );
    let centred_offset_local = [
        (offset[0] - 0.5) * cell_size,
        (offset[1] - 0.5) * cell_size,
        (offset[2] - 0.5) * cell_size,
    ];
    let centred_offset_in_common = mat3_mul_vec3(&parent_r, &centred_offset_local);
    [
        cell_centre[0] + centred_offset_in_common[0],
        cell_centre[1] + centred_offset_in_common[1],
        cell_centre[2] + centred_offset_in_common[2],
    ]
}

/// Index `i` of the first slot on `path` whose corresponding cell is a
/// `TangentBlock` (i.e., `path[i]` is the slot in path's parent that
/// reaches a TB cell). Returns `None` if no TB on the path.
fn first_tb_path_index(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
) -> Option<usize> {
    let mut node_id = world_root;
    for k in 0..path.depth() as usize {
        let n = library.get(node_id)?;
        let slot = path.slot(k) as usize;
        match n.children[slot] {
            Child::Node(child_id) => {
                if let Some(child_node) = library.get(child_id) {
                    if matches!(child_node.kind, NodeKind::TangentBlock { .. }) {
                        return Some(k);
                    }
                }
                node_id = child_id;
            }
            _ => return None,
        }
    }
    None
}

/// 3×3 matrix multiply, both column-major (`r[col][row]`).
fn matmul3x3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for c in 0..3 {
        for r in 0..3 {
            let mut s = 0.0f32;
            for k in 0..3 {
                s += a[k][r] * b[c][k];
            }
            out[c][r] = s;
        }
    }
    out
}

/// Apply a column-major 3×3 matrix to a 3-vector.
fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Apply the transpose of a column-major 3×3 matrix to a 3-vector.
fn mat3_transpose_mul_vec3(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Approximate equality for 3×3 matrices (tolerates f32 drift).
fn rotations_equal(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> bool {
    const EPS: f32 = 1e-5;
    for c in 0..3 {
        for r in 0..3 {
            if (a[c][r] - b[c][r]).abs() > EPS {
                return false;
            }
        }
    }
    true
}
