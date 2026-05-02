//! `(anchor, offset)` position with the offset held in `[0, 1)³`
//! by invariant. All zoom / motion / frame-projection arithmetic
//! lives here.

use super::path::{Path, Transition};
use super::WORLD_SIZE;
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

    /// Kind-aware renormalize. Walks the anchor through the world
    /// tree so that overflow on the wrap axis inside a
    /// `WrappedPlane` subtree wraps in place instead of bubbling.
    /// Returns the strongest transition observed during the renorm
    /// (any wrap event collapses to a single `WrappedPlaneWrap`;
    /// no-wrap steps return `Transition::None`).
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
        // The shader applies R^T when entering a TangentBlock, so
        // when the frame IS a TB (common_rot = R), the tail walk
        // must start with R^T so the GPU camera position matches
        // the shader's rotated-local convention. At the cell center
        // R^T is identity (rotation around center preserves center),
        // but at the edges the rotated position differs — without
        // this, the camera jumps by ~1 cell when crossing the TB
        // boundary away from center.
        let mut cur_rot = mat3_transpose(&common_rot);
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
            // Rotate into common ancestor's frame and add to centre.
            let centred_common = mat3_mul_vec3(&cur_rot, &centred_local);
            cur_centre = [
                cur_centre[0] + centred_common[0],
                cur_centre[1] + centred_common[1],
                cur_centre[2] + centred_common[2],
            ];

            // Update rotation if descending into a TangentBlock.
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

        // Apply the offset, also rotated by the cumulative chain
        // around the anchor cell's centre.
        let centred_offset_local = [
            (self.offset[0] - 0.5) * cur_size,
            (self.offset[1] - 0.5) * cur_size,
            (self.offset[2] - 0.5) * cur_size,
        ];
        let centred_offset_common = mat3_mul_vec3(&cur_rot, &centred_offset_local);
        let pos_common = [
            cur_centre[0] + centred_offset_common[0],
            cur_centre[1] + centred_offset_common[1],
            cur_centre[2] + centred_offset_common[2],
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

fn mat3_transpose(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
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
