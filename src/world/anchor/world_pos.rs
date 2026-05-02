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

    pub fn new(anchor: Path, offset: [f32; 3]) -> Self {
        let mut p = Self { anchor, offset };
        p.renormalize_cartesian();
        p
    }

    pub const fn root_origin() -> Self {
        Self { anchor: Path::root(), offset: [0.0, 0.0, 0.0] }
    }

    /// Precise anchor constructed directly as `slot` repeated `depth`
    /// times, with a fixed sub-cell `offset ∈ [0, 1)³`.
    pub fn uniform_column(slot: u8, depth: u8, offset: [f32; 3]) -> Self {
        debug_assert!((slot as usize) < 8, "slot must be < 8");
        debug_assert!((depth as usize) <= MAX_DEPTH, "depth exceeds MAX_DEPTH");
        let mut anchor = Path::root();
        for _ in 0..depth {
            anchor.push(slot);
        }
        Self::new(anchor, offset)
    }

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
                let (ok, wrapped) = self.anchor.step_neighbor_in_world(library, world_root, axis, 1);
                if !ok {
                    self.offset[axis] = 1.0 - f32::EPSILON;
                    break;
                }
                if wrapped {
                    transition = Transition::WrappedPlaneWrap { axis: axis as u8 };
                }
                guard += 1;
            }
            while self.offset[axis] < 0.0 && guard < 1 << 20 {
                self.offset[axis] += 1.0;
                let (ok, wrapped) = self.anchor.step_neighbor_in_world(library, world_root, axis, -1);
                if !ok {
                    self.offset[axis] = 0.0;
                    break;
                }
                if wrapped {
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

    pub fn zoom_in(&mut self) -> Transition {
        let mut coords = [0usize; 3];
        for i in 0..3 {
            let s = (self.offset[i] * 2.0).floor();
            coords[i] = s.clamp(0.0, 1.0) as usize;
            self.offset[i] = (self.offset[i] * 2.0 - s).clamp(0.0, 1.0 - f32::EPSILON);
        }
        let slot = slot_index(coords[0], coords[1], coords[2]) as u8;
        self.anchor.push(slot);
        Transition::None
    }

    pub fn offset_from(&self, other: &Self) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(&other.anchor) as usize;
        let mut common_size = WORLD_SIZE;
        for _ in 0..c {
            common_size /= 2.0;
        }
        let walk = |p: &Self| -> [f32; 3] {
            let mut pos = [0.0f32; 3];
            let mut size = common_size;
            for k in c..(p.anchor.depth() as usize) {
                let (sx, sy, sz) = slot_coords(p.anchor.slot(k) as usize);
                let child = size / 2.0;
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

    pub fn deepened_to(mut self, target_depth: u8) -> Self {
        while self.anchor.depth() < target_depth {
            self.zoom_in();
        }
        self
    }

    pub fn in_frame(&self, frame: &Path) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(frame) as usize;

        let mut pos_common = [0.0f32; 3];
        let mut size = WORLD_SIZE;
        for k in c..(self.anchor.depth() as usize) {
            let (sx, sy, sz) = slot_coords(self.anchor.slot(k) as usize);
            let child = size / 2.0;
            pos_common[0] += sx as f32 * child;
            pos_common[1] += sy as f32 * child;
            pos_common[2] += sz as f32 * child;
            size = child;
        }
        pos_common[0] += self.offset[0] * size;
        pos_common[1] += self.offset[1] * size;
        pos_common[2] += self.offset[2] * size;

        let mut frame_origin = [0.0f32; 3];
        let mut frame_size = WORLD_SIZE;
        for k in c..(frame.depth() as usize) {
            let (sx, sy, sz) = slot_coords(frame.slot(k) as usize);
            let child = frame_size / 2.0;
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

    /// Rotation-aware variant of [`in_frame`].
    pub fn in_frame_rot(
        &self,
        library: &NodeLibrary,
        world_root: NodeId,
        frame: &Path,
    ) -> [f32; 3] {
        let c = self.anchor.common_prefix_len(frame) as usize;

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

        let mut cur_centre = [WORLD_SIZE * 0.5; 3];
        let mut cur_size = WORLD_SIZE;
        let mut cur_rot = IDENTITY_ROTATION;
        let _ = common_rot;
        let mut have_node = true;
        let mut node = common_node;
        for k in c..(self.anchor.depth() as usize) {
            let slot = self.anchor.slot(k);
            let (sx, sy, sz) = slot_coords(slot as usize);
            let child_size = cur_size / 2.0;
            // In base-2, slot coords are 0 or 1. The centre of the
            // parent node is at cur_size/2 = child_size. Slot 0's
            // centre is at child_size/2, slot 1's at child_size*1.5.
            // Offset from parent centre: (sx - 0.5) * child_size.
            let centred_local = [
                (sx as f32 - 0.5) * child_size,
                (sy as f32 - 0.5) * child_size,
                (sz as f32 - 0.5) * child_size,
            ];
            let centred_common = mat3_mul_vec3(&cur_rot, &centred_local);
            cur_centre = [
                cur_centre[0] + centred_common[0],
                cur_centre[1] + centred_common[1],
                cur_centre[2] + centred_common[2],
            ];

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

        let mut frame_origin = [0.0f32; 3];
        let mut frame_size = WORLD_SIZE;
        for k in c..(frame.depth() as usize) {
            let (sx, sy, sz) = slot_coords(frame.slot(k) as usize);
            let child = frame_size / 2.0;
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
            let child = size / 2.0;
            let mut s = [0usize; 3];
            for i in 0..3 {
                let v = ((clamped[i] - origin[i]) / child).floor().clamp(0.0, 1.0) as usize;
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

    pub fn zoom_out(&mut self) -> Transition {
        let Some(slot) = self.anchor.pop() else { return Transition::None; };
        let (sx, sy, sz) = slot_coords(slot as usize);
        self.offset[0] = (self.offset[0] + sx as f32) / 2.0;
        self.offset[1] = (self.offset[1] + sy as f32) / 2.0;
        self.offset[2] = (self.offset[2] + sz as f32) / 2.0;
        debug_assert!(self.offset.iter().all(|&x| (0.0..1.0).contains(&x)));
        Transition::None
    }
}

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

fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}
