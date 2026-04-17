//! GPU-layout types matching the WGSL ray-march shader's structs.

use bytemuck::{Pod, Zeroable};

use crate::world::tree::NodeKind;

/// One child entry in the interleaved sparse-tree layout. 8 bytes.
///
/// The child entry is emitted inline into the `tree: Vec<u32>` buffer
/// immediately after its parent's 2-u32 header. For each non-empty
/// slot `s` at BFS position `b` the pack emits two u32s:
///
/// ```text
/// tree[header_offset[b] + 2 + rank*2 + 0]  = packed (tag|block_type|pad)
/// tree[header_offset[b] + 2 + rank*2 + 1]  = node_index (tag=2)
/// ```
///
/// - `tag` (u8): 1 = Block, 2 = Node. (tag=0 never appears.)
/// - `block_type` (u8): valid when `tag == 1` (LOD-flattened or
///   leaf block). For `tag == 2` it carries the child's
///   representative block — a hint the shader uses for the
///   empty-representative fast path without descending.
/// - `_pad` (u16): reserved for per-child flags.
/// - `node_index` (u32): BFS position of the child node when
///   `tag == 2`. Used to index `node_kinds[]` and `node_offsets[]`.
///   The header u32-offset in `tree[]` is
///   `node_offsets[node_index]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuChild {
    pub tag: u8,
    pub block_type: u8,
    pub _pad: u16,
    pub node_index: u32,
}

/// Per-packed-node parent pointer. Indexed by the child's BFS
/// position. Lets the shader pop upward without consulting a
/// pre-built ribbon: at pop time it reads `parent_info[current_idx]`,
/// recovers `(parent_node_idx, slot_in_parent)`, and applies the
/// same `pos = vec3(slot_xyz) + pos/3` rescale the ribbon code did.
///
/// 8 bytes per node — same encoding as the retired `GpuRibbonEntry`,
/// just keyed on the child's BFS index instead of the pop level.
///
/// `slot_and_flags` packs:
/// - low 5 bits: slot (0..27) the child occupies in its parent
/// - bit 31: `siblings_all_empty` — when set, every other slot of
///   `parent_node_idx` is absent from the pack (parent has
///   occupancy.count_ones() == 1). The shader uses this to fast-
///   exit empty shells with a single ray–box intersection.
///
/// World-root sentinel: `parent_node_idx == u32::MAX`. The shader
/// uses this to terminate pop loops.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq)]
pub struct GpuParentInfo {
    pub parent_node_idx: u32,
    pub slot_and_flags: u32,
}

pub const PARENT_SLOT_MASK: u32 = 0x1F;
pub const PARENT_SIBLINGS_ALL_EMPTY: u32 = 1 << 31;
pub const PARENT_NONE: u32 = u32::MAX;

impl GpuParentInfo {
    pub fn new(parent_node_idx: u32, slot: u8, siblings_all_empty: bool) -> Self {
        let flags = if siblings_all_empty { PARENT_SIBLINGS_ALL_EMPTY } else { 0 };
        Self {
            parent_node_idx,
            slot_and_flags: (slot as u32) | flags,
        }
    }

    pub fn root() -> Self {
        Self { parent_node_idx: PARENT_NONE, slot_and_flags: 0 }
    }

    pub fn slot(&self) -> u32 { self.slot_and_flags & PARENT_SLOT_MASK }
    pub fn siblings_all_empty(&self) -> bool {
        (self.slot_and_flags & PARENT_SIBLINGS_ALL_EMPTY) != 0
    }
    pub fn is_root(&self) -> bool { self.parent_node_idx == PARENT_NONE }
}

/// Per-packed-node metadata: which `NodeKind` this node is, plus
/// the per-kind data the shader needs to render its content.
/// Indexed by BFS position — the same `node_index` used in
/// `GpuChild::node_index`.
///
/// 16 bytes per node so the WGSL `array<NodeKindGpu>` aligns
/// cleanly. `kind` discriminant: 0 = Cartesian, 1 = CubedSphereBody,
/// 2 = CubedSphereFace.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuNodeKind {
    pub kind: u32,
    pub face: u32,
    pub inner_r: f32,
    pub outer_r: f32,
}

impl GpuNodeKind {
    pub fn from_node_kind(k: NodeKind) -> Self {
        match k {
            NodeKind::Cartesian => Self { kind: 0, face: 0, inner_r: 0.0, outer_r: 0.0 },
            NodeKind::CubedSphereBody { inner_r, outer_r } => Self {
                kind: 1, face: 0, inner_r, outer_r,
            },
            NodeKind::CubedSphereFace { face } => Self {
                kind: 2, face: face as u32, inner_r: 0.0, outer_r: 0.0,
            },
        }
    }
}

/// Camera uniforms in shader-frame coords. `pos`/`forward`/etc. are
/// in the current render frame's local `[0, 3)³` space.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCamera {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub forward: [f32; 3],
    pub _pad1: f32,
    pub right: [f32; 3],
    pub _pad2: f32,
    pub up: [f32; 3],
    pub fov: f32,
}

/// Block color palette — up to 256 RGBA colors indexed by block type.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPalette {
    pub colors: [[f32; 4]; 256],
}

impl Default for GpuPalette {
    fn default() -> Self {
        let mut colors = [[0.0f32; 4]; 256];
        for &(idx, _, color) in crate::world::palette::BUILTINS {
            colors[idx as usize] = color;
        }
        Self { colors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_child_size() {
        assert_eq!(std::mem::size_of::<GpuChild>(), 8);
    }

    #[test]
    fn gpu_node_kind_size() {
        assert_eq!(std::mem::size_of::<GpuNodeKind>(), 16);
    }

    #[test]
    fn gpu_parent_info_size() {
        assert_eq!(std::mem::size_of::<GpuParentInfo>(), 8);
    }

    #[test]
    fn gpu_parent_info_root_sentinel() {
        let r = GpuParentInfo::root();
        assert!(r.is_root());
        assert_eq!(r.parent_node_idx, PARENT_NONE);
    }

    #[test]
    fn gpu_parent_info_encoding_round_trip() {
        let p = GpuParentInfo::new(42, 19, true);
        assert_eq!(p.parent_node_idx, 42);
        assert_eq!(p.slot(), 19);
        assert!(p.siblings_all_empty());
        assert!(!p.is_root());

        let q = GpuParentInfo::new(1, 0, false);
        assert_eq!(q.slot(), 0);
        assert!(!q.siblings_all_empty());
    }

    #[test]
    fn from_node_kind_cartesian() {
        let k = GpuNodeKind::from_node_kind(NodeKind::Cartesian);
        assert_eq!(k.kind, 0);
    }

    #[test]
    fn from_node_kind_body_carries_radii() {
        let k = GpuNodeKind::from_node_kind(NodeKind::CubedSphereBody {
            inner_r: 0.12, outer_r: 0.45,
        });
        assert_eq!(k.kind, 1);
        assert!((k.inner_r - 0.12).abs() < 1e-7);
        assert!((k.outer_r - 0.45).abs() < 1e-7);
    }

    #[test]
    fn from_node_kind_face_carries_face_id() {
        let k = GpuNodeKind::from_node_kind(NodeKind::CubedSphereFace {
            face: crate::world::cubesphere::Face::PosX,
        });
        assert_eq!(k.kind, 2);
        assert_eq!(k.face, crate::world::cubesphere::Face::PosX as u32);
    }
}
