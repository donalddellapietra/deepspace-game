//! GPU-layout types matching the WGSL ray-march shader's structs.

use bytemuck::{Pod, Zeroable};

use crate::world::tree::NodeKind;

/// Per-node header in the sparse GPU layout. 8 bytes:
///
/// - `occupancy` (u32): low 27 bits form a bitmask where bit `s`
///   is set iff slot `s` is non-empty. Bits 27..31 are reserved for
///   per-node flags.
/// - `first_child` (u32): offset into the packed children buffer
///   (in `GpuChild` units) of this node's run of non-empty children.
///   Entries are stored in slot-ascending order; the `k`-th popcount
///   of occupancy bits below slot `s` is the rank of slot `s` within
///   the run.
///
/// Look up slot `s`:
/// ```text
/// let h = nodes[n];
/// let bit = 1u32 << s;
/// if h.occupancy & bit == 0 { EMPTY }
/// else {
///     let rank = (h.occupancy & (bit - 1)).count_ones();
///     children[h.first_child + rank]
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq, Default)]
pub struct NodeHeader {
    pub occupancy: u32,
    pub first_child: u32,
}

/// One child entry in the packed children buffer. 8 bytes.
/// Empty slots are absent from the buffer — they're encoded by
/// a clear bit in the parent's `NodeHeader.occupancy`. Only two
/// tag values appear: `1 = Block`, `2 = Node`.
///
/// - `tag` (u8): 1 = Block, 2 = Node. (tag=0 never appears.)
/// - `block_type` (u8): valid when `tag == 1` (LOD-flattened or
///   leaf block). For `tag == 2` it carries the child's
///   representative block — a hint the shader can use without
///   descending.
/// - `_pad` (u16): reserved for per-child flags.
/// - `node_index` (u32): buffer-local index, valid when `tag == 2`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuChild {
    pub tag: u8,
    pub block_type: u8,
    pub _pad: u16,
    pub node_index: u32,
}

/// Per-packed-node metadata: which `NodeKind` this node is, plus
/// the per-kind data the shader needs to render its content.
/// Indexed by the same buffer index used in `GpuChild::node_index`.
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
    fn node_header_size() {
        assert_eq!(std::mem::size_of::<NodeHeader>(), 8);
    }

    #[test]
    fn gpu_child_size() {
        assert_eq!(std::mem::size_of::<GpuChild>(), 8);
    }

    #[test]
    fn gpu_node_kind_size() {
        assert_eq!(std::mem::size_of::<GpuNodeKind>(), 16);
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
