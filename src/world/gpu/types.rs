//! GPU-layout types matching the WGSL ray-march shader's structs.

use bytemuck::{Pod, Zeroable};

use crate::world::tree::NodeKind;

/// One child entry in the interleaved sparse-tree layout. 12 bytes.
///
/// The child entry is emitted inline into the `tree: Vec<u32>` buffer
/// immediately after its parent's 2-u32 header. For each non-empty
/// slot `s` at BFS position `b` the pack emits three u32s:
///
/// ```text
/// tree[header_offset[b] + 2 + rank*3 + 0]  = packed (tag|block_type|pad)
/// tree[header_offset[b] + 2 + rank*3 + 1]  = node_index (tag=2)
/// tree[header_offset[b] + 2 + rank*3 + 2]  = child_occupancy (tag=2)
/// ```
///
/// The third u32 **inlines the child node's own occupancy mask**,
/// duplicated from `tree[node_offsets[node_index]]`. The shader uses
/// it at descend-time to initialise `cur_occupancy` without waiting
/// for the `node_offsets → tree[header_off]` dependent load chain —
/// turning a 3-hop critical path into a 1-hop one. For tag=1 leaves
/// this field is 0 (unused).
///
/// - `tag` (u8): 1 = Block, 2 = Node. (tag=0 never appears.)
/// - `block_type` (u8): valid when `tag == 1` (LOD-flattened or
///   leaf block). For `tag == 2` it carries the child's
///   representative block — a hint the shader uses for the
///   empty-representative fast path without descending.
/// - `_pad` (u16): content AABB (12 bits) + reserved for per-child flags.
/// - `node_index` (u32): BFS position of the child node when
///   `tag == 2`. Used to index `node_kinds[]` and `node_offsets[]`.
///   The header u32-offset in `tree[]` is
///   `node_offsets[node_index]`.
/// - `child_occupancy` (u32): 27-bit occupancy mask of the child
///   node (tag=2 only), inlined here so the descending ray has it
///   without chasing the node_offsets indirection.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuChild {
    pub tag: u8,
    pub block_type: u8,
    pub _pad: u16,
    pub node_index: u32,
    pub child_occupancy: u32,
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
///
/// The `jitter_x_px` / `jitter_y_px` slots carry a sub-pixel offset
/// applied to the NDC coordinates in the ray-march fragment shader.
/// Zero when TAAU is disabled (no visual change); non-zero when
/// TAAU is enabled so successive frames sample distinct sub-pixel
/// positions within each output pixel. Units: texels in the scaled-
/// resolution framebuffer (range `(-0.5, +0.5)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCamera {
    pub pos: [f32; 3],
    pub jitter_x_px: f32,
    pub forward: [f32; 3],
    pub jitter_y_px: f32,
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
        assert_eq!(std::mem::size_of::<GpuChild>(), 12);
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
