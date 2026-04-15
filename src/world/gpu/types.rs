//! GPU-layout types matching the WGSL ray-march shader's structs.

use bytemuck::{Pod, Zeroable};

use crate::world::tree::NodeKind;

/// One node in the GPU buffer = 27 GpuChild = 216 bytes.
pub const GPU_NODE_SIZE: usize = 27;

/// One child slot in the packed tree buffer. 8 bytes total:
///
/// - `tag` (u8): 0 = Empty, 1 = Block, 2 = Node
/// - `block_type` (u8): valid when `tag == 1` (LOD-flattened or
///   leaf block). For `tag == 2` it carries the child's
///   representative block — a hint the shader can use without
///   descending.
/// - `_pad` (u16): alignment.
/// - `node_index` (u32): buffer-local index, valid when `tag == 2`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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
    pub surface_r: f32,
    pub noise_scale: f32,
    pub noise_freq: f32,
    pub noise_seed: u32,
    pub surface_block: u32,
    pub core_block: u32,
    pub _pad: [u32; 2],
}

impl GpuNodeKind {
    pub fn from_node_kind(k: NodeKind) -> Self {
        match k {
            NodeKind::Cartesian => Self {
                kind: 0,
                face: 0,
                inner_r: 0.0,
                outer_r: 0.0,
                surface_r: 0.0,
                noise_scale: 0.0,
                noise_freq: 0.0,
                noise_seed: 0,
                surface_block: 0,
                core_block: 0,
                _pad: [0; 2],
            },
            NodeKind::CubedSphereBody {
                inner_r,
                outer_r,
                surface_r,
                noise_scale,
                noise_freq,
                noise_seed,
                surface_block,
                core_block,
            } => Self {
                kind: 1,
                face: 0,
                inner_r,
                outer_r,
                surface_r,
                noise_scale,
                noise_freq,
                noise_seed,
                surface_block: surface_block as u32,
                core_block: core_block as u32,
                _pad: [0; 2],
            },
            NodeKind::CubedSphereFace { face } => Self {
                kind: 2,
                face: face as u32,
                inner_r: 0.0,
                outer_r: 0.0,
                surface_r: 0.0,
                noise_scale: 0.0,
                noise_freq: 0.0,
                noise_seed: 0,
                surface_block: 0,
                core_block: 0,
                _pad: [0; 2],
            },
            NodeKind::CubedSphereProceduralFace { face } => Self {
                kind: 3,
                face: face as u32,
                inner_r: 0.0,
                outer_r: 0.0,
                surface_r: 0.0,
                noise_scale: 0.0,
                noise_freq: 0.0,
                noise_seed: 0,
                surface_block: 0,
                core_block: 0,
                _pad: [0; 2],
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
        assert_eq!(std::mem::size_of::<GpuNodeKind>(), 48);
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
            surface_r: 0.30,
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 0,
            surface_block: 1,
            core_block: 2,
        });
        assert_eq!(k.kind, 1);
        assert!((k.inner_r - 0.12).abs() < 1e-7);
        assert!((k.outer_r - 0.45).abs() < 1e-7);
        assert!((k.surface_r - 0.30).abs() < 1e-7);
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
