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
/// tree[header_offset[b] + 2 + rank*2 + 0]  = packed (tag|block_type_lo|block_type_hi|flags)
/// tree[header_offset[b] + 2 + rank*2 + 1]  = node_index (tag=2)
/// ```
///
/// - `tag` (u8): 1 = Block, 2 = Node. (tag=0 never appears.) Kept at
///   byte 0 so the shader's tag extraction is a simple `packed & 0xFFu`
///   regardless of palette width.
/// - `block_type_lo` + `block_type_hi` (u16 little-endian across bytes
///   1-2): palette index. Valid when `tag == 1` (LOD-flattened or
///   leaf block). For `tag == 2` it carries the child's representative
///   block — a hint the shader uses for the empty-representative fast
///   path without descending. Combined u16 gives 65 536 distinct
///   palette entries.
/// - `flags` (u8): reserved for per-child flags.
/// - `node_index` (u32): BFS position of the child node when
///   `tag == 2`. Used to index `node_kinds[]` and `node_offsets[]`.
///   The header u32-offset in `tree[]` is
///   `node_offsets[node_index]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuChild {
    pub tag: u8,
    pub block_type_lo: u8,
    pub block_type_hi: u8,
    pub flags: u8,
    pub node_index: u32,
}

impl GpuChild {
    #[inline]
    pub fn new(tag: u8, block_type: u16, flags: u8, node_index: u32) -> Self {
        let [lo, hi] = block_type.to_le_bytes();
        Self {
            tag,
            block_type_lo: lo,
            block_type_hi: hi,
            flags,
            node_index,
        }
    }

    #[inline]
    pub fn block_type(&self) -> u16 {
        u16::from_le_bytes([self.block_type_lo, self.block_type_hi])
    }
}

/// Per-packed-node metadata: which `NodeKind` this node is.
/// Indexed by BFS position — the same `node_index` used in
/// `GpuChild::node_index`.
///
/// 16 bytes per node so the WGSL `array<NodeKindGpu>` aligns
/// cleanly. `kind` discriminant: 0 = Cartesian, 1 = WrappedPlanet.
///
/// `geom_a/b/c` are neutral integer slots. WrappedPlanet packs
/// `(width, height, depth | (active_subdepth << 16))`. Cartesian
/// leaves them zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuNodeKind {
    pub kind: u32,
    pub geom_a: u32,
    pub geom_b: u32,
    pub geom_c: u32,
}

impl GpuNodeKind {
    pub fn from_node_kind(k: NodeKind) -> Self {
        match k {
            NodeKind::Cartesian => Self { kind: 0, geom_a: 0, geom_b: 0, geom_c: 0 },
            NodeKind::WrappedPlanet { width, height, depth, active_subdepth } => Self {
                kind: 1,
                geom_a: width as u32,
                geom_b: height as u32,
                geom_c: (depth as u32) | ((active_subdepth as u32) << 16),
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
    /// World → clip-space matrix (column-major). The ray-march writes
    /// this into `@builtin(frag_depth)` via the `fs_main_depth`
    /// entry point so the entity raster pass can z-test against it.
    /// When the raster pass is disabled, the matrix is still
    /// uploaded (trivial cost) but nothing reads it.
    pub view_proj: [[f32; 4]; 4],
}

/// One entity instance on the GPU: a bounding cube in the current
/// render frame's [0, 3)³ local coords plus a BFS idx into the
/// shared `tree[]` buffer for the entity's voxel subtree.
///
/// The shader ray-marches against `bbox_min`/`bbox_max`; on AABB
/// hit it either splats `representative_block` (sub-pixel entity,
/// skips the whole subtree descent) or transforms the ray into
/// the subtree's local [0, 3)³ space and calls `march_cartesian`
/// with a depth budget sized to the entity's on-screen pixel
/// count. `representative_block` is stored as u32 to keep the
/// GPU struct 16-byte aligned even though the CPU-side value is
/// a u16 palette index.
///
/// 32 bytes total (vec4-aligned for WGSL storage buffer).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct GpuEntity {
    pub bbox_min: [f32; 3],
    pub representative_block: u32,
    pub bbox_max: [f32; 3],
    pub subtree_bfs: u32,
}

// The former fixed-size `GpuPalette` uniform struct has been removed.
// Palette colors now live in a variable-length read-only storage
// buffer (see `ColorRegistry::to_gpu_palette` -> `Vec<[f32; 4]>`
// and `src/renderer/init.rs`'s palette buffer binding).

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
    fn from_node_kind_cartesian() {
        let k = GpuNodeKind::from_node_kind(NodeKind::Cartesian);
        assert_eq!(k.kind, 0);
    }

    #[test]
    fn from_node_kind_wrapped_planet_packs_dims() {
        let k = GpuNodeKind::from_node_kind(NodeKind::WrappedPlanet {
            width: 18,
            height: 9,
            depth: 3,
            active_subdepth: 2,
        });
        assert_eq!(k.kind, 1);
        assert_eq!(k.geom_a, 18);
        assert_eq!(k.geom_b, 9);
        assert_eq!(k.geom_c & 0xFFFF, 3);
        assert_eq!((k.geom_c >> 16) & 0xFF, 2);
    }
}
