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

/// Per-packed-node metadata: 80 bytes total.
///
/// `kind`: 0 = Cartesian, 1 = WrappedPlane, 2 = TangentBlock.
/// `dims_x/y/z`: slab dims for `WrappedPlane`; zero otherwise.
///
/// `rot_col0/1/2`: 3×3 rotation matrix (column-major) for
/// `TangentBlock`. Each column is `vec4<f32>` (xyz = column, w = 0)
/// for std140-style 16-byte vec3 alignment. Identity for the other
/// kinds (shader doesn't read it for those). `rot_col0.w` carries
/// the inscribed-cube `tb_scale`.
///
/// `cell_offset` (xyz, w padding): TB cell's displacement from its
/// natural slot centre, in parent-frame `[0, 3)³` units. Zero for
/// ordinary TBs (the cell sits at slot centre); non-zero for
/// SphericalWrappedPlane children that are repositioned onto the
/// sphere surface. Subtracted from the slot origin at TB child
/// dispatch (descent) and added back at ribbon pop (exit).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuNodeKind {
    pub kind: u32,
    pub dims_x: u32,
    pub dims_y: u32,
    pub dims_z: u32,
    pub rot_col0: [f32; 4],
    pub rot_col1: [f32; 4],
    pub rot_col2: [f32; 4],
    pub cell_offset: [f32; 4],
}

impl GpuNodeKind {
    pub fn from_node_kind(k: NodeKind) -> Self {
        let id_col0 = [1.0, 0.0, 0.0, 0.0];
        let id_col1 = [0.0, 1.0, 0.0, 0.0];
        let id_col2 = [0.0, 0.0, 1.0, 0.0];
        let zero_off = [0.0, 0.0, 0.0, 0.0];
        match k {
            NodeKind::Cartesian => Self {
                kind: 0, dims_x: 0, dims_y: 0, dims_z: 0,
                rot_col0: id_col0, rot_col1: id_col1, rot_col2: id_col2,
                cell_offset: zero_off,
            },
            NodeKind::WrappedPlane { dims, slab_depth: _ } => Self {
                kind: 1, dims_x: dims[0], dims_y: dims[1], dims_z: dims[2],
                rot_col0: id_col0, rot_col1: id_col1, rot_col2: id_col2,
                cell_offset: zero_off,
            },
            NodeKind::TangentBlock { rotation, cell_offset } => {
                let content_scale = inscribed_cube_scale(&rotation);
                Self {
                    kind: 2, dims_x: 0, dims_y: 0, dims_z: 0,
                    rot_col0: [rotation[0][0], rotation[0][1], rotation[0][2], content_scale],
                    rot_col1: [rotation[1][0], rotation[1][1], rotation[1][2], 0.0],
                    rot_col2: [rotation[2][0], rotation[2][1], rotation[2][2], 0.0],
                    cell_offset: [cell_offset[0], cell_offset[1], cell_offset[2], 0.0],
                }
            },
            NodeKind::SphericalWrappedPlane {
                dims, slab_depth, body_radius_cells, lat_max,
            } => {
                // kind=3. Slab dims in dims_x/y/z. Sphere params in
                // rot_col0 (.x=body_radius, .y=lat_max, .z=slab_depth
                // as f32; .w unused). Other rot_cols / cell_offset
                // unused (zeroed).
                Self {
                    kind: 3,
                    dims_x: dims[0], dims_y: dims[1], dims_z: dims[2],
                    rot_col0: [body_radius_cells, lat_max, slab_depth as f32, 0.0],
                    rot_col1: id_col1,
                    rot_col2: id_col2,
                    cell_offset: zero_off,
                }
            },
        }
    }
}

/// Largest uniform scale factor `s` such that the rotated cube
/// `R · ([0, 3)³ − 1.5) · s + 1.5` fits inside `[0, 3)³`. For each
/// world axis `i`, the rotated cube's half-extent is
/// `1.5 · s · Σ_j |R[j][i]|`. Setting that `≤ 1.5` gives
/// `s ≤ 1 / max_i(Σ_j |R[j][i]|)`. For a 45° Y rotation:
/// `s = 1/√2 ≈ 0.707`.
pub(crate) fn inscribed_cube_scale(r: &[[f32; 3]; 3]) -> f32 {
    let mut max_extent = 0.0f32;
    for i in 0..3 {
        let extent = r[0][i].abs() + r[1][i].abs() + r[2][i].abs();
        max_extent = max_extent.max(extent);
    }
    if max_extent < 1e-6 { 1.0 } else { (1.0 / max_extent).min(1.0) }
}

/// The single TangentBlock transform applied symmetrically across
/// shader, CPU raycast, anchor descent, and `in_frame_rot`.
///
/// **Descent** (parent → TB-storage frame):
///     `p' = R^T · (p − pivot) / tb_scale + pivot`
///     `d' = R^T · d / tb_scale`
/// **Pop** (TB-storage → parent frame), the inverse of descent:
///     `p' = R · (p − pivot) · tb_scale + pivot`
///     `d' = R · d · tb_scale`
///
/// `pivot` is `0.5` when working in unit-cell coords (anchor
/// descent / pop) and `1.5` when working in `[0, 3)³` coords
/// (shader / CPU raycast).
///
/// Rigid rotation + uniform scale → similarity transform → ray
/// parameter `t` is preserved between the two frames. That's why
/// the inner DDA's `sub.t` can be used directly as the world
/// parameter without rescaling.
#[derive(Clone, Copy, Debug)]
pub struct TbBoundary {
    pub r: [[f32; 3]; 3],
    pub tb_scale: f32,
    /// Cell's displacement from its natural slot centre, in
    /// parent-frame `[0, 3)³` units. Zero for ordinary TBs (the cell
    /// sits at slot centre); non-zero for SphericalWrappedPlane
    /// children. Subtracted at descent and added at exit by the
    /// boundary call site (NOT by `enter_point` / `exit_point` —
    /// those still operate purely on rotation+scale about pivot).
    pub cell_offset: [f32; 3],
}

impl TbBoundary {
    /// Build a `TbBoundary` from a `TangentBlock` rotation matrix.
    /// `cell_offset` defaults to zero (slot-centred cell).
    pub fn new(r: [[f32; 3]; 3]) -> Self {
        Self { r, tb_scale: inscribed_cube_scale(&r), cell_offset: [0.0; 3] }
    }

    /// Build from a `NodeKind`; returns `None` for non-TB kinds.
    pub fn from_kind(k: crate::world::tree::NodeKind) -> Option<Self> {
        if let crate::world::tree::NodeKind::TangentBlock { rotation, cell_offset } = k {
            Some(Self {
                r: rotation,
                tb_scale: inscribed_cube_scale(&rotation),
                cell_offset,
            })
        } else {
            None
        }
    }

    /// Descent: parent → TB-storage frame. `p' = R^T·(p−pivot)/s + pivot`.
    pub fn enter_point(&self, p: [f32; 3], pivot: f32) -> [f32; 3] {
        let c = [p[0] - pivot, p[1] - pivot, p[2] - pivot];
        // R^T · c (column-major r[col][row]).
        let rotated = [
            self.r[0][0] * c[0] + self.r[0][1] * c[1] + self.r[0][2] * c[2],
            self.r[1][0] * c[0] + self.r[1][1] * c[1] + self.r[1][2] * c[2],
            self.r[2][0] * c[0] + self.r[2][1] * c[1] + self.r[2][2] * c[2],
        ];
        [
            rotated[0] / self.tb_scale + pivot,
            rotated[1] / self.tb_scale + pivot,
            rotated[2] / self.tb_scale + pivot,
        ]
    }

    /// Descent direction. `d' = R^T·d/s` (no pivot — directions are
    /// translation-invariant).
    pub fn enter_dir(&self, d: [f32; 3]) -> [f32; 3] {
        [
            (self.r[0][0] * d[0] + self.r[0][1] * d[1] + self.r[0][2] * d[2]) / self.tb_scale,
            (self.r[1][0] * d[0] + self.r[1][1] * d[1] + self.r[1][2] * d[2]) / self.tb_scale,
            (self.r[2][0] * d[0] + self.r[2][1] * d[1] + self.r[2][2] * d[2]) / self.tb_scale,
        ]
    }

    /// Pop: TB-storage → parent frame. `p' = R·(p−pivot)·s + pivot`.
    pub fn exit_point(&self, p: [f32; 3], pivot: f32) -> [f32; 3] {
        let c = [
            (p[0] - pivot) * self.tb_scale,
            (p[1] - pivot) * self.tb_scale,
            (p[2] - pivot) * self.tb_scale,
        ];
        // R · c (column-major).
        [
            self.r[0][0] * c[0] + self.r[1][0] * c[1] + self.r[2][0] * c[2] + pivot,
            self.r[0][1] * c[0] + self.r[1][1] * c[1] + self.r[2][1] * c[2] + pivot,
            self.r[0][2] * c[0] + self.r[1][2] * c[1] + self.r[2][2] * c[2] + pivot,
        ]
    }

    /// Pop direction. `d' = R·d·s`.
    pub fn exit_dir(&self, d: [f32; 3]) -> [f32; 3] {
        let s = [d[0] * self.tb_scale, d[1] * self.tb_scale, d[2] * self.tb_scale];
        [
            self.r[0][0] * s[0] + self.r[1][0] * s[1] + self.r[2][0] * s[2],
            self.r[0][1] * s[0] + self.r[1][1] * s[1] + self.r[2][1] * s[2],
            self.r[0][2] * s[0] + self.r[1][2] * s[1] + self.r[2][2] * s[2],
        ]
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
        // 4 u32 (16) + 3 vec4 rotation (48) + 1 vec4 cell_offset (16) = 80 B.
        assert_eq!(std::mem::size_of::<GpuNodeKind>(), 80);
    }

    #[test]
    fn from_node_kind_cartesian() {
        let k = GpuNodeKind::from_node_kind(NodeKind::Cartesian);
        assert_eq!(k.kind, 0);
        assert_eq!(k.dims_x, 0);
        assert_eq!(k.dims_y, 0);
        assert_eq!(k.dims_z, 0);
    }

    #[test]
    fn from_node_kind_wrapped_plane_carries_dims() {
        let k = GpuNodeKind::from_node_kind(NodeKind::WrappedPlane {
            dims: [20, 10, 2],
            slab_depth: 3,
        });
        assert_eq!(k.kind, 1);
        assert_eq!(k.dims_x, 20);
        assert_eq!(k.dims_y, 10);
        assert_eq!(k.dims_z, 2);
    }

    #[test]
    fn from_node_kind_tangent_block_carries_rotation() {
        use crate::world::tree::IDENTITY_ROTATION;
        let k = GpuNodeKind::from_node_kind(NodeKind::TangentBlock {
            rotation: IDENTITY_ROTATION,
            cell_offset: [0.0; 3],
        });
        assert_eq!(k.kind, 2);
        // Identity rotation has inscribed scale 1.0.
        assert_eq!(k.rot_col0, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(k.rot_col1, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(k.rot_col2, [0.0, 0.0, 1.0, 0.0]);

        let r = [[0.5, 0.6, 0.7], [0.1, 0.2, 0.3], [0.9, 0.8, 0.4]];
        let k = GpuNodeKind::from_node_kind(NodeKind::TangentBlock {
            rotation: r,
            cell_offset: [0.0; 3],
        });
        let expected_scale = inscribed_cube_scale(&r);
        assert_eq!(k.rot_col0, [0.5, 0.6, 0.7, expected_scale]);
        assert_eq!(k.rot_col1, [0.1, 0.2, 0.3, 0.0]);
        assert_eq!(k.rot_col2, [0.9, 0.8, 0.4, 0.0]);
        assert_eq!(k.cell_offset, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn from_node_kind_tangent_block_carries_cell_offset() {
        use crate::world::tree::IDENTITY_ROTATION;
        let k = GpuNodeKind::from_node_kind(NodeKind::TangentBlock {
            rotation: IDENTITY_ROTATION,
            cell_offset: [0.7, -0.3, 1.4],
        });
        assert_eq!(k.cell_offset, [0.7, -0.3, 1.4, 0.0]);
    }
}
