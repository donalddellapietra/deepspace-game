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

/// Per-packed-node metadata: which `NodeKind` this node is, plus
/// per-kind data the shader needs. Indexed by BFS position — the
/// same `node_index` used in `GpuChild::node_index`.
///
/// 16 bytes per node so the WGSL `array<NodeKindGpu>` aligns cleanly.
/// `kind` discriminant:
/// - 0 = Cartesian: the standard slot-pick DDA arm.
/// - 1 = CubedSphereBody: `inner_r` / `outer_r` carry the sphere
///   shell radii in body cell-local `[0, 1)` units; `face` unused.
/// - 2 = CubedSphereFace: `face` carries the face index (0..5) for
///   the basis lookup; `inner_r` / `outer_r` unused.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuNodeKind {
    pub kind: u32,
    pub face: u32,
    pub inner_r: f32,
    pub outer_r: f32,
}

/// WGSL-side discriminant for [`GpuNodeKind::kind`]. Must stay in
/// sync with `NODE_KIND_*` constants in `bindings.wgsl` — the shader
/// switches on the u32 value.
pub const GPU_NODE_KIND_CARTESIAN: u32 = 0;
pub const GPU_NODE_KIND_CUBED_SPHERE_BODY: u32 = 1;
pub const GPU_NODE_KIND_CUBED_SPHERE_FACE: u32 = 2;

impl GpuNodeKind {
    pub fn from_node_kind(k: NodeKind) -> Self {
        match k {
            NodeKind::Cartesian => Self {
                kind: GPU_NODE_KIND_CARTESIAN,
                face: 0,
                inner_r: 0.0,
                outer_r: 0.0,
            },
            NodeKind::CubedSphereBody { inner_r, outer_r } => Self {
                kind: GPU_NODE_KIND_CUBED_SPHERE_BODY,
                face: 0,
                inner_r,
                outer_r,
            },
            NodeKind::CubedSphereFace { face } => Self {
                kind: GPU_NODE_KIND_CUBED_SPHERE_FACE,
                face: face as u32,
                inner_r: 0.0,
                outer_r: 0.0,
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

/// One entry in the shader-side seam-rotation table. 64 bytes,
/// std430-compatible. Indexed as `seam_table[face * 4 + edge]`
/// where `face ∈ 0..6` (Face discriminant) and
/// `edge ∈ 0..4` (`0 = -u`, `1 = +u`, `2 = -v`, `3 = +v`).
///
/// Mirrors `SeamEntry` in `bindings.wgsl`. The shader reads the
/// neighbor face and then rotates `ray_dir` via `R_seam` to get the
/// ray direction in the neighbor face's orthonormal `(u, v, n)` basis.
///
/// Layout choice: the 3×3 rotation is stored as three `vec4<f32>`
/// rows (with the last lane padded) so the WGSL struct's alignment
/// matches the CPU's `repr(C)` layout exactly without any per-row
/// std140 padding surprises. Each row lives in the first three lanes
/// of a `vec4`; the shader reconstructs the matrix row-by-row.
///
/// Precomputed at renderer init from `SEAM_TABLE` in
/// `src/world/cubesphere/seams.rs`; never mutated per-frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct GpuSeamEntry {
    /// Discriminant of the neighbor `Face` (0..=5).
    pub neighbor_face: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    /// Row 0 of `R_seam` (first 3 lanes), pad in lane 3.
    pub rotation_row0: [f32; 4],
    pub rotation_row1: [f32; 4],
    pub rotation_row2: [f32; 4],
}

/// Build the flat GPU-ready seam table from the CPU-side
/// `SEAM_TABLE`. Produces 24 entries in face-major / edge-minor
/// order — the same `[face * 4 + edge]` indexing the shader uses.
///
/// Called once per renderer at init — the table never changes.
pub fn build_seam_table() -> [GpuSeamEntry; 24] {
    use crate::world::cubesphere::seams::SEAM_TABLE;
    let mut out = [GpuSeamEntry::default(); 24];
    for (fi, face_rows) in SEAM_TABLE.iter().enumerate() {
        for (ei, entry) in face_rows.iter().enumerate() {
            let r = entry.rotation;
            out[fi * 4 + ei] = GpuSeamEntry {
                neighbor_face: entry.neighbor_face as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                rotation_row0: [r[0][0], r[0][1], r[0][2], 0.0],
                rotation_row1: [r[1][0], r[1][1], r[1][2], 0.0],
                rotation_row2: [r[2][0], r[2][1], r[2][2], 0.0],
            };
        }
    }
    out
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
    fn gpu_seam_entry_size() {
        // 64 bytes = 4 u32 header + 3 vec4 rotation rows.
        assert_eq!(std::mem::size_of::<GpuSeamEntry>(), 64);
    }

    #[test]
    fn build_seam_table_roundtrip() {
        use crate::world::cubesphere::seams::SEAM_TABLE;
        let flat = build_seam_table();
        for (fi, face_rows) in SEAM_TABLE.iter().enumerate() {
            for (ei, entry) in face_rows.iter().enumerate() {
                let g = flat[fi * 4 + ei];
                assert_eq!(
                    g.neighbor_face,
                    entry.neighbor_face as u32,
                    "face {fi} edge {ei} neighbor mismatch",
                );
                let r = entry.rotation;
                for col in 0..3 {
                    assert_eq!(g.rotation_row0[col], r[0][col], "row0 col{col}");
                    assert_eq!(g.rotation_row1[col], r[1][col], "row1 col{col}");
                    assert_eq!(g.rotation_row2[col], r[2][col], "row2 col{col}");
                }
            }
        }
        // Every edge must point to a valid face index.
        for entry in flat.iter() {
            assert!(entry.neighbor_face < 6);
        }
    }
}
