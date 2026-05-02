//! Voxel model import pipeline.
//!
//! Converts external voxel formats (`.vox`) into the base-2 tree.
//! The pipeline: parse → color map → tree build → NodeId ready to place.

pub mod vox;
pub mod vxs;
pub mod tree_builder;

/// Dispatch load by file extension. `.vox` → MagicaVoxel format;
/// `.vxs` → our custom sparse format for models exceeding the
/// 256-per-axis `.vox` limit. Both produce a `VoxelModel`.
pub fn load(
    path: &std::path::Path,
    registry: &mut crate::world::palette::ColorRegistry,
) -> Result<VoxelModel, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext.to_ascii_lowercase().as_str() {
        "vxs" => vxs::load(path, registry),
        _ => vox::load_first_model(path, registry),
    }
}

/// A parsed voxel model: flat 3D grid of palette indices.
/// `EMPTY_CELL` = empty/air, other values = palette index from
/// `ColorRegistry` (0 is a real color — Stone — not a sentinel).
pub struct VoxelModel {
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
    /// Flat row-major buffer (x fastest, then y, then z).
    pub data: Vec<u16>,
}

/// Value placed in `VoxelModel::data` for empty cells. Distinct from
/// every palette index so palette 0 (Stone) isn't mistaken for air.
pub const EMPTY_CELL: u16 = u16::MAX;

impl VoxelModel {
    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.size_y + y) * self.size_x + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u16 {
        self.data[self.index(x, y, z)]
    }
}

/// Sparse parsed voxel model — one (x, y, z, palette_idx) per
/// non-empty voxel. Skips the dense grid expansion; the tree build
/// then visits only occupied cells instead of the padded cube (for
/// Sponza-sized scenes this is a 50-80× reduction in visit count).
pub struct SparseVoxelModel {
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
    /// Non-empty voxels as `(x, y, z, palette_idx)`. The palette has
    /// already been merged into the caller-provided `ColorRegistry`
    /// so indices are ready to become `Child::Block`.
    pub voxels: Vec<(u32, u32, u32, u16)>,
}
