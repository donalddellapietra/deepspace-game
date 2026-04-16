//! Voxel model import pipeline.
//!
//! Converts external voxel formats (`.vox`) into the base-3 tree.
//! The pipeline: parse → color map → tree build → NodeId ready to place.

pub mod tree_builder;
pub mod vox;

/// A parsed voxel model: flat 3D grid of palette indices.
/// 0 = empty/air, 1-255 = palette index from `ColorRegistry`.
pub struct VoxelModel {
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
    /// Flat row-major buffer (x fastest, then y, then z).
    pub data: Vec<u8>,
}

impl VoxelModel {
    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.size_y + y) * self.size_x + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        self.data[self.index(x, y, z)]
    }
}
