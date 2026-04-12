//! Voxel model import pipeline.
//!
//! Converts external voxel formats (MagicaVoxel `.vox`, etc.) into the
//! game's content-addressed tree at leaf resolution (layer 12). Each
//! voxel in the source file maps to one leaf voxel.
//!
//! The pipeline has three stages:
//!
//! 1. **Parse** — read the file format into a [`VoxelModel`].
//! 2. **Color map** — palette colours are mapped to [`BlockType`] during
//!    parsing (see [`color_map`]).
//! 3. **Stamp** — chunk the model into 25³ leaves and install them into
//!    [`WorldState`] at a caller-supplied position (see [`stamp`]).

pub mod color_map;
pub mod stamp;
pub mod vox;

use crate::world::tree::Voxel;

/// A parsed voxel model ready for stamping into the world.
///
/// Coordinates are **model-local**: `(0, 0, 0)` is the model's
/// minimum corner. Dimensions are in leaf voxels. The flat buffer
/// is row-major with x varying fastest, then y, then z — matching
/// the tree's [`voxel_idx`](crate::world::tree::voxel_idx) layout.
pub struct VoxelModel {
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
    /// Flat `size_x * size_y * size_z` buffer. `EMPTY_VOXEL` (0) for
    /// air / absent voxels.
    pub data: Vec<Voxel>,
}

impl VoxelModel {
    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.size_y + y) * self.size_x + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.data[self.index(x, y, z)]
    }
}
