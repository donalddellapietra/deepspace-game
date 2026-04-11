//! Procedural generation primitives.
//!
//! The tree's infinite grassland has two leaf content patterns:
//!
//! * **Grass** — a `25³` grid filled with `BlockType::Grass`, used for
//!   leaves whose y-range in root-local coordinates is entirely at or
//!   below [`GROUND_Y_VOXELS`](super::state::GROUND_Y_VOXELS).
//! * **Air** — an empty `25³` grid, used for leaves whose y-range is
//!   entirely above the ground.
//!
//! Because the content is determined by y-range, x and z slots never
//! affect leaf identity. Every layer of the tree collapses to a
//! handful of library entries thanks to content-addressed dedup; see
//! [`WorldState::build_grassland_root`](super::state::WorldState::build_grassland_root)
//! for how the tree is assembled from these two primitives.

use super::tree::{empty_voxel_grid, filled_voxel_grid, voxel_from_block, VoxelGrid};
use crate::block::BlockType;

/// A `25³` grid of solid grass voxels.
pub fn generate_grass_leaf() -> VoxelGrid {
    filled_voxel_grid(voxel_from_block(Some(BlockType::Grass)))
}

/// A `25³` grid of empty (air) voxels.
pub fn generate_air_leaf() -> VoxelGrid {
    empty_voxel_grid()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{EMPTY_VOXEL, NODE_VOXELS};

    #[test]
    fn grass_leaf_is_all_grass() {
        let v = generate_grass_leaf();
        let grass = voxel_from_block(Some(BlockType::Grass));
        assert!(v.iter().all(|&x| x == grass));
    }

    #[test]
    fn air_leaf_is_all_empty() {
        let v = generate_air_leaf();
        assert!(v.iter().all(|&x| x == EMPTY_VOXEL));
        assert_eq!(v.len(), NODE_VOXELS);
    }

    #[test]
    fn grass_and_air_differ() {
        let g = generate_grass_leaf();
        let a = generate_air_leaf();
        assert_ne!(g.as_ref(), a.as_ref());
    }
}
