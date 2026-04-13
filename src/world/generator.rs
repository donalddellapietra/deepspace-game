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

use super::tree::{
    empty_voxel_grid, filled_voxel_grid, voxel_from_block, voxel_idx,
    VoxelGrid, NODE_VOXELS_PER_AXIS,
};
use crate::block::BlockType;

/// A `25³` grid of solid grass voxels.
pub fn generate_grass_leaf() -> VoxelGrid {
    filled_voxel_grid(voxel_from_block(Some(BlockType::Grass)))
}

/// A `25³` grid of empty (air) voxels.
pub fn generate_air_leaf() -> VoxelGrid {
    empty_voxel_grid()
}

// ---------------------------------------------------------- sphere generation

/// Parameters for a spherical planet.
pub struct SphereParams {
    pub center: [i64; 3],
    pub radius: i64,
}

/// Generate a leaf voxel grid by evaluating the sphere density field
/// at each voxel position. Voxels inside the sphere are coloured by
/// depth from the surface: Grass (< 3), Dirt (3–10), Stone (> 10).
pub fn generate_sphere_leaf(leaf_origin: [i64; 3], params: &SphereParams) -> VoxelGrid {
    let mut grid = empty_voxel_grid();
    let r = params.radius as f64;
    for z in 0..NODE_VOXELS_PER_AXIS {
        for y in 0..NODE_VOXELS_PER_AXIS {
            for x in 0..NODE_VOXELS_PER_AXIS {
                let dx = (leaf_origin[0] + x as i64 - params.center[0]) as f64;
                let dy = (leaf_origin[1] + y as i64 - params.center[1]) as f64;
                let dz = (leaf_origin[2] + z as i64 - params.center[2]) as f64;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < r {
                    let depth = r - dist;
                    let block = if depth < 3.0 {
                        BlockType::Grass
                    } else if depth < 10.0 {
                        BlockType::Dirt
                    } else {
                        BlockType::Stone
                    };
                    grid[voxel_idx(x, y, z)] = voxel_from_block(Some(block));
                }
            }
        }
    }
    grid
}

/// True when the axis-aligned box is entirely outside the sphere
/// (no voxel in the box can be inside). Uses the standard
/// closest-point-on-AABB distance test.
pub fn aabb_outside_sphere(
    origin: [i64; 3],
    extent: i64,
    params: &SphereParams,
) -> bool {
    let closest = [
        params.center[0].clamp(origin[0], origin[0] + extent),
        params.center[1].clamp(origin[1], origin[1] + extent),
        params.center[2].clamp(origin[2], origin[2] + extent),
    ];
    let dx = (closest[0] - params.center[0]) as f64;
    let dy = (closest[1] - params.center[1]) as f64;
    let dz = (closest[2] - params.center[2]) as f64;
    dx * dx + dy * dy + dz * dz >= (params.radius as f64) * (params.radius as f64)
}

/// True when the axis-aligned box is entirely inside a sphere of the
/// given `radius` centred at `center` (all 8 corners are within
/// `radius`). The caller may pass a reduced radius to guarantee a
/// minimum depth from the surface.
pub fn aabb_inside_sphere(
    origin: [i64; 3],
    extent: i64,
    center: [i64; 3],
    radius: i64,
) -> bool {
    if radius <= 0 {
        return false;
    }
    let r_sq = (radius as f64) * (radius as f64);
    for &dz in &[0i64, extent] {
        for &dy in &[0i64, extent] {
            for &dx in &[0i64, extent] {
                let px = (origin[0] + dx - center[0]) as f64;
                let py = (origin[1] + dy - center[1]) as f64;
                let pz = (origin[2] + dz - center[2]) as f64;
                if px * px + py * py + pz * pz >= r_sq {
                    return false;
                }
            }
        }
    }
    true
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
