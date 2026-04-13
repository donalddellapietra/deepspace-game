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

use fastnoise_lite::*;

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

// ---------------------------------------------------------- terrain noise

// Noise octaves. Amplitude in leaf voxels, wavelength in leaf voxels.
// Features at each scale become visible at the layer whose cell size
// is comparable to the wavelength. All amplitudes stay below layer-9
// cell size (3125 voxels) so terrain is invisible at layer 8 and below.
const MOUNTAIN_AMP: f64 = 80.0;
const MOUNTAIN_FREQ: f32 = 1.0 / 800.0;

const HILL_AMP: f64 = 20.0;
const HILL_FREQ: f32 = 1.0 / 200.0;

const DETAIL_AMP: f64 = 5.0;
const DETAIL_FREQ: f32 = 1.0 / 50.0;

/// Maximum possible terrain offset from the base radius. The AABB
/// culling checks use this as a margin so nodes near the surface
/// aren't incorrectly classified as all-air or all-solid.
pub const MAX_TERRAIN_AMPLITUDE: f64 = MOUNTAIN_AMP + HILL_AMP + DETAIL_AMP;

/// 3D noise generators for terrain. Sampled at world-space voxel
/// positions, so features wrap the sphere naturally with no seams.
pub struct TerrainNoise {
    mountain: FastNoiseLite,
    hill: FastNoiseLite,
    detail: FastNoiseLite,
}

impl TerrainNoise {
    pub fn new(seed: i32) -> Self {
        let make = |s: i32, freq: f32| {
            let mut n = FastNoiseLite::with_seed(s);
            n.set_noise_type(Some(NoiseType::OpenSimplex2));
            n.set_frequency(Some(freq));
            n
        };
        Self {
            mountain: make(seed, MOUNTAIN_FREQ),
            hill: make(seed.wrapping_add(1), HILL_FREQ),
            detail: make(seed.wrapping_add(2), DETAIL_FREQ),
        }
    }

    /// Terrain offset at a world-space position. Positive = surface
    /// pushed outward (mountain), negative = pushed inward (valley/ocean).
    pub fn offset(&self, x: f32, y: f32, z: f32) -> f64 {
        self.mountain.get_noise_3d(x, y, z) as f64 * MOUNTAIN_AMP
            + self.hill.get_noise_3d(x, y, z) as f64 * HILL_AMP
            + self.detail.get_noise_3d(x, y, z) as f64 * DETAIL_AMP
    }
}

// ---------------------------------------------------------- sphere generation

/// Parameters for a spherical planet.
pub struct SphereParams {
    pub center: [i64; 3],
    pub radius: i64,
    pub terrain: Option<TerrainNoise>,
}

/// Generate a leaf voxel grid by evaluating the sphere density field.
///
/// The surface radius at each voxel is `radius + terrain_offset(p)`.
/// Sea level equals the base `radius`. Where terrain dips below sea
/// level, water fills the gap. Block type depends on depth from the
/// terrain surface: Grass (< 3), Dirt (3–10), Stone (> 10).
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

                // Terrain offset pushes surface in/out from base radius.
                let terrain_r = match &params.terrain {
                    Some(t) => {
                        let wx = (leaf_origin[0] + x as i64) as f32;
                        let wy = (leaf_origin[1] + y as i64) as f32;
                        let wz = (leaf_origin[2] + z as i64) as f32;
                        r + t.offset(wx, wy, wz)
                    }
                    None => r,
                };

                if dist < terrain_r {
                    // Inside terrain surface.
                    let depth = terrain_r - dist;
                    let block = if depth < 3.0 {
                        BlockType::Grass
                    } else if depth < 10.0 {
                        BlockType::Dirt
                    } else {
                        BlockType::Stone
                    };
                    grid[voxel_idx(x, y, z)] = voxel_from_block(Some(block));
                } else if dist < r {
                    // Above terrain but below sea level → water.
                    grid[voxel_idx(x, y, z)] = voxel_from_block(Some(BlockType::Water));
                }
                // else: air (above sea level)
            }
        }
    }
    grid
}

/// True when the AABB is entirely outside the maximum possible
/// surface (radius + max terrain amplitude). Uses the standard
/// closest-point-on-AABB distance test.
pub fn aabb_outside_sphere(
    origin: [i64; 3],
    extent: i64,
    params: &SphereParams,
) -> bool {
    let max_r = params.radius as f64
        + if params.terrain.is_some() { MAX_TERRAIN_AMPLITUDE } else { 0.0 };
    let closest = [
        params.center[0].clamp(origin[0], origin[0] + extent),
        params.center[1].clamp(origin[1], origin[1] + extent),
        params.center[2].clamp(origin[2], origin[2] + extent),
    ];
    let dx = (closest[0] - params.center[0]) as f64;
    let dy = (closest[1] - params.center[1]) as f64;
    let dz = (closest[2] - params.center[2]) as f64;
    dx * dx + dy * dy + dz * dz >= max_r * max_r
}

/// True when the axis-aligned box is entirely inside a sphere of the
/// given `radius` centred at `center` (all 8 corners are within
/// `radius`). The caller passes a reduced radius to guarantee a
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
