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

// Single noise octave for terrain. Amplitude in leaf voxels.
// Kept modest so the surface band is thin and generation is fast
// in debug builds (~78M voxels × 1 noise call vs 3).
const TERRAIN_AMP: f64 = 40.0;
const TERRAIN_FREQ: f32 = 1.0 / 400.0;

/// Maximum possible terrain offset from the base radius.
pub const MAX_TERRAIN_AMPLITUDE: f64 = TERRAIN_AMP;

/// 3D noise generator for terrain. Sampled at center-relative voxel
/// positions, so features wrap the sphere naturally with no seams
/// and f32 precision is perfect (coords in ±600 range).
pub struct TerrainNoise {
    noise: FastNoiseLite,
}

impl TerrainNoise {
    pub fn new(seed: i32) -> Self {
        let mut noise = FastNoiseLite::with_seed(seed);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(TERRAIN_FREQ));
        Self { noise }
    }

    #[inline]
    pub fn offset(&self, x: f32, y: f32, z: f32) -> f64 {
        self.noise.get_noise_3d(x, y, z) as f64 * TERRAIN_AMP
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
    let amp = MAX_TERRAIN_AMPLITUDE;
    let stone_v = voxel_from_block(Some(BlockType::Stone));
    let water_v = voxel_from_block(Some(BlockType::Water));
    let has_terrain = params.terrain.is_some();

    for z in 0..NODE_VOXELS_PER_AXIS {
        for y in 0..NODE_VOXELS_PER_AXIS {
            for x in 0..NODE_VOXELS_PER_AXIS {
                let dx = (leaf_origin[0] + x as i64 - params.center[0]) as f64;
                let dy = (leaf_origin[1] + y as i64 - params.center[1]) as f64;
                let dz = (leaf_origin[2] + z as i64 - params.center[2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                // Early exit: skip noise for voxels clearly outside
                // or clearly deep inside the surface band.
                if has_terrain {
                    let outer = r + amp;
                    if dist_sq >= outer * outer {
                        continue; // definitely air
                    }
                    let inner = r - amp - 10.0;
                    if inner > 0.0 && dist_sq < inner * inner {
                        grid[voxel_idx(x, y, z)] = stone_v;
                        continue; // definitely deep stone
                    }
                }

                let dist = dist_sq.sqrt();

                let terrain_r = match &params.terrain {
                    Some(t) => r + t.offset(dx as f32, dy as f32, dz as f32),
                    None => r,
                };

                if dist < terrain_r {
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
                    grid[voxel_idx(x, y, z)] = water_v;
                }
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
