//! Biome-layered terrain generation.
//!
//! Terrain features exist at specific tree layers: mountains at layer
//! ~8, hills at layer ~10, surface detail at layer ~12. The terrain
//! system generates complete subtrees at the renderer's emit layer and
//! splices them in using [`install_subtree`] — the same code path as
//! block editing. No changes to the renderer or collision systems.
//!
//! Each subtree is built bottom-up: leaves are either classified as
//! all-air / all-solid (and reuse cached uniform towers) or generated
//! with noise-based terrain content. Parent nodes are built via the
//! existing [`downsample_from_library`].

use bevy::prelude::*;
use fastnoise_lite::*;

use super::edit::install_subtree;
use super::render::{CameraZoom, RADIUS_VIEW_CELLS};
use super::state::{world_extent_voxels, WorldState, GROUND_Y_VOXELS};
use super::tree::{
    downsample_from_library, empty_voxel_grid, filled_voxel_grid, slot_coords,
    uniform_children, voxel_from_block, voxel_idx, Children, NodeId, VoxelGrid,
    BRANCH_FACTOR, CHILDREN_PER_NODE, EMPTY_NODE, MAX_LAYER,
    NODE_VOXELS_PER_AXIS,
};
use super::view::{
    cell_size_at_layer, extent_for_layer, target_layer_for, WorldAnchor,
};
use crate::block::BlockType;

// ------------------------------------------------------------ biomes

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Biome {
    Grassland,
    Desert,
    Mountains,
    Tundra,
}

pub fn biome_blocks(biome: Biome) -> (BlockType, BlockType) {
    match biome {
        Biome::Grassland => (BlockType::Grass, BlockType::Dirt),
        Biome::Desert => (BlockType::Sand, BlockType::Sand),
        Biome::Mountains => (BlockType::Stone, BlockType::Stone),
        Biome::Tundra => (BlockType::Dirt, BlockType::Stone),
    }
}

// -------------------------------------------------------- terrain config

// Noise octaves. Each creates features at a specific tree-layer scale.
// Wavelength ≈ leaf voxels per noise cycle.
// Amplitude ≈ height variation in leaf voxels.
//
// At layer K, one cell = 5^(12-K) leaves. Features are visible when
// amplitude >= a few cells at that layer.

const MOUNTAIN_AMP: f64 = 800.0;
const MOUNTAIN_WAVELENGTH: f64 = 100_000.0;

const RIDGE_AMP: f64 = 200.0;
const RIDGE_WAVELENGTH: f64 = 10_000.0;

const HILL_AMP: f64 = 40.0;
const HILL_WAVELENGTH: f64 = 1_000.0;

const DETAIL_AMP: f64 = 8.0;
const DETAIL_WAVELENGTH: f64 = 200.0;

const MICRO_AMP: f64 = 2.0;
const MICRO_WAVELENGTH: f64 = 40.0;

pub struct TerrainConfig {
    mountain_noise: FastNoiseLite,
    ridge_noise: FastNoiseLite,
    hill_noise: FastNoiseLite,
    detail_noise: FastNoiseLite,
    micro_noise: FastNoiseLite,
    temperature_noise: FastNoiseLite,
    moisture_noise: FastNoiseLite,
    pub ground_y: f64,
}

impl TerrainConfig {
    pub fn new(seed: i32, ground_y: i64) -> Self {
        let make = |s: i32, freq: f32| -> FastNoiseLite {
            let mut n = FastNoiseLite::with_seed(s);
            n.set_noise_type(Some(NoiseType::OpenSimplex2));
            n.set_frequency(Some(freq));
            n
        };
        Self {
            mountain_noise: make(seed, 1.0 / MOUNTAIN_WAVELENGTH as f32),
            ridge_noise: make(seed.wrapping_add(1), 1.0 / RIDGE_WAVELENGTH as f32),
            hill_noise: make(seed.wrapping_add(2), 1.0 / HILL_WAVELENGTH as f32),
            detail_noise: make(seed.wrapping_add(3), 1.0 / DETAIL_WAVELENGTH as f32),
            micro_noise: make(seed.wrapping_add(4), 1.0 / MICRO_WAVELENGTH as f32),
            temperature_noise: make(seed.wrapping_add(100), 1.0 / 200_000.0),
            moisture_noise: make(seed.wrapping_add(200), 1.0 / 150_000.0),
            ground_y: ground_y as f64,
        }
    }

    /// Terrain surface height at (x, z) in leaf voxels.
    pub fn terrain_height(&self, x: f64, z: f64) -> f64 {
        let xf = x as f32;
        let zf = z as f32;
        self.ground_y
            + self.mountain_noise.get_noise_2d(xf, zf) as f64 * MOUNTAIN_AMP
            + self.ridge_noise.get_noise_2d(xf, zf) as f64 * RIDGE_AMP
            + self.hill_noise.get_noise_2d(xf, zf) as f64 * HILL_AMP
            + self.detail_noise.get_noise_2d(xf, zf) as f64 * DETAIL_AMP
            + self.micro_noise.get_noise_2d(xf, zf) as f64 * MICRO_AMP
    }

    pub fn biome_at(&self, x: f64, z: f64) -> Biome {
        let temp = self.temperature_noise.get_noise_2d(x as f32, z as f32);
        let moist = self.moisture_noise.get_noise_2d(x as f32, z as f32);
        if temp > 0.0 {
            if moist > 0.0 { Biome::Grassland } else { Biome::Desert }
        } else if moist > 0.0 {
            Biome::Mountains
        } else {
            Biome::Tundra
        }
    }

    /// Min and max terrain height across the XZ footprint of a node.
    /// Samples corners + centre. Conservative: adds amplitude of
    /// unsampled noise octaves as margin.
    fn height_bounds(&self, x0: f64, z0: f64, extent: f64) -> (f64, f64) {
        let samples = [
            self.terrain_height(x0, z0),
            self.terrain_height(x0 + extent, z0),
            self.terrain_height(x0, z0 + extent),
            self.terrain_height(x0 + extent, z0 + extent),
            self.terrain_height(x0 + extent * 0.5, z0 + extent * 0.5),
        ];
        let lo = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Margin for unsampled variation within the node.
        let margin = octave_margin(extent, MOUNTAIN_WAVELENGTH, MOUNTAIN_AMP)
            + octave_margin(extent, RIDGE_WAVELENGTH, RIDGE_AMP)
            + octave_margin(extent, HILL_WAVELENGTH, HILL_AMP)
            + octave_margin(extent, DETAIL_WAVELENGTH, DETAIL_AMP)
            + octave_margin(extent, MICRO_WAVELENGTH, MICRO_AMP);
        (lo - margin, hi + margin)
    }
}

fn octave_margin(extent: f64, wavelength: f64, amplitude: f64) -> f64 {
    amplitude * (extent / wavelength).min(1.0)
}

// ---------------------------------------------- subtree generation

/// Classification of a node's relationship to the terrain surface.
enum NodeClass {
    AllAir,
    AllSolid,
    Surface,
}

/// Classify a node: is it entirely above, entirely below, or does
/// it straddle the terrain surface?
fn classify_node(
    config: &TerrainConfig,
    origin: [i64; 3],
    extent: i64,
) -> NodeClass {
    let (h_min, h_max) = config.height_bounds(
        origin[0] as f64,
        origin[2] as f64,
        extent as f64,
    );
    let node_y_min = origin[1] as f64;
    let node_y_max = (origin[1] + extent) as f64;
    if node_y_min >= h_max {
        NodeClass::AllAir
    } else if node_y_max <= h_min {
        NodeClass::AllSolid
    } else {
        NodeClass::Surface
    }
}

/// Generate a terrain leaf (25³ voxel grid) at the given world origin.
pub fn generate_terrain_leaf(
    config: &TerrainConfig,
    origin: [i64; 3],
) -> VoxelGrid {
    let mut grid = empty_voxel_grid();
    let n = NODE_VOXELS_PER_AXIS;
    for lz in 0..n {
        for lx in 0..n {
            let wx = origin[0] + lx as i64;
            let wz = origin[2] + lz as i64;
            let h = config.terrain_height(wx as f64, wz as f64);
            let biome = config.biome_at(wx as f64, wz as f64);
            let (surface_bt, sub_bt) = biome_blocks(biome);
            let sv = voxel_from_block(Some(surface_bt));
            let subv = voxel_from_block(Some(sub_bt));
            let sy = h.floor() as i64;
            for ly in 0..n {
                let wy = origin[1] + ly as i64;
                if wy > sy {
                    // air
                } else if wy == sy {
                    grid[voxel_idx(lx, ly, lz)] = sv;
                } else {
                    grid[voxel_idx(lx, ly, lz)] = subv;
                }
            }
        }
    }
    grid
}

/// Build a complete terrain subtree from `layer` down to MAX_LAYER.
/// Returns the NodeId of the root of the subtree.
///
/// Uses classification to shortcut: all-air and all-solid nodes reuse
/// the cached uniform towers from WorldState. Only surface-crossing
/// leaves get noise-generated content.
pub fn build_terrain_subtree(
    world: &mut WorldState,
    config: &TerrainConfig,
    origin: [i64; 3],
    layer: u8,
) -> NodeId {
    let extent = layer_extent(layer);

    match classify_node(config, origin, extent) {
        NodeClass::AllAir => {
            return world.air_tower[layer as usize];
        }
        NodeClass::AllSolid => {
            let biome = config.biome_at(
                origin[0] as f64 + extent as f64 * 0.5,
                origin[2] as f64 + extent as f64 * 0.5,
            );
            let (_, sub_bt) = biome_blocks(biome);
            return world.solid_tower(&sub_bt, layer);
        }
        NodeClass::Surface => {}
    }

    // Surface node: if leaf, generate voxels directly.
    if layer == MAX_LAYER {
        let grid = generate_terrain_leaf(config, origin);
        return world.library.insert_leaf(grid);
    }

    // Non-leaf surface node: recurse into 125 children, then
    // build this node from their downsample.
    let child_ext = extent / BRANCH_FACTOR as i64;
    let mut child_ids = [EMPTY_NODE; CHILDREN_PER_NODE];
    for slot in 0..CHILDREN_PER_NODE {
        let (sx, sy, sz) = slot_coords(slot);
        let child_origin = [
            origin[0] + sx as i64 * child_ext,
            origin[1] + sy as i64 * child_ext,
            origin[2] + sz as i64 * child_ext,
        ];
        child_ids[slot] = build_terrain_subtree(
            world, config, child_origin, layer + 1,
        );
    }
    let children: Children = Box::new(child_ids);
    let voxels = downsample_from_library(&world.library, children.as_ref());
    world.library.insert_non_leaf(voxels, children)
}

/// Leaf-voxel extent of a node at `layer`.
fn layer_extent(layer: u8) -> i64 {
    let mut n = NODE_VOXELS_PER_AXIS as i64;
    for _ in 0..(MAX_LAYER - layer) {
        n *= BRANCH_FACTOR as i64;
    }
    n
}

// ---------------------------------------------- per-frame system

const GENERATION_BUDGET: usize = 8;

struct PristineNode {
    path: Vec<u8>,
    origin: [i64; 3],
}

/// Per-frame Bevy system: walk the tree to the emit layer, find
/// pristine nodes near the camera, generate terrain subtrees, and
/// splice them in via install_subtree.
pub fn terrain_generation_system(
    mut world: ResMut<WorldState>,
    anchor: Res<WorldAnchor>,
    zoom: Res<CameraZoom>,
    camera_q: Query<&Transform, With<Camera3d>>,
) {
    let (grass_id, air_id) = match (world.grass_leaf_id, world.air_leaf_id) {
        (Some(g), Some(a)) => (g, a),
        _ => return,
    };
    if world.terrain.is_none() {
        return;
    }
    let Ok(camera_tf) = camera_q.single() else { return };
    let camera_pos = camera_tf.translation;

    let target_layer = target_layer_for(zoom.layer);
    let emit_layer = target_layer.saturating_sub(1);
    let radius_bevy = RADIUS_VIEW_CELLS * cell_size_at_layer(zoom.layer);
    let radius_sq = radius_bevy * radius_bevy;
    let camera_coord = anchor.leaf_coord;

    // Phase 1: walk tree to emit_layer, collect pristine nodes.
    let mut pristine: Vec<PristineNode> = Vec::new();
    {
        struct Frame {
            node_id: NodeId,
            path: Vec<u8>,
            origin: [i64; 3],
            depth: u8,
        }
        let mut stack: Vec<Frame> = Vec::new();
        stack.push(Frame {
            node_id: world.root,
            path: Vec::new(),
            origin: [0; 3],
            depth: 0,
        });

        while let Some(frame) = stack.pop() {
            if pristine.len() >= GENERATION_BUDGET {
                break;
            }

            // AABB-sphere cull.
            let ext = extent_for_layer(frame.depth);
            let ax = (frame.origin[0] - camera_coord[0]) as f32;
            let ay = (frame.origin[1] - camera_coord[1]) as f32;
            let az = (frame.origin[2] - camera_coord[2]) as f32;
            let dx = (ax - camera_pos.x).max(0.0).max(camera_pos.x - (ax + ext));
            let dy = (ay - camera_pos.y).max(0.0).max(camera_pos.y - (ay + ext));
            let dz = (az - camera_pos.z).max(0.0).max(camera_pos.z - (az + ext));
            if dx * dx + dy * dy + dz * dz > radius_sq {
                continue;
            }

            // At emit layer: check if this node is pristine.
            if frame.depth == emit_layer {
                if is_pristine_subtree(&world, frame.node_id, grass_id, air_id) {
                    pristine.push(PristineNode {
                        path: frame.path,
                        origin: frame.origin,
                    });
                }
                continue;
            }

            // Descend into children.
            let Some(node) = world.library.get(frame.node_id) else { continue };
            let Some(children) = node.children.as_ref() else { continue };
            let child_ext = layer_extent(frame.depth + 1);
            for slot in 0..CHILDREN_PER_NODE {
                let cid = children[slot];
                if cid == EMPTY_NODE { continue; }
                let (sx, sy, sz) = slot_coords(slot);
                let mut child_path = frame.path.clone();
                child_path.push(slot as u8);
                stack.push(Frame {
                    node_id: cid,
                    path: child_path,
                    origin: [
                        frame.origin[0] + sx as i64 * child_ext,
                        frame.origin[1] + sy as i64 * child_ext,
                        frame.origin[2] + sz as i64 * child_ext,
                    ],
                    depth: frame.depth + 1,
                });
            }
        }
    }

    if pristine.is_empty() {
        return;
    }

    // Phase 2: generate and install.
    let terrain = world.terrain.take().unwrap();
    for node in &pristine {
        let subtree_id = build_terrain_subtree(
            &mut world,
            &terrain,
            node.origin,
            emit_layer,
        );
        install_subtree(&mut world, &node.path, subtree_id);
    }
    world.terrain = Some(terrain);
}

/// Check whether a node (and its descendants) are all pristine
/// grassland patterns. A node is pristine if it's one of the grass
/// or air leaf IDs, or if ALL of its children are pristine.
fn is_pristine_subtree(
    world: &WorldState,
    node_id: NodeId,
    grass_id: NodeId,
    air_id: NodeId,
) -> bool {
    if node_id == grass_id || node_id == air_id {
        return true;
    }
    let Some(node) = world.library.get(node_id) else { return false };
    let Some(children) = node.children.as_ref() else { return false };
    // A non-leaf is pristine if all its children are pristine.
    // The grassland world is built from only 2 leaf patterns and
    // uniform/mixed parents. Every non-leaf in the grassland is
    // composed entirely of those 2 patterns. An edited or terrain-
    // generated subtree will have at least one non-pristine child.
    children.iter().all(|&cid| is_pristine_subtree(world, cid, grass_id, air_id))
}

// ---------------------------------- test helper (no Bevy ECS needed)

/// Generate terrain for all pristine nodes at `target_layer` within
/// the subtree rooted at `path_prefix`. Returns the number of nodes
/// replaced. Used by tests.
pub fn generate_terrain_in_area(
    world: &mut WorldState,
    target_layer: u8,
    path_prefix: &[u8],
    budget: usize,
) -> usize {
    let (grass_id, air_id) = match (world.grass_leaf_id, world.air_leaf_id) {
        (Some(g), Some(a)) => (g, a),
        _ => return 0,
    };
    if world.terrain.is_none() {
        return 0;
    }

    // Walk from root following path_prefix to find the subtree root.
    let mut node_id = world.root;
    let mut origin: [i64; 3] = [0; 3];
    let mut extent = world_extent_voxels();
    for &slot in path_prefix {
        let child_ext = extent / BRANCH_FACTOR as i64;
        let (sx, sy, sz) = slot_coords(slot as usize);
        origin[0] += sx as i64 * child_ext;
        origin[1] += sy as i64 * child_ext;
        origin[2] += sz as i64 * child_ext;
        extent = child_ext;
        let Some(node) = world.library.get(node_id) else { return 0 };
        let Some(children) = node.children.as_ref() else { return 0 };
        node_id = children[slot as usize];
    }
    let start_depth = path_prefix.len() as u8;

    // Collect pristine nodes at target_layer.
    struct Frame { node_id: NodeId, path: Vec<u8>, origin: [i64; 3], depth: u8 }
    let mut pristine: Vec<PristineNode> = Vec::new();
    let mut stack: Vec<Frame> = vec![Frame {
        node_id,
        path: path_prefix.to_vec(),
        origin,
        depth: start_depth,
    }];

    while let Some(f) = stack.pop() {
        if pristine.len() >= budget { break; }
        if f.depth == target_layer {
            if is_pristine_subtree(world, f.node_id, grass_id, air_id) {
                pristine.push(PristineNode { path: f.path, origin: f.origin });
            }
            continue;
        }
        let Some(node) = world.library.get(f.node_id) else { continue };
        let Some(children) = node.children.as_ref() else { continue };
        let c_ext = layer_extent(f.depth + 1);
        for slot in 0..CHILDREN_PER_NODE {
            let cid = children[slot];
            if cid == EMPTY_NODE { continue; }
            let (sx, sy, sz) = slot_coords(slot);
            let mut cp = f.path.clone();
            cp.push(slot as u8);
            stack.push(Frame {
                node_id: cid, path: cp,
                origin: [
                    f.origin[0] + sx as i64 * c_ext,
                    f.origin[1] + sy as i64 * c_ext,
                    f.origin[2] + sz as i64 * c_ext,
                ],
                depth: f.depth + 1,
            });
        }
    }

    let count = pristine.len();
    if count == 0 { return 0; }

    let terrain = world.terrain.take().unwrap();
    for n in &pristine {
        let subtree_id = build_terrain_subtree(
            world, &terrain, n.origin, target_layer,
        );
        install_subtree(world, &n.path, subtree_id);
    }
    world.terrain = Some(terrain);
    count
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::edit::get_voxel;
    use crate::world::position::Position;
    use crate::world::tree::{voxel_from_block, EMPTY_VOXEL};
    use crate::world::view::{
        is_layer_pos_solid, layer_pos_from_leaf_coord_direct,
        position_from_leaf_coord, position_to_leaf_coord,
    };

    #[test]
    fn terrain_height_is_near_ground() {
        let config = TerrainConfig::new(42, GROUND_Y_VOXELS);
        let h = config.terrain_height(0.0, 0.0);
        let max_dev = MOUNTAIN_AMP + RIDGE_AMP + HILL_AMP + DETAIL_AMP + MICRO_AMP;
        let gy = GROUND_Y_VOXELS as f64;
        assert!(h >= gy - max_dev && h <= gy + max_dev, "h={h} gy={gy}");
    }

    #[test]
    fn leaf_at_surface_has_solid_and_air() {
        let config = TerrainConfig::new(42, GROUND_Y_VOXELS);
        let h = config.terrain_height(0.0, 0.0);
        let sy = h.floor() as i64;
        let origin = [0i64, sy - 12, 0i64];
        let grid = generate_terrain_leaf(&config, origin);
        let mut has_solid = false;
        let mut has_air = false;
        for y in 0..NODE_VOXELS_PER_AXIS {
            let v = grid[voxel_idx(12, y, 12)];
            if v == EMPTY_VOXEL { has_air = true; } else { has_solid = true; }
        }
        assert!(has_solid && has_air);
    }

    #[test]
    fn new_terrain_has_leaf_ids() {
        let world = WorldState::new_terrain(42);
        assert!(world.grass_leaf_id.is_some());
        assert!(world.air_leaf_id.is_some());
        assert!(world.terrain.is_some());
    }

    #[test]
    fn new_terrain_is_valid_grassland_initially() {
        let world = WorldState::new_terrain(42);
        let grassland = WorldState::new_grassland();
        assert_eq!(world.library.len(), grassland.library.len());
    }

    #[test]
    fn build_terrain_subtree_at_leaf_layer() {
        let mut world = WorldState::new_terrain(42);
        let config = world.terrain.take().unwrap();
        // Build a single leaf at a known position.
        let h = config.terrain_height(0.0, 0.0);
        let sy = h.floor() as i64;
        let origin = [0i64, sy - 12, 0i64];
        let id = build_terrain_subtree(&mut world, &config, origin, MAX_LAYER);
        // Should be a leaf (no children).
        let node = world.library.get(id).unwrap();
        assert!(node.children.is_none());
        // Should have mixed content (surface-crossing).
        let mut has_solid = false;
        let mut has_air = false;
        for i in 0..node.voxels.len() {
            if node.voxels[i] == EMPTY_VOXEL { has_air = true; } else { has_solid = true; }
        }
        assert!(has_solid && has_air);
        world.terrain = Some(config);
    }

    #[test]
    fn build_terrain_subtree_at_higher_layer() {
        let mut world = WorldState::new_terrain(42);
        let config = world.terrain.take().unwrap();
        // Build a layer-11 subtree (contains 5³=125 leaves).
        let h = config.terrain_height(0.0, 0.0);
        let sy = h.floor() as i64;
        // Put the subtree so it straddles the surface.
        let origin = [0i64, sy - 60, 0i64];
        let id = build_terrain_subtree(&mut world, &config, origin, MAX_LAYER - 1);
        // Should be a non-leaf with children.
        let node = world.library.get(id).unwrap();
        assert!(node.children.is_some());
        // Its voxels (downsample) should have mixed content.
        let mut has_solid = false;
        let mut has_air = false;
        for i in 0..node.voxels.len() {
            if node.voxels[i] == EMPTY_VOXEL { has_air = true; } else { has_solid = true; }
        }
        assert!(has_solid && has_air);
        world.terrain = Some(config);
    }

    #[test]
    fn build_terrain_subtree_all_air() {
        let mut world = WorldState::new_terrain(42);
        let config = world.terrain.take().unwrap();
        // A node far above the terrain should be classified AllAir.
        let origin = [0i64, GROUND_Y_VOXELS + 100_000, 0i64];
        let id = build_terrain_subtree(&mut world, &config, origin, MAX_LAYER);
        assert_eq!(id, world.air_tower[MAX_LAYER as usize]);
        world.terrain = Some(config);
    }

    #[test]
    fn generate_terrain_in_area_replaces_nodes() {
        let mut world = WorldState::new_terrain(42);
        let initial_root = world.root;
        let initial_len = world.library.len();
        // Generate at layer 11 in the ground-crossing region.
        // Path [slot(2,0,2)] * 4 puts us near the world centre.
        let path: Vec<u8> = vec![
            crate::world::tree::slot_index(2, 0, 2) as u8; 4
        ];
        let count = generate_terrain_in_area(&mut world, 11, &path, 100);
        assert!(count > 0, "should find pristine nodes");
        assert_ne!(world.root, initial_root);
        assert!(world.library.len() > initial_len);
    }

    #[test]
    fn terrain_ground_solid_at_every_view_layer() {
        let mut world = WorldState::new_terrain(42);

        // Get terrain height at world centre.
        let spawn = crate::player::spawn_position();
        let leaf_coord = position_to_leaf_coord(&spawn);
        let h = world.terrain.as_ref().unwrap()
            .terrain_height(leaf_coord[0] as f64, leaf_coord[2] as f64);
        let sy = h.floor() as i64;

        // Generate terrain near spawn at multiple layers (layer 9
        // covers layers 9-12, which is what the renderer would see
        // at zoom levels 7+).
        let path: Vec<u8> = spawn.path[..6].to_vec();
        generate_terrain_in_area(&mut world, 9, &path, 1000);

        // Probe below the terrain surface.
        let probe = [leaf_coord[0], sy - 1, leaf_coord[2]];

        for view_layer in 2..=MAX_LAYER {
            let target = target_layer_for(view_layer);
            if let Some(lp) = layer_pos_from_leaf_coord_direct(probe, target) {
                assert!(
                    is_layer_pos_solid(&world, &lp),
                    "ground should be solid at target layer {target} (view {view_layer})"
                );
            }
        }
    }
}
