//! Pre-computed column heightmap for NPC ground collision.
//!
//! Minecraft-style: one height value per XZ column at collision-layer
//! resolution. Pre-computed on startup, updated per-column on edit.
//! O(1) lookup, no frame budgets, no NaN fallbacks.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::state::WorldState;
use super::tree::MAX_LAYER;
use super::view::{
    cell_size_at_layer, is_layer_pos_solid,
    layer_pos_from_leaf_coord_direct, target_layer_for,
    WorldAnchor,
};

/// Column heightmap: absolute leaf Y of the top solid block per XZ column.
#[derive(Resource, Default)]
pub struct GroundHeightmap {
    /// Keyed by (leaf_x / block_size, leaf_z / block_size).
    /// Value is absolute leaf Y of the top face of the highest solid cell,
    /// or `None` if the column is entirely air.
    columns: HashMap<(i64, i64), i64>,
    /// The collision layer this was built for.
    collision_layer: u8,
    /// Block size in leaf voxels at the collision layer.
    block_size: i64,
    /// Whether initial generation is done.
    initialized: bool,
}

impl GroundHeightmap {
    /// O(1) ground height lookup. Returns bevy-space Y in the given anchor frame.
    /// Returns 0.0 if column has no solid blocks.
    pub fn ground_y(&self, anchor: &WorldAnchor, bevy_x: f32, bevy_z: f32) -> f32 {
        if !self.initialized || self.block_size == 0 {
            return 0.0;
        }
        let n = anchor.norm;
        let leaf_x = anchor.leaf_coord[0] + (bevy_x * n).floor() as i64;
        let leaf_z = anchor.leaf_coord[2] + (bevy_z * n).floor() as i64;
        let key = (
            leaf_x.div_euclid(self.block_size),
            leaf_z.div_euclid(self.block_size),
        );
        match self.columns.get(&key) {
            Some(&y) => (y - anchor.leaf_coord[1]) as f32 / n,
            None => 0.0,
        }
    }

    /// Update a single column after a terrain edit.
    /// Call with the bevy-space position of the edited block.
    pub fn update_column_at(
        &mut self,
        world: &WorldState,
        anchor: &WorldAnchor,
        bevy_x: f32,
        bevy_z: f32,
    ) {
        if self.block_size == 0 { return; }
        let n = anchor.norm;
        let leaf_x = anchor.leaf_coord[0] + (bevy_x * n).floor() as i64;
        let leaf_z = anchor.leaf_coord[2] + (bevy_z * n).floor() as i64;
        let key = (
            leaf_x.div_euclid(self.block_size),
            leaf_z.div_euclid(self.block_size),
        );
        let cx = key.0 * self.block_size + self.block_size / 2;
        let cz = key.1 * self.block_size + self.block_size / 2;
        let h = scan_column(world, self.collision_layer, self.block_size, cx, cz);
        if let Some(y) = h {
            self.columns.insert(key, y);
        } else {
            self.columns.remove(&key);
        }
        // Also update neighboring columns (edit might affect adjacent cells)
        for dz in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dz == 0 { continue; }
                let nk = (key.0 + dx, key.1 + dz);
                let nx = nk.0 * self.block_size + self.block_size / 2;
                let nz = nk.1 * self.block_size + self.block_size / 2;
                let nh = scan_column(world, self.collision_layer, self.block_size, nx, nz);
                if let Some(y) = nh {
                    self.columns.insert(nk, y);
                } else {
                    self.columns.remove(&nk);
                }
            }
        }
    }
}

/// System: generate the heightmap on first frame, regenerate on zoom change.
pub fn update_ground_heightmap(
    world: Res<WorldState>,
    anchor: Res<WorldAnchor>,
    zoom: Res<super::CameraZoom>,
    mut heightmap: ResMut<GroundHeightmap>,
) {
    let collision_layer = target_layer_for(zoom.layer);
    let block_size = cell_size_at_layer(collision_layer) as i64;

    // Regenerate if collision layer changed or not initialized.
    if heightmap.initialized && heightmap.collision_layer == collision_layer {
        return;
    }

    heightmap.columns.clear();
    heightmap.collision_layer = collision_layer;
    heightmap.block_size = block_size;

    // Scan the area around the anchor.
    // At leaf layer, RADIUS_VIEW_CELLS=32, so scan 64x64 columns.
    // At coarser layers, fewer columns needed.
    let radius_cells = super::render::RADIUS_VIEW_CELLS as i64;
    let anchor_cx = anchor.leaf_coord[0].div_euclid(block_size);
    let anchor_cz = anchor.leaf_coord[2].div_euclid(block_size);

    for dz in -radius_cells..=radius_cells {
        for dx in -radius_cells..=radius_cells {
            let cx = (anchor_cx + dx) * block_size + block_size / 2;
            let cz = (anchor_cz + dz) * block_size + block_size / 2;
            let key = (anchor_cx + dx, anchor_cz + dz);

            if let Some(y) = scan_column(&world, collision_layer, block_size, cx, cz) {
                heightmap.columns.insert(key, y);
            }
        }
    }

    heightmap.initialized = true;
    info!(
        "Ground heightmap: {} columns at layer {} (block_size={})",
        heightmap.columns.len(), collision_layer, block_size,
    );
}

/// Scan a single XZ column top-down to find the highest solid cell.
/// Probes at the collision layer AND one layer coarser to catch blocks
/// placed at the view layer.
/// Returns the absolute leaf Y of the top face, or None if all air.
fn scan_column(
    world: &WorldState,
    collision_layer: u8,
    block_size: i64,
    leaf_x: i64,
    leaf_z: i64,
) -> Option<i64> {
    // Scan from world top downward. The world is 5^12 voxels tall.
    // At collision_layer resolution, that's 5^12 / block_size cells.
    // But most are empty — scan a reasonable range around ground level.
    //
    // Ground is at GROUND_Y_VOXELS (~244M leaf voxels). Convert to
    // collision-layer cells.
    let ground_cell = super::state::GROUND_Y_VOXELS / block_size;
    let scan_range = 64i64; // cells above/below ground to scan

    let mut best_y: Option<i64> = None;

    for cell_y in (ground_cell - scan_range..=ground_cell + scan_range).rev() {
        let probe_y = cell_y * block_size + block_size / 2;
        let coord = [leaf_x, probe_y, leaf_z];

        if let Some(lp) = layer_pos_from_leaf_coord_direct(coord, collision_layer) {
            if is_layer_pos_solid(world, &lp) {
                best_y = Some((cell_y + 1) * block_size);
                break;
            }
        }
    }

    best_y
}
