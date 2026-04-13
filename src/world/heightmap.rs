//! Heightmap for NPC ground collision.
//!
//! Pre-computes a 2D grid of ground heights from the voxel tree so
//! NPC physics can sample O(1) instead of walking the tree per NPC.
//! Regenerated when the view layer or anchor changes.

use bevy::prelude::*;

use super::state::WorldState;
use super::view::{cell_size_at_layer, is_layer_pos_solid, layer_pos_from_bevy, target_layer_for, WorldAnchor};
use super::render::RADIUS_VIEW_CELLS;

/// Resolution of the heightmap grid per axis.
const HEIGHTMAP_RES: usize = 128;

/// Cached heightmap for NPC ground collision.
#[derive(Resource)]
pub struct NpcHeightmap {
    /// Ground Y in bevy space, indexed [z * HEIGHTMAP_RES + x].
    heights: Vec<f32>,
    /// World-space min corner (bevy coords, anchor-relative).
    world_min: Vec2,
    /// World-space size covered.
    world_size: Vec2,
    /// Cell size used to generate this heightmap.
    cell_size: f32,
    /// Anchor leaf coord when this was generated.
    anchor_coord: [i64; 3],
}

impl Default for NpcHeightmap {
    fn default() -> Self {
        Self {
            heights: Vec::new(),
            world_min: Vec2::ZERO,
            world_size: Vec2::ONE,
            cell_size: 1.0,
            anchor_coord: [0; 3],
        }
    }
}

impl NpcHeightmap {
    /// Sample the ground height at a bevy-space XZ position.
    /// Returns the Y coordinate of the top of the ground.
    pub fn sample(&self, x: f32, z: f32) -> f32 {
        if self.heights.is_empty() {
            return 0.0;
        }
        let u = (x - self.world_min.x) / self.world_size.x;
        let v = (z - self.world_min.y) / self.world_size.y;
        let tx = ((u * HEIGHTMAP_RES as f32) as usize).min(HEIGHTMAP_RES - 1);
        let tz = ((v * HEIGHTMAP_RES as f32) as usize).min(HEIGHTMAP_RES - 1);
        self.heights[tz * HEIGHTMAP_RES + tx]
    }

    /// Whether this heightmap needs regeneration.
    /// Only regenerate when the anchor has moved significantly
    /// (more than 1/4 of the heightmap extent) to avoid per-frame
    /// 16K raycasts while the player walks.
    pub fn needs_regen(&self, anchor: &WorldAnchor) -> bool {
        if self.heights.is_empty() {
            return true;
        }
        let dx = (anchor.leaf_coord[0] - self.anchor_coord[0]).abs();
        let dz = (anchor.leaf_coord[2] - self.anchor_coord[2]).abs();
        // Regenerate when moved more than 1/4 of the heightmap extent.
        let threshold = (self.world_size.x / self.cell_size / 4.0).max(8.0) as i64;
        dx > threshold || dz > threshold
    }
}

/// System that regenerates the heightmap when the anchor moves.
pub fn update_heightmap(
    world: Res<WorldState>,
    anchor: Res<WorldAnchor>,
    zoom: Res<super::CameraZoom>,
    mut heightmap: ResMut<NpcHeightmap>,
) {
    if !heightmap.needs_regen(&anchor) {
        return;
    }

    let cell = cell_size_at_layer(zoom.layer);
    let radius = RADIUS_VIEW_CELLS * cell;
    let target = target_layer_for(zoom.layer);

    let world_min = Vec2::new(-radius, -radius);
    let world_size = Vec2::new(radius * 2.0, radius * 2.0);

    heightmap.heights.resize(HEIGHTMAP_RES * HEIGHTMAP_RES, 0.0);

    for tz in 0..HEIGHTMAP_RES {
        for tx in 0..HEIGHTMAP_RES {
            let u = (tx as f32 + 0.5) / HEIGHTMAP_RES as f32;
            let v = (tz as f32 + 0.5) / HEIGHTMAP_RES as f32;
            let world_x = world_min.x + u * world_size.x;
            let world_z = world_min.y + v * world_size.y;

            let ground_y = raycast_ground(
                &world, target, &anchor, world_x, world_z, cell,
            );
            heightmap.heights[tz * HEIGHTMAP_RES + tx] = ground_y;
        }
    }

    heightmap.world_min = world_min;
    heightmap.world_size = world_size;
    heightmap.cell_size = cell;
    heightmap.anchor_coord = anchor.leaf_coord;

    info!(
        "Heightmap generated: {}x{} at cell_size={:.1}, radius={:.1}",
        HEIGHTMAP_RES, HEIGHTMAP_RES, cell, radius,
    );
}

/// Raycast downward to find ground height at an XZ position.
fn raycast_ground(
    world: &WorldState,
    target_layer: u8,
    anchor: &WorldAnchor,
    world_x: f32,
    world_z: f32,
    cell: f32,
) -> f32 {
    // Probe downward from a reasonable height.
    let max_cells: i32 = 32;
    for y_cell in (-max_cells..max_cells).rev() {
        let probe = Vec3::new(world_x, y_cell as f32 * cell + cell * 0.5, world_z);
        if let Some(lp) = layer_pos_from_bevy(probe, target_layer, anchor) {
            if is_layer_pos_solid(world, &lp) {
                return (y_cell + 1) as f32 * cell;
            }
        }
    }
    0.0
}
