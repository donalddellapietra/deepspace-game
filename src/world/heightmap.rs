//! Spatial ground height cache for NPC collision.
//!
//! Lazily caches ground heights per XZ cell using absolute leaf
//! coordinates as keys. Anchor-independent — no invalidation on
//! player movement or jumping. Only invalidated on terrain edits.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::state::WorldState;
use super::view::{
    cell_size_at_layer, is_layer_pos_solid, layer_pos_from_bevy,
    position_to_leaf_coord, target_layer_for, WorldAnchor,
};

/// Maximum tree-walk queries per frame to avoid spikes.
const MAX_QUERIES_PER_FRAME: usize = 64;

/// Spatial ground height cache keyed by absolute leaf XZ coordinates.
#[derive(Resource, Default)]
pub struct GroundCache {
    /// Ground height in bevy-space Y, keyed by absolute leaf (x, z).
    /// The Y value is relative to the current anchor and gets
    /// recomputed on cache miss.
    cache: HashMap<(i64, i64), f32>,
    /// Anchor used for the cached Y values. If anchor changes,
    /// Y values need adjustment (simple offset, no re-raycast).
    cached_anchor_y: i64,
    /// Queries this frame.
    queries_this_frame: usize,
}

impl GroundCache {
    /// Look up ground height at a bevy-space XZ, returning bevy-space Y.
    pub fn ground_y(
        &mut self,
        world: &WorldState,
        anchor: &WorldAnchor,
        view_layer: u8,
        bevy_x: f32,
        bevy_z: f32,
        cell: f32,
    ) -> f32 {
        // Convert bevy-space XZ to absolute leaf coordinates for the key.
        let leaf_x = anchor.leaf_coord[0] + bevy_x.floor() as i64;
        let leaf_z = anchor.leaf_coord[2] + bevy_z.floor() as i64;

        // Quantize to cell-sized buckets in leaf space.
        let extent = cell as i64;
        let key = (
            leaf_x.div_euclid(extent.max(1)),
            leaf_z.div_euclid(extent.max(1)),
        );

        if let Some(&cached_y) = self.cache.get(&key) {
            // Adjust for anchor Y change since this was cached.
            let anchor_dy = (self.cached_anchor_y - anchor.leaf_coord[1]) as f32;
            return cached_y + anchor_dy;
        }

        if self.queries_this_frame >= MAX_QUERIES_PER_FRAME {
            return 0.0;
        }

        let target = target_layer_for(view_layer);
        let y = raycast_ground(world, target, anchor, bevy_x, bevy_z, cell);
        self.cache.insert(key, y);
        self.cached_anchor_y = anchor.leaf_coord[1];
        self.queries_this_frame += 1;
        y
    }

    /// Invalidate cache entries near a bevy-space edit position.
    pub fn invalidate_near(
        &mut self,
        bevy_x: f32,
        bevy_z: f32,
        anchor: &WorldAnchor,
        cell: f32,
        radius: i32,
    ) {
        let leaf_x = anchor.leaf_coord[0] + bevy_x.floor() as i64;
        let leaf_z = anchor.leaf_coord[2] + bevy_z.floor() as i64;
        let extent = cell as i64;
        let cx = leaf_x.div_euclid(extent.max(1));
        let cz = leaf_z.div_euclid(extent.max(1));
        for dz in -(radius as i64)..=(radius as i64) {
            for dx in -(radius as i64)..=(radius as i64) {
                self.cache.remove(&(cx + dx, cz + dz));
            }
        }
    }

    /// Reset the per-frame query counter.
    pub fn reset_frame(&mut self) {
        self.queries_this_frame = 0;
    }
}

/// System that resets the per-frame query counter.
pub fn reset_ground_cache(mut cache: ResMut<GroundCache>) {
    cache.reset_frame();
}

/// Raycast downward to find ground height at a bevy-space XZ position.
fn raycast_ground(
    world: &WorldState,
    target_layer: u8,
    anchor: &WorldAnchor,
    world_x: f32,
    world_z: f32,
    cell: f32,
) -> f32 {
    // Probe from above down to find the highest solid cell.
    let max_cells: i32 = 32;
    for y_cell in (-max_cells..max_cells).rev() {
        let probe_y = y_cell as f32 * cell + cell * 0.5;
        let probe = Vec3::new(world_x, probe_y, world_z);
        if let Some(lp) = layer_pos_from_bevy(probe, target_layer, anchor) {
            if is_layer_pos_solid(world, &lp) {
                // Return the top face of this solid cell.
                return (y_cell + 1) as f32 * cell;
            }
        }
    }
    0.0
}
