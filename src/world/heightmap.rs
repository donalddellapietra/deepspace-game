//! Spatial ground height cache for NPC collision.
//!
//! Lazily caches ground heights per XZ cell. O(1) lookups after first
//! query. Invalidated surgically when terrain is edited.
//! No bulk regeneration — zero frame-time hitches.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::state::WorldState;
use super::view::{cell_size_at_layer, is_layer_pos_solid, layer_pos_from_bevy, target_layer_for, WorldAnchor};

/// Maximum tree-walk queries per frame to avoid spikes.
const MAX_QUERIES_PER_FRAME: usize = 128;

/// Spatial ground height cache.
#[derive(Resource, Default)]
pub struct GroundCache {
    /// Cached ground heights keyed by XZ cell coordinate.
    cache: HashMap<(i32, i32), f32>,
    /// Number of tree walks done this frame (reset each frame).
    queries_this_frame: usize,
}

impl GroundCache {
    /// Look up the ground height at a bevy-space XZ position.
    /// Returns cached value or queries the tree (up to frame budget).
    /// Falls back to 0.0 if budget exhausted and no cache entry exists.
    pub fn ground_y(
        &mut self,
        world: &WorldState,
        anchor: &WorldAnchor,
        view_layer: u8,
        x: f32,
        z: f32,
        cell: f32,
    ) -> f32 {
        let cx = (x / cell).floor() as i32;
        let cz = (z / cell).floor() as i32;
        let key = (cx, cz);

        if let Some(&y) = self.cache.get(&key) {
            return y;
        }

        // Budget check: don't do too many tree walks per frame.
        if self.queries_this_frame >= MAX_QUERIES_PER_FRAME {
            return 0.0;
        }

        let target = target_layer_for(view_layer);
        let y = raycast_ground(world, target, anchor, x, z, cell);
        self.cache.insert(key, y);
        self.queries_this_frame += 1;
        y
    }

    /// Invalidate cache entries near an edited position.
    /// Call this when terrain is edited at a given bevy-space position.
    pub fn invalidate_near(&mut self, x: f32, z: f32, cell: f32, radius: i32) {
        let cx = (x / cell).floor() as i32;
        let cz = (z / cell).floor() as i32;
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                self.cache.remove(&(cx + dx, cz + dz));
            }
        }
    }

    /// Reset the per-frame query counter. Call once at frame start.
    pub fn reset_frame(&mut self) {
        self.queries_this_frame = 0;
    }
}

/// System that resets the per-frame query counter.
pub fn reset_ground_cache(mut cache: ResMut<GroundCache>) {
    cache.reset_frame();
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
