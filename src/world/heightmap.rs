//! Spatial ground height cache for NPC collision.
//!
//! Lazily caches ground heights per XZ cell using absolute leaf
//! coordinates as keys. Anchor-independent -- no invalidation on
//! player movement or jumping. Only invalidated on terrain edits.
//!
//! Ground heights are stored as **absolute leaf Y coordinates** (`i64`)
//! so they never depend on which anchor was active at query time.
//! Conversion to bevy-space Y is done on the fly at the call site.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::state::WorldState;
use super::tree::MAX_LAYER;
use super::view::{
    cell_size_at_layer, is_layer_pos_solid,
    layer_pos_from_leaf_coord_direct, WorldAnchor,
};

/// Maximum tree-walk queries per frame to avoid spikes.
const MAX_QUERIES_PER_FRAME: usize = 64;

/// Spatial ground height cache keyed by absolute leaf XZ coordinates.
///
/// Stores ground height as an **absolute leaf Y coordinate** so the
/// cached value never needs anchor-relative adjustment.
#[derive(Resource, Default)]
pub struct GroundCache {
    /// Ground height as absolute leaf Y, keyed by quantised absolute
    /// leaf (x, z) at the collision-layer grid pitch.
    cache: HashMap<(i64, i64), i64>,
    /// Collision layer the cache was populated at. If the player
    /// changes zoom and the collision layer changes, the cache must
    /// be flushed because the solidity grid is different.
    cached_collision_layer: u8,
    /// Queries this frame.
    queries_this_frame: usize,
}

impl GroundCache {
    /// Look up ground height at a bevy-space XZ, returning bevy-space Y.
    ///
    /// `view_layer` is the current zoom layer. The collision layer is
    /// derived as `(view_layer + 1).min(MAX_LAYER)`, matching
    /// `collision.rs`.
    pub fn ground_y(
        &mut self,
        world: &WorldState,
        anchor: &WorldAnchor,
        view_layer: u8,
        bevy_x: f32,
        bevy_z: f32,
        _cell: f32, // kept for API compat; ignored internally
    ) -> f32 {
        let collision_layer = (view_layer + 1).min(MAX_LAYER);
        let block_size_i64 = cell_size_at_layer(collision_layer) as i64;

        // Flush cache if collision layer changed (zoom change).
        if self.cached_collision_layer != collision_layer {
            self.cache.clear();
            self.cached_collision_layer = collision_layer;
        }

        // Guard against non-finite inputs.
        if !bevy_x.is_finite() || !bevy_z.is_finite() {
            return f32::NAN;
        }

        // Convert bevy-space XZ to absolute leaf coordinates for the key.
        let leaf_x = anchor.leaf_coord[0] + bevy_x.floor() as i64;
        let leaf_z = anchor.leaf_coord[2] + bevy_z.floor() as i64;

        // Quantize to collision-layer cell-sized buckets in leaf space.
        let key = (
            leaf_x.div_euclid(block_size_i64),
            leaf_z.div_euclid(block_size_i64),
        );

        if let Some(&cached_leaf_y) = self.cache.get(&key) {
            // Convert absolute leaf Y to bevy-space Y.
            return (cached_leaf_y - anchor.leaf_coord[1]) as f32;
        }

        if self.queries_this_frame >= MAX_QUERIES_PER_FRAME {
            return f32::NAN; // signal "no data" instead of a plausible 0.0
        }

        let leaf_y = raycast_ground_integer(
            world,
            collision_layer,
            block_size_i64,
            leaf_x,
            leaf_z,
            anchor.leaf_coord[1],
        );
        self.cache.insert(key, leaf_y);
        self.queries_this_frame += 1;

        (leaf_y - anchor.leaf_coord[1]) as f32
    }

    /// Invalidate cache entries near a bevy-space edit position.
    pub fn invalidate_near(
        &mut self,
        bevy_x: f32,
        bevy_z: f32,
        anchor: &WorldAnchor,
        _cell: f32,
        radius: i32,
    ) {
        let collision_layer = self.cached_collision_layer;
        let block_size_i64 = cell_size_at_layer(collision_layer).max(1.0) as i64;
        let leaf_x = anchor.leaf_coord[0] + bevy_x.floor() as i64;
        let leaf_z = anchor.leaf_coord[2] + bevy_z.floor() as i64;
        let cx = leaf_x.div_euclid(block_size_i64);
        let cz = leaf_z.div_euclid(block_size_i64);
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

/// Raycast downward in integer leaf space to find ground height.
///
/// Works entirely in `i64` leaf coordinates and
/// `layer_pos_from_leaf_coord_direct`, avoiding any float-to-cell
/// alignment issues. Returns the absolute leaf Y of the top face of
/// the highest solid cell in the column.
fn raycast_ground_integer(
    world: &WorldState,
    collision_layer: u8,
    block_size_i64: i64,
    leaf_x: i64,
    leaf_z: i64,
    anchor_leaf_y: i64,
) -> i64 {
    // Snap X and Z to the centre of their collision-layer cell to
    // avoid boundary ambiguity.
    let probe_x = (leaf_x / block_size_i64) * block_size_i64 + block_size_i64 / 2;
    let probe_z = (leaf_z / block_size_i64) * block_size_i64 + block_size_i64 / 2;

    // Scan 64 cells above and below the anchor Y.
    let center_cell_y = anchor_leaf_y / block_size_i64;
    let max_cells: i64 = 64;

    for dy in (-max_cells..max_cells).rev() {
        let cell_y = center_cell_y + dy;
        let probe_y = cell_y * block_size_i64 + block_size_i64 / 2;
        let coord = [probe_x, probe_y, probe_z];
        if let Some(lp) = layer_pos_from_leaf_coord_direct(coord, collision_layer) {
            if is_layer_pos_solid(world, &lp) {
                // Top face of this solid cell in absolute leaf coords.
                return (cell_y + 1) * block_size_i64;
            }
        }
    }

    // Fallback: return the anchor Y (no ground found).
    anchor_leaf_y
}
