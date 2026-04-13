//! Heightmap generation from the voxel tree.
//!
//! Pre-computes a 2D R32Float texture of ground heights so the GPU
//! compute shader can sample it instead of walking the tree per NPC.
//! Regenerated when the view layer changes or terrain is edited.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::state::WorldState;
use super::view::{
    cell_size_at_layer, is_layer_pos_solid, layer_pos_from_bevy, target_layer_for, WorldAnchor,
};
use super::render::RADIUS_VIEW_CELLS;

/// Resolution of the heightmap in texels per axis.
/// 256x256 covers RADIUS_VIEW_CELLS in each direction at one texel per cell.
pub const HEIGHTMAP_RES: u32 = 256;

/// Resource holding the heightmap image and its world-space bounds.
#[derive(Resource)]
pub struct Heightmap {
    pub image: Handle<Image>,
    /// World-space min corner (bevy coords) of the heightmap.
    pub world_min: Vec2,
    /// World-space size of the heightmap.
    pub world_size: Vec2,
}

/// Generate or regenerate the heightmap from the voxel tree.
pub fn generate_heightmap(
    world: &WorldState,
    view_layer: u8,
    anchor: &WorldAnchor,
    images: &mut Assets<Image>,
) -> Heightmap {
    let cell = cell_size_at_layer(view_layer);
    let radius = RADIUS_VIEW_CELLS * cell;
    let target = target_layer_for(view_layer);

    let world_min = Vec2::new(-radius, -radius);
    let world_size = Vec2::new(radius * 2.0, radius * 2.0);

    let mut data = vec![0u8; (HEIGHTMAP_RES * HEIGHTMAP_RES * 4) as usize];

    for tz in 0..HEIGHTMAP_RES {
        for tx in 0..HEIGHTMAP_RES {
            // Map texel to world XZ (anchor-relative bevy coords).
            let u = (tx as f32 + 0.5) / HEIGHTMAP_RES as f32;
            let v = (tz as f32 + 0.5) / HEIGHTMAP_RES as f32;
            let world_x = world_min.x + u * world_size.x;
            let world_z = world_min.y + v * world_size.y;

            // Raycast down to find ground height.
            let ground_y = raycast_ground_y(world, target, anchor, world_x, world_z, cell);

            let offset = ((tz * HEIGHTMAP_RES + tx) * 4) as usize;
            data[offset..offset + 4].copy_from_slice(&ground_y.to_le_bytes());
        }
    }

    let image = Image::new(
        Extent3d {
            width: HEIGHTMAP_RES,
            height: HEIGHTMAP_RES,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    Heightmap {
        image: images.add(image),
        world_min,
        world_size,
    }
}

/// Raycast downward at a given XZ to find the Y of the top solid cell.
fn raycast_ground_y(
    world: &WorldState,
    target_layer: u8,
    anchor: &WorldAnchor,
    world_x: f32,
    world_z: f32,
    cell: f32,
) -> f32 {
    // Probe from high to low to find the first solid cell.
    let max_y_cells: i32 = 64;
    for y_cell in (-max_y_cells..max_y_cells).rev() {
        let probe = Vec3::new(world_x, y_cell as f32 * cell + cell * 0.5, world_z);
        if let Some(lp) = layer_pos_from_bevy(probe, target_layer, anchor) {
            if is_layer_pos_solid(world, &lp) {
                return (y_cell + 1) as f32 * cell;
            }
        }
    }
    0.0
}
