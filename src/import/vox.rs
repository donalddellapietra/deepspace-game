//! Parse MagicaVoxel `.vox` files into [`VoxelModel`].
//!
//! Uses the [`dot_vox`] crate for the low-level parse, then registers
//! palette colours as exact `Palette` entries via [`color_map`].

use std::path::Path;

use bevy::prelude::*;

use crate::block::{BslMaterial, Palette};
use crate::world::tree::EMPTY_VOXEL;

use super::color_map;
use super::VoxelModel;

/// Load a `.vox` file from disk and return the first model.
pub fn load_first_model(
    path: &Path,
    palette: &mut Palette,
    mat_assets: &mut Assets<BslMaterial>,
) -> Result<VoxelModel, String> {
    let data = dot_vox::load(
        path.to_str()
            .ok_or_else(|| "path contains non-UTF-8 characters".to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let model = data.models.first().ok_or("no models in .vox file")?;
    Ok(convert_model(model, &data.palette, palette, mat_assets))
}

/// Load a `.vox` file from an in-memory byte slice.
pub fn load_first_model_bytes(
    bytes: &[u8],
    palette: &mut Palette,
    mat_assets: &mut Assets<BslMaterial>,
) -> Result<VoxelModel, String> {
    let data = dot_vox::load_bytes(bytes).map_err(|e| e.to_string())?;
    let model = data.models.first().ok_or("no models in .vox file")?;
    Ok(convert_model(model, &data.palette, palette, mat_assets))
}

/// Load every model from a `.vox` file.
pub fn load_all_models(
    path: &Path,
    palette: &mut Palette,
    mat_assets: &mut Assets<BslMaterial>,
) -> Result<Vec<VoxelModel>, String> {
    let data = dot_vox::load(
        path.to_str()
            .ok_or_else(|| "path contains non-UTF-8 characters".to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(data
        .models
        .iter()
        .map(|m| convert_model(m, &data.palette, palette, mat_assets))
        .collect())
}

fn convert_model(
    model: &dot_vox::Model,
    vox_palette: &[dot_vox::Color],
    palette: &mut Palette,
    mat_assets: &mut Assets<BslMaterial>,
) -> VoxelModel {
    let sx = model.size.x as usize;
    let sy = model.size.y as usize;
    let sz = model.size.z as usize;

    let mut pal: [(u8, u8, u8, u8); 256] = [(0, 0, 0, 0); 256];
    for (i, c) in vox_palette.iter().enumerate().take(256) {
        pal[i] = (c.r, c.g, c.b, c.a);
    }
    let lut = color_map::build_palette_lut(&pal, palette, mat_assets);

    let mut data = vec![EMPTY_VOXEL; sx * sy * sz];

    // dot_vox coordinate system: (x, z, y) vs our (x, y, z).
    // MagicaVoxel Z is vertical; we use Y-up.
    let out_x = sx;
    let out_y = sz;
    let out_z = sy;

    for v in &model.voxels {
        let mx = v.x as usize;
        let my = v.z as usize;
        let mz = v.y as usize;
        if mx < out_x && my < out_y && mz < out_z {
            let idx = (mz * out_y + my) * out_x + mx;
            if let Some(&voxel) = lut.get(v.i as usize) {
                data[idx] = voxel;
            }
        }
    }

    VoxelModel {
        size_x: out_x,
        size_y: out_y,
        size_z: out_z,
        data,
    }
}
