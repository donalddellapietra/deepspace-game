//! Parse MagicaVoxel `.vox` files into [`VoxelModel`].

use std::path::Path;

use crate::world::palette::ColorRegistry;
use super::VoxelModel;

/// Load the first model from a `.vox` file on disk.
pub fn load_first_model(
    path: &Path,
    registry: &mut ColorRegistry,
) -> Result<VoxelModel, String> {
    let data = dot_vox::load(
        path.to_str().ok_or("path contains non-UTF-8 characters")?,
    ).map_err(|e| e.to_string())?;
    let model = data.models.first().ok_or("no models in .vox file")?;
    Ok(convert_model(model, &data.palette, registry))
}

/// Load the first model from in-memory bytes.
pub fn load_first_model_bytes(
    bytes: &[u8],
    registry: &mut ColorRegistry,
) -> Result<VoxelModel, String> {
    let data = dot_vox::load_bytes(bytes).map_err(|e| e.to_string())?;
    let model = data.models.first().ok_or("no models in .vox file")?;
    Ok(convert_model(model, &data.palette, registry))
}

fn convert_model(
    model: &dot_vox::Model,
    vox_palette: &[dot_vox::Color],
    registry: &mut ColorRegistry,
) -> VoxelModel {
    // Build a LUT mapping .vox palette index → our palette index.
    // Index 0 in .vox is reserved/unused; real colors are 1-255.
    let mut lut = [0u8; 256];
    for (i, c) in vox_palette.iter().enumerate().take(256) {
        if c.a == 0 {
            lut[i] = 0; // transparent → empty
        } else if let Some(idx) = registry.register(c.r, c.g, c.b, c.a) {
            lut[i] = idx;
        }
        // If registry is full, lut[i] stays 0 (empty). Acceptable fallback.
    }

    // MagicaVoxel coordinate system: (x, z, y) → our (x, y, z).
    // MagicaVoxel Z is vertical; we use Y-up.
    let out_x = model.size.x as usize;
    let out_y = model.size.z as usize; // their Z → our Y
    let out_z = model.size.y as usize; // their Y → our Z

    let mut data = vec![0u8; out_x * out_y * out_z];

    for v in &model.voxels {
        let x = v.x as usize;
        let y = v.z as usize; // their Z → our Y
        let z = v.y as usize; // their Y → our Z
        if x < out_x && y < out_y && z < out_z {
            let idx = (z * out_y + y) * out_x + x;
            data[idx] = lut[v.i as usize];
        }
    }

    VoxelModel { size_x: out_x, size_y: out_y, size_z: out_z, data }
}
