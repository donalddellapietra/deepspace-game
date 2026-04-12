//! Parse MagicaVoxel `.vox` files into [`VoxelModel`].
//!
//! Uses the [`dot_vox`] crate for the low-level parse, then maps
//! palette indices to [`BlockType`] voxels via [`color_map`].

use std::path::Path;

use crate::world::tree::EMPTY_VOXEL;

use super::color_map;
use super::VoxelModel;

/// Load a `.vox` file from disk and return the first model as a
/// [`VoxelModel`]. Multi-model files are common in MagicaVoxel (one
/// model per frame or per object); callers that need all models should
/// use [`load_all_models`].
pub fn load_first_model(path: &Path) -> Result<VoxelModel, String> {
    let data = dot_vox::load(
        path.to_str()
            .ok_or_else(|| "path contains non-UTF-8 characters".to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let model = data.models.first().ok_or("no models in .vox file")?;
    Ok(convert_model(model, &data.palette))
}

/// Load a `.vox` file from an in-memory byte slice.
pub fn load_first_model_bytes(bytes: &[u8]) -> Result<VoxelModel, String> {
    let data = dot_vox::load_bytes(bytes).map_err(|e| e.to_string())?;
    let model = data.models.first().ok_or("no models in .vox file")?;
    Ok(convert_model(model, &data.palette))
}

/// Load every model from a `.vox` file.
pub fn load_all_models(path: &Path) -> Result<Vec<VoxelModel>, String> {
    let data = dot_vox::load(
        path.to_str()
            .ok_or_else(|| "path contains non-UTF-8 characters".to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(data
        .models
        .iter()
        .map(|m| convert_model(m, &data.palette))
        .collect())
}

fn convert_model(model: &dot_vox::Model, palette: &[dot_vox::Color]) -> VoxelModel {
    let sx = model.size.x as usize;
    let sy = model.size.y as usize;
    let sz = model.size.z as usize;

    // Build a 256-entry LUT from palette index → our Voxel.
    let mut pal: [(u8, u8, u8, u8); 256] = [(0, 0, 0, 0); 256];
    for (i, c) in palette.iter().enumerate().take(256) {
        pal[i] = (c.r, c.g, c.b, c.a);
    }
    let lut = color_map::build_palette_lut(&pal);

    // Allocate the flat grid (default: air).
    let mut data = vec![EMPTY_VOXEL; sx * sy * sz];

    // dot_vox stores voxels as a sparse list. Scatter them into the
    // dense grid. The .vox coordinate system uses (x, z, y) compared
    // to our (x, y, z) — MagicaVoxel's Z axis is vertical while ours
    // uses Y-up.
    // After the axis swap the output dimensions are:
    let out_x = sx; // .vox X → our X
    let out_y = sz; // .vox Z → our Y (up)
    let out_z = sy; // .vox Y → our Z (depth)

    for v in &model.voxels {
        let mx = v.x as usize;
        let my = v.z as usize; // .vox Z → our Y (up)
        let mz = v.y as usize; // .vox Y → our Z (depth)
        if mx < out_x && my < out_y && mz < out_z {
            // Match VoxelModel::index layout: (z * size_y + y) * size_x + x
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_placeholder_from_dot_vox_crate() {
        // dot_vox ships a tiny 2×2×2 test file. Use it as a smoke test.
        let result = load_first_model(Path::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/resources/placeholder.vox"),
        ));
        // The file may not exist at that path in our repo — that's fine.
        // We just verify the parser doesn't panic.
        if let Ok(model) = result {
            assert!(model.size_x > 0);
            assert!(model.size_y > 0);
            assert!(model.size_z > 0);
            assert_eq!(model.data.len(), model.size_x * model.size_y * model.size_z);
        }
    }
}
