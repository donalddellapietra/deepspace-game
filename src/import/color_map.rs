//! Map RGBA palette colours to voxel indices via palette registration.
//!
//! Each `.vox` palette entry is registered as a new `Palette` entry
//! with its exact color, returning a fresh voxel index. Transparent
//! entries (`a == 0`) map to [`EMPTY_VOXEL`].

use bevy::prelude::*;

use crate::block::{PaletteMaterial, Palette, PaletteEntry};
use crate::world::tree::{Voxel, EMPTY_VOXEL};

/// Register .vox palette colors as new Palette entries (exact colors).
/// Returns a `[u8; 256]` mapping .vox palette indices to voxel indices.
/// Transparent entries (`a == 0`) map to `EMPTY_VOXEL`.
pub fn build_palette_lut(
    vox_palette: &[(u8, u8, u8, u8); 256],
    palette: &mut Palette,
    mat_assets: &mut Assets<PaletteMaterial>,
) -> [Voxel; 256] {
    let mut lut = [EMPTY_VOXEL; 256];
    let mut seen: std::collections::HashMap<(u8, u8, u8, u8), u8> =
        std::collections::HashMap::new();
    for (i, &(r, g, b, a)) in vox_palette.iter().enumerate() {
        if a == 0 {
            lut[i] = EMPTY_VOXEL;
            continue;
        }
        let key = (r, g, b, a);
        if let Some(&cached) = seen.get(&key) {
            lut[i] = cached;
            continue;
        }
        let color = if a == 255 {
            Color::srgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
        } else {
            Color::srgba(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            )
        };
        let alpha_mode = if a < 255 {
            AlphaMode::Blend
        } else {
            AlphaMode::Opaque
        };
        let voxel = palette.register(
            PaletteEntry {
                name: format!("vox_{}", i),
                color,
                roughness: 0.9,
                metallic: 0.0,
                alpha_mode,
            },
            mat_assets,
        );
        seen.insert(key, voxel);
        lut[i] = voxel;
    }
    lut
}
