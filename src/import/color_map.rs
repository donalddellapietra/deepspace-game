//! Map RGBA palette colours to voxel indices.
//!
//! Two modes:
//! 1. **Legacy nearest-match** — each `.vox` palette entry is matched to
//!    the nearest built-in `BlockType` by Euclidean distance in sRGB space.
//! 2. **Palette registration** — each `.vox` palette entry is registered as
//!    a new `Palette` entry with its exact color, returning a fresh voxel
//!    index. Transparent entries (`a == 0`) map to [`EMPTY_VOXEL`].

use bevy::prelude::*;

use crate::block::{BlockType, PaletteEntry, Palette};
use crate::world::tree::{voxel_from_block, Voxel, EMPTY_VOXEL};

/// Reference sRGB colours for each `BlockType`, extracted from
/// [`BlockType::color`] and stored as `(r, g, b)` in `0..=255`.
const BLOCK_COLORS: [(u8, u8, u8); 10] = [
    (128, 128, 128), // Stone
    (115, 77, 38),   // Dirt
    (77, 153, 51),   // Grass
    (140, 89, 38),   // Wood
    (51, 128, 26),   // Leaf
    (217, 204, 140), // Sand
    (51, 102, 204),  // Water
    (179, 77, 51),   // Brick
    (191, 191, 204), // Metal
    (217, 230, 255), // Glass
];

/// Map an RGBA colour to the closest `BlockType`, or `EMPTY_VOXEL`
/// when the alpha channel is zero. (Legacy nearest-match mode.)
pub fn rgba_to_voxel(r: u8, g: u8, b: u8, a: u8) -> Voxel {
    if a == 0 {
        return EMPTY_VOXEL;
    }
    let mut best_idx = 0usize;
    let mut best_dist = u32::MAX;
    for (i, &(br, bg, bb)) in BLOCK_COLORS.iter().enumerate() {
        let dr = r as i32 - br as i32;
        let dg = g as i32 - bg as i32;
        let db = b as i32 - bb as i32;
        let dist = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    voxel_from_block(Some(BlockType::ALL[best_idx]))
}

/// Pre-compute a full 256-entry palette lookup table using legacy
/// nearest-match to the 10 built-in block types.
pub fn build_palette_lut(palette: &[(u8, u8, u8, u8); 256]) -> [Voxel; 256] {
    let mut lut = [EMPTY_VOXEL; 256];
    for (i, &(r, g, b, a)) in palette.iter().enumerate() {
        lut[i] = rgba_to_voxel(r, g, b, a);
    }
    lut
}

/// Register .vox palette colors as new Palette entries (exact colors).
/// Returns a `[u8; 256]` mapping .vox palette indices to voxel indices.
/// Transparent entries (`a == 0`) map to `EMPTY_VOXEL`.
pub fn build_palette_lut_registered(
    vox_palette: &[(u8, u8, u8, u8); 256],
    palette: &mut Palette,
    mat_assets: &mut Assets<StandardMaterial>,
) -> [Voxel; 256] {
    let mut lut = [EMPTY_VOXEL; 256];
    // Cache to avoid registering duplicate colors multiple times.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_maps_to_empty() {
        assert_eq!(rgba_to_voxel(255, 0, 0, 0), EMPTY_VOXEL);
    }

    #[test]
    fn pure_green_maps_to_grass_or_leaf() {
        let v = rgba_to_voxel(0, 200, 0, 255);
        let bt = crate::world::tree::block_from_voxel(v).unwrap();
        assert!(
            bt == BlockType::Grass || bt == BlockType::Leaf,
            "expected Grass or Leaf, got {:?}",
            bt
        );
    }

    #[test]
    fn pure_grey_maps_to_stone() {
        let v = rgba_to_voxel(128, 128, 128, 255);
        assert_eq!(
            crate::world::tree::block_from_voxel(v),
            Some(BlockType::Stone)
        );
    }

    #[test]
    fn palette_lut_length() {
        let palette = [(128u8, 128, 128, 255); 256];
        let lut = build_palette_lut(&palette);
        assert_eq!(lut.len(), 256);
    }
}
