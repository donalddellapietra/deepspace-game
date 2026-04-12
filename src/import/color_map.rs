//! Map RGBA palette colours to [`BlockType`].
//!
//! Each `.vox` palette entry is matched to the nearest `BlockType` by
//! Euclidean distance in sRGB space. Fully transparent entries
//! (`a == 0`) map to [`EMPTY_VOXEL`].

use crate::block::BlockType;
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
/// when the alpha channel is zero.
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

/// Pre-compute a full 256-entry palette lookup table so each voxel
/// only needs an array index during model construction.
pub fn build_palette_lut(palette: &[(u8, u8, u8, u8); 256]) -> [Voxel; 256] {
    let mut lut = [EMPTY_VOXEL; 256];
    for (i, &(r, g, b, a)) in palette.iter().enumerate() {
        lut[i] = rgba_to_voxel(r, g, b, a);
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
