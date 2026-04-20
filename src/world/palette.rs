//! Color palette: maps `u16` block indices to RGBA colors.
//!
//! The palette is the single source of truth for block definitions.
//! Indices 0-9 are the builtin block types; 10..=65_534 are available
//! for imported model colors. `u16::MAX` (65_535) is reserved as a
//! "missing/unmapped" sentinel — never handed out by `register()`.
//!
//! Emptiness is NOT represented in the palette. `Child::Empty` is the
//! sole empty sentinel throughout the tree. Every palette entry is a
//! real color meant to be rendered.

use std::collections::HashMap;

/// Maximum number of palette entries (u16 range, minus `u16::MAX`
/// reserved as the "missing color" sentinel).
pub const MAX_ENTRIES: usize = 65_535;

/// Invalid / "no color assigned" sentinel. Never returned by
/// `register()`; callers that want to represent emptiness should
/// use `Child::Empty`, not this index.
pub const INVALID_COLOR: u16 = u16::MAX;

/// A single palette entry: name + RGBA color.
#[derive(Clone, Debug)]
pub struct PaletteEntry {
    pub name: String,
    pub color: [f32; 4],
}

/// The builtin block types. Builtins occupy indices 0..BUILTIN_COUNT;
/// imported model colors register into slots `BUILTIN_COUNT..`.
pub const BUILTINS: &[(u16, &str, [f32; 4])] = &[
    (0, "Stone", [0.50, 0.50, 0.50, 1.0]),
    (1, "Dirt", [0.45, 0.30, 0.15, 1.0]),
    (2, "Grass", [0.30, 0.60, 0.20, 1.0]),
    (3, "Wood", [0.55, 0.35, 0.15, 1.0]),
    (4, "Leaf", [0.20, 0.50, 0.10, 1.0]),
    (5, "Sand", [0.85, 0.80, 0.55, 1.0]),
    (6, "Water", [0.20, 0.40, 0.80, 1.0]),
    (7, "Brick", [0.70, 0.30, 0.20, 1.0]),
    (8, "Metal", [0.75, 0.75, 0.80, 1.0]),
    (9, "Glass", [0.85, 0.90, 1.00, 1.0]),
];

/// Named constants for the builtin indices — use these instead of
/// the old `BlockType` enum.
pub mod block {
    pub const STONE: u16 = 0;
    pub const DIRT: u16 = 1;
    pub const GRASS: u16 = 2;
    pub const WOOD: u16 = 3;
    pub const LEAF: u16 = 4;
    pub const SAND: u16 = 5;
    pub const WATER: u16 = 6;
    pub const BRICK: u16 = 7;
    pub const METAL: u16 = 8;
    pub const GLASS: u16 = 9;
    pub const BUILTIN_COUNT: u16 = 10;
}

pub struct ColorRegistry {
    entries: Vec<PaletteEntry>,
    /// Dedup: RGBA quantized to u8 -> palette index.
    seen: HashMap<(u8, u8, u8, u8), u16>,
}

impl ColorRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            entries: Vec::with_capacity(BUILTINS.len()),
            seen: HashMap::new(),
        };
        for &(expected_idx, name, color) in BUILTINS {
            let r = (color[0] * 255.0) as u8;
            let g = (color[1] * 255.0) as u8;
            let b = (color[2] * 255.0) as u8;
            let a = (color[3] * 255.0) as u8;
            let idx = reg.entries.len() as u16;
            debug_assert_eq!(
                idx, expected_idx,
                "BUILTINS ordering must match registration order"
            );
            reg.entries.push(PaletteEntry {
                name: name.to_string(),
                color,
            });
            reg.seen.insert((r, g, b, a), idx);
        }
        reg
    }

    /// Register an RGBA color (0-255 per channel). Returns the palette
    /// index. Deduplicates: identical colors return the same index.
    /// Returns `None` if the palette is full (`MAX_ENTRIES` entries).
    /// Transparent colors (a == 0) should not be registered — the
    /// caller should map them to `Child::Empty`.
    pub fn register(&mut self, r: u8, g: u8, b: u8, a: u8) -> Option<u16> {
        let key = (r, g, b, a);
        if let Some(&idx) = self.seen.get(&key) {
            return Some(idx);
        }
        if self.entries.len() >= MAX_ENTRIES {
            return None;
        }
        let idx = self.entries.len() as u16;
        self.entries.push(PaletteEntry {
            name: format!("color_{}", idx),
            color: [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ],
        });
        self.seen.insert(key, idx);
        Some(idx)
    }

    pub fn get(&self, index: u16) -> Option<&PaletteEntry> {
        self.entries.get(index as usize)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn name(&self, index: u16) -> &str {
        self.entries
            .get(index as usize)
            .map(|e| e.name.as_str())
            .unwrap_or("Unknown")
    }

    pub fn color(&self, index: u16) -> [f32; 4] {
        self.entries
            .get(index as usize)
            .map(|e| e.color)
            .unwrap_or([0.3, 0.3, 0.3, 1.0])
    }

    /// Return the full palette as a flat `Vec<[f32; 4]>` suitable for
    /// uploading as a storage buffer (one `vec4<f32>` per index).
    pub fn to_gpu_palette(&self) -> Vec<[f32; 4]> {
        self.entries.iter().map(|e| e.color).collect()
    }
}

impl Default for ColorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_registered() {
        let reg = ColorRegistry::new();
        assert_eq!(reg.len(), block::BUILTIN_COUNT as usize);
        assert_eq!(reg.name(block::STONE), "Stone");
        assert_eq!(reg.name(block::GLASS), "Glass");
    }

    #[test]
    fn register_dedup() {
        let mut reg = ColorRegistry::new();
        let a = reg.register(255, 0, 0, 255).unwrap();
        let b = reg.register(255, 0, 0, 255).unwrap();
        assert_eq!(a, b);
        assert_eq!(reg.len(), (block::BUILTIN_COUNT as usize) + 1);
    }

    #[test]
    fn register_distinct() {
        let mut reg = ColorRegistry::new();
        let a = reg.register(255, 0, 0, 255).unwrap();
        let b = reg.register(0, 255, 0, 255).unwrap();
        assert_ne!(a, b);
        assert_eq!(reg.len(), (block::BUILTIN_COUNT as usize) + 2);
    }

    #[test]
    fn gpu_palette_roundtrip() {
        let reg = ColorRegistry::new();
        let gp = reg.to_gpu_palette();
        assert_eq!(gp[block::STONE as usize], reg.color(block::STONE));
        assert_eq!(gp[block::GLASS as usize], reg.color(block::GLASS));
    }

    #[test]
    fn palette_overflow_returns_none() {
        let mut reg = ColorRegistry::new();
        // Burn through the entire palette with distinct RGBs.
        for i in 0..(MAX_ENTRIES - block::BUILTIN_COUNT as usize) {
            let r = (i & 0xff) as u8;
            let g = ((i >> 8) & 0xff) as u8;
            let b = ((i >> 16) & 0xff) as u8;
            // alpha varies too so we don't collide with prior iters
            let a = ((i >> 20) & 0xff) as u8;
            let res = reg.register(r, g, b, a.max(1));
            assert!(res.is_some(), "iter {} returned None early", i);
        }
        assert_eq!(reg.len(), MAX_ENTRIES);
        // One more should fail.
        let res = reg.register(123, 45, 67, 89);
        assert!(res.is_none(), "65_536th register should return None");
    }
}
