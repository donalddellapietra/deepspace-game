//! Color palette: maps `u8` block indices to RGBA colors.
//!
//! The palette is the single source of truth for block definitions.
//! Indices 0-9 are the builtin block types; 10-254 are available for
//! imported model colors. Index 255 is reserved (no dominant block).
//!
//! The palette is used both CPU-side (for UI, game logic) and GPU-side
//! (converted to `GpuPalette` for the shader).

use std::collections::HashMap;

use super::gpu::GpuPalette;

/// Maximum number of palette entries (u8 range, minus 255 reserved).
pub const MAX_ENTRIES: usize = 255;

/// A single palette entry: name + RGBA color.
#[derive(Clone, Debug)]
pub struct PaletteEntry {
    pub name: String,
    pub color: [f32; 4],
}

/// The builtin block types. Index 0 is reserved (empty sentinel in
/// `VoxelModel`), so builtins start at 1.
pub const BUILTINS: &[(u8, &str, [f32; 4])] = &[
    (1,  "Stone",  [0.50, 0.50, 0.50, 1.0]),
    (2,  "Dirt",   [0.45, 0.30, 0.15, 1.0]),
    (3,  "Grass",  [0.30, 0.60, 0.20, 1.0]),
    (4,  "Wood",   [0.55, 0.35, 0.15, 1.0]),
    (5,  "Leaf",   [0.20, 0.50, 0.10, 1.0]),
    (6,  "Sand",   [0.85, 0.80, 0.55, 1.0]),
    (7,  "Water",  [0.20, 0.40, 0.80, 1.0]),
    (8,  "Brick",  [0.70, 0.30, 0.20, 1.0]),
    (9,  "Metal",  [0.75, 0.75, 0.80, 1.0]),
    (10, "Glass",  [0.85, 0.90, 1.00, 1.0]),
];

/// Named constants for the builtin indices — use these instead of
/// the old `BlockType` enum. Index 0 is reserved (empty in VoxelModel).
pub mod block {
    pub const STONE: u8 = 1;
    pub const DIRT:  u8 = 2;
    pub const GRASS: u8 = 3;
    pub const WOOD:  u8 = 4;
    pub const LEAF:  u8 = 5;
    pub const SAND:  u8 = 6;
    pub const WATER: u8 = 7;
    pub const BRICK: u8 = 8;
    pub const METAL: u8 = 9;
    pub const GLASS: u8 = 10;
    pub const BUILTIN_COUNT: u8 = 11;
}

pub struct ColorRegistry {
    entries: Vec<PaletteEntry>,
    /// Dedup: RGBA quantized to u8 -> palette index.
    seen: HashMap<(u8, u8, u8, u8), u8>,
}

impl ColorRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            entries: Vec::with_capacity(MAX_ENTRIES),
            seen: HashMap::new(),
        };
        // Index 0 is reserved (empty sentinel in VoxelModel).
        reg.entries.push(PaletteEntry {
            name: "Empty".to_string(),
            color: [0.0, 0.0, 0.0, 0.0],
        });
        for &(_, name, color) in BUILTINS {
            let r = (color[0] * 255.0) as u8;
            let g = (color[1] * 255.0) as u8;
            let b = (color[2] * 255.0) as u8;
            let a = (color[3] * 255.0) as u8;
            let idx = reg.entries.len() as u8;
            reg.entries.push(PaletteEntry { name: name.to_string(), color });
            reg.seen.insert((r, g, b, a), idx);
        }
        reg
    }

    /// Register an RGBA color (0-255 per channel). Returns the palette
    /// index. Deduplicates: identical colors return the same index.
    /// Returns `None` if the palette is full (255 entries).
    /// Transparent colors (a == 0) should not be registered — the
    /// caller should map them to `Child::Empty`.
    pub fn register(&mut self, r: u8, g: u8, b: u8, a: u8) -> Option<u8> {
        let key = (r, g, b, a);
        if let Some(&idx) = self.seen.get(&key) {
            return Some(idx);
        }
        if self.entries.len() >= MAX_ENTRIES {
            return None;
        }
        let idx = self.entries.len() as u8;
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

    pub fn get(&self, index: u8) -> Option<&PaletteEntry> {
        self.entries.get(index as usize)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn name(&self, index: u8) -> &str {
        self.entries.get(index as usize)
            .map(|e| e.name.as_str())
            .unwrap_or("Unknown")
    }

    pub fn color(&self, index: u8) -> [f32; 4] {
        self.entries.get(index as usize)
            .map(|e| e.color)
            .unwrap_or([0.3, 0.3, 0.3, 1.0])
    }

    /// Convert to GPU format for the shader.
    pub fn to_gpu_palette(&self) -> GpuPalette {
        let mut colors = [[0.0f32; 4]; 256];
        for (i, entry) in self.entries.iter().enumerate() {
            colors[i] = entry.color;
        }
        GpuPalette { colors }
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
        // 1 reserved (empty) + 10 builtins = 11
        assert_eq!(reg.len(), 11);
        assert_eq!(reg.name(0), "Empty");
        assert_eq!(reg.name(block::STONE), "Stone");
        assert_eq!(reg.name(block::GLASS), "Glass");
    }

    #[test]
    fn register_dedup() {
        let mut reg = ColorRegistry::new();
        let a = reg.register(255, 0, 0, 255).unwrap();
        let b = reg.register(255, 0, 0, 255).unwrap();
        assert_eq!(a, b);
        assert_eq!(reg.len(), 12);
    }

    #[test]
    fn register_distinct() {
        let mut reg = ColorRegistry::new();
        let a = reg.register(255, 0, 0, 255).unwrap();
        let b = reg.register(0, 255, 0, 255).unwrap();
        assert_ne!(a, b);
        assert_eq!(reg.len(), 13);
    }

    #[test]
    fn gpu_palette_roundtrip() {
        let reg = ColorRegistry::new();
        let gp = reg.to_gpu_palette();
        assert_eq!(gp.colors[block::STONE as usize], reg.color(block::STONE));
        assert_eq!(gp.colors[block::GLASS as usize], reg.color(block::GLASS));
    }
}
