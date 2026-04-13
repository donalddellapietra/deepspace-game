pub mod bsl_material;
pub mod materials;

pub use bsl_material::BslMaterial;

use bevy::image::{ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bsl_material::{BslExtension, BslParams};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum BlockType {
    Stone = 0, Dirt = 1, Grass = 2, Wood = 3, Leaf = 4,
    Sand = 5, Water = 6, Brick = 7, Metal = 8, Glass = 9,
}

impl BlockType {
    pub const ALL: [Self; 10] = [
        Self::Stone, Self::Dirt, Self::Grass, Self::Wood, Self::Leaf,
        Self::Sand, Self::Water, Self::Brick, Self::Metal, Self::Glass,
    ];

    pub fn color(self) -> Color {
        match self {
            Self::Stone => Color::srgb(0.5, 0.5, 0.5),
            Self::Dirt  => Color::srgb(0.45, 0.3, 0.15),
            Self::Grass => Color::srgb(0.3, 0.6, 0.2),
            Self::Wood  => Color::srgb(0.55, 0.35, 0.15),
            Self::Leaf  => Color::srgb(0.2, 0.5, 0.1),
            Self::Sand  => Color::srgb(0.85, 0.8, 0.55),
            Self::Water => Color::srgba(0.2, 0.4, 0.8, 0.7),
            Self::Brick => Color::srgb(0.7, 0.3, 0.2),
            Self::Metal => Color::srgb(0.75, 0.75, 0.8),
            Self::Glass => Color::srgba(0.85, 0.9, 1.0, 0.3),
        }
    }

    pub fn roughness(self) -> f32 {
        match self { Self::Metal => 0.2, Self::Glass | Self::Water => 0.1, _ => 0.9 }
    }

    pub fn metallic(self) -> f32 {
        match self { Self::Metal => 0.9, _ => 0.0 }
    }

    pub fn alpha_mode(self) -> AlphaMode {
        match self { Self::Water | Self::Glass => AlphaMode::Blend, _ => AlphaMode::Opaque }
    }

    pub fn from_index(i: u8) -> Option<Self> {
        Self::ALL.get(i as usize).copied()
    }

    /// Return the 1-based voxel index for this built-in block type.
    pub fn voxel(&self) -> u8 {
        (*self as u8) + 1
    }

    pub fn texture_name(self) -> &'static str {
        match self {
            Self::Stone => "stone",
            Self::Dirt => "dirt",
            Self::Grass => "grass",
            Self::Wood => "wood",
            Self::Leaf => "leaf",
            Self::Sand => "sand",
            Self::Water => "water",
            Self::Brick => "brick",
            Self::Metal => "metal",
            Self::Glass => "glass",
        }
    }
}

// ---------------------------------------------------------------- Palette

/// A single palette entry.
#[derive(Clone, Debug)]
pub struct PaletteEntry {
    pub name: String,
    pub color: Color,
    pub roughness: f32,
    pub metallic: f32,
    pub alpha_mode: AlphaMode,
    pub texture: Option<Handle<Image>>,
}

/// The concrete material type the palette creates handles for.
/// Every caller that needs `&mut Assets<PaletteMaterial>` imports
/// this alias instead of reaching into `bsl_material` directly.
pub type PaletteMaterial = BslMaterial;

/// Runtime palette. Index 0 is reserved (EMPTY_VOXEL). Indices 1..=len are
/// valid block types. The first 10 are the built-in gameplay blocks.
#[derive(Resource)]
pub struct Palette {
    entries: Vec<PaletteEntry>,
    materials: Vec<Handle<PaletteMaterial>>,
}

impl Palette {
    /// Create a new palette pre-populated with the 10 built-in blocks.
    pub fn new(
        mat_assets: &mut Assets<PaletteMaterial>,
        asset_server: &AssetServer,
    ) -> Self {
        let mut palette = Self {
            entries: Vec::new(),
            materials: Vec::new(),
        };
        for bt in BlockType::ALL {
            let path = format!("textures/blocks/{}.png", bt.texture_name());
            let texture: Handle<Image> = asset_server
                .load_with_settings(path, |s: &mut ImageLoaderSettings| {
                    s.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                        address_mode_u: ImageAddressMode::Repeat,
                        address_mode_v: ImageAddressMode::Repeat,
                        address_mode_w: ImageAddressMode::Repeat,
                        mag_filter: ImageFilterMode::Nearest,
                        min_filter: ImageFilterMode::Nearest,
                        mipmap_filter: ImageFilterMode::Nearest,
                        ..default()
                    });
                });
            palette.register(
                PaletteEntry {
                    name: format!("{:?}", bt),
                    color: bt.color(),
                    roughness: bt.roughness(),
                    metallic: bt.metallic(),
                    alpha_mode: bt.alpha_mode(),
                    texture: Some(texture),
                },
                mat_assets,
            );
        }
        palette
    }

    /// Look up by voxel value (1-indexed). Returns `None` for 0 (empty)
    /// or out-of-range indices.
    pub fn get(&self, voxel: u8) -> Option<&PaletteEntry> {
        if voxel == 0 {
            return None;
        }
        self.entries.get((voxel - 1) as usize)
    }

    /// Get the material handle for a voxel value.
    pub fn material(&self, voxel: u8) -> Option<Handle<PaletteMaterial>> {
        if voxel == 0 {
            return None;
        }
        self.materials.get((voxel - 1) as usize).cloned()
    }

    /// Find an existing entry with the same color (within tolerance).
    /// Returns the 1-based voxel index if found.
    pub fn find_by_color(&self, color: Color) -> Option<u8> {
        let target = color.to_srgba();
        for (i, entry) in self.entries.iter().enumerate() {
            let c = entry.color.to_srgba();
            let dr = (c.red - target.red).abs();
            let dg = (c.green - target.green).abs();
            let db = (c.blue - target.blue).abs();
            let da = (c.alpha - target.alpha).abs();
            if dr < 0.004 && dg < 0.004 && db < 0.004 && da < 0.004 {
                return Some((i + 1) as u8);
            }
        }
        None
    }

    /// Add a new entry, create its material, return the voxel index.
    /// Deduplicates by color — if an entry with the same color already
    /// exists, returns its index without creating a new material.
    /// Panics if the palette is full (255 entries) and the color is new.
    pub fn register(
        &mut self,
        entry: PaletteEntry,
        mat_assets: &mut Assets<PaletteMaterial>,
    ) -> u8 {
        if let Some(existing) = self.find_by_color(entry.color) {
            return existing;
        }
        assert!(
            self.entries.len() < 255,
            "Palette full: cannot register more than 255 entries"
        );
        let subsurface = match entry.name.as_str() {
            "Leaf" | "Water" | "Glass" => 0.5,
            _ => 0.0,
        };
        let base_color = if entry.texture.is_some() {
            Color::WHITE
        } else {
            entry.color
        };
        let handle = mat_assets.add(BslMaterial {
            base: StandardMaterial {
                base_color,
                base_color_texture: entry.texture.clone(),
                perceptual_roughness: entry.roughness,
                metallic: entry.metallic,
                alpha_mode: entry.alpha_mode,
                ..default()
            },
            extension: BslExtension {
                params: BslParams {
                    subsurface_strength: subsurface,
                    ..default()
                },
            },
        });
        self.entries.push(entry);
        self.materials.push(handle);
        self.entries.len() as u8 // 1-based voxel index
    }

    /// Number of palette entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate with 1-based voxel indices.
    pub fn iter(&self) -> impl Iterator<Item = (u8, &PaletteEntry)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| ((i + 1) as u8, e))
    }
}

// ---------------------------------------------------------------- Plugin

pub struct BlockPlugin;
impl Plugin for BlockPlugin {
    fn build(&self, app: &mut App) {
        bsl_material::load_bsl_shader(app);
        app.add_systems(Startup, materials::init_palette);
    }
}
