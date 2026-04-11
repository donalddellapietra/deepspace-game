pub mod materials;

use bevy::prelude::*;

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
}

pub struct BlockPlugin;
impl Plugin for BlockPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, materials::init_block_materials);
    }
}
