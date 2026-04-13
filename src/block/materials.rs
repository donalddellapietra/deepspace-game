use bevy::prelude::*;
use super::{PaletteMaterial, Palette};

/// Initialize the Palette resource at startup.
pub fn init_palette(
    mut commands: Commands,
    mut materials: ResMut<Assets<PaletteMaterial>>,
) {
    let palette = Palette::new(&mut materials);
    commands.insert_resource(palette);
}
