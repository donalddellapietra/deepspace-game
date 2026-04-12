use bevy::prelude::*;
use super::Palette;

/// Initialize the Palette resource at startup.
pub fn init_palette(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let palette = Palette::new(&mut materials);
    commands.insert_resource(palette);
}
