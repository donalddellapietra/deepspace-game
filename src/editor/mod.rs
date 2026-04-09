pub mod tools;

use bevy::prelude::*;

use crate::block::BlockType;

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorState>()
            .add_systems(Update, (
                tools::drill_down,
                tools::drill_up,
                tools::place_block,
                tools::remove_block,
                tools::cycle_block_type,
                tools::save_as_template,
            ));
    }
}

#[derive(Resource)]
pub struct EditorState {
    pub selected_block: BlockType,
}

impl Default for EditorState {
    fn default() -> Self { Self { selected_block: BlockType::Stone } }
}
