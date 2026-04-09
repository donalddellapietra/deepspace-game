pub mod grid;
pub mod tools;

use bevy::prelude::*;

use crate::block::BlockType;
use crate::layer::GameLayer;

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorState>()
            .add_systems(OnEnter(GameLayer::Editing), grid::spawn_edit_grid)
            .add_systems(OnExit(GameLayer::Editing), grid::exit_edit_mode)
            .add_systems(
                Update,
                (tools::place_block, tools::remove_block, tools::cycle_block_type, tools::save_as_template)
                    .run_if(in_state(GameLayer::Editing)),
            )
            .add_systems(
                Update,
                (tools::enter_edit_mode, tools::cycle_model_slot, tools::place_cell, tools::remove_cell)
                    .run_if(in_state(GameLayer::World)),
            )
            .add_systems(
                Update,
                tools::exit_edit_shortcut.run_if(in_state(GameLayer::Editing)),
            );
    }
}

#[derive(Resource)]
pub struct EditorState {
    pub selected_block: BlockType,
}

impl Default for EditorState {
    fn default() -> Self {
        Self { selected_block: BlockType::Stone }
    }
}

#[derive(Resource)]
pub struct SharedCubeMesh(pub Handle<Mesh>);
