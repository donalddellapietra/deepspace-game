use bevy::prelude::*;

use crate::model::ModelId;

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GameLayer {
    #[default]
    World,   // Layer 1: place models in the world grid
    Editing, // Layer 0: edit blocks inside a single cell
}

/// Context for the cell currently being edited. Inserted before transitioning to Editing.
#[derive(Resource)]
pub struct EditingContext {
    pub cell_coord: IVec3,
    pub model_id: ModelId,
    /// Where the player was in world space before zooming in, so we can return them.
    pub return_position: Vec3,
}

pub struct LayerPlugin;

impl Plugin for LayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<GameLayer>();
    }
}
