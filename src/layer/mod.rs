use bevy::prelude::*;

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GameLayer {
    #[default]
    World,
    Editing,
}

/// Which cell is being edited. Only present during GameLayer::Editing.
#[derive(Resource)]
pub struct EditingContext {
    pub cell_coord: IVec3,
    pub return_position: Vec3,
}

pub struct LayerPlugin;

impl Plugin for LayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<GameLayer>();
    }
}
