use bevy::prelude::*;

/// Each entry records where the player was before drilling in.
#[derive(Clone)]
pub struct NavEntry {
    pub cell_coord: IVec3,
    pub return_position: Vec3,
}

/// Tracks drill-in/out navigation. Empty stack = at top layer.
#[derive(Resource, Default)]
pub struct ActiveLayer {
    pub nav_stack: Vec<NavEntry>,
}

impl ActiveLayer {
    pub fn is_top_layer(&self) -> bool { self.nav_stack.is_empty() }
    pub fn current_cell(&self) -> Option<IVec3> {
        self.nav_stack.last().map(|e| e.cell_coord)
    }
}

pub struct LayerPlugin;
impl Plugin for LayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveLayer>();
    }
}
