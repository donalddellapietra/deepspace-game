//! Content-addressed voxel tree. See `docs/architecture/` for the
//! design. Sub-modules:
//!
//! * [`tree`]      — `Node`, `NodeLibrary`, hashing, downsampling.
//! * [`position`]  — `Position`, `NodePath`, walking and neighbour walks.
//! * [`state`]     — `WorldState` (Bevy resource wrapping root + library).
//! * [`generator`] — procedural content (v1 returns grassland).
//! * [`edit`]      — leaf and higher-layer edit walks.
//! * [`collision`] — `SolidQuery`, AABB collision, ground check.
//! * [`render`]    — uniform-layer tree-walk renderer + `CameraZoom`.

pub mod collision;
pub mod edit;
pub mod generator;
pub mod position;
pub mod render;
pub mod state;
pub mod tree;

use bevy::prelude::*;

pub use render::CameraZoom;
pub use render::RenderState;
pub use state::WorldState;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldState>()
            .init_resource::<CameraZoom>()
            .init_resource::<RenderState>()
            .add_systems(Update, render::render_world);
    }
}
