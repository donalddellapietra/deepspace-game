//! Content-addressed voxel tree. See `docs/architecture/` for the
//! design. Sub-modules:
//!
//! * [`tree`]      — `Node`, `NodeLibrary`, hashing, downsampling.
//! * [`position`]  — `Position`, `NodePath`, walking and neighbour walks.
//! * [`state`]     — `WorldState` (Bevy resource wrapping root + library).
//! * [`generator`] — procedural content (v1 returns grassland).
//! * [`edit`]      — leaf and higher-layer edit walks.
//! * [`view`]      — the single home for Bevy ↔ layer-space math, the
//!                   `(L+2).min(MAX_LAYER)` target-layer rule, and the
//!                   view-cell solidity query. Every module that
//!                   needs to place something in Bevy space or ask
//!                   "what's in this view cell" imports from here.
//! * [`collision`] — AABB collision, ground check.
//! * [`render`]    — uniform-layer tree-walk renderer + `CameraZoom`.

pub mod collision;
pub mod edit;
pub mod generator;
pub mod position;
pub mod render;
pub mod state;
pub mod tree;
pub mod view;

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
