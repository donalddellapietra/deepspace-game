//! The world module is split into focused sub-modules, re-exported here:
//!
//! * [`chunk`]      — core voxel data (`Chunk`, `FlatWorld`).
//! * [`collision`]  — AABB collision + the `SolidQuery` trait.
//! * [`library`]    — content-addressed mesh cache (`MeshLibrary`) with
//!                    refcount + eviction.
//! * [`state`]      — `WorldState`, depth/zoom, and the editor write helpers
//!                    that keep the mesh library's refcounts correct.
//! * [`terrain`]    — deterministic `generate_chunk` and the streaming
//!                    terrain system.
//! * [`render`]     — `RenderState`, the render system, depth-aware
//!                    super-chunk / chunk rendering.

pub mod chunk;
pub mod collision;
pub mod library;
pub mod render;
pub mod state;
pub mod terrain;

use bevy::prelude::*;

pub use chunk::{Chunk, FlatWorld, SUPER};
pub use library::MeshLibrary;
pub use render::RenderState;
pub use state::{WorldState, MAX_DEPTH};
pub use terrain::StreamState;

/// How far the player can see, in *render entities* per axis. Same entity
/// budget at every depth (super-chunks at 0, chunks at 1/2).
pub const RENDER_DISTANCE: i32 = 24;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldState>()
            .init_resource::<RenderState>()
            .init_resource::<MeshLibrary>()
            .init_resource::<StreamState>()
            .add_systems(PreUpdate, terrain::generate_terrain)
            .add_systems(Update, render::render_world);
    }
}
