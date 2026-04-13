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
pub mod heightmap;
pub mod instanced_overlay;
pub mod npc_compute;
pub mod overlay;
pub mod position;
pub mod render;
pub mod state;
pub mod tree;
pub mod view;

use bevy::prelude::*;

pub use position::Position;
pub use render::CameraZoom;
pub use render::RenderState;
pub use state::WorldState;
pub use view::WorldAnchor;

/// Authoritative tree-space location of a Bevy entity.
///
/// Every entity that exists "somewhere in the world" carries this
/// component. The contents are a [`Position`] — a path of 12 slot
/// indices, an in-leaf voxel, and a sub-voxel fractional offset —
/// so the location is expressed entirely in bounded components
/// with no large `f32` or `i64` anywhere, regardless of how many
/// billions of leaves the entity is from the root corner.
///
/// The entity's `Transform.translation` is **derived** from this
/// component each frame (see `player::derive_transforms`), using
/// the current [`WorldAnchor`] to subtract out the anchor's leaf
/// coord in exact `i64` space before casting to `f32`. The net
/// result is that `Transform.translation` is always a small
/// anchor-relative offset, and a `f32` never has to resolve a
/// step size smaller than a leaf no matter where the entity sits.
///
/// Physics, collision, and anything else that needs to mutate a
/// location takes `&mut WorldPosition` and operates on the
/// [`Position`] inside it directly via the tree walks in
/// [`position`] — never through the derived `Transform`. See
/// `collision::move_and_collide` for the canonical example.
#[derive(Component, Clone, Debug)]
pub struct WorldPosition(pub Position);

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldState>()
            .init_resource::<CameraZoom>()
            .init_resource::<RenderState>()
            .init_resource::<render::RenderTimings>()
            // The `WorldAnchor` starts at its default (origin leaf)
            // so the renderer and any pre-spawn systems see a valid
            // resource, but the player plugin overwrites it in its
            // Startup system to match the spawn position. The two
            // inserts race-free because Startup systems run strictly
            // after all `init_resource` / `insert_resource` calls
            // made in plugin `build`.
            .init_resource::<WorldAnchor>()
            .init_resource::<overlay::OverlayList>()
            .init_resource::<heightmap::GroundCache>()
            .add_systems(
                Update,
                (
                    heightmap::reset_ground_cache,
                    render::render_world
                        .after(crate::player::derive_transforms)
                        .after(crate::npc::collect_overlays_from_buffer),
                ),
            );
    }
}
