//! World bootstrap presets used by app startup and debugging.
//!
//! Low-level generation stays in `worldgen`. This module owns
//! composition: which world we start with, whether it contains a
//! planet, and where the default spawn should be.

use super::anchor::{Path, WorldPos};
use super::state::WorldState;

mod cube_faces_test;
mod dodecahedron_test;
mod plain;
mod rhombic_dodecahedron_test;
mod rotated_cube_test;
mod vox;
mod wrapped_planet;

pub use cube_faces_test::{cube_faces_test_spawn, cube_faces_test_world};
pub use dodecahedron_test::{dodecahedron_test_spawn, dodecahedron_test_world};
pub use rhombic_dodecahedron_test::{
    rhombic_dodecahedron_test_spawn, rhombic_dodecahedron_test_world,
};
pub use rotated_cube_test::{rotated_cube_test_spawn, rotated_cube_test_world};

pub use plain::{
    carve_air_pocket, plain_surface_spawn, plain_test_world, plain_world,
    DEFAULT_PLAIN_LAYERS,
};
pub use vox::bootstrap_vox_model_world;
pub use wrapped_planet::{
    wrapped_planet_spawn, wrapped_planet_world, DEFAULT_WRAPPED_PLANET_CELL_SUBTREE_DEPTH,
    DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH, DEFAULT_WRAPPED_PLANET_SLAB_DEPTH,
    DEFAULT_WRAPPED_PLANET_SLAB_DIMS,
};

/// Re-export of [`crate::world::fractals::menger::menger_world`] for
/// existing call-sites (e.g. `gpu/pack.rs` baseline tests). The full
/// colored bootstrap lives in [`crate::world::fractals::menger`].
pub use crate::world::fractals::menger::menger_world;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WorldPreset {
    #[default]
    PlainTest,
    /// Menger sponge — canonical ternary fractal. Each non-empty
    /// cell subdivides into 20 non-empty + 7 empty children (the 7
    /// are the cube centroid + 6 face centroids). 74% occupancy
    /// per level, no uniform collapse — stresses the packer's
    /// preserved-detail path in a way plain/sphere don't.
    Menger,
    /// Sierpinski tetrahedron — 4 tetrahedral corners per level
    /// (trinary adaptation of PySpace's `FoldSierpinski + FoldScale(2)`).
    /// Very sparse: 4/27 cells filled.
    SierpinskiTet,
    /// Cantor dust in 3D — 8 corner cells per level (all coords ∈
    /// {0, 2}). The canonical ternary set extended to three dimensions.
    /// Colored as an 8-hue prismatic orbit trap.
    CantorDust,
    /// Jerusalem cross / axial plus — the complement of Menger: 7
    /// cells (body centre + 6 face centres) per level. A delicate
    /// self-similar scaffold of orthogonal rods.
    JerusalemCross,
    /// Stepped Sierpinski pyramid — 4 base corners + 1 apex per
    /// level. Ziggurat-like self-similarity with a distinct "up" axis.
    SierpinskiPyramid,
    /// Mausoleum — Menger geometry with authentic PySpace orbit-trap
    /// ochre palette. Structurally equivalent to [`Menger`] but
    /// painted with `OrbitMax((0.42, 0.38, 0.19))` derived RGB
    /// instead of the hybrid bronze+blue.
    Mausoleum,
    /// Edge scaffold — 12 edge-midpoint rods per level. Neon axial
    /// palette (cyan/magenta/yellow per orientation).
    EdgeScaffold,
    /// Hollow cube — 18-cell architectural shell (12 edges + 6
    /// faces, no corners or body). Brushed-steel + brass palette.
    HollowCube,
    /// Imported `.vox` / `.vxs` model placed inside a plain world.
    /// Uses the GLB→`.vxs`→tree pipeline (see `src/import/` and
    /// `tools/scene_voxelize/`). The model is planted at the center
    /// of a plain world of depth `plain_layers` (default 8), so
    /// camera spawn is reasonable out-of-the-box. Stresses the
    /// packer's real-content path: tens of thousands of unique
    /// library nodes, every visible cell is mesh detail.
    ///
    /// `interior_depth`: if > 0, each model voxel (originally a
    /// `Child::Block`) is expanded into a uniform subtree of that
    /// depth, so the voxel becomes a diggable cube. The world's
    /// total depth is chosen so the silhouette lands at the same
    /// ancestor cell regardless of `interior_depth` (so spawn
    /// framing is stable across runs).
    VoxModel {
        path: std::path::PathBuf,
        interior_depth: u8,
    },
    /// Canonical high-resolution mesh scene (Sponza, San Miguel,
    /// Bistro) voxelized offline by `tools/scene_voxelize/`. See
    /// [`crate::world::scenes`].
    Scene {
        id: crate::world::scenes::SceneId,
    },
    /// Planet + distant stars at varying ribbon depths. Validates
    /// that the ray-march preserves precision across deep pops —
    /// stars at ancestor-depth 1 through N−1 must all render.
    Stars,
    /// Wrapped-Cartesian planet (Phase 1: hardcoded slab).
    /// A `NodeKind::WrappedPlane` node is installed at
    /// `embedding_depth` levels below root, with a flat slab subtree
    /// of `slab_dims.x × slab_dims.y × slab_dims.z` leaf cells
    /// `slab_depth` levels below the WrappedPlane node. No wrap and
    /// no curvature in this phase — it just renders as a small
    /// rectangular patch of grass/dirt/stone embedded in empty space.
    WrappedPlanet {
        embedding_depth: u8,
        slab_dims: [u32; 3],
        slab_depth: u8,
        cell_subtree_depth: u8,
    },
    /// Step-1 unit primitive: a single `TangentBlock` at tree depth 3,
    /// rotated 45° around Y, in an otherwise empty world.
    RotatedCubeTest,
    /// Step-2 multi-TB primitive: an unrotated centre cube surrounded
    /// by six face-aligned `TangentBlock` cubes (a 3D plus / cross).
    /// Each face TB carries the rotation that maps its storage +Y to
    /// the world outward direction of that face. Validates the
    /// rotation-aware `renormalize_world` against the case of multiple
    /// distinct TB rotations at the same tree level.
    CubeFacesTest,
    /// Step-3 multi-TB primitive: 12 `TangentBlock` cubes at the 12
    /// edge-adjacent positions of root, each oriented toward its
    /// world edge direction (the rhombic-dodecahedron face-normals,
    /// = the FCC nearest-neighbour set, = the 3D analog of the
    /// hexagon). Plus an unrotated centre cube. Each rotation is a
    /// 45°-style rotation; without `tb_scale` the corners poke into
    /// adjacent slots — visual artifact only, not a math bug.
    RhombicDodecahedronTest,
    /// Step-4 multi-TB primitive: 12 `TangentBlock` cubes occupying
    /// the same 12 edge-adjacent slots as `RhombicDodecahedronTest`,
    /// but each rotated to a *regular* dodecahedron face normal —
    /// `(0, ±1, ±φ)` and cyclic permutations, where `φ = (1+√5)/2`.
    /// The slot positions are still rounded to the integer grid, but
    /// the rotations encode the exact non-axis-aligned, golden-ratio
    /// face directions of a true regular dodecahedron. Stresses
    /// `renormalize_world` against twelve distinct *non*-axis-aligned
    /// rotations at the same tree level.
    DodecahedronTest,
}

/// World-coordinate Y where entities naturally rest. `Some(y)` for
/// worlds with a single flat ground plane; `None` for sphere /
/// fractal presets where "resting height" is position-dependent.
/// Callers consume this to drop the Y component of entity velocity
/// so they don't drift off the ground during long sessions.
pub fn surface_y_for_preset(preset: &WorldPreset) -> Option<f32> {
    match preset {
        WorldPreset::PlainTest => Some(plain::PLAIN_SURFACE_Y),
        // Imported .vox worlds embed the model in a plain world;
        // they inherit the same sea level.
        WorldPreset::VoxModel { .. } => Some(plain::PLAIN_SURFACE_Y),
        // Every fractal preset leaves entities to fly
        // freely — they don't have a single horizontal ground plane
        // that a constant sea-level Y could track.
        WorldPreset::Menger
        | WorldPreset::SierpinskiTet
        | WorldPreset::CantorDust
        | WorldPreset::JerusalemCross
        | WorldPreset::SierpinskiPyramid
        | WorldPreset::Mausoleum
        | WorldPreset::EdgeScaffold
        | WorldPreset::HollowCube
        | WorldPreset::Stars
        | WorldPreset::Scene { .. } => None,
        // The wrapped planet has a flat slab top at fixed local-y,
        // but its world-y depends on embedding_depth and slot path
        // — entities don't auto-rest on it in Phase 1.
        WorldPreset::WrappedPlanet { .. } => None,
        WorldPreset::RotatedCubeTest => None,
        WorldPreset::CubeFacesTest => None,
        WorldPreset::RhombicDodecahedronTest => None,
        WorldPreset::DodecahedronTest => None,
    }
}

pub struct WorldBootstrap {
    pub world: WorldState,
    pub planet_path: Option<Path>,
    /// Spawn position as a path-anchored `WorldPos`. Constructed at
    /// shallow depth (where f32 decomposition is precise) then
    /// `deepened_to` the target anchor depth via pure slot arithmetic.
    pub default_spawn_pos: WorldPos,
    pub default_spawn_yaw: f32,
    pub default_spawn_pitch: f32,
    pub plain_layers: u8,
    /// Color registry populated by the bootstrap — contains every
    /// palette entry the world needs to render. Callers take this
    /// instead of constructing a fresh `ColorRegistry::new()` so that
    /// imported-model colors (from `.vox`/`.vxs`) survive into the
    /// render path. Always contains the builtins as a prefix.
    pub color_registry: crate::world::palette::ColorRegistry,
}

pub fn bootstrap_world(preset: WorldPreset, plain_layers: Option<u8>) -> WorldBootstrap {
    match preset {
        WorldPreset::PlainTest => {
            plain::bootstrap_plain_test_world(plain_layers.unwrap_or(DEFAULT_PLAIN_LAYERS))
        }
        WorldPreset::Menger => crate::world::fractals::menger::bootstrap_menger_world(
            plain_layers.unwrap_or(8),
        ),
        WorldPreset::SierpinskiTet => {
            crate::world::fractals::sierpinski_tet::bootstrap_sierpinski_tet_world(
                plain_layers.unwrap_or(8),
            )
        }
        WorldPreset::CantorDust => {
            crate::world::fractals::cantor_dust::bootstrap_cantor_dust_world(
                plain_layers.unwrap_or(8),
            )
        }
        WorldPreset::JerusalemCross => {
            crate::world::fractals::jerusalem_cross::bootstrap_jerusalem_cross_world(
                plain_layers.unwrap_or(8),
            )
        }
        WorldPreset::SierpinskiPyramid => {
            crate::world::fractals::sierpinski_pyramid::bootstrap_sierpinski_pyramid_world(
                plain_layers.unwrap_or(8),
            )
        }
        WorldPreset::Mausoleum => {
            crate::world::fractals::mausoleum::bootstrap_mausoleum_world(
                plain_layers.unwrap_or(8),
            )
        }
        WorldPreset::EdgeScaffold => {
            crate::world::fractals::edge_scaffold::bootstrap_edge_scaffold_world(
                plain_layers.unwrap_or(8),
            )
        }
        WorldPreset::HollowCube => {
            crate::world::fractals::hollow_cube::bootstrap_hollow_cube_world(
                plain_layers.unwrap_or(8),
            )
        }
        WorldPreset::VoxModel { path, interior_depth } => {
            vox::bootstrap_vox_model_world(&path, plain_layers.unwrap_or(8), interior_depth)
        }
        WorldPreset::Scene { id } => crate::world::scenes::bootstrap_scene_world(id),
        WorldPreset::Stars => {
            crate::world::stars::bootstrap_stars_world(plain_layers.unwrap_or(40))
        }
        WorldPreset::WrappedPlanet {
            embedding_depth,
            slab_dims,
            slab_depth,
            cell_subtree_depth,
        } => wrapped_planet::bootstrap_wrapped_planet_world(
            embedding_depth,
            slab_dims,
            slab_depth,
            cell_subtree_depth,
        ),
        WorldPreset::RotatedCubeTest => {
            rotated_cube_test::bootstrap_rotated_cube_test_world()
        }
        WorldPreset::CubeFacesTest => {
            cube_faces_test::bootstrap_cube_faces_test_world()
        }
        WorldPreset::RhombicDodecahedronTest => {
            rhombic_dodecahedron_test::bootstrap_rhombic_dodecahedron_test_world()
        }
        WorldPreset::DodecahedronTest => {
            dodecahedron_test::bootstrap_dodecahedron_test_world()
        }
    }
}
