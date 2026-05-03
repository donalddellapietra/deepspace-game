//! High-resolution mesh scenes imported via `tools/scene_voxelize/`.
//!
//! Each scene is a GLB/glTF voxelized offline into `.vxs` (see
//! `assets/scenes/sources/README.md` for how to generate them) and
//! loaded at runtime through the existing `.vxs` → tree pipeline
//! (`import::load` → `tree_builder::build_tree_with_interior`).
//!
//! The canonical scenes are classic graphics benchmarks:
//! Sponza, San Miguel, and Amazon Lumberyard Bistro. Each has a
//! natural physical scale — the `total_depth` below is picked so the
//! scene's voxel footprint occupies the inner tree layers with one
//! wrap of surrounding air for spawn framing.

use super::bootstrap::{WorldBootstrap, bootstrap_vox_model_world};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneId {
    /// Crytek Sponza (Khronos glTF Sample Models). Iconic interior
    /// atrium — the canonical rasterization/GI benchmark since 2010.
    Sponza,
    /// Morgan McGuire's San Miguel restaurant courtyard. 10.5M
    /// triangles of outdoor arcade with heavy foliage.
    SanMiguel,
    /// Amazon Lumberyard Bistro (NVIDIA ORCA). Paris street corner,
    /// interior + exterior, ~3M triangles.
    Bistro,
}

impl SceneId {
    pub const ALL: [SceneId; 3] = [SceneId::Sponza, SceneId::SanMiguel, SceneId::Bistro];

    /// Short lowercase slug matching the CLI flag and generated file
    /// basename (`assets/scenes/generated/<slug>.vxs`).
    pub fn slug(self) -> &'static str {
        match self {
            SceneId::Sponza => "sponza",
            SceneId::SanMiguel => "san_miguel",
            SceneId::Bistro => "bistro",
        }
    }

    /// Target total tree depth for this scene. Chosen so the voxel
    /// silhouette fills the inner world with 2 wraps of surrounding
    /// air — the model occupies `1/9` of the root cell per axis, with
    /// one extra wrap of air beyond its bounding box for the camera
    /// to navigate without immediately clipping.
    ///
    /// Sponza voxelizes to 384×384×320 at 16 v/m → silhouette 6 →
    /// total 8. San Miguel and Bistro voxelize larger; we'll tune
    /// their totals once they're generated.
    pub fn total_depth(self) -> u8 {
        match self {
            SceneId::Sponza => 8,
            SceneId::SanMiguel => 9,
            SceneId::Bistro => 9,
        }
    }

    /// Resolved path to the generated `.vxs`. Lives alongside the
    /// source GLB in `assets/scenes/` — both dir and contents are
    /// gitignored (multi-GB assets fetched on demand).
    pub fn path(self) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("assets/scenes/{}.vxs", self.slug()))
    }

    /// Parse a CLI-style slug (`sponza`, `san_miguel`, `san-miguel`,
    /// `bistro`) into a `SceneId`.
    pub fn from_slug(slug: &str) -> Option<SceneId> {
        let s = slug.to_ascii_lowercase().replace('-', "_");
        match s.as_str() {
            "sponza" => Some(SceneId::Sponza),
            "san_miguel" | "sanmiguel" => Some(SceneId::SanMiguel),
            "bistro" | "amazon_lumberyard_bistro" => Some(SceneId::Bistro),
            _ => None,
        }
    }
}

/// Bootstrap a scene preset. The scene's voxel footprint is planted
/// into a plain air world of `total_depth` layers via the existing
/// `.vxs` import pipeline.
///
/// Panics if the `.vxs` is missing — the caller is expected to have
/// generated it via `scripts/build-scenes.sh` first. The panic message
/// includes the exact generator command.
pub fn bootstrap_scene_world(id: SceneId) -> WorldBootstrap {
    let path = id.path();
    if !path.exists() {
        panic!(
            "scene `{}` not found at {:?}.\n\
             To generate it:\n  \
             1. scripts/fetch-glb-presets.sh {0}\n  \
             2. (cd tools/scene_voxelize && cargo run --release -- --models {0})\n\
             (source GLB must end up at assets/scenes/{0}.glb; .vxs will be written \
             next to it.)",
            id.slug(),
            path,
        );
    }
    bootstrap_vox_model_world(&path, id.total_depth(), 0)
}
