//! World bootstrap presets used by app startup and debugging.
//!
//! Low-level generation stays in `worldgen`. This module owns
//! composition: which world we start with and where the default
//! spawn should be.

use super::anchor::{Path, WorldPos, WORLD_SIZE};
use super::palette::block;
use super::state::WorldState;
use super::tree::{
    empty_children, slot_index, uniform_children, BRANCH, Child, NodeId, NodeKind, NodeLibrary,
    MAX_DEPTH,
};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WorldPreset {
    #[default]
    PlainTest,
    /// Menger sponge — canonical ternary fractal. Each non-empty
    /// cell subdivides into 20 non-empty + 7 empty children (the 7
    /// are the cube centroid + 6 face centroids). 74% occupancy
    /// per level, no uniform collapse — stresses the packer's
    /// preserved-detail path in a way plain doesn't.
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
}

pub const DEFAULT_PLAIN_LAYERS: u8 = 40;
const PLAIN_SURFACE_Y: f32 = 1.0;
const PLAIN_GRASS_THICKNESS: f32 = 0.05;
const PLAIN_DIRT_THICKNESS: f32 = 0.25;

/// World-coordinate Y where entities naturally rest. `Some(y)` for
/// worlds with a single flat ground plane; `None` for fractal
/// presets where "resting height" is position-dependent.
/// Callers consume this to drop the Y component of entity velocity
/// so they don't drift off the ground during long sessions.
pub fn surface_y_for_preset(preset: &WorldPreset) -> Option<f32> {
    match preset {
        WorldPreset::PlainTest => Some(PLAIN_SURFACE_Y),
        // Imported .vox worlds embed the model in a plain world;
        // they inherit the same sea level.
        WorldPreset::VoxModel { .. } => Some(PLAIN_SURFACE_Y),
        // Every fractal preset leaves entities to fly freely —
        // they don't have a single horizontal ground plane that a
        // constant sea-level Y could track.
        WorldPreset::Menger
        | WorldPreset::SierpinskiTet
        | WorldPreset::CantorDust
        | WorldPreset::JerusalemCross
        | WorldPreset::SierpinskiPyramid
        | WorldPreset::Mausoleum
        | WorldPreset::EdgeScaffold
        | WorldPreset::HollowCube
        | WorldPreset::Scene { .. } => None,
    }
}

pub struct WorldBootstrap {
    pub world: WorldState,
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
        WorldPreset::PlainTest => bootstrap_plain_test_world(plain_layers.unwrap_or(DEFAULT_PLAIN_LAYERS)),
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
        WorldPreset::VoxModel { path, interior_depth } => bootstrap_vox_model_world(
            &path, plain_layers.unwrap_or(8), interior_depth,
        ),
        WorldPreset::Scene { id } => crate::world::scenes::bootstrap_scene_world(id),
    }
}

/// Re-export of [`crate::world::fractals::menger::menger_world`] for
/// existing call-sites (e.g. `gpu/pack.rs` baseline tests). The full
/// colored bootstrap lives in [`crate::world::fractals::menger`].
pub use crate::world::fractals::menger::menger_world;

/// Load a `.vox` file via the import pipeline and embed it as the
/// child of a tree of `total_depth` levels. The model sits at the
/// center slot of each wrapper layer so the camera can spawn in
/// air nearby and look at it from any side.
///
/// Panics if the file doesn't exist or doesn't parse — this is a
/// bootstrap function, not a content runtime path.
pub(crate) fn bootstrap_vox_model_world(
    path: &std::path::Path,
    total_depth: u8,
    interior_depth: u8,
) -> WorldBootstrap {
    use crate::import::{self, tree_builder};
    use crate::world::tree::{CENTER_SLOT, empty_children};

    let total_depth = total_depth.max(1).min(MAX_DEPTH as u8);

    let mut lib = NodeLibrary::default();
    let mut registry = crate::world::palette::ColorRegistry::new();

    // Use the sparse path for `.vxs` when `interior_depth == 0` —
    // iterates only occupied voxels instead of the padded cube. For
    // Sponza (5 M voxels in a 729³ ≈ 388 M padded cube) this is
    // ~77× less work at load time.
    let ext_is_vxs = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("vxs"))
        .unwrap_or(false);

    let (size_x, size_y, size_z, model_root_id) = if ext_is_vxs && interior_depth == 0 {
        let t = std::time::Instant::now();
        let sparse = import::vxs::load_sparse(path, &mut registry)
            .unwrap_or_else(|e| panic!("failed to load {:?}: {}", path, e));
        eprintln!(
            "vox_world: loaded (sparse) {:?} ({}x{}x{} = {} voxels) in {:.2?}",
            path,
            sparse.size_x,
            sparse.size_y,
            sparse.size_z,
            sparse.voxels.len(),
            t.elapsed(),
        );
        let t = std::time::Instant::now();
        let root = tree_builder::build_tree_sparse(&sparse, &mut lib);
        eprintln!(
            "vox_world: sparse tree build in {:.2?} ({} library nodes)",
            t.elapsed(),
            lib.len(),
        );
        (
            sparse.size_x as usize,
            sparse.size_y as usize,
            sparse.size_z as usize,
            root,
        )
    } else {
        let model = import::load(path, &mut registry)
            .unwrap_or_else(|e| panic!("failed to load {:?}: {}", path, e));
        eprintln!(
            "vox_world: loaded {:?} ({}x{}x{} = {} voxels, interior_depth={})",
            path,
            model.size_x,
            model.size_y,
            model.size_z,
            model.data.iter().filter(|&&v| v != 0).count(),
            interior_depth,
        );
        let root =
            tree_builder::build_tree_with_interior(&model, &mut lib, interior_depth);
        (model.size_x, model.size_y, model.size_z, root)
    };

    // Silhouette depth: log3(max_dim). Each voxel then contributes
    // `interior_depth` additional levels (uniform-fill subtree),
    // so the model's full tree footprint is silhouette + interior.
    let max_dim = size_x.max(size_y).max(size_z).max(1);
    let mut dim = 1usize;
    let mut silhouette_depth: u8 = 0;
    while dim < max_dim {
        dim *= BRANCH;
        silhouette_depth += 1;
    }
    let model_depth = silhouette_depth.saturating_add(interior_depth);
    eprintln!(
        "vox_world: silhouette depth={}, interior depth={}, model total={}, library nodes after build={}",
        silhouette_depth, interior_depth, model_depth, lib.len(),
    );

    // Wrap in outer air layers up to target total_depth. Each wrap
    // places the current subtree at the center slot of a 27-child
    // node whose other 26 slots are Empty.
    //
    // When `vox_copies > 1`, the FINAL wrap layer fills multiple slots
    // instead of just the center — content-addressed dedup means all
    // copies share the same NodeId, so library size doesn't grow, but
    // ray-traversal working set does. Used for scene-scaling stress tests.
    let copies = std::env::var("VOX_COPIES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1)
        .clamp(1, 27);
    let mut current = model_root_id;
    let wraps = total_depth.saturating_sub(model_depth);
    for w in 0..wraps {
        let mut children = empty_children();
        if copies > 1 && w == wraps - 1 {
            // Outermost wrap: place `copies` instances across slots.
            for slot in 0..(copies as usize) {
                children[slot] = Child::Node(current);
            }
        } else {
            children[CENTER_SLOT] = Child::Node(current);
        }
        current = lib.insert(children);
    }
    eprintln!("vox_world: placed {} copies in outermost wrap", copies);
    eprintln!(
        "vox_world: wrapped {} layers, final tree depth={}, library total={}",
        wraps, wraps + model_depth, lib.len(),
    );

    lib.ref_inc(current);
    let world = WorldState { root: current, library: lib };

    // Compute the model's world-space bounds so spawn is always at a
    // position the camera can see the model from, regardless of the
    // model's voxel dimensions or how deep the wrap put it.
    //
    // Geometry: the model is wrapped at `CENTER_SLOT=(1,1,1)` of each
    // outer layer, so it lives in cell (1,1,1)^(wraps) of the root.
    // That cell's world extent is `[wrap_origin, wrap_origin+wrap_size]^3`
    // with `wrap_size = 3 * (1/3)^wraps` and `wrap_origin = 1.5 - wrap_size/2`.
    // Inside that cell, the model's voxel grid fills
    // `[0, size_axis/padded]` of the cell, where `padded = 3^silhouette_depth`
    // (silhouette-only; interior_depth adds layers *inside* each voxel but
    // doesn't change the voxel grid's extent inside the wrap cell).
    let padded = BRANCH.pow(silhouette_depth as u32) as f32;
    let wrap_size = 3.0 * (1.0 / BRANCH as f32).powi(wraps as i32);
    let wrap_origin = 1.5 - wrap_size / 2.0;
    let extent_x = wrap_size * (size_x as f32 / padded);
    let extent_y = wrap_size * (size_y as f32 / padded);
    let extent_z = wrap_size * (size_z as f32 / padded);
    let center_x = wrap_origin + extent_x / 2.0;
    let center_y = wrap_origin + extent_y / 2.0;
    let center_z = wrap_origin + extent_z / 2.0;
    eprintln!(
        "vox_world: model world bounds x=[{:.3}..{:.3}] y=[{:.3}..{:.3}] z=[{:.3}..{:.3}], center=({:.3},{:.3},{:.3})",
        wrap_origin, wrap_origin + extent_x,
        wrap_origin, wrap_origin + extent_y,
        wrap_origin, wrap_origin + extent_z,
        center_x, center_y, center_z,
    );

    // Spawn directly above the model's x/z centroid, outside the
    // wrap cell (y > wrap_cell_max), looking straight down.
    //
    // Why this spawn layout works reliably:
    //  - Camera x,z inside the model's voxel footprint → rays going
    //    down the pitch axis hit model cells (not empty siblings).
    //  - Camera y > 2.0 places it in a root cell ABOVE the wrap cell,
    //    so rays traverse an empty cell first, then enter the model
    //    tree via the wrap — no "inside-the-model-tree at spawn"
    //    edge case.
    //  - Pitch ≈ -π/2 keeps the ray direction aligned with the
    //    thin-slab Z axis of most humanoid GLBs (soldier, fox), so
    //    you see the whole silhouette in one view.
    let cam_x = center_x.clamp(0.05, 2.95);
    let cam_y = (wrap_origin + wrap_size + 0.5).min(2.95);
    let cam_z = center_z.clamp(0.05, 2.95);
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [cam_x, cam_y, cam_z],
        2,
    ).deepened_to(3);
    let yaw = 0.0;
    let pitch = -1.5;   // nearly straight down

    WorldBootstrap {
        world,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: yaw,
        default_spawn_pitch: pitch,
        plain_layers: total_depth,
        color_registry: registry,
    }
}

pub fn plain_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    let stone_l1 = library.insert(uniform_children(Child::Block(block::STONE)));
    let dirt_l1 = library.insert(uniform_children(Child::Block(block::DIRT)));
    let grass_l1 = library.insert(uniform_children(Child::Block(block::GRASS)));
    let wood_l1 = library.insert(uniform_children(Child::Block(block::WOOD)));
    let leaf_l1 = library.insert(uniform_children(Child::Block(block::LEAF)));
    let sand_l1 = library.insert(uniform_children(Child::Block(block::SAND)));
    let brick_l1 = library.insert(uniform_children(Child::Block(block::BRICK)));
    let air_l1 = library.insert(empty_children());

    let mut checker_children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                let slot = slot_index(x, y, z);
                checker_children[slot] = if (x + y + z) % 2 == 0 {
                    Child::Block(block::STONE)
                } else {
                    Child::Block(block::DIRT)
                };
            }
        }
    }
    let checker_l1 = library.insert(checker_children);

    let stone_l2 = library.insert(uniform_children(Child::Node(stone_l1)));

    let mut checker_ground_children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                let slot = slot_index(x, y, z);
                checker_ground_children[slot] = if (x + z) % 2 == 0 {
                    Child::Node(stone_l1)
                } else {
                    Child::Node(checker_l1)
                };
            }
        }
    }
    let checker_ground_l2 = library.insert(checker_ground_children);

    let mut grass_surface_children = empty_children();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            let slot_bottom = slot_index(x, 0, z);
            let slot_mid = slot_index(x, 1, z);
            let slot_top = slot_index(x, 2, z);
            grass_surface_children[slot_bottom] = Child::Node(dirt_l1);
            grass_surface_children[slot_mid] = Child::Node(grass_l1);
            grass_surface_children[slot_top] = Child::Node(air_l1);
        }
    }
    let grass_surface_l2 = library.insert(grass_surface_children);

    let air_l2 = library.insert(uniform_children(Child::Node(air_l1)));

    let mut features_children = empty_children();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            for y in 0..BRANCH {
                features_children[slot_index(x, y, z)] = Child::Node(air_l1);
            }
        }
    }
    features_children[slot_index(1, 0, 1)] = Child::Node(wood_l1);
    features_children[slot_index(1, 1, 1)] = Child::Node(wood_l1);
    features_children[slot_index(1, 2, 1)] = Child::Node(leaf_l1);
    features_children[slot_index(0, 0, 0)] = Child::Node(brick_l1);
    features_children[slot_index(0, 0, 1)] = Child::Node(brick_l1);
    features_children[slot_index(0, 0, 2)] = Child::Node(brick_l1);
    features_children[slot_index(0, 1, 0)] = Child::Node(brick_l1);
    features_children[slot_index(0, 1, 1)] = Child::Node(brick_l1);
    features_children[slot_index(0, 1, 2)] = Child::Node(brick_l1);
    features_children[slot_index(2, 0, 0)] = Child::Node(sand_l1);
    let features_l2 = library.insert(features_children);

    let mut root_children = empty_children();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            root_children[slot_index(x, 0, z)] = if (x + z) % 2 == 0 {
                Child::Node(stone_l2)
            } else {
                Child::Node(checker_ground_l2)
            };
            root_children[slot_index(x, 1, z)] = Child::Node(grass_surface_l2);
            root_children[slot_index(x, 2, z)] = if x == 1 && z == 1 {
                Child::Node(features_l2)
            } else {
                Child::Node(air_l2)
            };
        }
    }

    let root = library.insert(root_children);
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "Plain test world: {} library entries, root spans [0, {WORLD_SIZE}) per axis",
        world.library.len(),
    );
    world
}

pub fn plain_world(layers: u8) -> WorldState {
    assert!(layers > 0, "plain world must have at least one layer");
    assert!(
        (layers as usize) <= MAX_DEPTH,
        "plain world layers {} exceeds MAX_DEPTH {}",
        layers,
        MAX_DEPTH,
    );

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    enum UniformFill {
        Empty,
        Block(u16),
    }

    fn fill_for_range(y_min: f32, y_max: f32) -> Option<UniformFill> {
        let grass_min = PLAIN_SURFACE_Y - PLAIN_GRASS_THICKNESS;
        let dirt_min = PLAIN_SURFACE_Y - PLAIN_DIRT_THICKNESS;
        if y_min >= PLAIN_SURFACE_Y {
            Some(UniformFill::Empty)
        } else if y_max <= dirt_min {
            Some(UniformFill::Block(block::STONE))
        } else if y_min >= grass_min && y_max <= PLAIN_SURFACE_Y {
            Some(UniformFill::Block(block::GRASS))
        } else if y_min >= dirt_min && y_max <= grass_min {
            Some(UniformFill::Block(block::DIRT))
        } else {
            None
        }
    }

    fn uniform_subtree(
        lib: &mut NodeLibrary,
        cache: &mut HashMap<(u8, UniformFill), NodeId>,
        depth: u8,
        fill: UniformFill,
    ) -> NodeId {
        if let Some(&id) = cache.get(&(depth, fill)) {
            return id;
        }
        let id = if depth == 1 {
            match fill {
                UniformFill::Empty => lib.insert(empty_children()),
                UniformFill::Block(block_type) => lib.insert(uniform_children(Child::Block(block_type))),
            }
        } else {
            let child = uniform_subtree(lib, cache, depth - 1, fill);
            lib.insert(uniform_children(Child::Node(child)))
        };
        cache.insert((depth, fill), id);
        id
    }

    fn build_plain_subtree(
        lib: &mut NodeLibrary,
        cache: &mut HashMap<(u8, UniformFill), NodeId>,
        depth: u8,
        y_min: f32,
        y_max: f32,
    ) -> NodeId {
        if let Some(fill) = fill_for_range(y_min, y_max) {
            return uniform_subtree(lib, cache, depth, fill);
        }

        if depth == 1 {
            let mut children = empty_children();
            let child_size = (y_max - y_min) / BRANCH as f32;
            for z in 0..BRANCH {
                for y in 0..BRANCH {
                    let row_min = y_min + child_size * y as f32;
                    let row_max = row_min + child_size;
                    let child = match fill_for_range(row_min, row_max) {
                        Some(UniformFill::Empty) => Child::Empty,
                        Some(UniformFill::Block(block_type)) => Child::Block(block_type),
                        None => {
                            let mid = (row_min + row_max) * 0.5;
                            if mid >= PLAIN_SURFACE_Y {
                                Child::Empty
                            } else if mid >= PLAIN_SURFACE_Y - PLAIN_GRASS_THICKNESS {
                                Child::Block(block::GRASS)
                            } else if mid >= PLAIN_SURFACE_Y - PLAIN_DIRT_THICKNESS {
                                Child::Block(block::DIRT)
                            } else {
                                Child::Block(block::STONE)
                            }
                        }
                    };
                    for x in 0..BRANCH {
                        children[slot_index(x, y, z)] = child;
                    }
                }
            }
            return lib.insert(children);
        }

        let mut children = empty_children();
        let child_size = (y_max - y_min) / BRANCH as f32;
        for y in 0..BRANCH {
            let row_min = y_min + child_size * y as f32;
            let row_max = row_min + child_size;
            let child_id = build_plain_subtree(lib, cache, depth - 1, row_min, row_max);
            let row_child = Child::Node(child_id);
            for z in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] = row_child;
                }
            }
        }
        lib.insert(children)
    }

    let mut library = NodeLibrary::default();
    let mut uniform_cache = HashMap::new();
    let root = build_plain_subtree(&mut library, &mut uniform_cache, layers, 0.0, WORLD_SIZE);
    library.ref_inc(root);
    let world = WorldState { root, library };
    eprintln!(
        "Plain world: layers={}, library_entries={}, depth={}",
        layers,
        world.library.len(),
        world.tree_depth(),
    );
    world
}

/// Build a spawn position that tracks the dirt/grass boundary at any
/// anchor depth. The boundary at y ≈ 0.95 (PLAIN_SURFACE_Y −
/// PLAIN_GRASS_THICKNESS) is NOT ternary-rational: its base-3
/// expansion is 0.2211̄ (repeating "2211"). This means every render
/// frame at this y contains mixed dirt/grass children — the grid
/// between different block colors is always visible.
///
/// f64 arithmetic loses precision past depth ~34 (cell_size drops
/// below the f64 epsilon), so we use the exact periodic ternary
/// expansion directly.
pub fn plain_surface_spawn(anchor_depth: u8) -> WorldPos {
    // y ≈ 0.95 in [0, WORLD_SIZE=3). Ternary expansion:
    //   depth 1:  digit 0  (root y ∈ [0,1))
    //   depth 2+: repeating [2, 2, 1, 1]
    const Y_PATTERN: [usize; 4] = [2, 2, 1, 1];

    let mut path = Path::root();
    for d in 0..anchor_depth as usize {
        let y_row = if d == 0 { 0 } else { Y_PATTERN[(d - 1) % 4] };
        let slot = slot_index(1, y_row, 1); // x=1, z=1 center
        path.push(slot as u8);
    }

    // Camera near center of cell. After carve_air_pocket clears
    // this cell, the camera is in a 1-block air pocket looking at
    // surrounding dirt/grass blocks.
    WorldPos::new(path, [0.5, 0.5, 0.5])
}

/// Create a uniform air subtree of the given depth. All children are
/// recursively air nodes, so the render-frame tree walk can descend
/// through this region just like it would through a normal tree.
/// Content-addressed dedup means only O(depth) nodes are allocated
/// regardless of how wide the subtree is.
fn air_subtree(lib: &mut NodeLibrary, depth: u8) -> NodeId {
    if depth == 0 {
        return lib.insert(empty_children());
    }
    let child = air_subtree(lib, depth - 1);
    lib.insert(uniform_children(Child::Node(child)))
}

/// Ensure the camera's anchor path is tree-walkable down to
/// `anchor.depth()`, inserting a fresh empty Node at any slot that
/// was `Child::Empty` or `Child::Block` along the walk, and carving
/// an air subtree at the final slot of depth `anchor.depth() - 1`.
///
/// Two guarantees for the renderer:
/// 1. `compute_render_frame` can walk the anchor path all the way
///    down — critical for fractals (Menger's body-centre Empties,
///    Sierpinski's unused corner slots, etc.) where the path would
///    otherwise stall on a structural Empty at depth 2–3.
/// 2. The last cell (at `anchor.depth()`) is always air, so plain-
///    world spawn lands in an air pocket rather than inside a
///    dirt/grass block.
///
/// The expand-on-walk is a side effect the renderer *needs*; the
/// final-cell carve is a side effect plain-worlds *want*. Both
/// happen together because a single bottom-up rebuild stitches the
/// new child IDs upward through the whole anchor path.
///
/// The air subtree below the final cell extends to `total_depth` so
/// the user can zoom to any depth inside the cavity and still get a
/// deep, walkable render frame.
pub fn carve_air_pocket(world: &mut WorldState, anchor: &Path, total_depth: u8) {
    if anchor.depth() < 2 { return; }
    let slots = anchor.as_slice();
    let carve_depth = (anchor.depth() - 1) as usize;
    let mut node_stack: Vec<(NodeId, NodeKind)> = Vec::with_capacity(carve_depth + 1);
    let mut node_id = world.root;
    for &slot in &slots[..carve_depth] {
        let Some(node) = world.library.get(node_id) else { return };
        node_stack.push((node_id, node.kind));
        match node.children[slot as usize] {
            Child::Node(child_id) => node_id = child_id,
            // If the camera's anchor path crosses an Empty or Block
            // slot before reaching `carve_depth`, install a fresh
            // empty Node there so the walk can continue. Bottom-up
            // rebuild below stitches the new Node into the parent's
            // slot via the replacement chain — the rest of the
            // parent's siblings stay untouched. This is the fix
            // for fractals with structural empties (Menger's body-
            // centres, Sierpinski's 23 unused corners etc.): without
            // it, `compute_render_frame` stalls at the first empty
            // and the shader can never render cells small enough
            // for Nyquist to let a real Block leaf be visible,
            // which manifests as monochromatic LOD-terminal colour.
            // EntityRef should never appear in a terrain edit path
            // — entities only land in ephemeral scene-root nodes —
            // but if it does, treat it like an empty and install a
            // fresh air Node so carve can proceed without panicking.
            Child::Empty | Child::Block(_) | Child::EntityRef(_) => {
                node_id = world.library.insert(empty_children());
            }
        }
    }
    let Some(node) = world.library.get(node_id) else { return };
    node_stack.push((node_id, node.kind));

    let clear_slot = slots[carve_depth] as usize;

    // How many additional levels of air nodes below the cleared cell.
    // The cleared cell is at absolute depth `carve_depth + 1`.
    // We need air down to `total_depth`.
    let cleared_abs_depth = (carve_depth + 1) as u8;
    let air_depth = total_depth.saturating_sub(cleared_abs_depth);
    let air_node = if air_depth > 0 {
        Child::Node(air_subtree(&mut world.library, air_depth))
    } else {
        Child::Empty
    };

    // Build replacement bottom-up.
    let mut replacement: Option<NodeId> = None;
    for (i, &(nid, kind)) in node_stack.iter().enumerate().rev() {
        let Some(node) = world.library.get(nid) else { return };
        let mut new_children = node.children;
        if let Some(rep) = replacement {
            new_children[slots[i] as usize] = Child::Node(rep);
        } else {
            new_children[clear_slot] = air_node;
        }
        replacement = Some(world.library.insert_with_kind(new_children, kind));
    }
    if let Some(new_root) = replacement {
        world.swap_root(new_root);
    }
}

fn bootstrap_plain_test_world(plain_layers: u8) -> WorldBootstrap {
    let world = plain_world(plain_layers);
    let spawn_pos = plain_surface_spawn(8);
    // NOTE: don't carve here — the default spawn is at depth 8 and
    // carving here would clear a cell that's an ancestor of any deeper
    // --spawn-depth override. Carving happens in App::with_test_config
    // after the final spawn position is known.
    WorldBootstrap {
        world,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.45,
        plain_layers,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_world_honors_requested_depth() {
        let world = plain_world(40);
        assert_eq!(world.tree_depth(), 40);
    }
}
