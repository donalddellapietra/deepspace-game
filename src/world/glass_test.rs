//! Glass test world — a tiny hand-crafted scene designed to show off
//! per-block translucency.
//!
//! Layout (as viewed from the camera spawn looking +Z):
//!
//!  * A solid stone floor fills the bottom of the scene.
//!  * A translucent pale-cyan glass wall runs across the middle of the
//!    scene at a fixed z depth. Alpha ≈ 0.3 so everything behind it is
//!    visible but tinted.
//!  * A solid red brick wall sits further back, so the tint of the
//!    glass reads against a clearly-distinct background.
//!  * Three small solid colour blocks (blue, green, yellow) sit between
//!    the glass and the brick at varying depths — they should render
//!    with the glass tint laid over them.
//!
//! Tree depth is small (6) so each voxel projects to many pixels and
//! the translucent compositing path is easy to verify visually.

use crate::import::{tree_builder, VoxelModel, EMPTY_CELL};
use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, Child, NodeLibrary, CENTER_SLOT, MAX_DEPTH,
};

/// Side length of the voxel grid that holds the scene. 27 = 3³, which
/// is the silhouette size at tree depth 3 below the model-wrap layer.
/// Larger values produce finer detail but need more tree levels to
/// reach a comfortable camera standoff.
const GRID: usize = 27;

fn fill_box(
    model: &mut VoxelModel,
    x0: usize, x1: usize,
    y0: usize, y1: usize,
    z0: usize, z1: usize,
    palette_idx: u16,
) {
    for z in z0..z1.min(model.size_z) {
        for y in y0..y1.min(model.size_y) {
            for x in x0..x1.min(model.size_x) {
                let idx = model.index(x, y, z);
                model.data[idx] = palette_idx;
            }
        }
    }
}

/// Silhouette depth of the 27³ scene grid (log₃(27) = 3).
const SILHOUETTE_DEPTH: u8 = 3;

/// One wrap layer: scene sits in the center cell of the root so the
/// camera has empty air above/beside it to spawn in.
const WRAPS: u8 = 1;

/// Minimum tree depth: silhouette + wraps. At this depth every voxel
/// is a single Block leaf (not diggable); raise `plain-layers` to
/// give each voxel an interior subtree.
const MIN_DEPTH: u8 = SILHOUETTE_DEPTH + WRAPS;

pub fn bootstrap_glass_test_world(plain_layers: u8) -> WorldBootstrap {
    // `--plain-layers N` = total tree depth. The scene's physical
    // size and shape are fixed (one root-cell wrap, 27³ voxel
    // silhouette); what changes with N is how deep each voxel's
    // interior subtree is — i.e., how many layers you can drill
    // into a single glass/brick/colour block before hitting leaves.
    //
    // Mirrors the `--vox-model ... --vox-interior-depth N` pattern
    // (see `bootstrap_vox_model_world`) but derives interior depth
    // automatically from `plain-layers` so there's only one knob.
    let total_depth = plain_layers.max(MIN_DEPTH).min(MAX_DEPTH as u8);
    let interior_depth = total_depth.saturating_sub(SILHOUETTE_DEPTH + WRAPS);
    let mut registry = ColorRegistry::new();

    // Pale-cyan glass. Alpha 0.3 is well inside the translucent regime
    // (ALPHA_FLOOR = 0.02, ALPHA_CEIL = 0.95) so the block both tints
    // and passes through.
    let glass = registry
        .register(170, 220, 255, 77)
        .expect("palette not full");
    // Vivid blocks for the "behind the glass" sight lines. Full alpha
    // so they stay opaque — the whole point is to see them through a
    // translucent wall.
    let blue = registry
        .register(40, 90, 240, 255)
        .expect("palette not full");
    let green = registry
        .register(60, 220, 80, 255)
        .expect("palette not full");
    let yellow = registry
        .register(250, 220, 40, 255)
        .expect("palette not full");

    // Build a dense voxel grid and stamp the scene into it.
    let mut model = VoxelModel {
        size_x: GRID,
        size_y: GRID,
        size_z: GRID,
        data: vec![EMPTY_CELL; GRID * GRID * GRID],
    };

    // Stone floor — bottom 4 voxels.
    fill_box(&mut model, 0, GRID, 0, 4, 0, GRID, block::STONE);

    // Camera spawns at the +Z end of the scene and looks in -Z
    // (yaw = 0 → forward = (0, 0, -1) in this codebase). So the
    // scene is stacked so that the brick wall is at LOW z (far from
    // the camera), the glass wall is in the middle, and the coloured
    // blocks sit between them.

    // Brick wall at the "far" end (low z).
    fill_box(&mut model, 4, 24, 4, 16, 3, 5, block::BRICK);

    // Translucent glass wall in the middle. One voxel thick so the
    // compositing code sees it as a thin tinted pane.
    fill_box(&mut model, 4, 24, 4, 16, 14, 15, glass);

    // Solid cubes between the glass and the brick so everything
    // behind the glass shows up clearly tinted.
    fill_box(&mut model, 6, 10, 4, 8, 7, 11, blue);
    fill_box(&mut model, 12, 16, 4, 9, 8, 12, green);
    fill_box(&mut model, 18, 21, 4, 7, 9, 12, yellow);

    let mut lib = NodeLibrary::default();
    // Each non-empty voxel becomes a uniform subtree of depth
    // `interior_depth` — exactly like `bootstrap_vox_model_world`
    // with `--vox-interior-depth`. That's what makes every glass/
    // brick block drillable: `plain_layers − SILHOUETTE − WRAPS`
    // levels of recursion inside every cell.
    let model_root_id =
        tree_builder::build_tree_with_interior(&model, &mut lib, interior_depth);

    // Wrap once: scene sits in the center cell of the root.
    assert!((total_depth as usize) <= MAX_DEPTH);
    let mut root_children = empty_children();
    root_children[CENTER_SLOT] = Child::Node(model_root_id);
    let root_id = lib.insert(root_children);
    lib.ref_inc(root_id);
    let world = WorldState { root: root_id, library: lib };

    // Camera lives in the one wrap layer, in front of the scene.
    // Scene occupies root cell (1,1,1) which spans [1,2)³ in world
    // coords. Spawn at (1.5, 1.2, 2.3): horizontally centered on
    // the scene, just above the scene's floor in y, half-a-cell in
    // front of the scene's +Z face. Looking −Z (yaw=0) points the
    // camera through the glass wall toward the brick backdrop.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.5, 1.2, 2.3],
        WRAPS,
    );

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.1,
        plain_layers: total_depth,
        color_registry: registry,
    }
}
