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

/// Target total tree depth. Small enough that voxels project to many
/// pixels at the spawn distance, large enough to give the camera some
/// ancestor cells to sit in.
const TOTAL_DEPTH: u8 = 6;

pub fn bootstrap_glass_test_world(_plain_layers: u8) -> WorldBootstrap {
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
    let model_root = tree_builder::build_tree(&model, &mut lib);

    // GRID=27 → silhouette depth = 3. Wrap up to TOTAL_DEPTH by
    // placing the model at the center slot of each outer layer so
    // there's empty space above/below/around the scene for the
    // camera to sit in.
    let silhouette_depth = 3u8;
    let wraps = TOTAL_DEPTH.saturating_sub(silhouette_depth);
    assert!((TOTAL_DEPTH as usize) <= MAX_DEPTH);
    let mut current = model_root;
    for _ in 0..wraps {
        let mut children = empty_children();
        children[CENTER_SLOT] = Child::Node(current);
        current = lib.insert(children);
    }
    lib.ref_inc(current);
    let world = WorldState { root: current, library: lib };

    // Camera sits in front of the glass wall looking in -Z (yaw=0,
    // codebase's default forward is world -Z). The wrap puts the
    // model at cell (1,1,1) of the root, so the model's world-space
    // footprint is  [wrap_origin, wrap_origin + wrap_size]^3  with
    //   wrap_size   = 3 * (1/3)^wraps
    //   wrap_origin = 1.5 - wrap_size / 2
    let wrap_size = 3.0f32 * (1.0f32 / 3.0f32).powi(wraps as i32);
    let wrap_origin = 1.5 - wrap_size / 2.0;
    // Voxel coord → world coord.
    let voxel_to_world = |vx: f32| wrap_origin + (vx / GRID as f32) * wrap_size;
    // x: middle of scene. y: 3 voxels above the floor (floor is 4
    // thick). z: at voxel 22, a few cells ahead of the glass (z=14)
    // so there's room for the glass + content + brick backdrop in
    // the field of view.
    let cam_x = voxel_to_world(13.5).clamp(0.01, 2.99);
    let cam_y = voxel_to_world(7.5).clamp(0.01, 2.99);
    let cam_z = voxel_to_world(22.0).clamp(0.01, 2.99);
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [cam_x, cam_y, cam_z],
        2,
    )
    .deepened_to(TOTAL_DEPTH.min(8));

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.1,
        plain_layers: TOTAL_DEPTH,
        color_registry: registry,
    }
}
