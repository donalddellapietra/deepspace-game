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

/// Minimum tree depth: the hand-crafted 27³ scene occupies 3 levels
/// (silhouette), plus 1 wrap layer for the camera to sit in.
const MIN_DEPTH: u8 = 4;

pub fn bootstrap_glass_test_world(plain_layers: u8) -> WorldBootstrap {
    // `--plain-layers N` controls how many times the scene is wrapped
    // inside outer empty nodes — i.e., how small the scene is relative
    // to the world root. Larger N ⇒ scene placed deeper in the tree
    // ⇒ camera has to zoom in (deepen anchor) to see it at comparable
    // angular size. N=6 is the old default (3 wraps). N=20 nests it
    // 17 wraps deep: UI layer changes to reach the scene.
    let total_depth = plain_layers.max(MIN_DEPTH).min(MAX_DEPTH as u8);
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

    // GRID=27 → silhouette depth = 3. Wrap up to total_depth by
    // placing the model at the center slot of each outer layer so
    // there's empty space above/below/around the scene for the
    // camera to sit in.
    let silhouette_depth = 3u8;
    let wraps = total_depth.saturating_sub(silhouette_depth);
    assert!((total_depth as usize) <= MAX_DEPTH);
    let mut current = model_root;
    for _ in 0..wraps {
        let mut children = empty_children();
        children[CENTER_SLOT] = Child::Node(current);
        current = lib.insert(children);
    }
    lib.ref_inc(current);
    let world = WorldState { root: current, library: lib };

    // Camera sits in front of the glass wall looking in -Z (yaw=0,
    // codebase's default forward is world -Z). The scene lives at
    // path [CENTER_SLOT; wraps] — i.e., the center cell of each
    // wrap layer. Express the camera's position in the SCENE's own
    // frame so precision doesn't drop at deep wraps:
    //   scene frame spans [0, 3)³, covering 27³ voxels → 1 voxel =
    //   (1/9) frame units. Camera in voxel coords = (13.5, 7.5, 22).
    let mut scene_frame = Path::root();
    for _ in 0..wraps {
        scene_frame.push(CENTER_SLOT as u8);
    }
    let voxel_to_frame = |vx: f32| 3.0 * (vx / GRID as f32);
    let cam_x = voxel_to_frame(13.5);
    let cam_y = voxel_to_frame(7.5);
    let cam_z = voxel_to_frame(22.0);
    // Anchor depth: place it at the scene's silhouette depth (one
    // level above the voxel leaves) so the camera can freely move
    // within the scene without renormalize thrashing.
    let anchor_depth = (scene_frame.depth() + silhouette_depth)
        .min(MAX_DEPTH as u8);
    let spawn_pos = WorldPos::from_frame_local(
        &scene_frame,
        [cam_x, cam_y, cam_z],
        anchor_depth,
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
