//! Event-loop application struct and top-level frame driver.

use std::sync::Arc;

use winit::window::Window;

use crate::camera::Camera;
use crate::game_state::{GameUiState, SavedMeshes};
use crate::input::Keys;
use crate::player;
use crate::renderer::Renderer;
use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeId, NodeKind, MAX_DEPTH};

/// Levels shallower than the camera's anchor at which the render
/// frame sits. The frame walks down the camera's path until either
/// (a) it reaches `anchor_depth - RENDER_FRAME_K`, or (b) it would
/// cross into a non-Cartesian node — whichever happens first. The
/// non-Cartesian stop is required because the shader's main DDA
/// only knows how to march through Cartesian children at the frame
/// root (sphere body / face-cell roots are next-session work).
///
/// `RENDER_FRAME_MAX_DEPTH` was the historical pin at root (0) used
/// to validate the sphere dispatch; with the precision rewrite in
/// place it's now `MAX_DEPTH` so the walker can descend freely
/// through Cartesian zones.
pub const RENDER_FRAME_K: u8 = 3;
pub const RENDER_FRAME_MAX_DEPTH: u8 = MAX_DEPTH as u8;

pub mod cursor;
pub mod edit_actions;
pub mod event_loop;
pub mod frame;
pub mod input_handlers;
#[cfg(not(target_arch = "wasm32"))]
pub mod overlay_integration;
pub mod test_runner;

pub use frame::{aabb_world_to_frame, compute_render_frame, frame_origin_size_world};
pub use test_runner::TestConfig;

pub struct App {
    pub(super) window: Option<Arc<Window>>,
    pub(super) renderer: Option<Renderer>,
    pub(super) camera: Camera,
    pub(super) world: WorldState,
    pub(super) frozen: bool,
    pub(super) cursor_locked: bool,
    pub(super) keys: Keys,
    pub(super) last_frame: std::time::Instant,
    pub(super) tree_depth: u32,
    pub(super) palette: ColorRegistry,
    pub(super) saved_meshes: SavedMeshes,
    pub(super) save_mode: bool,
    pub(super) ui: GameUiState,
    pub(super) debug_overlay_visible: bool,
    pub(super) fps_smooth: f64,
    /// Path from `world.root` to the planet's body node. Used for
    /// spawn-position derivation and as a hint for future debug
    /// teleport / cursor logic; rendering reads the body via the
    /// normal tree walk + `NodeKind` dispatch.
    #[allow(dead_code)]
    pub(super) planet_path: Path,
    /// Headless test driver. Populated when CLI flags ask for
    /// scripted actions or screenshots. See `test_runner`.
    pub(super) test: Option<test_runner::TestRunner>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) frames_waited: u32,
}

impl App {
    pub fn new() -> Self {
        Self::with_test_config(TestConfig::default())
    }

    pub fn with_test_config(test_cfg: TestConfig) -> Self {
        // Build a Cartesian world tree, then insert the planet body
        // into its central depth-1 cell. After install, the planet
        // is a `NodeKind::CubedSphereBody` node living inside the
        // tree — there's no parallel `cs_planet` handle.
        let mut world = crate::world::worldgen::generate_world();
        let setup = crate::world::spherical_worldgen::demo_planet();
        let (new_root, planet_path) =
            crate::world::spherical_worldgen::install_at_root_center(
                &mut world.library, world.root, &setup,
            );
        world.swap_root(new_root);
        let tree_depth = world.tree_depth();
        eprintln!(
            "Planet inserted at path {:?}; library has {} nodes, depth {}",
            planet_path.as_slice(), world.library.len(), tree_depth,
        );

        // Spawn just above the planet's body cell. Body cell at
        // depth 1 slot 13 spans world `[1, 2)³`. Sphere outer_r
        // local = `setup.outer_r`, world = `outer_r * 1.0` (cell
        // size is 1 world unit at depth 1). Top of sphere at world
        // y = 1.5 + outer_r. Spawn slightly above.
        let body_top_y = 1.5 + setup.outer_r;
        let spawn_xyz = [1.5, (body_top_y + 0.05).min(WORLD_SIZE - 0.001), 1.5];
        debug_assert!(spawn_xyz.iter().all(|&v| (0.0..WORLD_SIZE).contains(&v)));

        let default_depth = ((tree_depth as i32 - 6 + 1).max(1) as u8).min(60);
        let anchor_depth = test_cfg.spawn_depth.unwrap_or(default_depth);
        let position = WorldPos::from_world_xyz(spawn_xyz, anchor_depth);

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                position,
                smoothed_up: [0.0, 1.0, 0.0],
                yaw: 0.0,
                // Steep pitch so the planet (below camera) fills
                // the lower half of the view at spawn.
                pitch: -1.2,
            },
            world,
            frozen: false,
            cursor_locked: false,
            keys: Keys::default(),
            last_frame: std::time::Instant::now(),
            tree_depth,
            palette: ColorRegistry::new(),
            saved_meshes: SavedMeshes::default(),
            save_mode: false,
            ui: GameUiState::new(),
            debug_overlay_visible: false,
            fps_smooth: 0.0,
            planet_path,
            test: test_runner::TestRunner::from_config(test_cfg),
            #[cfg(not(target_arch = "wasm32"))]
            webview: None,
            #[cfg(not(target_arch = "wasm32"))]
            frames_waited: 0,
        }
    }

    #[inline]
    pub(super) fn anchor_depth(&self) -> u32 {
        self.camera.position.anchor.depth() as u32
    }

    #[inline]
    pub(super) fn zoom_level(&self) -> i32 {
        (self.tree_depth as i32) - (self.anchor_depth() as i32) + 1
    }

    /// Walk the world tree to find the render-frame's NodeId.
    ///
    /// The frame can be **Cartesian** (shader runs the XYZ DDA
    /// from it) or **CubedSphereBody** (shader dispatches into
    /// sphere DDA at start-of-march, body fills the `[0, 3)³`
    /// frame). `CubedSphereFace` frames are out of scope this
    /// pass — the walker stops before entering a face cell.
    ///
    /// Cartesian descent is now safe because the GPU pack includes
    /// an ancestor ribbon back to the absolute world root, and
    /// the shader pops upward when rays exit the frame's [0, 3)³
    /// bubble. So content outside the frame stays visible: the
    /// planet at root.children[13] still renders even when the
    /// camera frame is deep in an empty Cartesian sibling subtree.
    pub(super) fn render_frame(&self) -> (Path, NodeId) {
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        frame::compute_render_frame(
            &self.world.library, self.world.root,
            &self.camera.position.anchor, desired_depth,
        )
    }

    /// `NodeKind` of the *intended* render-frame root from a tree
    /// walk (no buffer-truncation awareness). `upload_tree_lod`
    /// uses the effective frame instead — the one build_ribbon
    /// could actually reach. Kept for tests / debugging.
    #[allow(dead_code)]
    pub(super) fn render_frame_kind(&self) -> NodeKind {
        let (_, node_id) = self.render_frame();
        self.world.library.get(node_id)
            .map(|n| n.kind)
            .unwrap_or(NodeKind::Cartesian)
    }

    pub(super) fn update(&mut self, dt: f32) {
        player::update(&mut self.camera, dt);

        let (frame, _) = self.render_frame();
        let cam_local = self.camera.position.in_frame(&frame);
        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera_at(cam_local, 1.2));
        }
    }

    pub(super) fn step_chunk(&mut self, axis: usize, direction: i32) {
        if self.frozen { return; }
        self.camera.position.anchor.step_neighbor_cartesian(axis, direction);
        self.camera.position.offset = [0.5, 0.5, 0.5];
    }

    pub fn debug_teleport(&mut self, slots: &[u8], offset: [f32; 3]) {
        let mut anchor = Path::root();
        for &s in slots.iter().take(crate::world::tree::MAX_DEPTH) {
            anchor.push(s);
        }
        self.camera.position = WorldPos::new(anchor, offset);
        self.apply_zoom();
    }

    pub(super) fn log_location(&self) {
        let p = &self.camera.position;
        log::info!(
            "camera anchor depth={} slots={:?} offset={:?} world={:?}",
            p.anchor.depth(),
            p.anchor.as_slice(),
            p.offset,
            p.to_world_xyz(),
        );
    }
}

