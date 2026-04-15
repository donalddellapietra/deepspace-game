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
use crate::world::tree::{Child, NodeId};

/// Levels shallower than the camera's anchor at which the render
/// frame sits. Currently capped at root by `RENDER_FRAME_MAX_DEPTH`
/// while the path-anchored shader pipeline is being validated end-
/// to-end; once the sphere DDA fully runs in body-cell-local
/// coordinates (which it now does), this can rise.
pub const RENDER_FRAME_K: u8 = 3;
pub const RENDER_FRAME_MAX_DEPTH: u8 = 0;

pub mod cursor;
pub mod edit_actions;
pub mod event_loop;
pub mod input_handlers;
#[cfg(not(target_arch = "wasm32"))]
pub mod overlay_integration;

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
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) frames_waited: u32,
}

impl App {
    pub fn new() -> Self {
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

        let anchor_depth = ((tree_depth as i32 - 6 + 1).max(1) as u8).min(60);
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
    /// Currently `RENDER_FRAME_MAX_DEPTH = 0` so this just returns
    /// `world.root` — kept as a method so the frame-aware shader
    /// path stays connected and the cap can lift later.
    pub(super) fn render_frame(&self) -> (Path, NodeId) {
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        let mut frame = self.camera.position.anchor;
        frame.truncate(desired_depth);
        let mut node_id = self.world.root;
        let mut reached = 0u8;
        for k in 0..frame.depth() as usize {
            let Some(node) = self.world.library.get(node_id) else { break };
            let slot = frame.slot(k) as usize;
            match node.children[slot] {
                Child::Node(child_id) => {
                    node_id = child_id;
                    reached = (k as u8) + 1;
                }
                Child::Block(_) | Child::Empty => break,
            }
        }
        frame.truncate(reached);
        (frame, node_id)
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
