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
/// frame sits. Larger K = bigger packed subtree (more to render) but
/// more headroom before cell-scale drops below the local scale we
/// ship to the shader. 3 matches the spec's default.
pub const RENDER_FRAME_K: u8 = 3;

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
    pub(super) velocity: [f32; 3],
    pub(super) world: WorldState,
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
    pub(super) cs_planet: Option<crate::world::cubesphere::SphericalPlanet>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) frames_waited: u32,
}

impl App {
    pub fn new() -> Self {
        // Build the ENTIRE world — space tree AND spherical planet —
        // here, BEFORE the event loop starts.
        let mut world = crate::world::worldgen::generate_world();
        let tree_depth = world.tree_depth();

        let setup = crate::world::spherical_worldgen::demo_planet();
        let cs_planet = crate::world::spherical_worldgen::build(&mut world.library, &setup);
        eprintln!(
            "Spherical planet generated: 6 face subtrees, library now {} nodes",
            world.library.len(),
        );

        // Spawn just above the planet's north pole. `demo_planet`
        // centers the planet at the world origin so this naturally
        // lands inside `[0, WORLD_SIZE)^3`; assert to catch drift
        // if those tuning knobs change.
        let spawn_xyz = [
            setup.center[0],
            setup.center[1] + setup.outer_r + 0.3,
            setup.center[2],
        ];
        debug_assert!(
            spawn_xyz.iter().all(|&v| (0.0..WORLD_SIZE).contains(&v)),
            "spawn {:?} must be inside root [0, {})",
            spawn_xyz, WORLD_SIZE,
        );
        // Default anchor depth: `td - 6 + 1` reproduces the legacy
        // zoom feel (one level above the face subtree's per-block
        // cells at the planet's surface).
        let anchor_depth = ((tree_depth as i32 - 6 + 1).max(1) as u8).min(60);
        let position = WorldPos::from_world_xyz(spawn_xyz, anchor_depth);

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                position,
                smoothed_up: [0.0, 1.0, 0.0],
                yaw: 0.0,
                pitch: -0.3,
            },
            velocity: [0.0, 0.0, 0.0],
            world,
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
            cs_planet: Some(cs_planet),
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

    /// Render frame = camera's anchor truncated by `RENDER_FRAME_K`
    /// levels, further clamped so the walk from `world.root` lands
    /// on an actual `Node`. Returns `(path, node_id)` where
    /// `node_id` is the root of the subtree the shader will render
    /// and `path` is the path from `world.root` to that subtree.
    ///
    /// If the walk encounters a terminal (`Empty` or `Block`) before
    /// reaching the desired depth, the frame is truncated to the
    /// parent `Node` so the rendered subtree stays concrete — the
    /// shader can walk it without hitting a dangling path.
    pub(super) fn render_frame(&self) -> (Path, NodeId) {
        let desired_depth = self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8;
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

    #[inline]
    pub(super) fn zoom_level(&self) -> i32 {
        (self.tree_depth as i32) - (self.anchor_depth() as i32) + 1
    }

    pub(super) fn update(&mut self, dt: f32) {
        player::update(
            &mut self.camera,
            &mut self.velocity,
            &self.keys,
            self.cs_planet.as_ref(),
            &self.world.library,
            dt,
        );

        // Camera moved — push its new frame-local position to the
        // GPU so the next render consumes f32-safe coordinates.
        let (frame, _) = self.render_frame();
        let cam_local = self.camera.position.in_frame(&frame);
        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera_at(cam_local, 1.2));
        }
    }
}
