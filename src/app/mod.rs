//! Event-loop application struct and top-level frame driver.
//!
//! The [`App`] struct owns every piece of engine state and is the
//! `ApplicationHandler` that winit drives. Its surface area is
//! deliberately split across focused submodules:
//!
//! - [`cursor`] — cursor-lock / panel-sync state.
//! - [`overlay_integration`] — wry webview plumbing (native only).
//! - [`input_handlers`] — keyboard and mouse dispatch.
//! - [`edit_actions`] — break / place / highlight / zoom / GPU upload.
//! - [`event_loop`] — the `ApplicationHandler` implementation.
//!
//! `mod.rs` itself holds the struct definition, construction, and
//! the small per-frame `update()` dispatch. Anything larger than a
//! few lines belongs in one of the submodules above.

use std::sync::Arc;

use winit::window::Window;

use crate::camera::Camera;
use crate::game_state::{GameUiState, SavedMeshes};
use crate::input::Keys;
use crate::player;
use crate::renderer::Renderer;
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;

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
    /// Persistent velocity, integrated from gravity and damped each
    /// frame. Flight thrust is applied as direct per-frame
    /// displacement rather than accumulating here, so releasing WASD
    /// stops horizontal motion immediately while gravity keeps
    /// pulling.
    pub(super) velocity: [f32; 3],
    pub(super) world: WorldState,
    pub(super) cursor_locked: bool,
    pub(super) keys: Keys,
    pub(super) last_frame: std::time::Instant,
    /// Cached tree depth (recomputed only after edits). Used to clamp
    /// anchor depth so the player can't zoom past the tree's finest
    /// instantiated layer.
    pub(super) tree_depth: u32,
    pub(super) palette: ColorRegistry,
    pub(super) saved_meshes: SavedMeshes,
    pub(super) save_mode: bool,
    pub(super) ui: GameUiState,
    pub(super) debug_overlay_visible: bool,
    /// Exponentially smoothed FPS for the debug overlay.
    pub(super) fps_smooth: f64,
    /// Cubed-sphere demo planet — owns per-cell block storage and
    /// the shell geometry used for cursor raycasts and editing.
    pub(super) cs_planet: Option<crate::world::cubesphere::SphericalPlanet>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) frames_waited: u32,
}

impl App {
    pub fn new() -> Self {
        // Build the ENTIRE world — space tree AND spherical planet —
        // here, BEFORE the event loop starts. `resumed()` is called
        // synchronously by AppKit during its window-activation
        // handshake on macOS; if worldgen runs inside it, AppKit
        // never finishes activation and the NSWindow comes up non-
        // key (title bar grayed out, clicks ignored). Doing all
        // heavy work before `run_app` keeps `resumed()` a fast
        // "create window + hand pre-built data to GPU" path.
        let mut world = crate::world::worldgen::generate_world();
        let tree_depth = world.tree_depth();

        let setup = crate::world::spherical_worldgen::demo_planet();
        let cs_planet = crate::world::spherical_worldgen::build(&mut world.library, &setup);
        eprintln!(
            "Spherical planet generated: 6 face subtrees, library now {} nodes",
            world.library.len(),
        );

        // Spawn above the cubed-sphere planet's north pole.
        let spawn_pos = [setup.center[0], setup.center[1] + setup.outer_r + 0.3, setup.center[2]];

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                position: crate::world::coords::world_pos_from_f32(spawn_pos, 4),
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

    /// Per-frame update: advance camera physics, then push the new
    /// camera state to the GPU. Editing, highlights, and tree upload
    /// live in [`edit_actions`]; the `ApplicationHandler` in
    /// [`event_loop`] calls them in order.
    pub(super) fn update(&mut self, dt: f32) {
        // World-space width of one cell at the camera's current
        // anchor depth: `ROOT_EXTENT / 3^depth`. Feeds player gravity
        // and flight-thrust tuning so motion feels proportional to
        // the layer the player is operating at.
        let depth = self.camera.position.anchor.depth() as i32;
        let cell_size = crate::world::coords::ROOT_EXTENT / 3.0f32.powi(depth);

        player::update(
            &mut self.camera,
            &mut self.velocity,
            &self.keys,
            cell_size,
            self.cs_planet.as_ref(),
            &self.world.library,
            self.world.root,
            dt,
        );

        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera(1.2));
        }
    }
}
