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
    /// Debug freeze: when true, all teleport keys are ignored. Toggle
    /// with F. Provides a way to lock the camera in place while
    /// inspecting a frame.
    pub(super) debug_frozen: bool,
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
    /// Anchor path from `world.root` to the sphere body node. Used
    /// by gravity and render math to derive the body's world-space
    /// center without a separate `center: [f32; 3]` field.
    pub(super) body_anchor: crate::world::coords::Path,
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
        let setup = crate::world::spherical_worldgen::demo_planet();
        let scene = crate::world::spherical_worldgen::build(&setup);
        let tree_depth = scene.world.tree_depth();
        eprintln!(
            "Generated scene: body anchored at {:?}, tree_depth {}, library {} nodes",
            scene.body_anchor, tree_depth, scene.world.library.len(),
        );

        // Body's world-space center: the anchor cell's center.
        // Spawn well clear of the body so the player can survey the
        // whole planet without being inside the influence radius —
        // 1.0 world units away on the +Z side, clamped into the
        // root's `[0, 3)` extent.
        let body_center = crate::world::coords::world_pos_to_f32(
            &crate::world::coords::WorldPos { anchor: scene.body_anchor, offset: [0.5, 0.5, 0.5] },
        );
        // Camera default look direction is `-Z` (with a small -0.3
        // pitch, so it tilts down). Spawn 1.0 world units south of
        // the body on the `+Z` side so that "forward" looks straight
        // at the planet, just outside its influence radius.
        let spawn_pos = [
            body_center[0],
            body_center[1],
            (body_center[2] + 1.0).min(2.95),
        ];

        let world = scene.world;
        let cs_planet = scene.planet;
        let body_anchor = scene.body_anchor;

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
            debug_frozen: false,
            last_frame: std::time::Instant::now(),
            tree_depth,
            palette: ColorRegistry::new(),
            saved_meshes: SavedMeshes::default(),
            save_mode: false,
            ui: GameUiState::new(),
            debug_overlay_visible: false,
            fps_smooth: 0.0,
            cs_planet: Some(cs_planet),
            body_anchor,
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
        // Debug mode: no physics. Up-vector smoothing only.
        player::update(&mut self.camera, &mut self.velocity, dt);

        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera(1.2));
        }
    }
}
