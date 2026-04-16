//! Event-loop application struct and top-level frame driver.

use std::sync::Arc;

use winit::window::Window;

use crate::camera::Camera;
use crate::game_state::{GameUiState, SavedMeshes};
use crate::input::Keys;
use crate::player;
use crate::renderer::Renderer;
use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};
use crate::world::bootstrap;
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeKind, MAX_DEPTH};

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
pub const RENDER_FRAME_CONTEXT: u8 = 4;
pub mod cursor;
pub mod edit_actions;
pub mod event_loop;
pub mod frame;
pub mod input_handlers;
#[cfg(not(target_arch = "wasm32"))]
pub mod overlay_integration;
pub mod test_runner;

pub use frame::{
    compute_render_frame, frame_origin_size_world, with_render_margin, ActiveFrame,
    ActiveFrameKind,
};
pub use test_runner::TestConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LodUploadKey {
    pub root: u64,
    pub camera_anchor: Path,
    pub camera_offset_bits: [u32; 3],
    pub render_path: Path,
    pub logical_path: Path,
    pub kind_tag: u8,
}

impl LodUploadKey {
    pub(super) fn new(root: u64, camera: &WorldPos, frame: &ActiveFrame) -> Self {
        let kind_tag = match frame.kind {
            ActiveFrameKind::Cartesian => 0,
            ActiveFrameKind::Body { .. } => 1,
            ActiveFrameKind::Sphere(_) => 2,
        };
        Self {
            root,
            camera_anchor: camera.anchor,
            camera_offset_bits: camera.offset.map(f32::to_bits),
            render_path: frame.render_path,
            logical_path: frame.logical_path,
            kind_tag,
        }
    }
}

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
    pub(super) startup_profile_frames: u32,
    /// Path from `world.root` to the planet's body node. Used for
    /// spawn-position derivation and as a hint for future debug
    /// teleport / cursor logic; rendering reads the body via the
    /// normal tree walk + `NodeKind` dispatch.
    #[allow(dead_code)]
    pub(super) planet_path: Option<Path>,
    /// The actual frame the renderer is using right now. This may
    /// be shallower than `render_frame()` when GPU packing flattened
    /// a slot on the intended path and `build_ribbon` had to stop
    /// early.
    pub(super) active_frame: ActiveFrame,
    /// Headless test driver. Populated when CLI flags ask for
    /// scripted actions or screenshots. See `test_runner`.
    pub(super) test: Option<test_runner::TestRunner>,
    /// Last world/frame/camera tuple that required a full GPU tree
    /// repack + ribbon rebuild. If unchanged, we can keep the same
    /// tree/node-kind/ribbon buffers and just refresh camera/uniforms.
    pub(super) last_lod_upload_key: Option<LodUploadKey>,
    /// Deterministic renderer harness mode from the old deep-layers
    /// branch. Bypasses the native overlay/event-loop path so we can
    /// isolate renderer regressions from WKWebView issues.
    pub(super) render_harness: bool,
    /// Test/harness runs should measure renderer cost, not native
    /// vsync pacing. Interactive runs keep the default present mode.
    pub(super) low_latency_present: bool,
    pub(super) show_harness_window: bool,
    pub(super) disable_overlay: bool,
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
        let render_harness = test_cfg.render_harness;
        let low_latency_present = test_cfg.is_active();
        let show_harness_window = test_cfg.show_window;
        let disable_overlay = test_cfg.disable_overlay;
        let bootstrap = bootstrap::bootstrap_world(test_cfg.world_preset, Some(test_cfg.plain_layers()));
        let world = bootstrap.world;
        let tree_depth = world.tree_depth();
        let spawn_xyz = test_cfg.spawn_xyz.unwrap_or(bootstrap.default_spawn_xyz);
        debug_assert!(spawn_xyz.iter().all(|&v| (0.0..WORLD_SIZE).contains(&v)));

        let anchor_depth = test_cfg.spawn_depth.unwrap_or(bootstrap.default_spawn_depth);
        let position = WorldPos::from_world_xyz(spawn_xyz, anchor_depth);
        let desired_depth = (position.anchor.depth().saturating_sub(RENDER_FRAME_K))
            .min(RENDER_FRAME_MAX_DEPTH);
        let mut logical_path = position.anchor;
        logical_path.truncate(desired_depth);
        let active_frame = frame::with_render_margin(
            &world.library, world.root, &logical_path, RENDER_FRAME_CONTEXT,
        );
        eprintln!(
            "startup_perf initial_frame kind={:?} render_depth={} logical_depth={} desired_depth={} anchor_depth={}",
            active_frame.kind,
            active_frame.render_path.depth(),
            active_frame.logical_path.depth(),
            desired_depth,
            position.anchor.depth(),
        );
        let spawn_yaw = test_cfg.spawn_yaw.unwrap_or(bootstrap.default_spawn_yaw);
        let spawn_pitch = test_cfg.spawn_pitch.unwrap_or(bootstrap.default_spawn_pitch);
        eprintln!(
            "spawn: xyz={:?} anchor_depth={} yaw={} pitch={}",
            spawn_xyz, anchor_depth, spawn_yaw, spawn_pitch,
        );

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                position,
                smoothed_up: [0.0, 1.0, 0.0],
                yaw: spawn_yaw,
                // Steep pitch so the planet (below camera) fills
                // the lower half of the view at spawn. Override
                // via --spawn-pitch for screenshot-driven debug.
                pitch: spawn_pitch,
            },
            world,
            frozen: false,
            cursor_locked: test_cfg.spawn_depth.is_some() || test_cfg.screenshot.is_some(),
            keys: Keys::default(),
            last_frame: std::time::Instant::now(),
            tree_depth,
            palette: ColorRegistry::new(),
            saved_meshes: SavedMeshes::default(),
            save_mode: false,
            ui: GameUiState::new(),
            debug_overlay_visible: false,
            fps_smooth: 0.0,
            startup_profile_frames: 0,
            planet_path: bootstrap.planet_path,
            active_frame,
            test: test_runner::TestRunner::from_config(test_cfg),
            last_lod_upload_key: None,
            render_harness,
            low_latency_present,
            show_harness_window,
            disable_overlay,
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

    /// Resolve the active frame for the current zoom. Cartesian
    /// regions use a linear render root. Sphere regions keep the
    /// linear root at the containing body cell and carry an
    /// explicit face-cell window so render/edit share one layer
    /// definition.
    pub(super) fn render_frame(&self) -> ActiveFrame {
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        let mut logical_path = self.camera.position.anchor;
        logical_path.truncate(desired_depth);
        frame::with_render_margin(
            &self.world.library, self.world.root,
            &logical_path, RENDER_FRAME_CONTEXT,
        )
    }

    /// `NodeKind` of the *intended* render-frame root from a tree
    /// walk (no buffer-truncation awareness). `upload_tree_lod`
    /// uses the effective frame instead — the one build_ribbon
    /// could actually reach. Kept for tests / debugging.
    #[allow(dead_code)]
    pub(super) fn render_frame_kind(&self) -> NodeKind {
        match self.render_frame().kind {
            ActiveFrameKind::Cartesian => NodeKind::Cartesian,
            ActiveFrameKind::Body { inner_r, outer_r } => {
                NodeKind::CubedSphereBody { inner_r, outer_r }
            }
            ActiveFrameKind::Sphere(s) => NodeKind::CubedSphereFace { face: s.face },
        }
    }

    pub(super) fn update(&mut self, dt: f32) {
        player::update(&mut self.camera, dt);
        let cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&cam_gpu);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub(super) fn overlay_enabled(&self) -> bool {
        !self.render_harness && !self.disable_overlay
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
            "camera anchor depth={} slots={:?} offset={:?}",
            p.anchor.depth(),
            p.anchor.as_slice(),
            p.offset,
        );
    }

    pub(super) fn gpu_camera_for_frame(&self, frame: &ActiveFrame) -> crate::world::gpu::GpuCamera {
        let (cam_local, frame_scale) = match frame.kind {
            ActiveFrameKind::Sphere(sphere) => {
                let cam_local = self.camera.position.in_frame(&sphere.body_path);
                let (_origin, frame_size_world) = frame::frame_origin_size_world(&sphere.body_path);
                let frame_scale = WORLD_SIZE / frame_size_world.max(f32::MIN_POSITIVE);
                (cam_local, frame_scale)
            }
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                let cam_local = self.camera.position.in_frame(&frame.render_path);
                let (_origin, frame_size_world) = frame::frame_origin_size_world(&frame.render_path);
                let frame_scale = WORLD_SIZE / frame_size_world.max(f32::MIN_POSITIVE);
                (cam_local, frame_scale)
            }
        };
        if self.startup_profile_frames < 4 {
            eprintln!(
                "gpu_camera frame_kind={:?} render_path={:?} logical_path={:?} cam_local={:?}",
                frame.kind,
                frame.render_path.as_slice(),
                frame.logical_path.as_slice(),
                cam_local,
            );
        }
        let (fwd_world, right_world, up_world) = self.camera.basis();
        let fwd_local = crate::world::sdf::scale(crate::world::sdf::normalize(fwd_world), frame_scale);
        let right_local = crate::world::sdf::scale(crate::world::sdf::normalize(right_world), frame_scale);
        let up_local = crate::world::sdf::scale(crate::world::sdf::normalize(up_world), frame_scale);
        if self.startup_profile_frames < 4 {
            eprintln!(
                "gpu_camera basis world_fwd={:?} local_fwd={:?} local_right={:?} local_up={:?} frame_scale={}",
                fwd_world,
                fwd_local,
                right_local,
                up_local,
                frame_scale,
            );
        }
        self.camera.gpu_camera_with_basis(
            cam_local,
            fwd_local,
            right_local,
            up_local,
            1.2,
        )
    }
}
