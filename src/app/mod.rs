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
pub mod harness_emit;
pub mod input_handlers;
#[cfg(not(target_arch = "wasm32"))]
pub mod overlay_integration;
pub mod test_runner;

pub use frame::{
    compute_render_frame, with_render_margin, ActiveFrame,
    ActiveFrameKind,
};
pub use test_runner::TestConfig;

/// Cross-thread / cross-task signal back into the winit event loop.
/// The WASM path can't synchronously block on the async wgpu init, so
/// it spawns the future and posts the finished `Renderer` back via
/// this enum (native uses the same channel for symmetry). On WASM,
/// browser-window resize is also routed through here so the canvas
/// backing-store update and the renderer.resize stay in lockstep.
pub enum UserEvent {
    RendererReady(Box<Renderer>),
    /// Browser window resized. WASM-only: only sent from the
    /// `wasm_canvas_setup` resize closure. Native uses
    /// `WindowEvent::Resized` which already routes to both the
    /// renderer and the wry overlay — sending `UserEvent::Resize`
    /// natively would resize the renderer but skip the overlay,
    /// which is why the variant is gated.
    #[cfg(target_arch = "wasm32")]
    Resize(winit::dpi::PhysicalSize<u32>),
}

/// Pack is a pure function of `(library, root)`. The only field that
/// invalidates the GPU tree buffer is `root` — edits change it; pure
/// motion does not. Render path, visual depth, camera, etc. live in
/// per-frame uniforms (cheap to rewrite every frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LodUploadKey {
    pub root: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HighlightCacheKey {
    pub lod: LodUploadKey,
    pub render_path: Path,
    pub logical_path: Path,
    pub visual_depth: u8,
    pub yaw_bits: u32,
    pub pitch_bits: u32,
    pub cursor_locked: bool,
    pub epoch: u64,
}

impl HighlightCacheKey {
    pub(super) fn new(app: &App) -> Self {
        Self {
            lod: LodUploadKey::new(app.world.root),
            render_path: app.active_frame.render_path,
            logical_path: app.active_frame.logical_path,
            visual_depth: app.visual_depth().min(u8::MAX as u32) as u8,
            yaw_bits: app.camera.yaw.to_bits(),
            pitch_bits: app.camera.pitch.to_bits(),
            cursor_locked: app.cursor_locked,
            epoch: app.highlight_epoch,
        }
    }
}

impl LodUploadKey {
    pub(super) fn new(root: u64) -> Self {
        Self { root }
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
    pub(super) last_frame: web_time::Instant,
    pub(super) tree_depth: u32,
    pub(super) palette: ColorRegistry,
    pub(super) saved_meshes: SavedMeshes,
    pub(super) save_mode: bool,
    pub(super) ui: GameUiState,
    pub(super) debug_overlay_visible: bool,
    pub(super) fps_smooth: f64,
    pub(super) startup_profile_frames: u32,
    /// Path from `world.root` to the planet's body node. Used for
    /// spawn-position derivation and for camera-local sphere focus
    /// (`edit_actions::zoom::camera_local_sphere_focus_path`).
    pub(super) planet_path: Option<Path>,
    /// The actual frame the renderer is using right now. This may
    /// be shallower than `render_frame()` when GPU packing flattened
    /// a slot on the intended path and `build_ribbon` had to stop
    /// early.
    pub(super) active_frame: ActiveFrame,
    /// Headless test driver. Populated when CLI flags ask for
    /// scripted actions or screenshots. See `test_runner`.
    pub(super) test: Option<test_runner::TestRunner>,
    /// Last world root that triggered a full pack + GPU upload. If
    /// unchanged, the tree buffer is reusable — only the per-frame
    /// ribbon + camera uniforms rebuild.
    pub(super) last_lod_upload_key: Option<LodUploadKey>,
    /// Packed GPU tree state. Updated by `CachedTree::update_root`
    /// on edits — same function handles initial pack and edits.
    pub(super) cached_tree: Option<crate::world::gpu::CachedTree>,
    /// Deterministic renderer harness mode from the old deep-layers
    /// branch. Bypasses the native overlay/event-loop path so we can
    /// isolate renderer regressions from WKWebView issues.
    pub(super) render_harness: bool,
    /// Test/harness runs should measure renderer cost, not native
    /// vsync pacing. Interactive runs keep the default present mode.
    pub(super) low_latency_present: bool,
    pub(super) show_harness_window: bool,
    pub(super) disable_overlay: bool,
    pub(super) disable_highlight: bool,
    pub(super) forced_visual_depth: Option<u32>,
    pub(super) forced_edit_depth: Option<u32>,
    pub(super) harness_width: u32,
    pub(super) harness_height: u32,
    /// Whether to enable shader-side ray-step atomic counters in the
    /// fragment shader. Mirrors `TestConfig::shader_stats`; threaded
    /// into `Renderer::new` so the live event-loop `render()` path
    /// can emit per-frame shader stats when the `--shader-stats`
    /// flag is set.
    pub(super) shader_stats_enabled: bool,
    /// Nyquist LOD pixel threshold for the Cartesian shader.
    /// Threaded to `Renderer::new` where it's baked into the
    /// pipeline as a WGSL `override` constant.
    pub(super) lod_pixel_threshold: f32,
    /// Ribbon-level base detail budget. See bindings.wgsl
    /// `BASE_DETAIL_DEPTH`.
    pub(super) lod_base_depth: u32,
    /// When > 0, the live-surface `render()` path emits a
    /// `render_live_sample` line every N frames with CPU-side phase
    /// timings (acquire / encode / submit / present / total). 0
    /// disables. Set via `--live-sample-every N`.
    pub(super) live_sample_every_frames: u32,
    /// Whether TAAU is enabled. Threaded to `Renderer::new` so the
    /// live and harness paths share the same resolve setup.
    pub(super) taa_enabled: bool,
    /// How entities render — ray-march through tag=3 (default) or
    /// instanced raster (scales to 100k+). Set from CLI
    /// `--entity-render` and baked into Renderer::new.
    pub(super) entity_render_mode: crate::renderer::EntityRenderMode,
    /// World-coordinate Y where entities naturally rest. `Some` for
    /// flat worlds (sea level = a specific Y); `None` for sphere/
    /// fractal worlds where "resting height" is position-dependent.
    /// Entities snap to this Y at spawn and during motion, so a
    /// soldier walking around a flat world stays on the grass
    /// instead of drifting up through the sky over time.
    pub(super) entity_surface_y: Option<f32>,
    /// Block-interaction radius in anchor-cell units. Caps the
    /// cursor raycast distance so break/place only succeed when
    /// the target is within `interaction_radius × anchor_cell_size`
    /// of the camera. Mirrors the LOD shells: zoom-aware, no
    /// absolute-coordinate math.
    pub(super) interaction_radius_cells: u32,
    pub(super) last_highlight_raycast_ms: f64,
    pub(super) last_highlight_set_ms: f64,
    /// Cost of packing the tree into GPU-friendly form. Set by
    /// `upload_tree_lod`. `0.0` when the LOD key was reused.
    pub(super) last_pack_ms: f64,
    /// Cost of building the ancestor ribbon from the packed tree.
    /// Set by `upload_tree_lod`. `0.0` when the LOD key was reused.
    pub(super) last_ribbon_build_ms: f64,
    /// Number of packed nodes in the most recent packed tree (i.e.
    /// `nodes.len()` in the sparse layout). Static signal: tells the
    /// harness how much data the shader had to walk.
    pub(super) last_packed_node_count: u32,
    /// Length of the ancestor ribbon pushed to the GPU.
    pub(super) last_ribbon_len: u32,
    /// Effective `visual_depth` the renderer was run with. Diverges
    /// from anchor depth when LOD/force flags clamp the budget.
    pub(super) last_effective_visual_depth: u32,
    /// Whether `upload_tree_lod` reused a previously packed tree
    /// instead of repacking. When true, pack/ribbon_build are 0.
    pub(super) last_reused_gpu_tree: bool,
    pub(super) highlight_epoch: u64,
    pub(super) cached_highlight: Option<(HighlightCacheKey, Option<([f32; 3], [f32; 3])>)>,
    /// Last crosshair reticle state pushed to the overlay. Used by
    /// `update_highlight` to diff-push `CrosshairStateJs`: the
    /// overlay only receives an IPC message when the bit actually
    /// flips, not every frame. `None` = we've never pushed.
    pub(super) last_crosshair_sent: Option<crate::bridge::CrosshairStateJs>,
    /// Slot path from the last successful break/place edit. Used as a
    /// preserve_path during the next GPU pack so the packer keeps
    /// fine detail along the edit path visible, even when the camera
    /// is far enough from the surface that LOD would normally collapse it.
    pub(super) last_edit_slots: Option<Path>,
    /// All live entities. Flat Vec, no ECS — the old npc-instancing
    /// branch's 40× perf-over-ECS lesson. Visual content shared via
    /// `NodeLibrary`; position + override state per-entity.
    pub(super) entities: crate::world::entities::EntityStore,
    /// Scene root NodeId from the last `upload_tree_lod`. Held via
    /// `NodeLibrary::ref_inc` so its ephemeral ancestor chain
    /// survives between frames; released on the next upload when the
    /// new scene root replaces it.
    pub(super) active_scene_root: Option<crate::world::tree::NodeId>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) frames_waited: u32,
    /// Loopback into the winit event loop. Required on WASM so the
    /// spawned async renderer-init future can deliver the finished
    /// `Renderer` back via `UserEvent::RendererReady`, and so the
    /// browser-window resize closure can post `UserEvent::Resize`.
    /// Native keeps it for symmetry but doesn't currently send.
    pub(super) proxy: winit::event_loop::EventLoopProxy<UserEvent>,
    /// True after we kicked off the async renderer init (WASM only).
    /// Stops `ensure_started` from re-spawning the future on every
    /// `resumed` / `about_to_wait` callback before the renderer
    /// actually arrives.
    pub(super) renderer_init_started: bool,
    /// Captured args we need in `finish_init`, populated by
    /// `start_init` before the renderer comes online.
    pub(super) pending_init: Option<PendingInit>,
}

/// Args that `start_init` computes synchronously and `finish_init`
/// consumes once the renderer is ready.
pub(super) struct PendingInit {
    pub(super) source: String,
    pub(super) resumed_start: web_time::Instant,
    pub(super) window_elapsed: web_time::Duration,
    pub(super) prepare_elapsed: web_time::Duration,
    pub(super) pack_elapsed: web_time::Duration,
    pub(super) renderer_start: web_time::Instant,
    pub(super) node_count: usize,
    pub(super) tree_u32_count: usize,
}

impl App {
    pub fn new(proxy: winit::event_loop::EventLoopProxy<UserEvent>) -> Self {
        Self::with_test_config(TestConfig::default(), proxy)
    }

    pub fn with_test_config(
        test_cfg: TestConfig,
        proxy: winit::event_loop::EventLoopProxy<UserEvent>,
    ) -> Self {
        let render_harness = test_cfg.render_harness;
        let low_latency_present = test_cfg.is_active();
        let show_harness_window = test_cfg.show_window;
        let disable_overlay = test_cfg.disable_overlay;
        let disable_highlight = test_cfg.disable_highlight;
        let forced_visual_depth = test_cfg.force_visual_depth;
        let forced_edit_depth = test_cfg.force_edit_depth;
        let shader_stats_enabled = test_cfg.shader_stats;
        // Nyquist floor: sub-pixel rejection only. Primary LOD gate
        // is ribbon-level-based (`lod_base_depth`).
        let lod_pixel_threshold = test_cfg.lod_pixels.unwrap_or(1.0);
        let lod_base_depth = test_cfg.lod_base_depth.unwrap_or(8);
        let live_sample_every_frames = test_cfg.live_sample_every_frames.unwrap_or(0);
        let taa_enabled = test_cfg.taa;
        let entity_render_mode = test_cfg.entity_render_mode;
        let entity_surface_y = bootstrap::surface_y_for_preset(&test_cfg.world_preset);
        let interaction_radius_cells = test_cfg.interaction_radius.unwrap_or(6);
        let (harness_width, harness_height) = test_cfg.harness_size();
        let bootstrap = bootstrap::bootstrap_world(test_cfg.world_preset.clone(), Some(test_cfg.plain_layers()));
        let mut world = bootstrap.world;
        let bootstrap_color_registry = bootstrap.color_registry;
        let tree_depth = world.tree_depth();
        // Resolve spawn position: CLI --spawn-xyz overrides with
        // root-frame-local coords (precise at shallow depth, then
        // deepened via slot arithmetic); otherwise use bootstrap's
        // pre-built WorldPos.
        let position = match test_cfg.spawn_xyz {
            Some(xyz) => {
                debug_assert!(xyz.iter().all(|&v| (0.0..WORLD_SIZE).contains(&v)));
                let depth = test_cfg.spawn_depth.unwrap_or(
                    bootstrap.default_spawn_pos.anchor.depth(),
                );
                WorldPos::from_frame_local(&Path::root(), xyz, depth.min(12))
                    .deepened_to(depth)
            }
            None => {
                if let Some(depth) = test_cfg.spawn_depth {
                    if bootstrap.plain_layers > 0 {
                        // Surface-tracking spawn: follow the ground through
                        // the tree at the target depth instead of blindly
                        // deepening (which drifts away from the surface).
                        let pos = bootstrap::plain_surface_spawn(depth);
                        bootstrap::carve_air_pocket(&mut world, &pos.anchor, bootstrap.plain_layers);
                        pos
                    } else {
                        let mut pos = bootstrap.default_spawn_pos;
                        while pos.anchor.depth() > depth { pos.zoom_out(); }
                        pos = pos.deepened_to(depth);
                        pos
                    }
                } else {
                    // Carve the default spawn for plain worlds so the
                    // camera starts in air, not embedded in a block.
                    if bootstrap.plain_layers > 0 {
                        bootstrap::carve_air_pocket(&mut world, &bootstrap.default_spawn_pos.anchor, bootstrap.plain_layers);
                    }
                    bootstrap.default_spawn_pos
                }
            }
        };
        let anchor_depth = position.anchor.depth();
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
            "spawn: anchor_depth={} slots={:?} offset={:?} yaw={} pitch={}",
            anchor_depth, position.anchor.as_slice(), position.offset,
            spawn_yaw, spawn_pitch,
        );

        // Pull out fields that are used AFTER `from_config(test_cfg)`
        // consumes the config, since that call moves the struct.
        let spawn_entity_path = test_cfg.spawn_entity.clone();
        let spawn_entity_count = test_cfg.spawn_entity_count;

        let mut app = Self {
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
            last_frame: web_time::Instant::now(),
            tree_depth,
            palette: bootstrap_color_registry,
            saved_meshes: SavedMeshes::default(),
            save_mode: false,
            ui: GameUiState::new(),
            debug_overlay_visible: false,
            fps_smooth: 0.0,
            startup_profile_frames: if test_cfg.suppress_startup_logs { u32::MAX } else { 0 },
            planet_path: bootstrap.planet_path,
            active_frame,
            test: test_runner::TestRunner::from_config(test_cfg),
            last_lod_upload_key: None,
            cached_tree: None,
            render_harness,
            low_latency_present,
            show_harness_window,
            disable_overlay,
            disable_highlight,
            forced_visual_depth,
            forced_edit_depth,
            harness_width,
            harness_height,
            shader_stats_enabled,
            lod_pixel_threshold,
            lod_base_depth,
            live_sample_every_frames,
            taa_enabled,
            entity_render_mode,
            entity_surface_y,
            interaction_radius_cells,
            last_highlight_raycast_ms: 0.0,
            last_highlight_set_ms: 0.0,
            last_pack_ms: 0.0,
            last_ribbon_build_ms: 0.0,
            last_packed_node_count: 0,
            last_ribbon_len: 0,
            last_effective_visual_depth: 0,
            last_reused_gpu_tree: false,
            highlight_epoch: 0,
            last_crosshair_sent: None,
            cached_highlight: None,
            last_edit_slots: None,
            entities: crate::world::entities::EntityStore::new(),
            active_scene_root: None,
            #[cfg(not(target_arch = "wasm32"))]
            webview: None,
            #[cfg(not(target_arch = "wasm32"))]
            frames_waited: 0,
            proxy,
            renderer_init_started: false,
            pending_init: None,
        };
        if let Some(ref path) = spawn_entity_path {
            let count = spawn_entity_count.max(1);
            app.spawn_vox_entity_at_init(path, count);
        }
        app
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
        // Advance entities by velocity * dt. WorldPos renormalizes
        // on each entity, so cell-boundary crossings are handled
        // without the bbox computation seeing discontinuities.
        if !self.entities.is_empty() {
            self.entities.tick(&self.world.library, dt, self.entity_surface_y);
        }
        let cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        if let Some(renderer) = &mut self.renderer {
            renderer.update_camera(&cam_gpu);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub(super) fn overlay_enabled(&self) -> bool {
        !self.render_harness && !self.disable_overlay
    }

    /// True when the per-frame overlay state push + UI command poll
    /// should run. Native: gated on `overlay_enabled()` so render
    /// harness runs skip wry plumbing. WASM: always on — the React
    /// bridge has no harness mode.
    #[inline]
    pub(super) fn overlay_active(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        { self.overlay_enabled() }
        #[cfg(target_arch = "wasm32")]
        { true }
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
        let cam_local = match frame.kind {
            ActiveFrameKind::Sphere(sphere) => self.camera.position.in_frame(&sphere.body_path),
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                self.camera.position.in_frame(&frame.render_path)
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
        let fwd_local = crate::world::sdf::normalize(fwd_world);
        let right_local = crate::world::sdf::normalize(right_world);
        let up_local = crate::world::sdf::normalize(up_world);
        if self.startup_profile_frames < 4 {
            eprintln!(
                "gpu_camera basis world_fwd={:?} local_fwd={:?} local_right={:?} local_up={:?}",
                fwd_world,
                fwd_local,
                right_local,
                up_local,
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
