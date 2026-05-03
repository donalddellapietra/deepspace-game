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
use crate::world::tree::MAX_DEPTH;

/// `render_margin` passed to `with_render_margin`. For Cartesian
/// frames `min_render_depth = logical.depth()` so this constant is
/// dormant — render_path = logical_path regardless of K. It still
/// controls the spread between logical and render paths on Sphere
/// and Body frames, where the logical path continues through the
/// face subtree but the render walker stays at the containing body
/// cell. The `= 3` value is a historical pin kept for sphere path
/// stability.
pub const RENDER_FRAME_K: u8 = 3;
pub const RENDER_FRAME_MAX_DEPTH: u8 = MAX_DEPTH as u8;
pub const RENDER_FRAME_CONTEXT: u8 = 4;

/// Depth at which the render frame is rooted, *independent of
/// the user's zoom level*. The camera's `WorldPos` is deepened to
/// this path depth via `deepened_to` — using the stored f32 offset
/// to pick slots at every level beyond the user's anchor — and
/// that deep path drives `compute_render_frame`.
///
/// # Why this exists
///
/// Zooming is a UI/interaction concept: it controls `edit_depth`
/// (where break/place operations resolve), not what the camera
/// actually sees. Two cameras at the same world position should
/// render identical pixels regardless of their zoom state. Before
/// this constant existed, `desired_depth = anchor_depth - K`
/// leaked zoom directly into render-frame depth — so a fractal
/// looked like chunky cubes when zoomed out and fine mesh when
/// zoomed in, even though the camera hadn't moved.
///
/// # Bound
///
/// Disabled (= MAX_DEPTH) to let the render frame follow the
/// actual camera anchor. The prior 14 cap was an f32-precision
/// workaround for the shader's `ray_dir /= 3` per-pop scheme.
pub const RENDER_ANCHOR_DEPTH: u8 = MAX_DEPTH as u8;
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
    /// Monotonic counter incremented each time the user requests a
    /// debug-overlay-state copy via `[`. Forwarded to the UI which
    /// watches for changes and writes the formatted overlay text to
    /// the clipboard.
    pub(super) debug_copy_seq: u64,
    pub(super) fps_smooth: f64,
    pub(super) startup_profile_frames: u32,
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
    /// pipeline as a WGSL `override` constant. This is the sole
    /// visual LOD gate now that the old ribbon-shell descent
    /// budget has been retired.
    pub(super) lod_pixel_threshold: f32,
    /// When > 0, the live-surface `render()` path emits a
    /// `render_live_sample` line every N frames with CPU-side phase
    /// timings (acquire / encode / submit / present / total). 0
    /// disables. Set via `--live-sample-every N`.
    pub(super) live_sample_every_frames: u32,
    /// Whether TAAU is enabled. Threaded to `Renderer::new` so the
    /// live and harness paths share the same resolve setup.
    pub(super) taa_enabled: bool,
    /// How entities render — ray-march through tag=3 (default) or
    /// instanced raster (landed later). Set from CLI
    /// `--entity-render` and baked into Renderer::new.
    pub(super) entity_render_mode: crate::renderer::EntityRenderMode,
    /// Compile-time entity-branch kill switch. When true, the
    /// ray-march pipeline is built with `ENABLE_ENTITIES=false`
    /// so Naga DCEs the tag==3 branch + march_entity_subtree.
    /// Set via `--no-entities`. Default false = entities enabled.
    pub(super) disable_entities: bool,
    /// World-coordinate Y where entities naturally rest. `Some` for
    /// flat worlds (sea level = a specific Y); `None` for sphere/
    /// fractal worlds where "resting height" is position-dependent.
    /// Consumed by `EntityStore::tick` to zero out the Y velocity
    /// component so entities stay on the ground they spawned on.
    pub(super) entity_surface_y: Option<f32>,
    /// Cached subtree NodeId for the soldier model loaded from
    /// `assets/vox/soldier.vox` on first `spawn_test_entities` call.
    /// `None` until the first press of N or M. Caches the parsed
    /// model so repeat presses don't re-read the .vox file or
    /// re-register palette entries.
    pub(super) cached_soldier_subtree: Option<crate::world::tree::NodeId>,
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
    pub(super) last_path_diag: String,
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
    #[allow(dead_code)] // wasm-only consumer; native build keeps the field for symmetry
    pub(super) proxy: winit::event_loop::EventLoopProxy<UserEvent>,
    /// True after we kicked off the async renderer init (WASM only).
    /// Stops `ensure_started` from re-spawning the future on every
    /// `resumed` / `about_to_wait` callback before the renderer
    /// actually arrives.
    #[allow(dead_code)] // wasm-only
    pub(super) renderer_init_started: bool,
    /// Phase 3 Step 3.0 debug knob: constant curvature `A` to set
    /// on the renderer once it's ready (renderer is created
    /// asynchronously after App::new). Read from `--curvature A`.
    pub(super) startup_curvature_a: Option<f32>,
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
        let startup_curvature_a = test_cfg.curvature_a;
        // Nyquist floor: sub-pixel rejection only. This is the
        // sole visual LOD gate; the stack depth (MAX_STACK_DEPTH
        // in the shader) is the hard ceiling.
        let lod_pixel_threshold = test_cfg.lod_pixels.unwrap_or(1.0);
        let live_sample_every_frames = test_cfg.live_sample_every_frames.unwrap_or(0);
        let taa_enabled = test_cfg.taa;
        let entity_render_mode = test_cfg.entity_render_mode;
        let disable_entities = test_cfg.disable_entities;
        let entity_surface_y = bootstrap::surface_y_for_preset(&test_cfg.world_preset);
        let interaction_radius_cells = test_cfg.interaction_radius.unwrap_or(6);
        let (harness_width, harness_height) = test_cfg.harness_size();
        // Pass the raw `Option` through so each preset's
        // `unwrap_or(...)` default applies when the user didn't
        // pass --plain-layers. Fractals default to 8 (where Block
        // leaves are pixel-visible); plain/vox world default to 40.
        let bootstrap = bootstrap::bootstrap_world(test_cfg.world_preset.clone(), test_cfg.plain_layers);
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
        let mut logical_path = position.anchor;
        let desired_depth = logical_path.depth().saturating_sub(RENDER_FRAME_K);
        logical_path.truncate(desired_depth);
        // Render-frame root must not be a TangentBlock — see
        // `App::render_frame` for the rationale.
        while logical_path.depth() > 0
            && (path_lands_on_tangent_block(&world.library, world.root, &logical_path)
                || path_is_strict_descendant_of_spherical_wrapped_plane(
                    &world.library, world.root, &logical_path,
                ))
        {
            logical_path.truncate(logical_path.depth() - 1);
        }
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

        // Pull spawn-entity fields out of `test_cfg` before
        // `from_config` consumes it, so the init-time entity spawn
        // can run after the struct is fully built.
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
            debug_copy_seq: 0,
            fps_smooth: 0.0,
            startup_profile_frames: if test_cfg.suppress_startup_logs { u32::MAX } else { 0 },
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
            live_sample_every_frames,
            taa_enabled,
            entity_render_mode,
            disable_entities,
            entity_surface_y,
            cached_soldier_subtree: None,
            interaction_radius_cells,
            last_highlight_raycast_ms: 0.0,
            last_highlight_set_ms: 0.0,
            last_pack_ms: 0.0,
            last_ribbon_build_ms: 0.0,
            last_packed_node_count: 0,
            last_ribbon_len: 0,
            last_effective_visual_depth: 0,
            last_reused_gpu_tree: false,
            last_path_diag: String::new(),
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
            startup_curvature_a,
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
        let mut logical_path = self.camera.position.anchor;
        let desired_depth = logical_path.depth().saturating_sub(RENDER_FRAME_K);
        logical_path.truncate(desired_depth);
        // Render frame must satisfy two constraints:
        //   1. Its root is not a TangentBlock — the shader's TB dispatch
        //      fires at TB *child* entry, never at the frame root, so
        //      a TB-rooted frame would render axis-aligned.
        //   2. Its root is not a strict descendant of a
        //      SphericalWrappedPlane — the sphere DDA dispatch fires
        //      only when SphericalWP is the active frame, never deeper.
        //      A deeper frame would route through `march_cartesian`,
        //      which doesn't know how to find sphere-positioned cells.
        // Truncate one level at a time while either rule is violated.
        // (Stops at root, where neither applies by construction.)
        while logical_path.depth() > 0
            && (self.path_lands_on_tangent_block(&logical_path)
                || self.path_is_strict_descendant_of_spherical_wp(&logical_path))
        {
            logical_path.truncate(logical_path.depth() - 1);
        }
        frame::with_render_margin(
            &self.world.library, self.world.root,
            &logical_path, RENDER_FRAME_CONTEXT,
        )
    }

    fn path_lands_on_tangent_block(&self, path: &Path) -> bool {
        path_lands_on_tangent_block(&self.world.library, self.world.root, path)
    }

    fn path_is_strict_descendant_of_spherical_wp(&self, path: &Path) -> bool {
        path_is_strict_descendant_of_spherical_wrapped_plane(
            &self.world.library, self.world.root, path,
        )
    }

    pub(super) fn update(&mut self, dt: f32) {
        // Camera-relative continuous flight (Minecraft-creative-style).
        // W/S follow camera forward (with pitch — looking down + W
        // dives), A/D follow camera right, Space/Shift go world ±Y.
        // Speed is in current-anchor-cell units per second so the
        // feel stays consistent across zoom levels: at any depth one
        // second of W moves you the same fraction of the cell.
        if !self.frozen && self.cursor_locked && !self.ui.any_panel_open() {
            let (fwd, right, _up) = self.camera.basis();
            let mut delta = [0.0_f32; 3];
            if self.keys.w { delta = crate::world::sdf::add(delta, fwd); }
            if self.keys.s { delta = crate::world::sdf::sub(delta, fwd); }
            if self.keys.d { delta = crate::world::sdf::add(delta, right); }
            if self.keys.a { delta = crate::world::sdf::sub(delta, right); }
            if self.keys.space { delta[1] += 1.0; }
            if self.keys.shift { delta[1] -= 1.0; }
            let len = crate::world::sdf::length(delta);
            if len > 1e-6 {
                let speed_cells_per_sec = 4.0;
                let scale = speed_cells_per_sec * dt / len;
                // `delta` is in WORLD axes (camera basis is in world
                // frame). The offset stored in `WorldPos` lives in
                // the deepest cell's children frame — for paths that
                // cross a `TangentBlock`, that's `R^T · world`. Apply
                // the cumulative anchor rotation's transpose so the
                // step has the same axes as the offset BEFORE adding.
                // `renormalize_world` then handles cell-boundary
                // crossings (including TB rotation pivots) cell-locally.
                let step_world = [delta[0] * scale, delta[1] * scale, delta[2] * scale];
                let (anchor_rot, anchor_scale) = frame_path_chain(
                    &self.world.library,
                    self.world.root,
                    &self.camera.position.anchor,
                );
                let rotated = crate::world::mat3::transpose_mul_vec3(&anchor_rot, &step_world);
                let inv_scale = if anchor_scale > 1e-6 { 1.0 / anchor_scale } else { 1.0 };
                let step_local = [
                    rotated[0] * inv_scale,
                    rotated[1] * inv_scale,
                    rotated[2] * inv_scale,
                ];
                self.camera.position.add_local(
                    step_local,
                    &self.world.library,
                    self.world.root,
                );
            }
        }
        player::update(&mut self.camera, dt);
        // Advance entities by velocity * dt. WorldPos renormalizes
        // so cell-boundary crossings are handled transparently; on
        // worlds with a defined sea level, `tick` zeroes the Y
        // velocity so entities don't drift off the ground.
        if !self.entities.is_empty() {
            self.entities.tick(&self.world.library, self.world.root, dt, self.entity_surface_y);
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
            ActiveFrameKind::Cartesian
            | ActiveFrameKind::WrappedPlane { .. }
            | ActiveFrameKind::SphericalWrappedPlane { .. } => {
                // Rotation-aware: when the anchor path crosses a
                // TangentBlock, every slot offset past it (and the
                // final offset) must be rotated by the cumulative
                // chain rotation. Plain `in_frame` walks Cartesian
                // and gets the wrong world position for cameras
                // inside a rotated subtree.
                self.camera.position.in_frame_rot(
                    &self.world.library, self.world.root, &frame.render_path,
                )
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
        // Walk the frame path from world root; if any node along the
        // way is a TangentBlock with non-identity rotation, accumulate
        // R into `frame_rot`. The frame's local frame is rotated R
        // relative to world, so to express world-direction vectors in
        // frame-local we apply R^T.
        let frame_rot = frame_path_rotation(
            &self.world.library, self.world.root, &frame.render_path,
        );
        let fwd_world_rot = crate::world::mat3::transpose_mul_vec3(&frame_rot, &fwd_world);
        let right_world_rot = crate::world::mat3::transpose_mul_vec3(&frame_rot, &right_world);
        let up_world_rot = crate::world::mat3::transpose_mul_vec3(&frame_rot, &up_world);
        let fwd_local = crate::world::sdf::normalize(fwd_world_rot);
        let right_local = crate::world::sdf::normalize(right_world_rot);
        let up_local = crate::world::sdf::normalize(up_world_rot);
        if self.startup_profile_frames < 4 {
            eprintln!(
                "gpu_camera basis world_fwd={:?} frame_local_fwd={:?} local_right={:?} local_up={:?}",
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

/// True iff walking `path` from `world_root` passes THROUGH a
/// `SphericalWrappedPlane` node before reaching the leaf — i.e., the
/// leaf is a strict descendant of a SphericalWP. Used by render-frame
/// selection: the render frame must never descend BELOW a SphericalWP,
/// because the sphere DDA dispatch fires only when the SphericalWP is
/// the active frame; any deeper frame would route through plain
/// `march_cartesian`, which doesn't know how to find sphere-positioned
/// cells.
///
/// Returns false if the path's leaf IS the SphericalWP (that's allowed
/// — it's the active frame for sphere DDA), or if no SphericalWP is on
/// the path at all.
pub(super) fn path_is_strict_descendant_of_spherical_wrapped_plane(
    library: &crate::world::tree::NodeLibrary,
    world_root: crate::world::tree::NodeId,
    path: &crate::world::anchor::Path,
) -> bool {
    use crate::world::tree::{Child, NodeKind};
    if path.depth() == 0 {
        return false;
    }
    let mut node = world_root;
    for k in 0..(path.depth() as usize) {
        // Check the CURRENT node (= a strict ancestor of the leaf
        // since we haven't walked into the leaf yet).
        if let Some(n) = library.get(node) {
            if matches!(n.kind, NodeKind::SphericalWrappedPlane { .. }) {
                return true;
            }
        }
        // Walk to next.
        match library.get(node).map(|n| n.children[path.slot(k) as usize]) {
            Some(Child::Node(child_id)) => node = child_id,
            _ => return false,
        }
    }
    false
}

/// True iff walking `path` from `world_root` lands on a node whose
/// kind is `TangentBlock`. Returns false for paths that leave the
/// tree (Block / Empty / EntityRef terminus) or for the root path
/// (depth 0). Used by render-frame selection: the render frame's
/// root must never be a TB, so callers truncate the path until this
/// returns false.
pub(super) fn path_lands_on_tangent_block(
    library: &crate::world::tree::NodeLibrary,
    world_root: crate::world::tree::NodeId,
    path: &crate::world::anchor::Path,
) -> bool {
    use crate::world::tree::{Child, NodeKind};
    if path.depth() == 0 {
        return false;
    }
    let mut node = world_root;
    for k in 0..path.depth() as usize {
        let Some(parent) = library.get(node) else { return false };
        match parent.children[path.slot(k) as usize] {
            Child::Node(child_id) => node = child_id,
            _ => return false,
        }
    }
    library
        .get(node)
        .map(|n| matches!(n.kind, NodeKind::TangentBlock { .. }))
        .unwrap_or(false)
}

/// Walk `frame_path` from `world_root`, accumulating the cumulative
/// `TangentBlock` rotation **and** inscribed-cube shrink scale.
/// Returns `(R, scale)` where `R` takes the frame's local axes to
/// the world root frame and `scale = ∏ tb_scale` over every TB on
/// the path. Identity / 1.0 for any path with no rotated nodes.
pub(super) fn frame_path_chain(
    library: &crate::world::tree::NodeLibrary,
    world_root: crate::world::tree::NodeId,
    frame_path: &crate::world::anchor::Path,
) -> ([[f32; 3]; 3], f32) {
    use crate::world::gpu::TbBoundary;
    use crate::world::tree::{Child, IDENTITY_ROTATION};
    let mut rot = IDENTITY_ROTATION;
    let mut scale: f32 = 1.0;
    let mut node = world_root;
    for k in 0..(frame_path.depth() as usize) {
        let n = match library.get(node) {
            Some(n) => n,
            None => return (rot, scale),
        };
        match n.children[frame_path.slot(k) as usize] {
            Child::Node(child_id) => {
                if let Some(child_node) = library.get(child_id) {
                    if let Some(b) = TbBoundary::from_kind(child_node.kind) {
                        rot = crate::world::mat3::matmul(&rot, &b.r);
                        scale *= b.tb_scale;
                    }
                }
                node = child_id;
            }
            _ => return (rot, scale),
        }
    }
    (rot, scale)
}

/// Rotation-only variant of [`frame_path_chain`] — convenience for
/// callers that don't need the scale.
pub(super) fn frame_path_rotation(
    library: &crate::world::tree::NodeLibrary,
    world_root: crate::world::tree::NodeId,
    frame_path: &crate::world::anchor::Path,
) -> [[f32; 3]; 3] {
    frame_path_chain(library, world_root, frame_path).0
}

