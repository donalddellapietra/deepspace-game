//! wgpu renderer: full-screen ray march shader.
//!
//! Five buffers: tree (per-child), node_kinds (per-node), camera,
//! palette, uniforms, plus an ancestor-ribbon storage buffer. The
//! shader walks the unified tree and dispatches on `NodeKind` when
//! descending — there are no parallel buffers for sphere content,
//! no `cs_*` uniforms, and no absolute-coord shimming.

mod buffers;
mod draw;
mod init;
mod taa;

pub use draw::{OffscreenRenderTiming, ShaderStatsFrame};
pub use taa::{FrameSignature, TaaState};

/// Maximum ancestor-ribbon depth supported by the shader. Larger
/// ribbons get truncated at upload (anything beyond can't pop).
/// 64 covers MAX_DEPTH=63 with one slack.
pub const MAX_RIBBON_LEN: usize = 64;

/// `root_kind` discriminant — must mirror the WGSL `RootKind*`
/// constants in `bindings.wgsl`.
pub const ROOT_KIND_CARTESIAN: u32 = 0;
pub const ROOT_KIND_BODY: u32 = 1;
pub const ROOT_KIND_FACE: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuUniforms {
    pub root_index: u32,
    pub node_count: u32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub max_depth: u32,
    pub highlight_active: u32,
    /// 0 = Cartesian, 1 = CubedSphereBody. Mirrors the `RootKind*`
    /// constants. When 1, the shader dispatches into sphere DDA at
    /// start-of-march; the body fills the `[0, 3)³` frame, and
    /// `root_inner_r`/`root_outer_r` give the body's radii.
    pub root_kind: u32,
    /// Number of ancestor ribbon entries. 0 = frame is at world
    /// root, no pop possible.
    pub ribbon_count: u32,
    pub highlight_min: [f32; 4],
    pub highlight_max: [f32; 4],
    /// Body radii (used iff `root_kind == 1`). Stored in the body
    /// cell's local `[0, 1)` frame; the shader scales by 3.0
    /// (= WORLD_SIZE) to get shader-frame units.
    pub root_radii: [f32; 4],  // [inner_r, outer_r, _, _]
    pub root_face_meta: [u32; 4],
    pub root_face_bounds: [f32; 4],
    pub root_face_pop_pos: [f32; 4],
}

pub struct Renderer {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) surface: wgpu::Surface<'static>,
    pub(super) config: wgpu::SurfaceConfiguration,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    /// Interleaved tree buffer (4 B × u32). Each packed node
    /// occupies `2 + 2*popcount(occupancy)` u32s: a 2-u32 header
    /// (occupancy mask + first_child u32-offset in this same buffer)
    /// followed by inline child entries (each 2 u32s: packed
    /// tag/block_type/pad + BFS node index).
    pub(super) tree_buffer: wgpu::Buffer,
    /// BFS index → tree[] u32-offset of that node's header. Touched
    /// only on descent / ribbon pop (cold path).
    pub(super) node_offsets_buffer: wgpu::Buffer,
    pub(super) node_kinds_buffer: wgpu::Buffer,
    /// Running counts of what's currently uploaded to the GPU
    /// buffers (u32s for tree; element counts for the BFS-indexed
    /// side buffers). `update_tree` uses these to write only the
    /// appended tail via `queue.write_buffer`, avoiding a whole-
    /// buffer re-upload + GPU write-barrier stall on every edit.
    pub(super) uploaded_tree_u32s: u64,
    pub(super) uploaded_kinds_count: u64,
    pub(super) uploaded_offsets_count: u64,
    pub(super) camera_buffer: wgpu::Buffer,
    /// CPU-side mirror of the most recent GpuCamera uploaded via
    /// `update_camera()`, with `jitter_x_px` / `jitter_y_px` always
    /// zero regardless of the real GPU-side jitter. The TAA resolve
    /// path needs the un-jittered camera to reconstruct ray
    /// directions at pixel centers; it also stashes a copy as the
    /// `prev_camera` for next frame's reprojection.
    pub(super) last_camera: crate::world::gpu::GpuCamera,
    pub(super) palette_buffer: wgpu::Buffer,
    pub(super) uniforms_buffer: wgpu::Buffer,
    pub(super) ribbon_buffer: wgpu::Buffer,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) root_index: u32,
    pub(super) node_count: u32,
    pub(super) max_depth: u32,
    pub(super) highlight_active: u32,
    pub(super) highlight_min: [f32; 4],
    pub(super) highlight_max: [f32; 4],
    pub(super) root_kind: u32,
    pub(super) root_radii: [f32; 4],
    pub(super) root_face_meta: [u32; 4],
    pub(super) root_face_bounds: [f32; 4],
    pub(super) root_face_pop_pos: [f32; 4],
    pub(super) ribbon_count: u32,
    pub(super) offscreen_texture: Option<wgpu::Texture>,
    /// Second ray-march pipeline compiled to the TAAU entry point
    /// (`fs_main_taa`) with two color attachments — linear RGBA16F
    /// color and R32F hit-t. `None` when TAAU is disabled; the draw
    /// path then uses the single-attachment `pipeline`.
    pub(super) pipeline_taa: Option<wgpu::RenderPipeline>,
    /// TAAU state: history textures, resolve pipeline, jitter,
    /// previous camera. `None` when TAAU is disabled.
    pub(super) taa: Option<TaaState>,
    /// Optional GPU timestamp-query scaffolding. Present only when
    /// the adapter reports `Features::TIMESTAMP_QUERY`. Used by
    /// `render_offscreen` to measure the ray-march pass on the GPU
    /// side, not just the CPU-side `device.poll(Wait)` duration.
    pub(super) timestamp: Option<TimestampScratch>,
    /// Last queue.write_buffer durations (camera/ribbon/tree) in ms.
    /// Populated by the buffer-upload path so the harness can break
    /// "upload" into per-buffer sub-phases.
    pub(super) last_camera_write_ms: f64,
    pub(super) last_ribbon_write_ms: f64,
    pub(super) last_tree_write_ms: f64,
    pub(super) last_bind_group_rebuild_ms: f64,
    /// Shader-side atomic counters written by the fragment shader
    /// each frame (ray_count, hit_count, miss_count, max_iter_count,
    /// sum_steps_div4, max_steps, + 2 u32 pad). 32 bytes total.
    pub(super) shader_stats_buffer: wgpu::Buffer,
    /// Mappable COPY_DST shadow of `shader_stats_buffer`. Populated
    /// via `copy_buffer_to_buffer` at the end of the render pass,
    /// mapped after `poll(Wait)` so the harness can read the 8 u32s.
    pub(super) shader_stats_readback: wgpu::Buffer,
    /// When false, `render_offscreen` skips the stats clear / copy /
    /// map round-trip and returns a zeroed `ShaderStatsFrame`. The
    /// shader's atomic writes are compiled out via the `ENABLE_STATS`
    /// override so there's no per-pixel cost either.
    pub(super) shader_stats_enabled: bool,
    /// Rolling counter of live-surface frames rendered. Used by
    /// `render()` to periodically emit a `render_live_sample` line
    /// with per-phase CPU timings, so we can see the steady-state
    /// breakdown without waiting for a slow frame.
    pub(super) live_frame_counter: u64,
    /// When > 0, `render()` emits a `render_live_sample` every N
    /// frames. CPU-side only — no `device.poll(Wait)` stall. Set via
    /// `--live-sample-every N` CLI flag; 0 (default) disables.
    pub(super) live_sample_every_frames: u32,
}

/// GPU timestamp query resources. `query_set` holds two timestamp
/// slots (pass start, pass end); `resolve` is the COPY_SRC buffer
/// that `resolve_query_set` writes ticks into; `staging` is a
/// MAP_READ buffer used to read the ticks back on the CPU.
pub struct TimestampScratch {
    pub query_set: wgpu::QuerySet,
    pub resolve: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub period_ns: f32,
}

impl Renderer {
    /// Set the frame-root NodeKind to Cartesian (default).
    pub fn set_root_kind_cartesian(&mut self) {
        self.root_kind = ROOT_KIND_CARTESIAN;
        self.root_radii = [0.0; 4];
        self.root_face_meta = [0; 4];
        self.root_face_bounds = [0.0; 4];
        self.root_face_pop_pos = [0.0; 4];
        self.write_uniforms();
    }

    /// Set the frame-root NodeKind to CubedSphereBody with radii in
    /// the body cell's local `[0, 1)` frame.
    pub fn set_root_kind_body(&mut self, inner_r: f32, outer_r: f32) {
        self.root_kind = ROOT_KIND_BODY;
        self.root_radii = [inner_r, outer_r, 0.0, 0.0];
        self.root_face_meta = [0; 4];
        self.root_face_bounds = [0.0; 4];
        self.root_face_pop_pos = [0.0; 4];
        self.write_uniforms();
    }

    pub fn set_root_kind_face(
        &mut self,
        inner_r: f32,
        outer_r: f32,
        face_id: u32,
        subtree_depth: u32,
        bounds: [f32; 4],
        pop_pos: [f32; 3],
    ) {
        self.root_kind = ROOT_KIND_FACE;
        self.root_radii = [inner_r, outer_r, 0.0, 0.0];
        self.root_face_meta = [face_id, subtree_depth, 0, 0];
        self.root_face_bounds = bounds;
        self.root_face_pop_pos = [pop_pos[0], pop_pos[1], pop_pos[2], 0.0];
        self.write_uniforms();
    }

    pub fn set_highlight(&mut self, aabb: Option<([f32; 3], [f32; 3])>) {
        match aabb {
            Some((min, max)) => {
                self.highlight_active = 1;
                self.highlight_min = [min[0], min[1], min[2], 0.0];
                self.highlight_max = [max[0], max[1], max[2], 0.0];
            }
            None => { self.highlight_active = 0; }
        }
        self.write_uniforms();
    }

    pub fn set_max_depth(&mut self, depth: u32) {
        self.max_depth = depth;
        self.write_uniforms();
    }

    /// Update the BFS index the shader uses as the frame root.
    /// `CachedTree::update_root` emits in post-order DFS, so the
    /// world-root entry ends up LAST in the buffer (not at BFS idx
    /// 0 as a BFS-emit would produce). The current `root_bfs_idx`
    /// lives on `CachedTree` and is passed through here every frame
    /// so the shader starts at the correct node — callers should
    /// prefer `cache.root_bfs_idx` over any assumption about BFS 0.
    pub fn set_frame_root(&mut self, bfs_idx: u32) {
        self.root_index = bfs_idx;
        self.write_uniforms();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.offscreen_texture = None;
        if let Some(taa) = self.taa.as_mut() {
            taa.resize(&self.device, width, height);
        }
        self.write_uniforms();
    }

    /// Mark the TAAU history as invalid for the next few frames.
    /// Callers must invoke this whenever the render-frame root
    /// shifts (zoom, ribbon pop, teleport, spawn). A shift means
    /// the camera's local coordinate system is no longer the same
    /// as the one that produced the previous history, so naive
    /// reprojection would sample garbage. The frame-signature
    /// comparison inside `TaaState::begin_frame` also catches
    /// automatic changes via `set_frame_root` / `set_root_kind_*`,
    /// so this is primarily for app-level events.
    pub fn invalidate_taa_history(&mut self) {
        if let Some(taa) = self.taa.as_mut() {
            taa.invalidate_history();
        }
    }

    /// Whether TAAU is enabled for this renderer. Read-only — the
    /// flag is set at `new` time; toggling at runtime would require
    /// rebuilding pipelines.
    pub fn taa_enabled(&self) -> bool { self.taa.is_some() }

    /// Dimensions of the ray-march output attachment. Equal to the
    /// swapchain size when TAAU is off; equal to the TAAU half-res
    /// target when TAAU is on. The uniforms and shader jitter math
    /// both care about this, not the swapchain size.
    pub(super) fn march_dims(&self) -> (u32, u32) {
        match self.taa.as_ref() {
            Some(t) => (t.scaled_width, t.scaled_height),
            None => (self.config.width, self.config.height),
        }
    }

    pub(super) fn current_frame_signature(&self) -> FrameSignature {
        FrameSignature {
            root_index: self.root_index,
            root_kind: self.root_kind,
            ribbon_count: self.ribbon_count,
        }
    }
}
