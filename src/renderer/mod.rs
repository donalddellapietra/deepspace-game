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

pub use draw::{OffscreenRenderTiming, ShaderStatsFrame};

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
    /// Brick voxel storage. Each brick = 4921 u32s (27³ cells, 4 cells
    /// per u32). Touched only when the outer DDA dispatches into
    /// `march_brick`. May be a 1-u32 stub when no bricks exist.
    pub(super) brick_data_buffer: wgpu::Buffer,
    pub(super) camera_buffer: wgpu::Buffer,
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

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.offscreen_texture = None;
        self.write_uniforms();
    }
}
