//! wgpu renderer: full-screen ray march shader.
//!
//! Five buffers: tree (per-child), node_kinds (per-node), camera,
//! palette, uniforms, plus an ancestor-ribbon storage buffer. The
//! shader walks the unified tree and dispatches on `NodeKind` when
//! descending — there are no parallel buffers for sphere content,
//! no `cs_*` uniforms, and no absolute-coord shimming.

mod buffers;
#[allow(non_camel_case_types)]
pub mod cursor_probe;
mod draw;
mod init;

pub use cursor_probe::{CursorProbe, CursorProbeRaw};
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
    /// Highlight cell's slot path from world root, packed 4 slot
    /// bytes per u32 (byte 0 = depth 0). 4 × vec4<u32> = 16 u32s =
    /// 64 slot bytes = MAX_DEPTH. The shader does path-prefix
    /// matching instead of f32 AABB checks — precision-safe at any
    /// anchor depth (see `bindings.wgsl::Uniforms.highlight_path`).
    ///
    /// Layout MUST match the WGSL struct order in `bindings.wgsl`.
    pub highlight_path: [[u32; 4]; 4],
    pub highlight_path_depth: u32,
    pub _pad_highlight0: u32,
    pub _pad_highlight1: u32,
    pub _pad_highlight2: u32,
    /// Body radii (used iff `root_kind == 1`). Stored in the body
    /// cell's local `[0, 1)` frame; the shader scales by 3.0
    /// (= WORLD_SIZE) to get shader-frame units.
    pub root_radii: [f32; 4],  // [inner_r, outer_r, _, _]
    pub root_face_meta: [u32; 4],
    pub root_face_bounds: [f32; 4],
    pub root_face_pop_pos: [f32; 4],
    /// World-root-relative slot path of the active render frame —
    /// used by the shader to reconstruct full hit-cell paths for
    /// comparison against `highlight_path`.
    pub render_path: [[u32; 4]; 4],
    pub render_path_depth: u32,
    pub _pad_render0: u32,
    pub _pad_render1: u32,
    pub _pad_render2: u32,
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
    pub(super) highlight_path: [[u32; 4]; 4],
    pub(super) highlight_path_depth: u32,
    pub(super) render_path: [[u32; 4]; 4],
    pub(super) render_path_depth: u32,
    pub(super) root_kind: u32,
    pub(super) root_radii: [f32; 4],
    pub(super) root_face_meta: [u32; 4],
    pub(super) root_face_bounds: [f32; 4],
    pub(super) root_face_pop_pos: [f32; 4],
    pub(super) ribbon_count: u32,
    pub(super) offscreen_texture: Option<wgpu::Texture>,
    /// Integer downscale divisor for the ray-march pass. 1 = render
    /// at full `config.{width,height}`; 2 = render at half per-axis
    /// (quarter pixel count) into `ray_march_target` and blit up to
    /// the destination with a bilinear sampler. The ray-march cost
    /// drops linearly in pixel count; the blit is a single cheap
    /// fullscreen sample pass.
    pub(super) render_scale: u32,
    /// Scaled-size intermediate target for the ray-march pass, used
    /// when `render_scale > 1`. Sampled by the blit pipeline on
    /// upscale. `None` when `render_scale == 1` — the ray-march
    /// writes directly to the destination in that case.
    pub(super) ray_march_target: Option<wgpu::Texture>,
    pub(super) blit_pipeline: wgpu::RenderPipeline,
    pub(super) blit_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) blit_sampler: wgpu::Sampler,
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
    /// Last live-surface render sub-phase timings (ms). Populated by
    /// `render()` so slow-frame diagnostics can pinpoint whether
    /// cost is in encode, submit, CPU wait for GPU, or GPU pass.
    pub(super) last_render_encode_ms: f64,
    pub(super) last_render_submit_ms: f64,
    pub(super) last_render_wait_ms: f64,
    pub(super) last_gpu_pass_ms: f64,
    /// GPU-resident cursor-probe compute pipeline + output/staging
    /// buffers. Dispatched once per frame from `render_offscreen()`;
    /// the host reads the staging buffer via `read_cursor_probe()`
    /// for the highlight uniform and for break/place edits.
    pub(super) cursor_probe_gpu: cursor_probe::CursorProbe_Gpu,
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

    /// Block on the GPU finishing the in-flight cursor-probe copy,
    /// then map the staging buffer and decode the result. Zero-copy
    /// read: we keep the buffer mapped only for the duration of the
    /// decode (drop → unmap) so the next frame can re-fill it.
    ///
    /// Sub-millisecond in practice — the compute dispatch is one
    /// workgroup of size 1 following a single ray, and the staging
    /// buffer is 80 bytes. The harness probe commands and break/
    /// place call this synchronously; per-frame highlight updates
    /// call this too (the copy is already waiting by the time the
    /// next frame's highlight update runs).
    pub fn read_cursor_probe(&self) -> cursor_probe::CursorProbe {
        let slice = self.cursor_probe_gpu.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        // Pump the device until the map_async callback fires. The
        // copy was scheduled in the previous submit() so GPU work
        // is already done or in flight; the poll turns a potentially
        // long wait into a tight drain.
        let _ = self.device.poll(wgpu::PollType::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            _ => return cursor_probe::CursorProbe::default(),
        }
        let raw_bytes = slice.get_mapped_range();
        let raw: &cursor_probe::CursorProbeRaw = bytemuck::from_bytes(&raw_bytes);
        let decoded = cursor_probe::CursorProbe::decode(raw);
        drop(raw_bytes);
        self.cursor_probe_gpu.staging_buffer.unmap();
        decoded
    }

    /// Encode a fresh cursor-probe dispatch, submit, wait, and read.
    /// Use this when the camera state changed THIS frame (e.g. a
    /// scripted pitch rotation in the harness) and the previous
    /// frame's probe result is stale. The per-frame render path's
    /// readback in `read_cursor_probe` reuses the pipeline's staging
    /// buffer; this variant submits a standalone compute + copy so
    /// the current uniforms drive the ray.
    pub fn dispatch_and_read_cursor_probe_sync(&self) -> cursor_probe::CursorProbe {
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("cursor_probe_sync") },
        );
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cursor_probe_sync"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cursor_probe_gpu.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_bind_group(1, &self.cursor_probe_gpu.bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.cursor_probe_gpu.output_buffer, 0,
            &self.cursor_probe_gpu.staging_buffer, 0,
            cursor_probe::CURSOR_PROBE_BYTES,
        );
        self.queue.submit(Some(encoder.finish()));
        self.read_cursor_probe()
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

    /// Ship the highlight as a slot path instead of an f32 AABB.
    ///
    /// The shader's per-pixel `march` dispatch populates `hit_path`
    /// in `HitResult`; `main.wgsl` compares the two as packed slot
    /// bytes (4 per u32, byte 0 = depth 0). Prefix-match: if the
    /// walker's hit cell descends through `highlight_path[..depth]`
    /// the pixel glows. Avoids every f32 precision wall that
    /// `highlight_min`/`highlight_max` hit below `cell_size <
    /// ULP(frame_magnitude)`.
    ///
    /// `slots` is read up to `MAX_DEPTH = 64` entries; each slot
    /// value must be `0..27`. Empty slice = no highlight.
    pub fn set_highlight_path(&mut self, slots: &[u8]) {
        const MAX: usize = 64;
        let depth = slots.len().min(MAX);
        let mut packed = [[0u32; 4]; 4];
        for (i, &slot) in slots.iter().take(depth).enumerate() {
            let word = i / 16;
            let lane = (i / 4) % 4;
            let byte = i % 4;
            packed[word][lane] |= (slot as u32) << (byte * 8);
        }
        self.highlight_path = packed;
        self.highlight_path_depth = depth as u32;
        self.highlight_active = if depth > 0 { 1 } else { 0 };
        self.write_uniforms();
    }

    /// Ship the active render frame's slot path (from world root).
    /// The shader uses it to reconstruct a hit cell's full path for
    /// comparison against `highlight_path`.
    pub fn set_render_path(&mut self, slots: &[u8]) {
        const MAX: usize = 64;
        let depth = slots.len().min(MAX);
        let mut packed = [[0u32; 4]; 4];
        for (i, &slot) in slots.iter().take(depth).enumerate() {
            let word = i / 16;
            let lane = (i / 4) % 4;
            let byte = i % 4;
            packed[word][lane] |= (slot as u32) << (byte * 8);
        }
        self.render_path = packed;
        self.render_path_depth = depth as u32;
        self.write_uniforms();
    }

    pub fn set_max_depth(&mut self, depth: u32) {
        self.max_depth = depth;
        self.write_uniforms();
    }

    /// Update the BFS index the shader uses as the frame root. The
    /// root of the full packed tree is always BFS index 0; this lets
    /// the shader start rendering from a deeper node (the current
    /// render frame) without re-uploading the tree buffer.
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
        self.ray_march_target = None;
        self.write_uniforms();
    }

    /// Logical (ray-marched) render size after applying `render_scale`.
    /// Both axes are clamped to a 1-pixel floor so a degenerate
    /// `config` doesn't produce a zero-sized texture.
    pub(super) fn scaled_size(&self) -> (u32, u32) {
        let scale = self.render_scale.max(1);
        (
            (self.config.width / scale).max(1),
            (self.config.height / scale).max(1),
        )
    }
}
