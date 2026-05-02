//! wgpu renderer: full-screen ray march shader.
//!
//! Five buffers: tree (per-child), node_kinds (per-node), camera,
//! palette, uniforms, plus an ancestor-ribbon storage buffer. The
//! shader walks the unified tree and dispatches on `NodeKind` when
//! descending — there are no parallel buffers for sphere content,
//! no `cs_*` uniforms, and no absolute-coord shimming.

mod beam_mask;
mod buffers;
mod draw;
pub mod entity_raster;
pub mod heightmap;
mod init;
mod stats;
mod taa;

pub use entity_raster::{compute_view_proj, EntityRasterState, InstanceData};
pub use stats::{OffscreenRenderTiming, ShaderStatsFrame, WalkerProbeFrame};
pub use taa::{FrameSignature, TaaState};

/// How entities are rendered. Chosen at startup via
/// `--entity-render` and baked into the Renderer (pipelines and
/// buffers are allocated accordingly). Mutating at runtime isn't
/// supported — a toggle would require rebuilding the ray-march
/// pipeline and reallocating the depth texture.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum EntityRenderMode {
    #[default]
    /// Entities enter the tree as `Child::EntityRef(idx)` and are
    /// ray-marched through the tag=3 branch of `march_cartesian`.
    /// Default. Decent perf up to ~1k entities.
    RayMarch,
    /// Entities are rendered as instanced triangle meshes in a
    /// separate raster pass after the ray-march. Scales to 100k+.
    /// Incompatible with TAA (depth buffer handoff would need
    /// half-res adaptation). Landed in a later commit; ray-march
    /// is the only mode active on this branch until then.
    Raster,
}

/// Maximum ancestor-ribbon depth supported by the shader. Larger
/// ribbons get truncated at upload (anything beyond can't pop).
/// 64 covers MAX_DEPTH=63 with one slack.
pub const MAX_RIBBON_LEN: usize = 64;

/// `root_kind` discriminant — must mirror the WGSL `ROOT_KIND_*`
/// constants in `bindings.wgsl`. Phase 1: WrappedPlane renders
/// identically to Cartesian (the marcher does not yet branch on
/// root_kind); the constant exists so the uniform / shader layouts
/// stay in lockstep with the CPU side once Phase 2 / 3 add wrap and
/// curvature dispatch.
pub const ROOT_KIND_CARTESIAN: u32 = 0;
pub const ROOT_KIND_WRAPPED_PLANE: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuUniforms {
    pub root_index: u32,
    pub node_count: u32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub max_depth: u32,
    pub highlight_active: u32,
    /// 0 = Cartesian. Other discriminants used to dispatch sphere
    /// DDA branches; those are gone. Field retained so the layout
    /// matches the shader-side `Uniforms` struct exactly.
    pub root_kind: u32,
    /// Number of ancestor ribbon entries. 0 = frame is at world
    /// root, no pop possible.
    pub ribbon_count: u32,
    /// Number of live entities in the entity buffer (binding 10).
    /// Shader's tag=3 dispatch uses it as a validity gate; zero
    /// means the entity path is inert.
    pub entity_count: u32,
    pub _pad_entity: [u32; 3],
    pub highlight_min: [f32; 4],
    pub highlight_max: [f32; 4],
    /// Padding slot retained so the CPU-side `GpuUniforms` matches
    /// the WGSL `Uniforms` block byte-for-byte. Unused.
    pub _pad_radii: [f32; 4],
    /// `WrappedPlane` slab dimensions: `(dims_x, dims_y, dims_z,
    /// slab_depth)`. Populated when `root_kind ==
    /// ROOT_KIND_WRAPPED_PLANE`; the shader's X-wrap branch reads
    /// `slab_dims.x` and `slab_dims.w`. Zero-filled on Cartesian
    /// root frames. Mirrors `Uniforms.slab_dims` in
    /// `assets/shaders/bindings.wgsl`.
    pub slab_dims: [u32; 4],
    pub _pad_face_bounds: [f32; 4],
    pub _pad_face_pop_pos: [f32; 4],
    /// Visual debug paint mode. 0 = off (normal rendering); 1..=8 are
    /// the diagnostic paint modes in `march_debug.wgsl`. Lives in
    /// `.x`; `.yzw` reserved for per-mode tuning. Modes 7 and 8 are
    /// reserved placeholders for the wrapped-planet phases (Phase 2 /
    /// Phase 3 wire the underlying state).
    pub debug_mode: [u32; 4],
    /// `xy` = screen-space pixel to probe walker state for;
    /// `z` = non-zero means probing is active (0 disables all writes
    /// to `walker_probe`). `w` reserved.
    pub probe_pixel: [u32; 4],
    /// Phase 3 Step 3.0 curvature: `.x = A` is the per-step
    /// parabolic-drop coefficient (`drop = A · dist²` applied to
    /// `child_entry.y` at each descent). 0 = disabled; the marcher
    /// stays bit-identical to the flat path. `.yzw` reserved for
    /// k(altitude) ramp / R_inv / slab_surface_y in later steps.
    pub curvature: [f32; 4],
    /// Phase 3 REVISED — UV-sphere render mode toggle.
    /// `.x = 0` → flat slab DDA (current default).
    /// `.x = 1` → render WrappedPlane frame as a sphere.
    /// `.y = lat_max` (radians) — poles past this latitude return
    /// no-hit (banned). `.zw` reserved.
    pub planet_render: [f32; 4],
    /// Orthonormal rotation matrix applied at the descent boundary
    /// into any `NodeKind::TangentBlock` child (rows = basis vectors
    /// of the rotated child's local axes expressed in the parent
    /// frame). Stored as 3 vec4 rows (`.xyz` is the row, `.w` pad).
    /// Default identity (no rotation). The shader's `march_cartesian`
    /// applies `R^-1 = R^T` to the ray on push and `R` to the hit
    /// normal on return.
    pub tangent_rotation_row0: [f32; 4],
    pub tangent_rotation_row1: [f32; 4],
    pub tangent_rotation_row2: [f32; 4],
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
    /// BFS index → packed 12-bit content AABB in the low 12 bits.
    /// See `gpu::pack::CachedTree::aabbs`. Parallel to
    /// `node_offsets_buffer`; the ray-march descent uses it to cull
    /// rays that miss the subtree's occupied region.
    pub(super) aabbs_buffer: wgpu::Buffer,
    pub(super) node_kinds_buffer: wgpu::Buffer,
    /// Running counts of what's currently uploaded to the GPU
    /// buffers (u32s for tree; element counts for the BFS-indexed
    /// side buffers). `update_tree` uses these to write only the
    /// appended tail via `queue.write_buffer`, avoiding a whole-
    /// buffer re-upload + GPU write-barrier stall on every edit.
    pub(super) uploaded_tree_u32s: u64,
    pub(super) uploaded_kinds_count: u64,
    pub(super) uploaded_offsets_count: u64,
    pub(super) uploaded_aabbs_count: u64,
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
    /// `(dims_x, dims_y, dims_z, slab_depth)` for the active
    /// `WrappedPlane` render frame. Zero-filled when `root_kind ==
    /// ROOT_KIND_CARTESIAN`. Uploaded as `Uniforms.slab_dims`; the
    /// shader's X-wrap branch reads the X and W lanes.
    pub(super) slab_dims: [u32; 4],
    pub(super) ribbon_count: u32,
    /// Number of live entities. Drives the uniforms' `entity_count`
    /// (shader-side gate for the tag=3 dispatch path) and the
    /// instance-buffer dispatch count for the raster entity pass.
    pub(super) entity_count: u32,
    /// Per-entity storage buffer (binding 10). Populated each frame
    /// by `update_entities` with the live entity list; the ray-march
    /// shader indexes into it from tag=3 child entries.
    pub(super) entity_buffer: wgpu::Buffer,
    pub(super) uploaded_entities_count: u64,
    pub(super) offscreen_texture: Option<wgpu::Texture>,
    /// Beam-prepass mask: 1/BEAM_TILE_SIZE-per-axis R8Unorm render
    /// target populated by `coarse_pipeline` at the start of each
    /// frame. `fs_main` samples this 5 times per pixel and returns
    /// sky directly when every tap reads 0, skipping the
    /// register-constrained tree walk for sky tiles.
    pub(super) mask_texture: wgpu::Texture,
    pub(super) mask_view: wgpu::TextureView,
    /// 1×1 dummy `texture_2d<f32>` that replaces `mask_texture` at
    /// bind slot 8 during the coarse pass — a texture can't be
    /// simultaneously bound as a render target and a sampled input.
    /// `fs_coarse_mask` doesn't reference `coarse_mask`, so the
    /// dummy's contents never matter.
    pub(super) dummy_mask_view: wgpu::TextureView,
    /// Coarse-pass render pipeline. Same bind group layout as the
    /// main pipeline (one layout for both simplifies the code) but
    /// targets R8Unorm at 1/BEAM_TILE_SIZE resolution and uses the
    /// `fs_coarse_mask` fragment entry.
    pub(super) coarse_pipeline: wgpu::RenderPipeline,
    /// Bind group for the coarse pass. Identical to `bind_group`
    /// except slot 8 is bound to `dummy_mask_view` instead of
    /// `mask_view`.
    pub(super) coarse_bind_group: wgpu::BindGroup,
    /// When true, the coarse pass runs to populate the mask and
    /// `fs_main`'s 5-tap check culls sky tiles. When false, the
    /// mask is cleared to 1.0 (every pixel marches) and the coarse
    /// pass is skipped. Toggled per-frame by `set_beam_enabled`
    /// based on a CPU heuristic in the app: sparse root + camera
    /// inside occupied cell → enable; otherwise skip the coarse
    /// overhead and let the fine pass march every pixel directly.
    pub(super) beam_enabled: bool,
    /// Second ray-march pipeline compiled to the TAAU entry point
    /// (`fs_main_taa`) with two color attachments — linear RGBA16F
    /// color and R32F hit-t. `None` when TAAU is disabled; the draw
    /// path then uses the single-attachment `pipeline`.
    pub(super) pipeline_taa: Option<wgpu::RenderPipeline>,
    /// TAAU state: history textures, resolve pipeline, jitter,
    /// previous camera. `None` when TAAU is disabled.
    pub(super) taa: Option<TaaState>,
    /// Last queue.write_buffer durations (camera/ribbon/tree) in ms.
    /// Populated by the buffer-upload path so the harness can break
    /// "upload" into per-buffer sub-phases.
    pub(super) last_camera_write_ms: f64,
    pub(super) last_ribbon_write_ms: f64,
    pub(super) last_tree_write_ms: f64,
    pub(super) last_bind_group_rebuild_ms: f64,
    /// Shader-side atomic counters written by the fragment shader
    /// each frame. Layout matches the `ShaderStats` struct in
    /// `bindings.wgsl`; 64 bytes total (16 u32 slots).
    pub(super) shader_stats_buffer: wgpu::Buffer,
    /// Mappable COPY_DST shadow of `shader_stats_buffer`. Populated
    /// via `copy_buffer_to_buffer` at the end of the render pass,
    /// mapped after `poll(Wait)` so the harness can read it back.
    pub(super) shader_stats_readback: wgpu::Buffer,
    /// Per-pixel walker-state probe SSBO (binding 11). 64 bytes / 16
    /// u32s. Cleared each frame; `march_cartesian` writes the matching
    /// pixel's state if `uniforms.probe_pixel.z != 0` and the probe
    /// pixel matches `current_pixel`. Read back via
    /// `walker_probe_readback` after render.
    pub(super) walker_probe_buffer: wgpu::Buffer,
    pub(super) walker_probe_readback: wgpu::Buffer,
    /// Mirror of `uniforms.probe_pixel`. `xy` = pixel coords;
    /// `z` = active flag; `w` reserved. Set by
    /// `set_walker_probe_pixel`.
    pub(super) probe_pixel: [u32; 4],
    /// Mirror of `uniforms.debug_mode`. `.x` = mode (0..=8); `.yzw`
    /// reserved. Set by `set_debug_mode`.
    pub(super) debug_mode: [u32; 4],
    /// Mirror of `uniforms.curvature`. `.x = A` is the per-step
    /// parabolic-drop coefficient applied at every descent in
    /// `march_cartesian`. 0 = disabled (flat path). Set by
    /// `set_curvature_a` from a CLI debug flag in Step 3.0.
    pub(super) curvature: [f32; 4],
    /// Mirror of `uniforms.planet_render`. `.x` = render mode:
    /// 0 = flat slab DDA, 1 = UV-sphere. `.y` = lat_max (radians).
    pub(super) planet_render: [f32; 4],
    /// Orthonormal rotation matrix applied at descent into any
    /// `NodeKind::TangentBlock` child. Rows are the rotated frame's
    /// local axes expressed in the parent frame; identity = no
    /// rotation, marcher unchanged. Set via `set_tangent_rotation`.
    pub(super) tangent_rotation_row0: [f32; 4],
    pub(super) tangent_rotation_row1: [f32; 4],
    pub(super) tangent_rotation_row2: [f32; 4],
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
    /// Which entity render path this renderer is wired for. Baked
    /// at `Renderer::new` time; the raster path allocates the
    /// depth-aware ray-march pipeline, a depth texture, and an
    /// `EntityRasterState`.
    pub(super) entity_render_mode: EntityRenderMode,
    /// Depth texture used by the raster entity pass. Present iff
    /// `entity_render_mode == Raster`. The ray-march pass writes
    /// `@builtin(frag_depth)` via the `fs_main_depth` entry point;
    /// the raster pass runs second with `depth_compare: Less`.
    pub(super) depth_texture: Option<wgpu::Texture>,
    pub(super) depth_view: Option<wgpu::TextureView>,
    /// Ray-march pipeline compiled with a depth attachment + the
    /// `fs_main_depth` fragment entry point. Used only when
    /// `entity_render_mode == Raster`.
    pub(super) pipeline_with_depth: Option<wgpu::RenderPipeline>,
    /// Raster pass for entity meshes. Present iff
    /// `entity_render_mode == Raster`.
    pub(super) entity_raster: Option<EntityRasterState>,

    // --- Heightmap / entity physics (raster mode only) ---
    /// GPU heightmap-gen compute pipeline.
    pub(super) heightmap_gen: Option<heightmap::HeightmapGen>,
    /// GPU per-entity Y-clamp compute pipeline.
    pub(super) entity_heightmap_clamp: Option<heightmap::EntityHeightmapClamp>,
    /// Current heightmap texture + its uniforms. Reallocated when
    /// `delta` (collision depth relative to frame depth) changes.
    pub(super) heightmap_texture: Option<heightmap::HeightmapTexture>,
    /// True whenever the tree / frame root / collision depth
    /// changed since the last `heightmap_gen` dispatch.
    pub(super) heightmap_dirty: bool,
    /// Frame-root BFS index the heightmap was last generated for.
    pub(super) heightmap_frame_root_bfs: u32,
}

/// Depth attachment format for the raster entity pass.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub(super) fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("entity_raster_depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

impl Renderer {
    /// Set the frame-root NodeKind to Cartesian. Clears
    /// `slab_dims` so the X-wrap branch in the shader can't fire
    /// from a stale upload.
    pub fn set_root_kind_cartesian(&mut self) {
        self.root_kind = ROOT_KIND_CARTESIAN;
        self.slab_dims = [0; 4];
        self.write_uniforms();
    }

    /// Set the frame-root NodeKind to `WrappedPlane`, carrying the
    /// slab's `(dims_x, dims_y, dims_z, slab_depth)` for the
    /// shader's wrap branch. The shader uses `dims_x` and
    /// `slab_depth` to compute the wrap shift in slab-root local
    /// units; `dims_y` / `dims_z` are reserved for Phase 3.
    pub fn set_root_kind_wrapped_plane(&mut self, dims: [u32; 3], slab_depth: u8) {
        self.root_kind = ROOT_KIND_WRAPPED_PLANE;
        self.slab_dims = [dims[0], dims[1], dims[2], slab_depth as u32];
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

    /// Per-frame toggle for the beam prepass. Callers compute a
    /// cheap CPU heuristic (root occupancy popcount, camera's root
    /// cell) and set this; when false, the renderer skips the
    /// coarse pass and clears the mask to 1.0 so the fine pass's
    /// 5-tap check always passes — equivalent to running without
    /// P1 at all but keeping the shader path constant.
    pub fn set_beam_enabled(&mut self, enabled: bool) {
        self.beam_enabled = enabled;
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
        // Beam-prepass mask sizes to 1/BEAM_TILE_SIZE of the new
        // swapchain. Recreate both texture and bind group.
        let (mask_texture, mask_view) =
            self::beam_mask::create_mask_texture(&self.device, width, height);
        self.mask_texture = mask_texture;
        self.mask_view = mask_view;
        self.bind_group = self::buffers::make_bind_group(
            &self.device, &self.bind_group_layout,
            &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
            &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
            &self.shader_stats_buffer, &self.node_offsets_buffer,
            &self.aabbs_buffer,
            &self.mask_view,
            &self.entity_buffer,
            &self.walker_probe_buffer,
        );
        // coarse_bind_group uses the dummy mask view which doesn't
        // resize, so it stays valid across resizes.
        if let Some(taa) = self.taa.as_mut() {
            taa.resize(&self.device, width, height);
        }
        if self.entity_render_mode == EntityRenderMode::Raster {
            let (tex, view) = create_depth_texture(&self.device, width, height);
            self.depth_texture = Some(tex);
            self.depth_view = Some(view);
        }
        self.write_uniforms();
    }

    pub fn entity_render_mode(&self) -> EntityRenderMode { self.entity_render_mode }

    pub fn entity_raster_mut(&mut self) -> Option<&mut EntityRasterState> {
        self.entity_raster.as_mut()
    }

    /// Force a heightmap rebuild on the next frame.
    pub fn mark_heightmap_dirty(&mut self) {
        self.heightmap_dirty = true;
    }

    /// Ensure a heightmap texture of `3^delta` side exists.
    /// Reallocates if `delta` changed; no-op otherwise.
    pub fn ensure_heightmap(&mut self, delta: u32) {
        let cap_delta = delta.min(6);
        let need_alloc = match self.heightmap_texture.as_ref() {
            Some(h) => h.delta != cap_delta,
            None => true,
        };
        if need_alloc {
            self.heightmap_texture = Some(
                heightmap::HeightmapTexture::new(&self.device, &self.queue, cap_delta),
            );
            self.heightmap_dirty = true;
        }
    }

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

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

    /// Public wrapper for `march_dims` — used by the app layer to
    /// build the view+projection matrix with the right aspect ratio
    /// (the ray-march pass is half-res under TAAU).
    pub fn march_dims_public(&self) -> (u32, u32) { self.march_dims() }

    pub(super) fn current_frame_signature(&self) -> FrameSignature {
        FrameSignature {
            root_index: self.root_index,
            root_kind: self.root_kind,
            ribbon_count: self.ribbon_count,
        }
    }

    /// Enable or disable the walker-state probe for one pixel. When
    /// `active` is true, `march_cartesian` writes a 16-u32 record into
    /// `walker_probe` ONLY for the matching pixel each frame; the
    /// caller then calls `read_walker_probe` to get the values.
    /// `active=false` zeros the gate; the buffer is still cleared each
    /// frame so a later read returns `hit_flag=0`.
    pub fn set_walker_probe_pixel(&mut self, x: u32, y: u32, active: bool) {
        self.probe_pixel = [x, y, if active { 1 } else { 0 }, 0];
        self.write_uniforms();
    }

    /// Set the visual debug paint mode (see `march_debug.wgsl`).
    /// 0 = off (normal render). 1..=8 paint per-pixel diagnostics.
    /// Modes 7 / 8 are placeholders for the wrapped-planet phases —
    /// they return the unwired sentinel until Phase 2 / Phase 3 land.
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = [mode, 0, 0, 0];
        self.write_uniforms();
    }

    /// Phase 3 Step 3.0: set the constant curvature coefficient `A`
    /// for the per-step parabolic-drop bend `child_entry.y -= A·dist²`
    /// applied at every descent in `march_cartesian`. `A = 0` (the
    /// default) keeps the marcher bit-identical to the flat path.
    /// Wired via the `--curvature A` CLI debug flag; later Phase 3
    /// steps will compute `A` per-frame from camera altitude.
    pub fn set_curvature_a(&mut self, a: f32) {
        self.curvature = [a, 0.0, 0.0, 0.0];
        self.write_uniforms();
    }

    /// Phase 3 REVISED: enable UV-sphere render for the WrappedPlane
    /// frame. `mode = 0` → flat slab DDA (default), `mode = 1` →
    /// sphere intersect + (lon, lat) → cell. `lat_max` is the polar
    /// ban threshold in radians (e.g. 1.26 ≈ 72°).
    /// Install the rotation applied at descent into any
    /// `NodeKind::TangentBlock` child. `rows` is row-major: each
    /// row is the corresponding rotated-frame axis expressed in
    /// the parent frame. The matrix MUST be orthonormal — non-
    /// orthonormal matrices break the t-parameter invariance the
    /// DDA relies on (lengths inside the rotated subtree drift
    /// per descent and the cell-boundary side_dist becomes
    /// inconsistent). Pass identity to disable rotation.
    pub fn set_tangent_rotation(&mut self, rows: [[f32; 3]; 3]) {
        self.tangent_rotation_row0 = [rows[0][0], rows[0][1], rows[0][2], 0.0];
        self.tangent_rotation_row1 = [rows[1][0], rows[1][1], rows[1][2], 0.0];
        self.tangent_rotation_row2 = [rows[2][0], rows[2][1], rows[2][2], 0.0];
    }

    pub fn set_planet_render_sphere(&mut self, mode: u32, lat_max_rad: f32) {
        self.planet_render = [mode as f32, lat_max_rad, 0.0, 0.0];
        self.write_uniforms();
    }
}
