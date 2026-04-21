//! wgpu renderer: full-screen ray march shader.
//!
//! Five buffers: tree (per-child), node_kinds (per-node), camera,
//! palette, uniforms, plus an ancestor-ribbon storage buffer. The
//! shader walks the unified tree and dispatches on `NodeKind` when
//! descending — there are no parallel buffers for sphere content,
//! no `cs_*` uniforms, and no absolute-coord shimming.

mod buffers;
mod draw;
pub mod entity_raster;
pub mod heightmap;
mod init;
mod taa;

pub use draw::{OffscreenRenderTiming, ShaderStatsFrame};
pub use entity_raster::{compute_view_proj, EntityRasterState, InstanceData};
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
/// constants in `bindings.wgsl`.
pub const ROOT_KIND_CARTESIAN: u32 = 0;
pub const ROOT_KIND_BODY: u32 = 1;
pub const ROOT_KIND_FACE: u32 = 2;
/// Face-subtree render frame at face-subtree depth ≥ 3 — shader
/// dispatches the linearized local-frame sphere DDA in
/// `sphere_in_sub_frame`.
pub const ROOT_KIND_SPHERE_SUB: u32 = 3;

/// Max UVR pre-descent depth supported by the shader's SphereSub
/// walker. Each slot is packed as u32 (1 per 16-byte uniform array
/// element to keep std140/uniform alignment simple); tighten to a
/// u8-packed layout later if the uniform budget becomes a concern.
pub const MAX_SPHERE_SUB_DEPTH: usize = 64;

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
    /// Number of live entities in the entity buffer (binding 10).
    /// Shader's tag=3 dispatch uses it as a validity gate; zero
    /// means the entity path is inert.
    pub entity_count: u32,
    pub _pad_entity: [u32; 3],
    pub highlight_min: [f32; 4],
    pub highlight_max: [f32; 4],
    /// Body radii (used iff `root_kind == 1` or `3`). Stored in the
    /// body cell's local `[0, 1)` frame; the shader scales by 3.0
    /// (= WORLD_SIZE) to get shader-frame units.
    pub root_radii: [f32; 4],  // [inner_r, outer_r, _, _]
    pub root_face_meta: [u32; 4],
    pub root_face_bounds: [f32; 4],
    pub root_face_pop_pos: [f32; 4],

    // ───────── SphereSub fields (iff root_kind == 3) ─────────
    /// xyz = body-XYZ of local (0, 0, 0) at the DEEP sub-frame
    /// corner. w = `deep_scale` = `1/3^(deep_m - eval_m)` — the
    /// ratio between the deep sub-frame's cell size and the eval
    /// anchor's cell size. Shader multiplies `sub_face_corner.w`
    /// (eval_frame_size) by `deep_scale` to recover the deep
    /// `frame_size` for hit-cell reporting + neighbor-step.
    pub sub_c_body: [f32; 4],
    /// Jacobian columns. xyz used.
    pub sub_j_col0: [f32; 4],
    pub sub_j_col1: [f32; 4],
    pub sub_j_col2: [f32; 4],
    /// Inverse Jacobian columns. xyz used.
    pub sub_j_inv_col0: [f32; 4],
    pub sub_j_inv_col1: [f32; 4],
    pub sub_j_inv_col2: [f32; 4],
    /// xyz = `(un_eval, vn_eval, rn_eval)` face-normalized corner
    /// at the Jacobian's EVAL depth. w = `eval_frame_size =
    /// 1/3^eval_m`. The deep sub-frame's absolute corner equals the
    /// eval corner plus a symbolic offset carried by `sub_uvr_slots`
    /// from index `eval_m` onward; its cell size equals
    /// `eval_frame_size * deep_scale` (`deep_scale` in
    /// `sub_c_body.w`).
    pub sub_face_corner: [f32; 4],
    /// x = face index (0..5), y = UVR pre-descent prefix length
    /// (number of valid entries in `sub_uvr_slots`),
    /// z = face-root depth (= `body_path.depth() + 1` — the minimum
    /// prefix length before a neighbor-step bubble-up becomes a
    /// cross-face transition; the shader terminates the sphere-sub
    /// DDA rather than attempting the cross-face math, which is
    /// deferred to a follow-up commit), w = pad.
    pub sub_meta: [u32; 4],
    /// UVR slots the shader walker pre-descends from the face-root
    /// before starting intra-cell DDA. Only `sub_meta.y` entries are
    /// valid; the rest are zero. One u32 per slot keeps the std140
    /// alignment trivial (16-byte stride via `[u32; 4]` rows).
    pub sub_uvr_slots: [[u32; 4]; MAX_SPHERE_SUB_DEPTH / 4],
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
    pub(super) root_radii: [f32; 4],
    pub(super) root_face_meta: [u32; 4],
    pub(super) root_face_bounds: [f32; 4],
    pub(super) root_face_pop_pos: [f32; 4],
    pub(super) sub_c_body: [f32; 4],
    pub(super) sub_j_col0: [f32; 4],
    pub(super) sub_j_col1: [f32; 4],
    pub(super) sub_j_col2: [f32; 4],
    pub(super) sub_j_inv_col0: [f32; 4],
    pub(super) sub_j_inv_col1: [f32; 4],
    pub(super) sub_j_inv_col2: [f32; 4],
    pub(super) sub_face_corner: [f32; 4],
    pub(super) sub_meta: [u32; 4],
    pub(super) sub_uvr_slots: [[u32; 4]; MAX_SPHERE_SUB_DEPTH / 4],
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
    /// Frame root is Cartesian.
    pub fn set_root_kind_cartesian(&mut self) {
        self.root_kind = ROOT_KIND_CARTESIAN;
        self.root_radii = [0.0; 4];
        self.root_face_meta = [0; 4];
        self.root_face_bounds = [0.0; 4];
        self.root_face_pop_pos = [0.0; 4];
        self.clear_sub_fields();
        self.write_uniforms();
    }

    /// Frame root IS a CubedSphereBody cell — shader dispatches the
    /// whole-sphere march.
    pub fn set_root_kind_body(&mut self, inner_r: f32, outer_r: f32) {
        self.root_kind = ROOT_KIND_BODY;
        self.root_radii = [inner_r, outer_r, 0.0, 0.0];
        self.root_face_meta = [0; 4];
        self.root_face_bounds = [0.0; 4];
        self.root_face_pop_pos = [0.0; 4];
        self.clear_sub_fields();
        self.write_uniforms();
    }

    /// Frame root is inside a face subtree — shader dispatches the
    /// bounded face-window march. `bounds = (u_min, v_min, r_min,
    /// size)` in normalized face coords.
    pub fn set_root_kind_face(
        &mut self,
        inner_r: f32,
        outer_r: f32,
        face_id: u32,
        bounds: [f32; 4],
    ) {
        self.root_kind = ROOT_KIND_FACE;
        self.root_radii = [inner_r, outer_r, 0.0, 0.0];
        self.root_face_meta = [face_id, 0, 0, 0];
        self.root_face_bounds = bounds;
        self.root_face_pop_pos = [0.0; 4];
        self.clear_sub_fields();
        self.write_uniforms();
    }

    /// Frame root is a deep face-subtree cell — shader dispatches
    /// the linearized local-frame sphere DDA (`sphere_in_sub_frame`).
    ///
    /// `corner_and_size = (un, vn, rn, frame_size)` in face-normalized
    /// absolute coords. `c_body` is the body-XYZ of local (0,0,0).
    /// `j` is the local→body Jacobian (columns are ∂body/∂{u_l, v_l,
    /// r_l}); `j_inv` is its inverse. `uvr_prefix_slots` carries the
    /// UVR slots the shader walker pre-descends from the face-root
    /// (`set_frame_root` already points `root_index` at the face-root
    /// BFS index). Length is capped at `MAX_SPHERE_SUB_DEPTH`;
    /// callers pass the actual prefix (typically much shorter).
    /// `face_root_depth` = `body_path.depth() + 1` — the minimum UVR
    /// prefix length before a neighbor-step bubble-up becomes a
    /// cross-face transition (terminate threshold for the shader).
    /// `corner_and_size`: xyz = `(un_eval, vn_eval, rn_eval)`
    /// face-normalized corner at the Jacobian's EVAL depth
    /// (`eval_m ≤ MAX_EVAL_M`). w = `eval_frame_size = 1/3^eval_m`.
    ///
    /// `deep_scale = 1/3^(deep_m - eval_m)` — packed into
    /// `sub_c_body.w` so the shader can recover the deep cell size
    /// (`= eval_frame_size * deep_scale`) during hit-cell
    /// reconstruction and neighbor-step corner update. Equals
    /// `1.0` when `deep_m ≤ MAX_EVAL_M` (shallow sub-frame path).
    #[allow(clippy::too_many_arguments)]
    pub fn set_root_kind_sphere_sub(
        &mut self,
        inner_r: f32,
        outer_r: f32,
        face_id: u32,
        corner_and_size: [f32; 4],
        c_body: [f32; 3],
        deep_scale: f32,
        j: [[f32; 3]; 3],
        j_inv: [[f32; 3]; 3],
        uvr_prefix_slots: &[u8],
        face_root_depth: u32,
    ) {
        self.root_kind = ROOT_KIND_SPHERE_SUB;
        self.root_radii = [inner_r, outer_r, 0.0, 0.0];
        self.root_face_meta = [0; 4];
        self.root_face_bounds = [0.0; 4];
        self.root_face_pop_pos = [0.0; 4];
        self.sub_c_body = [c_body[0], c_body[1], c_body[2], deep_scale];
        self.sub_j_col0 = [j[0][0], j[0][1], j[0][2], 0.0];
        self.sub_j_col1 = [j[1][0], j[1][1], j[1][2], 0.0];
        self.sub_j_col2 = [j[2][0], j[2][1], j[2][2], 0.0];
        self.sub_j_inv_col0 = [j_inv[0][0], j_inv[0][1], j_inv[0][2], 0.0];
        self.sub_j_inv_col1 = [j_inv[1][0], j_inv[1][1], j_inv[1][2], 0.0];
        self.sub_j_inv_col2 = [j_inv[2][0], j_inv[2][1], j_inv[2][2], 0.0];
        self.sub_face_corner = corner_and_size;
        let prefix_len = uvr_prefix_slots.len().min(MAX_SPHERE_SUB_DEPTH);
        self.sub_meta = [face_id, prefix_len as u32, face_root_depth, 0];
        let mut slots = [[0u32; 4]; MAX_SPHERE_SUB_DEPTH / 4];
        for (i, &slot) in uvr_prefix_slots.iter().take(prefix_len).enumerate() {
            slots[i / 4][i % 4] = slot as u32;
        }
        self.sub_uvr_slots = slots;
        self.write_uniforms();
    }

    fn clear_sub_fields(&mut self) {
        self.sub_c_body = [0.0; 4];
        self.sub_j_col0 = [0.0; 4];
        self.sub_j_col1 = [0.0; 4];
        self.sub_j_col2 = [0.0; 4];
        self.sub_j_inv_col0 = [0.0; 4];
        self.sub_j_inv_col1 = [0.0; 4];
        self.sub_j_inv_col2 = [0.0; 4];
        self.sub_face_corner = [0.0; 4];
        self.sub_meta = [0; 4];
        self.sub_uvr_slots = [[0; 4]; MAX_SPHERE_SUB_DEPTH / 4];
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
            self::init::create_mask_texture(&self.device, width, height);
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
}
