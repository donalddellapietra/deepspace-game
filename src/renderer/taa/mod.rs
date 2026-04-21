//! TAAU: Temporal Anti-Aliasing Upscaling.
//!
//! When enabled, the ray-march pipeline runs at ½ per-axis (¼ pixel
//! count) into dedicated color + hit-t render targets, and a resolve
//! pass reprojects the previous frame's history into the current
//! frame's pixel grid, neighborhood-clamps it against the new half-
//! res samples, and blends. Camera projection is sub-pixel-jittered
//! by a Halton(2, 3) sequence so successive frames sample distinct
//! sub-pixel positions; after ~4 frames of accumulation the history
//! effectively supersamples the voxel scene.
//!
//! Net result: full-resolution crispness at half-res ray-march cost,
//! without the bilinear-blur of the prior `--render-scale` path.
//!
//! # Module layout
//!
//! - [`jitter`] — Halton sequence, per-frame sub-pixel offsets.
//! - [`history`] — Ping-pong RGBA16Float textures + validity tracking.
//! - [`pipeline`] — Resolve render pipeline construction.
//! - This module — [`TaaState`] ties it together: per-frame uniform
//!   upload, history swap, previous-camera tracking.
//!
//! The resolve shader lives in `assets/shaders/taa_resolve.wgsl`.
//!
//! # Lifecycle
//!
//! 1. `TaaState::new` — called once from `Renderer::new` when
//!    `--taa` is set. Creates pipelines, render targets, history.
//! 2. Every frame:
//!    a. [`begin_frame`] — pick new Halton sample, write the
//!       camera jitter into the GPU camera buffer, prepare resolve
//!       uniforms (current + previous camera, validity flag).
//!    b. Ray-march pass uses the `march_taa` pipeline to write into
//!       [`color_target`] + [`t_target`] at half-res.
//!    c. Resolve pass reads those + history, writes to swapchain +
//!       new history (MRT).
//!    d. [`end_frame`] — swap history textures, remember camera for
//!       next frame's reprojection.
//! 3. `invalidate_history` — called by the renderer whenever the
//!    frame-root or ribbon changes, because the CURRENT camera is
//!    suddenly in a different coordinate system than the PREVIOUS one
//!    and reprojection would sample garbage.

pub mod history;
pub mod jitter;
pub mod pipeline;

use crate::world::gpu::GpuCamera;

use history::HistoryPair;
use jitter::{jitter_offset, JITTER_COUNT};

/// Hard-coded downscale factor. 2 = quarter-pixel ray-march cost;
/// higher is possible but degrades quality faster than perf gains.
/// Changing this requires invalidating history.
pub const RENDER_SCALE: u32 = 2;

/// Format of the half-res color render target. Linear HDR so the
/// resolve clamp math is well-behaved; the ray-march shader's
/// gamma-corrected output still fits (values in `[0, 1]` after the
/// `pow(1/2.2)`), and RGBA16Float is filterable on every target.
pub const MARCH_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Format of the hit-t render target. R32Float fits the ray-space
/// distance without precision loss at any depth.
pub const MARCH_T_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Blend weight for current → history mix in the resolve shader.
/// 0.1 = 10% new / 90% history. Lower = smoother but more motion
/// lag; higher = more responsive but less detail accumulation.
pub const DEFAULT_BLEND_WEIGHT: f32 = 0.1;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TaaUniforms {
    pub cam_pos: [f32; 3],
    pub _pad0: f32,
    pub cam_forward: [f32; 3],
    pub _pad1: f32,
    pub cam_right: [f32; 3],
    pub _pad2: f32,
    pub cam_up: [f32; 3],
    pub cam_fov: f32,

    pub prev_cam_pos: [f32; 3],
    pub _pad3: f32,
    pub prev_cam_forward: [f32; 3],
    pub _pad4: f32,
    pub prev_cam_right: [f32; 3],
    pub _pad5: f32,
    pub prev_cam_up: [f32; 3],
    pub prev_cam_fov: f32,

    pub scaled_size: [f32; 2],
    pub full_size: [f32; 2],

    pub blend_weight: f32,
    pub history_valid: u32,
    pub _pad6: [f32; 2],
}

/// A stable-over-same-frame identifier for the render frame root.
/// If any field changes between frames, reprojection is unsafe —
/// the current `camera.pos` is in a different local [0, 3)³ than
/// the previous frame's, and `hit_pos_prev_frame ≠ hit_pos_current_frame`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FrameSignature {
    pub root_index: u32,
    /// Captures the sphere-remap mode (0 = Cartesian shading, 1 =
    /// sphere). A mode change invalidates TAA history just like a
    /// frame-root swap: lighting normals differ, so blending against
    /// last frame is unsafe.
    pub sphere_flag: u32,
    pub ribbon_count: u32,
}

/// Per-frame jitter counter. One tick per frame; wraps inside
/// [`JITTER_COUNT`] so the Halton sequence cycles cleanly.
pub struct JitterState {
    pub frame_index: u32,
    pub current: [f32; 2],
}

impl JitterState {
    pub fn new() -> Self { Self { frame_index: 0, current: [0.0, 0.0] } }

    /// Advance to the next Halton offset and return it. The caller
    /// is responsible for writing this into the camera buffer before
    /// the ray-march runs.
    pub fn tick(&mut self) -> [f32; 2] {
        self.current = jitter_offset(self.frame_index);
        self.frame_index = self.frame_index.wrapping_add(1);
        if self.frame_index >= JITTER_COUNT * 1024 {
            self.frame_index %= JITTER_COUNT;
        }
        self.current
    }
}

/// Full state owned by `Renderer` when `--taa` is enabled. Dropping
/// disables TAAU; the renderer falls back to the non-TAA path.
pub struct TaaState {
    pub full_width: u32,
    pub full_height: u32,
    pub scaled_width: u32,
    pub scaled_height: u32,

    // Render targets written by the TAA ray-march pipeline.
    pub color_target: wgpu::Texture,
    pub color_target_view: wgpu::TextureView,
    pub t_target: wgpu::Texture,
    pub t_target_view: wgpu::TextureView,

    // Resolve pipeline + its bind group layout and sampler.
    pub resolve_pipeline: wgpu::RenderPipeline,
    pub resolve_bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,

    // Uniforms uploaded each frame.
    pub uniforms_buffer: wgpu::Buffer,

    pub history: HistoryPair,
    pub jitter: JitterState,

    /// Camera used to render the PREVIOUS frame (for reprojection).
    /// `None` until the first frame has been rendered; treated as
    /// "history invalid" in that case.
    pub prev_camera: Option<GpuCamera>,
    /// Frame-root signature for the previous frame. Compared to
    /// current each frame — a mismatch forces history invalidation.
    pub prev_signature: FrameSignature,
}

impl TaaState {
    /// Build TaaState for the given swapchain format + full size.
    /// Half-res targets get `ceil(full/2)` so oddd resolutions don't
    /// drop a pixel row.
    pub fn new(
        device: &wgpu::Device,
        swapchain_format: wgpu::TextureFormat,
        full_width: u32,
        full_height: u32,
    ) -> Self {
        let (scaled_width, scaled_height) = scaled_dims(full_width, full_height);

        let (color_target, color_target_view) =
            make_march_target(device, scaled_width, scaled_height, MARCH_COLOR_FORMAT, "march_color");
        let (t_target, t_target_view) =
            make_march_target(device, scaled_width, scaled_height, MARCH_T_FORMAT, "march_t");

        let (resolve_pipeline, resolve_bind_group_layout) = pipeline::build_resolve_pipeline(
            device, swapchain_format,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa_uniforms"),
            size: std::mem::size_of::<TaaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let history = HistoryPair::new(device, full_width, full_height);

        Self {
            full_width, full_height,
            scaled_width, scaled_height,
            color_target, color_target_view,
            t_target, t_target_view,
            resolve_pipeline, resolve_bind_group_layout,
            sampler,
            uniforms_buffer,
            history,
            jitter: JitterState::new(),
            prev_camera: None,
            prev_signature: FrameSignature::default(),
        }
    }

    /// Re-allocate render targets + history for a new window size.
    /// Implicitly invalidates history.
    pub fn resize(&mut self, device: &wgpu::Device, full_width: u32, full_height: u32) {
        let (scaled_width, scaled_height) = scaled_dims(full_width, full_height);
        if full_width == self.full_width
            && full_height == self.full_height
            && scaled_width == self.scaled_width
            && scaled_height == self.scaled_height
        {
            return;
        }
        self.full_width = full_width;
        self.full_height = full_height;
        self.scaled_width = scaled_width;
        self.scaled_height = scaled_height;

        let (ct, ctv) =
            make_march_target(device, scaled_width, scaled_height, MARCH_COLOR_FORMAT, "march_color");
        self.color_target = ct;
        self.color_target_view = ctv;
        let (tt, ttv) =
            make_march_target(device, scaled_width, scaled_height, MARCH_T_FORMAT, "march_t");
        self.t_target = tt;
        self.t_target_view = ttv;

        self.history.resize(device, full_width, full_height);
        self.prev_camera = None; // force warmup on the new size.
    }

    /// Mark the next frame's resolve as "can't use history". Renderer
    /// calls this on any operation that breaks the prev↔current
    /// coordinate-system invariant.
    pub fn invalidate_history(&mut self) {
        self.history.invalidate();
        self.prev_camera = None;
    }

    /// Per-frame setup: tick the jitter, build + upload the TAA
    /// uniforms for this frame. Must be called after the caller has
    /// finalized the CURRENT camera but before the ray-march is
    /// encoded. Returns the jitter offset so the caller can stamp it
    /// into the camera uniform before upload.
    pub fn begin_frame(
        &mut self,
        queue: &wgpu::Queue,
        current_camera: &GpuCamera,
        current_signature: FrameSignature,
    ) -> [f32; 2] {
        let jitter = self.jitter.tick();

        // Invalidate history when the frame-root shifted — the
        // current camera's coordinate system has changed and prev
        // data is no longer comparable.
        if current_signature != self.prev_signature {
            self.history.invalidate();
            self.prev_camera = None;
        }

        let history_valid =
            if self.history.is_valid() && self.prev_camera.is_some() { 1u32 } else { 0u32 };
        let (prev_cam, prev_fov) = self
            .prev_camera
            .map(|c| (c, c.fov))
            .unwrap_or((*current_camera, current_camera.fov));

        let uniforms = TaaUniforms {
            cam_pos: current_camera.pos,
            _pad0: 0.0,
            cam_forward: current_camera.forward,
            _pad1: 0.0,
            cam_right: current_camera.right,
            _pad2: 0.0,
            cam_up: current_camera.up,
            cam_fov: current_camera.fov,

            prev_cam_pos: prev_cam.pos,
            _pad3: 0.0,
            prev_cam_forward: prev_cam.forward,
            _pad4: 0.0,
            prev_cam_right: prev_cam.right,
            _pad5: 0.0,
            prev_cam_up: prev_cam.up,
            prev_cam_fov: prev_fov,

            scaled_size: [self.scaled_width as f32, self.scaled_height as f32],
            full_size: [self.full_width as f32, self.full_height as f32],

            blend_weight: DEFAULT_BLEND_WEIGHT,
            history_valid,
            _pad6: [0.0, 0.0],
        };
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Stash the current frame's state so next frame's `begin_frame`
        // can supply it as `prev_*`. We capture AFTER uniforms are
        // built so the pre-jittered camera (which the march uses) gets
        // stamped — reprojection uses these same jittered coords.
        self.prev_camera = Some(*current_camera);
        self.prev_signature = current_signature;

        jitter
    }

    /// Build a bind group for the resolve pass bound to the current
    /// render targets and history textures. Cheap to rebuild each
    /// frame — wgpu bind groups are thin views over pre-existing
    /// resources.
    pub fn make_resolve_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa_resolve"),
            layout: &self.resolve_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.color_target_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.t_target_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(self.history.read_view()) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: self.uniforms_buffer.as_entire_binding() },
            ],
        })
    }

    /// Called once per frame AFTER the resolve pass has been recorded
    /// into the encoder. Ping-pongs history, ticks warmup counter.
    pub fn end_frame(&mut self) {
        self.history.end_frame();
    }
}

/// Divide `n` by [`RENDER_SCALE`] rounding up, with a one-pixel floor
/// so degenerate (0, _) sizes don't produce zero-sized textures.
fn scaled_dims(full_width: u32, full_height: u32) -> (u32, u32) {
    let w = (full_width + RENDER_SCALE - 1) / RENDER_SCALE;
    let h = (full_height + RENDER_SCALE - 1) / RENDER_SCALE;
    (w.max(1), h.max(1))
}

fn make_march_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &'static str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_dims_rounds_up() {
        assert_eq!(scaled_dims(1280, 720), (640, 360));
        assert_eq!(scaled_dims(1281, 721), (641, 361)); // ceil
        assert_eq!(scaled_dims(1, 1), (1, 1));
        assert_eq!(scaled_dims(0, 0), (1, 1)); // floor
    }

    #[test]
    fn taa_uniforms_size_is_multiple_of_16() {
        // WGSL uniform buffers require 16-byte alignment.
        let s = std::mem::size_of::<TaaUniforms>();
        assert_eq!(s % 16, 0, "TaaUniforms size {s} not 16-byte aligned");
    }

    #[test]
    fn signature_equality() {
        let a = FrameSignature { root_index: 0, sphere_flag: 0, ribbon_count: 0 };
        let b = FrameSignature { root_index: 0, sphere_flag: 0, ribbon_count: 0 };
        assert_eq!(a, b);
        let c = FrameSignature { root_index: 1, ..a };
        assert_ne!(a, c);
    }

    #[test]
    fn jitter_state_ticks_deterministically() {
        let mut j = JitterState::new();
        let first = j.tick();
        let second = j.tick();
        assert_ne!(first, second);
    }
}
