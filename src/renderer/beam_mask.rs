//! Beam-prepass mask: per-tile R8Unorm render target the coarse
//! pipeline writes to and the fine pass samples for early-out. The
//! shader's `BEAM_TILE_SIZE` constant must agree with the value
//! defined here.

/// Beam-prepass mask format. R8Unorm reads as f32 in the shader, and
/// the coarse fragment writes only 0.0 or 1.0 so quantisation is a
/// non-issue.
pub(super) const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Coarse-pass tile size in output pixels. MUST match `BEAM_TILE_SIZE`
/// in `bindings.wgsl`. Changes require rebuilding the shader module
/// (the const is compiled in, not an override).
pub(super) const BEAM_TILE_SIZE: u32 = 8;

pub(super) fn create_mask_texture(
    device: &wgpu::Device,
    swap_w: u32,
    swap_h: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    // Round up so edge tiles always exist; a non-aligned swapchain
    // size would otherwise sample one-past-end on the right/bottom
    // edges and read 0 (= sky) every time, dropping content.
    let w = (swap_w.max(1) + BEAM_TILE_SIZE - 1) / BEAM_TILE_SIZE;
    let h = (swap_h.max(1) + BEAM_TILE_SIZE - 1) / BEAM_TILE_SIZE;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("beam_mask"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MASK_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

pub(super) fn create_dummy_mask_view(device: &wgpu::Device) -> wgpu::TextureView {
    // 1×1 R8Unorm texture with any contents. The coarse bind group
    // slots it in at binding 8 — the coarse shader doesn't sample
    // `coarse_mask`, so the contents don't matter, only that SOME
    // texture is bound to satisfy the layout.
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("beam_mask_dummy"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MASK_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Pick the best present mode the surface supports, preferring the
/// caller's request and falling back to reasonable alternatives.
pub(super) fn select_present_mode(
    surface_caps: &wgpu::SurfaceCapabilities,
    requested: wgpu::PresentMode,
) -> wgpu::PresentMode {
    if surface_caps.present_modes.contains(&requested) {
        return requested;
    }
    if matches!(requested, wgpu::PresentMode::AutoNoVsync) {
        for candidate in [
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::FifoRelaxed,
        ] {
            if surface_caps.present_modes.contains(&candidate) {
                return candidate;
            }
        }
    }
    if matches!(requested, wgpu::PresentMode::AutoVsync) {
        for candidate in [wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox] {
            if surface_caps.present_modes.contains(&candidate) {
                return candidate;
            }
        }
    }
    requested
}
