//! Resolve pipeline construction. One shader module, one pipeline,
//! one bind-group layout. Nothing TAAU-specific leaks into the main
//! renderer init path beyond the single constructor call.
//!
//! The resolve pipeline writes to TWO color attachments at the same
//! resolution but different formats:
//!
//! - `@location(0)` → swapchain (BGRA8UnormSrgb, hardware gamma encode)
//! - `@location(1)` → new history (RGBA16Float, linear)
//!
//! wgpu handles MRT-with-heterogeneous-formats fine; each attachment
//! declares its own `ColorTargetState`.

use crate::shader_compose;

use super::history::HISTORY_FORMAT;

/// Build the resolve `(pipeline, bind_group_layout)` pair. `swapchain_format`
/// must match whatever the surface was configured with — typically an
/// sRGB variant of `Bgra8Unorm` or `Rgba8Unorm`.
pub fn build_resolve_pipeline(
    device: &wgpu::Device,
    swapchain_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("taa_resolve"),
        entries: &[
            // Half-res color (filterable float — we textureLoad, so
            // the `filterable` flag is technically unused here, but
            // keeping it set lets us swap to textureSample later
            // without re-declaring the layout).
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }, count: None,
            },
            // Half-res hit t (R32Float). R32F is not filterable by
            // default; declare as unfilterable so Metal/Vulkan don't
            // complain when the sampler is bound alongside.
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }, count: None,
            },
            // Full-res history (RGBA16F, filterable — we do
            // textureSample with a linear sampler for sub-texel
            // reprojection).
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }, count: None,
            },
            // Linear sampler, bound for the history texture's
            // reprojection sample.
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("taa_resolve"),
        source: wgpu::ShaderSource::Wgsl(shader_compose::compose("taa_resolve.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("taa_resolve"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("taa_resolve"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_resolve"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_resolve"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: HISTORY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bind_group_layout)
}
