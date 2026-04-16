//! `Renderer::new()` — one-time wgpu setup (device, surface, buffers,
//! pipeline, bind group). The hot-path upload + draw functions live
//! in `buffers.rs` / `draw.rs`.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuChild, GpuNodeKind, GpuPalette, GpuRibbonEntry};
use crate::world::tree::MAX_DEPTH;

use super::buffers::make_bind_group;
use super::{GpuUniforms, Renderer, ROOT_KIND_CARTESIAN};

impl Renderer {
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        tree_data: &[GpuChild],
        node_kinds: &[GpuNodeKind],
        root_index: u32,
        present_mode: wgpu::PresentMode,
    ) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("deepspace"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create GPU device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let requested_present_mode = present_mode;
        let present_mode = select_present_mode(&surface_caps, requested_present_mode);
        eprintln!(
            "renderer_present requested={requested:?} selected={selected:?} supported={supported:?}",
            requested = requested_present_mode,
            selected = present_mode,
            supported = surface_caps.present_modes,
        );
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let tree_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tree"),
            contents: bytemuck::cast_slice(tree_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let node_kinds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node_kinds"),
            contents: bytemuck::cast_slice(node_kinds),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let camera = GpuCamera {
            pos: [1.5, 1.75, 1.5],
            _pad0: 0.0,
            forward: [0.0, 0.0, -1.0],
            _pad1: 0.0,
            right: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            up: [0.0, 1.0, 0.0],
            fov: 1.2,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&camera),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let palette = GpuPalette::default();
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette"),
            contents: bytemuck::bytes_of(&palette),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let node_count = (tree_data.len() / 27) as u32;
        let uniforms = GpuUniforms {
            root_index,
            node_count,
            screen_width: config.width as f32,
            screen_height: config.height as f32,
            max_depth: MAX_DEPTH as u32,
            highlight_active: 0,
            root_kind: ROOT_KIND_CARTESIAN,
            ribbon_count: 0,
            highlight_min: [0.0; 4],
            highlight_max: [0.0; 4],
            root_radii: [0.0; 4],
            root_face_meta: [0; 4],
            root_face_bounds: [0.0; 4],
            root_face_pop_pos: [0.0; 4],
        };

        // Initial ribbon buffer is empty (just a stub of zero
        // entries; storage buffers can't be zero-sized so we
        // allocate one stub entry).
        let ribbon_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ribbon"),
            contents: bytemuck::cast_slice(&[GpuRibbonEntry { node_idx: 0, slot: 0 }]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ray_march"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // Ancestor ribbon (binding 5).
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let bind_group = make_bind_group(
            &device, &bind_group_layout,
            &tree_buffer, &camera_buffer, &palette_buffer,
            &uniforms_buffer, &node_kinds_buffer, &ribbon_buffer,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_march"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shader_compose::compose("main.wgsl").into()
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ray_march"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ray_march"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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

        Self {
            device, queue, surface, config, pipeline, bind_group_layout,
            tree_buffer, node_kinds_buffer,
            camera_buffer, palette_buffer, uniforms_buffer,
            ribbon_buffer,
            bind_group,
            root_index, node_count,
            max_depth: MAX_DEPTH as u32, highlight_active: 0,
            highlight_min: [0.0; 4], highlight_max: [0.0; 4],
            root_kind: ROOT_KIND_CARTESIAN,
            root_radii: [0.0; 4],
            root_face_meta: [0; 4],
            root_face_bounds: [0.0; 4],
            root_face_pop_pos: [0.0; 4],
            ribbon_count: 0,
            offscreen_texture: None,
        }
    }
}

/// Pick the best present mode the surface supports, preferring the
/// caller's request and falling back to reasonable alternatives.
fn select_present_mode(
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
