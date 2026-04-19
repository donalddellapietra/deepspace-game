//! `Renderer::new()` — one-time wgpu setup (device, surface, buffers,
//! pipeline, bind group). The hot-path upload + draw functions live
//! in `buffers.rs` / `draw.rs`.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuEntity, GpuNodeKind, GpuPalette, GpuRibbonEntry};
use crate::world::tree::MAX_DEPTH;

use super::buffers::make_bind_group;
use super::entity_raster::EntityRasterState;
use super::taa::{TaaState, MARCH_COLOR_FORMAT, MARCH_T_FORMAT};
use super::{
    create_depth_texture, EntityRenderMode, GpuUniforms, Renderer, TimestampScratch,
    DEPTH_FORMAT, ROOT_KIND_CARTESIAN,
};

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        tree: &[u32],
        node_kinds: &[GpuNodeKind],
        node_offsets: &[u32],
        root_bfs_index: u32,
        present_mode: wgpu::PresentMode,
        shader_stats_enabled: bool,
        lod_pixel_threshold: f32,
        lod_base_depth: u32,
        live_sample_every_frames: u32,
        taa_enabled: bool,
        entity_render_mode: EntityRenderMode,
    ) -> Self {
        // Raster entities require a depth handoff from the ray-march
        // pass; the TAA resolve would need to participate, which
        // we're not doing in this iteration. Fail loud so misuse is
        // caught at startup, not debug-hunted at render time.
        assert!(
            !(entity_render_mode == EntityRenderMode::Raster && taa_enabled),
            "--entity-render raster is incompatible with --taa (depth handoff would need half-res adaptation)",
        );
        // On web, winit's `inner_size` lags behind the canvas backing
        // store (request_inner_size doesn't apply synchronously and
        // ensure_started runs before any resize event), so we read the
        // canvas dimensions directly to pick a matching swapchain size.
        #[cfg(target_arch = "wasm32")]
        let size = {
            use winit::platform::web::WindowExtWebSys;
            window
                .canvas()
                .map(|c| winit::dpi::PhysicalSize::new(c.width(), c.height()))
                .unwrap_or_else(|| window.inner_size())
        };
        #[cfg(not(target_arch = "wasm32"))]
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

        let adapter_features = adapter.features();
        let want_timestamp = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if want_timestamp {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        // Storage buffers: tree, node_kinds, ribbon, shader_stats,
        // node_offsets, entities → 6 total. `downlevel_defaults` caps
        // at 4; bump to 8 for headroom.
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 8,
            ..wgpu::Limits::downlevel_defaults()
        }.using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("deepspace"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create GPU device");
        eprintln!(
            "renderer_features timestamp_query_supported={} enabled={}",
            adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY),
            device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        );
        eprintln!(
            "renderer_limits max_storage_buffer_binding_size={}",
            device.limits().max_storage_buffer_binding_size,
        );

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

        // Storage buffers cannot be zero-sized; stub any empty inputs.
        let stub_tree = [0u32, 2u32]; // one empty-node header
        let stub_offsets = [0u32];
        let tree_init: &[u32] = if tree.is_empty() { &stub_tree } else { tree };
        let offsets_init: &[u32] =
            if node_offsets.is_empty() { &stub_offsets } else { node_offsets };

        // Allocate tree / node_kinds / node_offsets with 1.5× headroom
        // past the initial pack size so the first handful of edits
        // patch in-place (via `queue.write_buffer`) without forcing a
        // full buffer recreate + bind-group rebuild.
        let alloc_with_headroom = |device: &wgpu::Device, label: &'static str, bytes: &[u8]| -> wgpu::Buffer {
            // Round UP to a multiple of 4: WebGPU requires storage
            // buffer binding sizes to be a multiple of 4 (and a whole
            // number of u32s for our u32-typed buffers). Metal is
            // looser and silently rounded. The assert traps any
            // future caller whose `T` isn't 4-aligned — would leave a
            // partial-element tail.
            debug_assert!(bytes.len() % 4 == 0, "alloc_with_headroom payload not 4-aligned");
            let min = bytes.len() as u64;
            let raw = (min.max(1) * 3 / 2).max(min + 4096);
            let size = raw.div_ceil(4) * 4;
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&b, 0, bytes);
            b
        };
        let tree_buffer = alloc_with_headroom(
            &device, "tree", bytemuck::cast_slice(tree_init),
        );
        let node_kinds_buffer = alloc_with_headroom(
            &device, "node_kinds", bytemuck::cast_slice(node_kinds),
        );
        let node_offsets_buffer = alloc_with_headroom(
            &device, "node_offsets", bytemuck::cast_slice(offsets_init),
        );

        let camera = GpuCamera {
            pos: [1.5, 1.75, 1.5],
            jitter_x_px: 0.0,
            forward: [0.0, 0.0, -1.0],
            jitter_y_px: 0.0,
            right: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            up: [0.0, 1.0, 0.0],
            fov: 1.2,
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
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

        let node_count = node_kinds.len() as u32;
        let uniforms = GpuUniforms {
            root_index: root_bfs_index,
            node_count,
            screen_width: config.width as f32,
            screen_height: config.height as f32,
            max_depth: MAX_DEPTH as u32,
            highlight_active: 0,
            root_kind: ROOT_KIND_CARTESIAN,
            ribbon_count: 0,
            entity_count: 0,
            _pad_entity: [0; 3],
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
            contents: bytemuck::cast_slice(&[GpuRibbonEntry { node_idx: 0, slot_bits: 0 }]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Entity buffer — same stub trick. `entity_count` in the
        // uniforms gates the shader's iteration, so the stub entry
        // is never read.
        let entity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entities"),
            contents: bytemuck::cast_slice(&[GpuEntity::default()]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shader_stats"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shader_stats_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shader_stats_readback"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // Node offsets (binding 7) — BFS index → u32-offset
                // of that node's header in `tree[]`. Cold path only
                // (touched on descent / ribbon pop).
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // Entities (binding 8) — flat `array<GpuEntity>` of
                // bounding-cube + subtree-BFS instances.
                wgpu::BindGroupLayoutEntry {
                    binding: 8, visibility: wgpu::ShaderStages::FRAGMENT,
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
            &shader_stats_buffer, &node_offsets_buffer,
            &entity_buffer,
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

        let lod_pixels_const = lod_pixel_threshold as f64;
        let enable_stats_const = if shader_stats_enabled { 1.0 } else { 0.0 };
        let base_detail_depth_const = lod_base_depth as f64;
        let override_constants: [(&str, f64); 3] = [
            ("ENABLE_STATS", enable_stats_const),
            ("LOD_PIXEL_THRESHOLD", lod_pixels_const),
            ("BASE_DETAIL_DEPTH", base_detail_depth_const),
        ];
        let frag_compilation_options = wgpu::PipelineCompilationOptions {
            constants: &override_constants,
            zero_initialize_workgroup_memory: false,
        };
        eprintln!(
            "renderer_pipeline lod_pixels={:.2} lod_base_depth={} shader_stats={}",
            lod_pixel_threshold, lod_base_depth, shader_stats_enabled,
        );
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
                compilation_options: frag_compilation_options.clone(),
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

        // When raster entities are enabled, compile a ray-march
        // pipeline variant with a Depth32Float attachment and the
        // `fs_main_depth` fragment entry point, which writes
        // `@builtin(frag_depth)` derived from `camera.view_proj`.
        // The subsequent entity raster pass z-tests against that
        // depth buffer. Keeping this as a separate pipeline (rather
        // than overriding `depth_stencil: None` on the default one)
        // means ray-march-only runs pay zero depth-write cost.
        let (pipeline_with_depth, depth_texture, depth_view, entity_raster) =
            if matches!(entity_render_mode, EntityRenderMode::Raster) {
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("ray_march_with_depth"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main_depth"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: frag_compilation_options.clone(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Always,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });
                let (tex, view) = create_depth_texture(&device, config.width, config.height);
                let raster = EntityRasterState::new(&device, config.format, DEPTH_FORMAT);
                (Some(pipeline), Some(tex), Some(view), Some(raster))
            } else {
                (None, None, None, None)
            };

        // When TAAU is enabled, compile a second ray-march pipeline
        // with the `fs_main_taa` entry point — two color attachments
        // (linear RGBA16F color + R32F hit t) at half-res. Both
        // pipelines share the same bind group layout so buffer state
        // is identical between them; only the fragment-output shape
        // and target formats differ.
        let (pipeline_taa, taa_state) = if taa_enabled {
            let pipeline_taa = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ray_march_taa"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main_taa"),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: MARCH_COLOR_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: MARCH_T_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::RED,
                        }),
                    ],
                    compilation_options: frag_compilation_options,
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
            let taa_state = TaaState::new(&device, config.format, config.width, config.height);
            eprintln!(
                "renderer_taa enabled scaled_size={}x{} full_size={}x{}",
                taa_state.scaled_width, taa_state.scaled_height,
                taa_state.full_width, taa_state.full_height,
            );
            (Some(pipeline_taa), Some(taa_state))
        } else {
            (None, None)
        };

        let timestamp = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("ray_march_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_staging"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let period_ns = queue.get_timestamp_period();
            Some(TimestampScratch { query_set, resolve, staging, period_ns })
        } else {
            None
        };

        let uploaded_tree_u32s = tree.len() as u64;
        let uploaded_kinds_count = node_kinds.len() as u64;
        let uploaded_offsets_count = node_offsets.len() as u64;
        Self {
            device, queue, surface, config, pipeline, bind_group_layout,
            tree_buffer, node_offsets_buffer, node_kinds_buffer,
            uploaded_tree_u32s,
            uploaded_kinds_count,
            uploaded_offsets_count,
            camera_buffer,
            last_camera: camera,
            palette_buffer, uniforms_buffer,
            ribbon_buffer,
            entity_buffer,
            uploaded_entities_count: 0,
            entity_count: 0,
            bind_group,
            root_index: root_bfs_index, node_count,
            max_depth: MAX_DEPTH as u32, highlight_active: 0,
            highlight_min: [0.0; 4], highlight_max: [0.0; 4],
            root_kind: ROOT_KIND_CARTESIAN,
            root_radii: [0.0; 4],
            root_face_meta: [0; 4],
            root_face_bounds: [0.0; 4],
            root_face_pop_pos: [0.0; 4],
            ribbon_count: 0,
            offscreen_texture: None,
            pipeline_taa,
            taa: taa_state,
            timestamp,
            last_camera_write_ms: 0.0,
            last_ribbon_write_ms: 0.0,
            last_tree_write_ms: 0.0,
            last_bind_group_rebuild_ms: 0.0,
            shader_stats_buffer,
            shader_stats_readback,
            shader_stats_enabled,
            live_frame_counter: 0,
            live_sample_every_frames,
            entity_render_mode,
            depth_texture,
            depth_view,
            pipeline_with_depth,
            entity_raster,
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
