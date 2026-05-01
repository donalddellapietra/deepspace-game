//! `Renderer::new()` — one-time wgpu setup (device, surface, buffers,
//! pipeline, bind group). The hot-path upload + draw functions live
//! in `buffers.rs` / `draw.rs`.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuEntity, GpuNodeKind, GpuRibbonEntry};
use crate::world::tree::MAX_DEPTH;

use super::buffers::make_bind_group;
use super::entity_raster::EntityRasterState;
use super::taa::{TaaState, MARCH_COLOR_FORMAT, MARCH_T_FORMAT};
use super::{GpuUniforms, Renderer, ROOT_KIND_CARTESIAN};

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

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        tree: &[u32],
        node_kinds: &[GpuNodeKind],
        node_offsets: &[u32],
        aabbs: &[u32],
        root_bfs_index: u32,
        present_mode: wgpu::PresentMode,
        shader_stats_enabled: bool,
        lod_pixel_threshold: f32,
        live_sample_every_frames: u32,
        taa_enabled: bool,
        entities_enabled: bool,
        entity_render_mode: crate::renderer::EntityRenderMode,
    ) -> Self {
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

        let required_features = wgpu::Features::empty();
        // Sparse tree needs 5 storage buffers (tree, node_kinds,
        // ribbon, shader_stats, node_offsets). `downlevel_defaults`
        // caps that at 4; bump to 8 (the WebGPU spec default) so
        // the limit is portable to the browser backend too.
        let required_limits = wgpu::Limits {
            // tree + palette + node_kinds + ribbon + shader_stats +
            // node_offsets + aabbs + entities = 8 storage buffers
            // minimum for the ray-march pipeline. Bump to 10 to
            // leave slack for compute passes (heightmap / physics)
            // that reuse this device.
            max_storage_buffers_per_shader_stage: 10,
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
        // aabbs is parallel to node_offsets: stub a single entry for
        // the empty-pack case so the binding stays valid.
        let stub_aabbs = [0u32];
        let aabbs_init: &[u32] = if aabbs.is_empty() { &stub_aabbs } else { aabbs };
        let aabbs_buffer = alloc_with_headroom(
            &device, "aabbs", bytemuck::cast_slice(aabbs_init),
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

        // Seed the palette buffer with the builtin colors. Later
        // uploads via `Renderer::update_palette` may grow the buffer
        // if an imported scene's palette exceeds the initial capacity.
        let builtin_palette: Vec<[f32; 4]> = crate::world::palette::BUILTINS
            .iter()
            .map(|&(_, _, c)| c)
            .collect();
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette"),
            contents: bytemuck::cast_slice(&builtin_palette),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            curvature: [0.0; 4],
            slab_dims: [0; 4],
            _pad_face_bounds: [0.0; 4],
            _pad_face_pop_pos: [0.0; 4],
        };

        // Initial ribbon buffer is empty (just a stub of zero
        // entries; storage buffers can't be zero-sized so we
        // allocate one stub entry).
        let ribbon_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ribbon"),
            contents: bytemuck::cast_slice(&[GpuRibbonEntry { node_idx: 0, slot_bits: 0 }]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Entity buffer (binding 10) — one-entry stub so the
        // storage binding isn't zero-sized. `entity_count` on the
        // uniforms stays 0 until entities spawn, so the shader's
        // tag=3 dispatch never reads this stub.
        let entity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entities"),
            contents: bytemuck::cast_slice(&[GpuEntity::default()]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                // Coarse beam-prepass mask (binding 8) — R8Unorm
                // texture populated by the coarse pipeline. Fine
                // pass reads a 5-tap neighborhood per pixel and
                // early-outs to sky when the tile is definitively
                // empty. Sample_type float, not filterable — we
                // use textureLoad, not textureSample.
                wgpu::BindGroupLayoutEntry {
                    binding: 8, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // Content AABBs (binding 9) — per-BFS 12-bit AABB in
                // the low 12 bits of each u32. Parallel to
                // `node_offsets`. Used by the ray-march descent cull.
                wgpu::BindGroupLayoutEntry {
                    binding: 9, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // Entities (binding 10) — flat `array<EntityGpu>` of
                // per-instance bounding-cube + subtree-BFS records.
                // Shader's tag=3 dispatch uses it when the ray hits
                // an `EntityRef(idx)` child cell.
                wgpu::BindGroupLayoutEntry {
                    binding: 10, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        // Beam-prepass mask texture. R8Unorm at 1/BEAM_TILE_SIZE per
        // axis — stores `fs_coarse_mask`'s per-tile hit flag. Resized
        // on window resize via `Renderer::resize`.
        let (mask_texture, mask_view) = create_mask_texture(
            &device, config.width, config.height,
        );
        // 1×1 dummy mask for the coarse bind group. The coarse shader
        // doesn't sample `coarse_mask` — it writes to the real one as
        // a render target. But the bind group layout requires
        // something at slot 8.
        let dummy_mask_view = create_dummy_mask_view(&device);

        let bind_group = make_bind_group(
            &device, &bind_group_layout,
            &tree_buffer, &camera_buffer, &palette_buffer,
            &uniforms_buffer, &node_kinds_buffer, &ribbon_buffer,
            &shader_stats_buffer, &node_offsets_buffer,
            &aabbs_buffer,
            &mask_view,
            &entity_buffer,
        );
        let coarse_bind_group = make_bind_group(
            &device, &bind_group_layout,
            &tree_buffer, &camera_buffer, &palette_buffer,
            &uniforms_buffer, &node_kinds_buffer, &ribbon_buffer,
            &shader_stats_buffer, &node_offsets_buffer,
            &aabbs_buffer,
            &dummy_mask_view,
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
        let enable_entities_const = if entities_enabled { 1.0 } else { 0.0 };
        let override_constants: [(&str, f64); 3] = [
            ("ENABLE_STATS", enable_stats_const),
            ("LOD_PIXEL_THRESHOLD", lod_pixels_const),
            ("ENABLE_ENTITIES", enable_entities_const),
        ];
        let frag_compilation_options = wgpu::PipelineCompilationOptions {
            constants: &override_constants,
            zero_initialize_workgroup_memory: false,
        };
        eprintln!(
            "renderer_pipeline lod_pixels={:.2} shader_stats={}",
            lod_pixel_threshold, shader_stats_enabled,
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

        // Beam-prepass coarse pipeline. Same pipeline layout, same
        // shader module, different fragment entry (fs_coarse_mask)
        // and render target (R8Unorm mask texture instead of the
        // swapchain). Writes per-tile hit flags so the fine pass can
        // skip sky tiles.
        let coarse_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ray_march_coarse"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_coarse_mask"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
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

        // Raster entity mode: build the depth-writing ray-march
        // pipeline (fs_main_depth) + depth texture + EntityRasterState
        // + heightmap compute pipelines. Compiling the depth pipeline
        // as a separate resource (rather than overriding the default)
        // means ray-march-only runs pay zero depth-write cost.
        let (pipeline_with_depth, depth_texture_opt, depth_view_opt, entity_raster, heightmap_gen, entity_heightmap_clamp) =
            if matches!(entity_render_mode, crate::renderer::EntityRenderMode::Raster) {
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
                        format: crate::renderer::DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Always,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });
                let (tex, view) = crate::renderer::create_depth_texture(
                    &device, config.width, config.height,
                );
                let raster = EntityRasterState::new(
                    &device, config.format, crate::renderer::DEPTH_FORMAT,
                );
                let hgen = super::heightmap::HeightmapGen::new(&device);
                let hclamp = super::heightmap::EntityHeightmapClamp::new(&device);
                (Some(pipeline), Some(tex), Some(view), Some(raster), Some(hgen), Some(hclamp))
            } else {
                (None, None, None, None, None, None)
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

        let uploaded_tree_u32s = tree.len() as u64;
        let uploaded_kinds_count = node_kinds.len() as u64;
        let uploaded_offsets_count = node_offsets.len() as u64;
        let uploaded_aabbs_count = aabbs.len() as u64;
        Self {
            device, queue, surface, config, pipeline, bind_group_layout,
            tree_buffer, node_offsets_buffer, aabbs_buffer, node_kinds_buffer,
            uploaded_tree_u32s,
            uploaded_kinds_count,
            uploaded_offsets_count,
            uploaded_aabbs_count,
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
            slab_dims: [0; 4],
            curvature: [0.0; 4],
            ribbon_count: 0,
            offscreen_texture: None,
            mask_texture,
            mask_view,
            dummy_mask_view,
            coarse_pipeline,
            coarse_bind_group,
            beam_enabled: true,
            pipeline_taa,
            taa: taa_state,
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
            depth_texture: depth_texture_opt,
            depth_view: depth_view_opt,
            pipeline_with_depth,
            entity_raster,
            heightmap_gen,
            entity_heightmap_clamp,
            heightmap_texture: None,
            heightmap_dirty: true,
            heightmap_frame_root_bfs: u32::MAX,
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
