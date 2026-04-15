//! wgpu renderer: full-screen ray march shader.
//!
//! Five buffers: tree (per-child), node_kinds (per-node), camera,
//! palette, uniforms. The shader walks the unified tree and
//! dispatches on `NodeKind` when descending — there are no parallel
//! buffers for sphere content, no `cs_*` uniforms, and no absolute-
//! coord shimming.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuChild, GpuNodeKind, GpuPalette, GpuRibbonEntry};
use crate::world::tree::MAX_DEPTH;

/// Maximum ancestor-ribbon depth supported by the shader. Larger
/// ribbons get truncated at upload (anything beyond can't pop).
/// 64 covers MAX_DEPTH=63 with one slack.
pub const MAX_RIBBON_LEN: usize = 64;

/// `root_kind` discriminant — must mirror the WGSL `RootKind*`
/// constants in `ray_march.wgsl`.
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tree_buffer: wgpu::Buffer,
    node_kinds_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    palette_buffer: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    ribbon_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    root_index: u32,
    node_count: u32,
    max_depth: u32,
    highlight_active: u32,
    highlight_min: [f32; 4],
    highlight_max: [f32; 4],
    root_kind: u32,
    root_radii: [f32; 4],
    root_face_meta: [u32; 4],
    root_face_bounds: [f32; 4],
    root_face_pop_pos: [f32; 4],
    ribbon_count: u32,
}

impl Renderer {
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        tree_data: &[GpuChild],
        node_kinds: &[GpuNodeKind],
        root_index: u32,
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

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
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
            world_pos: [1.5, 1.75, 1.5],
            _pad_world: 0.0,
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
                include_str!("../assets/shaders/ray_march.wgsl").into()
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
        }
    }

    /// Upload the ancestor ribbon (pop chain from frame's direct
    /// parent up to the absolute root). Resizes the ribbon buffer
    /// if needed and recreates the bind group.
    pub fn update_ribbon(&mut self, ribbon: &[GpuRibbonEntry]) {
        let truncated = if ribbon.len() > MAX_RIBBON_LEN {
            &ribbon[..MAX_RIBBON_LEN]
        } else {
            ribbon
        };
        // Always upload at least one entry — empty storage buffers
        // break the bind group.
        let stub_storage = [GpuRibbonEntry { node_idx: 0, slot: 0 }];
        let payload: &[GpuRibbonEntry] = if truncated.is_empty() {
            &stub_storage
        } else {
            truncated
        };

        self.ribbon_count = truncated.len() as u32;
        let needed = (payload.len() * std::mem::size_of::<GpuRibbonEntry>()) as u64;

        let mut recreate_bind_group = false;
        if needed > self.ribbon_buffer.size() {
            self.ribbon_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ribbon"),
                contents: bytemuck::cast_slice(payload),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            recreate_bind_group = true;
        } else {
            self.queue.write_buffer(&self.ribbon_buffer, 0, bytemuck::cast_slice(payload));
        }
        if recreate_bind_group {
            self.bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
            );
        }
        self.write_uniforms();
    }

    /// Set the frame-root NodeKind: Cartesian (default) or
    /// CubedSphereBody. For Body, also pass radii in the body
    /// cell's local `[0, 1)` frame.
    pub fn set_root_kind_cartesian(&mut self) {
        self.root_kind = ROOT_KIND_CARTESIAN;
        self.root_radii = [0.0; 4];
        self.root_face_meta = [0; 4];
        self.root_face_bounds = [0.0; 4];
        self.root_face_pop_pos = [0.0; 4];
        self.write_uniforms();
    }

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

    pub fn update_palette(&self, palette: &GpuPalette) {
        self.queue.write_buffer(&self.palette_buffer, 0, bytemuck::bytes_of(palette));
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
        self.write_uniforms();
    }

    pub fn update_camera(&self, camera: &GpuCamera) {
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn render_offscreen(&self) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen-frame"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen-frame"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("offscreen-ray-march"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.1, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
    }

    /// Render an off-screen frame and write a PNG to `path`. Used
    /// by the headless test driver so the agent can iterate on
    /// rendering issues without a window.
    pub fn capture_to_png(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let width = self.config.width;
        let height = self.config.height;
        let format = self.config.format;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bytes_per_pixel = 4u32;
        let unpadded_bpr = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        let buffer_size = (padded_bpr * height) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture-readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("capture-frame"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("capture-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.1, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::PollType::Wait)?;
        rx.recv()??;

        let raw = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded_bpr * height) as usize);
        let bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        for row in 0..height {
            let start = (row * padded_bpr) as usize;
            let end = start + unpadded_bpr as usize;
            for px in raw[start..end].chunks_exact(4) {
                if bgra {
                    pixels.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                } else {
                    pixels.extend_from_slice(px);
                }
            }
        }
        drop(raw);
        buffer.unmap();

        let file = std::fs::File::create(path)?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&pixels)?;

        Ok(())
    }

    pub fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let frame_start = std::time::Instant::now();
        let acquire_start = std::time::Instant::now();
        let output = self.surface.get_current_texture()?;
        let acquire_elapsed = acquire_start.elapsed();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let encode_start = std::time::Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ray_march"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.1, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        let encode_elapsed = encode_start.elapsed();
        let submit_start = std::time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        let submit_elapsed = submit_start.elapsed();
        let present_start = std::time::Instant::now();
        output.present();
        let present_elapsed = present_start.elapsed();
        let frame_elapsed = frame_start.elapsed();
        if frame_elapsed.as_secs_f64() * 1000.0 >= 30.0 {
            eprintln!(
                "renderer_slow acquire_ms={:.2} encode_ms={:.2} submit_ms={:.2} present_ms={:.2} total_ms={:.2}",
                acquire_elapsed.as_secs_f64() * 1000.0,
                encode_elapsed.as_secs_f64() * 1000.0,
                submit_elapsed.as_secs_f64() * 1000.0,
                present_elapsed.as_secs_f64() * 1000.0,
                frame_elapsed.as_secs_f64() * 1000.0,
            );
        }
        Ok(())
    }

    /// Re-upload the tree + node_kinds buffers after an edit or
    /// re-pack. Recreates the GPU buffers and bind group when the
    /// data outgrew the previous allocation.
    pub fn update_tree(
        &mut self,
        tree_data: &[GpuChild],
        node_kinds: &[GpuNodeKind],
        root_index: u32,
    ) {
        self.root_index = root_index;
        self.node_count = (tree_data.len() / 27) as u32;

        let tree_size = (tree_data.len() * std::mem::size_of::<GpuChild>()) as u64;
        let kinds_size = (node_kinds.len() * std::mem::size_of::<GpuNodeKind>()) as u64;

        let mut recreate_bind_group = false;

        if tree_size > self.tree_buffer.size() {
            self.tree_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tree"),
                contents: bytemuck::cast_slice(tree_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            recreate_bind_group = true;
        } else {
            self.queue.write_buffer(&self.tree_buffer, 0, bytemuck::cast_slice(tree_data));
        }

        if kinds_size > self.node_kinds_buffer.size() {
            self.node_kinds_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("node_kinds"),
                contents: bytemuck::cast_slice(node_kinds),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            recreate_bind_group = true;
        } else {
            self.queue.write_buffer(&self.node_kinds_buffer, 0, bytemuck::cast_slice(node_kinds));
        }

        if recreate_bind_group {
            self.bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
            );
        }

        self.write_uniforms();
    }

    fn write_uniforms(&self) {
        let uniforms = GpuUniforms {
            root_index: self.root_index,
            node_count: self.node_count,
            screen_width: self.config.width as f32,
            screen_height: self.config.height as f32,
            max_depth: self.max_depth,
            highlight_active: self.highlight_active,
            root_kind: self.root_kind,
            ribbon_count: self.ribbon_count,
            highlight_min: self.highlight_min,
            highlight_max: self.highlight_max,
            root_radii: self.root_radii,
            root_face_meta: self.root_face_meta,
            root_face_bounds: self.root_face_bounds,
            root_face_pop_pos: self.root_face_pop_pos,
        };
        self.queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    tree: &wgpu::Buffer,
    camera: &wgpu::Buffer,
    palette: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
    node_kinds: &wgpu::Buffer,
    ribbon: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ray_march"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tree.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: camera.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: palette.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: node_kinds.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: ribbon.as_entire_binding() },
        ],
    })
}
