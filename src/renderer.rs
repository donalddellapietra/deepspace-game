//! wgpu renderer: full-screen ray march shader.
//!
//! One render pipeline, one storage buffer (tree nodes), three uniform
//! buffers (camera, palette, uniforms). The fragment shader ray-marches
//! the base-3 tree with an iterative stack-based DDA.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuChild, GpuNodeMeta, GpuPalette};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuUniforms {
    pub root_index: u32,
    pub node_count: u32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub max_depth: u32,
    pub highlight_active: u32,
    pub _pad: [u32; 2],
    pub highlight_min: [f32; 4], // xyz, w unused
    pub highlight_max: [f32; 4], // xyz, w unused
    /// Cubed-sphere cursor highlight state — UI, not body geometry.
    /// x: `1.0` when active, `0.0` otherwise. y: highlight cell depth
    /// (`3^y` subdivisions per face axis). z, w: reserved.
    pub body_highlight_active: [f32; 4],
    /// Highlight cell `(face, iu, iv, ir)` at the depth above.
    pub body_highlight_cell: [f32; 4],
    /// Render frame: the packed tree's root node represents this
    /// sub-cube of world space. `xyz` = world-space min corner; `w` =
    /// world-space width of ONE root-cell (so the root node spans
    /// `[xyz, xyz + 3·w)`). When the render frame is the world root,
    /// this is `(0, 0, 0, 1)`.
    pub render_frame: [f32; 4],
    /// Sphere body's render-frame-local footprint. xyz = body cube's
    /// min corner in render-local units; w = cube side length. The
    /// shader uses this independent of where the camera's render
    /// root sits in the tree, so the body stays visible at any zoom.
    pub body_world: [f32; 4],
    /// `(inner_r, outer_r, _, _)` in body-local `[0, 1)` units. When
    /// `outer_r == 0` no body is rendered.
    pub body_radii: [f32; 4],
    /// Packed-buffer index of the body node. `u32::MAX` = no body.
    pub body_idx: u32,
    /// `1` when the render root sits inside a sphere subtree (body
    /// itself or any descendant face cell). The shader's flat-
    /// Cartesian DDA must skip the tree walk in that case — those
    /// cells are bulged voxels and the body pass renders them.
    pub render_root_in_sphere: u32,
    /// `1` when the render root is inside a face subtree and the
    /// shader should dispatch `march_face_chunk` (which renders the
    /// chunk's bulged voxels at the proper visible scale instead of
    /// the body pass's whole-shell view).
    pub face_chunk_active: u32,
    pub face_chunk_face: u32,
    /// Padding so the next `vec4<f32>` aligns to 16 bytes per WGSL
    /// uniform buffer rules (std140-like). Without this Rust packs
    /// 4 bytes shorter than the shader expects.
    pub _body_pad: [u32; 4],
    /// (u_lo, u_hi, v_lo, v_hi) of the chunk in face equal-angle
    /// coords [-1, 1].
    pub face_chunk_uv: [f32; 4],
    /// (r_lo, r_hi, _, _) — chunk's radial bounds in WORLD units
    /// (relative to body center).
    pub face_chunk_r: [f32; 4],
    /// Body's world center in render-frame-local coords. Shared by
    /// body pass and face-chunk pass.
    pub body_center_local: [f32; 4],
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tree_buffer: wgpu::Buffer,
    metas_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    palette_buffer: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    // Tracked state.
    root_index: u32,
    node_count: u32,
    max_depth: u32,
    highlight_active: u32,
    highlight_min: [f32; 4],
    highlight_max: [f32; 4],
    body_highlight_active: [f32; 4],
    body_highlight_cell: [f32; 4],
    render_frame: [f32; 4],
    body_world: [f32; 4],
    body_radii: [f32; 4],
    body_idx: u32,
    render_root_in_sphere: u32,
    face_chunk_active: u32,
    face_chunk_face: u32,
    face_chunk_uv: [f32; 4],
    face_chunk_r: [f32; 4],
    body_center_local: [f32; 4],
}

impl Renderer {
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        tree_data: &[GpuChild],
        tree_metas: &[GpuNodeMeta],
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

        // --- Buffers ---

        let tree_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tree"),
            contents: bytemuck::cast_slice(tree_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let metas_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node_metas"),
            contents: bytemuck::cast_slice(tree_metas),
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
            max_depth: 16,
            highlight_active: 0,
            _pad: [0; 2],
            highlight_min: [0.0; 4],
            highlight_max: [0.0; 4],
            body_highlight_active: [0.0; 4],
            body_highlight_cell: [0.0; 4],
            render_frame: [0.0, 0.0, 0.0, 1.0],
            body_world: [0.0; 4],
            body_radii: [0.0; 4],
            body_idx: u32::MAX,
            render_root_in_sphere: 0,
            face_chunk_active: 0,
            face_chunk_face: 0,
            _body_pad: [0; 4],
            face_chunk_uv: [0.0; 4],
            face_chunk_r: [0.0; 4],
            body_center_local: [0.0; 4],
        };
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group ---

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ray_march"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray_march"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tree_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: palette_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: metas_buffer.as_entire_binding() },
            ],
        });

        // --- Shader & Pipeline ---

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
            device,
            queue,
            surface,
            config,
            pipeline,
            bind_group_layout,
            tree_buffer,
            metas_buffer,
            camera_buffer,
            palette_buffer,
            uniforms_buffer,
            bind_group,
            root_index,
            node_count,
            max_depth: 16,
            highlight_active: 0,
            highlight_min: [0.0; 4],
            highlight_max: [0.0; 4],
            body_highlight_active: [0.0; 4],
            body_highlight_cell: [0.0; 4],
            render_frame: [0.0, 0.0, 0.0, 1.0],
            body_world: [0.0; 4],
            body_radii: [0.0; 4],
            body_idx: u32::MAX,
            render_root_in_sphere: 0,
            face_chunk_active: 0,
            face_chunk_face: 0,
            face_chunk_uv: [0.0; 4],
            face_chunk_r: [0.0; 4],
            body_center_local: [0.0; 4],
        }
    }

    /// Set (or clear with `None`) the active face-chunk for the
    /// current render root. When set, the shader's `march_face_chunk`
    /// renders the chunk's bulged voxels at the proper visible scale.
    pub fn set_face_chunk(
        &mut self,
        chunk: Option<crate::app::edit_actions::FaceChunk>,
        body_center_local: [f32; 3],
    ) {
        match chunk {
            Some(c) => {
                self.face_chunk_active = 1;
                self.face_chunk_face = c.face;
                self.face_chunk_uv = [c.u_lo, c.u_hi, c.v_lo, c.v_hi];
                self.face_chunk_r = [c.r_lo, c.r_hi, 0.0, 0.0];
            }
            None => {
                self.face_chunk_active = 0;
            }
        }
        self.body_center_local = [body_center_local[0], body_center_local[1], body_center_local[2], 0.0];
        self.write_uniforms();
    }

    pub fn set_render_root_in_sphere(&mut self, in_sphere: bool) {
        let v = if in_sphere { 1 } else { 0 };
        if self.render_root_in_sphere != v {
            self.render_root_in_sphere = v;
            self.write_uniforms();
        }
    }

    /// Set the body's render-frame-local footprint, kind payload,
    /// and packed-buffer index. `body_idx == u32::MAX` hides the
    /// body. Call each frame after recomputing the render frame.
    pub fn set_body(
        &mut self,
        world: [f32; 4],
        inner_r: f32,
        outer_r: f32,
        buf_idx: u32,
    ) {
        self.body_world = world;
        self.body_radii = [inner_r, outer_r, 0.0, 0.0];
        self.body_idx = buf_idx;
        self.write_uniforms();
    }

    /// Declare the world-space footprint of the packed tree's root.
    /// `origin` is the min corner in world units; `cell_size` is the
    /// width of one root-cell (the root node spans `[origin, origin +
    /// 3·cell_size)`). Call this whenever the render frame moves or
    /// rescales (e.g., camera crosses into a new depth-K ancestor).
    pub fn set_render_frame(&mut self, origin: [f32; 3], cell_size: f32) {
        self.render_frame = [origin[0], origin[1], origin[2], cell_size];
        self.write_uniforms();
    }

    /// Set or clear the cubed-sphere cursor highlight.
    /// `cell = (face, iu, iv, ir, depth)`. Indices live in the grid
    /// `3^depth` per axis, so `depth = 1` highlights 1 of 27 coarse
    /// "chunks" per face, and `depth = subtree_depth` highlights a
    /// single finest cell. The shader compares cell indices at the
    /// same depth so the wireframe shrinks/grows with depth.
    pub fn set_body_highlight(
        &mut self,
        cell: Option<(u32, u32, u32, u32, u32)>,
    ) {
        match cell {
            Some((face, i, j, k, depth)) => {
                self.body_highlight_active = [1.0, depth as f32, 0.0, 0.0];
                self.body_highlight_cell = [face as f32, i as f32, j as f32, k as f32];
            }
            None => {
                self.body_highlight_active = [0.0; 4];
                self.body_highlight_cell = [0.0; 4];
            }
        }
        self.write_uniforms();
    }

    pub fn update_palette(&self, palette: &GpuPalette) {
        self.queue.write_buffer(&self.palette_buffer, 0, bytemuck::bytes_of(palette));
    }

    /// Set the highlighted block AABB (or clear it with None).
    pub fn set_highlight(&mut self, aabb: Option<([f32; 3], [f32; 3])>) {
        match aabb {
            Some((min, max)) => {
                self.highlight_active = 1;
                self.highlight_min = [min[0], min[1], min[2], 0.0];
                self.highlight_max = [max[0], max[1], max[2], 0.0];
            }
            None => {
                self.highlight_active = 0;
            }
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

    pub fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    /// Render one frame to an offscreen texture in the surface format
    /// and write it to `path` as PNG. Used by the test runner to
    /// produce deterministic visual artifacts without depending on a
    /// human-visible window state.
    pub fn capture_to_png(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let width = self.config.width;
        let height = self.config.height;

        // The target format must match the pipeline's color target
        // (set up in `Renderer::new` from `surface.get_capabilities`).
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

        // wgpu requires `bytes_per_row` aligned to 256.
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

        // Map and read back. Using a channel + poll keeps it simple
        // without taking on a runtime.
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::PollType::Wait)?;
        rx.recv()??;

        let raw = slice.get_mapped_range();
        // Strip row padding into a tight RGBA8 buffer. Surface format
        // is normally `Bgra8UnormSrgb` on macOS — we swap channels so
        // the PNG comes out as RGBA.
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

        // PNG encode.
        let file = std::fs::File::create(path)?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&pixels)?;

        Ok(())
    }

    /// Re-upload the tree buffer after an edit. Recreates the buffer
    /// and bind group if the size changed.
    pub fn update_tree(
        &mut self,
        tree_data: &[GpuChild],
        tree_metas: &[GpuNodeMeta],
        root_index: u32,
    ) {
        // Guard against a runaway pack. A healthy world has on the
        // order of 10³ nodes; 10⁶ means something exploded — better
        // to skip the upload and keep last frame's data than to
        // allocate gigabytes of GPU memory and freeze the system.
        const MAX_NODES: usize = 1_000_000;
        if tree_data.len() / 27 > MAX_NODES {
            log::error!(
                "renderer: refusing tree upload of {} nodes (cap {}); keeping previous frame",
                tree_data.len() / 27, MAX_NODES,
            );
            return;
        }

        self.root_index = root_index;
        self.node_count = (tree_data.len() / 27) as u32;

        let new_tree_size = (tree_data.len() * std::mem::size_of::<GpuChild>()) as u64;
        let new_metas_size = (tree_metas.len() * std::mem::size_of::<GpuNodeMeta>()) as u64;

        let mut recreate_bind_group = false;

        if new_tree_size > self.tree_buffer.size() {
            self.tree_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tree"),
                contents: bytemuck::cast_slice(tree_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            recreate_bind_group = true;
        } else {
            self.queue.write_buffer(&self.tree_buffer, 0, bytemuck::cast_slice(tree_data));
        }

        if new_metas_size > self.metas_buffer.size() {
            self.metas_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("node_metas"),
                contents: bytemuck::cast_slice(tree_metas),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            recreate_bind_group = true;
        } else {
            self.queue.write_buffer(&self.metas_buffer, 0, bytemuck::cast_slice(tree_metas));
        }

        if recreate_bind_group {
            self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ray_march"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.tree_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.camera_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.palette_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.uniforms_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.metas_buffer.as_entire_binding() },
                ],
            });
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
            _pad: [0; 2],
            highlight_min: self.highlight_min,
            highlight_max: self.highlight_max,
            body_highlight_active: self.body_highlight_active,
            body_highlight_cell: self.body_highlight_cell,
            render_frame: self.render_frame,
            body_world: self.body_world,
            body_radii: self.body_radii,
            body_idx: self.body_idx,
            render_root_in_sphere: self.render_root_in_sphere,
            face_chunk_active: self.face_chunk_active,
            face_chunk_face: self.face_chunk_face,
            _body_pad: [0; 4],
            face_chunk_uv: self.face_chunk_uv,
            face_chunk_r: self.face_chunk_r,
            body_center_local: self.body_center_local,
        };
        self.queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}
