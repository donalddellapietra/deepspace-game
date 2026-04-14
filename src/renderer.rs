//! wgpu renderer: full-screen ray march shader.
//!
//! One render pipeline, one storage buffer (tree nodes), three uniform
//! buffers (camera, palette, uniforms). The fragment shader ray-marches
//! the base-3 tree with an iterative stack-based DDA.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuChild, GpuNodeKind, GpuPalette};

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
    /// Spherical planet: xyz = center, w = outer radius (0 disables).
    pub cs_planet: [f32; 4],
    /// x = inner radius. y/z/w reserved for later phases (cells
    /// visualization parameters or highlight-active flag).
    pub cs_params: [f32; 4],
    /// Reserved for Phase C cursor highlight.
    pub cs_highlight: [f32; 4],
    /// GPU buffer indices of the 6 face subtree roots.
    /// Layout: [PosX, NegX, PosY, NegY, PosZ, NegZ, unused, unused].
    /// Face `f` is looked up at `cs_face_roots[f]`.
    pub cs_face_roots: [u32; 8],
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tree_buffer: wgpu::Buffer,
    kinds_buffer: wgpu::Buffer,
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
    cs_planet: [f32; 4],
    cs_params: [f32; 4],
    cs_highlight: [f32; 4],
    cs_face_roots: [u32; 8],
}

impl Renderer {
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        tree_data: &[GpuChild],
        kinds_data: &[GpuNodeKind],
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

        let kinds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kinds"),
            contents: bytemuck::cast_slice(kinds_data),
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
            cs_planet: [0.0; 4],
            cs_params: [0.0; 4],
            cs_highlight: [0.0; 4],
            cs_face_roots: [0; 8],
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
                wgpu::BindGroupEntry { binding: 4, resource: kinds_buffer.as_entire_binding() },
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
            kinds_buffer,
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
            cs_planet: [0.0; 4],
            cs_params: [0.0; 4],
            cs_highlight: [0.0; 4],
            cs_face_roots: [0; 8],
        }
    }

    /// Provide the 6 face-subtree buffer indices for the spherical
    /// planet. Order: PosX, NegX, PosY, NegY, PosZ, NegZ.
    pub fn set_face_roots(&mut self, roots: [u32; 6]) {
        for (i, &r) in roots.iter().enumerate() {
            self.cs_face_roots[i] = r;
        }
        self.write_uniforms();
    }

    /// Set or clear the cubed-sphere cursor highlight.
    /// `cell = (face, iu, iv, ir, depth)`. Indices live in the grid
    /// `3^depth` per axis, so `depth = 1` highlights 1 of 27 coarse
    /// "chunks" per face, and `depth = subtree_depth` highlights a
    /// single finest cell. The shader compares cell indices at the
    /// same depth so the wireframe shrinks/grows with depth.
    pub fn set_cubed_sphere_highlight(
        &mut self,
        cell: Option<(u32, u32, u32, u32, u32)>,
    ) {
        match cell {
            Some((face, i, j, k, depth)) => {
                self.cs_highlight = [face as f32, i as f32, j as f32, k as f32];
                self.cs_params[2] = depth as f32;
                self.cs_params[3] = 1.0;
            }
            None => {
                self.cs_highlight = [0.0; 4];
                self.cs_params[2] = 0.0;
                self.cs_params[3] = 0.0;
            }
        }
        self.write_uniforms();
    }

    /// Configure the spherical planet's rendering footprint.
    /// - `outer_radius == 0` hides the planet.
    /// - `depth` = levels of recursion in each face subtree (3^depth
    ///   cells per face per axis). Needed by the shader so its DDA
    ///   knows the finest cell size to step to.
    pub fn set_cubed_sphere_planet(
        &mut self,
        center: [f32; 3],
        inner_radius: f32,
        outer_radius: f32,
        depth: u32,
    ) {
        self.cs_planet = [center[0], center[1], center[2], outer_radius];
        self.cs_params = [inner_radius, depth as f32, 0.0, self.cs_params[3]];
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

    /// Re-upload the tree buffer after an edit. Recreates the buffer
    /// and bind group if the size changed.
    pub fn update_tree(
        &mut self,
        tree_data: &[GpuChild],
        kinds_data: &[GpuNodeKind],
        root_index: u32,
    ) {
        self.root_index = root_index;
        self.node_count = (tree_data.len() / 27) as u32;

        let new_tree_size = (tree_data.len() * std::mem::size_of::<GpuChild>()) as u64;
        let new_kinds_size = (kinds_data.len() * std::mem::size_of::<GpuNodeKind>()) as u64;
        let tree_grew = new_tree_size > self.tree_buffer.size();
        let kinds_grew = new_kinds_size > self.kinds_buffer.size();

        if tree_grew {
            self.tree_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tree"),
                contents: bytemuck::cast_slice(tree_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            self.queue.write_buffer(&self.tree_buffer, 0, bytemuck::cast_slice(tree_data));
        }

        if kinds_grew {
            self.kinds_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("kinds"),
                contents: bytemuck::cast_slice(kinds_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            self.queue.write_buffer(&self.kinds_buffer, 0, bytemuck::cast_slice(kinds_data));
        }

        if tree_grew || kinds_grew {
            self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ray_march"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.tree_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.camera_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.palette_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.uniforms_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.kinds_buffer.as_entire_binding() },
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
            cs_planet: self.cs_planet,
            cs_params: self.cs_params,
            cs_highlight: self.cs_highlight,
            cs_face_roots: self.cs_face_roots,
        };
        self.queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}
