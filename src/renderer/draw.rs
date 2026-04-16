//! Frame rendering: on-surface, offscreen, and readback-to-PNG.

use super::Renderer;

#[derive(Debug, Clone, Copy, Default)]
pub struct OffscreenRenderTiming {
    pub texture_alloc_ms: f64,
    pub view_ms: f64,
    pub encode_ms: f64,
    pub submit_ms: f64,
    pub wait_ms: f64,
    pub total_ms: f64,
}

impl Renderer {
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

    pub fn render_offscreen(&mut self) -> OffscreenRenderTiming {
        let frame_start = std::time::Instant::now();
        let alloc_start = frame_start;
        if self.offscreen_texture.is_none() {
            self.offscreen_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
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
            }));
        }
        let texture_alloc_ms = alloc_start.elapsed().as_secs_f64() * 1000.0;
        let view_start = std::time::Instant::now();
        let texture = self
            .offscreen_texture
            .as_ref()
            .expect("offscreen texture initialized");
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let view_ms = view_start.elapsed().as_secs_f64() * 1000.0;
        let encode_start = std::time::Instant::now();
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
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
        let submit_start = std::time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        let submit_ms = submit_start.elapsed().as_secs_f64() * 1000.0;
        let wait_start = std::time::Instant::now();
        let _ = self.device.poll(wgpu::PollType::Wait);
        let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        if total_ms >= 10.0 {
            eprintln!(
                "renderer_offscreen_slow size={}x{} texture_alloc_ms={:.2} view_ms={:.2} encode_ms={:.2} submit_ms={:.2} wait_ms={:.2} total_ms={:.2}",
                self.config.width, self.config.height,
                texture_alloc_ms, view_ms, encode_ms, submit_ms, wait_ms, total_ms,
            );
        }
        OffscreenRenderTiming {
            texture_alloc_ms, view_ms, encode_ms, submit_ms, wait_ms, total_ms,
        }
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
}
