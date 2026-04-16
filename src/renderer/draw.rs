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
    /// Ray-march render pass duration as reported by the GPU itself,
    /// via the `TIMESTAMP_QUERY` scaffolding. `Some(0.0)` is a valid
    /// result (pass was truly trivial), `None` means the adapter
    /// does not support timestamp queries. On CPU-bound frames this
    /// will be much smaller than `wait_ms`; when they're close, the
    /// frame is genuinely GPU-bound.
    pub gpu_pass_ms: Option<f64>,
    /// Staging-buffer map_async + read-back cost for the timestamp
    /// values themselves. Included in `wait_ms`; broken out so it
    /// can be subtracted from a GPU-bound interpretation.
    pub gpu_readback_ms: f64,
    /// Time from `queue.submit` to the `on_submitted_work_done`
    /// callback firing. Captures *all* GPU work including the
    /// TBDR tile-resolve phase on Apple Silicon — which
    /// `gpu_pass_ms` (render-pass-boundary timestamps) can miss.
    /// When this is close to `wait_ms` and much larger than
    /// `gpu_pass_ms`, the cost is outside the measured pass:
    /// typically tile resolve, flush, or driver scheduling.
    pub submitted_done_ms: Option<f64>,
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
        let timestamp_writes = self.timestamp.as_ref().map(|ts| wgpu::RenderPassTimestampWrites {
            query_set: &ts.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
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
                timestamp_writes,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(ts) = self.timestamp.as_ref() {
            encoder.resolve_query_set(&ts.query_set, 0..2, &ts.resolve, 0);
            encoder.copy_buffer_to_buffer(&ts.resolve, 0, &ts.staging, 0, 16);
        }
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
        let done_slot: std::sync::Arc<std::sync::Mutex<Option<std::time::Instant>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let submit_start = std::time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        // Register the callback *after* submit: wgpu's
        // `on_submitted_work_done` fires when the queue is drained
        // up to the time of the call. Registering before submit
        // fires on the previous frame's completion (usually ~0 ms)
        // and tells us nothing about the current frame.
        {
            let done_slot = std::sync::Arc::clone(&done_slot);
            self.queue.on_submitted_work_done(move || {
                let mut slot = done_slot.lock().unwrap();
                *slot = Some(std::time::Instant::now());
            });
        }
        let submit_ms = submit_start.elapsed().as_secs_f64() * 1000.0;
        let wait_start = std::time::Instant::now();
        let _ = self.device.poll(wgpu::PollType::Wait);
        let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
        let submitted_done_ms = done_slot
            .lock()
            .ok()
            .and_then(|slot| slot.map(|t| t.duration_since(submit_start).as_secs_f64() * 1000.0));
        let (gpu_pass_ms, gpu_readback_ms) = self.read_timestamps();
        let total_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        if total_ms >= 10.0 {
            eprintln!(
                "renderer_offscreen_slow size={}x{} texture_alloc_ms={:.2} view_ms={:.2} encode_ms={:.2} submit_ms={:.2} wait_ms={:.2} gpu_pass_ms={} total_ms={:.2}",
                self.config.width, self.config.height,
                texture_alloc_ms, view_ms, encode_ms, submit_ms, wait_ms,
                gpu_pass_ms.map(|v| format!("{v:.2}")).unwrap_or_else(|| "na".into()),
                total_ms,
            );
        }
        OffscreenRenderTiming {
            texture_alloc_ms, view_ms, encode_ms, submit_ms, wait_ms, total_ms,
            gpu_pass_ms, gpu_readback_ms, submitted_done_ms,
        }
    }

    /// Map the staging buffer (if present), read back the two
    /// timestamp tick values, compute the delta in ms using the
    /// queue's ns-per-tick period. Returns `(pass_ms, readback_ms)`.
    /// `pass_ms` is `None` if the adapter did not enable the
    /// timestamp-query feature.
    fn read_timestamps(&self) -> (Option<f64>, f64) {
        let ts = match self.timestamp.as_ref() {
            Some(t) => t,
            None => return (None, 0.0),
        };
        let readback_start = std::time::Instant::now();
        let slice = ts.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        // map_async's callback is only driven when the device is
        // polled; poll(Wait) here is bounded because the submit
        // that resolved the query set already completed during the
        // earlier wait in render_offscreen.
        let _ = self.device.poll(wgpu::PollType::Wait);
        if rx.recv().is_err() {
            ts.staging.unmap();
            let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
            return (None, readback_ms);
        }
        let data = slice.get_mapped_range();
        let start_tick = u64::from_ne_bytes(data[0..8].try_into().unwrap());
        let end_tick = u64::from_ne_bytes(data[8..16].try_into().unwrap());
        drop(data);
        ts.staging.unmap();
        // Known wgpu/Metal quirk on Apple Silicon: the per-render-pass
        // start/end timestamp counters are not guaranteed to be
        // monotonic for fast passes — we sometimes see end < start by
        // a few milliseconds. Take the magnitude as a best-effort
        // estimate; `wait_ms` stays as the authoritative GPU-bound
        // signal on Metal. On vendors where the counters behave,
        // `abs` is a no-op vs. the direct subtraction.
        let delta_ticks = if end_tick >= start_tick {
            end_tick - start_tick
        } else {
            start_tick - end_tick
        };
        let ms = (delta_ticks as f64) * (ts.period_ns as f64) / 1_000_000.0;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        // Guard against Apple Silicon Metal's timestamp counters
        // returning nonsense for sub-ms passes (we've seen values
        // ~13e6 ms, = 3+ hours, from a frame that actually took
        // milliseconds). Cap at 5 seconds per pass — anything above
        // that is physically impossible in a render-harness run and
        // should be treated as "no reliable sample".
        const MAX_PLAUSIBLE_MS: f64 = 5000.0;
        let reported = if ms <= MAX_PLAUSIBLE_MS { Some(ms) } else { None };
        (reported, readback_ms)
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
