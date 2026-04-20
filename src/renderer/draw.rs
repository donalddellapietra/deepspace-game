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
    /// Shader-side per-ray counters for the frame. Populated by
    /// atomic writes in the fragment shader, read back by copy to
    /// the `shader_stats_readback` buffer + map_async.
    pub shader_stats: ShaderStatsFrame,
}

/// Decoded `shader_stats` buffer for one frame. `avg_steps` is
/// computed CPU-side as `sum_steps_div4 * 4 / ray_count` (the GPU
/// side stored div-by-4 to avoid u32 overflow).
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderStatsFrame {
    pub ray_count: u32,
    pub hit_count: u32,
    pub miss_count: u32,
    pub max_iter_count: u32,
    pub sum_steps_div4: u32,
    pub max_steps: u32,
    /// Per-branch breakdown of the ray_steps total. Each step lands
    /// in exactly one branch, so `sum_steps ≈ oob + empty + descend +
    /// lod_terminal` (the terminal-hit branch returns immediately
    /// without incrementing ray_steps for that iteration).
    pub sum_steps_oob_div4: u32,
    pub sum_steps_empty_div4: u32,
    pub sum_steps_node_descend_div4: u32,
    pub sum_steps_lod_terminal_div4: u32,
}

impl ShaderStatsFrame {
    pub fn avg_steps(&self) -> f64 {
        if self.ray_count == 0 {
            0.0
        } else {
            (self.sum_steps_div4 as f64 * 4.0) / self.ray_count as f64
        }
    }

    pub fn avg_steps_oob(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_oob_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_steps_empty(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_empty_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_steps_descend(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_node_descend_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_steps_lod_terminal(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_lod_terminal_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn hit_fraction(&self) -> f64 {
        if self.ray_count == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.ray_count as f64
        }
    }

    pub fn max_iter_fraction(&self) -> f64 {
        if self.ray_count == 0 {
            0.0
        } else {
            self.max_iter_count as f64 / self.ray_count as f64
        }
    }
}

impl Renderer {
    /// Lazily (re)allocate the scaled-down ray-march target. No-op
    /// when `render_scale == 1` — the ray-march writes directly to
    /// the destination in that case. On size/scale change the
    /// existing texture is dropped and replaced.
    pub(super) fn ensure_ray_march_target(&mut self) {
        if self.render_scale <= 1 { return; }
        let (w, h) = self.scaled_size();
        if let Some(tex) = self.ray_march_target.as_ref() {
            if tex.width() == w && tex.height() == h { return; }
        }
        self.ray_march_target = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ray_march_target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
    }

    /// Live-surface render path. Matches `render_offscreen`'s
    /// instrumentation when `shader_stats_enabled`: timestamp
    /// queries, `on_submitted_work_done` callback, stats-buffer
    /// clear/copy/readback. When disabled, the only extra cost over
    /// the old render() is an `on_submitted_work_done` registration
    /// (trivial). The enriched `renderer_slow` log fires whenever a
    /// frame exceeds 30ms total, reporting the per-phase breakdown
    /// the harness surfaces — so a live-game regression can be
    /// diagnosed from stderr without running the offscreen harness.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame_start = std::time::Instant::now();
        let acquire_start = std::time::Instant::now();
        let output = self.surface.get_current_texture()?;
        let acquire_elapsed = acquire_start.elapsed();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.ensure_ray_march_target();
        let use_blit = self.render_scale > 1 && self.ray_march_target.is_some();
        let scaled_view = self.ray_march_target.as_ref().map(|t| {
            t.create_view(&wgpu::TextureViewDescriptor::default())
        });
        let encode_start = std::time::Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        if self.shader_stats_enabled {
            encoder.clear_buffer(&self.shader_stats_buffer, 0, None);
        }
        let timestamp_writes = self.timestamp.as_ref().map(|ts| wgpu::RenderPassTimestampWrites {
            query_set: &ts.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
        {
            let march_view = if use_blit {
                scaled_view.as_ref().unwrap()
            } else {
                &view
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ray_march"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: march_view,
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
        // Cursor-probe compute pass — one ray from the crosshair
        // using the same `march()` the fragment shader uses. Output
        // is copied to the staging buffer so the CPU can map it
        // between frames for the highlight uniform + break/place.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cursor_probe"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cursor_probe_gpu.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_bind_group(1, &self.cursor_probe_gpu.bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.cursor_probe_gpu.output_buffer, 0,
            &self.cursor_probe_gpu.staging_buffer, 0,
            super::cursor_probe::CURSOR_PROBE_BYTES,
        );
        if use_blit {
            let src_view = scaled_view.as_ref().unwrap();
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit"),
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.blit_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_to_swapchain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(ts) = self.timestamp.as_ref() {
            encoder.resolve_query_set(&ts.query_set, 0..2, &ts.resolve, 0);
            encoder.copy_buffer_to_buffer(&ts.resolve, 0, &ts.staging, 0, 16);
        }
        if self.shader_stats_enabled {
            encoder.copy_buffer_to_buffer(
                &self.shader_stats_buffer, 0,
                &self.shader_stats_readback, 0,
                64,
            );
        }
        let encode_elapsed = encode_start.elapsed();
        let done_slot: std::sync::Arc<std::sync::Mutex<Option<std::time::Instant>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let submit_start = std::time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        {
            let done_slot = std::sync::Arc::clone(&done_slot);
            self.queue.on_submitted_work_done(move || {
                let mut slot = done_slot.lock().unwrap();
                *slot = Some(std::time::Instant::now());
            });
        }
        let submit_elapsed = submit_start.elapsed();
        let present_start = std::time::Instant::now();
        output.present();
        let present_elapsed = present_start.elapsed();
        let frame_elapsed = frame_start.elapsed();
        // Surface sub-phase timings for the event loop's slow-frame
        // diagnostic. `last_render_wait_ms` aliases the present()
        // duration — on macOS this is where vsync pacing / swap
        // chain backpressure shows up.
        self.last_render_encode_ms = encode_elapsed.as_secs_f64() * 1000.0;
        self.last_render_submit_ms = submit_elapsed.as_secs_f64() * 1000.0;
        self.last_render_wait_ms = present_elapsed.as_secs_f64() * 1000.0;
        // Poll + stats readback only when enabled, and only on slow
        // frames. The poll blocks the CPU on GPU completion, so we
        // don't want to stall every frame — but during a slowdown we
        // want the data.
        let slow = frame_elapsed.as_secs_f64() * 1000.0 >= 30.0;
        let (gpu_pass_ms, submitted_done_ms, shader_stats) =
            if self.shader_stats_enabled && slow {
                let _ = self.device.poll(wgpu::PollType::Wait);
                let submitted_done_ms = done_slot
                    .lock()
                    .ok()
                    .and_then(|s| s.map(|t| t.duration_since(submit_start).as_secs_f64() * 1000.0));
                let (gpu_pass_ms, _) = self.read_timestamps();
                let stats = self.read_shader_stats();
                (gpu_pass_ms, submitted_done_ms, stats)
            } else {
                (None, None, ShaderStatsFrame::default())
            };
        // Periodic steady-state sample: CPU-side timings only, no
        // device.poll(Wait) stall. Gives us acquire/encode/submit/
        // present/total on NORMAL frames, not just slow ones — which
        // is what we need to diagnose where the 16 ms budget is spent
        // at 60 FPS.
        self.live_frame_counter = self.live_frame_counter.wrapping_add(1);
        let sample = self.live_sample_every_frames > 0
            && self.live_frame_counter % self.live_sample_every_frames as u64 == 0;
        if sample && !slow {
            eprintln!(
                "render_live_sample frame={} acquire_ms={:.2} encode_ms={:.2} submit_ms={:.2} present_ms={:.2} total_ms={:.2}",
                self.live_frame_counter,
                acquire_elapsed.as_secs_f64() * 1000.0,
                encode_elapsed.as_secs_f64() * 1000.0,
                submit_elapsed.as_secs_f64() * 1000.0,
                present_elapsed.as_secs_f64() * 1000.0,
                frame_elapsed.as_secs_f64() * 1000.0,
            );
        }
        if slow {
            eprintln!(
                "renderer_slow acquire_ms={:.2} encode_ms={:.2} submit_ms={:.2} present_ms={:.2} total_ms={:.2} gpu_pass_ms={} submitted_done_ms={} rays={} hits={} miss={} max_iters={} avg_steps={:.1} max_steps={} avg_oob={:.1} avg_empty={:.1} avg_descend={:.1} avg_lod_terminal={:.1}",
                acquire_elapsed.as_secs_f64() * 1000.0,
                encode_elapsed.as_secs_f64() * 1000.0,
                submit_elapsed.as_secs_f64() * 1000.0,
                present_elapsed.as_secs_f64() * 1000.0,
                frame_elapsed.as_secs_f64() * 1000.0,
                gpu_pass_ms.map(|v| format!("{v:.2}")).unwrap_or_else(|| "na".into()),
                submitted_done_ms.map(|v| format!("{v:.2}")).unwrap_or_else(|| "na".into()),
                shader_stats.ray_count,
                shader_stats.hit_count,
                shader_stats.miss_count,
                shader_stats.max_iter_count,
                shader_stats.avg_steps(),
                shader_stats.max_steps,
                shader_stats.avg_steps_oob(),
                shader_stats.avg_steps_empty(),
                shader_stats.avg_steps_descend(),
                shader_stats.avg_steps_lod_terminal(),
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
        self.ensure_ray_march_target();
        let use_blit = self.render_scale > 1 && self.ray_march_target.is_some();
        let texture_alloc_ms = alloc_start.elapsed().as_secs_f64() * 1000.0;
        let view_start = std::time::Instant::now();
        let texture = self
            .offscreen_texture
            .as_ref()
            .expect("offscreen texture initialized");
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let scaled_view = self.ray_march_target.as_ref().map(|t| {
            t.create_view(&wgpu::TextureViewDescriptor::default())
        });
        let view_ms = view_start.elapsed().as_secs_f64() * 1000.0;
        let encode_start = std::time::Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen-frame"),
        });
        if self.shader_stats_enabled {
            // Zero the shader_stats buffer so atomics accumulate from
            // 0 this frame. `clear_buffer` is a GPU-side fill; no
            // CPU synchronization cost, and serializes with the
            // subsequent render pass on the same encoder.
            encoder.clear_buffer(&self.shader_stats_buffer, 0, None);
        }
        let timestamp_writes = self.timestamp.as_ref().map(|ts| wgpu::RenderPassTimestampWrites {
            query_set: &ts.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
        {
            let march_view = if use_blit {
                scaled_view.as_ref().unwrap()
            } else {
                &view
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("offscreen-ray-march"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: march_view,
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
        // Cursor-probe compute pass (same as in `render()`).
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cursor_probe"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cursor_probe_gpu.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_bind_group(1, &self.cursor_probe_gpu.bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.cursor_probe_gpu.output_buffer, 0,
            &self.cursor_probe_gpu.staging_buffer, 0,
            super::cursor_probe::CURSOR_PROBE_BYTES,
        );
        if use_blit {
            let src_view = scaled_view.as_ref().unwrap();
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit"),
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.blit_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_to_offscreen"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(ts) = self.timestamp.as_ref() {
            encoder.resolve_query_set(&ts.query_set, 0..2, &ts.resolve, 0);
            encoder.copy_buffer_to_buffer(&ts.resolve, 0, &ts.staging, 0, 16);
        }
        if self.shader_stats_enabled {
            encoder.copy_buffer_to_buffer(
                &self.shader_stats_buffer, 0,
                &self.shader_stats_readback, 0,
                64,
            );
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
        let shader_stats = if self.shader_stats_enabled {
            self.read_shader_stats()
        } else {
            ShaderStatsFrame::default()
        };
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
            gpu_pass_ms, gpu_readback_ms, submitted_done_ms, shader_stats,
        }
    }

    /// Map `shader_stats_readback`, decode the 6 u32 counters, and
    /// unmap. Called after the render-pass submit has completed on
    /// GPU (verified by the earlier `poll(Wait)`), and after the
    /// timestamp readback has already fired its own poll pass.
    fn read_shader_stats(&self) -> ShaderStatsFrame {
        let slice = self.shader_stats_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::Wait);
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            self.shader_stats_readback.unmap();
            return ShaderStatsFrame::default();
        }
        let data = slice.get_mapped_range();
        let read_u32 = |offset: usize| {
            u32::from_ne_bytes(data[offset..offset + 4].try_into().unwrap())
        };
        let stats = ShaderStatsFrame {
            ray_count: read_u32(0),
            hit_count: read_u32(4),
            miss_count: read_u32(8),
            max_iter_count: read_u32(12),
            sum_steps_div4: read_u32(16),
            max_steps: read_u32(20),
            sum_steps_oob_div4: read_u32(24),
            sum_steps_empty_div4: read_u32(28),
            sum_steps_node_descend_div4: read_u32(32),
            sum_steps_lod_terminal_div4: read_u32(36),
        };
        drop(data);
        self.shader_stats_readback.unmap();
        stats
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
