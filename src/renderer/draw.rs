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
    /// Time from `queue.submit` to the `on_submitted_work_done`
    /// callback firing. Authoritative GPU-bound signal on Apple
    /// Silicon: captures all GPU work including TBDR tile resolve.
    /// (A prior render-pass-boundary `gpu_pass_ms` timestamp-query
    /// metric was removed — Metal's per-pass timestamps were
    /// non-monotonic for fast passes and gave nonsense values.)
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
    /// Instrumentation counter: how many descent candidates would
    /// have been culled by `(child_occupancy & path_mask) == 0u`
    /// if the test ran BEFORE descending. Upper-bound on savings a
    /// real path-mask cull could deliver. Does not alter traversal.
    pub sum_steps_would_cull_div4: u32,
    /// Per-ray storage-buffer u32-load counters, split by which
    /// buffer is read. On Apple Silicon these are the dominant
    /// cost source (dependent chains stall L1); ALU counting on
    /// the same shader is not representative of real frame time.
    /// Populated only when ENABLE_STATS is true.
    pub sum_loads_tree_div4: u32,
    pub sum_loads_offsets_div4: u32,
    pub sum_loads_kinds_div4: u32,
    pub sum_loads_ribbon_div4: u32,
    /// Steps accumulated over rays that RETURNED a hit. Divided
    /// by 4 on the GPU side. Use with `hit_count` to compute avg
    /// steps per hit; `sum_steps_div4 - sum_steps_hits_div4` gives
    /// the per-miss total.
    pub sum_steps_hits_div4: u32,
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

    pub fn avg_steps_would_cull(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_would_cull_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_loads_tree(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_tree_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_offsets(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_offsets_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_kinds(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_kinds_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_ribbon(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_ribbon_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_total(&self) -> f64 {
        self.avg_loads_tree() + self.avg_loads_offsets()
            + self.avg_loads_kinds() + self.avg_loads_ribbon()
    }

    pub fn avg_steps_per_hit(&self) -> f64 {
        if self.hit_count == 0 { 0.0 }
        else { (self.sum_steps_hits_div4 as f64 * 4.0) / self.hit_count as f64 }
    }

    pub fn avg_steps_per_miss(&self) -> f64 {
        let miss_count = self.miss_count;
        if miss_count == 0 { return 0.0; }
        let miss_sum_div4 = self.sum_steps_div4.saturating_sub(self.sum_steps_hits_div4);
        (miss_sum_div4 as f64 * 4.0) / miss_count as f64
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
    /// Record the per-frame ray-march (and, when TAAU is enabled,
    /// the resolve) passes into `encoder`. Written once, called from
    /// both `render` and `render_offscreen` so the TAA branch lives
    /// in exactly one place.
    ///
    /// Pre-condition: the caller has already uploaded the current
    /// camera. On the TAA path this helper ticks the jitter counter
    /// and patches the camera buffer's jitter fields in-place, so
    /// the march pass sees the sub-pixel offset without requiring
    /// the caller to know about TAAU.
    fn record_frame_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dest_view: &wgpu::TextureView,
        march_label: &'static str,
    ) {
        // Stamp jitter into the camera buffer + upload resolve
        // uniforms. When TAAU is off this is a no-op.
        if self.taa.is_some() {
            self.prepare_taa_frame();
        }

        // --- Ray-march pass ---
        if self.pipeline_taa.is_some() {
            let taa = self.taa.as_ref().expect("pipeline_taa implies TaaState");
            let pipeline_taa = self.pipeline_taa.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(march_label),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &taa.color_target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05, g: 0.05, b: 0.1, a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &taa.t_target_view,
                        resolve_target: None,
                        // Cleared to 0 — if the march misses a pixel
                        // (can't happen for a fullscreen triangle but
                        // wgpu requires a valid clear op) the resolve
                        // would see a tiny t and reject the reproject.
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline_taa);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        } else {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(march_label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dest_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.1, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Resolve pass (TAA only) ---
        if let Some(taa) = self.taa.as_ref() {
            let bg = taa.make_resolve_bind_group(&self.device);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa_resolve"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: dest_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: taa.history.write_view(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&taa.resolve_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Tick the Halton jitter, build resolve uniforms, and patch the
    /// jitter bytes of the existing camera buffer. See `GpuCamera` —
    /// `jitter_x_px` sits at byte offset 12, `jitter_y_px` at offset
    /// 28 (they occupy the old `_pad0` / `_pad1` slots).
    fn prepare_taa_frame(&mut self) {
        let signature = self.current_frame_signature();
        let last_camera = self.last_camera;
        let jitter = self
            .taa
            .as_mut()
            .unwrap()
            .begin_frame(&self.queue, &last_camera, signature);
        self.queue.write_buffer(&self.camera_buffer, 12, bytemuck::bytes_of(&jitter[0]));
        self.queue.write_buffer(&self.camera_buffer, 28, bytemuck::bytes_of(&jitter[1]));
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
        let frame_start = web_time::Instant::now();
        let acquire_start = web_time::Instant::now();
        let output = self.surface.get_current_texture()?;
        let acquire_elapsed = acquire_start.elapsed();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let encode_start = web_time::Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        if self.shader_stats_enabled {
            encoder.clear_buffer(&self.shader_stats_buffer, 0, None);
        }
        self.record_frame_passes(&mut encoder, &view, "ray_march");
        if self.shader_stats_enabled {
            encoder.copy_buffer_to_buffer(
                &self.shader_stats_buffer, 0,
                &self.shader_stats_readback, 0,
                64,
            );
        }
        let encode_elapsed = encode_start.elapsed();
        let done_slot: std::sync::Arc<std::sync::Mutex<Option<web_time::Instant>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let submit_start = web_time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        // wgpu 25's WebGPU backend has not implemented
        // `on_submitted_work_done`; skip the latency probe on WASM.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let done_slot = std::sync::Arc::clone(&done_slot);
            self.queue.on_submitted_work_done(move || {
                let mut slot = done_slot.lock().unwrap();
                *slot = Some(web_time::Instant::now());
            });
        }
        let submit_elapsed = submit_start.elapsed();
        // Swap TAA history textures for next frame. Must happen AFTER
        // the submit above — the resolve pass wrote into
        // `history.write_view()`, so end_frame's swap promotes that
        // to next frame's `read_view()`.
        if let Some(taa) = self.taa.as_mut() {
            taa.end_frame();
        }
        let present_start = web_time::Instant::now();
        output.present();
        let present_elapsed = present_start.elapsed();
        let frame_elapsed = frame_start.elapsed();
        // Poll + stats readback only when enabled, and only on slow
        // frames. The poll blocks the CPU on GPU completion, so we
        // don't want to stall every frame — but during a slowdown we
        // want the data.
        let slow = frame_elapsed.as_secs_f64() * 1000.0 >= 30.0;
        let (submitted_done_ms, shader_stats) =
            if self.shader_stats_enabled && slow {
                let _ = self.device.poll(wgpu::PollType::Wait);
                let submitted_done_ms = done_slot
                    .lock()
                    .ok()
                    .and_then(|s| s.map(|t| t.duration_since(submit_start).as_secs_f64() * 1000.0));
                let stats = self.read_shader_stats();
                (submitted_done_ms, stats)
            } else {
                (None, ShaderStatsFrame::default())
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
                "renderer_slow acquire_ms={:.2} encode_ms={:.2} submit_ms={:.2} present_ms={:.2} total_ms={:.2} submitted_done_ms={} rays={} hits={} miss={} max_iters={} avg_steps={:.1} max_steps={} avg_oob={:.1} avg_empty={:.1} avg_descend={:.1} avg_lod_terminal={:.1} avg_would_cull={:.2} avg_loads_total={:.1} avg_loads_tree={:.1} avg_loads_offsets={:.1} avg_loads_kinds={:.2} avg_loads_ribbon={:.2}",
                acquire_elapsed.as_secs_f64() * 1000.0,
                encode_elapsed.as_secs_f64() * 1000.0,
                submit_elapsed.as_secs_f64() * 1000.0,
                present_elapsed.as_secs_f64() * 1000.0,
                frame_elapsed.as_secs_f64() * 1000.0,
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
                shader_stats.avg_steps_would_cull(),
                shader_stats.avg_loads_total(),
                shader_stats.avg_loads_tree(),
                shader_stats.avg_loads_offsets(),
                shader_stats.avg_loads_kinds(),
                shader_stats.avg_loads_ribbon(),
            );
        }
        Ok(())
    }

    pub fn render_offscreen(&mut self) -> OffscreenRenderTiming {
        let frame_start = web_time::Instant::now();
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
        let view_start = web_time::Instant::now();
        let texture = self
            .offscreen_texture
            .as_ref()
            .expect("offscreen texture initialized");
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let view_ms = view_start.elapsed().as_secs_f64() * 1000.0;
        let encode_start = web_time::Instant::now();
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
        self.record_frame_passes(&mut encoder, &view, "offscreen-ray-march");
        if self.shader_stats_enabled {
            encoder.copy_buffer_to_buffer(
                &self.shader_stats_buffer, 0,
                &self.shader_stats_readback, 0,
                64,
            );
        }
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
        let done_slot: std::sync::Arc<std::sync::Mutex<Option<web_time::Instant>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let submit_start = web_time::Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        // Register the callback *after* submit: wgpu's
        // `on_submitted_work_done` fires when the queue is drained
        // up to the time of the call. Registering before submit
        // fires on the previous frame's completion (usually ~0 ms)
        // and tells us nothing about the current frame.
        // wgpu 25's WebGPU backend does not implement this — skip on WASM.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let done_slot = std::sync::Arc::clone(&done_slot);
            self.queue.on_submitted_work_done(move || {
                let mut slot = done_slot.lock().unwrap();
                *slot = Some(web_time::Instant::now());
            });
        }
        let submit_ms = submit_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(taa) = self.taa.as_mut() {
            taa.end_frame();
        }
        let wait_start = web_time::Instant::now();
        let _ = self.device.poll(wgpu::PollType::Wait);
        let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
        let submitted_done_ms = done_slot
            .lock()
            .ok()
            .and_then(|slot| slot.map(|t| t.duration_since(submit_start).as_secs_f64() * 1000.0));
        let shader_stats = if self.shader_stats_enabled {
            self.read_shader_stats()
        } else {
            ShaderStatsFrame::default()
        };
        let total_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        if total_ms >= 10.0 {
            eprintln!(
                "renderer_offscreen_slow size={}x{} texture_alloc_ms={:.2} view_ms={:.2} encode_ms={:.2} submit_ms={:.2} wait_ms={:.2} submitted_done_ms={} total_ms={:.2}",
                self.config.width, self.config.height,
                texture_alloc_ms, view_ms, encode_ms, submit_ms, wait_ms,
                submitted_done_ms.map(|v| format!("{v:.2}")).unwrap_or_else(|| "na".into()),
                total_ms,
            );
        }
        OffscreenRenderTiming {
            texture_alloc_ms, view_ms, encode_ms, submit_ms, wait_ms, total_ms,
            submitted_done_ms, shader_stats,
        }
    }

    /// Map `shader_stats_readback`, decode the counter u32s, and
    /// unmap. Called after the render-pass submit has completed on
    /// GPU (verified by the earlier `poll(Wait)`).
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
            sum_steps_would_cull_div4: read_u32(40),
            sum_loads_tree_div4: read_u32(44),
            sum_loads_offsets_div4: read_u32(48),
            sum_loads_kinds_div4: read_u32(52),
            sum_loads_ribbon_div4: read_u32(56),
            sum_steps_hits_div4: read_u32(60),
        };
        drop(data);
        self.shader_stats_readback.unmap();
        stats
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
