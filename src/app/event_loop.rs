//! winit `ApplicationHandler` for the `App`.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::{WindowAttributes, WindowId};

use crate::renderer::Renderer;
use crate::world::gpu;

use super::App;

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {}

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        eprintln!("startup_perf callback: resumed");
        self.ensure_started(event_loop, "resumed");
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.ensure_started(event_loop, "about_to_wait");
        if self.render_harness {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        } else if let Some(window) = &self.window {
            let gap_ms = self.last_frame.elapsed().as_secs_f64() * 1000.0;
            if gap_ms >= 50.0 {
                eprintln!(
                    "redraw_kick gap_ms={:.2} zoom_level={} anchor_depth={}",
                    gap_ms,
                    self.zoom_level(),
                    self.anchor_depth(),
                );
                window.request_redraw();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
                #[cfg(not(target_arch = "wasm32"))]
                self.resize_overlay();
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state, .. },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                self.apply_key(code, pressed);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_scroll_zoom(delta);
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                self.apply_mouse(button);
            }

            WindowEvent::RedrawRequested => {
                self.handle_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.cursor_locked {
                const SENS: f64 = 0.003;
                self.camera.yaw -= (delta.0 * SENS) as f32;
                self.camera.pitch = (self.camera.pitch - (delta.1 * SENS) as f32)
                    .clamp(-1.5, 1.5);
            }
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        eprintln!("startup_perf callback: exiting");
    }
}

impl App {
    fn ensure_started(&mut self, event_loop: &ActiveEventLoop, source: &str) {
        if self.window.is_some() {
            return;
        }
        let resumed_start = std::time::Instant::now();
        eprintln!("startup_perf {source}: begin");

        let attrs = WindowAttributes::default()
            .with_title("Deep Space")
            .with_inner_size(winit::dpi::LogicalSize::new(self.harness_width, self.harness_height))
            .with_visible(!self.render_harness || self.show_harness_window);

        let window_start = std::time::Instant::now();
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let window_elapsed = window_start.elapsed();
        eprintln!("startup_perf {source}: window_created ms={:.2}", window_elapsed.as_secs_f64() * 1000.0);
        self.window = Some(window.clone());

        #[cfg(not(target_arch = "wasm32"))]
        let prepare_elapsed = if self.overlay_enabled() {
            let prepare_start = std::time::Instant::now();
            crate::platform::prepare_window(&window);
            prepare_start.elapsed()
        } else {
            std::time::Duration::ZERO
        };
        eprintln!("startup_perf {source}: window_prepared ms={:.2}", prepare_elapsed.as_secs_f64() * 1000.0);
        #[cfg(target_arch = "wasm32")]
        let prepare_elapsed = std::time::Duration::ZERO;

        let pack_start = std::time::Instant::now();
        let (tree_packed, node_kinds, node_offsets, _node_ids, root_index) =
            gpu::pack_tree(&self.world.library, self.world.root);
        let pack_elapsed = pack_start.elapsed();
        eprintln!(
            "startup_perf {source}: tree_packed ms={:.2} nodes={} tree_u32s={}",
            pack_elapsed.as_secs_f64() * 1000.0,
            node_kinds.len(),
            tree_packed.len(),
        );
        let renderer_start = std::time::Instant::now();
        let shader_stats_enabled = self.shader_stats_enabled;
        let lod_pixel_threshold = self.lod_pixel_threshold;
        let lod_base_depth = self.lod_base_depth;
        let renderer = pollster::block_on(
            Renderer::new(
                window,
                &tree_packed,
                &node_kinds,
                &node_offsets,
                root_index,
                if self.low_latency_present {
                    wgpu::PresentMode::AutoNoVsync
                } else {
                    wgpu::PresentMode::AutoVsync
                },
                shader_stats_enabled,
                lod_pixel_threshold,
                lod_base_depth,
                self.live_sample_every_frames,
                self.render_scale,
            ),
        );
        let renderer_elapsed = renderer_start.elapsed();
        eprintln!("startup_perf {source}: renderer_created ms={:.2}", renderer_elapsed.as_secs_f64() * 1000.0);
        let mut renderer = renderer;
        // Push the App's color registry into the GPU palette buffer.
        // Bootstrap presets that import models (e.g. --vox-model) add
        // per-model colors to this registry; without this upload the
        // shader still sees GpuPalette::default() (builtins only) and
        // every imported voxel renders as palette[0] = transparent black.
        renderer.update_palette(&self.palette.to_gpu_palette());
        if self.render_harness {
            renderer.resize(self.harness_width, self.harness_height);
            eprintln!(
                "startup_perf {source}: harness_resize width={} height={}",
                self.harness_width,
                self.harness_height,
            );
        }
        self.renderer = Some(renderer);
        let zoom_start = std::time::Instant::now();
        self.apply_zoom();
        let zoom_elapsed = zoom_start.elapsed();
        eprintln!("startup_perf {source}: apply_zoom ms={:.2}", zoom_elapsed.as_secs_f64() * 1000.0);
        #[cfg(not(target_arch = "wasm32"))]
        if self.overlay_enabled() {
            self.frames_waited = crate::app::overlay_integration::WAIT_FRAMES;
            let overlay_start = std::time::Instant::now();
            self.try_create_webview();
            eprintln!(
                "startup_perf {source}: overlay_create ms={:.2}",
                overlay_start.elapsed().as_secs_f64() * 1000.0,
            );
        }
        self.last_frame = std::time::Instant::now();
        if let Some(window) = &self.window {
            window.request_redraw();
        }
        eprintln!(
            "startup_perf {source} total_ms={:.2} window_ms={:.2} prepare_ms={:.2} pack_ms={:.2} renderer_ms={:.2} apply_zoom_ms={:.2} nodes={} tree_u32s={} kinds={}",
            resumed_start.elapsed().as_secs_f64() * 1000.0,
            window_elapsed.as_secs_f64() * 1000.0,
            prepare_elapsed.as_secs_f64() * 1000.0,
            pack_elapsed.as_secs_f64() * 1000.0,
            renderer_elapsed.as_secs_f64() * 1000.0,
            zoom_elapsed.as_secs_f64() * 1000.0,
            node_kinds.len(),
            tree_packed.len(),
            node_kinds.len(),
        );
    }

    /// Mouse-wheel zoom: change the camera's anchor depth by one.
    /// Pure anchor mutation via `WorldPos::zoom_in` / `zoom_out` —
    /// the world-space position of the camera is preserved exactly.
    /// No dolly, no translation; only the depth scale changes.
    fn handle_scroll_zoom(&mut self, delta: winit::event::MouseScrollDelta) {
        if self.frozen { return; }
        let y = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
        };
        // Positive y = scroll up = zoom in (deeper anchor).
        let step: i32 = if y > 0.0 { 1 } else if y < 0.0 { -1 } else { return };
        self.zoom_anchor(step);
    }

    fn handle_redraw(&mut self) {
        #[cfg(target_os = "macos")]
        {
            objc2::rc::autoreleasepool(|_| self.handle_redraw_inner());
            return;
        }
        #[cfg(not(target_os = "macos"))]
        self.handle_redraw_inner();
    }

    fn handle_redraw_inner(&mut self) {
        let frame_start = std::time::Instant::now();
        let overlay_start = frame_start;
        #[cfg(not(target_arch = "wasm32"))]
        if self.overlay_enabled() {
            let camera_local = match self.active_frame.kind {
                crate::app::ActiveFrameKind::Sphere(sphere) => {
                    self.camera.position.in_frame(&sphere.body_path)
                }
                crate::app::ActiveFrameKind::Cartesian | crate::app::ActiveFrameKind::Body { .. } => {
                    self.camera.position.in_frame(&self.active_frame.render_path)
                }
            };
            self.try_create_webview();
            self.inject_webview_input();
            self.poll_ui_commands();
            self.ui.push_to_overlay(&self.palette);
            crate::overlay::push_state(&crate::bridge::GameStateUpdate::DebugOverlay(
                crate::bridge::DebugOverlayStateJs {
                    visible: self.debug_overlay_visible,
                    fps: self.fps_smooth,
                    frame_time_ms: if self.fps_smooth > 0.0 {
                        1000.0 / self.fps_smooth
                    } else {
                        0.0
                    },
                    zoom_level: self.zoom_level(),
                    tree_depth: self.tree_depth,
                    edit_depth: self.edit_depth(),
                    visual_depth: self.visual_depth(),
                    camera_anchor_depth: self.camera.position.anchor.depth() as u32,
                    camera_local,
                    fov: 1.2,
                    node_count: self.world.library.len(),
                },
            ));
            self.flush_overlay();
        }
        #[cfg(not(target_arch = "wasm32"))]
        let overlay_elapsed = overlay_start.elapsed();
        #[cfg(target_arch = "wasm32")]
        let overlay_elapsed = std::time::Duration::ZERO;

        let now = frame_start;
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        if dt > 0.0 {
            let instant_fps = 1.0 / dt as f64;
            let alpha = (dt as f64 * 5.0).min(1.0);
            self.fps_smooth = self.fps_smooth * (1.0 - alpha) + instant_fps * alpha;
        }
        let update_start = std::time::Instant::now();
        self.update(dt);
        let update_elapsed = update_start.elapsed();

        let upload_start = std::time::Instant::now();
        self.upload_tree_lod();
        let upload_elapsed = upload_start.elapsed();

        let highlight_start = std::time::Instant::now();
        self.update_highlight();
        let highlight_elapsed = highlight_start.elapsed();

        let render_start = std::time::Instant::now();
        if let Some(renderer) = &mut self.renderer {
            match renderer.render() {
                Ok(()) => {}
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    let size = self.window.as_ref().unwrap().inner_size();
                    if let Some(r) = &mut self.renderer {
                        r.resize(size.width, size.height);
                    }
                }
                Err(e) => log::error!("Render error: {e:?}"),
            }
        }
        if let Some(renderer) = self.renderer.as_ref() {
            self.last_cursor_hit = Some(renderer.read_cursor_probe());
        }
        let render_elapsed = render_start.elapsed();

        let pre_tail_elapsed = frame_start.elapsed();
        if let Some(test) = self.test.as_mut() {
            let mut frame_sample = None;
            let mut cadence_sample = None;
            if test.frame >= test.fps_warmup_frames {
                test.perf_samples.record_frame(pre_tail_elapsed.as_secs_f64());
                frame_sample = Some(pre_tail_elapsed.as_secs_f64());
            }
            if test.frame >= test.cadence_warmup_frames {
                test.perf_samples.record_cadence(dt as f64);
                cadence_sample = Some(dt as f64);
            }
            test.monitor.record_frame(test.started_at.elapsed(), frame_sample, cadence_sample);
        }

        if self.startup_profile_frames < 12 {
            eprintln!(
                "startup_perf frame={} total_ms={:.2} dt_ms={:.2} fps_est={:.1} overlay_ms={:.2} update_ms={:.2} upload_ms={:.2} highlight_ms={:.2} render_ms={:.2}",
                self.startup_profile_frames,
                pre_tail_elapsed.as_secs_f64() * 1000.0,
                dt as f64 * 1000.0,
                if pre_tail_elapsed.as_secs_f64() > 0.0 { 1.0 / pre_tail_elapsed.as_secs_f64() } else { 0.0 },
                overlay_elapsed.as_secs_f64() * 1000.0,
                update_elapsed.as_secs_f64() * 1000.0,
                upload_elapsed.as_secs_f64() * 1000.0,
                highlight_elapsed.as_secs_f64() * 1000.0,
                render_elapsed.as_secs_f64() * 1000.0,
            );
        }
        self.startup_profile_frames = self.startup_profile_frames.saturating_add(1);
        // Slow-frame diagnostic: fires on any frame ≥10 ms, AND on
        // the next 4 frames after any slow frame so we can see the
        // recovery tail. Edit frames in a 20-layer soldier world
        // are the main source; this breakdown shows whether the
        // cost is in pack, upload (tree_write + bg_rebuild),
        // ribbon, or render internals (encode/submit/wait/gpu_pass).
        let total_ms = pre_tail_elapsed.as_secs_f64() * 1000.0;
        if total_ms >= 10.0 {
            self.slow_frame_tail = 4;
        }
        if total_ms >= 10.0 || self.slow_frame_tail > 0 {
            let frame_index = self.test.as_ref().map_or(0, |test| test.frame);
            let (tw, bg, rw, cw, gpu_pass, enc_ms, sub_ms, wait_ms) = self
                .renderer
                .as_ref()
                .map(|r| (
                    r.last_tree_write_ms,
                    r.last_bind_group_rebuild_ms,
                    r.last_ribbon_write_ms,
                    r.last_camera_write_ms,
                    r.last_gpu_pass_ms,
                    r.last_render_encode_ms,
                    r.last_render_submit_ms,
                    r.last_render_wait_ms,
                ))
                .unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
            let is_edit = !self.last_reused_gpu_tree && self.last_pack_ms < 10.0;
            eprintln!(
                "frame_breakdown frame={} total_ms={:.2} dt_ms={:.2} edit={} \
                 overlay_ms={:.2} update_ms={:.2} upload_ms={:.2} pack_ms={:.2} ribbon_build_ms={:.2} \
                 tree_write_ms={:.2} bg_rebuild_ms={:.2} ribbon_write_ms={:.2} camera_write_ms={:.2} \
                 highlight_ms={:.2} render_ms={:.2} render_encode_ms={:.2} render_submit_ms={:.2} render_wait_ms={:.2} gpu_pass_ms={:.2} \
                 zoom={} anchor_depth={} visual_depth={} lib_nodes={}",
                frame_index,
                total_ms,
                dt as f64 * 1000.0,
                is_edit,
                overlay_elapsed.as_secs_f64() * 1000.0,
                update_elapsed.as_secs_f64() * 1000.0,
                upload_elapsed.as_secs_f64() * 1000.0,
                self.last_pack_ms,
                self.last_ribbon_build_ms,
                tw, bg, rw, cw,
                highlight_elapsed.as_secs_f64() * 1000.0,
                render_elapsed.as_secs_f64() * 1000.0,
                enc_ms, sub_ms, wait_ms, gpu_pass,
                self.zoom_level(),
                self.anchor_depth(),
                self.visual_depth(),
                self.world.library.len(),
            );
            self.slow_frame_tail = self.slow_frame_tail.saturating_sub(1);
        }

        // Test driver runs AFTER the frame so the captured image
        // reflects what just rendered.
        let test_tail_start = std::time::Instant::now();
        self.tick_test_runner_after_frame();
        let test_tail_elapsed = test_tail_start.elapsed();

        let redraw_tail_start = std::time::Instant::now();
        if let Some(w) = &self.window {
            w.request_redraw();
        }
        let redraw_tail_elapsed = redraw_tail_start.elapsed();
        let frame_elapsed = frame_start.elapsed();
        if redraw_tail_elapsed.as_secs_f64() * 1000.0 >= 10.0
            || test_tail_elapsed.as_secs_f64() * 1000.0 >= 10.0
            || frame_elapsed.as_secs_f64() * 1000.0 >= 30.0
        {
            let frame_index = self.test.as_ref().map_or(0, |test| test.frame);
            eprintln!(
                "frame_tail frame={} pre_tail_ms={:.2} test_tail_ms={:.2} request_redraw_ms={:.2} handler_total_ms={:.2} dt_ms={:.2}",
                frame_index,
                pre_tail_elapsed.as_secs_f64() * 1000.0,
                test_tail_elapsed.as_secs_f64() * 1000.0,
                redraw_tail_elapsed.as_secs_f64() * 1000.0,
                frame_elapsed.as_secs_f64() * 1000.0,
                dt as f64 * 1000.0,
            );
        }

    }

    pub(super) fn zoom_anchor(&mut self, step: i32) {
        if step == 0 { return; }
        let cur = self.anchor_depth() as i32;
        let max_depth = crate::world::tree::MAX_DEPTH as i32;
        let new_depth = (cur + step).clamp(1, max_depth);
        if new_depth == cur { return; }
        if step > 0 {
            self.camera.position.zoom_in();
        } else {
            self.camera.position.zoom_out();
        }
        self.apply_zoom();
    }
}

impl App {
    fn print_perf_summary(&self) {
        if let Some(test) = self.test.as_ref() {
            eprintln!(
                "test_runner: perf summary samples={} avg_frame_fps={:.2} avg_cadence_fps={:.2} worst_frame_ms={:.2} worst_dt_ms={:.2}",
                test.perf_samples.count,
                test.perf_samples.avg_frame_fps().unwrap_or(0.0),
                test.perf_samples.avg_cadence_fps().unwrap_or(0.0),
                test.perf_samples.worst_frame_secs * 1000.0,
                test.perf_samples.worst_cadence_secs * 1000.0,
            );
        }
    }

    fn tick_test_runner_after_frame(&mut self) {
        // Borrow checker: collect commands into a local first.
        let (due, frame, frame_budget_done, timed_out, perf_active, exit_after, screenshot) = {
            let Some(test) = self.test.as_mut() else { return };
            test.frame += 1;
            let frame = test.frame;
            let due = test.drain_due();
            let frame_budget_done = frame + 1 >= test.exit_after_frames;
            let timed_out = test.timed_out();
            let perf_active = test.min_fps.is_some() || test.min_cadence_fps.is_some();
            let exit_after = test.exit_after_frames;
            let screenshot = test.screenshot_path.clone();
            (due, frame, frame_budget_done, timed_out, perf_active, exit_after, screenshot)
        };
        for cmd in due {
            self.handle_script_cmd(cmd, frame);
        }
        if let Some(path) = screenshot {
            let already_done = self.test.as_ref().is_some_and(|t| t.screenshot_done);
            if !already_done && (frame_budget_done || timed_out) {
                if let Some(r) = &mut self.renderer {
                    match r.capture_to_png(&path) {
                        Ok(()) => eprintln!("test_runner: screenshot saved to {path}"),
                        Err(e) => eprintln!("test_runner: screenshot failed: {e}"),
                    }
                }
                if let Some(t) = self.test.as_mut() {
                    t.screenshot_done = true;
                }
            }
        }
        let perf_failed = self.test.as_ref().is_some_and(|test| {
            let frame_failed = if let Some(min_fps) = test.min_fps {
                if test.frame < test.fps_warmup_frames || test.perf_samples.count == 0 {
                    false
                } else {
                    test.perf_samples.avg_frame_fps().is_some_and(|avg_fps| avg_fps < min_fps)
                }
            } else {
                false
            };
            let cadence_failed = if let Some(min_fps) = test.min_cadence_fps {
                if test.frame < test.cadence_warmup_frames || test.perf_samples.total_cadence_secs <= 0.0 {
                    false
                } else {
                    test.perf_samples.avg_cadence_fps().is_some_and(|avg_fps| avg_fps < min_fps)
                }
            } else {
                false
            };
            frame_failed || cadence_failed
        });
        if let Some(test) = self.test.as_ref() {
            if perf_failed {
                use std::sync::atomic::Ordering;
                test.monitor.perf_failed.store(true, Ordering::Relaxed);
            }
        }
        if frame % 60 == 0 {
            if let Some(test) = self.test.as_ref() {
                eprintln!(
                    "test_runner: heartbeat frame={} zoom_level={} anchor_depth={} avg_frame_fps={:.2} avg_cadence_fps={:.2}",
                    frame,
                    self.zoom_level(),
                    self.anchor_depth(),
                    test.perf_samples.avg_frame_fps().unwrap_or(0.0),
                    test.perf_samples.avg_cadence_fps().unwrap_or(0.0),
                );
            }
        }
        if timed_out {
            self.print_perf_summary();
            if perf_active {
                eprintln!("test_runner: timeout reached before satisfying perf test at frame {frame}, quitting");
                std::process::exit(1);
            } else {
                eprintln!("test_runner: timeout reached at frame {frame}, quitting");
                std::process::exit(0);
            }
        } else if frame >= exit_after {
            self.print_perf_summary();
            if perf_failed {
                eprintln!("test_runner: perf threshold failed at frame {frame}, quitting");
                std::process::exit(1);
            }
            eprintln!("test_runner: exit_after_frames={frame} reached, quitting");
            std::process::exit(0);
        }
    }
}
