//! winit `ApplicationHandler` for the `App`.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::{WindowAttributes, WindowId};

use crate::renderer::Renderer;
use crate::world::gpu;

use super::{App, PendingInit, UserEvent};

impl ApplicationHandler<UserEvent> for App {
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

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::RendererReady(renderer) => {
                self.finish_init(*renderer);
            }
            #[cfg(target_arch = "wasm32")]
            UserEvent::Resize(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width.max(1), size.height.max(1));
                }
            }
        }
    }
}

impl App {
    fn ensure_started(&mut self, event_loop: &ActiveEventLoop, source: &str) {
        if self.window.is_some() {
            // On WASM the window exists but renderer comes online
            // asynchronously — nothing more to do here.
            return;
        }
        let resumed_start = web_time::Instant::now();
        eprintln!("startup_perf {source}: begin");

        let attrs = WindowAttributes::default()
            .with_title("Deep Space")
            .with_inner_size(winit::dpi::LogicalSize::new(self.harness_width, self.harness_height))
            .with_visible(!self.render_harness || self.show_harness_window);

        let window_start = web_time::Instant::now();
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let window_elapsed = window_start.elapsed();
        eprintln!("startup_perf {source}: window_created ms={:.2}", window_elapsed.as_secs_f64() * 1000.0);
        self.window = Some(window.clone());

        // winit on wasm32 does not append the canvas to the document
        // body for us, and defaults the surface to 1×1 px. Append the
        // canvas, stamp the backing store to match the viewport, and
        // install a `window.onresize` closure that re-stamps + posts
        // `UserEvent::Resize` so the renderer follows the browser.
        #[cfg(target_arch = "wasm32")]
        wasm_canvas_setup(&window, self.proxy.clone());

        #[cfg(not(target_arch = "wasm32"))]
        let prepare_elapsed = if self.overlay_enabled() {
            let prepare_start = web_time::Instant::now();
            crate::platform::prepare_window(&window);
            prepare_start.elapsed()
        } else {
            web_time::Duration::ZERO
        };
        #[cfg(target_arch = "wasm32")]
        let prepare_elapsed = web_time::Duration::ZERO;
        eprintln!("startup_perf {source}: window_prepared ms={:.2}", prepare_elapsed.as_secs_f64() * 1000.0);

        let pack_start = web_time::Instant::now();
        let (tree_packed, node_kinds, node_offsets, aabbs, _node_ids, root_index) =
            gpu::pack_tree(&self.world.library, self.world.root);
        let pack_elapsed = pack_start.elapsed();
        eprintln!(
            "startup_perf {source}: tree_packed ms={:.2} nodes={} tree_u32s={}",
            pack_elapsed.as_secs_f64() * 1000.0,
            node_kinds.len(),
            tree_packed.len(),
        );

        let renderer_start = web_time::Instant::now();
        let present_mode = if self.low_latency_present {
            wgpu::PresentMode::AutoNoVsync
        } else {
            wgpu::PresentMode::AutoVsync
        };
        let shader_stats_enabled = self.shader_stats_enabled;
        let lod_pixel_threshold = self.lod_pixel_threshold;
        let live_sample_every = self.live_sample_every_frames;
        let taa_enabled = self.taa_enabled;
        // Default: entities enabled on every preset. `--no-entities`
        // flips the compile-time override so the shader's tag==3
        // branch DCEs for pure-fractal perf runs.
        let entities_enabled = !self.disable_entities;
        let entity_render_mode = self.entity_render_mode;
        let node_count = node_kinds.len();
        let tree_u32_count = tree_packed.len();

        self.pending_init = Some(PendingInit {
            source: source.to_string(),
            resumed_start,
            window_elapsed,
            prepare_elapsed,
            pack_elapsed,
            renderer_start,
            node_count,
            tree_u32_count,
        });

        #[cfg(not(target_arch = "wasm32"))]
        {
            let renderer = pollster::block_on(Renderer::new(
                window,
                &tree_packed,
                &node_kinds,
                &node_offsets,
                &aabbs,
                root_index,
                present_mode,
                shader_stats_enabled,
                lod_pixel_threshold,
                live_sample_every,
                taa_enabled,
                entities_enabled,
                entity_render_mode,
));
            self.finish_init(renderer);
        }

        #[cfg(target_arch = "wasm32")]
        {
            if self.renderer_init_started {
                return;
            }
            self.renderer_init_started = true;
            let proxy = self.proxy.clone();
            // Move owned copies into the future — slices can't cross the await.
            let tree_packed = tree_packed.to_vec();
            let node_kinds = node_kinds.to_vec();
            let node_offsets = node_offsets.to_vec();
            let aabbs = aabbs.to_vec();
            wasm_bindgen_futures::spawn_local(async move {
                // Re-stamp the canvas backing store right before
                // surface creation. Winit's request_inner_size on web
                // doesn't apply synchronously, so anything we set in
                // ensure_started can be reset by a resize event tick
                // before this spawn_local body actually runs.
                {
                    use winit::platform::web::WindowExtWebSys;
                    if let (Some(canvas), Some(web_window)) = (window.canvas(), web_sys::window()) {
                        let css_w = web_window.inner_width().ok()
                            .and_then(|v| v.as_f64()).unwrap_or(800.0);
                        let css_h = web_window.inner_height().ok()
                            .and_then(|v| v.as_f64()).unwrap_or(600.0);
                        let dpr = web_window.device_pixel_ratio();
                        canvas.set_width((css_w * dpr) as u32);
                        canvas.set_height((css_h * dpr) as u32);
                    }
                }
                let renderer = Renderer::new(
                    window,
                    &tree_packed,
                    &node_kinds,
                    &node_offsets,
                    &aabbs,
                    root_index,
                    present_mode,
                    shader_stats_enabled,
                    lod_pixel_threshold,
                    live_sample_every,
                    taa_enabled,
                    entities_enabled,
                    entity_render_mode,
                )
                .await;
                if proxy.send_event(UserEvent::RendererReady(Box::new(renderer))).is_err() {
                    log::error!("RendererReady: event loop already closed");
                }
            });
        }
    }

    /// Runs after the async (or sync) renderer init lands. Pulls the
    /// per-source timing breadcrumbs out of `pending_init` so the perf
    /// log line is identical on native and WASM.
    pub(super) fn finish_init(&mut self, mut renderer: Renderer) {
        let Some(pending) = self.pending_init.take() else {
            log::error!("finish_init called without pending_init — ignoring");
            return;
        };
        let renderer_elapsed = pending.renderer_start.elapsed();
        let source = pending.source;
        eprintln!("startup_perf {source}: renderer_created ms={:.2}", renderer_elapsed.as_secs_f64() * 1000.0);

        renderer.update_palette(&self.palette.to_gpu_palette());
        // Phase 3 Step 3.0: apply CLI curvature debug knob, if any.
        // `--curvature A` sets a constant per-step parabolic-drop
        // coefficient — the simplest form of the bend. Used to
        // validate the math on `--plain-world` before the k(altitude)
        // ramp lands in Step 3.4.
        if let Some(a) = self.startup_curvature_a {
            renderer.set_curvature_a(a);
        }
        // lat_max ≈ 72° = 1.26 rad — pole-band ban for the WrappedPlane
        // rotated-tangent-cube render. No-op for non-WrappedPlane frames.
        renderer.set_planet_lat_max(1.26);
        if self.render_harness {
            renderer.resize(self.harness_width, self.harness_height);
            eprintln!(
                "startup_perf {source}: harness_resize width={} height={}",
                self.harness_width,
                self.harness_height,
            );
        }
        self.renderer = Some(renderer);

        let zoom_start = web_time::Instant::now();
        self.apply_zoom();
        let zoom_elapsed = zoom_start.elapsed();
        eprintln!("startup_perf {source}: apply_zoom ms={:.2}", zoom_elapsed.as_secs_f64() * 1000.0);

        #[cfg(not(target_arch = "wasm32"))]
        if self.overlay_enabled() {
            self.frames_waited = crate::app::overlay_integration::WAIT_FRAMES;
            let overlay_start = web_time::Instant::now();
            self.try_create_webview();
            eprintln!(
                "startup_perf {source}: overlay_create ms={:.2}",
                overlay_start.elapsed().as_secs_f64() * 1000.0,
            );
        }

        self.last_frame = web_time::Instant::now();
        if let Some(window) = &self.window {
            window.request_redraw();
        }
        eprintln!(
            "startup_perf {source} total_ms={:.2} window_ms={:.2} prepare_ms={:.2} pack_ms={:.2} renderer_ms={:.2} apply_zoom_ms={:.2} nodes={} tree_u32s={} kinds={}",
            pending.resumed_start.elapsed().as_secs_f64() * 1000.0,
            pending.window_elapsed.as_secs_f64() * 1000.0,
            pending.prepare_elapsed.as_secs_f64() * 1000.0,
            pending.pack_elapsed.as_secs_f64() * 1000.0,
            renderer_elapsed.as_secs_f64() * 1000.0,
            zoom_elapsed.as_secs_f64() * 1000.0,
            pending.node_count,
            pending.tree_u32_count,
            pending.node_count,
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
        let frame_start = web_time::Instant::now();
        let overlay_start = frame_start;

        // wry/WKWebView pumping is native-only.
        #[cfg(not(target_arch = "wasm32"))]
        if self.overlay_enabled() {
            self.try_create_webview();
            self.inject_webview_input();
        }

        // Drain UI commands + push state on both platforms. Native:
        // wry IPC + evaluate_script. WASM: JS queues + window.__onGameState.
        if self.overlay_active() {
            self.poll_ui_commands();
            let camera_local = match self.active_frame.kind {
                crate::app::ActiveFrameKind::Cartesian
                | crate::app::ActiveFrameKind::WrappedPlane { .. } => {
                    self.camera.position.in_frame_rot(
                        &self.world.library,
                        self.world.root,
                        &self.active_frame.render_path,
                    )
                }
            };
            // Camera position in root-frame world coords (rotation-
            // aware so the value is correct even when the anchor
            // crosses a TangentBlock).
            let camera_root_xyz = self.camera.position.in_frame_rot(
                &self.world.library,
                self.world.root,
                &crate::world::anchor::Path::root(),
            );
            let anchor_depth = self.camera.position.anchor.depth();
            let anchor_cell_size_root =
                crate::world::anchor::WORLD_SIZE / 3.0_f32.powi(anchor_depth as i32);
            let anchor_slots_csv = self
                .camera
                .position
                .anchor
                .as_slice()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let active_frame_kind = match self.active_frame.kind {
                crate::app::ActiveFrameKind::Cartesian => "Cartesian".to_string(),
                crate::app::ActiveFrameKind::WrappedPlane { dims, slab_depth } => {
                    format!("WrappedPlane(dims={dims:?}, slab_d={slab_depth})")
                }
            };
            let render_path_csv = self
                .active_frame
                .render_path
                .as_slice()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let (tb_on_anchor_path, anchor_cumulative_yaw_deg) =
                tangent_block_chain_summary(
                    &self.world.library,
                    self.world.root,
                    &self.camera.position.anchor,
                );
            // Keep the UI's zoom_level in sync with the live anchor
            // depth. `edit_actions::zoom` updates it on explicit zoom
            // input, but startup spawns + bootstrap defaults need
            // this fallback or the on-screen "Layer N" indicator
            // stays stuck at 0 until the player presses a zoom key.
            self.ui.zoom_level = self.zoom_level();
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
                    camera_anchor_depth: anchor_depth as u32,
                    camera_local,
                    fov: 1.2,
                    node_count: self.world.library.len(),
                    camera_root_xyz,
                    anchor_cell_size_root,
                    anchor_slots_csv,
                    active_frame_kind,
                    render_path_csv,
                    tb_on_anchor_path,
                    anchor_cumulative_yaw_deg,
                    copy_seq: self.debug_copy_seq,
                },
            ));
        }

        // wry batch flush is native-only; WASM push_state is direct.
        #[cfg(not(target_arch = "wasm32"))]
        if self.overlay_enabled() {
            self.flush_overlay();
        }
        let overlay_elapsed = overlay_start.elapsed();

        let now = frame_start;
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        if dt > 0.0 {
            let instant_fps = 1.0 / dt as f64;
            let alpha = (dt as f64 * 5.0).min(1.0);
            self.fps_smooth = self.fps_smooth * (1.0 - alpha) + instant_fps * alpha;
        }
        let update_start = web_time::Instant::now();
        self.update(dt);
        let update_elapsed = update_start.elapsed();

        let upload_start = web_time::Instant::now();
        self.upload_tree_lod();
        let upload_elapsed = upload_start.elapsed();

        let highlight_start = web_time::Instant::now();
        self.update_highlight();
        let highlight_elapsed = highlight_start.elapsed();

        let render_start = web_time::Instant::now();
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
        if pre_tail_elapsed.as_secs_f64() * 1000.0 >= 30.0 {
            let frame_index = self.test.as_ref().map_or(0, |test| test.frame);
            eprintln!(
                "slow_frame frame={} total_ms={:.2} dt_ms={:.2} overlay_ms={:.2} update_ms={:.2} upload_ms={:.2} highlight_ms={:.2} render_ms={:.2} zoom_level={} anchor_depth={} visual_depth={} nodes={}",
                frame_index,
                pre_tail_elapsed.as_secs_f64() * 1000.0,
                dt as f64 * 1000.0,
                overlay_elapsed.as_secs_f64() * 1000.0,
                update_elapsed.as_secs_f64() * 1000.0,
                upload_elapsed.as_secs_f64() * 1000.0,
                highlight_elapsed.as_secs_f64() * 1000.0,
                render_elapsed.as_secs_f64() * 1000.0,
                self.zoom_level(),
                self.anchor_depth(),
                self.visual_depth(),
                self.world.library.len(),
            );
        }

        // Test driver runs AFTER the frame so the captured image
        // reflects what just rendered.
        let test_tail_start = web_time::Instant::now();
        self.tick_test_runner_after_frame();
        let test_tail_elapsed = test_tail_start.elapsed();

        let redraw_tail_start = web_time::Instant::now();
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

#[cfg(target_arch = "wasm32")]
fn wasm_canvas_setup(
    window: &Arc<winit::window::Window>,
    proxy: winit::event_loop::EventLoopProxy<UserEvent>,
) {
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;
    use winit::platform::web::WindowExtWebSys;

    let Some(canvas) = window.canvas() else {
        log::error!("wasm_canvas_setup: window.canvas() returned None");
        return;
    };
    let Some(web_window) = web_sys::window() else { return };
    let _ = web_window
        .document()
        .and_then(|d| d.body())
        .and_then(|b| b.append_child(&canvas).ok());

    fn viewport_size(w: &web_sys::Window) -> (u32, u32) {
        let css_w = w.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(800.0);
        let css_h = w.inner_height().ok().and_then(|v| v.as_f64()).unwrap_or(600.0);
        let dpr = w.device_pixel_ratio();
        ((css_w * dpr) as u32, (css_h * dpr) as u32)
    }

    let (phys_w, phys_h) = viewport_size(&web_window);
    canvas.set_width(phys_w);
    canvas.set_height(phys_h);
    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(phys_w, phys_h));
    log::info!("wasm: canvas {phys_w}x{phys_h} dpr={}", web_window.device_pixel_ratio());

    // Re-stamp canvas + post Resize on every browser-window resize.
    // The closure must be `forget()`-ed so it outlives this scope.
    let canvas_clone = canvas.clone();
    let web_window_clone = web_window.clone();
    let resize_cb = Closure::wrap(Box::new(move || {
        let (w, h) = viewport_size(&web_window_clone);
        canvas_clone.set_width(w);
        canvas_clone.set_height(h);
        let _ = proxy.send_event(UserEvent::Resize(winit::dpi::PhysicalSize::new(w, h)));
    }) as Box<dyn FnMut()>);
    let _ = web_window.add_event_listener_with_callback(
        "resize",
        resize_cb.as_ref().unchecked_ref(),
    );
    resize_cb.forget();
}

/// Walks the camera's anchor path from world root, looking for any
/// `NodeKind::TangentBlock` and accumulating its rotation. Returns
/// `(tb_on_path, cumulative_yaw_deg)` where:
/// - `tb_on_path` is `true` iff at least one TB was descended into.
/// - `cumulative_yaw_deg` is the cumulative Y-axis rotation in
///   degrees. For a single 45° TB on the path it's `45.0`. Useful
///   as a quick "the camera is in a rotated subtree" indicator
///   even when several TBs nest with non-Y rotations (read as
///   approximate yaw).
fn tangent_block_chain_summary(
    library: &crate::world::tree::NodeLibrary,
    world_root: crate::world::tree::NodeId,
    anchor: &crate::world::anchor::Path,
) -> (bool, f32) {
    use crate::world::tree::{Child, NodeKind};
    let mut rot: [[f32; 3]; 3] = crate::world::tree::IDENTITY_ROTATION;
    let mut tb_seen = false;
    let mut node = world_root;
    for k in 0..(anchor.depth() as usize) {
        let n = match library.get(node) {
            Some(n) => n,
            None => break,
        };
        let slot = anchor.slot(k) as usize;
        match n.children[slot] {
            Child::Node(child_id) => {
                if let Some(child_node) = library.get(child_id) {
                    if let NodeKind::TangentBlock { rotation: r } = child_node.kind {
                        tb_seen = true;
                        rot = matmul3x3_local(&rot, &r);
                    }
                }
                node = child_id;
            }
            _ => break,
        }
    }
    // Extract approximate yaw from rotation matrix. For a Y-axis
    // rotation R(θ): R[0][0] = cos(θ), R[2][0] = sin(θ). Take
    // atan2 of the column-0 X/Z components for column-major.
    let yaw_rad = rot[0][2].atan2(rot[0][0]);
    (tb_seen, yaw_rad.to_degrees())
}

fn matmul3x3_local(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for c in 0..3 {
        for r in 0..3 {
            let mut s = 0.0f32;
            for k in 0..3 {
                s += a[k][r] * b[c][k];
            }
            out[c][r] = s;
        }
    }
    out
}
