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
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Deep Space")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        #[cfg(not(target_arch = "wasm32"))]
        crate::platform::prepare_window(&window);

        let (tree_data, node_kinds, root_index) =
            gpu::pack_tree(&self.world.library, self.world.root);
        let renderer = pollster::block_on(
            Renderer::new(window, &tree_data, &node_kinds, root_index),
        );
        self.renderer = Some(renderer);
        self.apply_zoom();
        self.last_frame = std::time::Instant::now();
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
}

impl App {
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
        let cur = self.anchor_depth() as i32;
        let max_depth = (self.tree_depth as i32).max(1);
        let new_depth = (cur + step).clamp(1, max_depth);
        if new_depth == cur { return; }
        if step > 0 {
            self.camera.position.zoom_in();
        } else {
            self.camera.position.zoom_out();
        }
        self.apply_zoom();
    }

    fn handle_redraw(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
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
                    camera_pos: self.camera.world_pos_f32(),
                    fov: 1.2,
                    node_count: self.world.library.len(),
                },
            ));
            self.flush_overlay();
        }

        let now = std::time::Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        if dt > 0.0 {
            let instant_fps = 1.0 / dt as f64;
            let alpha = (dt as f64 * 5.0).min(1.0);
            self.fps_smooth = self.fps_smooth * (1.0 - alpha) + instant_fps * alpha;
        }
        self.update(dt);
        self.upload_tree_lod();
        self.update_highlight();

        if let Some(renderer) = &self.renderer {
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

        // Test driver runs AFTER the frame so the captured image
        // reflects what just rendered.
        self.tick_test_runner_after_frame();

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    fn tick_test_runner_after_frame(&mut self) {
        // Borrow checker: collect commands into a local first.
        let (due, frame, frame_budget_done, timed_out, exit_after, screenshot) = {
            let Some(test) = self.test.as_mut() else { return };
            test.frame += 1;
            let frame = test.frame;
            let due = test.drain_due();
            let frame_budget_done = frame + 1 >= test.exit_after_frames;
            let timed_out = test.timed_out();
            let exit_after = test.exit_after_frames;
            let screenshot = test.screenshot_path.clone();
            (due, frame, frame_budget_done, timed_out, exit_after, screenshot)
        };
        for cmd in due {
            match cmd {
                super::test_runner::ScriptCmd::Break => self.do_break(),
                super::test_runner::ScriptCmd::Place => self.do_place(),
                super::test_runner::ScriptCmd::Wait(_) => {}
            }
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
        if timed_out {
            eprintln!("test_runner: timeout reached at frame {frame}, quitting");
            std::process::exit(0);
        } else if frame >= exit_after {
            eprintln!("test_runner: exit_after_frames={frame} reached, quitting");
            std::process::exit(0);
        }
    }
}
