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

        // Initial pack uses the non-LOD `pack_tree` to get a
        // full-fidelity buffer on first upload; `apply_zoom` below
        // immediately re-packs with the ribbon-aware path.
        let (tree_data, node_kinds, root_index) =
            gpu::pack_tree(&self.world.library, self.world.root);
        // Single-entry ribbon pointing at the packed root — correct
        // for "no camera yet / no ribbon computed." `apply_zoom`
        // overwrites this below.
        let initial_ribbon = [gpu::GpuRibbonFrame {
            root_index,
            _pad0: 0,
            world_scale: 1.0,
            _pad1: 0,
            camera_local: [1.5, 1.75, 1.5, 0.0],
        }];
        let renderer = pollster::block_on(
            Renderer::new(window, &tree_data, &node_kinds, &initial_ribbon),
        );
        self.renderer = Some(renderer);
        self.upload_tree_lod();
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
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
