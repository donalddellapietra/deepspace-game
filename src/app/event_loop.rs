//! winit `ApplicationHandler` for the `App`.
//!
//! `resumed` creates the window + renderer + (previously-generated)
//! GPU state. `window_event` routes per-event behavior to the
//! submodule-provided methods on `App`. `device_event` handles
//! mouse-motion for camera look. All heavy lifting lives elsewhere
//! — this file is deliberately a thin dispatcher.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::{WindowAttributes, WindowId};

use crate::renderer::Renderer;
use crate::world::anchor::WorldPos;
use crate::world::edit;
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

        let (tree_data, root_index) = gpu::pack_tree(&self.world.library, self.world.root);
        let mut renderer = pollster::block_on(Renderer::new(window, &tree_data, root_index));
        if let Some(planet) = self.cs_planet.as_ref() {
            renderer.set_cubed_sphere_planet(
                planet.center,
                planet.inner_r,
                planet.outer_r,
                planet.depth,
            );
        }
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
    /// Mouse-wheel zoom: change the camera's anchor depth by one,
    /// then translate the camera along its forward ray so the block
    /// under the crosshair stays at the same apparent size. Each
    /// zoom step is 3× finer/coarser.
    ///
    /// The anchor depth is shifted via `WorldPos::zoom_in` /
    /// `zoom_out`, which preserve world-space position — only the
    /// representation changes. The 3× world-space dolly is a
    /// separate cosmetic move so the user sees "zoom" as zoom.
    fn handle_scroll_zoom(&mut self, delta: winit::event::MouseScrollDelta) {
        let y = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
        };
        // Positive y = scroll up = zoom in (finer = deeper anchor).
        let step: i32 = if y > 0.0 { 1 } else if y < 0.0 { -1 } else { return };
        // Clamp: zooming past root or past max depth is rejected.
        let new_anchor_depth = (self.anchor_depth() as i32 + step)
            .clamp(1, (self.tree_depth as i32).max(1));
        if new_anchor_depth == self.anchor_depth() as i32 { return; }

        // Compute dolly anchor: the world-space point under the
        // crosshair, or a fallback point ahead of the camera.
        let cam_world = self.camera.world_pos_f32();
        let ray_dir = self.camera.forward();
        let hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            cam_world,
            ray_dir,
            self.edit_depth(),
        );
        let anchor_world = if let Some(h) = hit {
            [
                cam_world[0] + ray_dir[0] * h.t,
                cam_world[1] + ray_dir[1] * h.t,
                cam_world[2] + ray_dir[2] * h.t,
            ]
        } else {
            let d = self.camera.cell_size() * 10.0;
            [
                cam_world[0] + ray_dir[0] * d,
                cam_world[1] + ray_dir[1] * d,
                cam_world[2] + ray_dir[2] * d,
            ]
        };

        // Dolly the camera along the ray by 3×^(−step).
        let scale = 3.0f32.powi(-step);
        let new_cam_world = [
            anchor_world[0] + (cam_world[0] - anchor_world[0]) * scale,
            anchor_world[1] + (cam_world[1] - anchor_world[1]) * scale,
            anchor_world[2] + (cam_world[2] - anchor_world[2]) * scale,
        ];
        self.camera.position = WorldPos::from_world_xyz(new_cam_world, new_anchor_depth as u8);

        self.apply_zoom();
    }

    /// One full frame: webview sync, per-frame state push, physics,
    /// editing queries, GPU repack, render.
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
