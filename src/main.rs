//! Deep Space — ray-marched voxel engine.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use deepspace_game::game_state::GameUiState;
use deepspace_game::renderer::Renderer;
use deepspace_game::world::collision::{self, PlayerPhysics};
use deepspace_game::world::edit;
use deepspace_game::world::gpu::{self, GpuCamera};
use deepspace_game::world::state::WorldState;

#[cfg(not(target_arch = "wasm32"))]
use deepspace_game::overlay;

// ------------------------------------------------------------ Camera

struct Camera {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [-sy * cp, sp, -cy * cp]
    }

    fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, -sy]
    }

    fn forward_xz(&self) -> [f32; 3] {
        [-self.yaw.sin(), 0.0, -self.yaw.cos()]
    }

    fn gpu_camera(&self, fov: f32) -> GpuCamera {
        let fwd = self.forward();
        let r = self.right();
        let up = [
            r[1] * fwd[2] - r[2] * fwd[1],
            r[2] * fwd[0] - r[0] * fwd[2],
            r[0] * fwd[1] - r[1] * fwd[0],
        ];
        GpuCamera {
            pos: self.pos,
            _pad0: 0.0,
            forward: fwd,
            _pad1: 0.0,
            right: r,
            _pad2: 0.0,
            up,
            fov,
        }
    }
}

// ------------------------------------------------------------ Input

#[derive(Default)]
struct Keys {
    w: bool, a: bool, s: bool, d: bool,
    space: bool, shift: bool,
}

impl Keys {
    fn apply(&mut self, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::KeyW => self.w = pressed,
            KeyCode::KeyA => self.a = pressed,
            KeyCode::KeyS => self.s = pressed,
            KeyCode::KeyD => self.d = pressed,
            KeyCode::Space => self.space = pressed,
            KeyCode::ShiftLeft => self.shift = pressed,
            _ => {}
        }
    }

    fn clear(&mut self) {
        *self = Keys::default();
    }
}

// ------------------------------------------------------------ App

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    world: WorldState,
    cursor_locked: bool,
    keys: Keys,
    last_frame: std::time::Instant,
    zoom_level: i32,
    /// Cached tree depth (recomputed only after edits).
    tree_depth: u32,
    physics: PlayerPhysics,
    ui: GameUiState,
    #[cfg(not(target_arch = "wasm32"))]
    webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    frames_waited: u32,
}

#[cfg(not(target_arch = "wasm32"))]
const WAIT_FRAMES: u32 = 10;

impl App {
    fn new() -> Self {
        let world = deepspace_game::world::worldgen::generate_world();
        let tree_depth = world.tree_depth();

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                pos: [1.5, 1.75, 1.5],
                yaw: 0.0,
                pitch: 0.0,
            },
            world,
            cursor_locked: false,
            keys: Keys::default(),
            last_frame: std::time::Instant::now(),
            zoom_level: 0,
            tree_depth,
            physics: PlayerPhysics::default(),
            ui: GameUiState::new(),
            #[cfg(not(target_arch = "wasm32"))]
            webview: None,
            #[cfg(not(target_arch = "wasm32"))]
            frames_waited: 0,
        }
    }

    fn update(&mut self, dt: f32) {
        // Speed: ~5 interaction-layer cells/second regardless of zoom.
        let td = self.tree_depth as i32;
        let cell_size = 1.0 / 3.0f32.powi(td - self.zoom_level);
        let speed = 5.0 * cell_size;

        let fwd = self.camera.forward_xz();
        let right = self.camera.right();

        let mut dx = 0.0f32;
        let mut dz = 0.0f32;

        if self.keys.w { dx += fwd[0]; dz += fwd[2]; }
        if self.keys.s { dx -= fwd[0]; dz -= fwd[2]; }
        if self.keys.d { dx += right[0]; dz += right[2]; }
        if self.keys.a { dx -= right[0]; dz -= right[2]; }

        // Jump.
        if self.keys.space {
            self.physics.jump();
        }

        // Normalize horizontal movement.
        let len = (dx * dx + dz * dz).sqrt();
        let move_xz = if len > 0.001 {
            let s = speed * dt / len;
            [dx * s, dz * s]
        } else {
            [0.0, 0.0]
        };

        // Swept-AABB collision against the tree.
        let edit_depth = self.edit_depth();
        let root = self.world.root;
        collision::move_and_collide(
            &mut self.camera.pos,
            &mut self.physics,
            move_xz,
            dt,
            cell_size,
            &self.world.library,
            root,
            edit_depth,
        );

        // Camera is at eye height above feet.
        let eye_height = collision::PLAYER_H * cell_size * 0.9;
        let mut gpu_pos = self.camera.pos;
        gpu_pos[1] += eye_height;

        if let Some(renderer) = &self.renderer {
            let mut cam = self.camera.gpu_camera(1.2);
            cam.pos = gpu_pos;
            renderer.update_camera(&cam);
        }
    }

    /// CPU raycast depth: tree_depth - zoom_level.
    /// zoom_level 0 = finest blocks, higher = coarser.
    fn edit_depth(&self) -> u32 {
        let td = self.tree_depth as i32;
        (td - self.zoom_level).max(1) as u32
    }

    /// GPU visual depth: edit_depth + 3 (see 27×27×27 detail).
    fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(8)
    }

    /// Clamp zoom and sync GPU depth.
    fn apply_zoom(&mut self) {
        let td = self.tree_depth as i32;
        self.zoom_level = self.zoom_level.clamp(0, (td - 1).max(0));
        self.ui.zoom_level = self.zoom_level;
        let vd = self.visual_depth();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
        }
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}",
            self.zoom_level, td, self.edit_depth(), vd
        );
    }

    fn eye_pos(&self) -> [f32; 3] {
        let td = self.tree_depth as i32;
        let cell_size = 1.0 / 3.0f32.powi(td - self.zoom_level);
        let eye_height = collision::PLAYER_H * cell_size * 0.9;
        [self.camera.pos[0], self.camera.pos[1] + eye_height, self.camera.pos[2]]
    }

    fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let hit = edit::cpu_raycast(
            &self.world.library, self.world.root,
            self.eye_pos(), ray_dir, self.edit_depth(),
        );
        if let Some(hit) = hit {
            if edit::break_block(&mut self.world, &hit) {
                self.upload_tree();
            }
        }
    }

    fn do_place(&mut self) {
        let Some(block_type) = self.ui.active_block_type() else { return };
        let ray_dir = self.camera.forward();
        let hit = edit::cpu_raycast(
            &self.world.library, self.world.root,
            self.eye_pos(), ray_dir, self.edit_depth(),
        );
        if let Some(hit) = hit {
            if edit::place_block(&mut self.world, &hit, block_type) {
                self.upload_tree();
            }
        }
    }

    fn upload_tree(&mut self) {
        self.tree_depth = self.world.tree_depth();
        self.upload_tree_lod();
    }

    /// Re-pack and upload the tree with LOD culling based on camera position.
    /// Called every frame so distant terrain stays flattened as the camera moves.
    fn upload_tree_lod(&mut self) {
        let (tree_data, root_index) = gpu::pack_tree_lod(
            &self.world.library,
            self.world.root,
            self.camera.pos,
            1440.0, // approximate screen height
            1.2,    // fov
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, root_index);
        }
    }

    fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let ray_dir = self.camera.forward();
        let hit = edit::cpu_raycast(
            &self.world.library, self.world.root,
            self.eye_pos(), ray_dir, self.edit_depth(),
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(hit.as_ref().map(edit::hit_aabb));
        }
    }

    fn lock_cursor(&mut self) {
        let Some(window) = &self.window else { return };
        self.cursor_locked = true;
        let _ = window.set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        window.set_cursor_visible(false);
        #[cfg(not(target_arch = "wasm32"))]
        {
            overlay::clear_passthrough();
            overlay::refocus_content_view(window);
        }
    }

    fn unlock_cursor(&mut self) {
        let Some(window) = &self.window else { return };
        self.cursor_locked = false;
        self.keys.clear();
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
    }

    fn sync_cursor_to_panels(&mut self) {
        if self.ui.any_panel_open() && self.cursor_locked {
            self.unlock_cursor();
        } else if !self.ui.any_panel_open() && !self.cursor_locked {
            self.lock_cursor();
        }
    }

    // ── Overlay integration (native only) ────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    fn try_create_webview(&mut self) {
        if self.webview.is_some() { return; }
        self.frames_waited += 1;
        if self.frames_waited < WAIT_FRAMES { return; }
        let Some(window) = &self.window else { return };
        if let Some(wv) = overlay::create_webview(window) {
            self.webview = Some(wv);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn inject_webview_input(&mut self) {
        for (code, pressed) in overlay::drain_forwarded_keys() {
            if let Some(key) = overlay::js_code_to_keycode(&code) {
                self.apply_key(key, pressed);
            }
        }
        for (button, pressed) in overlay::drain_forwarded_mouse() {
            if pressed {
                if let Some(btn) = overlay::js_button_to_mouse(button) {
                    self.apply_mouse(btn);
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_ui_commands(&mut self) {
        for cmd in overlay::poll_commands() {
            let panel_changed = self.ui.handle_command(cmd);
            if panel_changed { self.sync_cursor_to_panels(); }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn flush_overlay(&self) {
        if let Some(ref wv) = self.webview {
            overlay::flush_to_webview(wv);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resize_overlay(&self) {
        if let Some(ref wv) = self.webview {
            if let Some(window) = &self.window {
                overlay::resize_webview(wv, window);
            }
        }
    }

    // ── Unified input handlers ───────────────────────────────────

    fn apply_key(&mut self, code: KeyCode, pressed: bool) {
        self.keys.apply(code, pressed);

        if pressed && code == KeyCode::Escape {
            if self.ui.any_panel_open() {
                self.ui.handle_command(deepspace_game::bridge::UiCommand::CloseAllPanels);
                self.sync_cursor_to_panels();
            } else if self.cursor_locked {
                self.unlock_cursor();
            } else {
                self.lock_cursor();
            }
            return;
        }

        let panel_changed = self.ui.handle_key(code, pressed);
        if panel_changed { self.sync_cursor_to_panels(); }
    }

    fn apply_mouse(&mut self, button: MouseButton) {
        if !self.cursor_locked {
            if !self.ui.any_panel_open() {
                self.lock_cursor();
            }
        } else {
            match button {
                MouseButton::Left => self.do_break(),
                MouseButton::Right => self.do_place(),
                _ => {}
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = WindowAttributes::default()
            .with_title("Deep Space")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        let (tree_data, root_index) = gpu::pack_tree(&self.world.library, self.world.root);
        let renderer = pollster::block_on(Renderer::new(window, &tree_data, root_index));
        self.renderer = Some(renderer);
        self.last_frame = std::time::Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer { r.resize(size.width, size.height); }
                #[cfg(not(target_arch = "wasm32"))]
                self.resize_overlay();
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state, .. }, ..
            } => {
                let pressed = state == ElementState::Pressed;
                self.apply_key(code, pressed);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                // Scroll up = zoom in (finer), scroll down = zoom out (coarser).
                if y > 0.0 { self.zoom_level -= 1; }
                else if y < 0.0 { self.zoom_level += 1; }
                self.apply_zoom();
            }

            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                self.apply_mouse(button);
            }

            WindowEvent::RedrawRequested => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.try_create_webview();
                    self.inject_webview_input();
                    self.poll_ui_commands();
                    self.ui.push_to_overlay();
                    self.flush_overlay();
                }

                let now = std::time::Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;
                self.update(dt);
                self.upload_tree_lod(); // LOD repack every frame
                self.update_highlight();

                if let Some(renderer) = &self.renderer {
                    match renderer.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = self.window.as_ref().unwrap().inner_size();
                            if let Some(r) = &mut self.renderer { r.resize(size.width, size.height); }
                        }
                        Err(e) => log::error!("Render error: {e:?}"),
                    }
                }
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
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

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
