//! Deep Space — ray-marched voxel engine.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use deepspace_game::renderer::Renderer;
use deepspace_game::world::gpu::{self, GpuCamera};
use deepspace_game::world::state::WorldState;

// ------------------------------------------------------------ Camera

struct Camera {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

impl Camera {
    /// Forward direction (into screen at yaw=0).
    fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [-sy * cp, sp, -cy * cp]
    }

    /// Right direction (perpendicular to forward in XZ plane).
    fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, -sy]
    }

    /// Horizontal forward (no pitch, for WASD movement).
    fn forward_xz(&self) -> [f32; 3] {
        [-self.yaw.sin(), 0.0, -self.yaw.cos()]
    }

    fn gpu_camera(&self) -> GpuCamera {
        let fwd = self.forward();
        let r = self.right();
        // up = right × forward
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
            fov: 1.2,
        }
    }
}

// ------------------------------------------------------------ Input

#[derive(Default)]
struct Keys {
    w: bool, a: bool, s: bool, d: bool,
    space: bool, shift: bool,
}

// ------------------------------------------------------------ App

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    tree_data: Vec<gpu::GpuChild>,
    root_index: u32,
    cursor_locked: bool,
    keys: Keys,
    last_frame: std::time::Instant,
    zoom_level: i32,
}

impl App {
    fn new() -> Self {
        let world = WorldState::test_world();
        let (tree_data, root_index) = gpu::pack_tree(&world.library, world.root);
        log::info!(
            "World: {} nodes, root_index={}",
            tree_data.len() / 27, root_index,
        );

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                // Root spans [0,3) in shader space. 3 levels deep.
                // Root y=0: stone, y=1: grass surface, y=2: air+features.
                // Within y=1 (grass_surface_l2): sub-y=0 dirt, sub-y=1 grass, sub-y=2 air.
                // Grass top is at y = 1 + 2/3 = 1.667 in root space.
                // Start just above grass, in the air.
                pos: [1.5, 1.75, 1.5],
                yaw: 0.0,
                pitch: 0.0,
            },
            tree_data,
            root_index,
            cursor_locked: false,
            keys: Keys::default(),
            last_frame: std::time::Instant::now(),
            zoom_level: 0,
        }
    }

    fn update(&mut self, dt: f32) {
        let base_speed = 5.0;
        let speed = base_speed * 3.0f32.powi(self.zoom_level);

        let fwd = self.camera.forward_xz();
        let right = self.camera.right();

        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        let mut dz = 0.0f32;

        if self.keys.w { dx += fwd[0]; dz += fwd[2]; }
        if self.keys.s { dx -= fwd[0]; dz -= fwd[2]; }
        if self.keys.d { dx += right[0]; dz += right[2]; }
        if self.keys.a { dx -= right[0]; dz -= right[2]; }
        if self.keys.space { dy += 1.0; }
        if self.keys.shift { dy -= 1.0; }

        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len > 0.001 {
            let s = speed * dt / len;
            self.camera.pos[0] += dx * s;
            self.camera.pos[1] += dy * s;
            self.camera.pos[2] += dz * s;
        }

        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera());
        }
    }

    fn toggle_cursor_lock(&mut self) {
        let Some(window) = &self.window else { return };
        self.cursor_locked = !self.cursor_locked;
        if self.cursor_locked {
            let _ = window.set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            window.set_cursor_visible(false);
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
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

        let tree_data = self.tree_data.clone();
        let root_index = self.root_index;
        let renderer = pollster::block_on(Renderer::new(window, &tree_data, root_index));
        self.renderer = Some(renderer);
        self.last_frame = std::time::Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer { r.resize(size.width, size.height); }
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state, .. }, ..
            } => {
                let pressed = state == ElementState::Pressed;
                match code {
                    KeyCode::KeyW => self.keys.w = pressed,
                    KeyCode::KeyA => self.keys.a = pressed,
                    KeyCode::KeyS => self.keys.s = pressed,
                    KeyCode::KeyD => self.keys.d = pressed,
                    KeyCode::Space => self.keys.space = pressed,
                    KeyCode::ShiftLeft => self.keys.shift = pressed,
                    KeyCode::Escape if pressed => self.toggle_cursor_lock(),
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                if y > 0.0 { self.zoom_level += 1; }
                else if y < 0.0 { self.zoom_level -= 1; }
                self.zoom_level = self.zoom_level.clamp(-3, 10);
            }

            WindowEvent::MouseInput { state: ElementState::Pressed, .. } => {
                if !self.cursor_locked { self.toggle_cursor_lock(); }
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;
                self.update(dt);

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
                // Mouse right (positive delta.0) → yaw decreases → look right.
                self.camera.yaw -= (delta.0 * SENS) as f32;
                // Mouse up (negative delta.1) → pitch increases → look up.
                // macOS: delta.1 positive = mouse moves up physically.
                self.camera.pitch = (self.camera.pitch + (delta.1 * SENS) as f32)
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
