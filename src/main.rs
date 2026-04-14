//! Deep Space — ray-marched voxel engine.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use deepspace_game::renderer::Renderer;
use deepspace_game::world::edit;
use deepspace_game::world::gpu::{self, GpuCamera};
use deepspace_game::world::state::WorldState;
use deepspace_game::world::tree::BlockType;

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
    /// Apply a key press/release from a winit KeyCode.
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

    /// Clear all held keys (prevents "stuck keys" on focus transitions).
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
    selected_block: BlockType,
    #[cfg(not(target_arch = "wasm32"))]
    webview: Option<wry::WebView>,
    #[cfg(not(target_arch = "wasm32"))]
    frames_waited: u32,
    /// Whether the React UI currently has pointer focus.
    #[cfg(not(target_arch = "wasm32"))]
    ui_focused: bool,
}

/// Wait this many frames before creating the WebView, giving the
/// window and Metal surface time to initialise fully.
#[cfg(not(target_arch = "wasm32"))]
const WAIT_FRAMES: u32 = 10;

impl App {
    fn new() -> Self {
        let world = WorldState::test_world();

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
            selected_block: BlockType::Stone,
            #[cfg(not(target_arch = "wasm32"))]
            webview: None,
            #[cfg(not(target_arch = "wasm32"))]
            frames_waited: 0,
            #[cfg(not(target_arch = "wasm32"))]
            ui_focused: false,
        }
    }

    fn update(&mut self, dt: f32) {
        let speed = 5.0 * 3.0f32.powi(self.zoom_level);

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
            renderer.update_camera(&self.camera.gpu_camera(1.2));
        }
    }

    /// Compute the max ray depth for the current zoom level.
    /// zoom_level 0 → deepest (individual blocks), higher → coarser.
    fn edit_depth(&self) -> u32 {
        // The test world is 3 levels deep. Each zoom step removes one
        // level of descent, so the player edits coarser structures.
        // Clamp to at least 1 (can always target root children).
        let base_depth: u32 = 8; // max traversal depth
        base_depth.saturating_sub(self.zoom_level.max(0) as u32).max(1)
    }

    fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            self.camera.pos,
            ray_dir,
            self.edit_depth(),
        );
        if let Some(hit) = hit {
            if edit::break_block(&mut self.world, &hit) {
                self.upload_tree();
            }
        }
    }

    fn do_place(&mut self) {
        let ray_dir = self.camera.forward();
        let hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            self.camera.pos,
            ray_dir,
            self.edit_depth(),
        );
        if let Some(hit) = hit {
            if edit::place_block(&mut self.world, &hit, self.selected_block) {
                self.upload_tree();
            }
        }
    }

    fn upload_tree(&mut self) {
        let (tree_data, root_index) = gpu::pack_tree(&self.world.library, self.world.root);
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, root_index);
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

    fn toggle_cursor_lock(&mut self) {
        if self.cursor_locked {
            self.unlock_cursor();
        } else {
            self.lock_cursor();
        }
    }

    // ── Overlay integration (native only) ────────────────────────

    /// Try to create the WebView overlay after enough frames have passed.
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

    /// Drain forwarded key/mouse events from the WebView IPC and
    /// apply them to our input state.
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

    /// Process UI commands from React.
    #[cfg(not(target_arch = "wasm32"))]
    fn poll_ui_commands(&mut self) {
        use deepspace_game::bridge::UiCommand;
        for cmd in overlay::poll_commands() {
            match cmd {
                UiCommand::SelectHotbarSlot { slot } => {
                    if slot < 10 {
                        if let Some(bt) = slot_to_block(slot) {
                            self.selected_block = bt;
                        }
                    }
                }
                UiCommand::UiFocused { focused } => {
                    self.ui_focused = focused;
                }
                UiCommand::CloseAllPanels => {}
                UiCommand::ToggleInventory => {}
                UiCommand::ToggleColorPicker => {}
                _ => {}
            }
        }
    }

    /// Flush buffered state to the WebView.
    #[cfg(not(target_arch = "wasm32"))]
    fn flush_overlay(&self) {
        if let Some(ref wv) = self.webview {
            overlay::flush_to_webview(wv);
        }
    }

    /// Resize the WebView to match the window.
    #[cfg(not(target_arch = "wasm32"))]
    fn resize_overlay(&self) {
        if let Some(ref wv) = self.webview {
            if let Some(window) = &self.window {
                overlay::resize_webview(wv, window);
            }
        }
    }

    /// Push current game state to the React overlay.
    #[cfg(not(target_arch = "wasm32"))]
    fn push_state_to_overlay(&self) {
        use deepspace_game::bridge::*;

        let slots: Vec<SlotInfo> = (0..10).map(|i| {
            let bt = slot_to_block(i).unwrap_or(BlockType::Stone);
            let color = block_color(bt);
            SlotInfo {
                kind: "block",
                index: bt as u32,
                name: format!("{:?}", bt),
                color,
            }
        }).collect();

        overlay::push_state(&GameStateUpdate::Hotbar(HotbarState {
            active: block_to_slot(self.selected_block),
            slots,
            layer: self.zoom_level.max(0) as u8,
        }));

        overlay::push_state(&GameStateUpdate::ModeIndicator(ModeIndicatorStateJs {
            layer: self.zoom_level.max(0) as u8,
            save_mode: false,
            save_eligible: false,
            entity_edit_mode: false,
        }));
    }

    // ── Unified input handlers ───────────────────────────────────

    fn apply_key(&mut self, code: KeyCode, pressed: bool) {
        self.keys.apply(code, pressed);
        if pressed {
            match code {
                KeyCode::Escape => self.toggle_cursor_lock(),
                KeyCode::Digit1 => self.selected_block = BlockType::Stone,
                KeyCode::Digit2 => self.selected_block = BlockType::Dirt,
                KeyCode::Digit3 => self.selected_block = BlockType::Grass,
                KeyCode::Digit4 => self.selected_block = BlockType::Wood,
                KeyCode::Digit5 => self.selected_block = BlockType::Leaf,
                KeyCode::Digit6 => self.selected_block = BlockType::Sand,
                KeyCode::Digit7 => self.selected_block = BlockType::Brick,
                KeyCode::Digit8 => self.selected_block = BlockType::Metal,
                KeyCode::Digit9 => self.selected_block = BlockType::Glass,
                _ => {}
            }
        }
    }

    fn apply_mouse(&mut self, button: MouseButton) {
        if !self.cursor_locked {
            self.lock_cursor();
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
                if y > 0.0 { self.zoom_level -= 1; }
                else if y < 0.0 { self.zoom_level += 1; }
                self.zoom_level = self.zoom_level.clamp(-2, 5);

                let max_depth = (3 - self.zoom_level).clamp(1, 8) as u32;
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_max_depth(max_depth);
                }
                log::info!("Zoom level: {}, max_depth: {}, edit_depth: {}",
                    self.zoom_level, max_depth, self.edit_depth());
            }

            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                self.apply_mouse(button);
            }

            WindowEvent::RedrawRequested => {
                // ── Overlay lifecycle ──
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.try_create_webview();
                    self.inject_webview_input();
                    self.poll_ui_commands();
                    self.push_state_to_overlay();
                    self.flush_overlay();
                }

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
                self.camera.yaw -= (delta.0 * SENS) as f32;
                self.camera.pitch = (self.camera.pitch - (delta.1 * SENS) as f32)
                    .clamp(-1.5, 1.5);
            }
        }
    }
}

// ── Block ↔ slot helpers ─────────────────────────────────────────

fn slot_to_block(slot: usize) -> Option<BlockType> {
    match slot {
        0 => Some(BlockType::Stone),
        1 => Some(BlockType::Dirt),
        2 => Some(BlockType::Grass),
        3 => Some(BlockType::Wood),
        4 => Some(BlockType::Leaf),
        5 => Some(BlockType::Sand),
        6 => Some(BlockType::Brick),
        7 => Some(BlockType::Metal),
        8 => Some(BlockType::Glass),
        _ => None,
    }
}

fn block_to_slot(bt: BlockType) -> usize {
    match bt {
        BlockType::Stone => 0,
        BlockType::Dirt => 1,
        BlockType::Grass => 2,
        BlockType::Wood => 3,
        BlockType::Leaf => 4,
        BlockType::Sand => 5,
        BlockType::Brick => 6,
        BlockType::Metal => 7,
        BlockType::Glass => 8,
        _ => 0,
    }
}

fn block_color(bt: BlockType) -> [f32; 4] {
    match bt {
        BlockType::Stone => [0.5, 0.5, 0.5, 1.0],
        BlockType::Dirt  => [0.6, 0.4, 0.2, 1.0],
        BlockType::Grass => [0.3, 0.7, 0.2, 1.0],
        BlockType::Wood  => [0.6, 0.4, 0.15, 1.0],
        BlockType::Leaf  => [0.2, 0.6, 0.1, 1.0],
        BlockType::Sand  => [0.9, 0.85, 0.6, 1.0],
        BlockType::Water => [0.2, 0.4, 0.9, 0.7],
        BlockType::Brick => [0.7, 0.3, 0.2, 1.0],
        BlockType::Metal => [0.8, 0.8, 0.85, 1.0],
        BlockType::Glass => [0.7, 0.85, 0.9, 0.4],
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
