//! Deep Space — ray-marched voxel engine.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use deepspace_game::game_state::{GameUiState, HotbarItem, SavedMeshes};
use deepspace_game::renderer::Renderer;
use deepspace_game::world::edit;
use deepspace_game::world::sdf;
use deepspace_game::world::gpu::{self, GpuCamera};
use deepspace_game::world::palette::ColorRegistry;
use deepspace_game::world::state::WorldState;

#[cfg(not(target_arch = "wasm32"))]
use deepspace_game::overlay;

// ------------------------------------------------------------ Camera

/// Yaw/pitch camera in a locally-oriented frame.
///
/// `smoothed_up` tracks the local "up" direction (away from the
/// dominant planet's center, or world +Y in empty space), smoothed
/// over time so the horizon doesn't jerk when gravity switches.
/// `yaw` rotates around `smoothed_up`; `pitch` tilts the look vector
/// out of the tangent plane toward `smoothed_up`.
struct Camera {
    pos: [f32; 3],
    smoothed_up: [f32; 3],
    yaw: f32,
    pitch: f32,
}

impl Camera {
    /// Lerp `smoothed_up` toward `target_up` at rate `k` per dt.
    fn update_up(&mut self, target_up: [f32; 3], dt: f32) {
        let k = (dt * 4.0).min(1.0);
        let blended = [
            self.smoothed_up[0] + (target_up[0] - self.smoothed_up[0]) * k,
            self.smoothed_up[1] + (target_up[1] - self.smoothed_up[1]) * k,
            self.smoothed_up[2] + (target_up[2] - self.smoothed_up[2]) * k,
        ];
        self.smoothed_up = sdf::normalize(blended);
    }

    /// (forward, right, up) in world space, built from smoothed_up +
    /// yaw + pitch. Returned basis is orthonormal — `up` is
    /// perpendicular to `forward`.
    ///
    /// Yaw convention: positive yaw = turn LEFT (counterclockwise
    /// around up). This matches the original world-Y camera so
    /// mouse-look feels consistent on a planet.
    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let ref_up = self.smoothed_up;
        let (t_right, t_fwd) = sdf::tangent_basis(ref_up);
        let (sy, cy) = self.yaw.sin_cos();
        // Positive yaw rotates counterclockwise around `ref_up`.
        let horiz_fwd = sdf::sub(sdf::scale(t_fwd, cy), sdf::scale(t_right, sy));
        let horiz_right = sdf::add(sdf::scale(t_right, cy), sdf::scale(t_fwd, sy));
        let (sp, cp) = self.pitch.sin_cos();
        let fwd = sdf::normalize(sdf::add(
            sdf::scale(horiz_fwd, cp),
            sdf::scale(ref_up, sp),
        ));
        // Orthonormal `up` = right × forward (right-handed).
        let up = [
            horiz_right[1] * fwd[2] - horiz_right[2] * fwd[1],
            horiz_right[2] * fwd[0] - horiz_right[0] * fwd[2],
            horiz_right[0] * fwd[1] - horiz_right[1] * fwd[0],
        ];
        (fwd, horiz_right, sdf::normalize(up))
    }

    fn forward(&self) -> [f32; 3] { self.basis().0 }

    fn gpu_camera(&self, fov: f32) -> GpuCamera {
        let (fwd, r, up) = self.basis();
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
    palette: ColorRegistry,
    saved_meshes: SavedMeshes,
    save_mode: bool,
    ui: GameUiState,
    debug_overlay_visible: bool,
    /// Exponentially smoothed FPS for the debug overlay.
    fps_smooth: f64,
    /// Cubed-sphere demo planet — owns per-cell block storage and
    /// the shell geometry used for cursor raycasts and editing.
    cs_planet: Option<deepspace_game::world::cubesphere::CubeSpherePlanet>,
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

        // Spawn in empty space, facing toward the cubed-sphere demo
        // planet (which is placed at [1.5, 2.3, 1.5]).
        let spawn_pos = [1.5, 1.75, 1.5];

        Self {
            window: None,
            renderer: None,
            camera: Camera {
                pos: spawn_pos,
                smoothed_up: [0.0, 1.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
            },
            world,
            cursor_locked: false,
            keys: Keys::default(),
            last_frame: std::time::Instant::now(),
            // Start at the base terrain's block level. The base terrain
            // occupies the bottom 6 layers, so edit_depth = 6 means
            // zoom_level = tree_depth - 6.
            zoom_level: (tree_depth as i32 - 6).max(0),
            tree_depth,
            palette: ColorRegistry::new(),
            saved_meshes: SavedMeshes::default(),
            save_mode: false,
            ui: GameUiState::new(),
            debug_overlay_visible: false,
            fps_smooth: 0.0,
            cs_planet: None,
            #[cfg(not(target_arch = "wasm32"))]
            webview: None,
            #[cfg(not(target_arch = "wasm32"))]
            frames_waited: 0,
        }
    }

    fn update(&mut self, dt: f32) {
        let td = self.tree_depth as i32;
        let cell_size = 1.0 / 3.0f32.powi(td - self.zoom_level);

        // Camera's "up" relaxes toward world +Y. Later, when the
        // cubed-sphere planet gets gravity, we'll blend in the
        // planet's radial up here — for now this is pure flycam.
        self.camera.update_up([0.0, 1.0, 0.0], dt);

        // Creative/flycam movement: WASD along camera forward/right,
        // Space/Shift along camera up. No collision, no gravity force.
        let speed = 5.0 * cell_size;
        let (fwd, right, up) = self.camera.basis();
        let mut d = [0.0f32; 3];
        if self.keys.w { d = sdf::add(d, fwd); }
        if self.keys.s { d = sdf::sub(d, fwd); }
        if self.keys.d { d = sdf::add(d, right); }
        if self.keys.a { d = sdf::sub(d, right); }
        if self.keys.space { d = sdf::add(d, up); }
        if self.keys.shift { d = sdf::sub(d, up); }
        let l = sdf::length(d);
        if l > 1e-4 {
            let s = speed * dt / l;
            self.camera.pos = sdf::add(self.camera.pos, sdf::scale(d, s));
        }

        if let Some(renderer) = &self.renderer {
            renderer.update_camera(&self.camera.gpu_camera(1.2));
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
        (self.edit_depth() + 3).min(16)
    }

    /// Clamp zoom and sync GPU depth.
    fn apply_zoom(&mut self) {
        let td = self.tree_depth as i32;
        self.zoom_level = self.zoom_level.clamp(0, (td - 1).max(0));
        self.ui.zoom_level = self.zoom_level;
        let vd = self.visual_depth();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&self.camera.gpu_camera(1.2));
        }
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}",
            self.zoom_level, td, self.edit_depth(), vd
        );
    }

    /// If the cubed-sphere planet is hit closer than any tree block,
    /// write `new_block` (0 = erase) to the targeted cell and
    /// re-upload the per-cell buffer. Returns `true` if the edit
    /// happened so the caller can skip its tree-edit fallback.
    fn edit_cs_if_closer(&mut self, ray_dir: [f32; 3], new_block: u8) -> bool {
        // First: read-only probe of the cubed-sphere planet.
        let Some((cs_t, face, i, j)) = self.cs_planet.as_ref()
            .and_then(|p| p.hit_cell(self.camera.pos, ray_dir)) else {
            return false;
        };
        // Compare against the nearest tree block along this ray.
        let ed = self.edit_depth();
        let tree_hit = edit::cpu_raycast(
            &self.world.library, self.world.root,
            self.camera.pos, ray_dir, ed,
        );
        let tree_t = tree_hit.as_ref().map(|h| h.t).unwrap_or(f32::INFINITY);
        if cs_t >= tree_t { return false; }

        // Safe to take a mutable borrow now: we're no longer reading
        // anything else on `self` that conflicts.
        let Some(planet) = self.cs_planet.as_mut() else { return false };
        planet.set(face, i, j, new_block);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_cubed_sphere_blocks(&planet.blocks);
        }
        true
    }

    fn do_break(&mut self) {
        let ray_dir = self.camera.forward();

        // If the cubed-sphere planet's shell is closer than any
        // tree block, clear the targeted cell there. Break never
        // cares whether the cell was empty — we always write 0.
        if self.edit_cs_if_closer(ray_dir, 0) {
            return;
        }

        let hit = edit::cpu_raycast(
            &self.world.library, self.world.root,
            self.camera.pos, ray_dir, self.edit_depth(),
        );
        let Some(hit) = hit else { return };

        if self.save_mode {
            // Save mode: capture the subtree under the crosshair.
            // The hit path gives us (parent_id, slot) pairs from root.
            // We want to save the deepest Node in the path — that's
            // the natural "block" at the current zoom level.
            //
            // If the hit child is a Node, save it directly.
            // If it's a Block terminal, go one level up and save the
            // parent node (which contains this block as a child).
            use deepspace_game::world::tree::Child;
            let mut saved_id = None;
            if let Some(&(parent_id, slot)) = hit.path.last() {
                if let Some(node) = self.world.library.get(parent_id) {
                    match node.children[slot] {
                        Child::Node(child_id) => saved_id = Some(child_id),
                        Child::Block(_) | Child::Empty => {
                            // Hit a terminal — save the parent node instead.
                            saved_id = Some(parent_id);
                        }
                    }
                }
            }
            if let Some(node_id) = saved_id {
                self.world.library.ref_inc(node_id);
                let idx = self.saved_meshes.save(node_id);
                self.ui.slots[self.ui.active_slot] = HotbarItem::Mesh(idx);
                log::info!("Saved mesh #{idx} (node {node_id})");
            }
            self.save_mode = false;
            return;
        }

        if edit::break_block(&mut self.world, &hit) {
            self.upload_tree();
        }
    }

    fn do_place(&mut self) {
        let ray_dir = self.camera.forward();

        // Cubed-sphere first: place the active hotbar block into the
        // targeted cell if the sphere is closer. Mesh slots fall
        // through to the tree path.
        if let Some(block_type) = self.ui.active_block_type() {
            if self.edit_cs_if_closer(ray_dir, block_type) {
                return;
            }
        }

        let hit = edit::cpu_raycast(
            &self.world.library, self.world.root,
            self.camera.pos, ray_dir, self.edit_depth(),
        );
        let Some(hit) = hit else { return };

        match &self.ui.slots[self.ui.active_slot] {
            HotbarItem::Block(block_type) => {
                if edit::place_block(&mut self.world, &hit, *block_type) {
                    self.upload_tree();
                }
            }
            HotbarItem::Mesh(idx) => {
                let Some(saved) = self.saved_meshes.items.get(*idx) else { return };
                let node_id = saved.node_id;
                // Place the subtree adjacent to the hit face, same as blocks.
                if edit::place_child(
                    &mut self.world,
                    &hit,
                    deepspace_game::world::tree::Child::Node(node_id),
                ) {
                    self.upload_tree();
                }
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
                renderer.set_cubed_sphere_highlight(None);
            }
            return;
        }
        let ray_dir = self.camera.forward();
        let tree_hit = edit::cpu_raycast(
            &self.world.library, self.world.root,
            self.camera.pos, ray_dir, self.edit_depth(),
        );
        let tree_t = tree_hit.as_ref().map(|h| h.t).unwrap_or(f32::INFINITY);

        // Cubed-sphere cursor: hit the demo planet's outer shell and
        // report (face, i, j) for whichever cell the ray enters.
        let cs_hit = self.cs_planet.as_ref()
            .and_then(|p| p.hit_cell(self.camera.pos, ray_dir));
        let cs_t = cs_hit.map(|(t, ..)| t).unwrap_or(f32::INFINITY);

        if let Some(renderer) = &mut self.renderer {
            if cs_t < tree_t {
                // Cubed-sphere cell is in front — draw its bulged
                // outline, hide the tree AABB highlight.
                renderer.set_highlight(None);
                if let Some((_, face, i, j)) = cs_hit {
                    renderer.set_cubed_sphere_highlight(Some((face as u32, i, j)));
                }
            } else {
                // Tree block (or nothing) is in front — normal AABB.
                renderer.set_highlight(tree_hit.as_ref().map(edit::hit_aabb));
                renderer.set_cubed_sphere_highlight(None);
            }
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

        if pressed && code == KeyCode::BracketRight {
            self.debug_overlay_visible = !self.debug_overlay_visible;
            return;
        }

        if pressed && code == KeyCode::KeyV && self.cursor_locked {
            self.save_mode = !self.save_mode;
            log::info!("Save mode: {}", self.save_mode);
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
        let mut renderer = pollster::block_on(Renderer::new(window, &tree_data, root_index));
        // Demo cubed-sphere planet: 16 cells per face edge (96²
        // surface cells total), SDF-driven terrain. The SDF's own
        // radius is a touch smaller than the shell radius so noise
        // produces both inland terrain and empty "seas" where the
        // noise-displaced surface dips below the shell radius.
        {
            use deepspace_game::world::cubesphere::generate_from_sdf;
            use deepspace_game::world::palette::block;
            use deepspace_game::world::sdf::Planet as SdfPlanet;
            let cs_center = [1.5, 2.3, 1.5];
            let cs_radius = 0.35;
            let cells_per_face = 16u32;
            // SDF radius is slightly LARGER than the shell radius so
            // most cells read as "below surface" (= solid). Noise
            // then carves the occasional valley / sea where the
            // displaced surface dips below the shell.
            let sdf = SdfPlanet {
                center: cs_center,
                radius: cs_radius + 0.03,
                noise_scale: 0.05,
                noise_freq: 14.0,
                noise_seed: 2024,
                gravity: 9.8,
                influence_radius: cs_radius * 2.5,
                surface_block: block::GRASS,
                core_block: block::STONE,
            };
            let planet = generate_from_sdf(cs_center, cs_radius, cells_per_face, &sdf);
            renderer.set_cubed_sphere_planet(cs_center, cs_radius, cells_per_face);
            renderer.set_cubed_sphere_blocks(&planet.blocks);
            self.cs_planet = Some(planet);
        }
        self.renderer = Some(renderer);
        self.apply_zoom(); // sync renderer max_depth with initial zoom_level
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
                let old_zoom = self.zoom_level;
                // Scroll up = zoom in (finer), scroll down = zoom out (coarser).
                if y > 0.0 { self.zoom_level -= 1; }
                else if y < 0.0 { self.zoom_level += 1; }
                self.apply_zoom();
                let steps = self.zoom_level - old_zoom; // negative = zoomed in
                if steps != 0 {
                    // Move camera toward/away from the crosshair target so
                    // blocks at the new layer appear the same size on screen.
                    // Each layer is 3× finer, so scale distance by 3^steps.
                    let ray_dir = self.camera.forward();
                    let hit = edit::cpu_raycast(
                        &self.world.library, self.world.root,
                        self.camera.pos, ray_dir, self.edit_depth(),
                    );
                    let anchor = if let Some(h) = hit {
                        // Anchor at the hit point.
                        [
                            self.camera.pos[0] + ray_dir[0] * h.t,
                            self.camera.pos[1] + ray_dir[1] * h.t,
                            self.camera.pos[2] + ray_dir[2] * h.t,
                        ]
                    } else {
                        // No hit — anchor at a reasonable distance ahead.
                        let td = self.tree_depth as i32;
                        let cell_size = 1.0 / 3.0f32.powi(td - self.zoom_level);
                        let d = cell_size * 10.0;
                        [
                            self.camera.pos[0] + ray_dir[0] * d,
                            self.camera.pos[1] + ray_dir[1] * d,
                            self.camera.pos[2] + ray_dir[2] * d,
                        ]
                    };
                    let scale = 3.0f32.powi(-steps); // zoom in → 1/3, zoom out → 3
                    for i in 0..3 {
                        self.camera.pos[i] = anchor[i] + (self.camera.pos[i] - anchor[i]) * scale;
                    }
                }
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
                    self.ui.push_to_overlay(&self.palette);
                    overlay::push_state(&deepspace_game::bridge::GameStateUpdate::DebugOverlay(
                        deepspace_game::bridge::DebugOverlayStateJs {
                            visible: self.debug_overlay_visible,
                            fps: self.fps_smooth,
                            frame_time_ms: if self.fps_smooth > 0.0 { 1000.0 / self.fps_smooth } else { 0.0 },
                            zoom_level: self.zoom_level,
                            tree_depth: self.tree_depth,
                            edit_depth: self.edit_depth(),
                            visual_depth: self.visual_depth(),
                            camera_pos: self.camera.pos,
                            fov: 1.2,
                            node_count: self.world.library.len(),
                        },
                    ));
                    self.flush_overlay();
                }

                let now = std::time::Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                // Update smoothed FPS (EMA, ~0.5s window).
                if dt > 0.0 {
                    let instant_fps = 1.0 / dt as f64;
                    let alpha = (dt as f64 * 5.0).min(1.0); // smoothing factor
                    self.fps_smooth = self.fps_smooth * (1.0 - alpha) + instant_fps * alpha;
                }
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
