//! Keyboard and mouse dispatch on the `App`.
//!
//! These are the handlers that winit's event loop calls (directly
//! from the host OS) or that [`super::overlay_integration`] calls
//! (for events forwarded from the webview). They deliberately do
//! minimum routing and hand off to higher-level methods elsewhere:
//! editing goes to [`super::edit_actions`]; cursor-lock state lives
//! in [`super::cursor`].

use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use super::App;

impl App {
    pub(super) fn apply_key(&mut self, code: KeyCode, pressed: bool) {
        self.keys.apply(code, pressed);

        // Debug chunk-teleport. Fires on key-down only so each press
        // is one step — holding WASD doesn't auto-repeat. All
        // motion is gated by `cursor_locked` (so it doesn't fight
        // with typing into UI panels) and by `!self.frozen`.
        if pressed && self.cursor_locked && !self.ui.any_panel_open() {
            let step = match code {
                KeyCode::KeyW => Some((2usize, -1)), // forward = -Z
                KeyCode::KeyS => Some((2, 1)),       // back = +Z
                KeyCode::KeyA => Some((0, -1)),      // left = -X
                KeyCode::KeyD => Some((0, 1)),       // right = +X
                KeyCode::Space => Some((1, 1)),      // up = +Y
                KeyCode::ShiftLeft => Some((1, -1)), // down = -Y
                _ => None,
            };
            if let Some((axis, dir)) = step {
                self.step_chunk(axis, dir);
                return;
            }
            if code == KeyCode::KeyF {
                self.frozen = !self.frozen;
                log::info!("debug freeze: {}", self.frozen);
                return;
            }
            if code == KeyCode::KeyG {
                self.log_location();
                return;
            }
            if code == KeyCode::KeyN {
                self.spawn_test_entities(10);
                return;
            }
            if code == KeyCode::KeyM {
                self.spawn_test_entities(1000);
                return;
            }
            if code == KeyCode::KeyB {
                self.spawn_test_cubes(10);
                return;
            }
            if code == KeyCode::KeyV {
                self.spawn_test_cubes(1000);
                return;
            }
        }

        if pressed && code == KeyCode::Escape {
            if self.ui.any_panel_open() {
                self.ui
                    .handle_command(crate::bridge::UiCommand::CloseAllPanels);
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

        if pressed && code == KeyCode::F6 {
            self.cycle_sphere_debug_mode();
            return;
        }

        if pressed && code == KeyCode::KeyV && self.cursor_locked {
            self.save_mode = !self.save_mode;
            log::info!("Save mode: {}", self.save_mode);
            return;
        }

        let panel_changed = self.ui.handle_key(code, pressed);
        if panel_changed {
            self.sync_cursor_to_panels();
        }
    }

    /// Rotate `sphere_debug_mode` through `0..SPHERE_DEBUG_MODE_COUNT`
    /// and push to the renderer. Logs the new mode's name so the user
    /// can tell which visualization just got enabled.
    fn cycle_sphere_debug_mode(&mut self) {
        let next = (self.sphere_debug_mode + 1) % crate::renderer::SPHERE_DEBUG_MODE_COUNT;
        self.sphere_debug_mode = next;
        let name = crate::renderer::SPHERE_DEBUG_MODE_NAMES
            .get(next as usize)
            .copied()
            .unwrap_or("?");
        log::info!("sphere debug mode: {} ({})", next, name);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_sphere_debug_mode(next);
        }
    }

    pub(super) fn apply_mouse(&mut self, button: MouseButton) {
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
