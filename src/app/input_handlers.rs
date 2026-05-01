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

        // WASD/Space/Shift are CONTINUOUS — `App::update` reads the
        // held state every frame and flies the camera in its
        // current basis (Minecraft-creative-style). No discrete
        // teleport here. The remaining shortcuts below fire on key
        // down only, gated by cursor lock + no open UI panels.
        if pressed && self.cursor_locked && !self.ui.any_panel_open() {
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
