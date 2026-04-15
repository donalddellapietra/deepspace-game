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
    pub(super) fn apply_key(&mut self, code: KeyCode, pressed: bool, repeat: bool) {
        self.keys.apply(code, pressed);

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

        // Debug-mode movement: WASD/Space/Shift teleport one cell of
        // width `3^-camera.depth` in a world-frame direction. R
        // resets to the canonical inspection pose. All movement is
        // discrete — no held-key acceleration, no auto-repeat.
        if pressed && !repeat && self.cursor_locked {
            let dir: Option<[f32; 3]> = match code {
                KeyCode::KeyW => Some([0.0, 0.0, -1.0]),
                KeyCode::KeyS => Some([0.0, 0.0,  1.0]),
                KeyCode::KeyA => Some([-1.0, 0.0, 0.0]),
                KeyCode::KeyD => Some([ 1.0, 0.0, 0.0]),
                KeyCode::Space      => Some([0.0,  1.0, 0.0]),
                KeyCode::ShiftLeft  => Some([0.0, -1.0, 0.0]),
                KeyCode::KeyR => {
                    self.debug_reset_pose();
                    return;
                }
                _ => None,
            };
            if let Some(d) = dir {
                self.debug_teleport(d);
                return;
            }
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
