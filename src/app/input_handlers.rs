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

        if pressed && code == KeyCode::KeyF {
            self.debug_frozen = !self.debug_frozen;
            log::info!("Debug freeze: {}", self.debug_frozen);
            return;
        }

        if pressed && code == KeyCode::KeyT {
            self.teleport_to_body();
            return;
        }

        let panel_changed = self.ui.handle_key(code, pressed);
        if panel_changed {
            self.sync_cursor_to_panels();
        }
    }

    /// Hard-set the camera's anchor to the body's anchor cell at the
    /// camera's current depth (or the body's depth if shallower).
    /// Offset goes to `(0.5, 0.5, 0.5)` — center of that cell.
    /// Useful as a debug "warp to planet" when you've flown too far.
    fn teleport_to_body(&mut self) {
        use crate::world::coords::WorldPos;
        let mut anchor = self.body_anchor;
        let target_depth = self.camera.position.anchor.depth();
        let center_slot = crate::world::tree::CENTER_SLOT as u8;
        while anchor.depth() < target_depth {
            if !anchor.push(center_slot) { break; }
        }
        self.camera.position = WorldPos { anchor, offset: [0.5, 0.5, 0.5] };
        self.velocity = [0.0; 3];
        log::info!("Teleport to body anchor at depth {}", anchor.depth());
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
