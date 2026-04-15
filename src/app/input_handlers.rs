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

use crate::player::{self, CameraDir};

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

        if pressed {
            self.handle_debug_motion(code);
        }

        let panel_changed = self.ui.handle_key(code, pressed);
        if panel_changed {
            self.sync_cursor_to_panels();
        }
    }

    /// Debug movement: every keystroke teleports the camera by
    /// exactly one cell at its current anchor depth. Continuous
    /// physics is off — every position change goes through here.
    ///
    /// Bindings:
    ///   `W` / `S` — camera-forward / -backward, snapped to the
    ///       nearest world axis.
    ///   `A` / `D` — camera-left / -right, snapped likewise.
    ///   `Space` / `Shift` — world `+Y` / `-Y`.
    ///   `F` — toggle debug freeze (ignore movement keys).
    ///   `T` — teleport to the body's anchor cell at the camera's
    ///       current depth (offset reset to (0.5, 0.5, 0.5)).
    ///   `[` / `]` — placeholder; zoom is on the scroll wheel.
    fn handle_debug_motion(&mut self, code: KeyCode) {
        if code == KeyCode::KeyF {
            self.debug_frozen = !self.debug_frozen;
            log::info!("Debug freeze: {}", self.debug_frozen);
            return;
        }
        if self.debug_frozen { return; }
        if code == KeyCode::KeyT {
            self.teleport_to_body();
            return;
        }
        let lib = &self.world.library;
        let root = self.world.root;
        match code {
            KeyCode::KeyW => player::teleport_along_camera(&mut self.camera, CameraDir::Forward,  lib, root),
            KeyCode::KeyS => player::teleport_along_camera(&mut self.camera, CameraDir::Backward, lib, root),
            KeyCode::KeyA => player::teleport_along_camera(&mut self.camera, CameraDir::Left,     lib, root),
            KeyCode::KeyD => player::teleport_along_camera(&mut self.camera, CameraDir::Right,    lib, root),
            KeyCode::Space      => player::teleport_one_cell(&mut self.camera, 1,  1, lib, root),
            KeyCode::ShiftLeft  => player::teleport_one_cell(&mut self.camera, 1, -1, lib, root),
            _ => {}
        }
    }

    /// Hard-set the camera's anchor to the body's anchor cell at the
    /// camera's current depth (or the body's depth if shallower).
    /// Offset goes to `(0.5, 0.5, 0.5)` — center of that cell.
    fn teleport_to_body(&mut self) {
        use crate::world::coords::WorldPos;
        let mut anchor = self.body_anchor;
        // If the camera was deeper than the body's depth, push the
        // anchor down with center slots so we land at the same depth
        // the camera was operating at, just inside the body.
        let target_depth = self.camera.position.anchor.depth();
        let center_slot = crate::world::tree::CENTER_SLOT as u8;
        while anchor.depth() < target_depth {
            if !anchor.push(center_slot) { break; }
        }
        self.camera.position = WorldPos { anchor, offset: [0.5, 0.5, 0.5] };
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
