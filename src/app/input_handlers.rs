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

    /// Debug-only motion: every keystroke teleports the camera
    /// exactly one cell at its current anchor depth. Continuous
    /// physics is OFF — there is NO motion outside these keys.
    ///
    /// Bindings:
    ///   `W` / `S` — camera-forward / -backward, snapped to nearest world axis.
    ///   `A` / `D` — camera-left / -right, snapped likewise.
    ///   `Space` / `Shift` — world `+Y` / `-Y`.
    ///   `F` — toggle debug freeze (ignore movement keys).
    ///   `T` — teleport to the body's anchor cell at the camera's depth.
    ///   `1` / `2` / `3` — teleport to body north / south pole / center.
    fn handle_debug_motion(&mut self, code: KeyCode) {
        if code == KeyCode::KeyF {
            self.debug_frozen = !self.debug_frozen;
            log::info!("Debug freeze: {}", self.debug_frozen);
            return;
        }
        if self.debug_frozen { return; }
        if code == KeyCode::KeyT { self.teleport_to_body();        return; }
        if code == KeyCode::Digit1 { self.teleport_to_pole( 1);    return; }
        if code == KeyCode::Digit2 { self.teleport_to_pole(-1);    return; }
        if code == KeyCode::Digit3 { self.teleport_to_body_center(); return; }
        let lib = &self.world.library;
        let root = self.world.root;
        match code {
            KeyCode::KeyW       => player::teleport_along_camera(&mut self.camera, CameraDir::Forward,  lib, root),
            KeyCode::KeyS       => player::teleport_along_camera(&mut self.camera, CameraDir::Backward, lib, root),
            KeyCode::KeyA       => player::teleport_along_camera(&mut self.camera, CameraDir::Left,     lib, root),
            KeyCode::KeyD       => player::teleport_along_camera(&mut self.camera, CameraDir::Right,    lib, root),
            KeyCode::Space      => player::teleport_one_cell(&mut self.camera, 1,  1, lib, root),
            KeyCode::ShiftLeft  => player::teleport_one_cell(&mut self.camera, 1, -1, lib, root),
            _ => {}
        }
    }

    /// Teleport directly above (`pole=+1`) or below (`-1`) the body
    /// at the camera's current anchor depth. Lands one cell away
    /// from the body cell along the Y axis.
    fn teleport_to_pole(&mut self, pole: i8) {
        use crate::world::coords::WorldPos;
        use crate::world::tree::CENTER_SLOT;
        let mut anchor = self.body_anchor;
        let target_depth = self.camera.position.anchor.depth().max(self.body_anchor.depth());
        while anchor.depth() < target_depth {
            if !anchor.push(CENTER_SLOT as u8) { break; }
        }
        self.camera.position = WorldPos { anchor, offset: [0.5, 0.5, 0.5] };
        let lib = &self.world.library;
        let root = self.world.root;
        let _ = self.camera.position.add_local([0.0, pole as f32 * 1.5, 0.0], lib, root);
        log::info!("Teleport to pole {} at depth {}", pole, self.camera.position.anchor.depth());
    }

    /// Teleport to the body's anchor cell center at the camera's depth.
    fn teleport_to_body_center(&mut self) {
        use crate::world::coords::WorldPos;
        use crate::world::tree::CENTER_SLOT;
        let mut anchor = self.body_anchor;
        let target_depth = self.camera.position.anchor.depth().max(self.body_anchor.depth());
        while anchor.depth() < target_depth {
            if !anchor.push(CENTER_SLOT as u8) { break; }
        }
        self.camera.position = WorldPos { anchor, offset: [0.5, 0.5, 0.5] };
        log::info!("Teleport to body center at depth {}", self.camera.position.anchor.depth());
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
