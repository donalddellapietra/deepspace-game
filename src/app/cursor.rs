//! Cursor lock / unlock, plus the panel-open → cursor-state sync.
//!
//! The engine uses a locked cursor for first-person mouselook while
//! the game is focused, and releases the cursor when any UI panel
//! (inventory, color picker) opens. On macOS, locking also has to
//! re-route keyboard focus away from the wry WKWebView back to the
//! content view — see [`crate::platform::refocus_content_view`].

use winit::window::CursorGrabMode;

use super::App;

impl App {
    pub(super) fn lock_cursor(&mut self) {
        let Some(window) = &self.window else { return };
        self.cursor_locked = true;
        let _ = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        window.set_cursor_visible(false);
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::overlay::clear_passthrough();
            crate::platform::refocus_content_view(window);
        }
    }

    pub(super) fn unlock_cursor(&mut self) {
        let Some(window) = &self.window else { return };
        self.cursor_locked = false;
        self.keys.clear();
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
    }

    pub(super) fn sync_cursor_to_panels(&mut self) {
        if self.ui.any_panel_open() && self.cursor_locked {
            self.unlock_cursor();
        } else if !self.ui.any_panel_open() && !self.cursor_locked {
            self.lock_cursor();
        }
    }
}
