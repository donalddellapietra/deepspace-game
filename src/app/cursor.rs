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
        // Tell JS we initiated this release so the
        // pointerlockchange listener doesn't synthesize an ESC and
        // bounce-close whatever panel just opened.
        #[cfg(target_arch = "wasm32")]
        crate::overlay::notify_intentional_unlock();
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

    /// Drain UI commands from the overlay (WebView IPC on native,
    /// `window.__pollUiCommands` on WASM) and dispatch them to the
    /// game UI state. Panels that toggle as a result re-sync the
    /// cursor lock.
    pub(super) fn poll_ui_commands(&mut self) {
        for cmd in crate::overlay::poll_commands() {
            let panel_changed = self.ui.handle_command(cmd, &mut self.palette);
            if panel_changed {
                self.sync_cursor_to_panels();
            }
        }
        // `CreateBlock` registers a new palette entry; push the
        // updated palette to the GPU so subsequent place actions
        // actually paint with the right colour.
        if self.ui.palette_dirty {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.update_palette(&self.palette.to_gpu_palette());
            }
            self.ui.palette_dirty = false;
        }
    }
}
