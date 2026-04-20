//! Cursor highlight: ship the GPU cursor probe's hit slot path to
//! the shader so its path-prefix + sub-cell extension lights up that
//! exact cell.
//!
//! The GPU probe is the single source of truth — it runs the same
//! `march()` the fragment shader runs, so highlight, visible pixels,
//! and edits all converge on one hit. No CPU/GPU algorithm pair to
//! drift out of sync, no interaction-radius gate (the walker already
//! enforces Nyquist LOD termination), and no path-length mismatch
//! between what the user sees and what the next edit will target.

use crate::app::App;

impl App {
    pub(in crate::app) fn update_highlight(&mut self) {
        let raycast_start = std::time::Instant::now();
        let probe = if self.disable_highlight || !self.cursor_locked {
            None
        } else {
            self.renderer.as_ref().map(|r| r.read_cursor_probe())
        };
        self.last_highlight_raycast_ms =
            raycast_start.elapsed().as_secs_f64() * 1000.0;

        let set_start = std::time::Instant::now();
        let slots: Vec<u8> = probe
            .as_ref()
            .filter(|p| p.hit)
            .map(|p| p.slots.clone())
            .unwrap_or_default();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight_path(&slots);
        }
        self.last_highlight_set_ms =
            set_start.elapsed().as_secs_f64() * 1000.0;
    }
}
