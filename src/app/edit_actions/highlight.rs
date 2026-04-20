//! Cursor highlight: push the GPU-computed cursor hit path to the
//! renderer uniform. The GPU compute pipeline (`cursor_probe.wgsl`)
//! runs every frame and writes the crosshair ray's hit cell path
//! into a buffer that `test_runner`/`event_loop` maps back into
//! `App.last_cursor_hit`. This file just forwards that path — no
//! CPU raycast, no AABB, no frame-kind split, no algorithm that
//! could drift from the shader.

use crate::app::App;

impl App {
    pub(in crate::app) fn update_highlight(&mut self) {
        self.last_highlight_raycast_ms = 0.0;
        let set_start = std::time::Instant::now();
        let slots: Vec<u8> = match (self.disable_highlight, self.cursor_locked) {
            (false, true) => self
                .last_cursor_hit
                .as_ref()
                .filter(|h| h.hit)
                .map(|h| h.slots.clone())
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight_path(&slots);
        }
        self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
    }
}
