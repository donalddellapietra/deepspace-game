//! Cursor highlight: ray-cast, extract the hit cell's slot path
//! from the world root, and push it to the renderer. The shader
//! compares that path prefix-wise against each pixel's walker
//! descent — precision-safe at any anchor depth (no f32 AABB
//! representation involved).

use crate::app::{App, HighlightCacheKey};

impl App {
    pub(in crate::app) fn update_highlight(&mut self) {
        if self.disable_highlight {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight_path(&[]);
            }
            return;
        }
        if !self.cursor_locked {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight_path(&[]);
            }
            return;
        }
        let cache_key = HighlightCacheKey::new(self);
        if let Some((cached_key, cached_aabb)) = self.cached_highlight {
            if cached_key == cache_key {
                self.last_highlight_raycast_ms = 0.0;
                // Re-ship the path unconditionally — cheap write,
                // keeps the shader's uniform in sync after other
                // uniform updates (root_kind changes, etc.) that
                // also call write_uniforms().
                let _ = cached_aabb;
                return;
            }
        }
        let raycast_start = std::time::Instant::now();
        let tree_hit = self.frame_aware_raycast();
        self.last_highlight_raycast_ms = raycast_start.elapsed().as_secs_f64() * 1000.0;
        if self.startup_profile_frames < 16 {
            eprintln!(
                "highlight_update frame={} cursor_locked={} hit={}",
                self.startup_profile_frames,
                self.cursor_locked,
                tree_hit.is_some(),
            );
        }
        let set_start = std::time::Instant::now();
        if let Some(renderer) = &mut self.renderer {
            match &tree_hit {
                Some(hit) => {
                    let slots: Vec<u8> = hit.path.iter().map(|&(_, s)| s as u8).collect();
                    renderer.set_highlight_path(&slots);
                }
                None => renderer.set_highlight_path(&[]),
            }
        }
        self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
        self.cached_highlight = Some((cache_key, None));
    }
}
