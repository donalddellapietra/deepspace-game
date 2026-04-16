//! Cursor highlight: ray-cast, compute the hit's AABB in the current
//! frame's coords, and push to the renderer's highlight uniform.

use crate::world::aabb;

use super::super::{ActiveFrameKind, App, HighlightCacheKey};

impl App {
    pub(in crate::app) fn update_highlight(&mut self) {
        if self.disable_highlight {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        if !self.cursor_locked {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let cache_key = HighlightCacheKey::new(self);
        if let Some((cached_key, cached_aabb)) = self.cached_highlight {
            if cached_key == cache_key {
                self.last_highlight_raycast_ms = 0.0;
                let set_start = std::time::Instant::now();
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_highlight(cached_aabb);
                }
                self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
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
        let aabb = tree_hit.as_ref().map(|hit| match self.active_frame.kind {
            ActiveFrameKind::Sphere(_) => aabb::hit_aabb_body_local(&self.world.library, hit),
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                aabb::hit_aabb_in_frame_local(hit, &self.active_frame.render_path)
            }
        });
        let set_start = std::time::Instant::now();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
        self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
        self.cached_highlight = Some((cache_key, aabb));
    }
}
