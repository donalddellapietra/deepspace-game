//! Cursor highlight: ray-cast, compute the hit's AABB in the current
//! frame's coords, and push to the renderer's highlight uniform.
//!
//! Also derives the **crosshair** reticle's on-target bit from the
//! same raycast and pushes it to the HTML overlay. The crosshair
//! lives in the DOM, not the shader, so it stays pixel-crisp under
//! TAAU / resolution scaling / any future temporal effect. See
//! `docs/testing/proposed-perf-speedups.md` for the why.

use crate::bridge::{CrosshairStateJs, GameStateUpdate};
use crate::overlay;
use crate::world::aabb;

use crate::app::{ActiveFrameKind, App, HighlightCacheKey};

impl App {
    pub(in crate::app) fn update_highlight(&mut self) {
        if self.disable_highlight {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            self.push_crosshair(false, false);
            return;
        }
        // WrappedPlane frame: blocks are rotated tangent cubes, so the
        // axis-aligned `hit_aabb_in_frame_local` would draw at the
        // wrong place. Skip highlight; break/place still target the
        // correct slab cell because the raycast returns a path.
        if matches!(self.active_frame.kind, ActiveFrameKind::WrappedPlane { .. }) {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            self.push_crosshair(self.cursor_locked, false);
            return;
        }
        if !self.cursor_locked {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            // Crosshair is gameplay-only; hide while any menu is
            // open (cursor unlocked). CSS toggles the element's
            // visibility so the overlay stays cheap.
            self.push_crosshair(false, false);
            return;
        }
        let cache_key = HighlightCacheKey::new(self);
        if let Some((cached_key, cached_aabb)) = self.cached_highlight {
            if cached_key == cache_key {
                self.last_highlight_raycast_ms = 0.0;
                let set_start = web_time::Instant::now();
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_highlight(cached_aabb);
                }
                self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
                // Cached aabb tracks the raw hit: the cache key folds
                // in camera + world state, so identical key ⇒ same
                // on-target bit. Diff-push handles the no-op case.
                self.push_crosshair(true, cached_aabb.is_some());
                return;
            }
        }
        let raycast_start = web_time::Instant::now();
        let tree_hit = self.frame_aware_raycast();
        self.last_highlight_raycast_ms = raycast_start.elapsed().as_secs_f64() * 1000.0;
        // Reuse the raycast result for the crosshair reticle. Its
        // "on-target" color flip now matches the same voxel-hit
        // test the highlight AABB is derived from — no separate
        // raycast needed.
        self.push_crosshair(true, tree_hit.is_some());
        if self.startup_profile_frames < 16 {
            eprintln!(
                "highlight_update frame={} cursor_locked={} hit={}",
                self.startup_profile_frames,
                self.cursor_locked,
                tree_hit.is_some(),
            );
        }
        let aabb = tree_hit
            .as_ref()
            .map(|hit| aabb::hit_aabb_in_frame_local(hit, &self.active_frame.render_path));
        let set_start = web_time::Instant::now();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
        self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
        self.cached_highlight = Some((cache_key, aabb));
    }

    /// Diff-push the crosshair reticle state to the HTML overlay.
    /// Only emits an IPC message when either `visible` or `on_target`
    /// flips — most frames are no-ops, so the overhead is noise.
    fn push_crosshair(&mut self, visible: bool, on_target: bool) {
        let next = CrosshairStateJs { visible, on_target };
        if self.last_crosshair_sent.as_ref() == Some(&next) {
            return;
        }
        overlay::push_state(&GameStateUpdate::Crosshair(next.clone()));
        self.last_crosshair_sent = Some(next);
    }
}
