//! Unified cursor raycast: composes the world raycast with the
//! entity raycast and returns whichever hit (if any) is closer.
//!
//! World hits reuse the existing `frame_aware_raycast`. Entity hits
//! iterate `entities.entities` (v1 linear scan), ray-box against
//! each entity's bbox, then — on AABB hit — run the existing
//! `cpu_raycast` against the entity's subtree in local coords.
//! This reuses all the tree-walking machinery; entities are a
//! different starting root, nothing more.

use crate::world::anchor::{WorldPos, WORLD_SIZE};
use crate::world::raycast::{self, HitInfo};

use crate::app::App;

/// A cursor hit disambiguated between world content and an entity.
/// `path` in each inner HitInfo is rooted at:
/// - `world.root` for `World(..)`
/// - the entity's active subtree root for `Entity{..}`
pub enum CursorHit {
    World(HitInfo),
    Entity { entity_idx: u32, inner: HitInfo },
}

impl CursorHit {
    pub fn t(&self) -> f32 {
        match self {
            CursorHit::World(h) => h.t,
            CursorHit::Entity { inner, .. } => inner.t,
        }
    }
}

impl App {
    /// Cast a ray into the world AND into every entity; return
    /// whichever hit is closer (or `None` if both missed).
    pub(in crate::app) fn cursor_raycast(&self) -> Option<CursorHit> {
        let world_hit = self.frame_aware_raycast();
        let entity_hit = self.raycast_entities();
        match (world_hit, entity_hit) {
            (Some(w), Some((idx, e))) => {
                if e.t < w.t {
                    Some(CursorHit::Entity { entity_idx: idx, inner: e })
                } else {
                    Some(CursorHit::World(w))
                }
            }
            (Some(w), None) => Some(CursorHit::World(w)),
            (None, Some((idx, e))) => Some(CursorHit::Entity { entity_idx: idx, inner: e }),
            (None, None) => None,
        }
    }

    /// Raycast against every entity. Returns `(entity_idx, hit)` for
    /// the closest one, with `hit.path` rooted at the entity's
    /// active subtree. Uniform-scale transform means local `t` is
    /// directly comparable to world `t`.
    fn raycast_entities(&self) -> Option<(u32, HitInfo)> {
        if self.entities.is_empty() {
            return None;
        }
        let frame = self.active_frame.render_path;
        let frame_depth = frame.depth() as i32;
        let cam_local = self.camera.position.in_frame(&frame);
        let ray_dir = self.ray_dir_in_frame(&frame);

        let inv_dir = [
            if ray_dir[0].abs() > 1e-8 { 1.0 / ray_dir[0] } else { 1e10 },
            if ray_dir[1].abs() > 1e-8 { 1.0 / ray_dir[1] } else { 1e10 },
            if ray_dir[2].abs() > 1e-8 { 1.0 / ray_dir[2] } else { 1e10 },
        ];

        let mut best: Option<(u32, HitInfo)> = None;
        let mut best_t = f32::INFINITY;

        for (i, e) in self.entities.entities.iter().enumerate() {
            let anchor_depth = e.anchor.depth() as i32;
            if anchor_depth < frame_depth {
                continue;
            }
            let origin_pos = WorldPos::new_unchecked(e.anchor, [0.0, 0.0, 0.0]);
            let bbox_min = origin_pos.in_frame(&frame);
            let size = WORLD_SIZE / 3.0_f32.powi(anchor_depth - frame_depth);
            let bbox_max = [
                bbox_min[0] + size,
                bbox_min[1] + size,
                bbox_min[2] + size,
            ];

            let (t_enter, t_exit) = ray_aabb(cam_local, inv_dir, bbox_min, bbox_max);
            if t_enter >= t_exit || t_exit < 0.0 {
                continue;
            }
            if t_enter >= best_t {
                continue;
            }

            // Transform ray to entity-local [0, 3)³. Uniform scale
            // so `t_local == t_world` — directly comparable.
            let scale = 3.0 / size;
            let local_origin = [
                (cam_local[0] - bbox_min[0]) * scale,
                (cam_local[1] - bbox_min[1]) * scale,
                (cam_local[2] - bbox_min[2]) * scale,
            ];
            let local_dir = [
                ray_dir[0] * scale,
                ray_dir[1] * scale,
                ray_dir[2] * scale,
            ];

            // Edit-depth cap: descend as deep as the entity subtree
            // goes (6 is a reasonable upper bound matching the
            // shader's MAX_STACK_DEPTH-ish depth).
            const ENTITY_EDIT_DEPTH: u32 = 6;
            let hit = raycast::cpu_raycast(
                &self.world.library,
                e.active_root(),
                local_origin,
                local_dir,
                ENTITY_EDIT_DEPTH,
            );
            if let Some(h) = hit {
                if h.t < best_t {
                    best_t = h.t;
                    best = Some((i as u32, h));
                }
            }
        }
        best
    }
}

fn ray_aabb(
    origin: [f32; 3],
    inv_dir: [f32; 3],
    bmin: [f32; 3],
    bmax: [f32; 3],
) -> (f32, f32) {
    let t1 = [
        (bmin[0] - origin[0]) * inv_dir[0],
        (bmin[1] - origin[1]) * inv_dir[1],
        (bmin[2] - origin[2]) * inv_dir[2],
    ];
    let t2 = [
        (bmax[0] - origin[0]) * inv_dir[0],
        (bmax[1] - origin[1]) * inv_dir[1],
        (bmax[2] - origin[2]) * inv_dir[2],
    ];
    let t_enter = t1[0]
        .min(t2[0])
        .max(t1[1].min(t2[1]))
        .max(t1[2].min(t2[2]));
    let t_exit = t1[0]
        .max(t2[0])
        .min(t1[1].max(t2[1]))
        .min(t1[2].max(t2[2]));
    (t_enter, t_exit)
}
