//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified frame-aware raycast →
//! `break_block` / `place_block` pipeline. Cartesian and
//! cubed-sphere layers resolve through the same active-frame
//! contract; sphere layers use a bounded face window instead of a
//! separate coarser edit path.

mod break_place;
mod highlight;
mod spawn;
pub(crate) mod upload;
mod zoom;

use crate::world::anchor::Path;
use crate::world::{aabb, raycast};

use super::{ActiveFrameKind, App};

/// CPU-side ceiling for `visual_depth()` (feeds the sphere/face
/// walker's `uniforms.max_depth`). Picked equal to the tree's
/// absolute max so the sphere path isn't artificially capped —
/// the Cartesian path doesn't use this any more.
pub(super) const MAX_LOCAL_VISUAL_DEPTH: u32 = crate::world::tree::MAX_DEPTH as u32;
pub(super) const MAX_FOCUSED_FRAME_CAMERA_EXTENT: f32 = 8.0;
pub(super) const FRAME_VISUAL_MIN_PIXELS: f32 = 1.0;
pub(super) const FRAME_FOCUS_MIN_PIXELS: f32 = 1.0;

impl App {
    pub(super) fn ray_dir_in_frame(&self, _frame_path: &Path) -> [f32; 3] {
        // In Cartesian frames, all levels share the same axes — the
        // direction is identical in every frame.  The DDA only cares
        // about the *direction*, not the magnitude.  The old code
        // scaled by 3^depth which overflows f32 past depth ~20.
        crate::world::sdf::normalize(self.camera.forward())
    }

    /// Interaction distance cap in the given frame's local units.
    /// = `interaction_radius_cells × anchor_cell_size_in_frame`,
    /// where `anchor_cell_size_in_frame = 3 / 3^K` for K = anchor
    /// depth minus frame depth (K ≥ 0). `ray_dir_in_frame` is
    /// normalized, so `HitInfo.t` is a frame-local distance and
    /// can be compared directly to this value.
    pub(super) fn interaction_range_in_frame(&self, frame_path: &Path) -> f32 {
        let frame_depth = frame_path.depth();
        let anchor_depth = self.camera.position.anchor.depth();
        let k = anchor_depth.saturating_sub(frame_depth) as i32;
        let anchor_cell_size_in_frame = 3.0_f32.powi(1 - k);
        self.interaction_radius_cells as f32 * anchor_cell_size_in_frame
    }

    /// Cast a ray from the camera into the world using the same
    /// frame-aware machinery as the renderer: the cpu raycast
    /// runs in frame-local coordinates and pops upward via the
    /// camera's anchor when it exits the frame's bubble. This is
    /// what makes deep-zoom block placement land in the cell
    /// that's actually under the crosshair, instead of being
    /// pinned to the f32-precision wall of world XYZ.
    pub(in crate::app) fn frame_aware_raycast(&self) -> Option<raycast::HitInfo> {
        // Distance cap frame-path: sphere raycasts measure `t` in
        // body-frame local units; Cartesian raycasts measure `t` in
        // render-frame local units. The frame we use for the
        // anchor-cell-size math must match whichever path produced
        // the hit.
        let (hit, cap_frame_path) = match self.active_frame.kind {
            ActiveFrameKind::Sphere(sphere) => {
                let cam_body = self.camera.position.in_frame(&sphere.body_path);
                let ray_dir_local = self.ray_dir_in_frame(&sphere.body_path);
                let hit = raycast::cpu_raycast_in_sphere_frame(
                    &self.world.library,
                    self.world.root,
                    sphere.body_path.as_slice(),
                    cam_body,
                    cam_body,
                    ray_dir_local,
                    self.cs_edit_depth(),
                    sphere.face as u32,
                    sphere.face_u_min,
                    sphere.face_v_min,
                    sphere.face_r_min,
                    sphere.face_size,
                    sphere.inner_r,
                    sphere.outer_r,
                    sphere.face_depth,
                );
                (hit, sphere.body_path)
            }
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                // Raycast from the render frame — f32 can only represent
                // positions a few levels deeper than the frame root.
                // The pop loop handles finding hits at coarser depths
                // via slot-arithmetic frame transitions.
                let frame_path = self.active_frame.render_path;
                let cam_local = self.camera.position.in_frame(&frame_path);
                let ray_dir = self.ray_dir_in_frame(&frame_path);
                let hit = raycast::cpu_raycast_in_frame(
                    &self.world.library,
                    self.world.root,
                    frame_path.as_slice(),
                    cam_local,
                    ray_dir,
                    self.edit_depth(),
                    self.cs_edit_depth(),
                );
                if hit.is_none() && self.startup_profile_frames < 16 {
                    eprintln!(
                        "frame_raycast_cartesian_miss edit_depth={} render_path={:?}",
                        self.edit_depth(),
                        frame_path.as_slice(),
                    );
                }
                (hit, frame_path)
            }
        };
        // Enforce the interaction radius gate: drop hits beyond
        // `interaction_radius_cells × anchor_cell_size`. Same
        // cubic-shell LOD philosophy as the shader — out of range
        // of your current anchor locality, cursor shows no hit.
        let hit = hit.and_then(|h| {
            let max_t = self.interaction_range_in_frame(&cap_frame_path);
            if h.t <= max_t {
                Some(h)
            } else {
                if self.startup_profile_frames < 16 {
                    eprintln!(
                        "interaction_radius_reject t={:.4} max_t={:.4} cells={} anchor_depth={} frame_depth={}",
                        h.t,
                        max_t,
                        self.interaction_radius_cells,
                        self.camera.position.anchor.depth(),
                        cap_frame_path.depth(),
                    );
                }
                None
            }
        });
        if self.startup_profile_frames < 16 {
            eprintln!(
                "frame_raycast frame={} kind={:?} render_path={:?} logical_path={:?} cam_anchor={:?} hit={}",
                self.startup_profile_frames,
                self.active_frame.kind,
                self.active_frame.render_path.as_slice(),
                self.active_frame.logical_path.as_slice(),
                self.camera.position.anchor.as_slice(),
                hit.is_some(),
            );
            if let Some(ref h) = hit {
                let (aabb_min, aabb_max) = match self.active_frame.kind {
                    ActiveFrameKind::Sphere(_) => {
                        aabb::hit_aabb_body_local(&self.world.library, h)
                    }
                    ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                        aabb::hit_aabb_in_frame_local(h, &self.active_frame.render_path)
                    }
                };
                eprintln!(
                    "frame_raycast_hit path_len={} face={} t={} place_path_len={:?} terminal={} aabb_min={:?} aabb_max={:?} path_kinds={:?}",
                    h.path.len(),
                    h.face,
                    h.t,
                    h.place_path.as_ref().map(|p| p.len()),
                    self.debug_hit_terminal(h),
                    aabb_min,
                    aabb_max,
                    self.debug_path_kinds(&{
                        let mut p = Path::root();
                        for &(_, slot) in &h.path {
                            p.push(slot as u8);
                        }
                        p
                    }),
                );
            }
        }
        hit
    }

    fn debug_path_kinds(&self, path: &Path) -> Vec<String> {
        use crate::world::tree::{Child, NodeKind};

        let mut out = Vec::new();
        let mut node_id = self.world.root;
        out.push(format!("root:{:?}", self.world.library.get(node_id).map(|n| n.kind)));
        for (depth, &slot) in path.as_slice().iter().enumerate() {
            let kind = self.world.library.get(node_id).map(|n| n.kind);
            let next = self
                .world
                .library
                .get(node_id)
                .and_then(|n| match n.children[slot as usize] {
                    Child::Node(child_id) => Some(child_id),
                    Child::Block(block) => {
                        out.push(format!(
                            "d{} slot={} parent={kind:?} -> Block({block})",
                            depth + 1,
                            slot
                        ));
                        None
                    }
                    Child::Empty => {
                        out.push(format!(
                            "d{} slot={} parent={kind:?} -> Empty",
                            depth + 1,
                            slot
                        ));
                        None
                    }
                    Child::EntityRef(idx) => {
                        out.push(format!(
                            "d{} slot={} parent={kind:?} -> EntityRef({idx})",
                            depth + 1,
                            slot
                        ));
                        None
                    }
                });
            let Some(child_id) = next else { break };
            let child_kind = self.world.library.get(child_id).map(|n| n.kind);
            out.push(format!(
                "d{} slot={} parent={kind:?} -> node={child_id} kind={child_kind:?}",
                depth + 1,
                slot
            ));
            match child_kind {
                Some(NodeKind::Cartesian)
                | Some(NodeKind::CubedSphereBody { .. })
                | Some(NodeKind::CubedSphereFace { .. }) => {
                    node_id = child_id;
                }
                None => break,
            }
        }
        out
    }

    fn debug_hit_terminal(&self, hit: &raycast::HitInfo) -> String {
        use crate::world::tree::Child;

        let Some(&(node_id, slot)) = hit.path.last() else {
            return "empty-hit-path".to_string();
        };
        let Some(node) = self.world.library.get(node_id) else {
            return format!("missing-node node_id={node_id} slot={slot}");
        };
        match node.children[slot] {
            Child::Empty => format!("Empty node_id={node_id} slot={slot}"),
            Child::Block(block) => format!("Block({block}) node_id={node_id} slot={slot}"),
            Child::EntityRef(idx) => {
                format!("EntityRef({idx}) node_id={node_id} slot={slot}")
            }
            Child::Node(child_id) => {
                let desc = self
                    .world
                    .library
                    .get(child_id)
                    .map(|child| {
                        format!(
                            "Node({child_id}) kind={:?} uniform_type={} rep_block={}",
                            child.kind,
                            child.uniform_type,
                            child.representative_block
                        )
                    })
                    .unwrap_or_else(|| format!("Node({child_id}) missing"));
                format!("{desc} node_id={node_id} slot={slot}")
            }
        }
    }
}
