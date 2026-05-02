//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified frame-aware raycast →
//! `break_block` / `place_block` pipeline.

mod break_place;
mod highlight;
mod spawn;
pub(crate) mod upload;
mod zoom;

use crate::world::anchor::Path;
use crate::world::{aabb, raycast};

use super::{ActiveFrameKind, App};

/// CPU-side ceiling for `visual_depth()`.
pub(super) const MAX_LOCAL_VISUAL_DEPTH: u32 = crate::world::tree::MAX_DEPTH as u32;
pub(super) const MAX_FOCUSED_FRAME_CAMERA_EXTENT: f32 = 8.0;
pub(super) const FRAME_VISUAL_MIN_PIXELS: f32 = 1.0;
pub(super) const FRAME_FOCUS_MIN_PIXELS: f32 = 1.0;

impl App {
    pub(super) fn ray_dir_in_frame(&self, frame_path: &Path) -> [f32; 3] {
        let fwd = crate::world::sdf::normalize(self.camera.forward());
        let frame_rot = super::frame_path_rotation(
            &self.world.library, self.world.root, frame_path,
        );
        let rotated = super::mat3_transpose_mul_vec3(&frame_rot, &fwd);
        crate::world::sdf::normalize(rotated)
    }

    /// Interaction distance cap in the given frame's local units.
    /// = `interaction_radius_cells × anchor_cell_size_in_frame`,
    /// where `anchor_cell_size_in_frame = 3 / 3^K` for K = anchor
    /// depth minus frame depth (K ≥ 0). `ray_dir_in_frame` is
    /// normalized, so `HitInfo.t` is a frame-local distance and
    /// can be compared directly to this value.
    ///
    /// `K` is capped at `MAX_INTERACTION_K_BELOW_FRAME` to prevent
    /// exponential collapse when the render frame is locked above
    /// the anchor (sphere/tangent mode: WrappedPlane stays the
    /// frame even as anchor depth grows). Without the cap, every
    /// extra anchor level shrinks the interaction range by 1/3 —
    /// the cursor stops hitting anything within a few levels of
    /// "zoom" because the player's WORLD position to the cursor
    /// hit didn't shrink, only the scale we measured it against
    /// did. In Cartesian mode the frame follows the anchor with a
    /// render margin of ~3, so `K` is naturally bounded and the
    /// cap doesn't fire.
    pub(super) fn interaction_range_in_frame(&self, frame_path: &Path) -> f32 {
        const MAX_INTERACTION_K_BELOW_FRAME: i32 = 5;
        let frame_depth = frame_path.depth();
        let anchor_depth = self.camera.position.anchor.depth();
        let k = anchor_depth.saturating_sub(frame_depth) as i32;
        let k_capped = k.min(MAX_INTERACTION_K_BELOW_FRAME);
        let anchor_cell_size_in_frame = 3.0_f32.powi(1 - k_capped);
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
        let (hit, cap_frame_path) = match self.active_frame.kind {
            // WrappedPlane frame: dispatch the rotated-tangent-cube CPU
            // raycast so click-targeting matches the GPU visual.
            ActiveFrameKind::WrappedPlane { dims, slab_depth } => {
                let frame_path = self.active_frame.render_path;
                let cam_local = self.camera.position.in_frame(&frame_path);
                let ray_dir = self.ray_dir_in_frame(&frame_path);
                // lat_max kept in sync with the shader-side default
                // (1.26 rad ≈ 72°).
                let hit = raycast::cpu_raycast_wrapped_planet(
                    &self.world.library,
                    self.world.root,
                    frame_path.as_slice(),
                    cam_local,
                    ray_dir,
                    dims,
                    slab_depth,
                    1.26,
                    self.edit_depth(),
                );
                (hit, frame_path)
            }
            ActiveFrameKind::Cartesian => {
                let frame_path = self.active_frame.render_path;
                let mut cam_local = self.camera.position.in_frame(&frame_path);
                let ray_dir = self.ray_dir_in_frame(&frame_path);
                let tbc = self.active_frame.tb_center;
                if tbc[0] != 0.0 || tbc[1] != 0.0 || tbc[2] != 0.0 {
                    if let Some(rot) = self.find_frame_path_tb_rotation() {
                        let c = [cam_local[0]-tbc[0], cam_local[1]-tbc[1], cam_local[2]-tbc[2]];
                        cam_local = [
                            tbc[0] + rot[0][0]*c[0]+rot[0][1]*c[1]+rot[0][2]*c[2],
                            tbc[1] + rot[1][0]*c[0]+rot[1][1]*c[1]+rot[1][2]*c[2],
                            tbc[2] + rot[2][0]*c[0]+rot[2][1]*c[1]+rot[2][2]*c[2],
                        ];
                    }
                }
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
                let (aabb_min, aabb_max) =
                    aabb::hit_aabb_in_frame_local(h, &self.active_frame.render_path);
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
                | Some(NodeKind::WrappedPlane { .. })
                | Some(NodeKind::TangentBlock { .. }) => {
                    node_id = child_id;
                }
                None => break,
            }
        }
        out
    }

    fn find_frame_path_tb_rotation(&self) -> Option<[[f32; 3]; 3]> {
        use crate::world::tree::{Child, NodeKind};
        let mut node = self.world.root;
        for k in 0..self.active_frame.render_path.depth() as usize {
            let slot = self.active_frame.render_path.slot(k) as usize;
            let n = self.world.library.get(node)?;
            match n.children[slot] {
                Child::Node(child_id) => {
                    if let Some(child) = self.world.library.get(child_id) {
                        if let NodeKind::TangentBlock { rotation } = child.kind {
                            return Some(rotation);
                        }
                    }
                    node = child_id;
                }
                _ => return None,
            }
        }
        None
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
