//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified frame-aware raycast →
//! `break_block` / `place_block` pipeline. Cartesian and
//! cubed-sphere layers resolve through the same active-frame
//! contract; sphere layers use a bounded face window instead of a
//! separate coarser edit path.

mod break_place;
mod highlight;
pub(crate) mod upload;
mod zoom;

use crate::world::anchor::Path;
use crate::world::{aabb, raycast};

use super::App;

pub(super) const MAX_LOCAL_VISUAL_DEPTH: u32 = 12;
pub(super) const MAX_FOCUSED_FRAME_CAMERA_EXTENT: f32 = 8.0;
pub(super) const FRAME_VISUAL_MIN_PIXELS: f32 = 1.0;
pub(super) const FRAME_FOCUS_MIN_PIXELS: f32 = 1.0;

impl App {
    /// Trim a walker-found hit down to the current `edit_depth`. The
    /// walker returns a path all the way to the tree leaf so the
    /// cursor always hits something concrete; callers that act on
    /// the hit (break, place, highlight) want to operate at the
    /// user's current layer instead, so we strip trailing slots
    /// before returning. `place_path` receives the same trim so the
    /// "where would I place a new block" preview is layer-correct.
    fn truncate_hit_to_edit_depth(&self, mut hit: raycast::HitInfo) -> raycast::HitInfo {
        let edit_depth = self.edit_depth() as usize;
        if hit.path.len() > edit_depth {
            hit.path.truncate(edit_depth);
        }
        if let Some(ref mut pp) = hit.place_path {
            if pp.len() > edit_depth {
                pp.truncate(edit_depth);
            }
        }
        hit
    }

    /// Interaction distance cap in the given frame's local units.
    ///
    /// `interaction_radius_cells × anchor_cell_size_in_frame` with
    /// `anchor_cell_size_in_frame = 3 / 3^K`, `K = anchor_depth −
    /// frame_depth`. Reach scales with zoom: "N layer-sized cells
    /// away" stays meaningful whether you're zoomed out viewing a
    /// whole planet or zoomed in on a single block.
    ///
    /// Sphere worlds floor `anchor_cell_size` at the SDF's smallest
    /// representable cell. The SDF stops subdividing at
    /// `SDF_DETAIL_LEVELS` below each face root; below that the
    /// walker can't find cells smaller than the floor no matter how
    /// deep the anchor sits, so "12 anchor cells" below the floor
    /// would reject hits on content that physically exists. Converting
    /// to render-frame units multiplies the floor by `3^face_depth`
    /// (a deeper render frame makes the same physical cell span
    /// proportionally more frame-local units).
    pub(super) fn interaction_range_in_frame(&self, frame_path: &Path) -> f32 {
        let frame_depth = frame_path.depth();
        let anchor_depth = self.camera.position.anchor.depth();
        let k = anchor_depth.saturating_sub(frame_depth) as i32;
        let anchor_cell_size_in_frame = 3.0_f32.powi(1 - k);

        // For sphere-adjacent frames the reach floor is the body's
        // outer diameter (2 × outer_r × 3) in render-frame units —
        // cursor highlight and edit should be able to reach across
        // the whole planet, including rays that pass through
        // carved-air tunnels and hit solid on the far side. The old
        // `sdf_min_cell` scaling coupled reach to SDF recursion
        // depth and shrank inversely with detail; the corrected
        // floor is purely geometric (cursor reach scales with the
        // visible planet size, independent of voxel resolution).
        let effective_cell_size = match self.active_frame.kind {
            crate::app::ActiveFrameKind::Sphere(sphere) => {
                let body_depth = sphere.body_path.depth();
                let render_depth = self.active_frame.render_path.depth();
                let face_depth_i32 =
                    render_depth.saturating_sub(body_depth) as i32;
                // Body diameter in the render frame: `outer_r * 3 * 2`
                // at body-level, scaled by `3^face_depth` for each
                // level of descent into the face subtree.
                let body_diameter_in_frame = sphere.outer_r * 6.0
                    * 3.0_f32.powi(face_depth_i32);
                let reach_floor = body_diameter_in_frame
                    * (1.0 / self.interaction_radius_cells.max(1) as f32);
                anchor_cell_size_in_frame.max(reach_floor)
            }
            crate::app::ActiveFrameKind::Body { outer_r, .. } => {
                // Body frame: render root IS the body cell. Outer
                // diameter in body-cell [0, 3)³ = `outer_r * 6`.
                let body_diameter_in_frame = outer_r * 6.0;
                let reach_floor = body_diameter_in_frame
                    * (1.0 / self.interaction_radius_cells.max(1) as f32);
                anchor_cell_size_in_frame.max(reach_floor)
            }
            crate::app::ActiveFrameKind::Cartesian => anchor_cell_size_in_frame,
        };
        self.interaction_radius_cells as f32 * effective_cell_size
    }

    /// Cast a ray from the camera into the world using the same
    /// frame-aware machinery as the renderer: the cpu raycast
    /// runs in frame-local coordinates and pops upward via the
    /// camera's anchor when it exits the frame's bubble. This is
    /// what makes deep-zoom block placement land in the cell
    /// that's actually under the crosshair, instead of being
    /// pinned to the f32-precision wall of world XYZ.
    pub(in crate::app) fn frame_aware_raycast(&self) -> Option<raycast::HitInfo> {
        // Unified Cartesian raycast in the render_path frame.
        //
        // The only per-frame-kind adjustment is the ray direction:
        // when the render frame descends into a face subtree
        // (Sphere { face_depth >= 1 }), the forward vector must be
        // rotated into the face's local `(u_axis, v_axis, n_axis)`
        // basis so the Cartesian walker navigates the face subtree
        // correctly. Otherwise world axes are fine.
        //
        // At face_depth == 0 (the body cell IS the render frame) or
        // outside a sphere, world axes + `in_frame(render_path)` are
        // precision-clean because the frame is bounded.
        use crate::world::{cubesphere_local, sdf};
        let frame_path = self.active_frame.render_path;
        let cam_local = self.camera.position.in_frame(&frame_path);
        let fwd_world = sdf::normalize(self.camera.forward());
        let ray_dir = match self.active_frame.kind {
            crate::app::ActiveFrameKind::Sphere(sphere) if sphere.face_depth >= 1 => {
                sdf::normalize(cubesphere_local::world_vec_to_face_axes(
                    fwd_world, sphere.face,
                ))
            }
            _ => fwd_world,
        };
        let hit = raycast::cpu_raycast_in_frame(
            &self.world.library,
            self.world.root,
            frame_path.as_slice(),
            cam_local,
            ray_dir,
            self.raycast_max_depth(),
            self.cs_edit_depth(),
        );
        let (hit, cap_frame_path) = (hit, frame_path);
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

        // Truncate to edit_depth. The walker descends all the way to
        // the leaf so the cursor ALWAYS lands on something (bug:
        // cursor doesn't show when the leaf is small and the
        // anchor-depth coarse cell reads as empty). But break/place
        // — and the cursor highlight that previews them — must
        // operate at the user's current layer, not at the leaf.
        // Stripping the tail of `hit.path` / `hit.place_path` down to
        // `edit_depth` slots gives all consumers the correct
        // layer-N cell while still benefiting from the deeper walk.
        let hit = hit.map(|h| self.truncate_hit_to_edit_depth(h));
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
