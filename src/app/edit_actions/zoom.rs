//! Zoom: depth accessors + camera-aware frame selection + apply.

use crate::world::anchor::WORLD_SIZE;

use crate::app::frame;
use crate::app::{ActiveFrame, ActiveFrameKind, App};
use super::{
    FRAME_FOCUS_MIN_PIXELS, FRAME_VISUAL_MIN_PIXELS, MAX_FOCUSED_FRAME_CAMERA_EXTENT,
    MAX_LOCAL_VISUAL_DEPTH,
};

impl App {
    pub(in crate::app) fn edit_depth(&self) -> u32 {
        if let Some(depth) = self.forced_edit_depth {
            return depth.max(1).min(crate::world::tree::MAX_DEPTH as u32);
        }
        // Edit depth = anchor depth. The CPU raycast's pop loop
        // ensures the ray reaches the surface at this resolution.
        self.anchor_depth().max(1)
    }

    /// Cs edit-depth — retained as an alias for `edit_depth` so the
    /// frame-aware raycast call site stays unchanged across the
    /// sphere → cartesian migration. Same value either way.
    pub(in crate::app) fn cs_edit_depth(&self) -> u32 {
        self.edit_depth()
    }

    pub(in crate::app) fn visual_depth(&self) -> u32 {
        if let Some(depth) = self.forced_visual_depth {
            return depth.max(1).min(crate::world::tree::MAX_DEPTH as u32);
        }
        let local_target = self.edit_depth()
            .saturating_sub(self.active_frame.render_path.depth() as u32)
            .max(1);
        let pixels = self.frame_projected_pixels(&self.active_frame);
        let local_cap = if pixels <= FRAME_VISUAL_MIN_PIXELS {
            1
        } else {
            let extra = (pixels / FRAME_VISUAL_MIN_PIXELS).ln() / 3.0_f32.ln();
            extra.floor().max(1.0) as u32
        };
        local_target
            .min(local_cap)
            .min(MAX_LOCAL_VISUAL_DEPTH)
            .min(crate::world::tree::MAX_DEPTH as u32)
    }

    fn camera_fits_frame(&self, frame: &ActiveFrame) -> bool {
        match frame.kind {
            ActiveFrameKind::Cartesian | ActiveFrameKind::WrappedPlane { .. } => {
                let cam_local = self.camera.position.in_frame(&frame.render_path);
                cam_local.iter().all(|v| v.is_finite())
                    && cam_local.iter().all(|&v| {
                        (-MAX_FOCUSED_FRAME_CAMERA_EXTENT
                            ..=WORLD_SIZE + MAX_FOCUSED_FRAME_CAMERA_EXTENT)
                            .contains(&v)
                    })
            }
            ActiveFrameKind::SphereSubFrame(range) => {
                // Sub-frame is the camera's tangent column above the
                // sub-frame surface center. "Fits" iff the camera's
                // tangent offset (sub-frame local x, y) from the
                // sub-frame center is bounded by a small multiple of
                // the sub-frame's tangent extent. The radial axis
                // (sub-frame local z, the altitude direction) is
                // unbounded — a high-altitude camera looking down at
                // a tiny sub-frame patch must still "fit" the deep
                // frame, otherwise sub-frame dispatch never fires.
                let mut wp_path = frame.render_path;
                wp_path.truncate(range.wp_path_depth);
                let (fwd, right, up) = self.camera.basis();
                let local = crate::world::sphere_geom::camera_in_sphere_subframe(
                    &self.camera.position,
                    fwd, right, up,
                    &wp_path,
                    &range,
                    crate::world::anchor::WORLD_SIZE,
                );
                let r_sphere = crate::world::anchor::WORLD_SIZE / (2.0 * std::f32::consts::PI);
                let tangent_x = range.lon_extent() * r_sphere;
                let tangent_y = range.lat_extent() * r_sphere;
                let max_tangent = tangent_x.max(tangent_y).max(1e-9);
                let bound = MAX_FOCUSED_FRAME_CAMERA_EXTENT * max_tangent;
                local.origin.iter().all(|v| v.is_finite())
                    && local.origin[0].abs() <= bound
                    && local.origin[1].abs() <= bound
            }
        }
    }

    pub(in crate::app) fn frame_projected_pixels(&self, frame: &ActiveFrame) -> f32 {
        match frame.kind {
            ActiveFrameKind::Cartesian | ActiveFrameKind::WrappedPlane { .. } => {
                let cam_local = self.camera.position.in_frame(&frame.render_path);
                let to_center = crate::world::sdf::sub([1.5, 1.5, 1.5], cam_local);
                let dist = crate::world::sdf::length(to_center).max(0.05);
                let half_fov_recip = 720.0f32 / (2.0f32 * (1.2f32 * 0.5f32).tan());
                crate::world::anchor::WORLD_SIZE / dist * half_fov_recip
            }
            ActiveFrameKind::SphereSubFrame(range) => {
                // Sub-frame's "characteristic span" = max of its three
                // tangent-axis extents. Distance = camera origin
                // magnitude in sub-frame local coords (origin at
                // sub-frame center).
                let mut wp_path = frame.render_path;
                wp_path.truncate(range.wp_path_depth);
                let (fwd, right, up) = self.camera.basis();
                let local = crate::world::sphere_geom::camera_in_sphere_subframe(
                    &self.camera.position,
                    fwd, right, up,
                    &wp_path,
                    &range,
                    crate::world::anchor::WORLD_SIZE,
                );
                let dist = crate::world::sdf::length(local.origin).max(0.05);
                let span = range
                    .lon_extent()
                    .max(range.lat_extent())
                    .max(range.r_extent());
                let half_fov_recip = 720.0f32 / (2.0f32 * (1.2f32 * 0.5f32).tan());
                span / dist * half_fov_recip
            }
        }
    }

    pub(in crate::app) fn target_render_frame(&self) -> ActiveFrame {
        // Render frame depth is derived from `RENDER_ANCHOR_DEPTH`,
        // not the user's anchor depth — zoom controls interaction
        // layer, not what the camera renders. See `RENDER_ANCHOR_DEPTH`.
        //
        // Pop strategy: iterate `virtual_depth` (= the desired_depth
        // passed to `compute_render_frame`) downward until the camera
        // fits. For SphereSubFrame frames, reducing virtual_depth
        // shrinks the past-leaf range refinement — the sub-frame
        // grows toward the WP root until its tangent extent is large
        // enough to absorb the camera's f32 quantization noise.
        // Truncating `render_path` (the prior implementation) didn't
        // affect the virtual refinement and would collapse the kind
        // straight to Cartesian as soon as render_path dropped below
        // the WP — rendering the planet as a flat slab.
        let logical_path = self.camera.position.deepened_to(crate::app::RENDER_ANCHOR_DEPTH).anchor;
        let mut virtual_depth = crate::app::RENDER_ANCHOR_DEPTH
            .saturating_sub(crate::app::RENDER_FRAME_K)
            .min(crate::app::RENDER_FRAME_MAX_DEPTH);
        let mut frame = loop {
            let render = frame::compute_render_frame(
                &self.world.library,
                self.world.root,
                &logical_path,
                virtual_depth,
            );
            let candidate = ActiveFrame {
                render_path: render.render_path,
                logical_path,
                node_id: render.node_id,
                kind: render.kind,
            };
            let fits = self.camera_fits_frame(&candidate)
                && self.frame_projected_pixels(&candidate) >= FRAME_FOCUS_MIN_PIXELS;
            if fits || virtual_depth == 0 {
                break candidate;
            }
            virtual_depth = virtual_depth.saturating_sub(1);
        };
        // Tighten the render frame to the deeper context window
        // (mirrors `with_render_margin`). Skip when the frame is a
        // SphereSubFrame — those are already at the camera's logical
        // depth (`render_path` cannot be deepened past where the
        // library has Nodes).
        if !matches!(frame.kind, ActiveFrameKind::SphereSubFrame(_)) {
            let render = frame::with_render_margin(
                &self.world.library,
                self.world.root,
                &logical_path,
                crate::app::RENDER_FRAME_CONTEXT,
            );
            if render.render_path.depth() <= frame.render_path.depth() {
                frame = render;
            }
        }
        if self.startup_profile_frames < 4 {
            eprintln!(
                "target_frame local anchor_path={:?} render_path={:?} logical_path={:?} kind={:?}",
                self.camera.position.anchor.as_slice(),
                frame.render_path.as_slice(),
                frame.logical_path.as_slice(),
                frame.kind,
            );
        }
        if self.startup_profile_frames < 4 {
            let cam_local = match frame.kind {
                ActiveFrameKind::Cartesian | ActiveFrameKind::WrappedPlane { .. } => {
                    self.camera.position.in_frame(&frame.render_path)
                }
                ActiveFrameKind::SphereSubFrame(_) => {
                    self.camera.position.in_frame(&frame.render_path)
                }
            };
            eprintln!(
                "target_frame stable render_path={:?} logical_path={:?} kind={:?} cam_local={:?}",
                frame.render_path.as_slice(),
                frame.logical_path.as_slice(),
                frame.kind,
                cam_local,
            );
        }
        frame
    }

    pub fn apply_zoom(&mut self) {
        self.ui.zoom_level = self.zoom_level();
        let frame = self.target_render_frame();
        self.active_frame = frame;
        let vd = self.visual_depth();
        let cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&cam_gpu);
        }
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}, anchor_depth: {}, frame_depth: {}",
            self.zoom_level(), self.tree_depth as i32,
            self.edit_depth(), vd, self.anchor_depth(), frame.logical_path.depth(),
        );
    }
}
