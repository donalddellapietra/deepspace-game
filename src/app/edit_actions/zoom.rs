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
        let cam_local = match frame.kind {
            ActiveFrameKind::Cartesian
            | ActiveFrameKind::WrappedPlane { .. }
            | ActiveFrameKind::TangentBlock => {
                self.camera.position.in_frame(&frame.render_path)
            }
        };
        cam_local.iter().all(|v| v.is_finite())
            && cam_local.iter().all(|&v| {
                (-MAX_FOCUSED_FRAME_CAMERA_EXTENT
                    ..=WORLD_SIZE + MAX_FOCUSED_FRAME_CAMERA_EXTENT)
                    .contains(&v)
            })
    }

    pub(in crate::app) fn frame_projected_pixels(&self, frame: &ActiveFrame) -> f32 {
        let (cam_local, frame_center_local, frame_span) = match frame.kind {
            ActiveFrameKind::Cartesian
            | ActiveFrameKind::WrappedPlane { .. }
            | ActiveFrameKind::TangentBlock => (
                self.camera.position.in_frame(&frame.render_path),
                [1.5, 1.5, 1.5],
                crate::world::anchor::WORLD_SIZE,
            ),
        };
        let to_center = crate::world::sdf::sub(frame_center_local, cam_local);
        let dist = crate::world::sdf::length(to_center).max(0.05);
        let half_fov_recip = 720.0f32 / (2.0f32 * (1.2f32 * 0.5f32).tan());
        frame_span / dist * half_fov_recip
    }

    pub(in crate::app) fn target_render_frame(&self) -> ActiveFrame {
        // Render frame depth is derived from `RENDER_ANCHOR_DEPTH`,
        // not the user's anchor depth — zoom controls interaction
        // layer, not what the camera renders. See `RENDER_ANCHOR_DEPTH`.
        let frame = self.render_frame();
        let mut frame = frame;
        while frame.render_path.depth() > 0
            && (!self.camera_fits_frame(&frame)
                || self.frame_projected_pixels(&frame) < FRAME_FOCUS_MIN_PIXELS)
        {
            let logical_path = frame.logical_path;
            let mut shallower = frame.render_path;
            shallower.truncate(frame.render_path.depth().saturating_sub(1));
            let render = frame::compute_render_frame(
                &self.world.library,
                self.world.root,
                &shallower,
                shallower.depth(),
            );
            frame = ActiveFrame {
                render_path: render.render_path,
                logical_path,
                node_id: render.node_id,
                kind: render.kind,
            };
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
                ActiveFrameKind::Cartesian
                | ActiveFrameKind::WrappedPlane { .. }
                | ActiveFrameKind::TangentBlock => {
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
