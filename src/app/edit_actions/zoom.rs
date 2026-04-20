//! Zoom: depth accessors + camera-aware frame selection + apply.

use crate::world::anchor::{Path, WORLD_SIZE};

use crate::app::frame;
use crate::app::{ActiveFrame, App, RENDER_FRAME_CONTEXT, RENDER_FRAME_K, RENDER_FRAME_MAX_DEPTH};
use super::{
    FRAME_FOCUS_MIN_PIXELS, FRAME_VISUAL_MIN_PIXELS, MAX_FOCUSED_FRAME_CAMERA_EXTENT,
    MAX_LOCAL_VISUAL_DEPTH,
};

impl App {
    /// Depth at which break/place edits land — the user's current
    /// layer. A break at layer N removes a layer-N-sized cell,
    /// regardless of whether the raycast found a smaller leaf cell
    /// inside.
    pub(in crate::app) fn edit_depth(&self) -> u32 {
        if let Some(depth) = self.forced_edit_depth {
            return depth.max(1).min(crate::world::tree::MAX_DEPTH as u32);
        }
        self.anchor_depth().max(1)
    }

    /// Maximum depth the CPU raycast walker descends to. Decoupled
    /// from `edit_depth`: the walker needs to reach the actual leaf
    /// cell under the cursor so that (a) the cursor highlights what
    /// the user is visually pointing at, and (b) empty coarse
    /// representatives don't short-circuit the walk and cause "no
    /// hit" misfires near mixed-content boundaries.
    ///
    /// The caller truncates `hit.path` to `edit_depth()` slots before
    /// applying an edit, so this deeper walk never lets break/place
    /// land on a cell below the user's current layer.
    pub(in crate::app) fn raycast_max_depth(&self) -> u32 {
        (self.tree_depth as u32).max(1)
    }

    /// Sphere face-subtree edit depth — mirrors `edit_depth` so placed
    /// blocks are always at the user's current layer.
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
        // Unified: `in_frame(render_path)` works for every frame kind
        // because slot math is numerically shared regardless of
        // NodeKind. Any face-axis rotation (for Sphere face_depth>=1)
        // doesn't affect whether the camera is inside the frame bounds
        // — we check against the unrotated position.
        let cam_local = self.camera.position.in_frame(&frame.render_path);
        cam_local.iter().all(|v| v.is_finite())
            && cam_local.iter().all(|&v| {
                (-MAX_FOCUSED_FRAME_CAMERA_EXTENT
                    ..=WORLD_SIZE + MAX_FOCUSED_FRAME_CAMERA_EXTENT)
                    .contains(&v)
            })
    }

    /// When the camera is inside a planet body, build a path that
    /// descends into the appropriate face subtree toward the camera,
    /// to `desired_depth` total slots. The sphere SDF in the shader
    /// handles the silhouette regardless; this function's job is just
    /// to pick a render frame *inside* the face subtree content where
    /// the packed tree is rich, instead of letting the camera's raw
    /// anchor point into a uniform interior region.
    fn camera_local_sphere_focus_path(&self, desired_depth: u8) -> Option<Path> {
        let body_path = self.planet_path?;
        let cam_body = self.camera.position.in_frame(&body_path);
        // Pick the face whose outward normal has the largest component
        // of the camera's body-frame offset from center — that's the
        // face the camera is looking at / standing on.
        let d = [
            cam_body[0] - WORLD_SIZE * 0.5,
            cam_body[1] - WORLD_SIZE * 0.5,
            cam_body[2] - WORLD_SIZE * 0.5,
        ];
        let ax = d[0].abs();
        let ay = d[1].abs();
        let az = d[2].abs();
        let face_idx = if ax >= ay && ax >= az {
            if d[0] > 0.0 { 0 } else { 1 } // PosX / NegX
        } else if ay >= az {
            if d[1] > 0.0 { 2 } else { 3 } // PosY / NegY
        } else {
            if d[2] > 0.0 { 4 } else { 5 } // PosZ / NegZ
        };
        let mut path = body_path;
        path.push(crate::world::cubesphere::FACE_SLOTS[face_idx] as u8);
        // Within the face subtree the slot semantics are `(u, v, r)`
        // numerically identical to `(x, y, z)`. Walk toward the cell
        // under the camera's body-frame position — projected into
        // face subtree coords is simply `cam_body` minus the face
        // subtree's body-cell origin, divided by 1.0.
        //
        // We can't cheaply decompose `cam_body` into face-subtree
        // coords without the removed face-space helpers. Instead,
        // descend toward the center of the face subtree — that's
        // guaranteed to sit on the solid interior where SDF-varying
        // content lives.
        let target_depth = desired_depth.max(body_path.depth() + 1);
        while path.depth() < target_depth {
            path.push(13); // center slot = (1, 1, 1)
        }
        Some(path)
    }

    pub(in crate::app) fn frame_projected_pixels(&self, frame: &ActiveFrame) -> f32 {
        // Every frame is a bounded `[0, WORLD_SIZE)³` box in its own
        // local coords. Projected pixel span is `WORLD_SIZE / dist *
        // focal_px`, where dist is the camera's distance from the
        // frame center.
        let cam_local = self.camera.position.in_frame(&frame.render_path);
        let frame_center_local = [1.5, 1.5, 1.5];
        let frame_span = WORLD_SIZE;
        let to_center = crate::world::sdf::sub(frame_center_local, cam_local);
        let dist = crate::world::sdf::length(to_center).max(0.05);
        let half_fov_recip = 720.0f32 / (2.0f32 * (1.2f32 * 0.5f32).tan());
        frame_span / dist * half_fov_recip
    }

    pub(in crate::app) fn target_render_frame(&self) -> ActiveFrame {
        // Render frame depth is derived from `RENDER_ANCHOR_DEPTH`,
        // not the user's anchor depth — zoom controls interaction
        // layer, not what the camera renders. See `RENDER_ANCHOR_DEPTH`.
        let desired_depth = crate::app::RENDER_ANCHOR_DEPTH
            .saturating_sub(RENDER_FRAME_K as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        // For sphere worlds the render path should target the physical
        // face surface where the packed tree retains deep structure —
        // the camera's raw anchor can point into uniform interior
        // regions that the packer collapses, capping ribbon depth.
        // `camera_local_sphere_focus_path` maps the camera's body-
        // frame position into face coords and walks sub-cells down to
        // `desired_depth`, always landing on SDF-varying content.
        //
        // For Cartesian worlds the camera's anchor IS the path we
        // want (content richness is spread throughout the volume).
        let frame = self
            .camera_local_sphere_focus_path(desired_depth)
            .map(|path| {
                frame::with_render_margin(
                    &self.world.library,
                    self.world.root,
                    &path,
                    RENDER_FRAME_CONTEXT,
                )
            })
            .unwrap_or_else(|| self.render_frame());
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
            let cam_local = self.camera.position.in_frame(&frame.render_path);
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
