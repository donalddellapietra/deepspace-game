//! Zoom: depth accessors + camera-aware frame selection + apply.

use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::cubesphere::FACE_SLOTS;
use crate::world::cubesphere_local;

use crate::app::frame;
use crate::app::{ActiveFrame, ActiveFrameKind, App, RENDER_FRAME_CONTEXT, RENDER_FRAME_K, RENDER_FRAME_MAX_DEPTH};
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

    /// Sphere face-subtree edit depth.
    ///
    /// Use the same edit depth for both paths. The body/face wrappers
    /// are structural node kinds, not a reason to coarsen the user's
    /// interaction scale — the older cross-layer asymmetry
    /// (`anchor_depth - 4` vs `-1`) made placed blocks balloon as the
    /// player went deeper.
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
            ActiveFrameKind::Sphere(sphere) => {
                let cam_body = self.camera.position.in_frame(&sphere.body_path);
                if let Some(face_point) = cubesphere_local::body_point_to_face_space(
                    cam_body,
                    sphere.inner_r,
                    sphere.outer_r,
                    WORLD_SIZE,
                ) {
                    let scale = WORLD_SIZE / sphere.face_size;
                    [
                        (face_point.un - sphere.face_u_min) * scale,
                        (face_point.vn - sphere.face_v_min) * scale,
                        (face_point.rn - sphere.face_r_min) * scale,
                    ]
                } else {
                    [f32::NAN; 3]
                }
            }
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
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

    fn camera_local_sphere_focus_path(&self, desired_depth: u8) -> Option<Path> {
        let body_path = self.planet_path?;
        let mut node_id = self.world.root;
        for &slot in body_path.as_slice() {
            let Some(node) = self.world.library.get(node_id) else {
                eprintln!("sphere_focus: missing node for path {:?} at node_id={node_id}", body_path.as_slice());
                return None;
            };
            match node.children[slot as usize] {
                crate::world::tree::Child::Node(child_id) => node_id = child_id,
                other => {
                    eprintln!(
                        "sphere_focus: non-node child at slot={} for body_path={:?}: {:?}",
                        slot, body_path.as_slice(), other
                    );
                    return None;
                }
            }
        }
        let Some(body) = self.world.library.get(node_id) else {
            eprintln!("sphere_focus: missing body node_id={node_id}");
            return None;
        };
        let crate::world::tree::NodeKind::CubedSphereBody { inner_r, outer_r } = body.kind else {
            eprintln!("sphere_focus: path {:?} resolved to non-body kind {:?}", body_path.as_slice(), body.kind);
            return None;
        };
        let cam_body = self.camera.position.in_frame(&body_path);
        let ray_dir = crate::world::sdf::normalize(self.camera.forward());
        let Some(t) = cubesphere_local::ray_outer_sphere_hit(cam_body, ray_dir, outer_r, WORLD_SIZE) else {
            eprintln!(
                "sphere_focus: miss cam_body={:?} ray_dir={:?} body_path={:?}",
                cam_body, ray_dir, body_path.as_slice(),
            );
            return None;
        };
        let hit_body = crate::world::sdf::add(cam_body, crate::world::sdf::scale(ray_dir, t));
        let Some(face_point) = cubesphere_local::body_point_to_face_space(
            hit_body, inner_r, outer_r, WORLD_SIZE,
        ) else {
            eprintln!("sphere_focus: degenerate hit_body={:?}", hit_body);
            return None;
        };
        let face = face_point.face;
        let mut path = body_path;
        path.push(FACE_SLOTS[face as usize] as u8);

        let eps = 1e-5f32;
        let mut un = face_point.un.clamp(eps, 1.0 - eps);
        let mut vn = face_point.vn.clamp(eps, 1.0 - eps);
        let mut rn = face_point.rn.clamp(eps, 1.0 - eps);

        let target_depth = desired_depth.max(body_path.depth() + 1);
        let mut u_min = 0.0f32;
        let mut v_min = 0.0f32;
        let mut r_min = 0.0f32;
        let mut size = 1.0f32;
        while path.depth() < target_depth {
            let child_size = size / 3.0;
            let su = (((un - u_min) / child_size).floor() as i32).clamp(0, 2) as usize;
            let sv = (((vn - v_min) / child_size).floor() as i32).clamp(0, 2) as usize;
            let sr = (((rn - r_min) / child_size).floor() as i32).clamp(0, 2) as usize;
            path.push(crate::world::tree::slot_index(su, sv, sr) as u8);
            u_min += su as f32 * child_size;
            v_min += sv as f32 * child_size;
            r_min += sr as f32 * child_size;
            size = child_size;
            let inner_eps = eps.min(size * 0.25);
            un = un.clamp(u_min + inner_eps, u_min + size - inner_eps);
            vn = vn.clamp(v_min + inner_eps, v_min + size - inner_eps);
            rn = rn.clamp(r_min + inner_eps, r_min + size - inner_eps);
        }
        if self.startup_profile_frames < 16 {
            eprintln!(
                "sphere_focus: path={:?} desired_depth={} body_path={:?} face={:?}",
                path.as_slice(), desired_depth, body_path.as_slice(), face,
            );
        }
        Some(path)
    }

    pub(in crate::app) fn frame_projected_pixels(&self, frame: &ActiveFrame) -> f32 {
        let (cam_local, frame_center_local, frame_span) = match frame.kind {
            ActiveFrameKind::Sphere(sphere) => (
                self.camera.position.in_frame(&sphere.body_path),
                frame::frame_point_to_body([1.5, 1.5, 1.5], &sphere),
                (crate::world::anchor::WORLD_SIZE * sphere.face_size).max(1e-6),
            ),
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => (
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
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
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
            let cam_local = match frame.kind {
                ActiveFrameKind::Sphere(sphere) => self.camera.position.in_frame(&sphere.body_path),
                ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
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
