//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified frame-aware raycast →
//! `break_block` / `place_block` pipeline. Cartesian and
//! cubed-sphere layers now resolve through the same active-frame
//! contract; sphere layers use a bounded face window instead of a
//! separate coarser edit path.

use crate::game_state::HotbarItem;
use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::edit;
use crate::world::gpu;

use super::{
    App, ActiveFrame, ActiveFrameKind, LodUploadKey, RENDER_FRAME_CONTEXT, RENDER_FRAME_K,
    RENDER_FRAME_MAX_DEPTH,
};

const MAX_LOCAL_VISUAL_DEPTH: u32 = 8;
const MAX_BODY_VISUAL_DEPTH: u32 = 3;
const MAX_SPHERE_VISUAL_DEPTH: u32 = 3;
const MAX_FOCUSED_FRAME_CAMERA_EXTENT: f32 = 8.0;
const RENDER_FOV: f32 = 1.2;
const SPHERE_VISUAL_MIN_PIXELS: f32 = 48.0;
const MIN_SPHERE_VISUAL_DEPTH: u32 = 2;

impl App {
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

    pub(super) fn edit_depth(&self) -> u32 {
        self.anchor_depth().saturating_sub(1).max(1)
    }

    /// Sphere face-subtree edit depth.
    ///
    /// The old port kept sphere edits 3 levels coarser than
    /// Cartesian edits (`anchor_depth - 4` vs `anchor_depth - 1`).
    /// That created the exact cross-layer asymmetry the refactor is
    /// supposed to eliminate: at the same zoom, sphere placement
    /// targeted much larger cells, so placed blocks ballooned as the
    /// player went deeper.
    ///
    /// Use the same edit depth for both paths. The body/face wrappers
    /// are structural node kinds, not a reason to coarsen the user's
    /// interaction scale.
    pub(super) fn cs_edit_depth(&self) -> u32 {
        self.edit_depth()
    }

    pub(super) fn visual_depth(&self) -> u32 {
        let local_target = self.edit_depth()
            .saturating_sub(self.active_frame.render_path.depth() as u32)
            .max(1);
        match self.active_frame.kind {
            ActiveFrameKind::Body { inner_r, outer_r } => {
                self.visual_depth_cap_for_sphere(inner_r, outer_r, 1.0)
                    .min(MAX_BODY_VISUAL_DEPTH)
            }
            ActiveFrameKind::Sphere(sphere) => {
                self.visual_depth_cap_for_sphere_frame(&sphere)
                    .max(MIN_SPHERE_VISUAL_DEPTH)
                    .min(MAX_SPHERE_VISUAL_DEPTH)
                    .min(local_target.min(MAX_LOCAL_VISUAL_DEPTH))
            }
            ActiveFrameKind::Cartesian => local_target
                .min(MAX_LOCAL_VISUAL_DEPTH)
                .min(crate::world::tree::MAX_DEPTH as u32),
        }
    }

    fn visual_depth_cap_for_sphere(&self, inner_r: f32, outer_r: f32, face_size: f32) -> u32 {
        let screen_height = self
            .window
            .as_ref()
            .map(|w| w.inner_size().height.max(1) as f32)
            .unwrap_or(1080.0);
        let pixels_per_unit = screen_height / (2.0 * (RENDER_FOV * 0.5).tan());
        let cam_world = self.camera.world_pos_f32();
        let body_path = self.active_frame.render_path;
        let (body_origin, body_size) = super::frame::frame_origin_size_world(&body_path);
        let body_center = [
            body_origin[0] + body_size * 0.5,
            body_origin[1] + body_size * 0.5,
            body_origin[2] + body_size * 0.5,
        ];
        let outer_r_world = outer_r * body_size;
        let shell_world = (outer_r - inner_r) * body_size * face_size.max(1.0 / 3.0);
        if shell_world <= 0.0 {
            return 1;
        }
        let ray_dir = crate::world::sdf::normalize(self.camera.forward());
        let oc = [
            cam_world[0] - body_center[0],
            cam_world[1] - body_center[1],
            cam_world[2] - body_center[2],
        ];
        let b = oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2];
        let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - outer_r_world * outer_r_world;
        let disc = b * b - c;
        let surface_dist = if disc > 0.0 {
            let sq = disc.sqrt();
            let t_enter = (-b - sq).max(0.0);
            let t_exit = -b + sq;
            if t_enter > 0.0 { t_enter } else { t_exit.max(0.001) }
        } else {
            (crate::world::sdf::length(oc) - outer_r_world).abs().max(0.001)
        };
        let root_pixels = shell_world / surface_dist * pixels_per_unit;
        let mut depth = 1u32;
        let mut cell_pixels = root_pixels;
        while depth < MAX_LOCAL_VISUAL_DEPTH && cell_pixels >= SPHERE_VISUAL_MIN_PIXELS * 3.0 {
            depth += 1;
            cell_pixels /= 3.0;
        }
        depth.max(1)
    }

    fn visual_depth_cap_for_sphere_frame(&self, sphere: &super::frame::SphereFrame) -> u32 {
        let screen_height = self
            .window
            .as_ref()
            .map(|w| w.inner_size().height.max(1) as f32)
            .unwrap_or(1080.0);
        let pixels_per_unit = screen_height / (2.0 * (RENDER_FOV * 0.5).tan());
        let cam_world = self.camera.world_pos_f32();
        let (body_origin, body_size) = super::frame::frame_origin_size_world(&sphere.body_path);
        let body_center = [
            body_origin[0] + body_size * 0.5,
            body_origin[1] + body_size * 0.5,
            body_origin[2] + body_size * 0.5,
        ];
        let outer_r_world = sphere.outer_r * body_size;
        let shell_world =
            (sphere.outer_r - sphere.inner_r) * body_size * sphere.face_size.max(1.0 / 3.0);
        if shell_world <= 0.0 {
            return 1;
        }
        let ray_dir = crate::world::sdf::normalize(self.camera.forward());
        let oc = [
            cam_world[0] - body_center[0],
            cam_world[1] - body_center[1],
            cam_world[2] - body_center[2],
        ];
        let b = oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2];
        let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - outer_r_world * outer_r_world;
        let disc = b * b - c;
        let surface_dist = if disc > 0.0 {
            let sq = disc.sqrt();
            let t_enter = (-b - sq).max(0.0);
            let t_exit = -b + sq;
            if t_enter > 0.0 { t_enter } else { t_exit.max(0.001) }
        } else {
            (crate::world::sdf::length(oc) - outer_r_world).abs().max(0.001)
        };
        let root_pixels = shell_world / surface_dist * pixels_per_unit;
        let mut depth = 1u32;
        let mut cell_pixels = root_pixels;
        while depth < MAX_LOCAL_VISUAL_DEPTH && cell_pixels >= SPHERE_VISUAL_MIN_PIXELS * 3.0 {
            depth += 1;
            cell_pixels /= 3.0;
        }
        depth.max(1)
    }

    fn sphere_frame_fits_frustum(&self, sphere: &super::frame::SphereFrame) -> bool {
        let Some(window) = &self.window else {
            return false;
        };
        let size = window.inner_size();
        let aspect = if size.height == 0 {
            16.0 / 9.0
        } else {
            size.width.max(1) as f32 / size.height as f32
        };
        let half_fov_tan = (RENDER_FOV * 0.5).tan();
        let cam_world = self.camera.world_pos_f32();
        let (body_origin, body_size) = super::frame::frame_origin_size_world(&sphere.body_path);
        let body_center = [
            body_origin[0] + body_size * 0.5,
            body_origin[1] + body_size * 0.5,
            body_origin[2] + body_size * 0.5,
        ];
        let outer_r_world = sphere.outer_r * body_size;
        let (fwd, right, up) = self.camera.basis();
        let samples = [
            (0.0f32, 0.0f32),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
        ];
        let bounds_min = [
            sphere.face_u_min - sphere.face_size * 0.02,
            sphere.face_v_min - sphere.face_size * 0.02,
            sphere.face_r_min - sphere.face_size * 0.02,
        ];
        let bounds_max = [
            sphere.face_u_min + sphere.face_size * 1.02,
            sphere.face_v_min + sphere.face_size * 1.02,
            sphere.face_r_min + sphere.face_size * 1.02,
        ];
        let mut hits_considered = 0u32;

        for (sx, sy) in samples {
            let ndc_x = sx * aspect * half_fov_tan;
            let ndc_y = -sy * half_fov_tan;
            let ray_dir = crate::world::sdf::normalize([
                fwd[0] + right[0] * ndc_x + up[0] * ndc_y,
                fwd[1] + right[1] * ndc_x + up[1] * ndc_y,
                fwd[2] + right[2] * ndc_x + up[2] * ndc_y,
            ]);
            let oc = [
                cam_world[0] - body_center[0],
                cam_world[1] - body_center[1],
                cam_world[2] - body_center[2],
            ];
            let b = oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2];
            let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - outer_r_world * outer_r_world;
            let disc = b * b - c;
            if disc <= 0.0 {
                continue;
            }
            let sq = disc.sqrt();
            let t_enter = (-b - sq).max(0.0);
            let t_exit = -b + sq;
            let t = if t_enter > 0.0 { t_enter } else { t_exit };
            if t <= 0.0 {
                continue;
            }
            let hit_world = [
                cam_world[0] + ray_dir[0] * t,
                cam_world[1] + ray_dir[1] * t,
                cam_world[2] + ray_dir[2] * t,
            ];
            let Some(coord) = crate::world::cubesphere::world_to_coord(body_center, hit_world) else {
                continue;
            };
            hits_considered += 1;
            if coord.face != sphere.face {
                if self.startup_profile_frames < 8 {
                    eprintln!(
                        "sphere_fit reject reason=face sample=({sx:.1},{sy:.1}) sample_face={:?} frame_face={:?} render_path={:?}",
                        coord.face,
                        sphere.face,
                        sphere.frame_path.as_slice(),
                    );
                }
                return false;
            }
            let un = (coord.u + 1.0) * 0.5;
            let vn = (coord.v + 1.0) * 0.5;
            let rn = ((coord.r / body_size) - sphere.inner_r) / (sphere.outer_r - sphere.inner_r);
            let in_bounds = (bounds_min[0]..=bounds_max[0]).contains(&un)
                && (bounds_min[1]..=bounds_max[1]).contains(&vn)
                && (bounds_min[2]..=bounds_max[2]).contains(&rn);
            if !in_bounds {
                if self.startup_profile_frames < 8 {
                    eprintln!(
                        "sphere_fit reject reason=bounds sample=({sx:.1},{sy:.1}) un={:.6} vn={:.6} rn={:.6} bounds_u=[{:.6},{:.6}] bounds_v=[{:.6},{:.6}] bounds_r=[{:.6},{:.6}] render_path={:?}",
                        un, vn, rn,
                        bounds_min[0], bounds_max[0],
                        bounds_min[1], bounds_max[1],
                        bounds_min[2], bounds_max[2],
                        sphere.frame_path.as_slice(),
                    );
                }
                return false;
            }
        }

        if self.startup_profile_frames < 8 {
            eprintln!(
                "sphere_fit accept hits_considered={} render_path={:?} face_depth={} face_size={:.8}",
                hits_considered,
                sphere.frame_path.as_slice(),
                sphere.face_depth,
                sphere.face_size,
            );
        }
        hits_considered > 0
    }

    fn camera_fits_frame(&self, frame: &ActiveFrame) -> bool {
        let cam_local = super::point_world_to_frame(frame, self.camera.world_pos_f32());
        cam_local.iter().all(|v| v.is_finite())
            && cam_local.iter().all(|&v| {
                (-MAX_FOCUSED_FRAME_CAMERA_EXTENT
                    ..=WORLD_SIZE + MAX_FOCUSED_FRAME_CAMERA_EXTENT)
                    .contains(&v)
            })
    }

    fn camera_local_sphere_focus_path(&self, desired_depth: u8) -> Option<Path> {
        let body_path = self.planet_path;
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
                        slot,
                        body_path.as_slice(),
                        other
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
        let (body_origin, body_size) = super::frame::frame_origin_size_world(&body_path);
        let body_center = [
            body_origin[0] + body_size * 0.5,
            body_origin[1] + body_size * 0.5,
            body_origin[2] + body_size * 0.5,
        ];
        let cam_world = self.camera.world_pos_f32();
        let radial = crate::world::sdf::sub(cam_world, body_center);
        let radial_len = crate::world::sdf::length(radial);
        if radial_len <= 1e-6 {
            eprintln!("sphere_focus: radial_len too small cam_world={:?} body_center={:?}", cam_world, body_center);
            return None;
        }
        let radial_dir = crate::world::sdf::scale(radial, 1.0 / radial_len);
        let surface_world = crate::world::sdf::add(
            body_center,
            crate::world::sdf::scale(radial_dir, outer_r * body_size),
        );
        let Some(coord) = crate::world::cubesphere::world_to_coord(body_center, surface_world) else {
            eprintln!(
                "sphere_focus: world_to_coord failed body_center={:?} surface_world={:?}",
                body_center,
                surface_world
            );
            return None;
        };
        let mut path = body_path;
        path.push(crate::world::cubesphere::FACE_SLOTS[coord.face as usize] as u8);

        let eps = 1e-5f32;
        let mut un = ((coord.u + 1.0) * 0.5).clamp(eps, 1.0 - eps);
        let mut vn = ((coord.v + 1.0) * 0.5).clamp(eps, 1.0 - eps);
        let shell_world = (outer_r - inner_r) * body_size;
        if shell_world <= 0.0 {
            eprintln!("sphere_focus: non-positive shell_world={shell_world}");
            return None;
        }
        let mut rn = ((coord.r - inner_r * body_size) / shell_world).clamp(1.0 - eps, 1.0 - eps);

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
        eprintln!(
            "sphere_focus: path={:?} desired_depth={} body_path={:?} face={:?}",
            path.as_slice(),
            desired_depth,
            body_path.as_slice(),
            coord.face,
        );
        Some(path)
    }

    fn target_render_frame(&self) -> ActiveFrame {
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        let mut frame = self
            .camera_local_sphere_focus_path(desired_depth)
            .map(|path| {
                let logical = super::frame::compute_render_frame(
                    &self.world.library,
                    self.world.root,
                    &path,
                    path.depth(),
                );
                match logical.kind {
                    ActiveFrameKind::Sphere(sphere) => ActiveFrame {
                        render_path: sphere.body_path,
                        logical_path: logical.logical_path,
                        node_id: sphere.body_node_id,
                        kind: ActiveFrameKind::Body {
                            inner_r: sphere.inner_r,
                            outer_r: sphere.outer_r,
                        },
                    },
                    _ => super::frame::with_render_margin(
                        &self.world.library,
                        self.world.root,
                        &path,
                        RENDER_FRAME_CONTEXT,
                    ),
                }
            })
            .unwrap_or_else(|| self.render_frame());
        while !self.camera_fits_frame(&frame) && frame.render_path.depth() > 0 {
            let logical_path = frame.logical_path;
            let mut shallower = frame.render_path;
            shallower.truncate(frame.render_path.depth().saturating_sub(1));
            let render = super::frame::compute_render_frame(
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
                "target_frame stable render_path={:?} logical_path={:?} kind={:?} cam_local={:?}",
                frame.render_path.as_slice(),
                frame.logical_path.as_slice(),
                frame.kind,
                super::point_world_to_frame(&frame, self.camera.world_pos_f32()),
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

    /// Cast a ray from the camera into the world using the same
    /// frame-aware machinery as the renderer: the cpu raycast
    /// runs in frame-local coordinates and pops upward via the
    /// camera's anchor when it exits the frame's bubble. This is
    /// what makes deep-zoom block placement land in the cell
    /// that's actually under the crosshair, instead of being
    /// pinned to the f32-precision wall of world XYZ.
    pub(super) fn frame_aware_raycast(&self) -> Option<edit::HitInfo> {
        let cam_world = self.camera.world_pos_f32();
        let hit = match self.active_frame.kind {
            ActiveFrameKind::Sphere(sphere) => {
                let cam_body = super::frame::point_world_to_body_frame(&sphere, cam_world);
                edit::cpu_raycast_in_sphere_frame(
                    &self.world.library,
                    self.world.root,
                    self.active_frame.render_path.as_slice(),
                    cam_body,
                    cam_body,
                    crate::world::sdf::normalize(self.camera.forward()),
                    self.cs_edit_depth(),
                    sphere.face as u32,
                    sphere.face_u_min,
                    sphere.face_v_min,
                    sphere.face_r_min,
                    sphere.face_size,
                    sphere.inner_r,
                    sphere.outer_r,
                    sphere.face_depth,
                )
            }
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                let cam_local = self.camera.position.in_frame(&self.active_frame.render_path);
                let ray_dir =
                    super::world_dir_to_frame(&self.active_frame, cam_world, self.camera.forward());
                edit::cpu_raycast_in_frame(
                    &self.world.library,
                    self.world.root,
                    self.active_frame.render_path.as_slice(),
                    cam_local,
                    ray_dir,
                    self.edit_depth(),
                    self.cs_edit_depth(),
                )
            }
        };
        if self.startup_profile_frames < 16 {
            eprintln!(
                "frame_raycast frame={} kind={:?} render_path={:?} logical_path={:?} cam_world={:?} hit={}",
                self.startup_profile_frames,
                self.active_frame.kind,
                self.active_frame.render_path.as_slice(),
                self.active_frame.logical_path.as_slice(),
                cam_world,
                hit.is_some(),
            );
            if let Some(ref h) = hit {
                eprintln!(
                    "frame_raycast_hit path_len={} face={} t={} place_path_len={:?}",
                    h.path.len(),
                    h.face,
                    h.t,
                    h.place_path.as_ref().map(|p| p.len()),
                );
            }
        }
        hit
    }

    pub(super) fn do_break(&mut self) {
        let hit = self.frame_aware_raycast();
        let Some(hit) = hit else {
            eprintln!("do_break: no hit");
            return;
        };
        eprintln!(
            "do_break: hit path_len={} face={} place_path_len={:?}",
            hit.path.len(),
            hit.face,
            hit.place_path.as_ref().map(|p| p.len()),
        );

        if self.save_mode {
            use crate::world::tree::Child;
            let mut saved_id = None;
            if let Some(&(parent_id, slot)) = hit.path.last() {
                if let Some(node) = self.world.library.get(parent_id) {
                    match node.children[slot] {
                        Child::Node(child_id) => saved_id = Some(child_id),
                        Child::Block(_) | Child::Empty => saved_id = Some(parent_id),
                    }
                }
            }
            if let Some(node_id) = saved_id {
                self.world.library.ref_inc(node_id);
                let idx = self.saved_meshes.save(node_id);
                self.ui.slots[self.ui.active_slot] = HotbarItem::Mesh(idx);
                log::info!("Saved mesh #{idx} (node {node_id})");
            }
            self.save_mode = false;
            return;
        }

        let changed = edit::break_block(&mut self.world, &hit);
        eprintln!("do_break: changed={changed}");
        if changed {
            self.upload_tree();
        }
    }

    pub(super) fn do_place(&mut self) {
        let hit = self.frame_aware_raycast();
        let Some(hit) = hit else {
            eprintln!("do_place: no hit");
            return;
        };
        eprintln!(
            "do_place: hit path_len={} face={} place_path_len={:?} active_slot={}",
            hit.path.len(),
            hit.face,
            hit.place_path.as_ref().map(|p| p.len()),
            self.ui.active_slot,
        );

        match &self.ui.slots[self.ui.active_slot] {
            HotbarItem::Block(block_type) => {
                let changed = edit::place_block(&mut self.world, &hit, *block_type);
                eprintln!("do_place: block_type={} changed={changed}", block_type);
                if changed {
                    self.upload_tree();
                }
            }
            HotbarItem::Mesh(idx) => {
                let Some(saved) = self.saved_meshes.items.get(*idx) else { return };
                let node_id = saved.node_id;
                let changed = edit::place_child(
                    &mut self.world, &hit,
                    crate::world::tree::Child::Node(node_id),
                );
                eprintln!("do_place: mesh_idx={} node_id={} changed={changed}", idx, node_id);
                if changed {
                    self.upload_tree();
                }
            }
        }
    }

    pub(super) fn upload_tree(&mut self) {
        self.tree_depth = self.world.tree_depth();
        self.upload_tree_lod();
    }

    /// Pack the world tree (LOD-aware from the absolute root) and
    /// push it to the GPU, along with the ancestor ribbon that
    /// lets the shader pop from the frame back up to the absolute
    /// root.
    ///
    /// Pack runs in **world** coordinates (camera passed as world
    /// XYZ), so distance-LOD decisions are at world scale and the
    /// buffer is shared by all frame depths. The shader starts
    /// DDA at `frame_root_idx` with the camera in **frame-local**
    /// coordinates, and pops via the ribbon when rays exit the
    /// frame's `[0, 3)³` bubble.
    pub(super) fn upload_tree_lod(&mut self) {
        let intended_frame = self.target_render_frame();
        let cam_world = self.camera.world_pos_f32();
        let upload_key = LodUploadKey::new(self.world.root, cam_world, &intended_frame);
        let mut pack_elapsed = std::time::Duration::ZERO;
        let mut ribbon_elapsed = std::time::Duration::ZERO;
        let reused_gpu_tree = self.last_lod_upload_key == Some(upload_key);

        if !reused_gpu_tree {
            let pack_start = std::time::Instant::now();
            // Preserve the intended frame path through the pack so
            // build_ribbon can walk it. Without this, uniform-empty
            // Cartesian siblings on the camera's path get flattened
            // and the ribbon stops at world.root — defeating frame
            // descent and pinning camera precision regardless of zoom.
            let preserve_paths = [
                intended_frame.render_path.as_slice(),
                intended_frame.logical_path.as_slice(),
            ];
            let (tree_data, node_kinds, _world_root_idx) = gpu::pack_tree_lod_preserving(
                &self.world.library,
                self.world.root,
                cam_world,
                1440.0,
                1.2,
                &preserve_paths,
            );
            pack_elapsed = pack_start.elapsed();

            // build_ribbon may stop short of the intended frame when
            // pack LOD-flattened a sibling on the way down (uniform-
            // empty Cartesian children become tag=0 leaves). The
            // shader can only operate at the depth the buffer
            // actually reached, so we recompute cam_local against the
            // truncated path.
            let ribbon_start = std::time::Instant::now();
            let r = gpu::build_ribbon(&tree_data, intended_frame.render_path.as_slice());
            ribbon_elapsed = ribbon_start.elapsed();
            let effective_path = super::frame::frame_from_slots(&r.reached_slots);
            let effective_render = super::frame::compute_render_frame(
                &self.world.library,
                self.world.root,
                &effective_path,
                effective_path.depth(),
            );
            self.active_frame = ActiveFrame {
                render_path: effective_render.render_path,
                logical_path: intended_frame.logical_path,
                node_id: effective_render.node_id,
                kind: effective_render.kind,
            };
            self.last_lod_upload_key = Some(upload_key);

            if let Some(renderer) = &mut self.renderer {
                renderer.update_tree(&tree_data, &node_kinds, r.frame_root_idx);
                renderer.update_ribbon(&r.ribbon);
            }
        }

        let effective_visual_depth = self.visual_depth();
        let cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(effective_visual_depth);
            renderer.update_camera(&cam_gpu);
            match self.active_frame.kind {
                ActiveFrameKind::Sphere(sphere) => {
                    renderer.set_root_kind_face(
                        sphere.inner_r,
                        sphere.outer_r,
                        sphere.face as u32,
                        sphere.face_depth,
                        [sphere.face_u_min, sphere.face_v_min, sphere.face_r_min, sphere.face_size],
                        super::frame::point_world_to_body_frame(&sphere, cam_world),
                    );
                }
                ActiveFrameKind::Body { inner_r, outer_r } => {
                    renderer.set_root_kind_body(inner_r, outer_r);
                }
                ActiveFrameKind::Cartesian => renderer.set_root_kind_cartesian(),
            }
        }
        if self.startup_profile_frames < 12 {
            eprintln!(
                "startup_perf upload_tree_lod frame={} reused={} pack_ms={:.2} ribbon_ms={:.2} frame_depth={} render_depth={} visual_depth={} kind={:?}",
                self.startup_profile_frames,
                reused_gpu_tree,
                pack_elapsed.as_secs_f64() * 1000.0,
                ribbon_elapsed.as_secs_f64() * 1000.0,
                self.active_frame.logical_path.depth(),
                self.active_frame.render_path.depth(),
                effective_visual_depth,
                self.active_frame.kind,
            );
        }
    }
    pub(super) fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let tree_hit = self.frame_aware_raycast();
        if self.startup_profile_frames < 16 {
            eprintln!(
                "highlight_update frame={} cursor_locked={} hit={}",
                self.startup_profile_frames,
                self.cursor_locked,
                tree_hit.is_some(),
            );
        }
        let aabb_world = tree_hit.as_ref().map(|h| edit::hit_aabb(&self.world.library, h));
        let aabb = aabb_world.map(|(mn, mx)| match self.active_frame.kind {
            ActiveFrameKind::Sphere(sphere) => {
                super::frame::aabb_world_to_body_frame(&sphere, mn, mx)
            }
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                super::aabb_world_to_frame(&self.active_frame, mn, mx)
            }
        });
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
    }
}
