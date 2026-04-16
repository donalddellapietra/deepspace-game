//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified frame-aware raycast →
//! `break_block` / `place_block` pipeline. Cartesian and
//! cubed-sphere layers now resolve through the same active-frame
//! contract; sphere layers use a bounded face window instead of a
//! separate coarser edit path.

use crate::game_state::HotbarItem;
use crate::world::cubesphere::FACE_SLOTS;
use crate::world::cubesphere_local;
use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::edit;
use crate::world::gpu;

use super::{
    App, ActiveFrame, ActiveFrameKind, HighlightCacheKey, LodUploadKey, RENDER_FRAME_CONTEXT, RENDER_FRAME_K,
    RENDER_FRAME_MAX_DEPTH,
};

const MAX_LOCAL_VISUAL_DEPTH: u32 = 12;
const MAX_FOCUSED_FRAME_CAMERA_EXTENT: f32 = 8.0;
const FRAME_VISUAL_MIN_PIXELS: f32 = 1.0;
const FRAME_FOCUS_MIN_PIXELS: f32 = 192.0;

impl App {
    fn ray_dir_in_frame(&self, _frame_path: &Path) -> [f32; 3] {
        // In Cartesian frames, all levels share the same axes — the
        // direction is identical in every frame.  The DDA only cares
        // about the *direction*, not the magnitude.  The old code
        // scaled by 3^depth which overflows f32 past depth ~20.
        crate::world::sdf::normalize(self.camera.forward())
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

    fn debug_hit_terminal(&self, hit: &edit::HitInfo) -> String {
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

    pub(super) fn edit_depth(&self) -> u32 {
        if let Some(depth) = self.forced_edit_depth {
            return depth.max(1).min(crate::world::tree::MAX_DEPTH as u32);
        }
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
        let cam_body = self.camera.position.in_frame(&body_path);
        let ray_dir = crate::world::sdf::normalize(self.camera.forward());
        let Some(t) = cubesphere_local::ray_outer_sphere_hit(cam_body, ray_dir, outer_r, WORLD_SIZE) else {
            eprintln!(
                "sphere_focus: miss cam_body={:?} ray_dir={:?} body_path={:?}",
                cam_body,
                ray_dir,
                body_path.as_slice(),
            );
            return None;
        };
        let hit_body = crate::world::sdf::add(cam_body, crate::world::sdf::scale(ray_dir, t));
        let Some(face_point) = cubesphere_local::body_point_to_face_space(
            hit_body,
            inner_r,
            outer_r,
            WORLD_SIZE,
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
                path.as_slice(),
                desired_depth,
                body_path.as_slice(),
                face,
            );
        }
        Some(path)
    }

    fn frame_projected_pixels(&self, frame: &ActiveFrame) -> f32 {
        let (cam_local, frame_center_local, frame_span) = match frame.kind {
            ActiveFrameKind::Sphere(sphere) => (
                self.camera.position.in_frame(&sphere.body_path),
                super::frame::frame_point_to_body([1.5, 1.5, 1.5], &sphere),
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

    fn target_render_frame(&self) -> ActiveFrame {
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        let frame = self
            .camera_local_sphere_focus_path(desired_depth)
            .map(|path| {
                super::frame::with_render_margin(
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

    /// Cast a ray from the camera into the world using the same
    /// frame-aware machinery as the renderer: the cpu raycast
    /// runs in frame-local coordinates and pops upward via the
    /// camera's anchor when it exits the frame's bubble. This is
    /// what makes deep-zoom block placement land in the cell
    /// that's actually under the crosshair, instead of being
    /// pinned to the f32-precision wall of world XYZ.
    pub(super) fn frame_aware_raycast(&self) -> Option<edit::HitInfo> {
        let hit = match self.active_frame.kind {
            ActiveFrameKind::Sphere(sphere) => {
                let cam_body = self.camera.position.in_frame(&sphere.body_path);
                let ray_dir_local = self.ray_dir_in_frame(&sphere.body_path);
                edit::cpu_raycast_in_sphere_frame(
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
                )
            }
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                let frame_path = &self.active_frame.render_path;
                let mut hit = None;
                let cam_local = self.camera.position.in_frame(frame_path);
                let ray_dir = self.ray_dir_in_frame(frame_path);
                let min_depth = frame_path.depth() as u32 + 1;
                let mut depth = self.edit_depth();
                while depth >= min_depth {
                    hit = edit::cpu_raycast_in_frame(
                        &self.world.library,
                        self.world.root,
                        frame_path.as_slice(),
                        cam_local,
                        ray_dir,
                        depth,
                        self.cs_edit_depth(),
                    );
                    if hit.is_some() || depth == min_depth {
                        break;
                    }
                    depth -= 1;
                }
                if hit.is_none() && self.startup_profile_frames < 16 {
                    eprintln!(
                        "frame_raycast_cartesian_fallback_failed edit_depth={} min_depth={} render_path={:?}",
                        self.edit_depth(),
                        min_depth,
                        frame_path.as_slice(),
                    );
                }
                hit
            }
        };
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
                        edit::hit_aabb_body_local(&self.world.library, h)
                    }
                    ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                        edit::hit_aabb_in_frame_local(h, &self.active_frame.render_path)
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
        self.highlight_epoch = self.highlight_epoch.wrapping_add(1);
        self.cached_highlight = None;
        self.upload_tree_lod();
    }

    /// Pack the world tree (LOD-aware from the root) and push it
    /// to the GPU, along with the ancestor ribbon that lets the
    /// shader pop from the active frame back up the tree.
    ///
    /// The packer now uses the camera's path-anchored `WorldPos`
    /// directly. Every LOD decision is made in the current node's
    /// local `[0, 3)³` frame via `WorldPos::in_frame`, so upload,
    /// render, and edit all agree on the same locality model.
    pub(super) fn upload_tree_lod(&mut self) {
        let intended_frame = self.target_render_frame();
        let effective_visual_depth = self.visual_depth();
        let upload_key = LodUploadKey::new(
            self.world.root,
            &self.camera.position,
            &intended_frame,
            effective_visual_depth.min(u8::MAX as u32) as u8,
        );
        let mut pack_elapsed = std::time::Duration::ZERO;
        let mut ribbon_elapsed = std::time::Duration::ZERO;
        let reused_gpu_tree = self.last_lod_upload_key == Some(upload_key);

        if !reused_gpu_tree {
            let pack_start = std::time::Instant::now();
            let (tree_data, node_kinds, _world_root_idx) = {
                let mut preserve_path_storage = vec![intended_frame.render_path];
                if intended_frame.logical_path != intended_frame.render_path {
                    preserve_path_storage.push(intended_frame.logical_path);
                }
                let preserve_paths: Vec<&[u8]> =
                    preserve_path_storage.iter().map(Path::as_slice).collect();
                gpu::pack_tree_lod_preserving(
                    &self.world.library,
                    self.world.root,
                    &self.camera.position,
                    1440.0,
                    1.2,
                    &preserve_paths,
                )
            };
            pack_elapsed = pack_start.elapsed();
            let packed_node_count = tree_data.len() / 27;

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
            if self.render_harness {
                eprintln!(
                    "render_harness_pack kind={:?} cartesian_lod_enabled={} packed_nodes={} library_nodes={}",
                    intended_frame.kind,
                    true,
                    packed_node_count,
                    self.world.library.len(),
                );
            }
        }

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
                        self.camera.position.in_frame(&sphere.body_path),
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
        if self.disable_highlight {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        if !self.cursor_locked {
            self.last_highlight_raycast_ms = 0.0;
            self.last_highlight_set_ms = 0.0;
            self.cached_highlight = None;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let cache_key = HighlightCacheKey::new(self);
        if let Some((cached_key, cached_aabb)) = self.cached_highlight {
            if cached_key == cache_key {
                self.last_highlight_raycast_ms = 0.0;
                let set_start = std::time::Instant::now();
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_highlight(cached_aabb);
                }
                self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
                return;
            }
        }
        let raycast_start = std::time::Instant::now();
        let tree_hit = self.frame_aware_raycast();
        self.last_highlight_raycast_ms = raycast_start.elapsed().as_secs_f64() * 1000.0;
        if self.startup_profile_frames < 16 {
            eprintln!(
                "highlight_update frame={} cursor_locked={} hit={}",
                self.startup_profile_frames,
                self.cursor_locked,
                tree_hit.is_some(),
            );
        }
        let aabb = tree_hit.as_ref().map(|hit| match self.active_frame.kind {
            ActiveFrameKind::Sphere(_) => edit::hit_aabb_body_local(&self.world.library, hit),
            ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
                edit::hit_aabb_in_frame_local(hit, &self.active_frame.render_path)
            }
        });
        let set_start = std::time::Instant::now();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
        self.last_highlight_set_ms = set_start.elapsed().as_secs_f64() * 1000.0;
        self.cached_highlight = Some((cache_key, aabb));
    }
}
