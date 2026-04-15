//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified frame-aware raycast →
//! `break_block` / `place_block` pipeline. Cartesian and
//! cubed-sphere layers now resolve through the same active-frame
//! contract; sphere layers use a bounded face window instead of a
//! separate coarser edit path.

use crate::game_state::HotbarItem;
use crate::world::edit;
use crate::world::gpu;

use super::{
    App, ActiveFrame, ActiveFrameKind, LodUploadKey, RENDER_FRAME_CONTEXT, RENDER_FRAME_K,
    RENDER_FRAME_MAX_DEPTH,
};

const MAX_LOCAL_VISUAL_DEPTH: u32 = 8;

impl App {
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
        local_target
            .min(MAX_LOCAL_VISUAL_DEPTH)
            .min(crate::world::tree::MAX_DEPTH as u32)
    }

    fn target_render_frame(&self) -> ActiveFrame {
        let desired_depth = (self.anchor_depth().saturating_sub(RENDER_FRAME_K as u32) as u8)
            .min(RENDER_FRAME_MAX_DEPTH);
        let anchor_frame = self.render_frame();
        let cam_world = self.camera.world_pos_f32();
        let cam_local = self.camera.position.in_frame(&anchor_frame.render_path);
        let ray_dir = super::world_dir_to_frame(&anchor_frame, cam_world, self.camera.forward());

        let Some(hit) = edit::cpu_raycast_in_frame(
            &self.world.library,
            self.world.root,
            anchor_frame.render_path.as_slice(),
            cam_local,
            ray_dir,
            self.edit_depth(),
            self.edit_depth(),
        ) else {
            if self.startup_profile_frames < 4 {
                eprintln!(
                    "target_frame miss desired_depth={} anchor_path={:?} anchor_kind={:?}",
                    desired_depth,
                    anchor_frame.render_path.as_slice(),
                    anchor_frame.kind,
                );
            }
            return anchor_frame;
        };

        let mut focus_path = crate::world::anchor::Path::root();
        for &(_, slot) in hit.path.iter().take(desired_depth as usize) {
            focus_path.push(slot as u8);
        }
        if focus_path.depth() == 0 {
            return anchor_frame;
        }

        let focus_frame = super::frame::with_render_margin(
            &self.world.library,
            self.world.root,
            &focus_path,
            RENDER_FRAME_CONTEXT,
        );
        if self.startup_profile_frames < 4 {
            eprintln!(
                "target_frame hit_path_len={} desired_depth={} focus_path={:?} anchor_kind={:?} focus_kind={:?}",
                hit.path.len(),
                desired_depth,
                focus_path.as_slice(),
                anchor_frame.kind,
                focus_frame.kind,
            );
        }
        match focus_frame.kind {
            ActiveFrameKind::Sphere(_) | ActiveFrameKind::Body { .. } => focus_frame,
            ActiveFrameKind::Cartesian => anchor_frame,
        }
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
        let cam_local = self.camera.position.in_frame(&self.active_frame.render_path);
        let ray_dir = super::world_dir_to_frame(&self.active_frame, cam_world, self.camera.forward());
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

    pub(super) fn do_break(&mut self) {
        let hit = self.frame_aware_raycast();
        let Some(hit) = hit else { return };

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
        if changed {
            self.upload_tree();
        }
    }

    pub(super) fn do_place(&mut self) {
        let hit = self.frame_aware_raycast();
        let Some(hit) = hit else { return };

        match &self.ui.slots[self.ui.active_slot] {
            HotbarItem::Block(block_type) => {
                if edit::place_block(&mut self.world, &hit, *block_type) {
                    self.upload_tree();
                }
            }
            HotbarItem::Mesh(idx) => {
                let Some(saved) = self.saved_meshes.items.get(*idx) else { return };
                let node_id = saved.node_id;
                if edit::place_child(
                    &mut self.world, &hit,
                    crate::world::tree::Child::Node(node_id),
                ) {
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
            let (tree_data, node_kinds, _world_root_idx) = gpu::pack_tree_lod_preserving(
                &self.world.library, self.world.root, cam_world, 1440.0, 1.2,
                intended_frame.logical_path.as_slice(),
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
        let aabb_world = tree_hit.as_ref().map(|h| edit::hit_aabb(&self.world.library, h));
        let aabb = aabb_world.map(|(mn, mx)| super::aabb_world_to_frame(&self.active_frame, mn, mx));
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
    }
}
