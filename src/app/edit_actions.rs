//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified `edit::cpu_raycast` →
//! `break_block` / `place_block` pipeline. The planet is part of
//! the same tree, so there's no longer a sphere-specific code
//! path here. (CPU raycast traversal of `CubedSphereBody` cells
//! is a follow-up; for now hits inside the body cell are treated
//! as Cartesian, which means edits on the planet's interior
//! aren't yet supported. Rendering of the planet works.)

use crate::game_state::HotbarItem;
use crate::world::edit;
use crate::world::gpu;

use super::App;

const MAX_LOCAL_VISUAL_DEPTH: u32 = 4;

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
            .saturating_sub(self.active_frame.depth() as u32)
            .max(1);
        local_target
            .min(MAX_LOCAL_VISUAL_DEPTH)
            .min(crate::world::tree::MAX_DEPTH as u32)
    }

    pub fn apply_zoom(&mut self) {
        self.ui.zoom_level = self.zoom_level();
        let vd = self.visual_depth();
        let (frame, _) = self.render_frame();
        self.active_frame = frame;
        let chain = super::frame::FrameKindChain::build(
            &self.world.library, self.world.root, &self.active_frame,
        );
        let cam_world = self.camera.position.to_world_xyz_in(&self.world.library, self.world.root);
        let cam_local = super::frame::position_in_frame(
            &self.world.library,
            self.world.root,
            &self.active_frame,
            &self.camera.position,
        );
        let (fwd_world, right_world, up_world) = self.camera.basis();
        let face_info = super::frame::face_frame_info(&self.active_frame, &chain);
        let (fwd_local, right_local, up_local) = if let Some(info) = face_info {
            let cam_body = super::frame::frame_point_to_body(cam_local, &info);
            (
                super::frame::body_dir_to_frame(cam_body, fwd_world, &info),
                super::frame::body_dir_to_frame(cam_body, right_world, &info),
                super::frame::body_dir_to_frame(cam_body, up_world, &info),
            )
        } else {
            (
                super::frame::world_dir_to_frame(&self.active_frame, &chain, cam_world, fwd_world),
                super::frame::world_dir_to_frame(&self.active_frame, &chain, cam_world, right_world),
                super::frame::world_dir_to_frame(&self.active_frame, &chain, cam_world, up_world),
            )
        };
        let cam_gpu = self.camera.gpu_camera_with_basis(
            cam_local,
            fwd_local,
            right_local,
            up_local,
            1.2,
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&cam_gpu);
        }
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}, anchor_depth: {}, frame_depth: {}",
            self.zoom_level(), self.tree_depth as i32,
            self.edit_depth(), vd, self.anchor_depth(), frame.depth(),
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
        let chain = super::frame::FrameKindChain::build(
            &self.world.library, self.world.root, &self.active_frame,
        );
        let cam_world = self.camera.position.to_world_xyz_in(&self.world.library, self.world.root);
        let cam_local = super::frame::position_in_frame(
            &self.world.library,
            self.world.root,
            &self.active_frame,
            &self.camera.position,
        );
        let ray_dir = if let Some(info) = super::frame::face_frame_info(&self.active_frame, &chain) {
            let cam_body = super::frame::frame_point_to_body(cam_local, &info);
            super::frame::body_dir_to_frame(cam_body, self.camera.forward(), &info)
        } else {
            super::frame::world_dir_to_frame(
                &self.active_frame, &chain, cam_world, self.camera.forward(),
            )
        };
        edit::cpu_raycast_in_frame(
            &self.world.library, self.world.root,
            self.active_frame.as_slice(), cam_local, ray_dir,
            self.edit_depth(), self.cs_edit_depth(),
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
        let (intended_frame, _frame_root_id) = self.render_frame();
        let cam_world = self.camera.position.to_world_xyz_in(&self.world.library, self.world.root);
        // Preserve the intended frame path through the pack so
        // build_ribbon can walk it. Without this, uniform-empty
        // Cartesian siblings on the camera's path get flattened
        // and the ribbon stops at world.root — defeating frame
        // descent and pinning camera precision regardless of zoom.
        let (tree_data, node_kinds, _world_root_idx) = gpu::pack_tree_lod_preserving(
            &self.world.library, self.world.root, cam_world, 1440.0, 1.2,
            intended_frame.as_slice(),
        );
        // build_ribbon may stop short of the intended frame when
        // pack LOD-flattened a sibling on the way down (uniform-
        // empty Cartesian children become tag=0 leaves). The
        // shader can only operate at the depth the buffer
        // actually reached, so we recompute cam_local against the
        // truncated path.
        let r = gpu::build_ribbon(&tree_data, intended_frame.as_slice());
        let effective_frame = super::frame::frame_from_slots(&r.reached_slots);
        self.active_frame = effective_frame;
        let chain = super::frame::FrameKindChain::build(
            &self.world.library, self.world.root, &effective_frame,
        );
        let cam_local = super::frame::position_in_frame(
            &self.world.library,
            self.world.root,
            &effective_frame,
            &self.camera.position,
        );
        let (fwd_world, right_world, up_world) = self.camera.basis();
        let face_info = super::frame::face_frame_info(&effective_frame, &chain);
        let (fwd_local, right_local, up_local) = if let Some(info) = face_info {
            let cam_body = super::frame::frame_point_to_body(cam_local, &info);
            (
                super::frame::body_dir_to_frame(cam_body, fwd_world, &info),
                super::frame::body_dir_to_frame(cam_body, right_world, &info),
                super::frame::body_dir_to_frame(cam_body, up_world, &info),
            )
        } else {
            (
                super::frame::world_dir_to_frame(&effective_frame, &chain, cam_world, fwd_world),
                super::frame::world_dir_to_frame(&effective_frame, &chain, cam_world, right_world),
                super::frame::world_dir_to_frame(&effective_frame, &chain, cam_world, up_world),
            )
        };
        let cam_gpu = self.camera.gpu_camera_with_basis(
            cam_local,
            fwd_local,
            right_local,
            up_local,
            1.2,
        );
        // Frame kind depends on the EFFECTIVE frame, not the
        // intended one.
        let frame_kind = self.world.library
            .get(self.frame_root_id_for(&effective_frame))
            .map(|n| n.kind)
            .unwrap_or(crate::world::tree::NodeKind::Cartesian);
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &node_kinds, r.frame_root_idx);
            renderer.update_ribbon(&r.ribbon);
            renderer.update_camera(&cam_gpu);
            if let Some(info) = face_info {
                let pop_pos = super::frame::frame_point_to_body(cam_local, &info);
                renderer.set_root_kind_face(
                    info.inner_r,
                    info.outer_r,
                    info.face as u32,
                    info.subtree_depth as u32,
                    [info.u_lo, info.v_lo, info.r_lo, info.size],
                    pop_pos,
                );
            } else {
                match frame_kind {
                crate::world::tree::NodeKind::CubedSphereBody { inner_r, outer_r } => {
                    renderer.set_root_kind_body(inner_r, outer_r);
                }
                _ => {
                    renderer.set_root_kind_cartesian();
                }
                }
            }
        }
    }

    /// Walk the world tree from world.root following `path`
    /// returning the NodeId reached. Used by upload_tree_lod to
    /// look up the *effective* frame's NodeKind after build_ribbon
    /// truncated.
    pub(super) fn frame_root_id_for(&self, path: &crate::world::anchor::Path) -> crate::world::tree::NodeId {
        let mut node = self.world.root;
        for k in 0..path.depth() as usize {
            let Some(n) = self.world.library.get(node) else { break };
            let slot = path.slot(k) as usize;
            match n.children[slot] {
                crate::world::tree::Child::Node(child) => { node = child; }
                _ => break,
            }
        }
        node
    }

    pub(super) fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let tree_hit = self.frame_aware_raycast();
        let chain = super::frame::FrameKindChain::build(
            &self.world.library, self.world.root, &self.active_frame,
        );
        let aabb_world = tree_hit.as_ref().map(|h| edit::hit_aabb(&self.world.library, h));
        // Transform AABB from world coords to frame-local coords.
        // Shader expects highlight in the same frame as `camera.pos`.
        let aabb = aabb_world.map(|(mn, mx)| {
            super::aabb_world_to_frame(&self.active_frame, &chain, mn, mx)
        });
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::TestConfig;

    #[test]
    fn perf_smoke_cpu_paths() {
        let mut app = App::with_test_config(TestConfig {
            spawn_depth: Some(12),
            ..Default::default()
        });
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            app.upload_tree_lod();
        }
        let upload_ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;

        let t1 = std::time::Instant::now();
        for _ in 0..50 {
            let _ = app.frame_aware_raycast();
        }
        let raycast_ms = t1.elapsed().as_secs_f64() * 1000.0 / 50.0;

        eprintln!(
            "perf_smoke_cpu_paths avg_ms upload_tree_lod={:.3} frame_aware_raycast={:.3}",
            upload_ms, raycast_ms,
        );
    }
}
