//! GPU upload: pack the world tree LOD-aware and push the packed
//! buffers (tree, node_kinds, node_offsets, parent_info) plus the
//! per-frame uniforms to the renderer.
//!
//! Ancestor pop-up no longer needs a side ribbon: the shader walks
//! `parent_info[current_idx]` on each pop. The CPU still needs the
//! frame's BFS index to seed `uniforms.root_index`, which is what
//! `walk_to_frame_root` returns.

use crate::world::anchor::Path;
use crate::world::gpu;

use crate::app::frame;
use crate::app::{ActiveFrame, ActiveFrameKind, App, LodUploadKey};
use super::SHELL_PRESERVE_DEPTH;

impl App {
    pub(in crate::app) fn upload_tree(&mut self) {
        self.tree_depth = self.world.tree_depth();
        self.highlight_epoch = self.highlight_epoch.wrapping_add(1);
        self.cached_highlight = None;
        self.upload_tree_lod();
    }

    /// Pack the world tree (LOD-aware from the root) and push it
    /// to the GPU. The packer emits a `parent_info[]` array
    /// alongside the tree; the shader uses it to pop upward from
    /// the active frame without a side ribbon.
    ///
    /// The packer uses the camera's path-anchored `WorldPos`
    /// directly. Every LOD decision is made in the current node's
    /// local `[0, 3)³` frame via `WorldPos::in_frame`, so upload,
    /// render, and edit all agree on the same locality model.
    pub(in crate::app) fn upload_tree_lod(&mut self) {
        let intended_frame = self.target_render_frame();
        let effective_visual_depth = self.visual_depth();
        let upload_key = LodUploadKey::new(
            self.world.root,
            &self.camera.position,
            &intended_frame,
            effective_visual_depth.min(u8::MAX as u32) as u8,
        );
        let mut pack_elapsed = std::time::Duration::ZERO;
        let reused_gpu_tree = self.last_lod_upload_key == Some(upload_key);
        // Workload counters — recorded every frame, even when the
        // LOD key hit, so the harness sees what the frame was asked
        // to do (not just how long each phase took).
        self.last_effective_visual_depth = effective_visual_depth;
        self.last_reused_gpu_tree = reused_gpu_tree;

        if !reused_gpu_tree {
            let pack_start = std::time::Instant::now();
            let (tree_packed, node_kinds, node_offsets, parent_info, _world_root_idx) = {
                let mut preserve_path_storage = vec![intended_frame.render_path];
                if intended_frame.logical_path != intended_frame.render_path {
                    preserve_path_storage.push(intended_frame.logical_path);
                }
                // If a recent edit landed deeper than the render frame,
                // preserve the exact slot path so the packer keeps
                // fine-grained detail along the edit visible.
                if let Some(ref edit_path) = self.last_edit_slots {
                    preserve_path_storage.push(*edit_path);
                }
                let preserve_paths: Vec<&[u8]> =
                    preserve_path_storage.iter().map(Path::as_slice).collect();
                // Preserve regions: keep enough detail at each
                // ribbon-ancestor shell so the shader's LOD-bounded
                // DDA can descend into the packed tree. Each shell
                // gets SHELL_PRESERVE_DEPTH levels of un-collapsed
                // nodes; the LOD check in the shader decides how
                // deep to actually go at render time.
                let mut preserve_regions = Vec::new();
                if matches!(intended_frame.kind, ActiveFrameKind::Cartesian)
                    && !intended_frame.render_path.is_root()
                {
                    let rd = intended_frame.render_path.depth();
                    preserve_regions.push((
                        intended_frame.render_path,
                        SHELL_PRESERVE_DEPTH,
                    ));
                    for d in (1..rd).rev() {
                        let mut ancestor = intended_frame.render_path;
                        ancestor.truncate(d);
                        preserve_regions.push((ancestor, SHELL_PRESERVE_DEPTH));
                    }
                }
                gpu::pack_tree_lod_selective(
                    &self.world.library,
                    self.world.root,
                    &self.camera.position,
                    1440.0,
                    1.2,
                    &preserve_paths,
                    &preserve_regions,
                )
            };
            pack_elapsed = pack_start.elapsed();
            let packed_node_count = node_kinds.len();
            self.last_packed_node_count = packed_node_count as u32;

            // The packer's LOD pass may flatten a sibling on the
            // intended frame path (uniform-empty Cartesian children
            // disappear from the occupancy mask). The shader can
            // only operate at the depth the buffer actually reached,
            // so we walk the packed tree and recompute cam_local
            // against the truncated path.
            let (frame_root_idx, reached_slots) = gpu::walk_to_frame_root(
                &tree_packed,
                &node_offsets,
                intended_frame.render_path.as_slice(),
            );
            let effective_path = frame::frame_from_slots(&reached_slots);
            let effective_render = frame::compute_render_frame(
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
                renderer.update_tree(
                    &tree_packed,
                    &node_kinds,
                    &node_offsets,
                    &parent_info,
                    frame_root_idx,
                );
            }
            if self.render_harness {
                eprintln!(
                    "render_harness_pack kind={:?} cartesian_lod_enabled={} packed_nodes={} tree_u32s={} library_nodes={}",
                    intended_frame.kind,
                    true,
                    packed_node_count,
                    tree_packed.len(),
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
        self.last_pack_ms = pack_elapsed.as_secs_f64() * 1000.0;
        if self.startup_profile_frames < 12 {
            eprintln!(
                "startup_perf upload_tree_lod frame={} reused={} pack_ms={:.2} frame_depth={} render_depth={} visual_depth={} kind={:?}",
                self.startup_profile_frames,
                reused_gpu_tree,
                self.last_pack_ms,
                self.active_frame.logical_path.depth(),
                self.active_frame.render_path.depth(),
                effective_visual_depth,
                self.active_frame.kind,
            );
        }
    }
}
