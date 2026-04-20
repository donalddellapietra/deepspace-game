//! GPU upload: keep the cached packed tree in sync with `world.root`
//! and push per-frame uniforms (camera, ribbon) to the renderer.
//!
//! The packed tree is content-addressed: `CachedTree::update_root`
//! emits (or reuses) nodes keyed by `NodeId`. Initial pack and edit
//! path go through the same function — the edit case is fast because
//! every sibling of the edit path is already in the cache.

use crate::app::{ActiveFrame, ActiveFrameKind, App, LodUploadKey};
use crate::app::frame;
use crate::world::gpu;

impl App {
    pub(in crate::app) fn upload_tree(&mut self) {
        self.tree_depth = self.world.tree_depth();
        self.highlight_epoch = self.highlight_epoch.wrapping_add(1);
        self.cached_highlight = None;
        self.upload_tree_lod();
    }

    pub(in crate::app) fn upload_tree_lod(&mut self) {
        let intended_frame = self.target_render_frame();
        let effective_visual_depth = self.visual_depth();
        let upload_key = LodUploadKey::new(self.world.root);
        let reused_gpu_tree = self.last_lod_upload_key == Some(upload_key);
        self.last_effective_visual_depth = effective_visual_depth;
        self.last_reused_gpu_tree = reused_gpu_tree;

        let mut pack_elapsed = std::time::Duration::ZERO;
        if !reused_gpu_tree {
            let pack_start = std::time::Instant::now();
            let cache = self.cached_tree.get_or_insert_with(gpu::CachedTree::new);
            let len_before = cache.tree.len();
            cache.update_root(&self.world.library, self.world.root);
            pack_elapsed = pack_start.elapsed();
            let appended_u32s = cache.tree.len().saturating_sub(len_before);
            self.last_packed_node_count = cache.node_offsets.len() as u32;

            if let Some(renderer) = &mut self.renderer {
                renderer.update_tree(
                    &cache.tree,
                    &cache.node_kinds,
                    &cache.node_offsets,
                    cache.root_bfs_idx,
                );
            }
            self.last_lod_upload_key = Some(upload_key);

            if self.render_harness {
                eprintln!(
                    "render_harness_pack kind={:?} packed_nodes={} library_nodes={} appended_u32s={} pack_ms={:.3}",
                    intended_frame.kind,
                    self.last_packed_node_count,
                    self.world.library.len(),
                    appended_u32s,
                    pack_elapsed.as_secs_f64() * 1000.0,
                );
            }
        }

        // Ribbon rebuilds every frame against the cached tree.
        let ribbon_start = std::time::Instant::now();
        let cache = self
            .cached_tree
            .as_ref()
            .expect("cached_tree populated on first upload_tree_lod");
        let r = gpu::build_ribbon(
            &cache.tree,
            &cache.node_offsets,
            cache.root_bfs_idx,
            intended_frame.render_path.as_slice(),
        );
        let ribbon_elapsed = ribbon_start.elapsed();
        self.last_ribbon_len = r.ribbon.len() as u32;
        let effective_path = frame::frame_from_slots(&r.reached_slots);
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
        if let Some(renderer) = &mut self.renderer {
            renderer.set_frame_root(r.frame_root_idx);
            renderer.update_ribbon(&r.ribbon);
            // Ship the effective render_path's slot sequence so the
            // shader can reconstruct full hit-cell paths for the
            // highlight comparison.
            renderer.set_render_path(self.active_frame.render_path.as_slice());
        }

        let cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(effective_visual_depth);
            renderer.update_camera(&cam_gpu);
            match self.active_frame.kind {
                ActiveFrameKind::Sphere(sphere) => {
                    // Face-depth-conditional dispatch (option B from
                    // docs/history/sphere-locality-refactor-plan.md):
                    //
                    // * face_depth == 0 — whole face visible in the
                    //   viewport. Keep the curved equal-angle
                    //   `march_face_root` rendering so the planet
                    //   silhouette is round. Body-frame math here is
                    //   precision-safe because anchor depth relative
                    //   to body stays shallow at this zoom level.
                    //
                    // * face_depth ≥ 1 — camera has zoomed into a
                    //   sub-face whose angular extent is small. Local
                    //   curvature over the sub-cell is sub-pixel, so
                    //   dispatching `march_cartesian` on the face
                    //   subtree (with face-axis-rotated camera basis
                    //   from `gpu_camera_for_frame`) is visually
                    //   indistinguishable from the curved version
                    //   AND eliminates the body-frame precision wall
                    //   at `anchor_depth ≥ 20` that drove the
                    //   refactor. See user guidance "lose curvature
                    //   after a couple of layers".
                    if sphere.face_depth == 0 {
                        renderer.set_root_kind_face(
                            sphere.inner_r, sphere.outer_r,
                            sphere.face as u32, sphere.face_depth,
                            [sphere.face_u_min, sphere.face_v_min, sphere.face_r_min, sphere.face_size],
                            self.camera.position.in_frame(&sphere.body_path),
                        );
                    } else {
                        renderer.set_root_kind_cartesian();
                    }
                }
                ActiveFrameKind::Body { inner_r, outer_r } => {
                    renderer.set_root_kind_body(inner_r, outer_r);
                }
                ActiveFrameKind::Cartesian => renderer.set_root_kind_cartesian(),
            }
        }
        self.last_pack_ms = pack_elapsed.as_secs_f64() * 1000.0;
        self.last_ribbon_build_ms = ribbon_elapsed.as_secs_f64() * 1000.0;
        if self.startup_profile_frames < 12 {
            eprintln!(
                "startup_perf upload_tree_lod frame={} reused={} pack_ms={:.2} ribbon_ms={:.2} frame_depth={} render_depth={} visual_depth={} kind={:?}",
                self.startup_profile_frames,
                reused_gpu_tree,
                self.last_pack_ms,
                self.last_ribbon_build_ms,
                self.active_frame.logical_path.depth(),
                self.active_frame.render_path.depth(),
                effective_visual_depth,
                self.active_frame.kind,
            );
        }
    }
}
