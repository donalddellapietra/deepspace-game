//! GPU upload: keep the cached packed tree in sync with `world.root`
//! and push per-frame uniforms (camera, ribbon) to the renderer.
//!
//! The packed tree is content-addressed: `CachedTree::update_root`
//! emits (or reuses) nodes keyed by `NodeId`. Initial pack and edit
//! path go through the same function — the edit case is fast because
//! every sibling of the edit path is already in the cache.

use crate::app::{ActiveFrame, ActiveFrameKind, App, LodUploadKey};
use crate::app::frame;
use crate::world::anchor::WORLD_SIZE;
use crate::world::gpu::{self, GpuEntity};

impl App {
    pub(in crate::app) fn upload_tree(&mut self) {
        self.tree_depth = self.world.tree_depth();
        self.highlight_epoch = self.highlight_epoch.wrapping_add(1);
        self.cached_highlight = None;
        self.upload_tree_lod();
    }

    /// Pack every entity's active root into the shared GPU tree
    /// buffer, then build a per-frame `GpuEntity` list and upload it
    /// to the entity buffer. Runs at the end of `upload_tree_lod`
    /// so `self.active_frame.render_path` is already set.
    ///
    /// Entity subtrees live in the same `tree[]` as the world root —
    /// the content-addressed cache means 10k identical entities
    /// share one packed subtree at zero cost.
    pub(in crate::app) fn upload_entities(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else { return };
        if self.entities.is_empty() {
            renderer.update_entities(&[]);
            return;
        }
        let Some(cache) = self.cached_tree.as_mut() else { return };
        let len_before_u32 = cache.tree.len();
        let kinds_before = cache.node_kinds.len();
        for e in self.entities.entities.iter_mut() {
            e.bfs_idx = cache.ensure_root(&self.world.library, e.active_root());
        }
        // If packing entity content appended new nodes, push the
        // tail to GPU — same append-only path world edits use.
        if cache.tree.len() != len_before_u32 || cache.node_kinds.len() != kinds_before {
            renderer.update_tree(
                &cache.tree,
                &cache.node_kinds,
                &cache.node_offsets,
                cache.root_bfs_idx,
            );
        }
        // Build per-frame GPU records. `pos.in_frame(frame)` gives
        // the entity's bbox_min corner in the current frame's local
        // [0, 3)³ coords — sub-cell offset shifts this continuously
        // each frame, driving smooth motion under `EntityStore::tick`.
        //
        // Skip entities at or above the frame depth (v1: entities
        // must be smaller than the frame cell; larger entities
        // would contain the camera — handled later with LOD).
        let frame = self.active_frame.render_path;
        let frame_depth = frame.depth() as i32;
        let mut gpu = Vec::with_capacity(self.entities.len());
        for e in &self.entities.entities {
            let anchor_depth = e.pos.anchor.depth() as i32;
            if anchor_depth < frame_depth {
                continue;
            }
            let bbox_min = e.pos.in_frame(&frame);
            let size = WORLD_SIZE / 3.0_f32.powi(anchor_depth - frame_depth);
            gpu.push(GpuEntity {
                bbox_min,
                _pad0: 0.0,
                bbox_max: [bbox_min[0] + size, bbox_min[1] + size, bbox_min[2] + size],
                subtree_bfs: e.bfs_idx,
            });
        }
        renderer.update_entities(&gpu);
    }

    pub(in crate::app) fn upload_tree_lod(&mut self) {
        let intended_frame = self.target_render_frame();
        let effective_visual_depth = self.visual_depth();
        let upload_key = LodUploadKey::new(self.world.root);
        let reused_gpu_tree = self.last_lod_upload_key == Some(upload_key);
        self.last_effective_visual_depth = effective_visual_depth;
        self.last_reused_gpu_tree = reused_gpu_tree;

        let mut pack_elapsed = web_time::Duration::ZERO;
        if !reused_gpu_tree {
            let pack_start = web_time::Instant::now();
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
        let ribbon_start = web_time::Instant::now();
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
        }

        let cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(effective_visual_depth);
            renderer.update_camera(&cam_gpu);
            match self.active_frame.kind {
                ActiveFrameKind::Sphere(sphere) => {
                    renderer.set_root_kind_face(
                        sphere.inner_r, sphere.outer_r,
                        sphere.face as u32, sphere.face_depth,
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
        self.last_ribbon_build_ms = ribbon_elapsed.as_secs_f64() * 1000.0;

        // Entity pass: pack entity subtrees into the shared tree
        // buffer (if not already) and upload per-frame bboxes. Runs
        // every frame — entity bboxes depend on the render frame,
        // which shifts as the camera zooms.
        self.upload_entities();

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
