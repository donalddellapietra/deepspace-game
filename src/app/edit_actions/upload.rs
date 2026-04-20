//! GPU upload: keep the cached packed tree in sync with `world.root`
//! and push per-frame uniforms (camera, ribbon) to the renderer.
//!
//! The packed tree is content-addressed: `CachedTree::update_root`
//! emits (or reuses) nodes keyed by `NodeId`. Initial pack and edit
//! path go through the same function — the edit case is fast because
//! every sibling of the edit path is already in the cache.

use crate::app::{ActiveFrame, App, LodUploadKey};
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
            // Per-frame dispatch. Cartesian = pure Cartesian walker.
            // Body = top-level `sphere_in_cell` with the body at
            // (0, 0, 0)..(3, 3, 3) in render frame. Sphere = render
            // root is inside the face subtree; `sphere_in_cell` gets
            // the body's bounding box expressed in render-frame
            // coords (computed via path arithmetic below), and the
            // shader walks the same curved DDA with precision
            // inherited from render-frame-local math.
            match self.active_frame.kind {
                crate::app::ActiveFrameKind::Cartesian => {
                    renderer.set_root_kind_cartesian();
                }
                crate::app::ActiveFrameKind::Body { inner_r, outer_r } => {
                    renderer.set_root_kind_body(inner_r, outer_r);
                }
                crate::app::ActiveFrameKind::Sphere(sphere) => {
                    // Body origin in render-frame coords: position of
                    // the body cell's (0, 0, 0) corner. Computed via
                    // the SHARED f32 path from render_path to
                    // body_path; `body_cell_size_in_frame =
                    // WORLD_SIZE ÷ 3^(render_depth − body_depth)`.
                    use crate::world::anchor::WORLD_SIZE;
                    let render_depth = self.active_frame.render_path.depth();
                    let body_depth = sphere.body_path.depth();
                    let scale = 3.0_f32.powi(
                        render_depth.saturating_sub(body_depth) as i32,
                    );
                    let body_size_in_frame = WORLD_SIZE * scale;
                    // Position of the body's local `(0, 0, 0)` corner
                    // expressed in the render frame. WorldPos::new
                    // with offset `(0, 0, 0)` at body_path is that
                    // corner; `in_frame(render_path)` projects it to
                    // render-frame coords via the common-prefix
                    // arithmetic (precision-safe).
                    let body_corner = crate::world::anchor::WorldPos::new(
                        sphere.body_path,
                        [0.0, 0.0, 0.0],
                    ).in_frame(&self.active_frame.render_path);
                    renderer.set_root_kind_sphere(
                        sphere.inner_r, sphere.outer_r,
                        body_corner, body_size_in_frame,
                    );
                }
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
