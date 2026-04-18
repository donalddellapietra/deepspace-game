//! GPU upload: pack the world tree and build the ancestor ribbon,
//! then push both (plus the per-frame uniforms) to the renderer.
//!
//! The pack is a pure function of `(library, root)`. LOD lives in
//! the shader; pure camera motion never repacks. An edit (which
//! changes `world.root`) is the only thing that invalidates the
//! buffer.

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

    /// Pack the world tree (content-only, no LOD) and push it to the
    /// GPU along with the ancestor ribbon. Camera uniforms refresh
    /// every frame; the packed tree only re-uploads when `root` or a
    /// view-structural field changes (render_path, logical_path,
    /// visual_depth, kind_tag).
    pub(in crate::app) fn upload_tree_lod(&mut self) {
        let intended_frame = self.target_render_frame();
        let effective_visual_depth = self.visual_depth();
        let upload_key = LodUploadKey::new(self.world.root);
        let mut pack_elapsed = std::time::Duration::ZERO;
        let mut ribbon_elapsed = std::time::Duration::ZERO;
        let reused_gpu_tree = self.last_lod_upload_key == Some(upload_key);
        self.last_effective_visual_depth = effective_visual_depth;
        self.last_reused_gpu_tree = reused_gpu_tree;

        // The tree buffer only depends on (library, root). Everything
        // else — camera, render frame, ribbon — is recomputed every
        // frame and uploaded cheaply. Pack runs when root changes.
        if !reused_gpu_tree {
            let pack_start = std::time::Instant::now();
            let (tree_packed, node_kinds, node_offsets, _world_root_idx) =
                gpu::pack_tree(&self.world.library, self.world.root);
            pack_elapsed = pack_start.elapsed();
            let packed_node_count = node_kinds.len();
            self.last_packed_node_count = packed_node_count as u32;

            if let Some(renderer) = &mut self.renderer {
                renderer.update_tree(&tree_packed, &node_kinds, &node_offsets, 0);
            }
            self.last_lod_upload_key = Some(upload_key);
            self.cached_tree = Some(CachedTree { tree: tree_packed, node_offsets });

            if self.render_harness {
                eprintln!(
                    "render_harness_pack kind={:?} packed_nodes={} library_nodes={}",
                    intended_frame.kind,
                    packed_node_count,
                    self.world.library.len(),
                );
            }
        }

        // Ribbon depends on (packed tree, render_path). Pure camera
        // motion can change render_path, so rebuild each frame using
        // the cached tree buffer. It's a short walk (≤ render depth).
        let ribbon_start = std::time::Instant::now();
        let cached = self.cached_tree.as_ref().expect("cached_tree populated above");
        let r = gpu::build_ribbon(
            &cached.tree,
            &cached.node_offsets,
            intended_frame.render_path.as_slice(),
        );
        ribbon_elapsed = ribbon_start.elapsed();
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

/// Cached packed tree so the per-frame ribbon walk has something to
/// traverse without re-packing. Populated on every real pack; lives
/// as long as the current LOD upload key.
pub(crate) struct CachedTree {
    pub tree: Vec<u32>,
    pub node_offsets: Vec<u32>,
}
