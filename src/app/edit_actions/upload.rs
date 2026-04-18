//! GPU upload: keep the cached packed tree in sync with `world.root`
//! and push per-frame uniforms (camera, ribbon) to the renderer.
//!
//! The packed tree is content-addressed: `CachedTree::update_root`
//! emits (or reuses) nodes keyed by `NodeId`. Initial pack and edit
//! path go through the same function — the edit case is fast because
//! every sibling of the edit path is already in the cache.
//!
//! ## Scene-root overlay
//!
//! When entities exist, `upload_tree_lod` builds a per-frame scene
//! root via `world::scene::build_scene_root` that overlays
//! `Child::EntityRef(idx)` onto terrain at every entity's anchor
//! slot. The render pack root becomes the scene root; the ribbon and
//! render-frame computation walk the scene tree, so the shader's
//! unified `march_cartesian` encounters entity cells naturally as
//! `tag=3` DDA leaves — no separate hash-grid pass, no gather.
//!
//! When no entities exist the scene root collapses to `world.root`
//! unchanged (content-addressed fast path).

use crate::app::{ActiveFrame, ActiveFrameKind, App, LodUploadKey};
use crate::app::frame;
use crate::world::gpu::{self, GpuEntity};
use crate::world::scene::{self, EntityPath};

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

        // --- Entity subtree pre-pack ---
        let cache = self.cached_tree.get_or_insert_with(gpu::CachedTree::new);
        for e in self.entities.entities.iter_mut() {
            e.bfs_idx = cache.ensure_root(&self.world.library, e.active_root());
        }

        // --- Pack WORLD.ROOT for ribbon & LOD baseline ---
        // Ribbon is computed against terrain only — NOT the scene
        // overlay. This keeps `ribbon_level` (shader's distance-based
        // detail budget) identical to a no-entity run, so terrain
        // renders with the same LOD splat regardless of entity
        // presence. The scene overlay's job is to make entities
        // findable from the frame, not to alter terrain detail.
        let upload_key = LodUploadKey::new(self.world.root);
        let reused_gpu_tree = self.last_lod_upload_key == Some(upload_key);
        self.last_effective_visual_depth = effective_visual_depth;
        self.last_reused_gpu_tree = reused_gpu_tree;

        let mut pack_elapsed = web_time::Duration::ZERO;
        if !reused_gpu_tree {
            let pack_start = web_time::Instant::now();
            let cache = self
                .cached_tree
                .as_mut()
                .expect("cached_tree inserted above");
            let len_before = cache.tree.len();
            cache.update_root(&self.world.library, self.world.root);
            pack_elapsed = pack_start.elapsed();
            let appended_u32s = cache.tree.len().saturating_sub(len_before);
            self.last_packed_node_count = cache.node_offsets.len() as u32;
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

        // --- Build + pack scene root ---
        let intended_render_path = intended_frame.render_path;
        let mut entity_paths: Vec<EntityPath> = Vec::with_capacity(self.entities.len());
        for (idx, e) in self.entities.entities.iter().enumerate() {
            let anchor = e.pos.anchor;
            if anchor.depth() == 0 {
                continue;
            }
            let anchor_slots = anchor.as_slice();
            entity_paths.push(EntityPath {
                entity_idx: idx as u32,
                path_slots: anchor_slots.to_vec(),
            });
        }
        let scene_result = scene::build_scene_root(
            &mut self.world.library,
            self.world.root,
            &entity_paths,
        );
        let scene_root = scene_result.node_id;

        self.world.library.ref_inc(scene_root);
        if let Some(prev) = self.active_scene_root.replace(scene_root) {
            self.world.library.ref_dec(prev);
        }

        let cache = self
            .cached_tree
            .as_mut()
            .expect("cached_tree present");
        let scene_root_bfs = cache.ensure_root(&self.world.library, scene_root);
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(
                &cache.tree,
                &cache.node_kinds,
                &cache.node_offsets,
                cache.root_bfs_idx,
            );
        }

        // --- Ribbon on TERRAIN ---
        let ribbon_start = web_time::Instant::now();
        let cache = self
            .cached_tree
            .as_ref()
            .expect("cached_tree populated on first upload_tree_lod");
        let r = gpu::build_ribbon(
            &cache.tree,
            &cache.node_offsets,
            cache.root_bfs_idx,
            intended_render_path.as_slice(),
        );
        let ribbon_elapsed = ribbon_start.elapsed();
        self.last_ribbon_len = r.ribbon.len() as u32;

        // --- Frame root: SCENE at terrain's reached depth ---
        // Walk scene_root down `reached_slots` in the LIBRARY (not
        // the pack) to find the scene-side equivalent of the node
        // terrain's ribbon ended at. That NodeId's BFS becomes the
        // shader's `frame_root_idx`. Rays hitting terrain-sibling
        // slots of the scene frame render identically to baseline
        // (same packed NodeIds via content-addressed dedup); rays
        // hitting the scene-overlay slot follow the ephemeral chain
        // down to the entity.
        let effective_path = frame::frame_from_slots(&r.reached_slots);
        let mut scene_frame_id = scene_root;
        for &slot in r.reached_slots.iter() {
            let Some(node) = self.world.library.get(scene_frame_id) else { break };
            match node.children[slot as usize] {
                crate::world::tree::Child::Node(id) => scene_frame_id = id,
                _ => break,
            }
        }
        let cache = self.cached_tree.as_mut().expect("cached_tree present");
        let scene_frame_bfs = cache.ensure_root(
            &self.world.library, scene_frame_id,
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(
                &cache.tree,
                &cache.node_kinds,
                &cache.node_offsets,
                cache.root_bfs_idx,
            );
        }
        let _ = scene_root_bfs;

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
            renderer.set_frame_root(scene_frame_bfs);
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

        // --- Upload entities buffer ---
        // Entity bbox is in the SCENE frame's [0, 3)^3 coords —
        // computed from the entity's `in_frame(&effective_path)`
        // plus its anchor cell size. The shader's tag=3 branch
        // ray-AABB tests this bbox directly (the scene tree's cell
        // just locates the entity coarsely; motion within the anchor
        // cell shifts the bbox sub-cell every frame, which the
        // ray-box test picks up without rebuilding the tree).
        let frame_world_size = crate::world::anchor::WORLD_SIZE;
        let mut gpu_entities: Vec<GpuEntity> = Vec::with_capacity(self.entities.len());
        for e in &self.entities.entities {
            let rep = self
                .world
                .library
                .get(e.active_root())
                .map(|n| n.representative_block as u32)
                .unwrap_or(255);
            let anchor_depth = e.pos.anchor.depth() as i32;
            let frame_depth = effective_path.depth() as i32;
            let depth_delta = (anchor_depth - frame_depth).max(0) as u32;
            let size = frame_world_size / 3.0_f32.powi(depth_delta as i32);
            let bbox_min = e.pos.in_frame(&effective_path);
            let bbox_max = [
                bbox_min[0] + size,
                bbox_min[1] + size,
                bbox_min[2] + size,
            ];
            gpu_entities.push(GpuEntity {
                bbox_min,
                representative_block: rep,
                bbox_max,
                subtree_bfs: e.bfs_idx,
            });
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.update_entities(&gpu_entities);
        }
        if self.render_harness {
            eprintln!(
                "entity_upload entities={} scene_root_id={} scene_frame_bfs={} ribbon_len={}",
                gpu_entities.len(),
                scene_root,
                scene_frame_bfs,
                r.ribbon.len(),
            );
        }

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
