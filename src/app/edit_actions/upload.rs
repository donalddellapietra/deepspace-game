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
use crate::renderer::{compute_view_proj, EntityRenderMode, InstanceData};
use crate::world::gpu::{self, GpuEntity};
use crate::world::scene::{self, EntityPath};
use crate::world::tree::NodeId;

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
        // In raster mode entities are not drawn through the ray-march
        // DDA, so their `bfs_idx` is never read. Skip the pre-pack
        // loop entirely — at 10k entities this was the dominant
        // per-frame CPU cost.
        let raster_pre = self
            .renderer
            .as_ref()
            .map(|r| r.entity_render_mode() == EntityRenderMode::Raster)
            .unwrap_or(false);
        let cache = self.cached_tree.get_or_insert_with(gpu::CachedTree::new);
        if !raster_pre {
            for e in self.entities.entities.iter_mut() {
                e.bfs_idx = cache.ensure_root(&self.world.library, e.active_root());
            }
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

        // Any repack means the tree nodes that back the heightmap
        // may have changed shape — flag it so the next frame's
        // `record_frame_passes` reruns the gen compute pass. The
        // renderer also detects frame-root changes internally;
        // this is the edit-invalidation half.
        if !reused_gpu_tree {
            if let Some(renderer) = &mut self.renderer {
                renderer.mark_heightmap_dirty();
            }
        }

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
        // When the renderer is configured for raster-entity rendering,
        // entities are drawn as instanced triangle meshes in a
        // separate pass — they do NOT need to appear in the world
        // tree as Child::EntityRef cells. Skip the overlay so the
        // ray-march walks terrain alone and the scene root collapses
        // to `world.root` (content-addressed fast path).
        let raster_mode = self
            .renderer
            .as_ref()
            .map(|r| r.entity_render_mode() == EntityRenderMode::Raster)
            .unwrap_or(false);
        let intended_render_path = intended_frame.render_path;
        let mut entity_paths: Vec<EntityPath> = Vec::with_capacity(self.entities.len());
        if !raster_mode {
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

        let mut cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        // Fill in view_proj for the raster-entity path (and for
        // consistency; ray-march ignores it when raster is off).
        // The matrix uses the march-target aspect ratio so the
        // depth values `fs_main_depth` writes match what the raster
        // pipeline computes from the same world hits.
        if let Some(renderer) = &self.renderer {
            let (w, h) = renderer.march_dims_public();
            let aspect = w as f32 / h as f32;
            cam_gpu.view_proj = compute_view_proj(
                cam_gpu.pos, cam_gpu.forward, cam_gpu.right, cam_gpu.up,
                cam_gpu.fov, aspect,
            );
        }
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
        // Skip the GpuEntity build loop entirely in raster mode —
        // the shader's entity buffer is unused there (entity_count=0
        // gates the march's inner loop). Building 10k GpuEntity
        // structs per frame just to discard them dominated "upload".
        if !raster_mode {
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
        } else if let Some(renderer) = &mut self.renderer {
            renderer.update_entities(&[]);
        }

        // --- Raster entity path ---
        if raster_mode {
            // Build per-entity instance data in the same frame-local
            // coordinates as the ray-march would use. bbox_min + size
            // are already computed above; mesh NodeId comes from the
            // entity's active_root (override_root if edited, else
            // shared subtree). ensure_mesh extracts + uploads the
            // triangle mesh the first time each unique NodeId is seen.
            let palette = self.palette.to_gpu_palette();
            let mut per_entity: Vec<(NodeId, InstanceData)> =
                Vec::with_capacity(self.entities.len());
            // Entities below this projected screen size are drawn as
            // a single colored cube (LOD impostor). Sized to match
            // the ray-march's sub-pixel representative-block splat —
            // a soldier 20 pixels tall is already a blob of color.
            let lod_cube_pixels: f32 = 24.0;
            let march_h = self
                .renderer
                .as_ref()
                .map(|r| r.march_dims_public().1)
                .unwrap_or(360);
            let focal_px = march_h as f32 / (2.0 * (0.6_f32).tan());
            // Hoist loop invariants: the camera, frame depth, and
            // pixel-size threshold squared are all constant across
            // entities — at 10k entities the per-iteration HashMap
            // lookups and path recomputation dominated what should
            // be a ~200k-float-op loop.
            let cam = self.camera.position.in_frame(&effective_path);
            let frame_depth = effective_path.depth() as i32;
            let lod_cube_node = crate::renderer::entity_raster::LOD_CUBE_NODE;
            for e in &self.entities.entities {
                let anchor_depth = e.pos.anchor.depth() as i32;
                let depth_delta = (anchor_depth - frame_depth).max(0);
                let size = frame_world_size / 3.0_f32.powi(depth_delta);
                let translate = e.pos.in_frame(&effective_path);
                let half = size * 0.5;
                let dx = translate[0] + half - cam[0];
                let dy = translate[1] + half - cam[1];
                let dz = translate[2] + half - cam[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                // pixel_size = size / dist * focal_px
                // pixel_size < threshold  ⇔  size * focal_px < threshold * dist
                //                         ⇔  (size*focal_px)^2 < threshold^2 * dist^2
                // Avoid the sqrt in the hot path.
                let num_sq = (size * focal_px) * (size * focal_px);
                let cutoff = lod_cube_pixels * lod_cube_pixels * dist_sq;
                let use_cube = num_sq < cutoff;
                let root = e.active_root();
                let (mesh_id, tint) = if use_cube {
                    let rep = self
                        .world
                        .library
                        .get(root)
                        .map(|n| n.representative_block as usize)
                        .unwrap_or(255)
                        .min(255);
                    let c = palette.colors[rep];
                    (lod_cube_node, [c[0], c[1], c[2], 1.0])
                } else {
                    (root, [1.0, 1.0, 1.0, 1.0])
                };
                per_entity.push((mesh_id, InstanceData {
                    translate,
                    scale: size,
                    tint,
                }));
            }
            // Size the heightmap for this frame's entity depth.
            // Collision granularity is entity_anchor_depth + 1, so
            // delta = (anchor_depth + 1) - frame_depth. If entities
            // live at different depths (rare), we use the deepest
            // to keep the heightmap fine enough for everyone.
            let max_entity_depth = self
                .entities
                .entities
                .iter()
                .map(|e| e.pos.anchor.depth() as i32)
                .max()
                .unwrap_or(frame_depth);
            let heightmap_delta = (max_entity_depth + 1 - frame_depth).max(0) as u32;
            if let Some(renderer) = &mut self.renderer {
                renderer.ensure_heightmap(heightmap_delta);
            }
            if let Some(renderer) = &mut self.renderer {
                // Snapshot the device+queue handles so we can borrow
                // the raster state mutably without clashing with the
                // palette borrow.
                let device = renderer.device().clone();
                let queue = renderer.queue().clone();
                let aspect = {
                    let (w, h) = renderer.march_dims_public();
                    w as f32 / h as f32
                };
                let view_proj = compute_view_proj(
                    cam_gpu.pos, cam_gpu.forward, cam_gpu.right, cam_gpu.up,
                    cam_gpu.fov, aspect,
                );
                let library = &self.world.library;
                if let Some(raster) = renderer.entity_raster_mut() {
                    for (node_id, _) in &per_entity {
                        raster.ensure_mesh(&device, library, *node_id, &palette.colors);
                    }
                    raster.update_view_proj(&queue, view_proj);
                    raster.update_instances(&device, &queue, &per_entity);
                    if self.render_harness {
                        eprintln!(
                            "entity_raster_upload entities={} unique_meshes={} batches={} instances={}",
                            per_entity.len(),
                            raster.cached_meshes(),
                            raster.batch_count(),
                            raster.total_instances(),
                        );
                    }
                }
            }
        } else if self.render_harness {
            eprintln!(
                "entity_upload entities={} scene_root_id={} scene_frame_bfs={} ribbon_len={}",
                self.entities.len(),
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
