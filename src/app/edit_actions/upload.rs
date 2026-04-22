//! GPU upload: keep the cached packed tree in sync with `world.root`
//! and push per-frame uniforms (camera, ribbon) to the renderer.
//!
//! The packed tree is content-addressed: `CachedTree::update_root`
//! emits (or reuses) nodes keyed by `NodeId`. Initial pack and edit
//! path go through the same function — the edit case is fast because
//! every sibling of the edit path is already in the cache.

use crate::app::{ActiveFrame, ActiveFrameKind, App, LodUploadKey};
use crate::app::frame;
use crate::renderer::{compute_view_proj, EntityRenderMode, InstanceData};
use crate::world::gpu::{self, GpuEntity};
use crate::world::scene::{self, EntityPath};
use crate::world::tree::NodeId;

/// Heuristic: should the renderer run the beam-prepass (P1) for this
/// frame? Returns true iff:
///   1. The root (= frame-root) node's occupancy popcount is ≤ 10.
///      Dense roots (Menger at 20/27, plain worlds filling every slot)
///      have near-100% hit_fraction in the fine pass, so coarse-cull
///      savings are zero and its cost is pure overhead.
///   2. The camera's current root cell is OCCUPIED. This distinguishes
///      "inside sparse content with long empty channels" (Jerusalem
///      nucleus) — where P1 wins big — from "outside content looking
///      in" (Jerusalem corner) — where the fine pass is already fast.
///
/// Both checks are 2-3 u32 reads + a popcount. Zero-cost heuristic,
/// no readback, no lag. Revisited every `upload_tree_lod` call
/// (which fires on camera move + edits).
fn compute_beam_enable(
    cache: &gpu::CachedTree,
    frame_root_bfs: u32,
    camera_pos: [f32; 3],
) -> bool {
    let root_header = cache.node_offsets.get(frame_root_bfs as usize).copied();
    let Some(off) = root_header else { return false; };
    let Some(&occupancy) = cache.tree.get(off as usize) else { return false; };
    if occupancy.count_ones() > 10 {
        return false; // Dense — P1 can't cull anything meaningful.
    }
    // Sparse root: decide by camera position. Inside an occupied
    // cell → rays traverse long internal empty channels (Jerusalem
    // nucleus) → P1 wins. Inside an empty cell → rays shoot into
    // nearby content and hit fast (Jerusalem corner, Cantor default
    // corner spawn) → fine pass is already fast enough that the
    // coarse pass's own cost swamps any savings.
    let cx = camera_pos[0].floor().clamp(0.0, 2.0) as u32;
    let cy = camera_pos[1].floor().clamp(0.0, 2.0) as u32;
    let cz = camera_pos[2].floor().clamp(0.0, 2.0) as u32;
    let slot = cx + cy * 3 + cz * 9;
    (occupancy >> slot) & 1 != 0
}

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

        // Pre-pack every entity's active subtree so its BFS idx is
        // ready when we build the entity GPU buffer below. The cache
        // dedups by NodeId, so 10k copies of the same soldier share
        // a single pack of ~100 unique nodes.
        // In raster mode, entity subtree BFS indices are unused (the
        // ray-march's tag=3 branch is dead — the raster pass draws
        // entities as meshes). Skip the pre-pack loop, which at 10k
        // entities was the dominant per-frame CPU cost.
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

        // Tree repack → heightmap source may have changed → flag a
        // rebuild. The renderer's record_frame_passes also detects
        // frame-root changes internally; this is the edit half.
        if !reused_gpu_tree {
            if let Some(renderer) = &mut self.renderer {
                renderer.mark_heightmap_dirty();
            }
        }

        let mut pack_elapsed = web_time::Duration::ZERO;
        if !reused_gpu_tree {
            let pack_start = web_time::Instant::now();
            let cache = self.cached_tree.as_mut().expect("cache inserted above");
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

        // --- Build + pack scene root (overlay) ---
        // Entities in ray-march mode enter the tree as Child::EntityRef
        // cells at their anchor slot, wrapped in a per-frame ephemeral
        // ancestor chain. Terrain-only slots share NodeIds with
        // `world.root` via content-addressed dedup, so the scene root
        // reuses ~100% of the already-packed terrain buffer.
        // Raster mode: skip the scene overlay entirely. Entities draw
        // through the instanced raster pass, not as Child::EntityRef
        // cells, so the scene root collapses to `world.root` (content-
        // addressed fast path) and the ribbon stays pure-terrain.
        let raster_mode = self
            .renderer
            .as_ref()
            .map(|r| r.entity_render_mode() == EntityRenderMode::Raster)
            .unwrap_or(false);
        let intended_render_path = intended_frame.render_path;
        let mut entity_paths: Vec<EntityPath> =
            Vec::with_capacity(self.entities.len());
        if !raster_mode {
            for (idx, e) in self.entities.entities.iter().enumerate() {
                let anchor = e.pos.anchor;
                if anchor.depth() == 0 { continue; }
                entity_paths.push(EntityPath {
                    entity_idx: idx as u32,
                    path_slots: anchor.as_slice().to_vec(),
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
        let cache = self.cached_tree.as_mut().expect("cached_tree present");
        let _scene_root_bfs = cache.ensure_root(&self.world.library, scene_root);

        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(
                &cache.tree,
                &cache.node_kinds,
                &cache.node_offsets,
                &cache.aabbs,
                cache.root_bfs_idx,
            );
        }

        // --- Ribbon on TERRAIN ---
        let ribbon_intended_path: crate::world::anchor::Path = intended_render_path;
        let ribbon_start = web_time::Instant::now();
        let cache = self.cached_tree.as_ref().expect("cached_tree");
        let r = gpu::build_ribbon(
            &cache.tree,
            &cache.node_offsets,
            cache.root_bfs_idx,
            ribbon_intended_path.as_slice(),
        );
        let ribbon_elapsed = ribbon_start.elapsed();
        self.last_ribbon_len = r.ribbon.len() as u32;

        // --- Frame root: SCENE at terrain's reached depth ---
        // Walk scene_root down the ribbon's reached slot path to
        // find the scene-side equivalent node the shader should
        // use as its frame root. Terrain-sibling slots render
        // identically to baseline (same packed NodeIds via dedup);
        // scene-overlay slots hit Child::EntityRef leaves.
        let effective_path = frame::frame_from_slots(&r.reached_slots);
        let mut scene_frame_id = scene_root;
        for &slot in r.reached_slots.iter() {
            let Some(node) = self.world.library.get(scene_frame_id) else { break };
            match node.children[slot as usize] {
                crate::world::tree::Child::Node(id) => scene_frame_id = id,
                _ => break,
            }
        }
        let cache = self.cached_tree.as_mut().expect("cached_tree");
        let scene_frame_bfs = cache.ensure_root(
            &self.world.library, scene_frame_id,
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(
                &cache.tree,
                &cache.node_kinds,
                &cache.node_offsets,
                &cache.aabbs,
                cache.root_bfs_idx,
            );
        }

        // GPU ribbon may have truncated below the intended depth —
        // re-resolve the active frame at the effective truncated depth.
        self.active_frame = {
            let effective_render = frame::compute_render_frame(
                &self.world.library,
                self.world.root,
                &self.camera.position,
                effective_path.depth(),
            );
            ActiveFrame {
                render_path: effective_render.render_path,
                logical_path: intended_frame.logical_path,
                node_id: effective_render.node_id,
                kind: effective_render.kind,
            }
        };
        if let Some(renderer) = &mut self.renderer {
            renderer.set_frame_root(scene_frame_bfs);
            renderer.update_ribbon(&r.ribbon);
        }

        let mut cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        // Fill in view_proj so fs_main_depth's clip.z/clip.w matches
        // the raster pipeline's own projection. Seeded as identity at
        // camera construction; the ray-march path ignores it when
        // raster is disabled.
        if let Some(renderer) = &self.renderer {
            let (w, h) = renderer.march_dims_public();
            let aspect = w as f32 / h as f32;
            cam_gpu.view_proj = compute_view_proj(
                cam_gpu.pos, cam_gpu.forward, cam_gpu.right, cam_gpu.up,
                cam_gpu.fov, aspect,
            );
        }

        // Beam-prepass (P1) heuristic: worth running only when the
        // scene is sparse at the root AND the camera sits inside an
        // occupied cell (nucleus-like: rays traverse internal empty
        // channels → miss-heavy → culling wins).
        //
        // Dense-root scenes (Menger, plain worlds) or sparse-root
        // scenes where the camera is in an EMPTY root cell (looking
        // into content from outside, corner-like) don't benefit from
        // the cull — the coarse pass is pure overhead there. The
        // renderer skips it and just clears the mask to 1.0 so the
        // fine pass marches every pixel unconditionally.
        // Beam prepass is Cartesian-only; sphere/body frames have
        // their own cubemap-based culling inside the sphere DDA.
        let beam_enabled = matches!(self.active_frame.kind, ActiveFrameKind::Cartesian)
            && compute_beam_enable(
                self.cached_tree.as_ref().expect("cached_tree populated"),
                r.frame_root_idx,
                cam_gpu.pos,
            );

        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(effective_visual_depth);
            renderer.set_beam_enabled(beam_enabled);
            renderer.update_camera(&cam_gpu);
            match self.active_frame.kind {
                ActiveFrameKind::Body { inner_r, outer_r } => {
                    renderer.set_root_kind_body(inner_r, outer_r);
                }
                ActiveFrameKind::Cartesian => {
                    renderer.set_root_kind_cartesian();
                }
            }
        }
        self.last_pack_ms = pack_elapsed.as_secs_f64() * 1000.0;
        self.last_ribbon_build_ms = ribbon_elapsed.as_secs_f64() * 1000.0;

        // --- Entity GPU buffer upload ---
        // Bbox is in the render frame's `[0, 3)^3` local coords —
        // `e.pos.in_frame(&effective_path)` plus the entity's
        // anchor cell size (scaled by depth delta between anchor
        // and frame). The shader's tag=3 branch ray-AABB-tests this
        // bbox directly so sub-cell motion shifts the entity
        // every frame without rebuilding the tree.
        let frame_world_size = crate::world::anchor::WORLD_SIZE;
        let frame_depth = effective_path.depth() as i32;
        if !raster_mode {
            let mut gpu_entities: Vec<GpuEntity> =
                Vec::with_capacity(self.entities.len());
            for e in &self.entities.entities {
                let rep = self
                    .world
                    .library
                    .get(e.active_root())
                    .map(|n| n.representative_block as u32)
                    .unwrap_or(0xFFFEu32);
                let anchor_depth = e.pos.anchor.depth() as i32;
                let depth_delta = (anchor_depth - frame_depth).max(0);
                let size = frame_world_size / 3.0_f32.powi(depth_delta);
                let bbox_min = e.pos.in_frame(&effective_path);
                let bbox_max = [
                    bbox_min[0] + size, bbox_min[1] + size, bbox_min[2] + size,
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
            if self.render_harness && !self.entities.is_empty() {
                eprintln!(
                    "entity_upload entities={} scene_frame_bfs={} ribbon_len={}",
                    gpu_entities.len(),
                    scene_frame_bfs,
                    r.ribbon.len(),
                );
            }
        } else if let Some(renderer) = &mut self.renderer {
            // Raster mode: ray-march sees zero entities.
            renderer.update_entities(&[]);
        }

        // --- Raster entity path ---
        if raster_mode {
            let palette = self.palette.to_gpu_palette();
            let mut per_entity: Vec<(NodeId, InstanceData)> =
                Vec::with_capacity(self.entities.len());
            let lod_cube_pixels: f32 = 24.0;
            let march_h = self
                .renderer
                .as_ref()
                .map(|r| r.march_dims_public().1)
                .unwrap_or(360);
            let focal_px = march_h as f32 / (2.0 * (0.6_f32).tan());
            let cam = self.camera.position.in_frame(&effective_path);
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
                        .unwrap_or(0);
                    let c = palette
                        .get(rep)
                        .copied()
                        .unwrap_or([0.5, 0.5, 0.5, 1.0]);
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
                        raster.ensure_mesh(&device, library, *node_id, &palette);
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
