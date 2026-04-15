//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! The rendering path is driven in the **render frame** — an
//! ancestor of the camera's anchor a few levels up. Every value
//! that reaches the shader (camera position, planet center, block
//! highlight AABB) is expressed in that frame's local
//! `[0, WORLD_SIZE)³` system so f32 precision is preserved at any
//! anchor depth. Absolute world XYZ is reserved for CPU-side
//! operations (cursor raycast, gravity, editing) where cell-scale
//! noise at deep anchors is invisible.

use crate::editing;
use crate::game_state::HotbarItem;
use crate::world::anchor::WORLD_SIZE;
use crate::world::edit;
use crate::world::gpu;

use super::App;

impl App {
    /// CPU raycast depth: one less than the camera's anchor depth.
    pub(super) fn edit_depth(&self) -> u32 {
        self.anchor_depth().saturating_sub(1).max(1)
    }

    /// GPU visual depth: edit_depth + 3.
    pub(super) fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(16)
    }

    pub(super) fn apply_zoom(&mut self) {
        self.ui.zoom_level = self.zoom_level();
        let vd = self.visual_depth();
        let (frame, _) = self.render_frame();
        let cam_local = self.camera.position.in_frame(&frame);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&self.camera.gpu_camera_at(cam_local, 1.2));
        }
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}, anchor_depth: {}, frame_depth: {}",
            self.zoom_level(),
            self.tree_depth as i32,
            self.edit_depth(),
            vd,
            self.anchor_depth(),
            frame.depth(),
        );
    }

    pub(super) fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let edit_depth = self.edit_depth();
        let zoom_level = self.zoom_level();
        let camera_pos = self.camera.world_pos_f32();

        if editing::try_cs_break(
            &mut self.world,
            self.cs_planet.as_mut(),
            camera_pos,
            ray_dir,
            zoom_level,
            edit_depth,
        ) {
            return;
        }

        let hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            camera_pos,
            ray_dir,
            self.edit_depth(),
        );
        let Some(hit) = hit else { return };

        if self.save_mode {
            use crate::world::tree::Child;
            let mut saved_id = None;
            if let Some(&(parent_id, slot)) = hit.path.last() {
                if let Some(node) = self.world.library.get(parent_id) {
                    match node.children[slot] {
                        Child::Node(child_id) => saved_id = Some(child_id),
                        Child::Block(_) | Child::Empty => {
                            saved_id = Some(parent_id);
                        }
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

        if edit::break_block(&mut self.world, &hit) {
            self.upload_tree();
        }
    }

    pub(super) fn do_place(&mut self) {
        let ray_dir = self.camera.forward();
        let edit_depth = self.edit_depth();
        let zoom_level = self.zoom_level();
        let camera_pos = self.camera.world_pos_f32();

        if let Some(block_type) = self.ui.active_block_type() {
            if editing::try_cs_place(
                &mut self.world,
                self.cs_planet.as_mut(),
                camera_pos,
                ray_dir,
                zoom_level,
                edit_depth,
                block_type,
            ) {
                return;
            }
        }

        let hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            camera_pos,
            ray_dir,
            self.edit_depth(),
        );
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
                    &mut self.world,
                    &hit,
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

    /// Pack and upload the tree subtree rooted at the current render
    /// frame. The shader sees the frame as its `[0, WORLD_SIZE)` root;
    /// camera position, planet center, and planet radii are all
    /// transformed into the same local frame so f32 stays sub-cell
    /// accurate at any anchor depth.
    pub(super) fn upload_tree_lod(&mut self) {
        let (frame, frame_root) = self.render_frame();
        let frame_scale = 3.0f32.powi(frame.depth() as i32);
        let cam_local = self.camera.position.in_frame(&frame);

        let mut roots: Vec<u64> = vec![frame_root];
        if let Some(p) = self.cs_planet.as_ref() {
            roots.extend_from_slice(&p.face_roots);
        }
        let (tree_data, root_indices) = gpu::pack_tree_lod_multi(
            &self.world.library,
            &roots,
            cam_local,
            1440.0,
            1.2,
        );

        let Some(renderer) = &mut self.renderer else { return; };
        renderer.update_tree(&tree_data, root_indices[0]);
        if let Some(planet) = self.cs_planet.as_ref() {
            let face_roots: [u32; 6] = [
                root_indices[1], root_indices[2], root_indices[3],
                root_indices[4], root_indices[5], root_indices[6],
            ];
            renderer.set_face_roots(face_roots);

            // Transform the planet into the render frame. `in_frame`
            // handles the in-branch case precisely and the
            // cross-branch case via common-ancestor composition —
            // either way we get shader-appropriate frame-local
            // coordinates. Deepen first so tail-path walk spans
            // the finer scale.
            let deepened = planet.center_worldpos.deepened_to(frame.depth());
            let local_center = deepened.in_frame(&frame);
            renderer.set_cubed_sphere_planet(
                local_center,
                planet.inner_r * frame_scale,
                planet.outer_r * frame_scale,
                planet.depth,
            );

            // Path-anchored camera-from-sphere-center vector. The
            // shader uses this as `oc` directly — it never
            // recomputes `camera.pos - cs_center`, so the shell
            // ray-march keeps sub-cell precision at any anchor
            // depth. The diff is bounded in f32 by the camera /
            // planet common-ancestor cell size, not by WORLD_SIZE.
            let oc = self.camera.position.offset_from(&planet.center_worldpos);
            renderer.set_cs_oc(oc);
        }

        renderer.update_camera(&self.camera.gpu_camera_at(cam_local, 1.2));
    }

    pub(super) fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
                renderer.set_cubed_sphere_highlight(None);
            }
            return;
        }
        let ray_dir = self.camera.forward();
        let camera_pos = self.camera.world_pos_f32();
        let tree_hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            camera_pos,
            ray_dir,
            self.edit_depth(),
        );
        let tree_t = tree_hit.as_ref().map(|h| h.t).unwrap_or(f32::INFINITY);

        let cs_depth = editing::cs_edit_depth(self.cs_planet.as_ref(), self.zoom_level());
        let cs_hit = self.cs_planet.as_ref().and_then(|p| {
            p.raycast(&self.world.library, camera_pos, ray_dir, cs_depth)
        });
        let cs_t = cs_hit.as_ref().map(|h| h.t).unwrap_or(f32::INFINITY);

        // Resolve the block AABB into frame-local coords before we
        // take the mutable renderer borrow — read-only `self`
        // access can't overlap the `&mut renderer`.
        let highlight_local = if cs_t < tree_t {
            None
        } else if let Some(aabb) = tree_hit.as_ref().map(edit::hit_aabb) {
            let (frame, _) = self.render_frame();
            let frame_scale = 3.0f32.powi(frame.depth() as i32);
            let mut frame_origin = [0.0f32; 3];
            let mut size = WORLD_SIZE;
            for k in 0..frame.depth() as usize {
                let (sx, sy, sz) = crate::world::tree::slot_coords(frame.slot(k) as usize);
                let child = size / 3.0;
                frame_origin[0] += sx as f32 * child;
                frame_origin[1] += sy as f32 * child;
                frame_origin[2] += sz as f32 * child;
                size = child;
            }
            let to_local = |w: [f32; 3]| [
                (w[0] - frame_origin[0]) * frame_scale,
                (w[1] - frame_origin[1]) * frame_scale,
                (w[2] - frame_origin[2]) * frame_scale,
            ];
            Some((to_local(aabb.0), to_local(aabb.1)))
        } else {
            None
        };

        let Some(renderer) = &mut self.renderer else { return };
        if cs_t < tree_t {
            renderer.set_highlight(None);
            if let Some(h) = cs_hit {
                renderer.set_cubed_sphere_highlight(Some((
                    h.face as u32, h.iu, h.iv, h.ir, h.depth,
                )));
            }
        } else {
            renderer.set_highlight(highlight_local);
            renderer.set_cubed_sphere_highlight(None);
        }
    }
}
