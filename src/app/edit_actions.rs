//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! These methods wrap the free-function helpers in
//! [`crate::editing`] with the engine state plumbing they need
//! (world reference, camera, zoom level, renderer handle). Most of
//! the actual logic lives in `editing`; this file is the glue layer
//! that owns the `&mut self` access pattern.

use crate::editing;
use crate::game_state::HotbarItem;
use crate::world::edit;
use crate::world::gpu;

use super::App;

impl App {
    /// CPU raycast depth: one less than the camera's anchor depth, so
    /// edits target the cell the camera is currently anchored in.
    pub(super) fn edit_depth(&self) -> u32 {
        self.anchor_depth().saturating_sub(1).max(1)
    }

    /// GPU visual depth: edit_depth + 3 (see 27×27×27 detail).
    pub(super) fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(16)
    }

    /// Sync renderer max_depth and camera uniforms after a zoom or
    /// worldgen change.
    pub(super) fn apply_zoom(&mut self) {
        self.ui.zoom_level = self.zoom_level();
        let vd = self.visual_depth();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&self.camera.gpu_camera(1.2));
        }
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}, anchor_depth: {}",
            self.zoom_level(),
            self.tree_depth as i32,
            self.edit_depth(),
            vd,
            self.anchor_depth(),
        );
    }

    pub(super) fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let edit_depth = self.edit_depth();
        let zoom_level = self.zoom_level();
        let camera_pos = self.camera.world_pos_f32();

        // Spherical-tree break: clear the targeted subtree if the
        // planet is hit closer than any Cartesian tree block.
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

    /// Re-pack and upload the tree with LOD culling based on camera position.
    /// Called every frame so distant terrain stays flattened as the camera moves.
    pub(super) fn upload_tree_lod(&mut self) {
        let cam_world = self.camera.world_pos_f32();
        let mut roots: Vec<u64> = vec![self.world.root];
        if let Some(p) = self.cs_planet.as_ref() {
            roots.extend_from_slice(&p.face_roots);
        }
        let (tree_data, root_indices) = gpu::pack_tree_lod_multi(
            &self.world.library,
            &roots,
            cam_world,
            1440.0,
            1.2,
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, root_indices[0]);
            if self.cs_planet.is_some() {
                let face_roots: [u32; 6] = [
                    root_indices[1], root_indices[2], root_indices[3],
                    root_indices[4], root_indices[5], root_indices[6],
                ];
                renderer.set_face_roots(face_roots);
            }
        }
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

        if let Some(renderer) = &mut self.renderer {
            if cs_t < tree_t {
                renderer.set_highlight(None);
                if let Some(h) = cs_hit {
                    renderer.set_cubed_sphere_highlight(Some((
                        h.face as u32, h.iu, h.iv, h.ir, h.depth,
                    )));
                }
            } else {
                renderer.set_highlight(tree_hit.as_ref().map(edit::hit_aabb));
                renderer.set_cubed_sphere_highlight(None);
            }
        }
    }
}
