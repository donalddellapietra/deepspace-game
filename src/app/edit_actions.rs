//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified `edit::cpu_raycast` →
//! `break_block` / `place_block` pipeline. The planet is part of
//! the same tree, so there's no longer a sphere-specific code
//! path here. (CPU raycast traversal of `CubedSphereBody` cells
//! is a follow-up; for now hits inside the body cell are treated
//! as Cartesian, which means edits on the planet's interior
//! aren't yet supported. Rendering of the planet works.)

use crate::game_state::HotbarItem;
use crate::world::edit;
use crate::world::gpu;

use super::App;

impl App {
    pub(super) fn edit_depth(&self) -> u32 {
        self.anchor_depth().saturating_sub(1).max(1)
    }

    /// Face-subtree depth at which sphere edits land — picks a
    /// user-visible cell granularity instead of the planet's
    /// deepest sub-pixel resolution. Without this cap the walker
    /// descends all 20 face-subtree levels and breaks/places
    /// modify cells smaller than a pixel — they succeed but are
    /// invisible. Mirrors the old `cs_edit_depth = 4 + (15 -
    /// zoom_level)` formula, clamped to `[1, 14]`.
    ///
    /// The 14 cap is tactical and disappears with the unified
    /// driver refactor (camera-precision via virtual-root + face
    /// DDA in cell-local frames).
    pub(super) fn cs_edit_depth(&self) -> u32 {
        ((self.anchor_depth() as i32) - 4).clamp(1, 14) as u32
    }

    pub(super) fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(16)
    }

    pub fn apply_zoom(&mut self) {
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
            self.zoom_level(), self.tree_depth as i32,
            self.edit_depth(), vd, self.anchor_depth(), frame.depth(),
        );
    }

    pub(super) fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let camera_pos = self.camera.world_pos_f32();
        let hit = edit::cpu_raycast_with_face_depth(
            &self.world.library, self.world.root,
            camera_pos, ray_dir, self.edit_depth(),
            self.cs_edit_depth(),
        );
        eprintln!("do_break: hit={:?}",
            hit.as_ref().map(|h| (h.path.len(), h.face, h.t)));
        let Some(hit) = hit else { return };

        if self.save_mode {
            use crate::world::tree::Child;
            let mut saved_id = None;
            if let Some(&(parent_id, slot)) = hit.path.last() {
                if let Some(node) = self.world.library.get(parent_id) {
                    match node.children[slot] {
                        Child::Node(child_id) => saved_id = Some(child_id),
                        Child::Block(_) | Child::Empty => saved_id = Some(parent_id),
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

        let changed = edit::break_block(&mut self.world, &hit);
        eprintln!("do_break: break_block returned {}", changed);
        if changed {
            self.upload_tree();
        }
    }

    pub(super) fn do_place(&mut self) {
        let ray_dir = self.camera.forward();
        let camera_pos = self.camera.world_pos_f32();
        let hit = edit::cpu_raycast_with_face_depth(
            &self.world.library, self.world.root,
            camera_pos, ray_dir, self.edit_depth(),
            self.cs_edit_depth(),
        );
        eprintln!("do_place: hit={:?}",
            hit.as_ref().map(|h| (h.path.len(), h.face, h.t)));
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
                    &mut self.world, &hit,
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

    /// Pack the tree subtree at the current render frame and push
    /// it (along with the parallel `node_kinds` buffer) to the GPU.
    /// Also updates the renderer's root_kind dispatch so the shader
    /// knows whether to enter Cartesian DDA or sphere DDA.
    pub(super) fn upload_tree_lod(&mut self) {
        let (frame, frame_root) = self.render_frame();
        let cam_local = self.camera.position.in_frame(&frame);
        let (tree_data, node_kinds, root_index) = gpu::pack_tree_lod(
            &self.world.library, frame_root, cam_local, 1440.0, 1.2,
        );
        let frame_kind = self.render_frame_kind();
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &node_kinds, root_index);
            renderer.update_camera(&self.camera.gpu_camera_at(cam_local, 1.2));
            match frame_kind {
                crate::world::tree::NodeKind::CubedSphereBody { inner_r, outer_r } => {
                    renderer.set_root_kind_body(inner_r, outer_r);
                }
                _ => {
                    renderer.set_root_kind_cartesian();
                }
            }
        }
    }

    pub(super) fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let ray_dir = self.camera.forward();
        let camera_pos = self.camera.world_pos_f32();
        let tree_hit = edit::cpu_raycast_with_face_depth(
            &self.world.library, self.world.root,
            camera_pos, ray_dir, self.edit_depth(),
            self.cs_edit_depth(),
        );
        let aabb_world = tree_hit.as_ref().map(|h| edit::hit_aabb(&self.world.library, h));
        // Transform AABB from world coords to frame-local coords.
        // Shader expects highlight in the same frame as `camera.pos`.
        let (frame, _) = self.render_frame();
        let (frame_origin, frame_size) = frame_origin_size_world(&frame);
        let scale = crate::world::anchor::WORLD_SIZE / frame_size;
        let aabb = aabb_world.map(|(mn, mx)| {
            (
                [
                    (mn[0] - frame_origin[0]) * scale,
                    (mn[1] - frame_origin[1]) * scale,
                    (mn[2] - frame_origin[2]) * scale,
                ],
                [
                    (mx[0] - frame_origin[0]) * scale,
                    (mx[1] - frame_origin[1]) * scale,
                    (mx[2] - frame_origin[2]) * scale,
                ],
            )
        });
        if let Some((mn, mx)) = &aabb {
            eprintln!("highlight (frame-local): min={:?} max={:?} size={:?}",
                mn, mx, [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]]);
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
    }
}

/// World-space origin and size of the cell at `path`. Origin is the
/// world XYZ of the cell's `(0, 0, 0)` corner; size is its side
/// length in world units.
fn frame_origin_size_world(path: &crate::world::anchor::Path) -> ([f32; 3], f32) {
    use crate::world::anchor::WORLD_SIZE;
    use crate::world::tree::slot_coords;
    let mut origin = [0.0f32; 3];
    let mut size = WORLD_SIZE;
    for k in 0..path.depth() as usize {
        let (sx, sy, sz) = slot_coords(path.slot(k) as usize);
        let child = size / 3.0;
        origin[0] += sx as f32 * child;
        origin[1] += sy as f32 * child;
        origin[2] += sz as f32 * child;
        size = child;
    }
    (origin, size)
}
