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
    /// CPU raycast depth: how far below the render root to descend
    /// when testing the cursor ray. Derived from the camera's path
    /// depth (the "zoom anchor") minus the render-root depth. Under
    /// the path-based model introduced in step 7, zoom IS
    /// `camera.position.depth`.
    pub(super) fn edit_depth(&self) -> u32 {
        let cd = self.camera.position.depth as i32;
        let rrd = self.render_root_depth() as i32;
        (cd - rrd).max(1) as u32
    }

    /// Legacy "zoom_level" = `tree_depth - edit_depth`. Kept as a
    /// getter for UI/overlay code that still reports the old number
    /// (0 = finest, higher = coarser). Step 10 retires the UI field.
    pub(super) fn zoom_level(&self) -> i32 {
        self.tree_depth as i32 - self.edit_depth() as i32
    }

    /// How many leading slots of the camera's path sit above the
    /// render root. Pinned `RENDER_FRAME_K` levels above the camera
    /// so the shader's `[0, 3)³` traversal volume stays a small f32
    /// magnitude regardless of camera depth — which is what
    /// eliminates the per-layer absolute-coords jitter.
    pub(super) fn render_root_depth(&self) -> u8 {
        const RENDER_FRAME_K: u8 = 3;
        self.camera.position.depth.saturating_sub(RENDER_FRAME_K)
    }

    /// NodeId of the render root — walks `camera.position.path[..]`
    /// from the tree root. Because the path was built via
    /// `Position::from_world_pos_in_tree` (NodeKind-aware), each
    /// slot along the walk resolves to the correct tree node even
    /// where the path threads through body/face subtrees. Falls back
    /// to the tree root if the walk hits a non-Node slot.
    pub(super) fn render_root_id(&self) -> crate::world::tree::NodeId {
        use crate::world::tree::Child;
        let target = self.render_root_depth() as usize;
        let mut id = self.world.root;
        for k in 0..target {
            let Some(node) = self.world.library.get(id) else {
                return self.world.root;
            };
            let slot = self.camera.position.path[k] as usize;
            match node.children[slot] {
                Child::Node(child) => id = child,
                _ => return self.world.root,
            }
        }
        id
    }

    /// Camera XYZ in the render root's local `[0, 3)³` frame.
    /// NodeKind-aware: crossing a body ancestor reconstructs the
    /// cartesian XYZ inside the body cell via sphere geometry.
    pub(super) fn camera_pos_in_render_frame(&self) -> [f32; 3] {
        self.camera.position.pos_in_ancestor_frame_in_tree(
            self.render_root_depth(),
            &self.world.library,
            self.world.root,
        )
    }

    /// GPU visual depth: edit_depth + 3 (see 27×27×27 detail).
    pub(super) fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(16)
    }

    /// Sync GPU max depth + camera uniform with the current camera
    /// depth. Called after any zoom change.
    pub(super) fn apply_zoom(&mut self) {
        let td = self.tree_depth as u8;
        // Clamp camera depth into [1, tree_depth] via zoom_in/out.
        // Step 7 puts zoom behind Position — the camera can't end up
        // at an invalid depth through normal input, but defensive
        // clamping covers edge cases (e.g. tree shrank after edits).
        while self.camera.position.depth < 1 {
            self.camera.position.zoom_in();
        }
        while self.camera.position.depth > td && td > 0 {
            self.camera.position.zoom_out();
        }
        let vd = self.visual_depth();
        self.ui.zoom_level = self.zoom_level();
        let pos = self.camera_pos_in_render_frame();
        let gpu_cam = self.camera.gpu_camera(1.2, pos);
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&gpu_cam);
        }
        log::info!(
            "Camera depth: {}/{} (zoom_level {}), edit: {}, visual: {}",
            self.camera.position.depth,
            td,
            self.zoom_level(),
            self.edit_depth(),
            vd,
        );
    }

    pub(super) fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let edit_depth = self.edit_depth();
        let zoom_level = self.zoom_level();
        let camera_pos = self.camera_pos_in_render_frame();

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
            self.camera_pos_in_render_frame(),
            ray_dir,
            self.edit_depth(),
        );
        let Some(hit) = hit else { return };

        if self.save_mode {
            // Save mode: capture the subtree under the crosshair.
            // The hit path gives us (parent_id, slot) pairs from root.
            // We want to save the deepest Node in the path — that's
            // the natural "block" at the current zoom level.
            //
            // If the hit child is a Node, save it directly.
            // If it's a Block terminal, go one level up and save the
            // parent node (which contains this block as a child).
            use crate::world::tree::Child;
            let mut saved_id = None;
            if let Some(&(parent_id, slot)) = hit.path.last() {
                if let Some(node) = self.world.library.get(parent_id) {
                    match node.children[slot] {
                        Child::Node(child_id) => saved_id = Some(child_id),
                        Child::Block(_) | Child::Empty => {
                            // Hit a terminal — save the parent node instead.
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
        let camera_pos = self.camera_pos_in_render_frame();

        // Spherical place: fill the cell adjacent to the first solid
        // cell with the active hotbar block. Meshes fall through to
        // the Cartesian tree placer below.
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
            self.camera_pos_in_render_frame(),
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
                // Place the subtree adjacent to the hit face, same as blocks.
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
        // Single unified tree — the sphere's face subtrees live as
        // children of the body node in the tree, so one BFS packs
        // them all. No separate face-root uploads any more.
        let (tree_data, kinds_data, root_idx) = gpu::pack_tree_lod(
            &self.world.library,
            self.render_root_id(),
            self.camera_pos_in_render_frame(),
            1440.0,
            1.2,
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &kinds_data, root_idx);
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
        let tree_hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            self.camera_pos_in_render_frame(),
            ray_dir,
            self.edit_depth(),
        );

        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(tree_hit.as_ref().map(edit::hit_aabb));
        }
    }
}
