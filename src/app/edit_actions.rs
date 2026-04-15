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
    /// Rebuild `camera.position` at `new_depth` from a target world
    /// XYZ via `from_world_pos_in_tree`. Used by debug teleports and
    /// scroll-zoom — both want "same world spot, possibly different
    /// path depth."
    pub(super) fn reanchor_camera(&mut self, world: [f32; 3], new_depth: u8) {
        let clamped = [
            world[0].clamp(0.0, 3.0 - f32::EPSILON * 4.0),
            world[1].clamp(0.0, 3.0 - f32::EPSILON * 4.0),
            world[2].clamp(0.0, 3.0 - f32::EPSILON * 4.0),
        ];
        self.camera.position = crate::world::position::Position::from_world_pos_in_tree(
            clamped,
            new_depth,
            &self.world.library,
            self.world.root,
        );
    }

    /// Debug-mode teleport: jump exactly one cell of width
    /// `3^-camera.depth` along a world-frame direction.
    pub(super) fn debug_teleport(&mut self, dir: [f32; 3]) {
        let depth = self.camera.position.depth;
        let cell_size = 3.0f32.powi(-(depth as i32));
        let world = self.camera.position.pos_in_ancestor_frame_in_tree(
            0,
            &self.world.library,
            self.world.root,
        );
        let target = [
            world[0] + dir[0] * cell_size,
            world[1] + dir[1] * cell_size,
            world[2] + dir[2] * cell_size,
        ];
        eprintln!(
            "[teleport] dir={:?} from={:?} to={:?} cell_size={:.3e}",
            dir, world, target, cell_size,
        );
        self.reanchor_camera(target, depth);
        self.apply_zoom();
    }

    /// Reset to a known inspection pose at world (1.5, 2.0, 1.5)
    /// looking straight down. Bound to `R`.
    pub(super) fn debug_reset_pose(&mut self) {
        let depth = self.camera.position.depth;
        self.reanchor_camera([1.5, 2.0, 1.5], depth);
        self.camera.yaw = 0.0;
        self.camera.pitch = -std::f32::consts::FRAC_PI_2;
        self.camera.smoothed_up = [0.0, 1.0, 0.0];
        self.apply_zoom();
    }

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
    /// render root.
    ///
    /// Pinned `K` levels above the camera so the shader's `[0, 3)³`
    /// traversal volume stays in a small f32 magnitude regardless of
    /// camera depth — that's what removes the per-layer "absolute
    /// coords" jitter & precision wall.
    ///
    /// Capped at the body node's tree depth (`PLANET_BODY_DEPTH`)
    /// so the render root never descends INSIDE a face subtree.
    /// Inside a face subtree the shader can only see the immediate
    /// neighborhood of the camera (mostly empty above the SDF surface),
    /// so the planet would render as plain sky. Capping at body depth
    /// keeps the whole sphere in the packed subtree and lets the
    /// shader's `kinds[root]==1 → march_sphere_body` dispatch handle
    /// the entire planet from one render call.
    pub(super) fn render_root_depth(&self) -> u8 {
        const RENDER_FRAME_K: u8 = 3;
        const PLANET_BODY_DEPTH: u8 = 1;
        let desired = self.camera.position.depth.saturating_sub(RENDER_FRAME_K);
        desired.min(PLANET_BODY_DEPTH)
    }

    /// NodeId of the render root — walks `camera.position.path[..]`
    /// from the tree root via NodeKind-aware indexing (the camera's
    /// path was built by `from_world_pos_in_tree` so each slot
    /// resolves to the correct node at every level). Falls back to
    /// `world.root` if the walk hits a non-Node slot.
    pub(super) fn render_root_id(&self) -> crate::world::tree::NodeId {
        use crate::world::tree::Child;
        let target = self.render_root_depth() as usize;
        let mut id = self.world.root;
        for k in 0..target {
            let Some(node) = self.world.library.get(id) else { return self.world.root };
            let slot = self.camera.position.path[k] as usize;
            match node.children[slot] {
                Child::Node(child) => id = child,
                _ => return self.world.root,
            }
        }
        id
    }

    /// Camera XYZ in the render root's local `[0, 3)³` frame.
    /// NodeKind-aware: passing through a body ancestor reconstructs
    /// the cartesian XYZ inside the body cell via sphere geometry.
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
        let mut roots: Vec<u64> = vec![self.render_root_id()];
        if let Some(p) = self.cs_planet.as_ref() {
            roots.extend_from_slice(&p.face_roots);
        }
        let (tree_data, kinds_data, root_indices) = gpu::pack_tree_lod_multi(
            &self.world.library,
            &roots,
            self.camera_pos_in_render_frame(),
            1440.0,
            1.2,
        );
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &kinds_data, root_indices[0]);
            if self.cs_planet.is_some() {
                let face_roots: [u32; 6] = [
                    root_indices[1], root_indices[2], root_indices[3],
                    root_indices[4], root_indices[5], root_indices[6],
                ];
                renderer.set_face_roots(face_roots);
            }
        }
        // The `cs_planet` uniform is in WHATEVER frame the shader is
        // rendering in. With dynamic render frame the shader's
        // `[0, 3)³` traversal volume is the render-root cell, not
        // tree root, so the planet center / radii must be shifted
        // and scaled into that frame each frame.
        self.upload_cs_planet_in_render_frame();
    }

    /// Recompute `cs_planet` and `cs_params` uniforms in the render
    /// root's `[0, 3)³` frame and upload them. The shader's sphere
    /// DDA uses these values; if they don't match the camera's frame
    /// the ray and the sphere live in different worlds.
    fn upload_cs_planet_in_render_frame(&mut self) {
        let rrd = self.render_root_depth() as usize;
        let path = self.camera.position.path;
        let (planet_center, planet_inner, planet_outer, planet_depth) =
            match self.cs_planet.as_ref() {
                Some(p) => (p.center, p.inner_r, p.outer_r, p.depth),
                None => return,
            };
        let mut origin = [0.0f64; 3];
        let mut cell_size_world = 1.0f64;
        for k in 0..rrd {
            let (sx, sy, sz) = crate::world::tree::slot_coords(path[k] as usize);
            origin[0] += sx as f64 * cell_size_world;
            origin[1] += sy as f64 * cell_size_world;
            origin[2] += sz as f64 * cell_size_world;
            cell_size_world /= 3.0;
        }
        let render_extent_world = if rrd == 0 { 3.0 } else { cell_size_world * 3.0 };
        let scale = 3.0 / render_extent_world;
        let center_render = [
            ((planet_center[0] as f64 - origin[0]) * scale) as f32,
            ((planet_center[1] as f64 - origin[1]) * scale) as f32,
            ((planet_center[2] as f64 - origin[2]) * scale) as f32,
        ];
        let outer_render = (planet_outer as f64 * scale) as f32;
        let inner_render = (planet_inner as f64 * scale) as f32;
        if let Some(renderer) = &mut self.renderer {
            renderer.set_cubed_sphere_planet(center_render, inner_render, outer_render, planet_depth);
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
        let tree_hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            self.camera_pos_in_render_frame(),
            ray_dir,
            self.edit_depth(),
        );
        let tree_t = tree_hit.as_ref().map(|h| h.t).unwrap_or(f32::INFINITY);

        // Cubed-sphere cursor. Highlight depth comes from the same
        // `cs_edit_depth` the break path uses, so what you see is
        // exactly what you break.
        let cs_depth = editing::cs_edit_depth(self.cs_planet.as_ref(), self.zoom_level());
        let cs_hit = self.cs_planet.as_ref().and_then(|p| {
            p.raycast(&self.world.library, self.camera_pos_in_render_frame(), ray_dir, cs_depth)
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
