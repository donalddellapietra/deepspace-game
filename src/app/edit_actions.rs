//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! These methods wrap the free-function helpers in
//! [`crate::editing`] with the engine state plumbing they need
//! (world reference, camera, zoom level, renderer handle). Most of
//! the actual logic lives in `editing`; this file is the glue layer
//! that owns the `&mut self` access pattern.

use crate::editing;
use crate::game_state::HotbarItem;
use crate::world::coords::ROOT_EXTENT;
use crate::world::edit;
use crate::world::gpu;
use crate::world::tree::{slot_coords, Child, NodeId};

use super::App;

/// Render-frame ancestor offset. Set to a saturating-large value so
/// the render root is ALWAYS the world root, regardless of camera
/// anchor depth — zooming changes the layer the player operates at
/// without shifting the rendered view.
pub(super) const RENDER_FRAME_K: u8 = 255;

impl App {
    /// CPU raycast depth: the camera's anchor depth. Zoom is no
    /// longer a separate "edit depth" knob — the raycast resolves at
    /// whatever layer the camera is anchored in.
    pub(super) fn edit_depth(&self) -> u32 {
        let d = self.camera.position.anchor.depth() as u32;
        d.max(1)
    }

    /// GPU visual depth: `edit_depth + 3` (the ~27×27×27 surround
    /// you can see from your cell).
    pub(super) fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(16)
    }

    /// Legacy zoom_level value exposed to the overlay / debug UI.
    /// Preserved so external consumers (the React overlay) keep
    /// rendering the old numeric scale: `tree_depth - anchor_depth`.
    pub(super) fn legacy_zoom_level(&self) -> i32 {
        (self.tree_depth as i32 - self.camera.position.anchor.depth() as i32).max(0)
    }

    /// Sync the renderer and overlay with the current anchor depth.
    /// Called after every zoom_in/zoom_out and after worldgen, so a
    /// single entry point owns the "depth changed — redraw" plumbing.
    pub(super) fn apply_zoom(&mut self) {
        // Clamp the camera's anchor depth into the instantiated
        // portion of the tree. If the player has zoomed past the
        // finest layer the tree has spelled out, pop back up.
        let max_depth = (self.tree_depth.saturating_sub(1)).max(1) as u8;
        while self.camera.position.anchor.depth() > max_depth {
            self.camera.position.zoom_out();
        }
        let zl = self.legacy_zoom_level();
        self.ui.zoom_level = zl;
        let vd = self.visual_depth();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&self.camera.gpu_camera(1.2));
        }
        log::info!(
            "Anchor depth: {}/{} (legacy zoom: {}), edit_depth: {}, visual: {}",
            self.camera.position.anchor.depth(),
            self.tree_depth,
            zl,
            self.edit_depth(),
            vd
        );
    }

    pub(super) fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let edit_depth = self.edit_depth();
        let camera_pos = crate::world::coords::world_pos_to_f32(&self.camera.position);

        // Spherical-tree break: clear the targeted subtree if the
        // planet is hit closer than any Cartesian tree block.
        if editing::try_cs_break(
            &mut self.world,
            self.cs_planet.as_mut(),
            &self.body_anchor,
            camera_pos,
            ray_dir,
            edit_depth,
        ) {
            return;
        }

        let hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            crate::world::coords::world_pos_to_f32(&self.camera.position),
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
        let camera_pos = crate::world::coords::world_pos_to_f32(&self.camera.position);

        // Spherical place: fill the cell adjacent to the first solid
        // cell with the active hotbar block. Meshes fall through to
        // the Cartesian tree placer below.
        if let Some(block_type) = self.ui.active_block_type() {
            if editing::try_cs_place(
                &mut self.world,
                self.cs_planet.as_mut(),
                &self.body_anchor,
                camera_pos,
                ray_dir,
                edit_depth,
                block_type,
            ) {
                return;
            }
        }

        let hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            crate::world::coords::world_pos_to_f32(&self.camera.position),
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

    /// Compute the render frame for the current camera: the
    /// ancestor Path at `depth - K`, its root `NodeId`, its world-
    /// space min corner, and the world width of one root-cell.
    ///
    /// If the render-frame path resolves to a non-Node child (an
    /// uninstantiated uniform region the content-addressed tree
    /// hasn't spelled out at that depth), falls back to the world
    /// root. The world-generated empty tree is fully instantiated so
    /// this fallback only fires on partially-built worlds.
    pub(super) fn render_frame(&self) -> ([f32; 3], f32, NodeId) {
        let cam = &self.camera.position;
        let cam_depth = cam.anchor.depth();
        let rf_depth = cam_depth.saturating_sub(RENDER_FRAME_K);
        // Walk from world.root down rf_depth slots, tracking world
        // origin. Root cells are 1.0 wide (span [0, ROOT_EXTENT)).
        let mut node_id = self.world.root;
        let mut origin = [0.0f32; 3];
        let mut cell_size = 1.0f32;
        for i in 0..rf_depth as usize {
            let slot = cam.anchor.slots()[i];
            let Some(node) = self.world.library.get(node_id) else {
                return (origin, cell_size, self.world.root);
            };
            match node.children[slot as usize] {
                Child::Node(child_id) => {
                    let (sx, sy, sz) = slot_coords(slot as usize);
                    origin[0] += sx as f32 * cell_size;
                    origin[1] += sy as f32 * cell_size;
                    origin[2] += sz as f32 * cell_size;
                    cell_size /= 3.0;
                    node_id = child_id;
                }
                _ => {
                    // Ancestor isn't instantiated; render from world
                    // root. Shouldn't happen for the generated empty
                    // tree, which is uniformly subdivided.
                    return ([0.0; 3], 1.0, self.world.root);
                }
            }
        }
        // After walking `rf_depth` levels, `cell_size` is the width
        // of one cell AT that depth = the width of one root-cell of
        // the render-frame node. `origin` is its world-space min
        // corner. The render-frame node itself spans
        // `[origin, origin + 3 * cell_size) = ROOT_EXTENT / 3^rf_depth`.
        let _ = ROOT_EXTENT; // ROOT_EXTENT is baked into the numbers above.
        (origin, cell_size, node_id)
    }

    /// Walk `body_anchor` from `world.root` and confirm a
    /// `CubedSphereBody` still lives at the end. Updates
    /// `cs_planet.body_node` to match. Returns the body's current
    /// `NodeId`, or `None` if the anchor no longer leads to a body
    /// (e.g., a Cartesian edit overwrote it).
    fn refresh_body_node(&mut self) -> Option<crate::world::tree::NodeId> {
        use crate::world::tree::{Child, NodeKind};
        let Some(planet) = self.cs_planet.as_mut() else { return None; };
        let mut id = self.world.root;
        for &slot in self.body_anchor.slots() {
            let node = self.world.library.get(id)?;
            match node.children[slot as usize] {
                Child::Node(child) => id = child,
                _ => return None,
            }
        }
        let node = self.world.library.get(id)?;
        if !matches!(node.kind, NodeKind::CubedSphereBody { .. }) {
            return None;
        }
        planet.body_node = id;
        Some(id)
    }

    /// Re-pack and upload the tree with LOD culling based on camera position.
    /// Called every frame so distant terrain stays flattened as the camera moves.
    pub(super) fn upload_tree_lod(&mut self) {
        let (rf_origin, rf_cell, rf_node) = self.render_frame();

        // Re-resolve the body from `body_anchor` each frame. The
        // walk catches cases where a Cartesian edit has replaced the
        // body cell — stale ids can't outlive an edit and sneak into
        // the packer. `self.cs_planet.body_node` is refreshed to
        // match the current tree state.
        let body_node = self.refresh_body_node();
        let mut roots: Vec<u64> = vec![rf_node];
        if let Some(b) = body_node { roots.push(b); }

        let (tree_data, tree_metas, root_indices) = gpu::pack_tree_lod_multi_with_frame(
            &self.world.library,
            &roots,
            crate::world::coords::world_pos_to_f32(&self.camera.position),
            1440.0,
            1.2,
            rf_origin,
            rf_cell,
        );

        // Derive the body's world frame from its anchor path, so the
        // shader doesn't have to re-walk the tree. Body occupies one
        // cell at `body_anchor.depth()` in world units.
        let (body_world, body_idx) = if body_node.is_some() {
            use crate::world::coords::{world_pos_to_f32, WorldPos, ROOT_EXTENT};
            let cube_w = ROOT_EXTENT / 3f32.powi(self.body_anchor.depth() as i32);
            let origin = world_pos_to_f32(&WorldPos {
                anchor: self.body_anchor,
                offset: [0.0, 0.0, 0.0],
            });
            ([origin[0], origin[1], origin[2], cube_w], root_indices[1])
        } else {
            ([0.0; 4], u32::MAX)
        };

        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &tree_metas, root_indices[0]);
            renderer.set_render_frame(rf_origin, rf_cell);
            renderer.set_body(body_world, body_idx);
        }
    }

    pub(super) fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
                renderer.set_body_highlight(None);
            }
            return;
        }
        let ray_dir = self.camera.forward();
        let tree_hit = edit::cpu_raycast(
            &self.world.library,
            self.world.root,
            crate::world::coords::world_pos_to_f32(&self.camera.position),
            ray_dir,
            self.edit_depth(),
        );
        let tree_t = tree_hit.as_ref().map(|h| h.t).unwrap_or(f32::INFINITY);

        // Cubed-sphere cursor. Highlight depth comes from the same
        // `cs_edit_depth` the break path uses, so what you see is
        // exactly what you break.
        let cs_depth = editing::cs_edit_depth(self.cs_planet.as_ref(), self.edit_depth());
        let cs_hit = self.cs_planet.as_ref().and_then(|p| {
            p.raycast(&self.world.library, crate::world::coords::world_pos_to_f32(&self.camera.position), ray_dir, cs_depth)
        });
        let cs_t = cs_hit.as_ref().map(|h| h.t).unwrap_or(f32::INFINITY);

        if let Some(renderer) = &mut self.renderer {
            if cs_t < tree_t {
                renderer.set_highlight(None);
                if let Some(h) = cs_hit {
                    renderer.set_body_highlight(Some((
                        h.face as u32, h.iu, h.iv, h.ir, h.depth,
                    )));
                }
            } else {
                renderer.set_highlight(tree_hit.as_ref().map(edit::hit_aabb));
                renderer.set_body_highlight(None);
            }
        }
    }
}
