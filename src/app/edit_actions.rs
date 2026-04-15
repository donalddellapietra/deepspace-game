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

    /// Face-subtree depth at which sphere edits land. Picks a
    /// user-visible cell granularity matching the UI's layer
    /// readout: placing at "Layer N" produces a block whose cell
    /// size is exactly 3^N cells per face axis. The previous
    /// `anchor - 4` formula was off by one — placements at Layer
    /// N landed at face-subtree depth N-1 ("Layer N+1 block").
    ///
    /// No numerical depth cap. Any ceiling would be an admission
    /// the shader's rendering doesn't honor path-anchored
    /// precision (camera in root-scale f32 breaks at anchor ~15,
    /// `cells_d = pow(3, depth)` overflows f32-integer-exact at
    /// depth 16). Those are shader bugs to fix, not budgets to
    /// ration. The lower bound of 1 is the only actual content
    /// constraint — depth 0 would place above the face root.
    pub(super) fn cs_edit_depth(&self) -> u32 {
        ((self.anchor_depth() as i32) - 3).max(1) as u32
    }

    pub(super) fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(16)
    }

    pub fn apply_zoom(&mut self) {
        self.ui.zoom_level = self.zoom_level();
        let vd = self.visual_depth();
        let world_pos = self.camera.world_pos_f32();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&self.camera.gpu_camera_at(world_pos, 1.2));
        }
        // Rebuild the ribbon + tree pack from the new anchor.
        self.upload_tree_lod();
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}, anchor_depth: {}",
            self.zoom_level(), self.tree_depth as i32,
            self.edit_depth(), vd, self.anchor_depth(),
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

    /// Pack the tree from world root into one buffer and build the
    /// ribbon uniform. Every ribbon frame's node is kept walkable by
    /// passing the camera's anchor as the packer's `preserve_path`,
    /// so LOD flattening can't strand a frame root.
    pub(super) fn upload_tree_lod(&mut self) {
        let ribbon = self.render_ribbon();
        let world_pos = self.camera.world_pos_f32();
        let anchor = self.camera.position.anchor;
        let (tree_data, node_kinds, _world_root_idx, visited) = gpu::pack_tree_lod(
            &self.world.library, self.world.root,
            world_pos, 1440.0, 1.2,
            &anchor,
        );

        // Compute the planet uniform. Walks `planet_path` to find the
        // current body node (paths persist across edits via content-
        // addressed rebuild, but the NodeId at the end of the path
        // changes each time the subtree mutates — look it up fresh).
        let planet_gpu = self.compute_planet_uniform(&visited);
        // Map each ribbon frame's NodeId to its buffer index. If a
        // frame's node wasn't reachable (shouldn't happen with the
        // preserve_path pass, but defensive), fall back to the root.
        let mut gpu_ribbon: Vec<gpu::GpuRibbonFrame> = Vec::with_capacity(ribbon.len());
        for f in &ribbon {
            let root_index = visited.get(&f.node_id).copied().unwrap_or(0);
            gpu_ribbon.push(gpu::GpuRibbonFrame {
                root_index,
                _pad0: 0,
                world_scale: f.world_scale,
                _pad1: 0,
                camera_local: [f.camera_local[0], f.camera_local[1], f.camera_local[2], 0.0],
            });
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &node_kinds, &gpu_ribbon);
            renderer.set_planet(planet_gpu);
            renderer.update_camera(&self.camera.gpu_camera_at(world_pos, 1.2));
        }
    }

    /// Compute the planet uniform from the current world state and
    /// camera. Uses path-anchored `offset_from` so `oc_world` stays
    /// bounded (magnitude ≤ body cell size in world units)
    /// regardless of the camera's anchor depth — this is the
    /// precision-safe replacement for passing the world-scale
    /// camera/center through a frame-local scaling.
    ///
    /// `active=0` is returned when the planet isn't reachable via
    /// `planet_path` (e.g., the planet hasn't been installed yet or
    /// the pack truncated before the body) — the shader skips its
    /// sphere pass entirely.
    fn compute_planet_uniform(
        &self,
        visited: &std::collections::HashMap<crate::world::tree::NodeId, u32>,
    ) -> gpu::GpuPlanet {
        use crate::world::anchor::WorldPos;
        use crate::world::tree::{Child, NodeKind};

        // Walk planet_path to find the body NodeId.
        let mut node_id = self.world.root;
        for k in 0..self.planet_path.depth() as usize {
            let Some(node) = self.world.library.get(node_id) else {
                return gpu::GpuPlanet::default();
            };
            let slot = self.planet_path.slot(k) as usize;
            match node.children[slot] {
                Child::Node(cid) => node_id = cid,
                _ => return gpu::GpuPlanet::default(),
            }
        }
        let body_id = node_id;
        let Some(body_node) = self.world.library.get(body_id) else {
            return gpu::GpuPlanet::default();
        };
        let (inner_r_local, outer_r_local) = match body_node.kind {
            NodeKind::CubedSphereBody { inner_r, outer_r } => (inner_r, outer_r),
            _ => return gpu::GpuPlanet::default(),
        };

        // Body cell size in world units. At depth 1, cell_size =
        // WORLD_SIZE / 3 = 1.0. More generally: WORLD_SIZE / 3^depth.
        let body_depth = self.planet_path.depth();
        let body_cell_size = crate::world::anchor::WORLD_SIZE
            / (3.0_f32).powi(body_depth as i32);
        let inner_r_world = inner_r_local * body_cell_size;
        let outer_r_world = outer_r_local * body_cell_size;

        // Body center in WorldPos form, then path-anchored offset
        // from the camera. `offset_from` composes positions in their
        // common ancestor's frame — the body node IS the common
        // ancestor when the camera is inside the body's subtree, so
        // precision is bounded by body_cell_size * 1e-7 in that case.
        let body_center = WorldPos::new(self.planet_path, [0.5, 0.5, 0.5]);
        let oc = self.camera.position.offset_from(&body_center);

        // Walker's max_term_depth cap for sphere boundary math. The
        // sphere DDA floors cell_eps at the cap so ray advancement
        // stays representable in f32 regardless of content depth;
        // deeper edits still register as hits via the walker's per-
        // sample block lookup. Value chosen empirically: 3^12 ≈ 5e5
        // keeps `cells_d` integer-exact in f32 and cell-eps
        // advancement above f32 ULP at typical ray distances.
        let max_term_depth = 12;

        let body_node_index = visited.get(&body_id).copied().unwrap_or(u32::MAX);
        if body_node_index == u32::MAX {
            return gpu::GpuPlanet::default();
        }

        gpu::GpuPlanet {
            enabled: 1,
            body_node_index,
            inner_r_world,
            outer_r_world,
            oc_world: [oc[0], oc[1], oc[2], 0.0],
            max_term_depth,
            _pad: [0; 3],
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
        let aabb = tree_hit.as_ref().map(|h| edit::hit_aabb(&self.world.library, h));
        if let Some((mn, mx)) = &aabb {
            eprintln!("highlight: min={:?} max={:?} size={:?}",
                mn, mx, [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]]);
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
    }
}
