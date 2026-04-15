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
    /// user-visible cell granularity. Formula: anchor_depth - 4
    /// (4 layers of headroom between visible scale and edit
    /// scale). Lower bound 1 (can't edit at the body root).
    ///
    /// Upper bound is now `MAX_DEPTH` rather than the historical
    /// 14, because:
    ///
    /// - The face DDA boundaries are Kahan-compensated (precision
    ///   ~1 ULP regardless of depth).
    /// - The CPU raycast runs in frame-local coords with ribbon
    ///   ascent (cell precision bounded by frame depth, not
    ///   absolute path).
    /// - The shader's ribbon pop keeps `ray_dir` unit so the
    ///   1e-8 parallel-axis check no longer underflows.
    pub(super) fn cs_edit_depth(&self) -> u32 {
        ((self.anchor_depth() as i32) - 4)
            .clamp(1, crate::world::tree::MAX_DEPTH as i32) as u32
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

    /// Cast a ray from the camera into the world using the same
    /// frame-aware machinery as the renderer: the cpu raycast
    /// runs in frame-local coordinates and pops upward via the
    /// camera's anchor when it exits the frame's bubble. This is
    /// what makes deep-zoom block placement land in the cell
    /// that's actually under the crosshair, instead of being
    /// pinned to the f32-precision wall of world XYZ.
    pub(super) fn frame_aware_raycast(&self) -> Option<edit::HitInfo> {
        let ray_dir = self.camera.forward();
        let (frame, _) = self.render_frame();
        let cam_local = self.camera.position.in_frame(&frame);
        edit::cpu_raycast_in_frame(
            &self.world.library, self.world.root,
            frame.as_slice(), cam_local, ray_dir,
            self.edit_depth(), self.cs_edit_depth(),
        )
    }

    pub(super) fn do_break(&mut self) {
        let hit = self.frame_aware_raycast();
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
        let hit = self.frame_aware_raycast();
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

    /// Pack the world tree (LOD-aware from the absolute root) and
    /// push it to the GPU, along with the ancestor ribbon that
    /// lets the shader pop from the frame back up to the absolute
    /// root.
    ///
    /// Pack runs in **world** coordinates (camera passed as world
    /// XYZ), so distance-LOD decisions are at world scale and the
    /// buffer is shared by all frame depths. The shader starts
    /// DDA at `frame_root_idx` with the camera in **frame-local**
    /// coordinates, and pops via the ribbon when rays exit the
    /// frame's `[0, 3)³` bubble.
    pub(super) fn upload_tree_lod(&mut self) {
        let (intended_frame, _frame_root_id) = self.render_frame();
        let cam_world = self.camera.world_pos_f32();
        // Preserve the intended frame path through the pack so
        // build_ribbon can walk it. Without this, uniform-empty
        // Cartesian siblings on the camera's path get flattened
        // and the ribbon stops at world.root — defeating frame
        // descent and pinning camera precision regardless of zoom.
        let (tree_data, node_kinds, _world_root_idx) = gpu::pack_tree_lod_preserving(
            &self.world.library, self.world.root, cam_world, 1440.0, 1.2,
            intended_frame.as_slice(),
        );
        // build_ribbon may stop short of the intended frame when
        // pack LOD-flattened a sibling on the way down (uniform-
        // empty Cartesian children become tag=0 leaves). The
        // shader can only operate at the depth the buffer
        // actually reached, so we recompute cam_local against the
        // truncated path.
        let r = gpu::build_ribbon(&tree_data, intended_frame.as_slice());
        let mut effective_frame = crate::world::anchor::Path::root();
        for &slot in &r.reached_slots {
            effective_frame.push(slot);
        }
        let cam_local = self.camera.position.in_frame(&effective_frame);
        // Frame kind depends on the EFFECTIVE frame, not the
        // intended one.
        let frame_kind = self.world.library
            .get(self.frame_root_id_for(&effective_frame))
            .map(|n| n.kind)
            .unwrap_or(crate::world::tree::NodeKind::Cartesian);
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &node_kinds, r.frame_root_idx);
            renderer.update_ribbon(&r.ribbon);
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

    /// Walk the world tree from world.root following `path`
    /// returning the NodeId reached. Used by upload_tree_lod to
    /// look up the *effective* frame's NodeKind after build_ribbon
    /// truncated.
    fn frame_root_id_for(&self, path: &crate::world::anchor::Path) -> crate::world::tree::NodeId {
        let mut node = self.world.root;
        for k in 0..path.depth() as usize {
            let Some(n) = self.world.library.get(node) else { break };
            let slot = path.slot(k) as usize;
            match n.children[slot] {
                crate::world::tree::Child::Node(child) => { node = child; }
                _ => break,
            }
        }
        node
    }

    pub(super) fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let tree_hit = self.frame_aware_raycast();
        if let Some(h) = &tree_hit {
            eprintln!(
                "highlight tree_hit: path.len={} face={} t={} place_path.len={:?} edit_depth={} cs_edit_depth={}",
                h.path.len(), h.face, h.t,
                h.place_path.as_ref().map(|p| p.len()),
                self.edit_depth(), self.cs_edit_depth(),
            );
        }
        let aabb_world = tree_hit.as_ref().map(|h| edit::hit_aabb(&self.world.library, h));
        // Transform AABB from world coords to frame-local coords.
        // Shader expects highlight in the same frame as `camera.pos`.
        let (frame, _) = self.render_frame();
        let aabb = aabb_world.map(|(mn, mx)| super::aabb_world_to_frame(&frame, mn, mx));
        if let Some((mn, mx)) = &aabb {
            eprintln!("highlight (frame-local): min={:?} max={:?} size={:?}",
                mn, mx, [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]]);
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
    }
}

