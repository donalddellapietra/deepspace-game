//! `do_break` / `do_place`: cursor-driven edits that feed the hit
//! into `world::edit` and re-upload on success.
//!
//! The cursor raycast returns either a world hit (existing behavior)
//! or an entity hit. Entity edits bypass the world tree entirely —
//! they clone-on-write the entity's subtree via
//! `propagate_edit_on_library` and stash the new root in
//! `entity.override_root`. The world root is untouched; only the
//! per-entity NodeId changes. 10k identical NPCs stay deduped
//! except for the one whose voxel you just damaged.

use crate::game_state::HotbarItem;
use crate::world::anchor::Path;
use crate::world::edit;
use crate::world::raycast::HitInfo;
use crate::world::tree::Child;

use super::CursorHit;
use crate::app::App;

impl App {
    pub(in crate::app) fn do_break(&mut self) {
        let Some(hit) = self.cursor_raycast() else {
            eprintln!("do_break: no hit");
            return;
        };
        match hit {
            CursorHit::World(h) => self.do_break_world(h),
            CursorHit::Entity { entity_idx, inner } => {
                self.do_break_entity(entity_idx, inner);
            }
        }
    }

    pub(in crate::app) fn do_place(&mut self) {
        let Some(hit) = self.cursor_raycast() else {
            eprintln!("do_place: no hit");
            return;
        };
        match hit {
            CursorHit::World(h) => self.do_place_world(h),
            CursorHit::Entity { entity_idx, inner } => {
                // Placement into an entity: for v1 we support "break"
                // (click = turn voxel to air). Placement into entities
                // is a follow-up — it needs a face-direction notion
                // that works inside the entity's local frame.
                eprintln!(
                    "do_place_entity: not implemented yet, entity_idx={} path_len={}",
                    entity_idx, inner.path.len(),
                );
            }
        }
    }

    fn do_break_entity(&mut self, entity_idx: u32, inner: HitInfo) {
        eprintln!(
            "do_break_entity: entity_idx={} path_len={} face={}",
            entity_idx, inner.path.len(), inner.face,
        );
        if inner.path.is_empty() {
            return;
        }
        let new_root = edit::propagate_edit_on_library(
            &mut self.world.library,
            &inner.path,
            Child::Empty,
        );
        let Some(new_root) = new_root else { return };
        self.entities
            .set_override(&mut self.world.library, entity_idx, new_root);
        self.upload_tree();
    }

    fn do_break_world(&mut self, hit: HitInfo) {
        eprintln!(
            "do_break: hit path_len={} face={} place_path_len={:?}",
            hit.path.len(),
            hit.face,
            hit.place_path.as_ref().map(|p| p.len()),
        );

        if self.save_mode {
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

        // Store the edit path's slot sequence so upload_tree_lod can
        // preserve it, making the fine edit visible in the packed tree
        // even when the camera is far enough that LOD would collapse it.
        let mut edit_slots = Path::root();
        for &(_, slot) in &hit.path {
            edit_slots.push(slot as u8);
        }
        self.last_edit_slots = Some(edit_slots);

        let changed = edit::break_block(&mut self.world, &hit);
        eprintln!("do_break: changed={changed}");
        self.harness_emit_edit("broke", &hit, changed);
        if changed {
            self.upload_tree();
        }
    }

    fn do_place_world(&mut self, hit: HitInfo) {
        eprintln!(
            "do_place: hit path_len={} face={} place_path_len={:?} active_slot={}",
            hit.path.len(),
            hit.face,
            hit.place_path.as_ref().map(|p| p.len()),
            self.ui.active_slot,
        );

        let mut edit_slots = Path::root();
        for &(_, slot) in &hit.path {
            edit_slots.push(slot as u8);
        }
        self.last_edit_slots = Some(edit_slots);

        match &self.ui.slots[self.ui.active_slot] {
            HotbarItem::Block(block_type) => {
                let changed = edit::place_block(&mut self.world, &hit, *block_type);
                eprintln!("do_place: block_type={} changed={changed}", block_type);
                self.harness_emit_edit("placed", &hit, changed);
                if changed {
                    self.upload_tree();
                }
            }
            HotbarItem::Mesh(idx) => {
                let Some(saved) = self.saved_meshes.items.get(*idx) else { return };
                let node_id = saved.node_id;
                let changed = edit::place_child(
                    &mut self.world, &hit,
                    Child::Node(node_id),
                );
                eprintln!("do_place: mesh_idx={} node_id={} changed={changed}", idx, node_id);
                self.harness_emit_edit("placed", &hit, changed);
                if changed {
                    self.upload_tree();
                }
            }
        }
    }
}
