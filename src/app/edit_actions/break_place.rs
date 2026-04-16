//! `do_break` / `do_place`: cursor-driven edits that feed the hit
//! into `world::edit` and re-upload on success.

use crate::game_state::HotbarItem;
use crate::world::anchor::Path;
use crate::world::edit;

use crate::app::App;

impl App {
    pub(in crate::app) fn do_break(&mut self) {
        let hit = self.frame_aware_raycast();
        let Some(hit) = hit else {
            eprintln!("do_break: no hit");
            return;
        };
        eprintln!(
            "do_break: hit path_len={} face={} place_path_len={:?}",
            hit.path.len(),
            hit.face,
            hit.place_path.as_ref().map(|p| p.len()),
        );

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

    pub(in crate::app) fn do_place(&mut self) {
        let hit = self.frame_aware_raycast();
        let Some(hit) = hit else {
            eprintln!("do_place: no hit");
            return;
        };
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
                    crate::world::tree::Child::Node(node_id),
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
