//! GPU cursor probe → edit HitInfo conversion.
//!
//! The shader's `march()` is the single source of truth for what the
//! crosshair ray hits. `probe_hit` dispatches a fresh single-ray
//! compute with the current camera uniform, waits for the GPU, then
//! walks the library along the returned slot path to materialise the
//! `(NodeId, slot)` pairs edits need. For hits deeper than the
//! walker's `MAX_STACK_DEPTH` cap we extend by descending into a
//! ray-aligned slot (chosen from the hit normal) until `edit_depth`
//! or until the library runs out of Node children.
//!
//! There is no parallel CPU raycast. The walker's result is the hit
//! — this module's job is only to translate slot bytes into edit-
//! ready structures.

use crate::app::App;
use crate::renderer::cursor_probe::CursorProbe;
use crate::world::anchor::Path;
use crate::world::edit::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeKind};

impl App {
    /// Dispatch a fresh cursor probe with the current camera state,
    /// wait for the GPU, and build a `HitInfo`. Using a standalone
    /// sync dispatch (instead of reading the last frame's probe) is
    /// required for scripted probe-direction changes: the per-frame
    /// probe in `render()` uses whatever camera uniform was written
    /// before the frame submit, so a mid-frame pitch rotation (e.g.
    /// `probe_down` rotating to face-down) would otherwise read a
    /// stale hit from the old camera orientation.
    pub(in crate::app) fn probe_hit(&self) -> Option<HitInfo> {
        let cam_gpu = self.gpu_camera_for_frame(&self.active_frame);
        let renderer = self.renderer.as_ref()?;
        renderer.queue.write_buffer(
            &renderer.camera_buffer,
            0,
            bytemuck::bytes_of(&cam_gpu),
        );
        let probe = renderer.dispatch_and_read_cursor_probe_sync();
        self.probe_to_hit_info(&probe)
    }

    fn probe_to_hit_info(&self, probe: &CursorProbe) -> Option<HitInfo> {
        if !probe.hit {
            return None;
        }
        let edit_depth = self.edit_depth() as usize;
        let trim = probe.slots.len().min(edit_depth);
        let slots = &probe.slots[..trim];
        let pad_slot = pad_slot_for_face(probe.face);
        let path = self.walk_library_by_slots_padded(slots, edit_depth, pad_slot)?;
        let place_path = self.derive_place_path(&path);
        Some(HitInfo {
            path,
            face: probe.face as u32,
            t: probe.t,
            place_path,
        })
    }

    /// Walk `world.library` from `world.root` through the given slot
    /// sequence, then extend with `pad_slot` descent until
    /// `target_depth` or until the library hits a non-Node child. If
    /// `pad_slot` resolves to Empty/Block at a padding step, fall
    /// back to the first Node sibling — the walker flagged this
    /// region as a hit, so any Node child keeps us inside its
    /// content.
    fn walk_library_by_slots_padded(
        &self,
        slots: &[u8],
        target_depth: usize,
        pad_slot: usize,
    ) -> Option<Vec<(NodeId, usize)>> {
        if slots.is_empty() && target_depth == 0 {
            return None;
        }
        let mut path = Vec::with_capacity(target_depth.max(slots.len()));
        let mut current = self.world.root;
        for &slot in slots {
            let slot = slot as usize;
            if slot >= 27 {
                return None;
            }
            let node = self.world.library.get(current)?;
            path.push((current, slot));
            match node.children[slot] {
                Child::Node(child_id) => current = child_id,
                Child::Block(_) | Child::Empty => {
                    return Some(path);
                }
            }
        }
        while path.len() < target_depth {
            let node = match self.world.library.get(current) {
                Some(n) => n,
                None => break,
            };
            let mut chosen_slot = pad_slot;
            let mut chosen_child = node.children[pad_slot];
            if !matches!(chosen_child, Child::Node(_)) {
                for s in 0..27 {
                    if let Child::Node(_) = node.children[s] {
                        chosen_slot = s;
                        chosen_child = node.children[s];
                        break;
                    }
                }
            }
            path.push((current, chosen_slot));
            match chosen_child {
                Child::Node(child_id) => current = child_id,
                Child::Block(_) | Child::Empty => break,
            }
        }
        Some(path)
    }

    /// Derive a `place_path` (the cell a `place_block` should land in)
    /// for the given hit. For Cartesian hits returns `None` — the
    /// edit code's face → xyz-delta fallback handles them. For hits
    /// whose immediate parent is a `CubedSphereFace`, steps back one
    /// cell along the face subtree's `-r` (radial-outward) axis.
    fn derive_place_path(
        &self,
        hit_path: &[(NodeId, usize)],
    ) -> Option<Vec<(NodeId, usize)>> {
        let (parent_id, _) = *hit_path.last()?;
        let parent_kind = self.world.library.get(parent_id)?.kind;
        if !matches!(parent_kind, NodeKind::CubedSphereFace { .. }) {
            return None;
        }

        let mut target = Path::root();
        for &(_, slot) in hit_path {
            target.push(slot as u8);
        }
        // r-axis is z in `slot_index(x, y, z) = z*9 + y*3 + x`, so
        // axis index 2 = -r.
        target.step_neighbor_cartesian(2, -1);

        self.walk_library_by_slots_padded(
            target.as_slice(),
            target.depth() as usize,
            slot_index(1, 1, 1),
        )
    }
}

/// Map a `face` id (from the GPU probe's encoded hit normal) to the
/// child slot the ray would enter if we descended one more level.
/// `face = 0/1 = ±X`, `2/3 = ±Y`, `4/5 = ±Z`; a positive face id
/// means the surface normal points in the positive axis, which means
/// the ray was travelling in the negative direction on that axis — so
/// the next descent should pick the 0-side of that axis, with the
/// other two axes at centre. Fallback to centre on unknown face.
fn pad_slot_for_face(face: u8) -> usize {
    match face {
        0 => slot_index(2, 1, 1), // ray going +X
        1 => slot_index(0, 1, 1), // ray going -X
        2 => slot_index(1, 2, 1), // ray going +Y
        3 => slot_index(1, 0, 1), // ray going -Y
        4 => slot_index(1, 1, 2), // ray going +Z
        5 => slot_index(1, 1, 0), // ray going -Z
        _ => slot_index(1, 1, 1),
    }
}
