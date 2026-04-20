//! GPU cursor probe → edit HitInfo conversion.
//!
//! The shader's `march()` is the single source of truth for what the
//! crosshair ray hits. The `cursor_probe` compute dispatch runs that
//! same ray every frame and writes the full world-root-relative slot
//! path of the hit cell. This module walks the library along those
//! slots to materialise the `(NodeId, slot)` pairs edits need, and
//! derives a `place_path` for hits that land inside a face subtree
//! (where the walker's hit normal doesn't map cleanly to Cartesian
//! slot arithmetic).
//!
//! This replaces the old CPU-side `frame_aware_raycast`: there is now
//! exactly one ray-march algorithm (on the GPU), and the CPU is a
//! thin bookkeeping layer that translates its output into edit-space
//! structures.

use crate::app::App;
use crate::renderer::cursor_probe::CursorProbe;
use crate::world::anchor::Path;
use crate::world::edit::HitInfo;
use crate::world::tree::{Child, NodeId, NodeKind};

impl App {
    /// Dispatch / read the cursor probe and build a `HitInfo` from
    /// the returned slot path. Returns `None` if the probe missed or
    /// the slot path can't be walked in the library (e.g. the tree
    /// changed since the probe dispatch — very rare).
    pub(in crate::app) fn probe_hit(&self) -> Option<HitInfo> {
        let renderer = self.renderer.as_ref()?;
        let probe = renderer.read_cursor_probe();
        self.probe_to_hit_info(&probe)
    }

    fn probe_to_hit_info(&self, probe: &CursorProbe) -> Option<HitInfo> {
        if !probe.hit {
            return None;
        }
        let path = self.walk_library_by_slots(&probe.slots)?;
        let place_path = self.derive_place_path(&path);
        Some(HitInfo {
            path,
            face: probe.face as u32,
            t: probe.t,
            place_path,
        })
    }

    /// Walk `world.library` from `world.root`, descending one slot at
    /// a time, returning the `(parent_id, slot)` pair at each step.
    /// Mirrors what `raycast::build_frame_chain` does but with slot
    /// input instead of a ray.
    fn walk_library_by_slots(&self, slots: &[u8]) -> Option<Vec<(NodeId, usize)>> {
        if slots.is_empty() {
            return None;
        }
        let mut path = Vec::with_capacity(slots.len());
        let mut current = self.world.root;
        for (i, &slot) in slots.iter().enumerate() {
            let slot = slot as usize;
            if slot >= 27 {
                return None;
            }
            let node = self.world.library.get(current)?;
            path.push((current, slot));
            let is_last = i + 1 == slots.len();
            match node.children[slot] {
                Child::Node(child_id) => current = child_id,
                Child::Block(_) | Child::Empty => {
                    if !is_last {
                        // Hit descended past a non-Node child. Keep
                        // whatever we resolved — the walker emitted
                        // `is_last` here, so trailing mismatch only
                        // happens with stale probes; degrade to the
                        // shorter path rather than failing outright.
                        path.truncate(i + 1);
                        return Some(path);
                    }
                }
            }
        }
        Some(path)
    }

    /// Derive a `place_path` (the cell a `place_block` should land in)
    /// for the given hit. For Cartesian hits returns `None` — the
    /// edit code's face → xyz-delta fallback handles them. For hits
    /// whose immediate parent is a `CubedSphereFace`, steps back one
    /// cell along the face subtree's `-r` (radial-outward) axis. That
    /// covers the common "aim at surface, break/place" use case.
    ///
    /// Deep-grazing or face-crossing rays can still miss (the empty
    /// cell the ray entered from might be in a sibling face subtree
    /// rather than `r_slot - 1` of the current one). Those rare cases
    /// currently fall through to `place_child`'s face fallback, which
    /// is wrong for face subtrees but doesn't corrupt state — the
    /// edit just doesn't land. A future shader-side `prev_empty_path`
    /// emission would cover them exactly.
    fn derive_place_path(
        &self,
        hit_path: &[(NodeId, usize)],
    ) -> Option<Vec<(NodeId, usize)>> {
        let (parent_id, _) = *hit_path.last()?;
        let parent_kind = self.world.library.get(parent_id)?.kind;
        if !matches!(parent_kind, NodeKind::CubedSphereFace { .. }) {
            return None;
        }

        // Build an anchor path from the hit slots so `step_neighbor_cartesian`
        // can carry/borrow through ancestors precisely (no f32).
        let mut target = Path::root();
        for &(_, slot) in hit_path {
            target.push(slot as u8);
        }
        // r-axis is z in `slot_index(x, y, z) = z*9 + y*3 + x`, so
        // axis index 2 = -r.
        target.step_neighbor_cartesian(2, -1);

        self.walk_library_by_slots(target.as_slice())
    }
}
