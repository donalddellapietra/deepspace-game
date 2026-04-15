//! Ancestor ribbon: the chain that lets the shader pop upward
//! when a ray exits the render frame's `[0, 3)³` bubble.
//!
//! The ribbon is computed AFTER the tree pack: it walks the GPU
//! buffer along the frame path, recording (ancestor_node_idx,
//! slot) pairs. The shader pops in order — `ribbon[0]` is the
//! frame's direct parent (one level above the frame), `ribbon[1]`
//! is the grandparent, and so on up to the absolute world root.
//!
//! At each pop, the shader rescales the ray:
//!
//! ```text
//! parent_pos = vec3(slot_xyz) + frame_pos / 3.0
//! parent_dir = frame_dir / 3.0
//! ```
//!
//! This brings the ray into the parent's `[0, 3)³` frame coords
//! and the DDA continues unchanged at the parent's buffer index.

use bytemuck::{Pod, Zeroable};

use super::types::{GpuChild, GPU_NODE_SIZE};

/// One entry in the ancestor ribbon. The shader pops from the
/// frame upward; `ribbon[0]` is the frame's direct parent, then
/// `ribbon[1]` the grandparent, etc., up to the absolute root.
///
/// `node_idx` is the buffer index of the ancestor's node. `slot`
/// is the slot in the ancestor that contained the level the ray
/// is popping FROM — the shader uses `slot_coords(slot)` to add
/// the integer offset when remapping the ray into the ancestor's
/// frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct GpuRibbonEntry {
    pub node_idx: u32,
    pub slot: u32,
}

/// What `build_ribbon` returns: the chosen frame root in the
/// buffer, the pop-ordered ribbon, and the slot prefix actually
/// reached. `reached_slots.len()` is the **effective frame depth**
/// — which can be shorter than the requested `frame_slots` when
/// the pack flattened a Cartesian sibling on the way down. The
/// caller MUST recompute the camera's `in_frame` projection using
/// `reached_slots` (truncated to its actual length), otherwise
/// the camera is in coords for a frame the shader doesn't have.
#[derive(Clone, Debug)]
pub struct RibbonResult {
    pub frame_root_idx: u32,
    pub ribbon: Vec<GpuRibbonEntry>,
    pub reached_slots: Vec<u8>,
}

/// Walk the GPU buffer from index 0 (world root) along
/// `frame_slots`, following Node-tagged children. Returns the
/// frame root in the buffer, the pop-ordered ribbon, and the
/// slot prefix that the walker actually reached (which may be
/// shorter than `frame_slots` when the pack LOD-flattened a
/// Cartesian sibling at some depth).
///
/// Empty `frame_slots` ⇒ `frame_root_idx = 0`, empty ribbon,
/// empty `reached_slots`.
pub fn build_ribbon(tree: &[GpuChild], frame_slots: &[u8]) -> RibbonResult {
    let mut walk: Vec<u32> = Vec::with_capacity(frame_slots.len() + 1);
    walk.push(0);
    let mut reached_slots: Vec<u8> = Vec::with_capacity(frame_slots.len());
    let mut current = 0u32;
    for &slot in frame_slots {
        let idx = (current as usize) * GPU_NODE_SIZE + slot as usize;
        if idx >= tree.len() { break; }
        let child = tree[idx];
        if child.tag != 2 { break; }
        current = child.node_index;
        walk.push(current);
        reached_slots.push(slot);
    }
    let frame_root_idx = *walk.last().unwrap();
    let depth = reached_slots.len();
    let mut ribbon = Vec::with_capacity(depth);
    for pop in 0..depth {
        let ancestor_idx = walk[depth - 1 - pop];
        let slot = reached_slots[depth - 1 - pop];
        ribbon.push(GpuRibbonEntry {
            node_idx: ancestor_idx,
            slot: slot as u32,
        });
    }
    RibbonResult { frame_root_idx, ribbon, reached_slots }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::anchor::WorldPos;
    use crate::world::tree::{empty_children, uniform_children, Child, NodeKind, NodeLibrary, CENTER_SLOT};
    use super::super::pack::pack_tree_lod;

    #[test]
    fn gpu_ribbon_entry_size() {
        // Must match WGSL `RibbonEntry { node_idx: u32, slot: u32 }`.
        assert_eq!(std::mem::size_of::<GpuRibbonEntry>(), 8);
    }

    fn one_node_tree() -> Vec<GpuChild> {
        vec![GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }; 27]
    }

    fn two_node_tree(parent_slot: u8) -> Vec<GpuChild> {
        // Two nodes in buffer: index 0 (parent) has Node child at
        // `parent_slot` pointing to index 1. Child at idx 1 has
        // 27 Empty children.
        let mut data = vec![GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }; 54];
        data[parent_slot as usize] = GpuChild { tag: 2, block_type: 0, _pad: 0, node_index: 1 };
        data
    }

    #[test]
    fn empty_path_gives_empty_ribbon() {
        let tree = one_node_tree();
        let RibbonResult { frame_root_idx: frame_idx, ribbon, .. } = build_ribbon(&tree, &[]);
        assert_eq!(frame_idx, 0);
        assert!(ribbon.is_empty());
    }

    #[test]
    fn single_step() {
        let tree = two_node_tree(13);
        let RibbonResult { frame_root_idx: frame_idx, ribbon, .. } = build_ribbon(&tree, &[13]);
        assert_eq!(frame_idx, 1);
        assert_eq!(ribbon.len(), 1);
        assert_eq!(ribbon[0], GpuRibbonEntry { node_idx: 0, slot: 13 });
    }

    #[test]
    fn stops_at_non_node_child() {
        // Path requests slot 13 → Node, then slot 5 → ... but child
        // at idx 1 has only Empty children. Walker stops at frame=1.
        let tree = two_node_tree(13);
        let RibbonResult { frame_root_idx: frame_idx, ribbon, .. } = build_ribbon(&tree, &[13, 5]);
        assert_eq!(frame_idx, 1);
        assert_eq!(ribbon.len(), 1);
    }

    #[test]
    fn multi_step_pop_order() {
        // Three nodes: 0 → slot 16 → 1 → slot 8 → 2.
        let mut data = vec![GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }; 81];
        data[16] = GpuChild { tag: 2, block_type: 0, _pad: 0, node_index: 1 };
        data[27 + 8] = GpuChild { tag: 2, block_type: 0, _pad: 0, node_index: 2 };

        let RibbonResult { frame_root_idx: frame_idx, ribbon, .. } = build_ribbon(&data, &[16, 8]);
        assert_eq!(frame_idx, 2);
        assert_eq!(ribbon.len(), 2);
        // Pop order: ribbon[0] = direct parent (idx 1, came from
        // slot 8); ribbon[1] = grandparent (idx 0, came from slot 16).
        assert_eq!(ribbon[0], GpuRibbonEntry { node_idx: 1, slot: 8 });
        assert_eq!(ribbon[1], GpuRibbonEntry { node_idx: 0, slot: 16 });
    }

    #[test]
    fn out_of_bounds_index_safe() {
        // Crafted child with node_index past tree end: walker must
        // not panic.
        let mut data = vec![GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 }; 27];
        data[5] = GpuChild { tag: 2, block_type: 0, _pad: 0, node_index: 999 };
        // Walking [5] → tag=2 advances to idx 999, then [5] → idx
        // 999*27+5 = 26978 which is past tree.len(). Walker breaks.
        let RibbonResult { frame_root_idx: frame_idx, ribbon, .. } = build_ribbon(&data, &[5, 5]);
        assert_eq!(frame_idx, 999);
        assert_eq!(ribbon.len(), 1);
    }

    fn planet_world() -> crate::world::state::WorldState {
        let mut lib = NodeLibrary::default();
        let leaf_air = lib.insert(empty_children());
        let mut root_children = uniform_children(Child::Node(leaf_air));
        let body_id = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        root_children[CENTER_SLOT] = Child::Node(body_id);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        crate::world::state::WorldState { root, library: lib }
    }

    fn camera_at(xyz: [f32; 3]) -> WorldPos {
        WorldPos::from_world_xyz(xyz, 10)
    }

    #[test]
    fn ribbon_for_path_into_body_in_planet_world() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        let RibbonResult { frame_root_idx: frame_idx, ribbon, .. } = build_ribbon(&data, &[13]);
        assert!(frame_idx > 0, "body packed at non-zero index");
        assert_eq!(ribbon.len(), 1);
        assert_eq!(ribbon[0].node_idx, 0, "world root at index 0");
        assert_eq!(ribbon[0].slot, 13);
    }

    #[test]
    fn reached_slots_truncated_when_pack_flattens_sibling() {
        // Plant a Cartesian sibling that pack will flatten because
        // it's uniform-empty. Walking deep into it via build_ribbon
        // should stop at the world root with reached_slots empty.
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        // Slot 16 is uniform-empty Cartesian Node — pack flattens
        // it to tag=0. Asking the ribbon to descend into [16, 13]
        // should stop at depth 0.
        let r = build_ribbon(&data, &[16, 13]);
        assert_eq!(r.frame_root_idx, 0, "stayed at world root");
        assert!(r.ribbon.is_empty());
        assert!(r.reached_slots.is_empty(),
            "no slot reachable past flattened sibling");
    }

    #[test]
    fn frame_root_at_world_root_yields_empty_ribbon_in_planet_world() {
        let world = planet_world();
        let camera = camera_at([0.5, 0.5, 0.5]);
        let (data, _kinds, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        let RibbonResult { frame_root_idx: frame_idx, ribbon, .. } = build_ribbon(&data, &[]);
        assert_eq!(frame_idx, 0);
        assert!(ribbon.is_empty());
    }
}
