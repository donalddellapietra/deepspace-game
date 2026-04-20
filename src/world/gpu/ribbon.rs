//! Ancestor ribbon: the chain that lets the shader pop upward
//! when a ray exits the render frame's `[0, 3)³` bubble.
//!
//! The ribbon is computed AFTER the tree pack: it walks the
//! interleaved GPU buffer along the frame path, recording
//! (ancestor_bfs_idx, slot) pairs. The shader pops in order —
//! `ribbon[0]` is the frame's direct parent (one level above the
//! frame), `ribbon[1]` is the grandparent, and so on up to the
//! absolute world root.
//!
//! At each pop, the shader rescales the ray:
//!
//! ```text
//! parent_pos = vec3(slot_xyz) + frame_pos / 3.0
//! parent_dir = frame_dir / 3.0
//! ```
//!
//! This brings the ray into the parent's `[0, 3)³` frame coords and
//! the DDA continues at the parent's BFS index (looked up via
//! `node_offsets[]` to find the parent's header in `tree[]`).

use bytemuck::{Pod, Zeroable};

/// One entry in the ancestor ribbon. The shader pops from the
/// frame upward; `ribbon[0]` is the frame's direct parent, then
/// `ribbon[1]` the grandparent, etc., up to the absolute root.
///
/// `node_idx` is the BFS position of the ancestor's node — the
/// shader maps this to a `tree[]` u32-offset via `node_offsets[]`.
///
/// `slot_bits` packs two things into a u32:
/// - Low 5 bits: slot (0..27) in the ancestor that contained the
///   level the ray is popping FROM.
/// - Bit 31: `siblings_all_empty` flag. When set, every child slot
///   of the ancestor OTHER than `slot` is empty. The shader uses
///   this to fast-exit the whole shell with a single ray–box
///   intersection instead of DDA-traversing empty cells.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct GpuRibbonEntry {
    pub node_idx: u32,
    pub slot_bits: u32,
}

/// Bit mask for the `siblings_all_empty` flag in `slot_bits`.
pub const SIBLINGS_ALL_EMPTY_BIT: u32 = 1 << 31;
/// Bit mask isolating the slot index (0..27) in `slot_bits`.
pub const SLOT_MASK: u32 = 0x1F;

impl GpuRibbonEntry {
    pub fn new(node_idx: u32, slot: u8, siblings_all_empty: bool) -> Self {
        let flags = if siblings_all_empty { SIBLINGS_ALL_EMPTY_BIT } else { 0 };
        Self {
            node_idx,
            slot_bits: (slot as u32) | flags,
        }
    }

    pub fn slot(&self) -> u32 { self.slot_bits & SLOT_MASK }

    pub fn siblings_all_empty(&self) -> bool {
        (self.slot_bits & SIBLINGS_ALL_EMPTY_BIT) != 0
    }
}

/// What `build_ribbon` returns: the chosen frame root (BFS idx) in
/// the pack, the pop-ordered ribbon, and the slot prefix actually
/// reached. `reached_slots.len()` is the **effective frame depth**
/// — which can be shorter than the requested `frame_slots` when the
/// pack flattened a Cartesian sibling on the way down. The caller
/// MUST recompute the camera's `in_frame` projection using
/// `reached_slots` (truncated to its actual length), otherwise the
/// camera is in coords for a frame the shader doesn't have.
#[derive(Clone, Debug)]
pub struct RibbonResult {
    pub frame_root_idx: u32,
    pub ribbon: Vec<GpuRibbonEntry>,
    pub reached_slots: Vec<u8>,
}

/// Walk the interleaved GPU buffer from `world_root_bfs` along
/// `frame_slots`, following Node-tagged children. Returns the frame
/// root's BFS index, the pop-ordered ribbon, and the slot prefix
/// that the walker actually reached (which may be shorter than
/// `frame_slots` when pack uniform-flattened a Cartesian sibling at
/// some depth).
///
/// The world root used to be at BFS idx 0; with the recursive
/// memoized pack it's the last-appended entry (post-order DFS
/// emission), so callers pass it explicitly.
///
/// Empty `frame_slots` ⇒ `frame_root_idx = world_root_bfs`, empty
/// ribbon, empty `reached_slots`.
pub fn build_ribbon(
    tree: &[u32],
    node_offsets: &[u32],
    world_root_bfs: u32,
    frame_slots: &[u8],
) -> RibbonResult {
    let mut walk: Vec<u32> = Vec::with_capacity(frame_slots.len() + 1);
    walk.push(world_root_bfs);
    let mut reached_slots: Vec<u8> = Vec::with_capacity(frame_slots.len());
    let mut current = world_root_bfs;
    for &slot in frame_slots {
        let Some(entry) = sparse_child(tree, node_offsets, current, slot) else { break };
        if entry.tag != 2 { break; }
        current = entry.node_index;
        walk.push(current);
        reached_slots.push(slot);
    }
    let frame_root_idx = *walk.last().unwrap();
    let depth = reached_slots.len();
    let mut ribbon = Vec::with_capacity(depth);
    for pop in 0..depth {
        let ancestor_idx = walk[depth - 1 - pop];
        let slot = reached_slots[depth - 1 - pop];
        let siblings_all_empty =
            ancestor_siblings_all_empty(tree, node_offsets, ancestor_idx);
        ribbon.push(GpuRibbonEntry::new(ancestor_idx, slot, siblings_all_empty));
    }
    RibbonResult { frame_root_idx, ribbon, reached_slots }
}

/// Simple decoded GpuChild used by the ribbon builder. Distinct from
/// `super::types::GpuChild` so the ribbon code is self-contained.
struct DecodedChild {
    tag: u8,
    node_index: u32,
}

/// Look up slot `slot` at the node with BFS idx `bfs_idx`. Returns
/// `None` when the slot is out of bounds or empty; otherwise the
/// decoded child entry.
fn sparse_child(
    tree: &[u32],
    node_offsets: &[u32],
    bfs_idx: u32,
    slot: u8,
) -> Option<DecodedChild> {
    let header_off = *node_offsets.get(bfs_idx as usize)? as usize;
    let occupancy = *tree.get(header_off)?;
    let bit = 1u32 << slot;
    if occupancy & bit == 0 {
        return None;
    }
    let first_child = *tree.get(header_off + 1)? as usize;
    let rank = (occupancy & (bit - 1)).count_ones() as usize;
    let off = first_child + rank * 2;
    let packed = *tree.get(off)?;
    let tag = (packed & 0xFF) as u8;
    let node_index = *tree.get(off + 1)?;
    Some(DecodedChild { tag, node_index })
}

/// True when every child of the node OTHER than the one we just
/// popped out of is empty. In the sparse layout, this is equivalent
/// to `occupancy.count_ones() == 1` — the only set bit MUST be the
/// slot we descended through (otherwise the walker couldn't have
/// reached the child in the first place), so a single-bit occupancy
/// implies all siblings are empty.
fn ancestor_siblings_all_empty(
    tree: &[u32],
    node_offsets: &[u32],
    bfs_idx: u32,
) -> bool {
    let Some(&header_off) = node_offsets.get(bfs_idx as usize) else { return false };
    let Some(&occupancy) = tree.get(header_off as usize) else { return false };
    occupancy.count_ones() == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::anchor::WorldPos;
    use crate::world::tree::{empty_children, uniform_children, Child, NodeKind, NodeLibrary, CENTER_SLOT};
    use super::super::pack::pack_tree;

    #[test]
    fn gpu_ribbon_entry_size() {
        assert_eq!(std::mem::size_of::<GpuRibbonEntry>(), 8);
    }

    /// Encode a GpuChild's first u32 (tag + block_type_lo + block_type_hi + flags).
    fn encode_packed(tag: u8, block_type: u16) -> u32 {
        (tag as u32) | ((block_type as u32) << 8)
    }

    /// A single empty node in the interleaved layout (2 u32 header,
    /// no children).
    fn one_node_tree() -> (Vec<u32>, Vec<u32>) {
        let tree = vec![0u32 /* occupancy */, 2u32 /* first_child */];
        let offsets = vec![0u32];
        (tree, offsets)
    }

    /// Two-node tree: root has a single Node child at `parent_slot`
    /// pointing to BFS idx 1 (which is itself an empty node).
    ///
    /// Layout:
    ///   offset 0: root header  (occupancy=1<<slot, first_child=2)
    ///   offset 2: root's child (packed tag=2, node_index=1)
    ///   offset 4: child header (occupancy=0, first_child=6)
    fn two_node_tree(parent_slot: u8) -> (Vec<u32>, Vec<u32>) {
        let mut tree: Vec<u32> = Vec::new();
        // Root header at offset 0.
        tree.push(1u32 << parent_slot); // occupancy
        tree.push(2);                    // first_child
        // Root's one child at offset 2.
        tree.push(encode_packed(2, 0));
        tree.push(1); // node_index = BFS idx 1
        // Child header at offset 4.
        tree.push(0);
        tree.push(6);
        let offsets = vec![0u32, 4u32];
        (tree, offsets)
    }

    #[test]
    fn empty_path_gives_empty_ribbon() {
        let (tree, offsets) = one_node_tree();
        let RibbonResult { frame_root_idx, ribbon, .. } =
            build_ribbon(&tree, &offsets, 0, &[]);
        assert_eq!(frame_root_idx, 0);
        assert!(ribbon.is_empty());
    }

    #[test]
    fn single_step() {
        let (tree, offsets) = two_node_tree(13);
        let RibbonResult { frame_root_idx, ribbon, .. } =
            build_ribbon(&tree, &offsets, 0, &[13]);
        assert_eq!(frame_root_idx, 1);
        assert_eq!(ribbon.len(), 1);
        assert_eq!(ribbon[0].node_idx, 0);
        assert_eq!(ribbon[0].slot(), 13);
        // Root has occupancy with exactly one bit set → flag set.
        assert!(ribbon[0].siblings_all_empty());
    }

    #[test]
    fn stops_at_non_node_child() {
        // Path requests slot 13 → Node, then slot 5 → empty at child.
        // Walker stops at frame=1.
        let (tree, offsets) = two_node_tree(13);
        let RibbonResult { frame_root_idx, ribbon, .. } =
            build_ribbon(&tree, &offsets, 0, &[13, 5]);
        assert_eq!(frame_root_idx, 1);
        assert_eq!(ribbon.len(), 1);
    }

    #[test]
    fn multi_step_pop_order() {
        // Three nodes: 0 → slot 16 → 1 → slot 8 → 2.
        // Layout:
        //   offset 0: root header (occupancy=1<<16, first_child=2)
        //   offset 2: root child (tag=2, node_index=1)
        //   offset 4: node 1 header (occupancy=1<<8, first_child=6)
        //   offset 6: node 1 child (tag=2, node_index=2)
        //   offset 8: node 2 header (occupancy=0, first_child=10)
        let mut tree: Vec<u32> = Vec::new();
        tree.extend_from_slice(&[1u32 << 16, 2]);              // root
        tree.extend_from_slice(&[encode_packed(2, 0), 1]);     // root's child
        tree.extend_from_slice(&[1u32 << 8, 6]);               // node 1
        tree.extend_from_slice(&[encode_packed(2, 0), 2]);     // node 1's child
        tree.extend_from_slice(&[0, 10]);                       // node 2
        let offsets = vec![0u32, 4, 8];

        let RibbonResult { frame_root_idx, ribbon, .. } =
            build_ribbon(&tree, &offsets, 0, &[16, 8]);
        assert_eq!(frame_root_idx, 2);
        assert_eq!(ribbon.len(), 2);
        // Pop order: ribbon[0] = direct parent (idx 1, from slot 8);
        // ribbon[1] = grandparent (idx 0, from slot 16).
        assert_eq!(ribbon[0].node_idx, 1);
        assert_eq!(ribbon[0].slot(), 8);
        assert_eq!(ribbon[1].node_idx, 0);
        assert_eq!(ribbon[1].slot(), 16);
    }

    #[test]
    fn out_of_bounds_index_safe() {
        // A tag=2 child pointing past the nodes array: walker must
        // not panic, just stop the descent.
        let mut tree: Vec<u32> = Vec::new();
        tree.extend_from_slice(&[1u32 << 5, 2]);
        tree.extend_from_slice(&[encode_packed(2, 0), 999]);
        let offsets = vec![0u32];
        let RibbonResult { frame_root_idx, ribbon, .. } =
            build_ribbon(&tree, &offsets, 0, &[5, 5]);
        assert_eq!(frame_root_idx, 999);
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
        WorldPos::from_frame_local(&crate::world::anchor::Path::root(), xyz, 2)
            .deepened_to(10)
    }

    #[test]
    fn ribbon_for_path_into_body_in_planet_world() {
        let world = planet_world();
        let (tree, _kinds, offsets, _, _node_ids, root_idx) = pack_tree(&world.library, world.root);
        let RibbonResult { frame_root_idx, ribbon, .. } =
            build_ribbon(&tree, &offsets, root_idx, &[13]);
        assert_ne!(frame_root_idx, root_idx, "body packed at a different BFS idx");
        assert_eq!(ribbon.len(), 1);
        assert_eq!(ribbon[0].node_idx, root_idx, "first pop is the world root");
        assert_eq!(ribbon[0].slot(), 13);
    }

    #[test]
    fn reached_slots_truncated_when_pack_flattens_sibling() {
        let world = planet_world();
        let (tree, _kinds, offsets, _, _node_ids, root_idx) = pack_tree(&world.library, world.root);
        // Slot 16 is uniform-empty Cartesian → absent from pack.
        // Descent into [16, 13] stops at depth 0.
        let r = build_ribbon(&tree, &offsets, root_idx, &[16, 13]);
        assert_eq!(r.frame_root_idx, root_idx, "stayed at world root");
        assert!(r.ribbon.is_empty());
        assert!(r.reached_slots.is_empty(),
            "no slot reachable past flattened sibling");
    }

    #[test]
    fn frame_root_at_world_root_yields_empty_ribbon_in_planet_world() {
        let world = planet_world();
        let (tree, _kinds, offsets, _, _node_ids, root_idx) = pack_tree(&world.library, world.root);
        let RibbonResult { frame_root_idx, ribbon, .. } =
            build_ribbon(&tree, &offsets, root_idx, &[]);
        assert_eq!(frame_root_idx, root_idx);
        assert!(ribbon.is_empty());
    }
}
