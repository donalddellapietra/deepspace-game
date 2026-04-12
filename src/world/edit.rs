//! Edit walks: single-voxel leaf edits and bulk higher-layer edits.
//!
//! See `docs/architecture/editing.md` for the full design. Every
//! edit walks up the tree from the touched layer to the root,
//! minting a fresh `NodeId` at each ancestor via the library. The
//! library's content dedup means most ancestors recycle existing
//! ids when the change is small.

use super::position::{LayerPos, Position, NODE_PATH_LEN};
use super::state::WorldState;
use super::tree::{
    downsample, downsample_updated_slot, filled_voxel_grid, slot_index,
    uniform_children, voxel_idx, NodeId, Voxel, VoxelGrid, BRANCH_FACTOR,
    CHILDREN_PER_NODE, MAX_LAYER, NODE_VOXELS_PER_AXIS,
};

/// Construct the tree path that reaches the subtree "under" a view
/// cell named by `lp`. Mirrors the `slot_a = c/5, slot_b = c%5`
/// decomposition that `edit_at_layer_pos` uses internally: starting
/// from `lp.path` (length `lp.layer`), push one or two extra slots
/// until we reach the layer that one view cell actually corresponds
/// to.
///
/// The returned path length equals `min(lp.layer + 2, MAX_LAYER)`,
/// so:
///
/// - `lp.layer <= MAX_LAYER - 2` → `path.len() == lp.layer + 2`
///   (one clean layer-`(L+2)` subtree per view cell).
/// - `lp.layer == MAX_LAYER - 1` → `path.len() == MAX_LAYER`
///   (one leaf; the second descent is clamped).
/// - `lp.layer == MAX_LAYER` → `path.len() == MAX_LAYER` (the
///   leaf containing `lp.cell` as one voxel).
///
/// Call this from any site that needs to address the subtree beneath
/// a hovered / clicked cell — `resolve_node_at_lp` in save mode,
/// `place_block` when placing a saved mesh, etc. — rather than
/// re-deriving the decomposition inline.
pub fn subtree_path_for_layer_pos(lp: &LayerPos) -> Vec<u8> {
    let mut path = lp.path.clone();
    if lp.layer >= MAX_LAYER {
        return path;
    }
    let b = BRANCH_FACTOR as u8;
    let slot_a = slot_index(
        (lp.cell[0] / b) as usize,
        (lp.cell[1] / b) as usize,
        (lp.cell[2] / b) as usize,
    );
    path.push(slot_a as u8);
    if lp.layer + 1 < MAX_LAYER {
        let slot_b = slot_index(
            (lp.cell[0] % b) as usize,
            (lp.cell[1] % b) as usize,
            (lp.cell[2] % b) as usize,
        );
        path.push(slot_b as u8);
    }
    path
}

/// Edit a single leaf voxel to `voxel`. Propagates up the tree and
/// atomically replaces `world.root`.
pub fn edit_leaf(world: &mut WorldState, position: &Position, voxel: Voxel) {
    // Walk down from the root, collecting one (ancestor_id, child_slot)
    // pair per layer. After the loop, `current_id` points at the leaf.
    let mut descent: Vec<(NodeId, usize)> = Vec::with_capacity(NODE_PATH_LEN);
    let mut current_id = world.root;
    for layer_idx in 0..NODE_PATH_LEN {
        let slot = position.path[layer_idx] as usize;
        descent.push((current_id, slot));
        let node = world
            .library
            .get(current_id)
            .expect("edit_leaf: descent hit a missing node");
        let children = node
            .children
            .as_ref()
            .expect("edit_leaf: descent hit a leaf above MAX_LAYER");
        current_id = children[slot];
    }

    // Build the new leaf's voxel grid.
    let new_leaf_voxels: VoxelGrid = {
        let leaf = world
            .library
            .get(current_id)
            .expect("edit_leaf: missing leaf");
        let mut v = leaf.voxels.clone();
        v[voxel_idx(
            position.voxel[0] as usize,
            position.voxel[1] as usize,
            position.voxel[2] as usize,
        )] = voxel;
        v
    };
    let mut new_child_id = world.library.insert_leaf(new_leaf_voxels);

    // Walk back up, rebuilding each ancestor with the new child id.
    // Only one slot changed per layer, so we incrementally patch that
    // slot's 5³ parent region instead of re-downsampling all 125
    // children from scratch.
    for &(parent_id, slot) in descent.iter().rev() {
        let (new_voxels, new_children) = {
            let parent = world
                .library
                .get(parent_id)
                .expect("edit_leaf: parent missing");
            let old_children = parent
                .children
                .as_ref()
                .expect("edit_leaf: non-leaf expected on walk up");
            let mut new_children = old_children.clone();
            new_children[slot] = new_child_id;
            let new_child_voxels = &world
                .library
                .get(new_child_id)
                .expect("edit_leaf: new child missing")
                .voxels;
            let new_voxels = downsample_updated_slot(
                &parent.voxels,
                new_child_voxels,
                slot,
            );
            (new_voxels, new_children)
        };
        new_child_id = world.library.insert_non_leaf(new_voxels, new_children);
    }

    // Transfer the external ref from old root to new root. Order
    // matters: ref_inc first so that if old == new (round-trip edit)
    // the refcount never hits zero.
    world.library.ref_inc(new_child_id);
    let old_root = world.root;
    world.root = new_child_id;
    world.library.ref_dec(old_root);
}

/// Bulk edit: replace the node at `path_prefix` (of length
/// `target_layer`, 0..=MAX_LAYER) with a solid chain of `voxel`.
///
/// - `path_prefix.len() == MAX_LAYER` → replace the leaf with a
///   filled leaf.
/// - `path_prefix.len() < MAX_LAYER` → build a solid-voxel subtree
///   from the leaf up to `target_layer` and splice it in.
/// - `path_prefix.len() == 0` → replace the entire world root.
pub fn edit_at_layer(world: &mut WorldState, path_prefix: &[u8], voxel: Voxel) {
    let target_layer = path_prefix.len();
    assert!(target_layer <= MAX_LAYER as usize);

    // Build solid-voxel chain. chain_id starts as a leaf and is
    // wrapped in a non-leaf once per layer up until it reaches
    // target_layer.
    let leaf_voxels = filled_voxel_grid(voxel);
    let mut chain_id = world.library.insert_leaf(leaf_voxels);
    let mut chain_layer = MAX_LAYER as usize;
    while chain_layer > target_layer {
        let voxels = {
            let node = world
                .library
                .get(chain_id)
                .expect("edit_at_layer: chain node missing");
            let refs: [&VoxelGrid; CHILDREN_PER_NODE] =
                std::array::from_fn(|_| &node.voxels);
            downsample(refs)
        };
        let children = uniform_children(chain_id);
        chain_id = world.library.insert_non_leaf(voxels, children);
        chain_layer -= 1;
    }
    // chain_id is now at exactly `target_layer`.

    // Walk down path_prefix collecting ancestors at depths 0..target_layer.
    let mut descent: Vec<(NodeId, usize)> = Vec::with_capacity(target_layer);
    let mut current_id = world.root;
    for layer_idx in 0..target_layer {
        let slot = path_prefix[layer_idx] as usize;
        descent.push((current_id, slot));
        let node = world
            .library
            .get(current_id)
            .expect("edit_at_layer: missing ancestor");
        let children = node
            .children
            .as_ref()
            .expect("edit_at_layer: non-leaf expected on descent");
        current_id = children[slot];
    }

    // Walk back up, splicing chain_id at the target_layer position.
    // Incremental downsample: one slot changes per layer.
    let mut new_child_id = chain_id;
    for &(parent_id, slot) in descent.iter().rev() {
        let (new_voxels, new_children) = {
            let parent = world
                .library
                .get(parent_id)
                .expect("edit_at_layer: parent missing");
            let old_children = parent
                .children
                .as_ref()
                .expect("edit_at_layer: non-leaf expected on walk up");
            let mut new_children = old_children.clone();
            new_children[slot] = new_child_id;
            let new_child_voxels = &world
                .library
                .get(new_child_id)
                .expect("edit_at_layer: new child missing")
                .voxels;
            let new_voxels = downsample_updated_slot(
                &parent.voxels,
                new_child_voxels,
                slot,
            );
            (new_voxels, new_children)
        };
        new_child_id = world.library.insert_non_leaf(new_voxels, new_children);
    }

    world.library.ref_inc(new_child_id);
    let old_root = world.root;
    world.root = new_child_id;
    world.library.ref_dec(old_root);
}

/// Apply an edit named by a [`LayerPos`]. Dispatches on the view
/// layer to one of three shapes:
///
/// - **`lp.layer == MAX_LAYER`** (leaf view): single-cell edit on the
///   clicked leaf voxel. Equivalent to calling [`edit_leaf`] directly.
/// - **`lp.layer == MAX_LAYER - 1`**: fills the `5³` region inside the
///   leaf one level below the view node that this cell summarises.
/// - **`lp.layer <= MAX_LAYER - 2`**: replaces the whole layer-`(L + 2)`
///   subtree beneath the clicked cell with a canonical "solid X"
///   chain. The click's `(cx, cy, cz)` decomposes into two slot
///   steps: `slot_a = (c / 5)` and `slot_b = (c % 5)`.
///
/// See `docs/architecture/editing.md` and the 2D prototype's
/// `World::edit_at` for the derivation of the three cases.
pub fn edit_at_layer_pos(world: &mut WorldState, lp: &LayerPos, voxel: Voxel) {
    assert!(lp.layer as usize == lp.path.len());
    assert!(lp.layer <= MAX_LAYER);
    assert!(lp.cell[0] < NODE_VOXELS_PER_AXIS as u8);
    assert!(lp.cell[1] < NODE_VOXELS_PER_AXIS as u8);
    assert!(lp.cell[2] < NODE_VOXELS_PER_AXIS as u8);

    let cx = lp.cell[0];
    let cy = lp.cell[1];
    let cz = lp.cell[2];
    let b = BRANCH_FACTOR as u8;

    if lp.layer == MAX_LAYER {
        // Leaf-view: single-voxel edit. Synthesise a `Position`
        // pointing at the target cell and fall through.
        assert!(lp.path.len() == NODE_PATH_LEN);
        let mut path = [0u8; NODE_PATH_LEN];
        for (i, &s) in lp.path.iter().enumerate() {
            path[i] = s;
        }
        let position = Position {
            path,
            voxel: [cx, cy, cz],
            offset: [0.5, 0.5, 0.5],
        };
        edit_leaf(world, &position, voxel);
        return;
    }

    if lp.layer == MAX_LAYER - 1 {
        // One layer above leaves: the clicked cell summarises a 5³
        // region of one specific leaf below us. Walk down into that
        // leaf, clone its voxels, fill the 5³ region, and call
        // `install_subtree` via `edit_leaf`'s walk.
        let child_slot = slot_index(
            (cx / b) as usize,
            (cy / b) as usize,
            (cz / b) as usize,
        );
        let mut leaf_path_vec: Vec<u8> = lp.path.clone();
        leaf_path_vec.push(child_slot as u8);
        debug_assert_eq!(leaf_path_vec.len(), NODE_PATH_LEN);

        // Locate the leaf so we can read its current voxel grid.
        let mut leaf_id = world.root;
        for &slot in &leaf_path_vec {
            let node = world
                .library
                .get(leaf_id)
                .expect("edit_at_layer_pos: descent hit missing node");
            let children = node
                .children
                .as_ref()
                .expect("edit_at_layer_pos: non-leaf expected on descent");
            leaf_id = children[slot as usize];
        }
        let mut new_voxels: VoxelGrid = world
            .library
            .get(leaf_id)
            .expect("edit_at_layer_pos: leaf missing")
            .voxels
            .clone();
        let rx0 = ((cx % b) * b) as usize;
        let ry0 = ((cy % b) * b) as usize;
        let rz0 = ((cz % b) * b) as usize;
        for dz in 0..BRANCH_FACTOR {
            for dy in 0..BRANCH_FACTOR {
                for dx in 0..BRANCH_FACTOR {
                    new_voxels[voxel_idx(rx0 + dx, ry0 + dy, rz0 + dz)] = voxel;
                }
            }
        }
        let new_leaf_id = world.library.insert_leaf(new_voxels);
        install_subtree(world, &leaf_path_vec, new_leaf_id);
        return;
    }

    // Two or more layers above leaves: replace the whole layer-(L+2)
    // subtree with a solid-voxel chain. (cx, cy, cz) decomposes into
    // two more slot steps to reach that subtree.
    let slot_a = slot_index(
        (cx / b) as usize,
        (cy / b) as usize,
        (cz / b) as usize,
    );
    let slot_b = slot_index(
        (cx % b) as usize,
        (cy % b) as usize,
        (cz % b) as usize,
    );
    let mut sub_path: Vec<u8> = lp.path.clone();
    sub_path.push(slot_a as u8);
    sub_path.push(slot_b as u8);
    let target_layer = (lp.layer + 2) as usize;

    // Build (or recycle via dedup) a solid-voxel chain rooted at
    // `target_layer`. Same construction as `edit_at_layer` below.
    let leaf_voxels = filled_voxel_grid(voxel);
    let mut chain_id = world.library.insert_leaf(leaf_voxels);
    let mut chain_layer = MAX_LAYER as usize;
    while chain_layer > target_layer {
        let voxels = {
            let node = world
                .library
                .get(chain_id)
                .expect("edit_at_layer_pos: chain node missing");
            let refs: [&VoxelGrid; CHILDREN_PER_NODE] =
                std::array::from_fn(|_| &node.voxels);
            downsample(refs)
        };
        let children = uniform_children(chain_id);
        chain_id = world.library.insert_non_leaf(voxels, children);
        chain_layer -= 1;
    }

    install_subtree(world, &sub_path, chain_id);
}

/// Replace the node at `ancestor_slots` with `new_node_id`, then walk
/// back up to the root re-downsampling the one affected slot per
/// ancestor and interning each new ancestor. The length of
/// `ancestor_slots` equals the layer of the replaced node.
///
/// Public so the save-mode placement flow can splice a previously
/// captured subtree back into the world at a matching layer.
pub fn install_subtree(world: &mut WorldState, ancestor_slots: &[u8], new_node_id: NodeId) {
    let mut descent: Vec<(NodeId, usize)> = Vec::with_capacity(ancestor_slots.len());
    let mut current_id = world.root;
    for &slot in ancestor_slots {
        descent.push((current_id, slot as usize));
        let node = world
            .library
            .get(current_id)
            .expect("install_subtree: descent hit missing node");
        let children = node
            .children
            .as_ref()
            .expect("install_subtree: non-leaf expected on descent");
        current_id = children[slot as usize];
    }

    // Incremental downsample: one slot changes per layer.
    let mut child_id = new_node_id;
    for &(parent_id, slot) in descent.iter().rev() {
        let (new_voxels, new_children) = {
            let parent = world
                .library
                .get(parent_id)
                .expect("install_subtree: parent missing");
            let old_children = parent
                .children
                .as_ref()
                .expect("install_subtree: non-leaf expected on walk up");
            let mut new_children = old_children.clone();
            new_children[slot] = child_id;
            let new_child_voxels = &world
                .library
                .get(child_id)
                .expect("install_subtree: new child missing")
                .voxels;
            let new_voxels = downsample_updated_slot(
                &parent.voxels,
                new_child_voxels,
                slot,
            );
            (new_voxels, new_children)
        };
        child_id = world.library.insert_non_leaf(new_voxels, new_children);
    }

    world.library.ref_inc(child_id);
    let old_root = world.root;
    world.root = child_id;
    world.library.ref_dec(old_root);
}

/// Read a leaf voxel by walking down the tree. Used by tests and
/// future collision / raycasting code.
pub fn get_voxel(world: &WorldState, position: &Position) -> Voxel {
    let mut current_id = world.root;
    for &slot in &position.path {
        let node = world
            .library
            .get(current_id)
            .expect("get_voxel: descent hit missing node");
        let children = node
            .children
            .as_ref()
            .expect("get_voxel: non-leaf expected on descent");
        current_id = children[slot as usize];
    }
    let leaf = world
        .library
        .get(current_id)
        .expect("get_voxel: missing leaf");
    leaf.voxels[voxel_idx(
        position.voxel[0] as usize,
        position.voxel[1] as usize,
        position.voxel[2] as usize,
    )]
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockType;
    use crate::world::tree::{voxel_from_block, EMPTY_VOXEL};

    fn grass() -> Voxel {
        voxel_from_block(Some(BlockType::Grass))
    }
    fn stone() -> Voxel {
        voxel_from_block(Some(BlockType::Stone))
    }

    #[test]
    fn edit_leaf_changes_root_id() {
        let mut world = WorldState::new_grassland();
        let initial_root = world.root;
        let mut pos = Position::origin();
        pos.voxel = [12, 5, 3];
        edit_leaf(&mut world, &pos, stone());
        assert_ne!(world.root, initial_root);
    }

    #[test]
    fn edit_leaf_places_requested_voxel() {
        let mut world = WorldState::new_grassland();
        let mut pos = Position::origin();
        pos.voxel = [12, 5, 3];
        assert_eq!(get_voxel(&world, &pos), grass());
        edit_leaf(&mut world, &pos, stone());
        assert_eq!(get_voxel(&world, &pos), stone());
    }

    #[test]
    fn other_voxels_unaffected() {
        let mut world = WorldState::new_grassland();
        let mut pos = Position::origin();
        pos.voxel = [12, 5, 3];
        edit_leaf(&mut world, &pos, stone());

        // Some other voxel in the same leaf should still be grass.
        let mut other = Position::origin();
        other.voxel = [1, 1, 1];
        assert_eq!(get_voxel(&world, &other), grass());
    }

    #[test]
    fn edit_leaf_round_trip_has_original_content() {
        let mut world = WorldState::new_grassland();
        let initial_len = world.library.len();

        let mut pos = Position::origin();
        pos.voxel = [12, 5, 3];
        edit_leaf(&mut world, &pos, stone());
        edit_leaf(&mut world, &pos, grass());

        // After a round trip, every sampled voxel is grass again.
        let mut p = Position::origin();
        for &vx in &[0u8, 3, 12, 20] {
            for &vy in &[0u8, 5, 17, 24] {
                p.voxel = [vx, vy, 7];
                assert_eq!(get_voxel(&world, &p), grass());
            }
        }
        // And the library should be back to its starting size.
        // (A round trip evicts all the intermediate edit-path nodes.)
        assert_eq!(world.library.len(), initial_len);
    }

    #[test]
    fn edit_at_layer_fills_whole_leaf() {
        let mut world = WorldState::new_grassland();
        // path_prefix = the full path (MAX_LAYER entries) replaces
        // one leaf entirely with stone.
        let path_prefix = [0u8; MAX_LAYER as usize];
        edit_at_layer(&mut world, &path_prefix, stone());

        // Every voxel inside the target leaf is stone.
        let mut pos = Position::origin();
        for &vx in &[0u8, 7, 12, 24] {
            for &vy in &[0u8, 7, 12, 24] {
                for &vz in &[0u8, 7, 12, 24] {
                    pos.voxel = [vx, vy, vz];
                    assert_eq!(get_voxel(&world, &pos), stone());
                }
            }
        }
    }

    #[test]
    fn edit_at_layer_higher_builds_shallow_chain() {
        let mut world = WorldState::new_grassland();
        // Target layer 6 (path_prefix length 6) — a small subtree.
        let path_prefix = [0u8; 6];
        edit_at_layer(&mut world, &path_prefix, stone());

        // Any leaf inside that subtree should now be stone.
        let mut pos = Position::origin();
        pos.voxel = [12, 12, 12];
        assert_eq!(get_voxel(&world, &pos), stone());
    }

    #[test]
    fn edit_at_layer_zero_replaces_entire_world() {
        let mut world = WorldState::new_grassland();
        edit_at_layer(&mut world, &[], stone());
        // Every leaf everywhere is now stone.
        let mut pos = Position::origin();
        pos.voxel = [0, 0, 0];
        assert_eq!(get_voxel(&world, &pos), stone());
        // After filling the root with stone, the new world is again
        // one library entry per layer.
        assert_eq!(world.library.len(), (MAX_LAYER as usize) + 1);
    }

    #[test]
    fn edit_leaf_paint_empty_changes_content() {
        let mut world = WorldState::new_grassland();
        let mut pos = Position::origin();
        pos.voxel = [3, 4, 5];
        edit_leaf(&mut world, &pos, EMPTY_VOXEL);
        assert_eq!(get_voxel(&world, &pos), EMPTY_VOXEL);
    }

    #[test]
    fn repeat_same_edit_is_noop() {
        let mut world = WorldState::new_grassland();
        let mut pos = Position::origin();
        pos.voxel = [10, 10, 10];
        edit_leaf(&mut world, &pos, stone());
        let mid_root = world.root;
        let mid_len = world.library.len();

        edit_leaf(&mut world, &pos, stone());
        assert_eq!(world.root, mid_root);
        assert_eq!(world.library.len(), mid_len);
    }

    // ----------------------------------------------- edit_at_layer_pos

    #[test]
    fn edit_at_layer_pos_leaf_layer_is_single_voxel_edit() {
        let mut world = WorldState::new_grassland();

        // Build a leaf position, project it to a LayerPos at MAX_LAYER.
        let mut leaf_pos = Position::origin();
        leaf_pos.voxel = [12, 5, 3];
        let lp = LayerPos::from_leaf(&leaf_pos, MAX_LAYER);
        edit_at_layer_pos(&mut world, &lp, stone());

        // The targeted voxel is stone.
        assert_eq!(get_voxel(&world, &leaf_pos), stone());
        // A sibling voxel in the same leaf is still grass.
        let mut sibling = Position::origin();
        sibling.voxel = [1, 1, 1];
        assert_eq!(get_voxel(&world, &sibling), grass());
    }

    #[test]
    fn edit_at_layer_pos_one_above_leaf_fills_5x5x5_region() {
        let mut world = WorldState::new_grassland();

        // Pick a LayerPos at layer MAX_LAYER - 1 inside the all-zero
        // path. Cell (2, 3, 4) maps to:
        //   child_slot = (2/5, 3/5, 4/5) = (0, 0, 0)
        //   region base = ((2%5)*5, (3%5)*5, (4%5)*5) = (10, 15, 20)
        let lp = LayerPos {
            path: vec![0; (MAX_LAYER - 1) as usize],
            cell: [2, 3, 4],
            layer: MAX_LAYER - 1,
        };
        edit_at_layer_pos(&mut world, &lp, stone());

        // Every voxel in the 5³ region [10..15, 15..20, 20..25] of
        // the all-zero-path leaf should be stone.
        let mut p = Position::origin();
        for &x in &[10u8, 11, 12, 13, 14] {
            for &y in &[15u8, 16, 17, 18, 19] {
                for &z in &[20u8, 21, 22, 23, 24] {
                    p.voxel = [x, y, z];
                    assert_eq!(get_voxel(&world, &p), stone());
                }
            }
        }

        // A voxel outside the 5³ region is still grass.
        p.voxel = [0, 0, 0];
        assert_eq!(get_voxel(&world, &p), grass());
        p.voxel = [9, 15, 20];
        assert_eq!(get_voxel(&world, &p), grass());
    }

    #[test]
    fn edit_at_layer_pos_higher_layer_replaces_subtree() {
        let mut world = WorldState::new_grassland();
        // View layer MAX_LAYER - 2 = 10. Cell (0, 0, 0) maps to
        // slot_a = (0, 0, 0), slot_b = (0, 0, 0). Replaces the
        // layer-12 (leaf) subtree at path [0; 10, 0, 0] with a
        // solid-stone leaf.
        let lp = LayerPos {
            path: vec![0; (MAX_LAYER - 2) as usize],
            cell: [0, 0, 0],
            layer: MAX_LAYER - 2,
        };
        edit_at_layer_pos(&mut world, &lp, stone());

        // The all-zero-path leaf is now solid stone.
        let mut p = Position::origin();
        for &vx in &[0u8, 7, 12, 24] {
            for &vy in &[0u8, 7, 12, 24] {
                for &vz in &[0u8, 7, 12, 24] {
                    p.voxel = [vx, vy, vz];
                    assert_eq!(get_voxel(&world, &p), stone());
                }
            }
        }
    }

    #[test]
    fn edit_at_layer_pos_higher_layer_with_nonzero_cell_picks_correct_subtree() {
        let mut world = WorldState::new_grassland();
        // View layer 10, cell (6, 2, 0) decomposes as:
        //   slot_a = (6/5, 2/5, 0/5) = (1, 0, 0)
        //   slot_b = (6%5, 2%5, 0%5) = (1, 2, 0)
        // So the target leaf is at path [0; 10, slot(1,0,0), slot(1,2,0)].
        let lp = LayerPos {
            path: vec![0; 10],
            cell: [6, 2, 0],
            layer: 10,
        };
        edit_at_layer_pos(&mut world, &lp, stone());

        // Construct the matching leaf Position and confirm it reads
        // back as stone (the whole leaf was replaced).
        let mut p = Position::origin();
        p.path[10] = crate::world::tree::slot_index(1, 0, 0) as u8;
        p.path[11] = crate::world::tree::slot_index(1, 2, 0) as u8;
        for &vx in &[0u8, 12, 24] {
            for &vy in &[0u8, 12, 24] {
                for &vz in &[0u8, 12, 24] {
                    p.voxel = [vx, vy, vz];
                    assert_eq!(get_voxel(&world, &p), stone());
                }
            }
        }

        // A different leaf (path [0; 12]) is unaffected and still grass.
        let mut sibling = Position::origin();
        sibling.voxel = [12, 12, 12];
        assert_eq!(get_voxel(&world, &sibling), grass());
    }
}
