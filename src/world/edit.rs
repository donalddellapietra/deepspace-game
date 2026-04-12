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
/// from `lp.path()` (length `lp.layer`), push one or two extra slots
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
    let mut path: Vec<u8> = lp.path().to_vec();
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

/// Descend `path` (length `0..=NODE_PATH_LEN`) from `world.root` and
/// return the id of the node living at that path. A zero-length path
/// returns the root. Panics if the descent hits a missing node or a
/// premature leaf.
fn descend_to(world: &WorldState, path: &[u8]) -> NodeId {
    let mut current_id = world.root;
    for &slot in path {
        let node = world
            .library
            .get(current_id)
            .expect("descend_to: descent hit a missing node");
        let children = node
            .children
            .as_ref()
            .expect("descend_to: non-leaf expected on descent");
        current_id = children[slot as usize];
    }
    current_id
}

/// Build (or recycle via dedup) a "solid `voxel`" subtree rooted at
/// `target_layer` and return its `NodeId`. `target_layer == MAX_LAYER`
/// returns just the solid leaf; each layer above adds one non-leaf
/// whose 125 children are all the layer-below id.
///
/// Thanks to content dedup, repeated calls with the same `voxel` and
/// `target_layer` return the same id without allocating new library
/// entries — this is how layer-K edits stay cheap.
fn build_solid_chain(
    world: &mut WorldState,
    voxel: Voxel,
    target_layer: usize,
) -> NodeId {
    assert!(target_layer <= MAX_LAYER as usize);
    let leaf_voxels = filled_voxel_grid(voxel);
    let mut chain_id = world.library.insert_leaf(leaf_voxels);
    let mut chain_layer = MAX_LAYER as usize;
    while chain_layer > target_layer {
        let voxels = {
            let node = world
                .library
                .get(chain_id)
                .expect("build_solid_chain: chain node missing");
            let refs: [&VoxelGrid; CHILDREN_PER_NODE] =
                std::array::from_fn(|_| &node.voxels);
            downsample(refs)
        };
        let children = uniform_children(chain_id);
        chain_id = world.library.insert_non_leaf(voxels, children);
        chain_layer -= 1;
    }
    chain_id
}

/// Apply an edit named by a [`LayerPos`]. Dispatches on the view
/// layer to one of three shapes, each of which computes a
/// `(path, new_node_id)` pair and then routes through
/// [`install_subtree`]:
///
/// - **`lp.layer == MAX_LAYER`** (leaf view): single-cell edit. Walk
///   down to the target leaf, clone its voxel grid, patch the one
///   cell, intern the new leaf, and install it at the full leaf path.
/// - **`lp.layer == MAX_LAYER - 1`**: fills the `5³` region inside the
///   leaf one level below the view node that this cell summarises.
///   Walk down to that leaf, clone, fill, intern, install.
/// - **`lp.layer <= MAX_LAYER - 2`**: replaces the whole layer-`(L + 2)`
///   subtree beneath the clicked cell with a canonical "solid X"
///   chain. The click's `(cx, cy, cz)` decomposes into two slot
///   steps: `slot_a = (c / 5)` and `slot_b = (c % 5)`.
///
/// See `docs/architecture/editing.md` and the 2D prototype's
/// `World::edit_at` for the derivation of the three cases.
pub fn edit_at_layer_pos(world: &mut WorldState, lp: &LayerPos, voxel: Voxel) {
    assert!(lp.layer as usize == lp.path().len());
    assert!(lp.layer <= MAX_LAYER);
    assert!(lp.cell[0] < NODE_VOXELS_PER_AXIS as u8);
    assert!(lp.cell[1] < NODE_VOXELS_PER_AXIS as u8);
    assert!(lp.cell[2] < NODE_VOXELS_PER_AXIS as u8);

    let cx = lp.cell[0];
    let cy = lp.cell[1];
    let cz = lp.cell[2];
    let b = BRANCH_FACTOR as u8;

    if lp.layer == MAX_LAYER {
        // Leaf-view: single-voxel edit. Read the target leaf, clone
        // its voxel grid, patch one cell, intern, install.
        let leaf_path = lp.path();
        debug_assert_eq!(leaf_path.len(), NODE_PATH_LEN);
        let leaf_id = descend_to(world, leaf_path);
        let mut new_voxels: VoxelGrid = world
            .library
            .get(leaf_id)
            .expect("edit_at_layer_pos: leaf missing")
            .voxels
            .clone();
        new_voxels[voxel_idx(cx as usize, cy as usize, cz as usize)] = voxel;
        let new_leaf_id = world.library.insert_leaf(new_voxels);
        install_subtree(world, leaf_path, new_leaf_id);
        return;
    }

    if lp.layer == MAX_LAYER - 1 {
        // One layer above leaves: the clicked cell summarises a 5³
        // region of one specific leaf below us. Walk down into that
        // leaf, clone its voxels, fill the 5³ region, intern, install.
        let child_slot = slot_index(
            (cx / b) as usize,
            (cy / b) as usize,
            (cz / b) as usize,
        );
        let mut leaf_path_vec: Vec<u8> = lp.path().to_vec();
        leaf_path_vec.push(child_slot as u8);
        debug_assert_eq!(leaf_path_vec.len(), NODE_PATH_LEN);

        let leaf_id = descend_to(world, &leaf_path_vec);
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
    let mut sub_path: Vec<u8> = lp.path().to_vec();
    sub_path.push(slot_a as u8);
    sub_path.push(slot_b as u8);
    let target_layer = (lp.layer + 2) as usize;

    let chain_id = build_solid_chain(world, voxel, target_layer);
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

    world.swap_root(child_id);
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

    /// Helper: project a leaf `Position` into a `LayerPos` at
    /// `MAX_LAYER` for the leaf-case tests that used to call
    /// `edit_leaf` directly.
    fn edit_single_leaf_voxel(world: &mut WorldState, pos: &Position, voxel: Voxel) {
        let lp = LayerPos::from_leaf(pos, MAX_LAYER);
        edit_at_layer_pos(world, &lp, voxel);
    }

    #[test]
    fn edit_leaf_changes_root_id() {
        let mut world = WorldState::new_grassland();
        let initial_root = world.root;
        let mut pos = Position::origin();
        pos.voxel = [12, 5, 3];
        edit_single_leaf_voxel(&mut world, &pos, stone());
        assert_ne!(world.root, initial_root);
    }

    #[test]
    fn edit_leaf_places_requested_voxel() {
        let mut world = WorldState::new_grassland();
        let mut pos = Position::origin();
        pos.voxel = [12, 5, 3];
        assert_eq!(get_voxel(&world, &pos), grass());
        edit_single_leaf_voxel(&mut world, &pos, stone());
        assert_eq!(get_voxel(&world, &pos), stone());
    }

    #[test]
    fn other_voxels_unaffected() {
        let mut world = WorldState::new_grassland();
        let mut pos = Position::origin();
        pos.voxel = [12, 5, 3];
        edit_single_leaf_voxel(&mut world, &pos, stone());

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
        edit_single_leaf_voxel(&mut world, &pos, stone());
        edit_single_leaf_voxel(&mut world, &pos, grass());

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
    fn install_subtree_full_path_replaces_one_leaf() {
        // `install_subtree` with a full-length path splices a new leaf
        // into exactly one leaf slot without disturbing its siblings.
        // Covers the same scenario as the old `edit_at_layer` test for
        // a path prefix of length `MAX_LAYER` — the whole leaf gets
        // replaced with solid stone.
        let mut world = WorldState::new_grassland();
        let full_path = [0u8; MAX_LAYER as usize];
        let solid_stone_leaf = build_solid_chain(&mut world, stone(), MAX_LAYER as usize);
        install_subtree(&mut world, &full_path, solid_stone_leaf);

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
        // Layer-4 view cell (0, 0, 0) decomposes to slot_a = slot_b = 0,
        // so the target subtree is at path [0; 6] — layer 6, matching
        // the old `edit_at_layer(&[0; 6], ...)` test.
        let lp = LayerPos::from_parts(&vec![0u8; 4], [0, 0, 0], 4);
        edit_at_layer_pos(&mut world, &lp, stone());

        // Any leaf inside that subtree should now be stone.
        let mut pos = Position::origin();
        pos.voxel = [12, 12, 12];
        assert_eq!(get_voxel(&world, &pos), stone());
    }

    #[test]
    fn install_subtree_at_root_replaces_entire_world() {
        // `install_subtree` is public (save-mode uses it) and callable
        // with an empty ancestor path, which replaces the whole world
        // root. `edit_at_layer_pos` can't reach target layer 0 itself
        // — its minimum is `lp.layer + 2 == 2` — so this exercises
        // `install_subtree`'s empty-descent path directly.
        let mut world = WorldState::new_grassland();
        let root_chain = build_solid_chain(&mut world, stone(), 0);
        install_subtree(&mut world, &[], root_chain);

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
        edit_single_leaf_voxel(&mut world, &pos, EMPTY_VOXEL);
        assert_eq!(get_voxel(&world, &pos), EMPTY_VOXEL);
    }

    #[test]
    fn repeat_same_edit_is_noop() {
        let mut world = WorldState::new_grassland();
        let mut pos = Position::origin();
        pos.voxel = [10, 10, 10];
        edit_single_leaf_voxel(&mut world, &pos, stone());
        let mid_root = world.root;
        let mid_len = world.library.len();

        edit_single_leaf_voxel(&mut world, &pos, stone());
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
        let lp = LayerPos::from_parts(
            &vec![0; (MAX_LAYER - 1) as usize],
            [2, 3, 4],
            MAX_LAYER - 1,
        );
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
        let lp = LayerPos::from_parts(
            &vec![0; (MAX_LAYER - 2) as usize],
            [0, 0, 0],
            MAX_LAYER - 2,
        );
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
        let lp = LayerPos::from_parts(&vec![0; 10], [6, 2, 0], 10);
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
