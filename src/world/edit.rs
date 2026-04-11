//! Edit walks: single-voxel leaf edits and bulk higher-layer edits.
//!
//! See `docs/architecture/editing.md` for the full design. Every
//! edit walks up the tree from the touched layer to the root,
//! minting a fresh `NodeId` at each ancestor via the library. The
//! library's content dedup means most ancestors recycle existing
//! ids when the change is small.

use super::state::WorldState;
use super::position::{Position, NODE_PATH_LEN};
use super::tree::{
    downsample, downsample_from_library, filled_voxel_grid, uniform_children,
    voxel_idx, NodeId, Voxel, VoxelGrid, CHILDREN_PER_NODE, MAX_LAYER,
};

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
    for &(parent_id, slot) in descent.iter().rev() {
        let new_children = {
            let parent = world
                .library
                .get(parent_id)
                .expect("edit_leaf: parent missing");
            let old_children = parent
                .children
                .as_ref()
                .expect("edit_leaf: non-leaf expected on walk up");
            let mut c = old_children.clone();
            c[slot] = new_child_id;
            c
        };
        let new_voxels =
            downsample_from_library(&world.library, new_children.as_ref());
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
    let mut new_child_id = chain_id;
    for &(parent_id, slot) in descent.iter().rev() {
        let new_children = {
            let parent = world
                .library
                .get(parent_id)
                .expect("edit_at_layer: parent missing");
            let old_children = parent
                .children
                .as_ref()
                .expect("edit_at_layer: non-leaf expected on walk up");
            let mut c = old_children.clone();
            c[slot] = new_child_id;
            c
        };
        let new_voxels =
            downsample_from_library(&world.library, new_children.as_ref());
        new_child_id = world.library.insert_non_leaf(new_voxels, new_children);
    }

    world.library.ref_inc(new_child_id);
    let old_root = world.root;
    world.root = new_child_id;
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
}
