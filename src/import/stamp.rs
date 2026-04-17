//! Stamp a [`VoxelModel`] into [`WorldState`] at leaf resolution.
//!
//! The model's `(0, 0, 0)` corner is placed at the supplied
//! [`Position`]'s leaf voxel. The model is then chunked into 25³
//! leaf-sized tiles; each tile is inserted into the library and
//! installed into the tree via [`install_subtree`].
//!
//! Voxels in the model that are [`EMPTY_VOXEL`] are **transparent** —
//! they preserve whatever the world already has at that location
//! rather than overwriting it with air. This lets you stamp a
//! non-rectangular model without carving a box-shaped hole.

use crate::world::edit::install_subtree;
use crate::world::position::Position;
use crate::world::state::WorldState;
use crate::world::tree::{
    voxel_idx, NodeId, EMPTY_VOXEL, NODE_VOXELS_PER_AXIS,
};

use super::VoxelModel;

/// Stamp `model` into `world` with its origin at `anchor`.
///
/// Returns the number of leaves that were actually modified (useful
/// for diagnostics / progress reporting). Leaves where every model
/// voxel is `EMPTY_VOXEL` are skipped entirely.
pub fn stamp_model(world: &mut WorldState, anchor: &Position, model: &VoxelModel) -> usize {
    let nvpa = NODE_VOXELS_PER_AXIS;

    // How many leaves does the model span along each axis?
    let leaves_x = (model.size_x + nvpa - 1) / nvpa;
    let leaves_y = (model.size_y + nvpa - 1) / nvpa;
    let leaves_z = (model.size_z + nvpa - 1) / nvpa;

    let mut modified = 0usize;

    for lz in 0..leaves_z {
        for ly in 0..leaves_y {
            for lx in 0..leaves_x {
                // Model-local voxel range covered by this leaf tile.
                let mx0 = lx * nvpa;
                let my0 = ly * nvpa;
                let mz0 = lz * nvpa;

                // Quick check: does this tile contain any non-empty
                // model voxels? If not, skip it entirely.
                if !tile_has_content(model, mx0, my0, mz0) {
                    continue;
                }

                // Compute the world Position of this tile's (0,0,0)
                // corner by stepping from the anchor.
                let Some(tile_pos) = offset_position(
                    anchor,
                    mx0 as i32,
                    my0 as i32,
                    mz0 as i32,
                ) else {
                    continue; // walked off the world edge
                };

                // Read the existing leaf at this position so we can
                // merge (preserve existing voxels where the model is
                // transparent).
                let (leaf_path, existing_id) = descend_to_leaf(world, &tile_pos);
                let mut new_voxels = world
                    .library
                    .get(existing_id)
                    .expect("stamp: leaf missing")
                    .voxels
                    .clone();

                let mut any_changed = false;
                for dz in 0..nvpa {
                    let mz = mz0 + dz;
                    if mz >= model.size_z {
                        break;
                    }
                    for dy in 0..nvpa {
                        let my = my0 + dy;
                        if my >= model.size_y {
                            break;
                        }
                        for dx in 0..nvpa {
                            let mx = mx0 + dx;
                            if mx >= model.size_x {
                                break;
                            }
                            let v = model.get(mx, my, mz);
                            if v != EMPTY_VOXEL {
                                let idx = voxel_idx(dx, dy, dz);
                                if new_voxels[idx] != v {
                                    new_voxels[idx] = v;
                                    any_changed = true;
                                }
                            }
                        }
                    }
                }

                if !any_changed {
                    continue;
                }

                let new_leaf_id = world.library.insert_leaf(new_voxels);
                install_subtree(world, &leaf_path, new_leaf_id);
                modified += 1;
            }
        }
    }

    modified
}

/// Check whether any model voxel in the 25³ tile starting at
/// `(mx0, my0, mz0)` is non-empty.
fn tile_has_content(model: &VoxelModel, mx0: usize, my0: usize, mz0: usize) -> bool {
    let nvpa = NODE_VOXELS_PER_AXIS;
    for dz in 0..nvpa {
        let mz = mz0 + dz;
        if mz >= model.size_z {
            break;
        }
        for dy in 0..nvpa {
            let my = my0 + dy;
            if my >= model.size_y {
                break;
            }
            for dx in 0..nvpa {
                let mx = mx0 + dx;
                if mx >= model.size_x {
                    break;
                }
                if model.get(mx, my, mz) != EMPTY_VOXEL {
                    return true;
                }
            }
        }
    }
    false
}

/// Offset a `Position` by `(dx, dy, dz)` leaf voxels. Returns `None`
/// if the result walks off the world boundary.
fn offset_position(base: &Position, dx: i32, dy: i32, dz: i32) -> Option<Position> {
    let mut p = *base;
    if !p.step_voxels(0, dx) {
        return None;
    }
    if !p.step_voxels(1, dy) {
        return None;
    }
    if !p.step_voxels(2, dz) {
        return None;
    }
    // Snap sub-voxel offset to zero — we want the integer corner.
    p.offset = [0.0; 3];
    Some(p)
}

/// Walk from the root down to the leaf that contains `pos` and return
/// `(leaf_path, leaf_node_id)`.
fn descend_to_leaf(world: &WorldState, pos: &Position) -> (Vec<u8>, NodeId) {
    let path: Vec<u8> = pos.path.to_vec();
    let mut current_id = world.root;
    for &slot in &path {
        let node = world
            .library
            .get(current_id)
            .expect("descend_to_leaf: missing node on descent");
        let children = node
            .children
            .as_ref()
            .expect("descend_to_leaf: non-leaf expected");
        current_id = children[slot as usize];
    }
    (path, current_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockType;
    use crate::world::edit::get_voxel;
    use crate::world::tree::{voxel_from_block, EMPTY_VOXEL};

    fn stone() -> u8 {
        voxel_from_block(Some(BlockType::Stone))
    }

    /// Stamp a tiny 3×3×3 model at the origin and verify the voxels
    /// land in the right place.
    #[test]
    fn stamp_small_model_at_origin() {
        let mut world = WorldState::default();
        let anchor = Position::origin();

        let size = 3;
        let v = stone();
        let data = vec![v; size * size * size];
        let model = VoxelModel {
            size_x: size,
            size_y: size,
            size_z: size,
            data,
        };

        let modified = stamp_model(&mut world, &anchor, &model);
        assert_eq!(modified, 1); // fits in one leaf

        // Check the stamped voxels.
        for dz in 0..size {
            for dy in 0..size {
                for dx in 0..size {
                    let mut p = Position::origin();
                    p.voxel = [dx as u8, dy as u8, dz as u8];
                    assert_eq!(get_voxel(&world, &p), v);
                }
            }
        }

        // A voxel outside the model is still grass (the default world).
        let mut outside = Position::origin();
        outside.voxel = [10, 10, 10];
        assert_eq!(
            get_voxel(&world, &outside),
            voxel_from_block(Some(BlockType::Grass))
        );
    }

    /// An all-empty model should modify zero leaves.
    #[test]
    fn stamp_empty_model_is_noop() {
        let mut world = WorldState::default();
        let anchor = Position::origin();
        let model = VoxelModel {
            size_x: 10,
            size_y: 10,
            size_z: 10,
            data: vec![EMPTY_VOXEL; 1000],
        };
        let modified = stamp_model(&mut world, &anchor, &model);
        assert_eq!(modified, 0);
    }

    /// A model larger than 25 spans multiple leaves.
    #[test]
    fn stamp_multi_leaf_model() {
        let mut world = WorldState::default();
        let anchor = Position::origin();

        let size = 30; // spans 2 leaves on each axis
        let v = stone();
        let data = vec![v; size * size * size];
        let model = VoxelModel {
            size_x: size,
            size_y: size,
            size_z: size,
            data,
        };

        let modified = stamp_model(&mut world, &anchor, &model);
        assert!(modified > 1, "expected multiple leaves, got {modified}");

        // Check a voxel in the second leaf along x.
        let mut p = Position::origin();
        p.step_voxels(0, 26); // past the first leaf's 25 voxels
        assert_eq!(get_voxel(&world, &p), v);
    }

    /// Transparent model voxels (EMPTY_VOXEL) preserve the existing
    /// world content.
    #[test]
    fn stamp_preserves_existing_on_transparent() {
        let mut world = WorldState::default();
        let anchor = Position::origin();

        let mut data = vec![EMPTY_VOXEL; 25 * 25 * 25];
        // Only set voxel (0, 0, 0) to stone.
        data[0] = stone();
        let model = VoxelModel {
            size_x: 25,
            size_y: 25,
            size_z: 25,
            data,
        };

        stamp_model(&mut world, &anchor, &model);

        // (0,0,0) is stone.
        let mut p = Position::origin();
        p.voxel = [0, 0, 0];
        assert_eq!(get_voxel(&world, &p), stone());

        // (1,0,0) is still grass (the model had EMPTY_VOXEL there).
        p.voxel = [1, 0, 0];
        assert_eq!(
            get_voxel(&world, &p),
            voxel_from_block(Some(BlockType::Grass))
        );
    }

}
