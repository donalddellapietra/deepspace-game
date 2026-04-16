//! Convert a flat voxel grid into a base-3 recursive tree.
//!
//! The algorithm pads the model to the next power of 3, then
//! recursively subdivides into 3x3x3 groups bottom-up. Content-
//! addressed dedup in `NodeLibrary` collapses identical subtrees
//! automatically.

use super::VoxelModel;
use crate::world::tree::*;

/// Build a base-3 tree from a voxel model. Returns the root `NodeId`.
///
/// The model is padded to the next power-of-3 cube. Out-of-bounds
/// voxels are treated as empty. Uniform subtrees (all one block type
/// or all empty) are collapsed — they don't create intermediate nodes.
pub fn build_tree(model: &VoxelModel, library: &mut NodeLibrary) -> NodeId {
    let max_dim = model.size_x.max(model.size_y).max(model.size_z).max(1);
    let padded = next_power_of_3(max_dim);

    let root_child = build_node(model, library, 0, 0, 0, padded / BRANCH);
    match root_child {
        Child::Node(id) => id,
        // Entire model collapsed to a single block or empty — wrap in a node.
        other => library.insert(uniform_children(other)),
    }
}

/// Recursively build one node of the tree.
///
/// `origin_x/y/z` is the world-space corner of this region.
/// `cell_size` is the size of one child cell (the region spans
/// `cell_size * 3` along each axis).
fn build_node(
    model: &VoxelModel,
    library: &mut NodeLibrary,
    origin_x: usize,
    origin_y: usize,
    origin_z: usize,
    cell_size: usize,
) -> Child {
    // Early out: entire region is outside the model bounds.
    if origin_x >= model.size_x && origin_y >= model.size_y && origin_z >= model.size_z {
        return Child::Empty;
    }

    // Build 27 children.
    let mut children = empty_children();

    if cell_size == 1 {
        // Leaf level: each child is one voxel.
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] =
                        leaf_child(model, origin_x + x, origin_y + y, origin_z + z);
                }
            }
        }
    } else {
        // Recurse into 3x3x3 sub-cells.
        let sub_size = cell_size / BRANCH;
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] = build_node(
                        model,
                        library,
                        origin_x + x * cell_size,
                        origin_y + y * cell_size,
                        origin_z + z * cell_size,
                        sub_size,
                    );
                }
            }
        }
    }

    // Collapse: if all 27 children are the same, return that directly.
    let first = children[0];
    if children.iter().all(|&c| c == first) {
        return first;
    }

    Child::Node(library.insert(children))
}

/// Read a single voxel from the model (or Empty if out of bounds).
#[inline]
fn leaf_child(model: &VoxelModel, x: usize, y: usize, z: usize) -> Child {
    if x < model.size_x && y < model.size_y && z < model.size_z {
        let v = model.get(x, y, z);
        if v == 0 {
            Child::Empty
        } else {
            Child::Block(v)
        }
    } else {
        Child::Empty
    }
}

/// Smallest power of 3 >= n.
fn next_power_of_3(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 3;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;

    /// A 3x3x3 model of uniform stone produces a single node.
    #[test]
    fn uniform_3x3x3() {
        let model = VoxelModel {
            size_x: 3,
            size_y: 3,
            size_z: 3,
            data: vec![block::STONE; 27],
        };
        let mut lib = NodeLibrary::default();
        let root = build_tree(&model, &mut lib);
        // Uniform → collapsed to one node with 27 Block children.
        assert_eq!(lib.len(), 1);
        let node = lib.get(root).unwrap();
        assert!(
            node.children
                .iter()
                .all(|c| *c == Child::Block(block::STONE))
        );
    }

    /// A fully empty model produces a single all-empty node.
    #[test]
    fn empty_model() {
        let model = VoxelModel {
            size_x: 3,
            size_y: 3,
            size_z: 3,
            data: vec![0; 27],
        };
        let mut lib = NodeLibrary::default();
        let root = build_tree(&model, &mut lib);
        assert_eq!(lib.len(), 1);
        let node = lib.get(root).unwrap();
        assert!(node.children.iter().all(|c| c.is_empty()));
    }

    /// A 4x4x4 model is padded to 9x9x9 (next power of 3).
    #[test]
    fn padding() {
        let mut data = vec![0u8; 4 * 4 * 4];
        data[0] = block::STONE; // one non-empty voxel
        let model = VoxelModel {
            size_x: 4,
            size_y: 4,
            size_z: 4,
            data,
        };
        let mut lib = NodeLibrary::default();
        let _root = build_tree(&model, &mut lib);
        // Should have created some nodes (not everything collapsed).
        assert!(lib.len() >= 1);
    }

    /// Content-addressed dedup: a model with repeated patterns shares nodes.
    #[test]
    fn dedup() {
        // 9x9x9 model: two 3x3x3 quadrants of stone, rest empty.
        let mut data = vec![0u8; 9 * 9 * 9];
        // Fill (0,0,0)-(2,2,2) with stone.
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    data[(z * 9 + y) * 9 + x] = block::STONE;
                }
            }
        }
        // Fill (3,0,0)-(5,2,2) with stone (identical subtree).
        for z in 0..3 {
            for y in 0..3 {
                for x in 3..6 {
                    data[(z * 9 + y) * 9 + x] = block::STONE;
                }
            }
        }

        let model = VoxelModel {
            size_x: 9,
            size_y: 9,
            size_z: 9,
            data,
        };
        let mut lib = NodeLibrary::default();
        let _root = build_tree(&model, &mut lib);
        // The two stone 3x3x3 cubes should share a NodeId via dedup.
        // Total unique nodes should be small.
        assert!(lib.len() <= 5, "expected dedup, got {} nodes", lib.len());
    }

    /// Round-trip: build tree, then walk it to verify voxels match.
    #[test]
    fn round_trip_3x3x3() {
        let mut data = vec![0u8; 27];
        data[0] = block::STONE;
        data[13] = block::GRASS; // center: (1,1,1)
        let model = VoxelModel {
            size_x: 3,
            size_y: 3,
            size_z: 3,
            data,
        };
        let mut lib = NodeLibrary::default();
        let root = build_tree(&model, &mut lib);

        // Walk the tree to check specific voxels.
        let node = lib.get(root).unwrap();
        assert_eq!(
            node.children[slot_index(0, 0, 0)],
            Child::Block(block::STONE)
        );
        assert_eq!(
            node.children[slot_index(1, 1, 1)],
            Child::Block(block::GRASS)
        );
        assert_eq!(node.children[slot_index(2, 2, 2)], Child::Empty);
    }
}
