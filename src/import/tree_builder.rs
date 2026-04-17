//! Convert a flat voxel grid into a base-3 recursive tree.
//!
//! The algorithm pads the model to the next power of 3, then
//! recursively subdivides into 3×3×3 groups bottom-up. Content-
//! addressed dedup in `NodeLibrary` collapses identical subtrees
//! automatically.
//!
//! ## Brick materialization
//!
//! When a region fits in a brick losslessly (its content is entirely
//! terminal voxel data), we insert a `NodeKind::Brick` with dense
//! byte storage instead of a nested Cartesian tree. This replaces
//! `log3(side)` levels of recursion with one flat grid — dramatically
//! fewer GPU buffer reads per ray and no per-cell tag dispatch.
//!
//! `MAX_BRICK_SIDE` picks the largest brick we'll emit. Supported
//! values are `3`, `9`, and `27` (1-3 levels of recursion collapsed
//! per brick). Higher values use more memory per brick but cover more
//! voxels per descent.

use crate::world::tree::*;
use super::VoxelModel;

/// Largest brick side to materialize during import. Must be one of
/// `BRICK_SIDES` (3, 9, or 27). `9` is the default: a 9³ brick
/// carries 729 voxels in 183 u32s of GPU storage vs 2196+ u32s of
/// equivalent sparse nodes, and a ray through a 9³ brick amortizes
/// the per-entry setup cost over meaningfully more cells.
pub const MAX_BRICK_SIDE: u8 = 9;

/// Content depth of the library node at `id`, in the sense
/// `total_depth = wraps + content_depth(root)` so voxels land at the
/// same world depth whether or not bricks were materialized.
///
/// - `NodeKind::Brick { side = S }` → `log3(S)` content levels (one
///   tree node, `log3(S)` content levels).
/// - `NodeKind::Cartesian/Sphere/Face` → `1 + max_over_children`.
/// - A leaf `Child::Block` / `Child::Empty` contributes 0 (no
///   subdivision).
pub fn content_depth(library: &NodeLibrary, id: NodeId) -> u8 {
    let Some(node) = library.get(id) else { return 0 };
    if node.is_brick() {
        // A side-S brick carries log3(S) levels of world-tree
        // subdivision in a single tree node: side=3 replaces 1
        // Cartesian level, side=9 replaces 2, side=27 replaces 3.
        // The total `content_depth` the brick contributes equals
        // the number of Cartesian levels it stands in for.
        let mut log3 = 0u8;
        let mut s = node.brick_side as u32;
        while s > 1 { s /= 3; log3 += 1; }
        return log3;
    }
    let mut max_child = 0u8;
    for child in &node.children {
        let d = match child {
            Child::Empty | Child::Block(_) => 0,
            Child::Node(cid) => content_depth(library, *cid),
        };
        if d > max_child { max_child = d; }
    }
    1 + max_child
}

/// Build a base-3 tree from a voxel model. Returns the root `NodeId`.
///
/// The model is padded to the next power-of-3 cube. Out-of-bounds
/// voxels are treated as empty. Uniform subtrees (all one block type
/// or all empty) collapse to a single `Child`. Subtrees that fit in
/// a brick become `NodeKind::Brick` (unless `DEEPSPACE_NO_BRICK` is
/// set, in which case pure-Cartesian output is produced for A/B).
pub fn build_tree(model: &VoxelModel, library: &mut NodeLibrary) -> NodeId {
    let max_side = if std::env::var("DEEPSPACE_NO_BRICK").is_ok() {
        // Emit no bricks — fall back to the "collapse-to-single-block"
        // path by requiring a side larger than any voxel region we
        // ever process. We pick 3 but the build_node code also has to
        // honour the env var; handled via explicit path below.
        0
    } else {
        match std::env::var("DEEPSPACE_BRICK_SIDE").ok().as_deref() {
            Some("3") => 3,
            Some("9") => 9,
            Some("27") => 27,
            _ => MAX_BRICK_SIDE,
        }
    };
    if max_side == 0 {
        build_tree_no_bricks(model, library)
    } else {
        build_tree_with_max_brick_side(model, library, max_side)
    }
}

/// Build a pure-Cartesian tree (no brick materialization). Used by
/// the `DEEPSPACE_NO_BRICK` A/B path so we can compare bricks vs
/// fully-sparse visually and perf-wise from a single binary.
pub fn build_tree_no_bricks(
    model: &VoxelModel,
    library: &mut NodeLibrary,
) -> NodeId {
    let max_dim = model.size_x.max(model.size_y).max(model.size_z).max(1);
    let padded = next_power_of_3(max_dim);
    let root_child = build_node_cartesian(model, library, 0, 0, 0, padded / BRANCH);
    match root_child {
        Child::Node(id) => id,
        other => library.insert(uniform_children(other)),
    }
}

/// Cartesian-only build (no brick path). Structurally identical to
/// the pre-brickmap builder: leaf level fills `[Child; 27]` with
/// terminal voxels, intermediate levels recurse.
fn build_node_cartesian(
    model: &VoxelModel,
    library: &mut NodeLibrary,
    origin_x: usize,
    origin_y: usize,
    origin_z: usize,
    cell_size: usize,
) -> Child {
    if origin_x >= model.size_x && origin_y >= model.size_y && origin_z >= model.size_z {
        return Child::Empty;
    }
    let mut children = empty_children();
    if cell_size == 1 {
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] = leaf_child(
                        model, origin_x + x, origin_y + y, origin_z + z,
                    );
                }
            }
        }
    } else {
        let sub = cell_size / BRANCH;
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] = build_node_cartesian(
                        model, library,
                        origin_x + x * cell_size,
                        origin_y + y * cell_size,
                        origin_z + z * cell_size,
                        sub,
                    );
                }
            }
        }
    }
    let first = children[0];
    if matches!(first, Child::Empty | Child::Block(_))
        && children.iter().all(|&c| c == first)
    {
        return first;
    }
    Child::Node(library.insert(children))
}

#[inline]
fn leaf_child(model: &VoxelModel, x: usize, y: usize, z: usize) -> Child {
    if x < model.size_x && y < model.size_y && z < model.size_z {
        let v = model.get(x, y, z);
        if v == 0 { Child::Empty } else { Child::Block(v) }
    } else {
        Child::Empty
    }
}

/// `build_tree` with an explicit max brick side (for tests and A/B).
pub fn build_tree_with_max_brick_side(
    model: &VoxelModel,
    library: &mut NodeLibrary,
    max_brick_side: u8,
) -> NodeId {
    assert!(
        BRICK_SIDES.contains(&max_brick_side),
        "invalid max_brick_side {max_brick_side}; must be one of {:?}",
        BRICK_SIDES,
    );

    let max_dim = model.size_x.max(model.size_y).max(model.size_z).max(1);
    let padded = next_power_of_3(max_dim);

    let root_child = build_node(model, library, 0, 0, 0, padded / BRANCH, max_brick_side);
    match root_child {
        Child::Node(id) => id,
        // Entire model collapsed to a single value — wrap it.
        other => library.insert(uniform_children(other)),
    }
}

/// Recursively build one tree region.
///
/// `origin_{x,y,z}` is the region's world-space corner; `cell_size`
/// is the size of one child cell (the region spans `cell_size * 3`
/// along each axis). Returns a `Child` representing the entire region:
///
/// - `Child::Empty` if the region is fully empty
/// - `Child::Block(bt)` if it's uniformly one block type
/// - `Child::Node(id)` otherwise, where `id` may be a `Brick`
///   (terminal content fits in a brick ≤ `max_brick_side`) or a
///   `Cartesian` node (mixed subtrees).
fn build_node(
    model: &VoxelModel,
    library: &mut NodeLibrary,
    origin_x: usize,
    origin_y: usize,
    origin_z: usize,
    cell_size: usize,
    max_brick_side: u8,
) -> Child {
    if origin_x >= model.size_x && origin_y >= model.size_y && origin_z >= model.size_z {
        return Child::Empty;
    }

    // ── brick path ──
    // A region at cell_size = S covers 3·S voxels per axis. We can
    // materialize it as a brick of side (3·S) provided that side is
    // in `BRICK_SIDES` and ≤ `max_brick_side`. This bottom-up read
    // is lossless — every cell is a terminal voxel from the model.
    let brick_side_here = match cell_size {
        1 => 3,
        3 => 9,
        9 => 27,
        _ => 0,
    };
    if brick_side_here > 0 && brick_side_here <= max_brick_side {
        return build_brick(model, library, origin_x, origin_y, origin_z, brick_side_here);
    }

    // ── recursive path ──
    // Children are sub-regions that recurse until they hit the brick
    // level. Each child returns a Child::{Empty, Block, Node}.
    let sub_size = cell_size / BRANCH;
    let mut children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                children[slot_index(x, y, z)] = build_node(
                    model, library,
                    origin_x + x * cell_size,
                    origin_y + y * cell_size,
                    origin_z + z * cell_size,
                    sub_size,
                    max_brick_side,
                );
            }
        }
    }

    // Uniform collapse: if all 27 children are the same terminal,
    // return that terminal directly (no node created).
    let first = children[0];
    if matches!(first, Child::Empty | Child::Block(_))
        && children.iter().all(|&c| c == first)
    {
        return first;
    }

    Child::Node(library.insert(children))
}

/// Read `side³` voxels from the model into a brick and insert it.
/// Collapses fully-uniform regions to a single `Child` so the parent
/// can either emit a `Child::Block` / `Child::Empty` directly or
/// dedup via the library.
fn build_brick(
    model: &VoxelModel,
    library: &mut NodeLibrary,
    origin_x: usize,
    origin_y: usize,
    origin_z: usize,
    side: u8,
) -> Child {
    let s = side as usize;
    let mut cells = vec![BRICK_EMPTY; s * s * s];
    let mut first: Option<u8> = None;
    let mut uniform = true;
    for cz in 0..s {
        for cy in 0..s {
            for cx in 0..s {
                let voxel = if origin_x + cx < model.size_x
                    && origin_y + cy < model.size_y
                    && origin_z + cz < model.size_z
                {
                    let v = model.get(origin_x + cx, origin_y + cy, origin_z + cz);
                    if v == 0 { BRICK_EMPTY } else { v }
                } else {
                    BRICK_EMPTY
                };
                let idx = cz * s * s + cy * s + cx;
                cells[idx] = voxel;
                if uniform {
                    match first {
                        None => first = Some(voxel),
                        Some(f) if f == voxel => {}
                        _ => uniform = false,
                    }
                }
            }
        }
    }

    if uniform {
        return match first {
            Some(v) if v == BRICK_EMPTY => Child::Empty,
            Some(v) => Child::Block(v),
            None => Child::Empty,
        };
    }

    Child::Node(library.insert_brick(cells, side))
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

    /// A 3x3x3 model of uniform stone collapses to a single Block
    /// child on the root wrapper (not even a node).
    #[test]
    fn uniform_3x3x3() {
        let model = VoxelModel {
            size_x: 3, size_y: 3, size_z: 3,
            data: vec![block::STONE; 27],
        };
        let mut lib = NodeLibrary::default();
        let root = build_tree(&model, &mut lib);
        // Uniform region collapses to Child::Block, which gets wrapped
        // in one uniform Cartesian node at the root.
        let node = lib.get(root).unwrap();
        assert!(node.children.iter().all(|c| *c == Child::Block(block::STONE)));
    }

    #[test]
    fn empty_model() {
        let model = VoxelModel {
            size_x: 3, size_y: 3, size_z: 3,
            data: vec![0; 27],
        };
        let mut lib = NodeLibrary::default();
        let root = build_tree(&model, &mut lib);
        let node = lib.get(root).unwrap();
        assert!(node.children.iter().all(|c| c.is_empty()));
    }

    /// A mixed 3x3x3 model emits a size-3 brick as the root.
    #[test]
    fn mixed_3x3x3_becomes_brick_when_allowed() {
        let mut data = vec![0u8; 27];
        data[0] = block::STONE;
        data[13] = block::GRASS;
        let model = VoxelModel { size_x: 3, size_y: 3, size_z: 3, data };

        let mut lib = NodeLibrary::default();
        let root = build_tree_with_max_brick_side(&model, &mut lib, 3);
        // The entire 27-voxel model fits in a side-3 brick, so the
        // root itself IS the brick (no Cartesian wrapper).
        let node = lib.get(root).unwrap();
        assert!(node.is_brick());
        assert_eq!(node.brick_side, 3);
        // Probe a couple of cells to verify content.
        assert_eq!(node.brick_voxel(0, 0, 0), block::STONE);
        assert_eq!(node.brick_voxel(1, 1, 1), block::GRASS);
        assert_eq!(node.brick_voxel(2, 2, 2), BRICK_EMPTY);
    }

    /// With max_brick_side=3, a 9-wide region becomes a recursive
    /// node of brick children (not a single size-9 brick).
    #[test]
    fn max_side_3_caps_brick_growth() {
        let mut data = vec![0u8; 9 * 9 * 9];
        data[0] = block::STONE;
        data[80] = block::GRASS;
        let model = VoxelModel { size_x: 9, size_y: 9, size_z: 9, data };

        let mut lib = NodeLibrary::default();
        let _root = build_tree_with_max_brick_side(&model, &mut lib, 3);
        // Library should contain at least one size-3 brick but no larger.
        let any_brick = lib.nodes_iter().any(|n| n.is_brick());
        assert!(any_brick);
        let any_big_brick = lib.nodes_iter().any(|n| n.is_brick() && n.brick_side > 3);
        assert!(!any_big_brick);
    }

    /// With max_brick_side=9, a 9-wide mixed region becomes one
    /// size-9 brick (one-shot).
    #[test]
    fn max_side_9_emits_big_brick() {
        let mut data = vec![0u8; 9 * 9 * 9];
        data[0] = block::STONE;
        data[80] = block::GRASS;
        let model = VoxelModel { size_x: 9, size_y: 9, size_z: 9, data };

        let mut lib = NodeLibrary::default();
        let _root = build_tree_with_max_brick_side(&model, &mut lib, 9);
        let any_side_9 = lib.nodes_iter().any(|n| n.is_brick() && n.brick_side == 9);
        assert!(any_side_9, "expected a side-9 brick, found: {:?}",
            lib.nodes_iter().map(|n| (n.kind, n.brick_side)).collect::<Vec<_>>());
    }

    #[test]
    fn dedup_across_bricks() {
        // Two identical 3³ regions should dedup to one brick.
        let mut data = vec![0u8; 9 * 3 * 3];
        // (0,0,0)..(2,0,0) stone; (3,0,0)..(5,0,0) stone — same pattern shifted.
        for x in 0..3 { data[x] = block::STONE; }
        for x in 3..6 { data[x] = block::STONE; }
        let model = VoxelModel { size_x: 9, size_y: 3, size_z: 3, data };

        let mut lib = NodeLibrary::default();
        let _root = build_tree_with_max_brick_side(&model, &mut lib, 3);
        let brick_count = lib.nodes_iter()
            .filter(|n| n.is_brick() && n.brick_side == 3)
            .count();
        // Both 3³ sub-regions hold the same content, so dedup gives 1.
        assert_eq!(brick_count, 1, "bricks should dedup");
    }

    #[test]
    fn padding_from_4x4x4() {
        let mut data = vec![0u8; 4 * 4 * 4];
        data[0] = block::STONE;
        let model = VoxelModel { size_x: 4, size_y: 4, size_z: 4, data };
        let mut lib = NodeLibrary::default();
        let _root = build_tree(&model, &mut lib);
        assert!(lib.len() >= 1);
    }
}
