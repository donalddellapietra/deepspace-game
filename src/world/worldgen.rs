//! Worldgen: build the content-addressed voxel tree.
//!
//! Produces a noise-displaced landscape inside the root cell,
//! `[0, WORLD_SIZE)³`. The recursion bottoms out at `TARGET_DEPTH`
//! finest cells per axis; above/below the surface is filled with
//! dedup'd uniform air / stone subtrees so unique node count stays
//! small regardless of depth.

use super::anchor::WORLD_SIZE;
use super::palette::block;
use super::sdf;
use super::state::WorldState;
use super::tree::*;

/// Finest cells per axis = 3^TARGET_DEPTH. At depth 6 the world is
/// 729 cells across each axis; enough detail to see terrain without
/// the recursion cost of the old 21-level empty-tree.
const TARGET_DEPTH: u32 = 6;

/// Levels that sample the SDF. Below this, each straddling cell
/// commits to solid or empty and wraps the remaining depth in a
/// uniform filler — the usual cap that keeps unique-node count
/// bounded by `27^SDF_DETAIL_LEVELS`.
const SDF_DETAIL_LEVELS: u32 = 4;

pub fn generate_world() -> WorldState {
    let mut lib = NodeLibrary::default();
    let root_child = build_region(
        &mut lib,
        [0.0, 0.0, 0.0],
        WORLD_SIZE,
        TARGET_DEPTH,
        SDF_DETAIL_LEVELS,
    );
    let root = match root_child {
        Child::Node(id) => id,
        Child::Block(b) => lib.insert(uniform_children(Child::Block(b))),
        Child::Empty => lib.insert(empty_children()),
    };
    lib.ref_inc(root);
    let world = WorldState { root, library: lib };
    eprintln!(
        "Generated voxel world: {} unique nodes, depth {}",
        world.library.len(), world.tree_depth(),
    );
    world
}

/// Distance to a heightmap surface at world-space `p`. Negative
/// below ground, positive above. The surface sits at `y_surface =
/// WORLD_SIZE * 0.5 + noise * amplitude`.
fn surface_distance(p: [f32; 3]) -> f32 {
    let noise = sdf::noise3d([p[0] * 2.5, 0.0, p[2] * 2.5], 17);
    let y_surface = WORLD_SIZE * 0.5 + noise * 0.25;
    p[1] - y_surface
}

fn block_at(p: [f32; 3]) -> u8 {
    let d = surface_distance(p);
    if d < -0.3 { block::STONE }
    else if d < -0.05 { block::DIRT }
    else { block::GRASS }
}

/// Recursive builder for a Cartesian region. `origin` is the world-
/// space min corner of the region and `size` is its extent along
/// each axis. `depth` is the remaining tree depth; `sdf_budget`
/// caps SDF-recursion.
fn build_region(
    lib: &mut NodeLibrary,
    origin: [f32; 3],
    size: f32,
    depth: u32,
    sdf_budget: u32,
) -> Child {
    let center = [origin[0] + size * 0.5, origin[1] + size * 0.5, origin[2] + size * 0.5];
    let d_center = surface_distance(center);
    let cell_rad = size * 0.866; // ~sqrt(3)/2 for the corner radius.

    // Fully above the surface → empty.
    if d_center > cell_rad {
        if depth == 0 { return Child::Empty; }
        return Child::Node(build_uniform_empty(lib, depth));
    }
    // Fully below → solid with the center sample's block type.
    if d_center < -cell_rad {
        let b = block_at(center);
        if depth == 0 { return Child::Block(b); }
        return lib.build_uniform_subtree(b, depth);
    }

    if depth == 0 {
        return if d_center < 0.0 { Child::Block(block_at(center)) } else { Child::Empty };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            lib.build_uniform_subtree(block_at(center), depth)
        } else {
            Child::Node(build_uniform_empty(lib, depth))
        };
    }

    let mut children = empty_children();
    let child_size = size / 3.0;
    for cz in 0..BRANCH {
        for cy in 0..BRANCH {
            for cx in 0..BRANCH {
                let child_origin = [
                    origin[0] + cx as f32 * child_size,
                    origin[1] + cy as f32 * child_size,
                    origin[2] + cz as f32 * child_size,
                ];
                children[slot_index(cx, cy, cz)] = build_region(
                    lib, child_origin, child_size, depth - 1, sdf_budget - 1,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

fn build_uniform_empty(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::edit;

    #[test]
    fn generated_world_has_content() {
        let w = generate_world();
        // Below the middle, expect solid.
        assert!(edit::is_solid_at(&w.library, w.root, [1.5, 0.5, 1.5], 6));
        // Above, expect empty.
        assert!(!edit::is_solid_at(&w.library, w.root, [1.5, 2.5, 1.5], 6));
    }
}
