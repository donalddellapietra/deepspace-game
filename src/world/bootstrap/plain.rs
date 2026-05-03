//! Plain (flat-ground) world bootstrap and the editor `plain_test_world`.

use std::collections::HashMap;

use super::WorldBootstrap;
use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, BRANCH, Child, NodeId, NodeKind, NodeLibrary,
    MAX_DEPTH,
};

pub const DEFAULT_PLAIN_LAYERS: u8 = 40;
pub(super) const PLAIN_SURFACE_Y: f32 = 1.0;
const PLAIN_GRASS_THICKNESS: f32 = 0.05;
const PLAIN_DIRT_THICKNESS: f32 = 0.25;

pub fn plain_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    let stone_l1 = library.insert(uniform_children(Child::Block(block::STONE)));
    let dirt_l1 = library.insert(uniform_children(Child::Block(block::DIRT)));
    let grass_l1 = library.insert(uniform_children(Child::Block(block::GRASS)));
    let wood_l1 = library.insert(uniform_children(Child::Block(block::WOOD)));
    let leaf_l1 = library.insert(uniform_children(Child::Block(block::LEAF)));
    let sand_l1 = library.insert(uniform_children(Child::Block(block::SAND)));
    let brick_l1 = library.insert(uniform_children(Child::Block(block::BRICK)));
    let air_l1 = library.insert(empty_children());

    let mut checker_children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                let slot = slot_index(x, y, z);
                checker_children[slot] = if (x + y + z) % 2 == 0 {
                    Child::Block(block::STONE)
                } else {
                    Child::Block(block::DIRT)
                };
            }
        }
    }
    let checker_l1 = library.insert(checker_children);

    let stone_l2 = library.insert(uniform_children(Child::Node(stone_l1)));

    let mut checker_ground_children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                let slot = slot_index(x, y, z);
                checker_ground_children[slot] = if (x + z) % 2 == 0 {
                    Child::Node(stone_l1)
                } else {
                    Child::Node(checker_l1)
                };
            }
        }
    }
    let checker_ground_l2 = library.insert(checker_ground_children);

    let mut grass_surface_children = empty_children();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            let slot_bottom = slot_index(x, 0, z);
            let slot_mid = slot_index(x, 1, z);
            let slot_top = slot_index(x, 2, z);
            grass_surface_children[slot_bottom] = Child::Node(dirt_l1);
            grass_surface_children[slot_mid] = Child::Node(grass_l1);
            grass_surface_children[slot_top] = Child::Node(air_l1);
        }
    }
    let grass_surface_l2 = library.insert(grass_surface_children);

    let air_l2 = library.insert(uniform_children(Child::Node(air_l1)));

    let mut features_children = empty_children();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            for y in 0..BRANCH {
                features_children[slot_index(x, y, z)] = Child::Node(air_l1);
            }
        }
    }
    features_children[slot_index(1, 0, 1)] = Child::Node(wood_l1);
    features_children[slot_index(1, 1, 1)] = Child::Node(wood_l1);
    features_children[slot_index(1, 2, 1)] = Child::Node(leaf_l1);
    features_children[slot_index(0, 0, 0)] = Child::Node(brick_l1);
    features_children[slot_index(0, 0, 1)] = Child::Node(brick_l1);
    features_children[slot_index(0, 0, 2)] = Child::Node(brick_l1);
    features_children[slot_index(0, 1, 0)] = Child::Node(brick_l1);
    features_children[slot_index(0, 1, 1)] = Child::Node(brick_l1);
    features_children[slot_index(0, 1, 2)] = Child::Node(brick_l1);
    features_children[slot_index(2, 0, 0)] = Child::Node(sand_l1);
    let features_l2 = library.insert(features_children);

    let mut root_children = empty_children();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            root_children[slot_index(x, 0, z)] = if (x + z) % 2 == 0 {
                Child::Node(stone_l2)
            } else {
                Child::Node(checker_ground_l2)
            };
            root_children[slot_index(x, 1, z)] = Child::Node(grass_surface_l2);
            root_children[slot_index(x, 2, z)] = if x == 1 && z == 1 {
                Child::Node(features_l2)
            } else {
                Child::Node(air_l2)
            };
        }
    }

    let root = library.insert(root_children);
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "Plain test world: {} library entries, root spans [0, {WORLD_SIZE}) per axis",
        world.library.len(),
    );
    world
}

pub fn plain_world(layers: u8) -> WorldState {
    assert!(layers > 0, "plain world must have at least one layer");
    assert!(
        (layers as usize) <= MAX_DEPTH,
        "plain world layers {} exceeds MAX_DEPTH {}",
        layers,
        MAX_DEPTH,
    );

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    enum UniformFill {
        Empty,
        Block(u16),
    }

    fn fill_for_range(y_min: f32, y_max: f32) -> Option<UniformFill> {
        let grass_min = PLAIN_SURFACE_Y - PLAIN_GRASS_THICKNESS;
        let dirt_min = PLAIN_SURFACE_Y - PLAIN_DIRT_THICKNESS;
        if y_min >= PLAIN_SURFACE_Y {
            Some(UniformFill::Empty)
        } else if y_max <= dirt_min {
            Some(UniformFill::Block(block::STONE))
        } else if y_min >= grass_min && y_max <= PLAIN_SURFACE_Y {
            Some(UniformFill::Block(block::GRASS))
        } else if y_min >= dirt_min && y_max <= grass_min {
            Some(UniformFill::Block(block::DIRT))
        } else {
            None
        }
    }

    fn uniform_subtree(
        lib: &mut NodeLibrary,
        cache: &mut HashMap<(u8, UniformFill), NodeId>,
        depth: u8,
        fill: UniformFill,
    ) -> NodeId {
        if let Some(&id) = cache.get(&(depth, fill)) {
            return id;
        }
        let id = if depth == 1 {
            match fill {
                UniformFill::Empty => lib.insert(empty_children()),
                UniformFill::Block(block_type) => lib.insert(uniform_children(Child::Block(block_type))),
            }
        } else {
            let child = uniform_subtree(lib, cache, depth - 1, fill);
            lib.insert(uniform_children(Child::Node(child)))
        };
        cache.insert((depth, fill), id);
        id
    }

    fn build_plain_subtree(
        lib: &mut NodeLibrary,
        cache: &mut HashMap<(u8, UniformFill), NodeId>,
        depth: u8,
        y_min: f32,
        y_max: f32,
    ) -> NodeId {
        if let Some(fill) = fill_for_range(y_min, y_max) {
            return uniform_subtree(lib, cache, depth, fill);
        }

        if depth == 1 {
            let mut children = empty_children();
            let child_size = (y_max - y_min) / BRANCH as f32;
            for z in 0..BRANCH {
                for y in 0..BRANCH {
                    let row_min = y_min + child_size * y as f32;
                    let row_max = row_min + child_size;
                    let child = match fill_for_range(row_min, row_max) {
                        Some(UniformFill::Empty) => Child::Empty,
                        Some(UniformFill::Block(block_type)) => Child::Block(block_type),
                        None => {
                            let mid = (row_min + row_max) * 0.5;
                            if mid >= PLAIN_SURFACE_Y {
                                Child::Empty
                            } else if mid >= PLAIN_SURFACE_Y - PLAIN_GRASS_THICKNESS {
                                Child::Block(block::GRASS)
                            } else if mid >= PLAIN_SURFACE_Y - PLAIN_DIRT_THICKNESS {
                                Child::Block(block::DIRT)
                            } else {
                                Child::Block(block::STONE)
                            }
                        }
                    };
                    for x in 0..BRANCH {
                        children[slot_index(x, y, z)] = child;
                    }
                }
            }
            return lib.insert(children);
        }

        let mut children = empty_children();
        let child_size = (y_max - y_min) / BRANCH as f32;
        for y in 0..BRANCH {
            let row_min = y_min + child_size * y as f32;
            let row_max = row_min + child_size;
            let child_id = build_plain_subtree(lib, cache, depth - 1, row_min, row_max);
            let row_child = Child::Node(child_id);
            for z in 0..BRANCH {
                for x in 0..BRANCH {
                    children[slot_index(x, y, z)] = row_child;
                }
            }
        }
        lib.insert(children)
    }

    let mut library = NodeLibrary::default();
    let mut uniform_cache = HashMap::new();
    let root = build_plain_subtree(&mut library, &mut uniform_cache, layers, 0.0, WORLD_SIZE);
    library.ref_inc(root);
    let world = WorldState { root, library };
    eprintln!(
        "Plain world: layers={}, library_entries={}, depth={}",
        layers,
        world.library.len(),
        world.tree_depth(),
    );
    world
}

/// Build a spawn position that tracks the dirt/grass boundary at any
/// anchor depth. The boundary at y ≈ 0.95 (PLAIN_SURFACE_Y −
/// PLAIN_GRASS_THICKNESS) is NOT ternary-rational: its base-3
/// expansion is 0.2211̄ (repeating "2211"). This means every render
/// frame at this y contains mixed dirt/grass children — the grid
/// between different block colors is always visible.
///
/// f64 arithmetic loses precision past depth ~34 (cell_size drops
/// below f64 resolution at 0.95). This direct construction bypasses
/// f64 entirely: we encode the exact ternary digit pattern as path
/// slots, achieving precision-perfect spawning at any anchor depth.
pub fn plain_surface_spawn(anchor_depth: u8) -> WorldPos {
    // y ≈ 0.95 in [0, WORLD_SIZE=3). Ternary expansion:
    //   depth 1:  digit 0  (root y ∈ [0,1))
    //   depth 2+: repeating [2, 2, 1, 1]
    const Y_PATTERN: [usize; 4] = [2, 2, 1, 1];

    let mut path = Path::root();
    for d in 0..anchor_depth as usize {
        let y_row = if d == 0 { 0 } else { Y_PATTERN[(d - 1) % 4] };
        let slot = slot_index(1, y_row, 1); // x=1, z=1 center
        path.push(slot as u8);
    }

    // Camera near center of cell. After carve_air_pocket clears
    // this cell, the camera is in a 1-block air pocket looking at
    // surrounding dirt/grass blocks.
    WorldPos::new(path, [0.5, 0.5, 0.5])
}

/// Create a uniform air subtree of the given depth. All children are
/// recursively air nodes, so the render-frame tree walk can descend
/// through this region just like it would through a normal tree.
/// Content-addressed dedup means only O(depth) nodes are allocated
/// regardless of how wide the subtree is.
fn air_subtree(lib: &mut NodeLibrary, depth: u8) -> NodeId {
    if depth == 0 {
        return lib.insert(empty_children());
    }
    let child = air_subtree(lib, depth - 1);
    lib.insert(uniform_children(Child::Node(child)))
}

/// Ensure the camera's anchor path is tree-walkable down to
/// `anchor.depth()`, inserting a fresh empty Node at any slot that
/// was `Child::Empty` or `Child::Block` along the walk, and carving
/// an air subtree at the final slot of depth `anchor.depth() - 1`.
///
/// Two guarantees for the renderer:
/// 1. `compute_render_frame` can walk the anchor path all the way
///    down — critical for fractals (Menger's body-centre Empties,
///    Sierpinski's unused corner slots, etc.) where the path would
///    otherwise stall on a structural Empty at depth 2–3.
/// 2. The last cell (at `anchor.depth()`) is always air, so plain-
///    world spawn lands in an air pocket rather than inside a
///    dirt/grass block.
///
/// The expand-on-walk is a side effect the renderer *needs*; the
/// final-cell carve is a side effect plain-worlds *want*. Both
/// happen together because a single bottom-up rebuild stitches the
/// new child IDs upward through the whole anchor path.
///
/// The air subtree below the final cell extends to `total_depth` so
/// the user can zoom to any depth inside the cavity and still get a
/// deep, walkable render frame.
pub fn carve_air_pocket(world: &mut WorldState, anchor: &Path, total_depth: u8) {
    if anchor.depth() < 2 { return; }
    let slots = anchor.as_slice();
    let carve_depth = (anchor.depth() - 1) as usize;
    let mut node_stack: Vec<(NodeId, NodeKind)> = Vec::with_capacity(carve_depth + 1);
    let mut node_id = world.root;
    for &slot in &slots[..carve_depth] {
        let Some(node) = world.library.get(node_id) else { return };
        node_stack.push((node_id, node.kind));
        match node.children[slot as usize] {
            Child::Node(child_id) => node_id = child_id,
            // If the camera's anchor path crosses an Empty or Block
            // slot before reaching `carve_depth`, install a fresh
            // empty Node there so the walk can continue. Bottom-up
            // rebuild below stitches the new Node into the parent's
            // slot via the replacement chain — the rest of the
            // parent's siblings stay untouched. This is the fix
            // for fractals with structural empties (Menger's body-
            // centres, Sierpinski's 23 unused corners etc.): without
            // it, `compute_render_frame` stalls at the first empty
            // and the shader can never render cells small enough
            // for Nyquist to let a real Block leaf be visible,
            // which manifests as monochromatic LOD-terminal colour.
            // EntityRef should never appear in a terrain edit path
            // — entities only land in ephemeral scene-root nodes —
            // but if it does, treat it like an empty and install a
            // fresh air Node so carve can proceed without panicking.
            Child::Empty | Child::Block(_) | Child::EntityRef(_) => {
                node_id = world.library.insert(empty_children());
            }
        }
    }
    let Some(node) = world.library.get(node_id) else { return };
    node_stack.push((node_id, node.kind));

    let clear_slot = slots[carve_depth] as usize;

    // How many additional levels of air nodes below the cleared cell.
    // The cleared cell is at absolute depth `carve_depth + 1`.
    // We need air down to `total_depth`.
    let cleared_abs_depth = (carve_depth + 1) as u8;
    let air_depth = total_depth.saturating_sub(cleared_abs_depth);
    let air_node = if air_depth > 0 {
        Child::Node(air_subtree(&mut world.library, air_depth))
    } else {
        Child::Empty
    };

    // Build replacement bottom-up.
    let mut replacement: Option<NodeId> = None;
    for (i, &(nid, kind)) in node_stack.iter().enumerate().rev() {
        let Some(node) = world.library.get(nid) else { return };
        let mut new_children = node.children;
        if let Some(rep) = replacement {
            new_children[slots[i] as usize] = Child::Node(rep);
        } else {
            new_children[clear_slot] = air_node;
        }
        replacement = Some(world.library.insert_with_kind(new_children, kind));
    }
    if let Some(new_root) = replacement {
        world.swap_root(new_root);
    }
}

pub(super) fn bootstrap_plain_test_world(plain_layers: u8) -> WorldBootstrap {
    let world = plain_world(plain_layers);
    let spawn_pos = plain_surface_spawn(8);
    // NOTE: don't carve here — the default spawn is at depth 8 and
    // carving here would clear a cell that's an ancestor of any deeper
    // --spawn-depth override. Carving happens in App::with_test_config
    // after the final spawn position is known.
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -0.45,
        plain_layers,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_world_honors_requested_depth() {
        let world = plain_world(40);
        assert_eq!(world.tree_depth(), 40);
    }
}
