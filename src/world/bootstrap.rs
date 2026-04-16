//! World bootstrap presets used by app startup and debugging.
//!
//! Low-level generation stays in `worldgen` and `spherical_worldgen`.
//! This module owns composition: which world we start with, whether it
//! contains a planet, and where the default spawn should be.

use super::anchor::{Path, WorldPos, WORLD_SIZE};
use super::palette::block;
use super::state::WorldState;
use super::tree::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorldPreset {
    #[default]
    PlainTest,
    DemoSphere,
}

pub const DEFAULT_PLAIN_LAYERS: u8 = 40;
const PLAIN_SURFACE_Y: f32 = 1.0;
const PLAIN_GRASS_THICKNESS: f32 = 0.05;
const PLAIN_DIRT_THICKNESS: f32 = 0.25;

pub struct WorldBootstrap {
    pub world: WorldState,
    pub planet_path: Option<Path>,
    /// Spawn position as a path-anchored `WorldPos`. Constructed at
    /// shallow depth (where f32 decomposition is precise) then
    /// `deepened_to` the target anchor depth via pure slot arithmetic.
    pub default_spawn_pos: WorldPos,
    pub default_spawn_yaw: f32,
    pub default_spawn_pitch: f32,
    pub plain_layers: u8,
}

pub fn bootstrap_world(preset: WorldPreset, plain_layers: Option<u8>) -> WorldBootstrap {
    match preset {
        WorldPreset::DemoSphere => bootstrap_demo_sphere_world(),
        WorldPreset::PlainTest => bootstrap_plain_test_world(plain_layers.unwrap_or(DEFAULT_PLAIN_LAYERS)),
    }
}

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
        Block(u8),
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

fn bootstrap_demo_sphere_world() -> WorldBootstrap {
    let mut world = crate::world::worldgen::generate_world();
    let setup = crate::world::spherical_worldgen::demo_planet();
    let (new_root, planet_path) =
        crate::world::spherical_worldgen::install_at_root_center(
            &mut world.library,
            world.root,
            &setup,
        );
    world.swap_root(new_root);
    let tree_depth = world.tree_depth();
    eprintln!(
        "Demo sphere world: planet_path={:?}, library_entries={}, depth={}",
        planet_path.as_slice(),
        world.library.len(),
        tree_depth,
    );

    let body_top_y = 1.5 + setup.outer_r;
    // Construct at shallow depth (2) where f32 decomposition is
    // precise, then deepen to anchor depth 16 via pure slot arithmetic.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.5, (body_top_y + 0.05).min(WORLD_SIZE - 0.001), 1.5],
        2,
    ).deepened_to(16);
    WorldBootstrap {
        world,
        planet_path: Some(planet_path),
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: -1.2,
        plain_layers: 0,
    }
}

/// Build a spawn position that tracks the dirt/grass boundary at any
/// anchor depth. The boundary at y ≈ 0.95 (PLAIN_SURFACE_Y −
/// PLAIN_GRASS_THICKNESS) is NOT ternary-rational: its base-3
/// expansion is 0.2211̄ (repeating "2211"). This means every render
/// frame at this y contains mixed dirt/grass children — the grid
/// between different block colors is always visible.
///
/// f64 arithmetic loses precision past depth ~34 (cell_size drops
/// below the f64 epsilon), so we use the exact periodic ternary
/// expansion directly.
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

/// Carve a 3x3x3-block air cavity at the camera's anchor position.
///
/// Clears the cell at `anchor.depth() - 1` (the parent of the leaf)
/// to `Child::Empty`. Since that parent cell contains a 3x3x3 grid
/// of leaf blocks, this creates a 27-block air cavity. The camera
/// inside the cavity can see 3x3 grids on each wall — with the
/// y-axis showing different materials where the dirt/grass boundary
/// crosses the cavity.
///
/// The carve is 1 level above the anchor's leaf, which is still
/// below the render frame (anchor − RENDER_FRAME_K), so it doesn't
/// affect frame computation.
pub fn carve_air_pocket(world: &mut WorldState, anchor: &Path) {
    if anchor.depth() < 2 { return; }
    let slots = anchor.as_slice();
    // Walk to depth (anchor_depth - 2): the grandparent of the leaf.
    // We'll clear the child at slots[anchor_depth - 2] which is the
    // parent cell of the leaf.
    let carve_depth = (anchor.depth() - 1) as usize; // slot index of parent cell
    let mut node_stack: Vec<(NodeId, NodeKind)> = Vec::with_capacity(carve_depth);
    let mut node_id = world.root;
    for &slot in &slots[..carve_depth] {
        let Some(node) = world.library.get(node_id) else { return };
        node_stack.push((node_id, node.kind));
        match node.children[slot as usize] {
            Child::Node(child_id) => node_id = child_id,
            _ => return,
        }
    }
    let Some(node) = world.library.get(node_id) else { return };
    node_stack.push((node_id, node.kind));

    // The cell to clear is at slots[carve_depth - 1] in node_stack.last().
    // Wait — carve_depth is the slot index we want to CLEAR at.
    // node_stack.last() is the node that CONTAINS slots[carve_depth].
    // But actually, we walked up to (but not including) carve_depth in
    // the forward walk. So node_stack has `carve_depth` entries.
    // node_stack[carve_depth-1] is the node at depth carve_depth-1,
    // and the child at slots[carve_depth-1] is the cell to clear.

    let clear_slot = slots[carve_depth] as usize; // the parent-of-leaf slot

    // Build replacement bottom-up.
    let mut replacement: Option<NodeId> = None;
    for (i, &(nid, kind)) in node_stack.iter().enumerate().rev() {
        let Some(node) = world.library.get(nid) else { return };
        let mut new_children = node.children;
        if let Some(rep) = replacement {
            new_children[slots[i] as usize] = Child::Node(rep);
        } else {
            // Bottom of chain: clear the parent-of-leaf cell.
            new_children[clear_slot] = Child::Empty;
        }
        replacement = Some(world.library.insert_with_kind(new_children, kind));
    }
    if let Some(new_root) = replacement {
        world.swap_root(new_root);
    }
}

fn bootstrap_plain_test_world(plain_layers: u8) -> WorldBootstrap {
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
