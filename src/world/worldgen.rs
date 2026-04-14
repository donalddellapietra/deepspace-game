//! World generator: builds a 6-level deep world with varied terrain.
//!
//! The world has structure at every scale:
//!   Level 1: individual block materials (stone, dirt, grass, etc.)
//!   Level 2: 3×3×3 block clusters (uniform fills, checkerboards, surfaces)
//!   Level 3: 9×9×9 terrain chunks (ground layers, trees, buildings)
//!   Level 4: 27×27×27 terrain sections (terrain with features)
//!   Level 5: 81×81×81 biome-scale regions
//!   Level 6: 243×243×243 world root
//!
//! Content-addressing means shared structure costs no extra memory.
//! A uniform stone layer reused 100 times is still one node.

use super::palette::block;
use super::state::WorldState;
use super::tree::*;

/// Simple deterministic hash for pseudo-random terrain variety.
fn pos_hash(x: usize, y: usize, z: usize, seed: u32) -> u32 {
    let mut h = seed;
    h = h.wrapping_mul(374761393).wrapping_add(x as u32).wrapping_mul(668265263);
    h = h.wrapping_mul(374761393).wrapping_add(y as u32).wrapping_mul(668265263);
    h = h.wrapping_mul(374761393).wrapping_add(z as u32).wrapping_mul(668265263);
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    h
}

/// Build a 6-level deep world.
pub fn generate_world() -> WorldState {
    let mut lib = NodeLibrary::default();

    // ═══════════════════════════════════════════════════════════════
    // Level 1: 3×3×3 uniform block clusters (building materials)
    // ═══════════════════════════════════════════════════════════════

    let stone_l1 = lib.insert(uniform_children(Child::Block(block::STONE)));
    let dirt_l1 = lib.insert(uniform_children(Child::Block(block::DIRT)));
    let grass_l1 = lib.insert(uniform_children(Child::Block(block::GRASS)));
    let wood_l1 = lib.insert(uniform_children(Child::Block(block::WOOD)));
    let leaf_l1 = lib.insert(uniform_children(Child::Block(block::LEAF)));
    let sand_l1 = lib.insert(uniform_children(Child::Block(block::SAND)));
    let brick_l1 = lib.insert(uniform_children(Child::Block(block::BRICK)));
    let air_l1 = lib.insert(empty_children());

    let gravel_l1 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let h = pos_hash(x, y, z, 42);
                    c[slot_index(x, y, z)] = if h % 3 == 0 {
                        Child::Block(block::DIRT)
                    } else {
                        Child::Block(block::STONE)
                    };
                }
            }
        }
        lib.insert(c)
    };

    // ═══════════════════════════════════════════════════════════════
    // Level 2: 3×3×3 of L1 nodes (terrain building blocks)
    // ═══════════════════════════════════════════════════════════════

    let stone_l2 = lib.insert(uniform_children(Child::Node(stone_l1)));
    let sand_l2 = lib.insert(uniform_children(Child::Node(sand_l1)));
    let air_l2 = lib.insert(uniform_children(Child::Node(air_l1)));

    let grass_surface_l2 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(dirt_l1);
                c[slot_index(x, 1, z)] = Child::Node(grass_l1);
                c[slot_index(x, 2, z)] = Child::Node(air_l1);
            }
        }
        lib.insert(c)
    };

    let sand_surface_l2 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(sand_l1);
                c[slot_index(x, 1, z)] = Child::Node(sand_l1);
                c[slot_index(x, 2, z)] = Child::Node(air_l1);
            }
        }
        lib.insert(c)
    };

    let rocky_surface_l2 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(stone_l1);
                c[slot_index(x, 1, z)] = Child::Node(gravel_l1);
                c[slot_index(x, 2, z)] = Child::Node(air_l1);
            }
        }
        lib.insert(c)
    };

    let trunk_l2 = {
        let mut c = uniform_children(Child::Node(air_l1));
        c[slot_index(1, 0, 1)] = Child::Node(wood_l1);
        c[slot_index(1, 1, 1)] = Child::Node(wood_l1);
        c[slot_index(1, 2, 1)] = Child::Node(wood_l1);
        lib.insert(c)
    };

    let canopy_l2 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let h = pos_hash(x, y, z, 77);
                    c[slot_index(x, y, z)] = if h % 4 == 0 {
                        Child::Node(air_l1)
                    } else {
                        Child::Node(leaf_l1)
                    };
                }
            }
        }
        lib.insert(c)
    };

    let wall_l2 = lib.insert(uniform_children(Child::Node(brick_l1)));

    // ═══════════════════════════════════════════════════════════════
    // Level 3: 3×3×3 of L2 nodes (terrain chunks = 9×9×9 blocks)
    // ═══════════════════════════════════════════════════════════════

    let grassland_l3 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(stone_l2);
                c[slot_index(x, 1, z)] = Child::Node(grass_surface_l2);
                c[slot_index(x, 2, z)] = Child::Node(air_l2);
            }
        }
        lib.insert(c)
    };

    let desert_l3 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(sand_l2);
                c[slot_index(x, 1, z)] = Child::Node(sand_surface_l2);
                c[slot_index(x, 2, z)] = Child::Node(air_l2);
            }
        }
        lib.insert(c)
    };

    let rocky_l3 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(stone_l2);
                c[slot_index(x, 1, z)] = Child::Node(rocky_surface_l2);
                c[slot_index(x, 2, z)] = Child::Node(air_l2);
            }
        }
        lib.insert(c)
    };

    let forest_l3 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(stone_l2);
                c[slot_index(x, 1, z)] = Child::Node(grass_surface_l2);
                c[slot_index(x, 2, z)] = Child::Node(air_l2);
            }
        }
        c[slot_index(1, 2, 1)] = Child::Node(trunk_l2);
        lib.insert(c)
    };

    let dense_forest_l3 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(stone_l2);
                c[slot_index(x, 1, z)] = Child::Node(grass_surface_l2);
                let h = pos_hash(x, 0, z, 123);
                c[slot_index(x, 2, z)] = if h % 3 == 0 {
                    Child::Node(trunk_l2)
                } else if h % 3 == 1 {
                    Child::Node(canopy_l2)
                } else {
                    Child::Node(air_l2)
                };
            }
        }
        lib.insert(c)
    };

    let mountain_l3 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(stone_l2);
                c[slot_index(x, 1, z)] = Child::Node(stone_l2);
                c[slot_index(x, 2, z)] = if x == 1 && z == 1 {
                    Child::Node(rocky_surface_l2)
                } else {
                    Child::Node(air_l2)
                };
            }
        }
        lib.insert(c)
    };

    let village_l3 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(stone_l2);
                c[slot_index(x, 1, z)] = Child::Node(grass_surface_l2);
                c[slot_index(x, 2, z)] = Child::Node(air_l2);
            }
        }
        c[slot_index(1, 2, 1)] = Child::Node(wall_l2);
        c[slot_index(0, 2, 1)] = Child::Node(wall_l2);
        c[slot_index(1, 2, 0)] = Child::Node(wall_l2);
        lib.insert(c)
    };

    let air_l3 = lib.insert(uniform_children(Child::Node(air_l2)));

    // ═══════════════════════════════════════════════════════════════
    // Level 4: 3×3×3 of L3 nodes (terrain sections = 27×27×27)
    // ═══════════════════════════════════════════════════════════════

    let plains_l4 = lib.insert(uniform_children(Child::Node(grassland_l3)));

    let forest_l4 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let h = pos_hash(x, y, z, 200);
                    c[slot_index(x, y, z)] = if y >= 2 {
                        Child::Node(air_l3)
                    } else if y == 1 {
                        if h % 2 == 0 { Child::Node(forest_l3) }
                        else { Child::Node(dense_forest_l3) }
                    } else {
                        Child::Node(grassland_l3)
                    };
                }
            }
        }
        lib.insert(c)
    };

    let mountain_l4 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    c[slot_index(x, y, z)] = if y >= 2 {
                        Child::Node(air_l3)
                    } else if y == 1 {
                        let h = pos_hash(x, 0, z, 300);
                        if h % 3 == 0 { Child::Node(mountain_l3) }
                        else { Child::Node(rocky_l3) }
                    } else {
                        Child::Node(rocky_l3)
                    };
                }
            }
        }
        lib.insert(c)
    };

    let desert_l4 = lib.insert(uniform_children(Child::Node(desert_l3)));

    let village_l4 = {
        let mut c = uniform_children(Child::Node(grassland_l3));
        c[slot_index(1, 1, 1)] = Child::Node(village_l3);
        c[slot_index(0, 1, 0)] = Child::Node(forest_l3);
        c[slot_index(2, 1, 2)] = Child::Node(forest_l3);
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 2, z)] = Child::Node(air_l3);
            }
        }
        lib.insert(c)
    };

    // ═══════════════════════════════════════════════════════════════
    // Level 5: 3×3×3 of L4 nodes (biome regions = 81×81×81)
    // ═══════════════════════════════════════════════════════════════

    let temperate_l5 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let h = pos_hash(x, y, z, 400);
                    c[slot_index(x, y, z)] = match (y, h % 5) {
                        (2, _) => Child::Node(plains_l4),
                        (1, 0) | (1, 3) => Child::Node(forest_l4),
                        (1, 1) => Child::Node(village_l4),
                        (_, _) => Child::Node(plains_l4),
                    };
                }
            }
        }
        lib.insert(c)
    };

    let highland_l5 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let h = pos_hash(x, y, z, 500);
                    c[slot_index(x, y, z)] = match (y, h % 3) {
                        (1, 1) => Child::Node(forest_l4),
                        _ => Child::Node(mountain_l4),
                    };
                }
            }
        }
        lib.insert(c)
    };

    let arid_l5 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let h = pos_hash(x, y, z, 600);
                    c[slot_index(x, y, z)] = if y == 0 && h % 4 == 0 {
                        Child::Node(mountain_l4)
                    } else {
                        Child::Node(desert_l4)
                    };
                }
            }
        }
        lib.insert(c)
    };

    // ═══════════════════════════════════════════════════════════════
    // Level 6 (root): 3×3×3 of L5 nodes (the world = 243×243×243)
    // ═══════════════════════════════════════════════════════════════

    let mut root_children = empty_children();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            root_children[slot_index(x, 0, z)] = Child::Node(highland_l5);
            root_children[slot_index(x, 2, z)] = Child::Node(temperate_l5);
            let h = pos_hash(x, 1, z, 700);
            root_children[slot_index(x, 1, z)] = match h % 3 {
                0 => Child::Node(temperate_l5),
                1 => Child::Node(highland_l5),
                _ => Child::Node(arid_l5),
            };
        }
    }

    let root = lib.insert(root_children);
    lib.ref_inc(root);

    let world = WorldState { root, library: lib };
    let d = world.tree_depth();
    eprintln!(
        "Generated world: {} unique nodes, depth {}, 3^{} = {} blocks per axis",
        world.library.len(), d, d, 3u64.pow(d),
    );

    world
}
