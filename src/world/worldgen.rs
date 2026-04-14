//! World generator: builds a 20-level deep world with varied terrain.
//!
//! Bottom 6 levels have hand-crafted terrain detail (blocks, surfaces,
//! trees, buildings, biomes). Levels 7-20 are composed algorithmically
//! by mixing lower-level variants with positional hashing.
//!
//! Content-addressing means the node count stays small (~100 unique
//! nodes) even at 20 levels deep (3^20 ≈ 3.5 billion blocks per axis).

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

/// Build a 20-level deep world.
pub fn generate_world() -> WorldState {
    let mut lib = NodeLibrary::default();

    // ═══════════════════════════════════════════════════════════════
    // Level 1: 3×3×3 uniform block clusters
    // ═══════════════════════════════════════════════════════════════

    let stone_l1 = lib.insert(uniform_children(Child::Block(BlockType::Stone)));
    let dirt_l1 = lib.insert(uniform_children(Child::Block(BlockType::Dirt)));
    let grass_l1 = lib.insert(uniform_children(Child::Block(BlockType::Grass)));
    let wood_l1 = lib.insert(uniform_children(Child::Block(BlockType::Wood)));
    let leaf_l1 = lib.insert(uniform_children(Child::Block(BlockType::Leaf)));
    let sand_l1 = lib.insert(uniform_children(Child::Block(BlockType::Sand)));
    let brick_l1 = lib.insert(uniform_children(Child::Block(BlockType::Brick)));
    let air_l1 = lib.insert(empty_children());

    let gravel_l1 = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let h = pos_hash(x, y, z, 42);
                    c[slot_index(x, y, z)] = if h % 3 == 0 {
                        Child::Block(BlockType::Dirt)
                    } else {
                        Child::Block(BlockType::Stone)
                    };
                }
            }
        }
        lib.insert(c)
    };

    // ═══════════════════════════════════════════════════════════════
    // Level 2: terrain building blocks (surfaces, trunks, walls)
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
    // Level 3: terrain chunks (9×9×9 blocks)
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
    // Level 4: terrain sections (27×27×27)
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
    // Level 5: biome regions (81×81×81)
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
    // Level 6: continents (243×243×243)
    // ═══════════════════════════════════════════════════════════════

    let continent_a = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(highland_l5);
                c[slot_index(x, 2, z)] = Child::Node(temperate_l5);
                let h = pos_hash(x, 1, z, 700);
                c[slot_index(x, 1, z)] = match h % 3 {
                    0 => Child::Node(temperate_l5),
                    1 => Child::Node(highland_l5),
                    _ => Child::Node(arid_l5),
                };
            }
        }
        lib.insert(c)
    };

    let continent_b = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(highland_l5);
                c[slot_index(x, 2, z)] = Child::Node(arid_l5);
                let h = pos_hash(x, 1, z, 710);
                c[slot_index(x, 1, z)] = match h % 3 {
                    0 => Child::Node(arid_l5),
                    1 => Child::Node(highland_l5),
                    _ => Child::Node(temperate_l5),
                };
            }
        }
        lib.insert(c)
    };

    let continent_c = {
        let mut c = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                c[slot_index(x, 0, z)] = Child::Node(highland_l5);
                c[slot_index(x, 2, z)] = Child::Node(temperate_l5);
                let h = pos_hash(x, 1, z, 720);
                c[slot_index(x, 1, z)] = match h % 3 {
                    0 => Child::Node(highland_l5),
                    _ => Child::Node(arid_l5),
                };
            }
        }
        lib.insert(c)
    };

    // ═══════════════════════════════════════════════════════════════
    // Levels 7-20: composed algorithmically.
    //
    // Every level preserves the y-structure from the terrain:
    //   y=0: underground (denser, more stone/mountain)
    //   y=1: surface (mixed terrain — the interesting layer)
    //   y=2: sky (air-heavy, sparser)
    //
    // This ensures the camera always has navigable space above
    // ground at every zoom scale. Each level creates 3 surface
    // variants and 2 underground variants for xz variety.
    // ═══════════════════════════════════════════════════════════════

    // We need an air node at the continent scale for sky layers.
    // Build one by uniform-wrapping air upward from air_l3.
    let mut air_at_level = air_l3;
    for _ in 4..=6 {
        air_at_level = lib.insert(uniform_children(Child::Node(air_at_level)));
    }
    // air_at_level is now an all-air node at level 6 (continent scale).

    // Track surface variants (terrain with air above) and underground
    // variants (solid terrain) separately.
    let mut surface_variants: Vec<NodeId> = vec![continent_a, continent_b, continent_c];
    let mut underground_variants: Vec<NodeId> = vec![continent_a, continent_c];

    for level in 7..=20 {
        let seed_base = level as u32 * 1000;
        let n_surf = surface_variants.len();
        let n_under = underground_variants.len();

        let mut new_surface = Vec::new();
        let mut new_underground = Vec::new();

        // 3 surface variants: y=0 underground, y=1 mixed surface, y=2 air.
        for variant in 0..3u32 {
            let mut c = empty_children();
            for z in 0..BRANCH {
                for x in 0..BRANCH {
                    // y=0: underground
                    let h0 = pos_hash(x, 0, z, seed_base + variant * 100);
                    c[slot_index(x, 0, z)] = Child::Node(
                        underground_variants[h0 as usize % n_under]
                    );
                    // y=1: surface (the interesting terrain)
                    let h1 = pos_hash(x, 1, z, seed_base + variant * 100 + 37);
                    c[slot_index(x, 1, z)] = Child::Node(
                        surface_variants[h1 as usize % n_surf]
                    );
                    // y=2: sky (all air)
                    c[slot_index(x, 2, z)] = Child::Node(air_at_level);
                }
            }
            new_surface.push(lib.insert(c));
        }

        // 2 underground variants: solid terrain at all y levels.
        for variant in 0..2u32 {
            let mut c = empty_children();
            for z in 0..BRANCH {
                for y in 0..BRANCH {
                    for x in 0..BRANCH {
                        let h = pos_hash(x, y, z, seed_base + 500 + variant * 100);
                        c[slot_index(x, y, z)] = Child::Node(
                            underground_variants[h as usize % n_under]
                        );
                    }
                }
            }
            new_underground.push(lib.insert(c));
        }

        // Wrap air up one more level for the next iteration.
        air_at_level = lib.insert(uniform_children(Child::Node(air_at_level)));

        surface_variants = new_surface;
        underground_variants = new_underground;
    }

    // Final root: y-structured.
    let mut root_children = empty_children();
    let n_surf = surface_variants.len();
    let n_under = underground_variants.len();
    for z in 0..BRANCH {
        for x in 0..BRANCH {
            let h = pos_hash(x, 0, z, 99999);
            root_children[slot_index(x, 0, z)] = Child::Node(
                underground_variants[h as usize % n_under]
            );
            let h = pos_hash(x, 1, z, 99998);
            root_children[slot_index(x, 1, z)] = Child::Node(
                surface_variants[h as usize % n_surf]
            );
            root_children[slot_index(x, 2, z)] = Child::Node(air_at_level);
        }
    }

    let root = lib.insert(root_children);
    lib.ref_inc(root);

    let world = WorldState { root, library: lib };
    let d = world.tree_depth();
    eprintln!(
        "Generated world: {} unique nodes, depth {}, 3^{} ≈ {:.2e} blocks per axis",
        world.library.len(), d, d, 3.0f64.powi(d as i32),
    );

    world
}
