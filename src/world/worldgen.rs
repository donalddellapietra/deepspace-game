//! World generator: builds a 21-level deep world from blocks to planets.
//!
//! The y-structure is maintained at every level so the camera at
//! [1.5, 1.75, 1.5] always sits on a surface. The camera path
//! enters cells (1, y, 1) where y follows the pattern 1, 2, 0, 2, 0, ...
//! from root downward (the base-3 expansion of 1.75).

use super::palette::block;
use super::state::WorldState;
use super::tree::*;

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

/// Which y-cell the camera enters at `depth` levels below root.
/// Camera y=1.75, base-3: 1.202020...
fn camera_y(depth: usize) -> usize {
    match depth {
        0 => 1,
        d if d % 2 == 1 => 2,
        _ => 0,
    }
}

const TARGET_DEPTH: usize = 21;


pub fn generate_world() -> WorldState {
    let mut lib = NodeLibrary::default();

    // ═══ LEVEL 1: uniform block clusters ═══

    let stone1 = lib.insert(uniform_children(Child::Block(block::STONE)));
    let dirt1 = lib.insert(uniform_children(Child::Block(block::DIRT)));
    let grass1 = lib.insert(uniform_children(Child::Block(block::GRASS)));
    let wood1 = lib.insert(uniform_children(Child::Block(block::WOOD)));
    let leaf1 = lib.insert(uniform_children(Child::Block(block::LEAF)));
    let sand1 = lib.insert(uniform_children(Child::Block(block::SAND)));
    let brick1 = lib.insert(uniform_children(Child::Block(block::BRICK)));
    let water1 = lib.insert(uniform_children(Child::Block(block::WATER)));
    let air1 = lib.insert(empty_children());

    let gravel1 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = if pos_hash(x, y, z, 42) % 3 == 0 {
                Child::Block(block::DIRT)
            } else {
                Child::Block(block::STONE)
            };
        }}}
        lib.insert(c)
    };

    // ═══ LEVEL 2: terrain surfaces ═══

    let stone2 = lib.insert(uniform_children(Child::Node(stone1)));
    let sand2 = lib.insert(uniform_children(Child::Node(sand1)));
    let water2 = lib.insert(uniform_children(Child::Node(water1)));
    let air2 = lib.insert(uniform_children(Child::Node(air1)));

    let grass_sfc2 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(dirt1);
            c[slot_index(x, 1, z)] = Child::Node(grass1);
            c[slot_index(x, 2, z)] = Child::Node(air1);
        }}
        lib.insert(c)
    };

    let sand_sfc2 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(sand1);
            c[slot_index(x, 1, z)] = Child::Node(sand1);
            c[slot_index(x, 2, z)] = Child::Node(air1);
        }}
        lib.insert(c)
    };

    let rocky_sfc2 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone1);
            c[slot_index(x, 1, z)] = Child::Node(gravel1);
            c[slot_index(x, 2, z)] = Child::Node(air1);
        }}
        lib.insert(c)
    };

    let trunk2 = {
        let mut c = uniform_children(Child::Node(air1));
        c[slot_index(1, 0, 1)] = Child::Node(wood1);
        c[slot_index(1, 1, 1)] = Child::Node(wood1);
        c[slot_index(1, 2, 1)] = Child::Node(wood1);
        lib.insert(c)
    };

    let canopy2 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = if pos_hash(x, y, z, 77) % 4 == 0 {
                Child::Node(air1)
            } else {
                Child::Node(leaf1)
            };
        }}}
        lib.insert(c)
    };

    let wall2 = lib.insert(uniform_children(Child::Node(brick1)));

    // ═══ LEVEL 3: terrain chunks (9×9×9) ═══

    let grassland3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone2);
            c[slot_index(x, 1, z)] = Child::Node(grass_sfc2);
            c[slot_index(x, 2, z)] = Child::Node(air2);
        }}
        lib.insert(c)
    };

    let desert3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(sand2);
            c[slot_index(x, 1, z)] = Child::Node(sand_sfc2);
            c[slot_index(x, 2, z)] = Child::Node(air2);
        }}
        lib.insert(c)
    };

    let rocky3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone2);
            c[slot_index(x, 1, z)] = Child::Node(rocky_sfc2);
            c[slot_index(x, 2, z)] = Child::Node(air2);
        }}
        lib.insert(c)
    };

    let forest3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone2);
            c[slot_index(x, 1, z)] = Child::Node(grass_sfc2);
            c[slot_index(x, 2, z)] = Child::Node(air2);
        }}
        c[slot_index(1, 2, 1)] = Child::Node(trunk2);
        lib.insert(c)
    };

    let dense_forest3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone2);
            c[slot_index(x, 1, z)] = Child::Node(grass_sfc2);
            c[slot_index(x, 2, z)] = if pos_hash(x, 0, z, 123) % 3 == 0 {
                Child::Node(trunk2)
            } else if pos_hash(x, 0, z, 123) % 3 == 1 {
                Child::Node(canopy2)
            } else {
                Child::Node(air2)
            };
        }}
        lib.insert(c)
    };

    let mountain3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone2);
            c[slot_index(x, 1, z)] = Child::Node(stone2);
            c[slot_index(x, 2, z)] = if x == 1 && z == 1 {
                Child::Node(rocky_sfc2)
            } else {
                Child::Node(air2)
            };
        }}
        lib.insert(c)
    };

    let village3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone2);
            c[slot_index(x, 1, z)] = Child::Node(grass_sfc2);
            c[slot_index(x, 2, z)] = Child::Node(air2);
        }}
        c[slot_index(1, 2, 1)] = Child::Node(wall2);
        c[slot_index(0, 2, 1)] = Child::Node(wall2);
        c[slot_index(1, 2, 0)] = Child::Node(wall2);
        lib.insert(c)
    };

    let ocean3 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(stone2);
            c[slot_index(x, 1, z)] = Child::Node(water2);
            c[slot_index(x, 2, z)] = Child::Node(air2);
        }}
        lib.insert(c)
    };

    // ═══ LEVEL 4: terrain sections (27×27×27) ═══

    let plains4 = lib.insert(uniform_children(Child::Node(grassland3)));
    let desert4 = lib.insert(uniform_children(Child::Node(desert3)));
    let ocean4 = lib.insert(uniform_children(Child::Node(ocean3)));

    let forest4 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = match y {
                2 => Child::Node(grassland3),
                1 => if pos_hash(x, y, z, 200) % 2 == 0 {
                    Child::Node(forest3)
                } else {
                    Child::Node(dense_forest3)
                },
                _ => Child::Node(grassland3),
            };
        }}}
        lib.insert(c)
    };

    let mountain4 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = match y {
                2 => Child::Node(rocky3),
                1 => if pos_hash(x, 0, z, 300) % 3 == 0 {
                    Child::Node(mountain3)
                } else {
                    Child::Node(rocky3)
                },
                _ => Child::Node(rocky3),
            };
        }}}
        lib.insert(c)
    };

    let village4 = {
        let mut c = uniform_children(Child::Node(grassland3));
        c[slot_index(1, 1, 1)] = Child::Node(village3);
        c[slot_index(0, 1, 0)] = Child::Node(forest3);
        c[slot_index(2, 1, 2)] = Child::Node(forest3);
        lib.insert(c)
    };

    // ═══ LEVEL 5: biome regions (81×81×81) ═══

    let temperate5 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = match (y, pos_hash(x, y, z, 400) % 5) {
                (_, 0) | (_, 3) => Child::Node(forest4),
                (_, 1) => Child::Node(village4),
                (_, _) => Child::Node(plains4),
            };
        }}}
        lib.insert(c)
    };

    let highland5 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = if pos_hash(x, y, z, 500) % 3 == 1 {
                Child::Node(forest4)
            } else {
                Child::Node(mountain4)
            };
        }}}
        lib.insert(c)
    };

    let arid5 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = if pos_hash(x, y, z, 600) % 4 == 0 {
                Child::Node(mountain4)
            } else {
                Child::Node(desert4)
            };
        }}}
        lib.insert(c)
    };

    let coastal5 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = if pos_hash(x, y, z, 650) % 3 == 0 {
                Child::Node(ocean4)
            } else {
                Child::Node(plains4)
            };
        }}}
        lib.insert(c)
    };

    // ═══ LEVEL 6: world (243×243×243) — the base terrain ═══

    let mut l6ch = empty_children();
    for z in 0..3 { for x in 0..3 {
        l6ch[slot_index(x, 0, z)] = Child::Node(highland5);
        l6ch[slot_index(x, 2, z)] = Child::Node(temperate5);
        l6ch[slot_index(x, 1, z)] = match pos_hash(x, 1, z, 700) % 3 {
            0 => Child::Node(temperate5),
            1 => Child::Node(highland5),
            _ => Child::Node(arid5),
        };
    }}
    let terrain6 = lib.insert(l6ch);

    // ═══ LEVELS 7-21: wrap terrain to planet scale ═══
    //
    // At each wrapping level we build a node where:
    //   - The camera-path cell contains the chain (the deeper world)
    //   - All other cells are filled with the L6 terrain/solid/sky
    //     (same depth, no cascading depth growth)
    //
    // The biome_pool gives visual variety — different L6-depth nodes
    // for different positions. Since they're all depth 6, the wrapping
    // node depth = 1 + max(6, chain_depth) = chain_depth + 1.

    // Build uniform solid and sky chains up to depth 6.
    let mut solid = stone1;
    let mut sky = air1;
    for _ in 0..5 {
        solid = lib.insert(uniform_children(Child::Node(solid)));
        sky = lib.insert(uniform_children(Child::Node(sky)));
    }
    let solid6 = solid;
    let sky6 = sky;

    let ocean6 = {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(highland5);
            c[slot_index(x, 1, z)] = Child::Node(coastal5);
            c[slot_index(x, 2, z)] = Child::Node(arid5);
        }}
        lib.insert(c)
    };

    // Pool of L6-depth biome nodes for filling terrain cells.
    let biome_pool = [terrain6, ocean6, {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(highland5);
            c[slot_index(x, 1, z)] = Child::Node(coastal5);
            c[slot_index(x, 2, z)] = Child::Node(temperate5);
        }}
        lib.insert(c)
    }, {
        let mut c = empty_children();
        for z in 0..3 { for x in 0..3 {
            c[slot_index(x, 0, z)] = Child::Node(arid5);
            c[slot_index(x, 1, z)] = Child::Node(arid5);
            c[slot_index(x, 2, z)] = Child::Node(highland5);
        }}
        lib.insert(c)
    }];

    let mut chain = terrain6;

    for level in 7..=TARGET_DEPTH {
        let depth = TARGET_DEPTH - level;
        let cy = camera_y(depth);

        let mut ch = empty_children();

        if depth <= 1 {
            // Space: planet at center, void elsewhere.
            ch[slot_index(1, 1, 1)] = Child::Node(chain);
            if depth == 1 {
                // Moon.
                ch[slot_index(2, 1, 0)] = Child::Node(solid6);
            }
        } else if depth <= 5 {
            // Planet: core below, terrain+ocean middle, sky above.
            for z in 0..3 { for x in 0..3 {
                ch[slot_index(x, 0, z)] = Child::Node(solid6);
                let h = pos_hash(x, level, z, 900) as usize;
                ch[slot_index(x, 1, z)] = if h % 3 == 0 {
                    Child::Node(ocean6)
                } else {
                    Child::Node(biome_pool[h % biome_pool.len()])
                };
                ch[slot_index(x, 2, z)] = Child::Node(sky6);
            }}
            ch[slot_index(1, cy, 1)] = Child::Node(chain);
        } else {
            // Terrain: solid below, varied terrain middle, sky above.
            for z in 0..3 { for x in 0..3 {
                ch[slot_index(x, 0, z)] = Child::Node(solid6);
                let h = pos_hash(x, level, z, 1000 + level as u32 * 37) as usize;
                ch[slot_index(x, 1, z)] =
                    Child::Node(biome_pool[h % biome_pool.len()]);
                ch[slot_index(x, 2, z)] = Child::Node(sky6);
            }}
            ch[slot_index(1, cy, 1)] = Child::Node(chain);
        }

        chain = lib.insert(ch);
    }

    lib.ref_inc(chain);

    let world = WorldState { root: chain, library: lib };
    let d = world.tree_depth();
    eprintln!(
        "Generated world: {} unique nodes, depth {}, 3^{} blocks per axis",
        world.library.len(), d, d,
    );

    world
}
