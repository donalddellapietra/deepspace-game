//! World generator: builds a 21-level deep world from blocks to planets.

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

    // ═══ L1: block clusters ═══
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

    // ═══ L2: terrain surfaces ═══
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

    // ═══ L3: terrain chunks (9×9×9) ═══
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

    // ═══ L4: terrain sections (27×27×27) ═══
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

    // ═══ L5: biome regions (81×81×81) ═══
    let temperate5 = {
        let mut c = empty_children();
        for z in 0..3 { for y in 0..3 { for x in 0..3 {
            c[slot_index(x, y, z)] = match (y, pos_hash(x, y, z, 400) % 5) {
                (_, 0) | (_, 3) => Child::Node(forest4),
                (_, 1) => Child::Node(village4),
                _ => Child::Node(plains4),
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
    let _coastal5 = {
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

    // ═══ L6: base terrain (243×243×243) ═══
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

    // ═══ LEVELS 7-21: wrapping ═══
    //
    // Each wrapping level N builds new fillers of depth N from the
    // previous level's fillers (depth N-1). The wrapping node uses
    // the PREVIOUS fillers (depth N-1) alongside the chain (depth N-1),
    // so the wrapping node has depth 1 + (N-1) = N.
    //
    // The new fillers are stored for the NEXT iteration.

    // Starting fillers: depth 6 (matches terrain6).
    let mut solid_prev = {
        let mut s = stone1;
        for _ in 0..5 { s = lib.insert(uniform_children(Child::Node(s))); }
        s
    };
    let mut sky_prev = {
        let mut s = air1;
        for _ in 0..5 { s = lib.insert(uniform_children(Child::Node(s))); }
        s
    };
    let mut surface_prev = terrain6;

    let mut chain = terrain6;

    for level in 7..=TARGET_DEPTH {
        let depth = TARGET_DEPTH - level;
        let cy = camera_y(depth);

        // Build new fillers (depth = level) from prev fillers (depth = level-1).
        let solid_new = lib.insert(uniform_children(Child::Node(solid_prev)));
        let sky_new = lib.insert(uniform_children(Child::Node(sky_prev)));
        let surface_new = {
            let mut c = empty_children();
            for z in 0..3 { for x in 0..3 {
                c[slot_index(x, 0, z)] = Child::Node(solid_prev);
                let h = pos_hash(x, level, z, 800 + level as u32 * 37) as usize;
                // Mix different surface styles at y=1.
                c[slot_index(x, 1, z)] = match h % 4 {
                    0 => Child::Node(surface_prev),
                    1 | 2 => Child::Node(surface_prev),
                    _ => Child::Node(solid_prev), // occasional "lake" of solid
                };
                c[slot_index(x, 2, z)] = Child::Node(sky_prev);
            }}
            lib.insert(c)
        };

        // Build the wrapping node using PREVIOUS fillers (depth N-1).
        let mut ch = empty_children();

        if depth <= 1 {
            // Space: planet at center, void elsewhere.
            ch[slot_index(1, 1, 1)] = Child::Node(chain);
            if depth == 1 {
                ch[slot_index(2, 1, 0)] = Child::Node(solid_prev); // moon
            }
        } else if depth <= 5 {
            // Planet: core, surface+ocean, atmosphere.
            for z in 0..3 { for x in 0..3 {
                ch[slot_index(x, 0, z)] = Child::Node(solid_prev);
                let h = pos_hash(x, level, z, 900) as usize;
                ch[slot_index(x, 1, z)] = if h % 3 == 0 {
                    Child::Node(solid_prev) // ocean-like
                } else {
                    Child::Node(surface_prev)
                };
                ch[slot_index(x, 2, z)] = Child::Node(sky_prev);
            }}
            ch[slot_index(1, cy, 1)] = Child::Node(chain);
        } else {
            // Terrain: underground, varied surface, sky.
            for z in 0..3 { for x in 0..3 {
                ch[slot_index(x, 0, z)] = Child::Node(solid_prev);
                let h = pos_hash(x, level, z, 1000 + level as u32 * 37) as usize;
                ch[slot_index(x, 1, z)] = if h % 4 == 0 {
                    Child::Node(solid_prev) // rocky patch
                } else {
                    Child::Node(surface_prev)
                };
                ch[slot_index(x, 2, z)] = Child::Node(sky_prev);
            }}
            ch[slot_index(1, cy, 1)] = Child::Node(chain);
        }

        chain = lib.insert(ch);

        // Advance fillers for next iteration.
        solid_prev = solid_new;
        sky_prev = sky_new;
        surface_prev = surface_new;
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
