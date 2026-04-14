//! Worldgen: a small solar system of SDF-driven planets + a star.
//!
//! The root of the tree is the "space layer": a 3³ arrangement of
//! mostly-empty slots with a handful of planets placed at chosen
//! positions in world space [0, 3). One of the "planets" is actually
//! a star — a larger radius, emissive `core_block` = `STAR_CORE`,
//! `surface_block` = `STAR_SURFACE`. The star is generated the exact
//! same way as any other planet: by recursively sampling its SDF.
//! Emission intensity is stored per palette entry and inherits
//! through the tree's LOD averaging (representative_block), so the
//! star glows correctly even when viewed from far away as a single
//! coarse-cell in the cascaded tree.

use super::palette::block;
use super::sdf::{Planet, Vec3};
use super::state::WorldState;
use super::tree::*;

const TARGET_DEPTH: usize = 21;

/// Max levels the SDF recursion descends. Cells straddling the
/// surface are subdivided up to this depth; below it we commit to
/// solid/empty by the center sample.
const SDF_DETAIL_LEVELS: u32 = 5;
const SDF_SAFETY: f32 = 0.0;

pub fn generate_world() -> WorldState {
    let mut lib = NodeLibrary::default();

    // Planet 0: spawn planet (earth-like).
    // Planet 1: small sandy moon.
    // Planet 2: large volcanic/lava planet.
    // Planet 3: STAR — much larger, emissive. Acts as the light
    //   source in the scene; the shader computes per-point
    //   illumination toward the star's surface.
    let planets = vec![
        Planet {
            center: [0.8, 0.9, 0.8],
            radius: 0.30,
            noise_scale: 0.04,
            noise_freq: 45.0,
            noise_seed: 1337,
            gravity: 9.8,
            influence_radius: 0.7,
            surface_block: block::GRASS,
            core_block: block::STONE,
        },
        Planet {
            center: [0.4, 1.7, 1.3],
            radius: 0.14,
            noise_scale: 0.018,
            noise_freq: 90.0,
            noise_seed: 7,
            gravity: 5.0,
            influence_radius: 0.30,
            surface_block: block::SAND,
            core_block: block::STONE,
        },
        Planet {
            center: [1.6, 1.1, 1.9],
            radius: 0.22,
            noise_scale: 0.05,
            noise_freq: 35.0,
            noise_seed: 99,
            gravity: 11.0,
            influence_radius: 0.5,
            surface_block: block::DIRT,
            core_block: block::STONE,
        },
        // The star. Positioned far across the world from the spawn
        // planet so it reads as a distant giant sun.
        Planet {
            center: [2.55, 2.35, 2.55],
            radius: 0.45,
            noise_scale: 0.08,
            noise_freq: 20.0,
            noise_seed: 4242,
            gravity: 0.0,          // no gravity from a star in this toy sim
            influence_radius: 0.0, // no camera reorientation to it
            surface_block: block::STAR_SURFACE,
            core_block: block::STAR_CORE,
        },
    ];

    // Per-planet filler subtrees: deep interior cells use the
    // planet's core block, not a global stone default, so the star
    // glows in its interior and stone planets stay stone.
    let max_sub_depth = TARGET_DEPTH as u32;
    let empty_leaf = lib.insert(empty_children());
    let mut empty_fillers: Vec<NodeId> = vec![empty_leaf];
    for _ in 1..=max_sub_depth {
        let e = *empty_fillers.last().unwrap();
        empty_fillers.push(lib.insert(uniform_children(Child::Node(e))));
    }

    let mut planet_fillers: Vec<Vec<NodeId>> = Vec::new();
    for p in &planets {
        let leaf = lib.insert(uniform_children(Child::Block(p.core_block)));
        let mut fs = vec![leaf];
        for _ in 1..=max_sub_depth {
            let s = *fs.last().unwrap();
            fs.push(lib.insert(uniform_children(Child::Node(s))));
        }
        planet_fillers.push(fs);
    }

    // Build the root (space layer) by recursively sampling all
    // planets into a single subtree of depth TARGET_DEPTH.
    let mut root_children = empty_children();
    let root_scale = 1.0f32;
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let origin = [x as f32, y as f32, z as f32];
                let child = build_space_subtree(
                    &mut lib, &planets, origin, root_scale,
                    TARGET_DEPTH as u32 - 1,
                    SDF_DETAIL_LEVELS,
                    &planet_fillers, &empty_fillers,
                );
                root_children[slot_index(x, y, z)] = child;
            }
        }
    }
    let root = lib.insert(root_children);
    lib.ref_inc(root);

    let world = WorldState { root, library: lib, planets };
    let d = world.tree_depth();
    eprintln!(
        "Generated solar system: {} planets (including 1 star), {} unique nodes, depth {}",
        world.planets.len(), world.library.len(), d,
    );
    world
}

fn build_space_subtree(
    lib: &mut NodeLibrary,
    planets: &[Planet],
    origin: Vec3,
    scale: f32,
    depth: u32,
    sdf_budget: u32,
    planet_fillers: &[Vec<NodeId>],
    empty: &[NodeId],
) -> Child {
    let half = scale * 0.5;
    let center = [origin[0] + half, origin[1] + half, origin[2] + half];
    let half_diag = scale * 0.8660254; // sqrt(3)/2

    // Track the nearest planet (by SDF) so we can pick its fillers
    // and surface block when committing.
    let mut min_sdf = f32::INFINITY;
    let mut nearest_idx: usize = 0;
    for (i, p) in planets.iter().enumerate() {
        let d = p.distance(center);
        if d < min_sdf {
            min_sdf = d;
            nearest_idx = i;
        }
    }

    // Cell safely outside every planet: empty filler.
    if min_sdf - half_diag - SDF_SAFETY > 0.0 {
        if depth == 0 { return Child::Empty; }
        return Child::Node(empty[(depth - 1) as usize]);
    }

    let planet = &planets[nearest_idx];
    let fillers = &planet_fillers[nearest_idx];

    // Cell safely inside the nearest planet: planet-specific core filler.
    if min_sdf + half_diag + SDF_SAFETY < 0.0 {
        if depth == 0 { return Child::Block(planet.core_block); }
        return Child::Node(fillers[(depth - 1) as usize]);
    }

    // Leaf or SDF budget exhausted: decide by center sample.
    if depth == 0 || sdf_budget == 0 {
        let is_solid = min_sdf < 0.0;
        if depth == 0 {
            return if is_solid {
                Child::Block(planet.block_at(center))
            } else {
                Child::Empty
            };
        }
        return Child::Node(if is_solid {
            fillers[(depth - 1) as usize]
        } else {
            empty[(depth - 1) as usize]
        });
    }

    // Near-surface interior cell: subdivide.
    let child_scale = scale / 3.0;
    let mut children = empty_children();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let co = [
                    origin[0] + x as f32 * child_scale,
                    origin[1] + y as f32 * child_scale,
                    origin[2] + z as f32 * child_scale,
                ];
                children[slot_index(x, y, z)] = build_space_subtree(
                    lib, planets, co, child_scale, depth - 1, sdf_budget - 1,
                    planet_fillers, empty,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::edit;

    #[test]
    fn world_has_planets_and_star() {
        let w = generate_world();
        assert!(w.planets.len() >= 3);
        // At least one planet should have emissive core.
        let star = w.planets.iter().find(|p| p.core_block == block::STAR_CORE);
        assert!(star.is_some(), "expected at least one star");
    }

    #[test]
    fn planet_center_is_solid() {
        let w = generate_world();
        let depth = w.tree_depth();
        let c = w.planets[0].center;
        assert!(edit::is_solid_at(&w.library, w.root, c, depth),
            "planet center should be solid");
    }

    #[test]
    fn deep_space_is_empty() {
        let w = generate_world();
        let depth = w.tree_depth();
        assert!(!edit::is_solid_at(&w.library, w.root, [0.05, 2.9, 0.05], depth),
            "deep space should be empty");
    }

    #[test]
    fn dominant_planet_near_first() {
        let w = generate_world();
        let c = w.planets[0].center;
        let near = [c[0] + 0.05, c[1], c[2]];
        let p = w.dominant_planet(near).expect("should be inside first planet");
        assert_eq!(p.center, w.planets[0].center);
    }

    #[test]
    fn dominant_planet_none_in_void() {
        let w = generate_world();
        assert!(w.dominant_planet([0.05, 0.05, 0.05]).is_none());
    }
}
