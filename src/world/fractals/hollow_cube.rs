//! Hollow cube — 7/8 slots (all except one corner).
//!
//! # Structure
//!
//! In the base-3 tree, the hollow cube kept the 18-cell shell (edges +
//! faces, no corners or body centre). In a base-2 octree there is no
//! shell/interior distinction (all 8 slots are corners).
//!
//! We use the same 7/8 pattern as the Menger analogue: all slots
//! except (0,0,0). This gives a cube with one missing corner at every
//! recursion level — each sub-cube also has its (0,0,0) corner
//! removed, producing an increasingly intricate lace-like structure.
//! 7/8 = 87.5% occupancy per level.
//!
//! # Coloring
//!
//! Two structural roles, steel-architectural palette:
//! - 3 face-adjacent slots (1 coord nonzero): polished brass
//! - 4 distant slots (2+ coords nonzero): brushed steel
//!
//! The 3:4 face-adjacent:distant ratio means far-LOD averaging tilts
//! toward the distant colour (57% steel, 43% brass -> cool industrial
//! average).

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

fn hollow_cube_world(depth: u8, face: u16, edge: u16) -> WorldState {
    let mut slots: Vec<Slot> = Vec::with_capacity(7);
    for z in 0u8..2 {
        for y in 0u8..2 {
            for x in 0u8..2 {
                if x == 0 && y == 0 && z == 0 { continue; } // omit origin corner
                let nonzero = (x as u8) + (y as u8) + (z as u8);
                // 1 nonzero coord = face-adjacent (edge colour = brass)
                // 2+ nonzero coords = distant (face colour = steel)
                match nonzero {
                    1 => slots.push((x, y, z, edge)),
                    _ => slots.push((x, y, z, face)),
                }
            }
        }
    }
    debug_assert_eq!(slots.len(), 7, "hollow cube must have 7 slots");
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_hollow_cube_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    let mut registry = ColorRegistry::new();
    let face = registry.register(95, 105, 120, 255).unwrap();   // brushed steel
    let edge = registry.register(185, 150, 95, 255).unwrap();   // polished brass

    let world = hollow_cube_world(depth, face, edge);

    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.8, 1.8, 1.8],
        2,
    )
    .deepened_to(8);
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: -0.615,
        plain_layers: depth,
        color_registry: registry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hollow_cube_has_expected_depth() {
        let w = hollow_cube_world(5, 11, 12);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn hollow_cube_dedup() {
        let w = hollow_cube_world(6, 11, 12);
        assert_eq!(w.library.len(), 6);
    }

    #[test]
    fn exactly_7_slots() {
        let w = hollow_cube_world(1, 11, 12);
        // At depth 1 we have one root node with 7 Block children.
        assert_eq!(w.library.len(), 1);
    }
}
