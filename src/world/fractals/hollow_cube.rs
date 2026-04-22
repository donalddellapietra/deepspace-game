//! Hollow cube — 18-cell cube shell (edges + faces, no corners or body).
//!
//! # Structure
//!
//! The 18 cells that have at least one coord = 1 **and** at least
//! one coord ∈ {0, 2}:
//!
//! - 12 edge-midpoints (exactly one coord = 1)
//! - 6 face-centres (exactly two coords = 1)
//!
//! Excluded: 8 cube corners (all coords ∈ {0, 2}) and 1 body centre
//! (all coords = 1). Each level produces a lace-like shell whose
//! sub-cells are also hollow cubes — recurses into an increasingly
//! intricate architectural surface.
//!
//! # Source
//!
//! No direct PySpace scene. It's a natural complement to Menger
//! (which keeps corners + edges) and Jerusalem cross (body + faces):
//! hollow cube keeps edges + faces, giving the third and final
//! structural partition of the 3×3×3 by "count of centre
//! coordinates" role.
//!
//! # Coloring
//!
//! Two structural roles, steel-architectural palette: faces get a
//! cooler brushed-metal tone, edges get a warmer polished-brass
//! tone. The 12:6 edge:face ratio means far-LOD averaging tilts
//! toward the edge colour (66% brass, 33% steel → warm industrial
//! average). Consistent with PySpace's `mausoleum` aesthetic but
//! sharper — no ochre, steel + brass instead.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

fn hollow_cube_world(depth: u8, face: u16, edge: u16) -> WorldState {
    let mut slots: Vec<Slot> = Vec::with_capacity(18);
    for z in 0u8..3 {
        for y in 0u8..3 {
            for x in 0u8..3 {
                let center_axes = (x == 1) as u8 + (y == 1) as u8 + (z == 1) as u8;
                // 1 = edge-midpoint, 2 = face-centre. Skip 0 (corner) and 3 (body).
                match center_axes {
                    1 => slots.push((x, y, z, edge)),
                    2 => slots.push((x, y, z, face)),
                    _ => {}
                }
            }
        }
    }
    debug_assert_eq!(slots.len(), 18, "hollow cube must have 18 slots");
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
        [2.8, 2.8, 2.8],
        2,
    )
    .deepened_to(8);
    WorldBootstrap {
        world,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: -0.615,
        plain_layers: depth,
        color_registry: registry,
        body_path: None,
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
    fn exactly_18_slots() {
        let w = hollow_cube_world(1, 11, 12);
        // At depth 1 we have one root node with 18 Block children.
        assert_eq!(w.library.len(), 1);
    }
}
