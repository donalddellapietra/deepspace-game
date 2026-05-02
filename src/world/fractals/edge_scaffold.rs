//! Edge scaffold — 6 face-adjacent slots per level.
//!
//! # Structure
//!
//! In the base-3 tree, the edge scaffold kept 12 edge-midpoint cells
//! (exactly one coord = 1). In a base-2 octree there are no edge
//! midpoints (no center coordinate).
//!
//! Instead we keep the 6 slots that are NOT on the body diagonal:
//! all slots except the two body-diagonal corners (0,0,0) and
//! (1,1,1). Each kept slot has exactly 1 or 2 coords that differ
//! from (0,0,0), forming an edge-adjacent shell. 6/8 = 75% occupancy.
//!
//! ```text
//!   (1,0,0)  (0,1,0)  (0,0,1)  (1,1,0)  (1,0,1)  (0,1,1)
//! ```
//!
//! Recursing gives a self-similar lattice that omits the two opposite
//! corners at every level — an increasingly intricate structural
//! framework with diagonal voids.
//!
//! # Coloring — three-axis hue split
//!
//! We split the 6 slots by which single axis is "aligned" (has the
//! same value) relative to the two removed diagonal corners. Slots
//! with z matching get one colour, y another, x a third. The result
//! reads as an RGB-coded orthogonal lattice.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// The 6 slots excluding the body-diagonal corners (0,0,0) and (1,1,1).
const SCAFFOLD: [(u8, u8, u8); 6] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
];

fn edge_scaffold_world(depth: u8, x_axis: u16, y_axis: u16, z_axis: u16) -> WorldState {
    // Colour by which coordinate is zero (the "aligned" axis):
    //   z=0: (1,0,0), (0,1,0), (1,1,0) -> group by pair sums
    // Actually, simpler: colour by the number of 1s.
    //   1 nonzero: (1,0,0)=x, (0,1,0)=y, (0,0,1)=z
    //   2 nonzero: (1,1,0), (1,0,1), (0,1,1) — the zero coord picks axis
    let slots: Vec<Slot> = SCAFFOLD
        .iter()
        .map(|&(x, y, z)| {
            let block = if z == 0 {
                x_axis   // XY plane slots
            } else if y == 0 {
                y_axis   // XZ plane slots
            } else {
                z_axis   // YZ plane slots
            };
            (x, y, z, block)
        })
        .collect();
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_edge_scaffold_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    // Neon cyan / magenta / yellow — electric axial palette that
    // survives LOD averaging (each pair still averages to a distinct
    // mid-tone, unlike 8-corner rainbow which averages to gray).
    let mut registry = ColorRegistry::new();
    let x_axis = registry.register(80, 210, 235, 255).unwrap();    // cyan
    let y_axis = registry.register(225, 85, 180, 255).unwrap();    // magenta
    let z_axis = registry.register(240, 215, 70, 255).unwrap();    // yellow

    let world = edge_scaffold_world(depth, x_axis, y_axis, z_axis);

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
    fn scaffold_has_expected_depth() {
        let w = edge_scaffold_world(5, 11, 12, 13);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn scaffold_dedup() {
        let w = edge_scaffold_world(6, 11, 12, 13);
        assert_eq!(w.library.len(), 6);
    }
}
