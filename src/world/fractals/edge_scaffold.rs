//! Edge scaffold — 12 edge-midpoint cells per level.
//!
//! # Structure
//!
//! The 12 cells where exactly one coordinate is the centre (1) and
//! the other two are at the boundary ({0, 2}) — the midpoints of
//! the 12 edges of a cube:
//!
//! ```text
//!   X-axis edges (y,z ∈ {0,2}):  (1,0,0) (1,0,2) (1,2,0) (1,2,2)
//!   Y-axis edges (x,z ∈ {0,2}):  (0,1,0) (0,1,2) (2,1,0) (2,1,2)
//!   Z-axis edges (x,y ∈ {0,2}):  (0,0,1) (0,2,1) (2,0,1) (2,2,1)
//! ```
//!
//! Recursing gives a self-similar lattice of orthogonal rods —
//! "Menger without the corners, Jerusalem without the faces". Very
//! airy silhouette, feels like structural framework.
//!
//! # Source
//!
//! PySpace has no direct analog (it would require a complement-fold
//! we don't have primitives for). We include it because it's the
//! third natural 3×3×3 vocabulary element alongside Menger's corners
//! and Jerusalem's body+faces.
//!
//! # Coloring — three-axis hue split
//!
//! Each of the 12 rods has a natural "axis orientation" (the
//! coordinate that equals 1 tells you which axis the rod lies
//! along). We use that to paint all X-axis rods one colour,
//! Y-axis rods another, Z-axis rods a third. The result reads as
//! an RGB-coded orthogonal lattice — echoing the way PySpace's
//! `OrbitMinAbs(1.0)` assigns distinct hues to each axis based on
//! which coordinate the fold-iterate stayed closest to.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

fn edge_scaffold_world(depth: u8, x_axis: u16, y_axis: u16, z_axis: u16) -> WorldState {
    let mut slots: Vec<Slot> = Vec::with_capacity(12);
    for z in 0u8..3 {
        for y in 0u8..3 {
            for x in 0u8..3 {
                let cx = (x == 1) as u8;
                let cy = (y == 1) as u8;
                let cz = (z == 1) as u8;
                // Exactly one coord equals center, and the other two
                // are at the boundary (not center = 0 or 2).
                if cx + cy + cz != 1 { continue; }
                // Skip if any non-center coord is also 1 (shouldn't
                // happen given the above, but guard anyway).
                let block = if cx == 1 {
                    x_axis
                } else if cy == 1 {
                    y_axis
                } else {
                    z_axis
                };
                slots.push((x, y, z, block));
            }
        }
    }
    debug_assert_eq!(slots.len(), 12, "edge scaffold must have 12 slots");
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
    // These hues are chosen to be perceptually distant on the
    // colour wheel so even heavy mixing doesn't collapse to neutral.
    let mut registry = ColorRegistry::new();
    let x_axis = registry.register(80, 210, 235, 255).unwrap();    // cyan
    let y_axis = registry.register(225, 85, 180, 255).unwrap();    // magenta
    let z_axis = registry.register(240, 215, 70, 255).unwrap();    // yellow

    let world = edge_scaffold_world(depth, x_axis, y_axis, z_axis);

    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [2.8, 2.8, 2.8],
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
        proto_subtree_root: None,
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
