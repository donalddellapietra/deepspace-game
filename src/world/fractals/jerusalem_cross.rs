//! Jerusalem cross — L-shape fractal, sparse complement of the Menger analogue.
//!
//! # Structure
//!
//! In the base-3 tree, the Jerusalem cross kept the 7 "interior"
//! cells (6 face centres + 1 body centre) — the complement of Menger.
//! In a base-2 octree there are no face centres or body centre, so a
//! literal port is impossible.
//!
//! Instead we use an L-shaped pattern of 3 slots:
//!
//! ```text
//!   (0,0,0)  (1,0,0)  (0,1,0)
//! ```
//!
//! This forms an L in the XY plane at z=0. 3/8 = 37.5% occupancy.
//! Recursing produces a delicate angular scaffold — each L-arm
//! branches into smaller Ls at every zoom level.
//!
//! # Source
//!
//! PySpace doesn't have a direct analog, but this is in spirit a
//! sparse structural fractal. The L-shape provides interesting
//! asymmetric self-similarity.
//!
//! # Coloring
//!
//! Architectural two-tone: the corner (0,0,0) gets a warm gold
//! "nucleus", the 2 arm cells get a darker ochre. At every zoom the
//! scaffold reads as connected lattice -> core -> connected lattice,
//! evoking PySpace's `mausoleum` orbit-trap palette
//! `(0.42, 0.38, 0.19)`.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// L-shape cells: origin corner + two arms along X and Y.
const L_SHAPE: [(u8, u8, u8); 3] = [
    (0, 0, 0),   // corner (nucleus)
    (1, 0, 0),   // +X arm
    (0, 1, 0),   // +Y arm
];

fn jerusalem_cross_world(depth: u8, nucleus: u16, rod: u16) -> WorldState {
    let slots: Vec<Slot> = L_SHAPE
        .iter()
        .map(|&(x, y, z)| {
            let block = if (x, y, z) == (0, 0, 0) { nucleus } else { rod };
            (x, y, z, block)
        })
        .collect();
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_jerusalem_cross_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    let mut registry = ColorRegistry::new();
    // Mausoleum orbit-trap palette, authentic PySpace RGB:
    //   `OrbitMax((0.42, 0.38, 0.19))` x 1.0 -> (107, 97, 48) rod
    //                                    x 1.75 -> (178, 161, 81) highlight
    // Nucleus uses the highlight tone so the single-cell core pops
    // against the 2-rod scaffold at every recursion level.
    let nucleus = registry.register(178, 161, 81, 255).unwrap();
    let rod = registry.register(107, 97, 48, 255).unwrap();

    let world = jerusalem_cross_world(depth, nucleus, rod);

    // Far-diagonal pose: from the (+X,+Y,+Z) corner looking back
    // along the body diagonal.
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
    fn cross_has_expected_depth() {
        let w = jerusalem_cross_world(5, 11, 12);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn cross_dedup() {
        let w = jerusalem_cross_world(6, 11, 12);
        assert_eq!(w.library.len(), 6);
    }
}
