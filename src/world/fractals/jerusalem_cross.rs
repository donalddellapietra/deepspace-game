//! Jerusalem cross — the complement of Menger.
//!
//! # Structure
//!
//! Where Menger removes the 7 "interior" cells (6 face centres + 1
//! body centre), this fractal keeps *only those 7*. Each level is a
//! 3-axis plus sign embedded in a 3×3×3 cube: three orthogonal rods
//! passing through the central cell.
//!
//! ```text
//!   (1,1,0) ──┐   ┌── (1,1,2)
//!             │   │
//!   (0,1,1) ──●── (2,1,1)         ● = (1,1,1) body centre
//!             │   │
//!   (1,0,1) ──┘   └── (1,2,1)
//! ```
//!
//! Zooming in reveals smaller crosses joined at every rod end — a
//! delicate axial scaffold, the exact opposite of Menger's corner-rib
//! weave.
//!
//! # Source
//!
//! PySpace doesn't have a direct analog, but this is in spirit the
//! "inverse Menger" you can construct from `FoldAbs` + `FoldPlane`
//! with inverted half-spaces. Our trinary subdivision makes it
//! essentially free to express.
//!
//! # Coloring
//!
//! Architectural two-tone: the body centre gets a warm gold
//! "nucleus", the 6 rod cells get a darker ochre. At every zoom the
//! scaffold reads as connected lattice → core → connected lattice,
//! evoking PySpace's `mausoleum` orbit-trap palette
//! `(0.42, 0.38, 0.19)`.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// Cross cells: body centre + 6 face centres.
const CROSS: [(u8, u8, u8); 7] = [
    (1, 1, 1),   // body centre
    (0, 1, 1),   // -X face
    (2, 1, 1),   // +X face
    (1, 0, 1),   // -Y face
    (1, 2, 1),   // +Y face
    (1, 1, 0),   // -Z face
    (1, 1, 2),   // +Z face
];

fn jerusalem_cross_world(depth: u8, nucleus: u16, rod: u16) -> WorldState {
    let slots: Vec<Slot> = CROSS
        .iter()
        .map(|&(x, y, z)| {
            let block = if (x, y, z) == (1, 1, 1) { nucleus } else { rod };
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
    //   `OrbitMax((0.42, 0.38, 0.19))` × 1.0 → (107, 97, 48) rod
    //                                    × 1.75 → (178, 161, 81) highlight
    // Nucleus uses the highlight tone so the single-cell core pops
    // against the 6-rod scaffold at every recursion level.
    let nucleus = registry.register(178, 161, 81, 255).unwrap();
    let rod = registry.register(107, 97, 48, 255).unwrap();

    let world = jerusalem_cross_world(depth, nucleus, rod);

    // Far-diagonal pose (see `scripts/test-fractals.sh`): from the
    // (+X,+Y,+Z) corner looking back along the body diagonal. The
    // three orthogonal rods converge toward the camera, giving the
    // cross its signature ornate filigree silhouette.
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
