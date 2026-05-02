//! Cantor dust 3D — tetrahedral 4-corner pattern per level.
//!
//! # Structure
//!
//! In the original base-3 tree, Cantor dust kept the 8 corner cells
//! (coords ∈ {0, 2}) of a 3×3×3 subdivision. In a base-2 octree all
//! 8 slots ARE corners, so a literal port would be a full cube (not
//! interesting as a fractal).
//!
//! Instead we pick 4 alternating corners forming a tetrahedral pattern:
//! (0,0,0), (1,1,0), (1,0,1), (0,1,1). Each pair of occupied slots
//! differs in exactly 2 coordinates — a regular tetrahedron inscribed
//! in the cube. 4/8 = 50% occupancy per level.
//!
//! # Source
//!
//! PySpace doesn't ship a Cantor-dust fractal (its DE toolbox doesn't
//! really fit the "isolated points" structure). We include it here
//! because its sparse, symmetric structure stress-tests the renderer's
//! traversal with a high empty-space ratio.
//!
//! # Coloring — orbit-trap rainbow
//!
//! We colour each of the 4 corners with a different hue, then every
//! level inherits the same role assignment so zooming in reveals the
//! same prism.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// 4 tetrahedral corners of the 2×2×2 subdivision.
const CORNERS: [(u8, u8, u8); 4] = [
    (0, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
];

/// A colour wheel of 4 vivid hues (RGB 0-255, full-opacity alpha).
/// Ordered to match `CORNERS` so corner `i` gets hue `i`.
const HUES: [(u8, u8, u8); 4] = [
    (230,  70,  70),   // red
    (230, 210,  70),   // yellow
    ( 70, 130, 230),   // blue
    (230,  95, 185),   // magenta
];

fn cantor_dust_world(depth: u8, hue_indices: [u16; 4]) -> WorldState {
    let slots: Vec<Slot> = CORNERS
        .iter()
        .zip(hue_indices.iter())
        .map(|(&(x, y, z), &h)| (x, y, z, h))
        .collect();
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_cantor_dust_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    let mut registry = ColorRegistry::new();
    let mut hue_ids = [0u16; 4];
    for (i, &(r, g, b)) in HUES.iter().enumerate() {
        hue_ids[i] = registry.register(r, g, b, 255).unwrap();
    }

    let world = cantor_dust_world(depth, hue_ids);

    // Far-diagonal pose: symmetric body-diagonal vantage, so all 4
    // tetrahedral corners appear around the frame.
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
    fn cantor_dust_has_expected_depth() {
        let w = cantor_dust_world(5, [11, 12, 13, 14]);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn cantor_dust_dedup() {
        // 4 distinct block ids at the leaf level. Every level above
        // reuses the same structure -> one node per level.
        let w = cantor_dust_world(6, [11, 12, 13, 14]);
        assert_eq!(w.library.len(), 6);
    }
}
