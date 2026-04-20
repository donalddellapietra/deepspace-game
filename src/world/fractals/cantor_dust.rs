//! Cantor dust 3D — 8 corner cells per level.
//!
//! # Structure
//!
//! Cantor dust is the canonical ternary set: keep the two outer
//! thirds of `[0, 1]`, remove the middle third, recurse. In 3D that
//! means keep cells where every coord is ∈ {0, 2} — exactly the 8
//! cube corners of the 3×3×3 subdivision. No edges, no face cells,
//! no body centre.
//!
//! # Source
//!
//! PySpace doesn't ship a Cantor-dust fractal (its DE toolbox doesn't
//! really fit the "isolated points" structure). We include it here
//! because the ternary subdivision makes it natural: it's the sparsest
//! non-trivial fractal our tree supports, and it stress-tests the
//! renderer's traversal with a very high empty-space ratio.
//!
//! # Coloring — orbit-trap rainbow
//!
//! PySpace's `mandelbox` / `test_fractal` produce prismatic colour via
//! `OrbitInitInf` + `OrbitMinAbs(1.0)`: each trapped point lands at a
//! different distance from the origin, giving different hues. The
//! 8-corner structure of Cantor dust is perfect for this — we colour
//! each of the 8 corners with a different hue around the colour wheel,
//! then every level inherits the same role assignment so zooming in
//! reveals the same prism.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// The 8 cube corners of the 3×3×3 subdivision.
const CORNERS: [(u8, u8, u8); 8] = [
    (0, 0, 0),
    (2, 0, 0),
    (0, 2, 0),
    (2, 2, 0),
    (0, 0, 2),
    (2, 0, 2),
    (0, 2, 2),
    (2, 2, 2),
];

/// A colour wheel of 8 vivid hues (RGB 0-255, full-opacity alpha).
/// Ordered to match `CORNERS` so corner `i` gets hue `i`. The palette
/// is loosely a prismatic wheel: red → orange → yellow → lime → green
/// → cyan → blue → magenta. Saturation is high because Cantor dust
/// cells are small and need to read well against dark background.
const HUES: [(u8, u8, u8); 8] = [
    (230,  70,  70),   // red
    (235, 150,  55),   // orange
    (230, 210,  70),   // yellow
    ( 90, 210,  90),   // green
    ( 70, 220, 200),   // cyan
    ( 70, 130, 230),   // blue
    (170,  90, 220),   // indigo
    (230,  95, 185),   // magenta
];

fn cantor_dust_world(depth: u8, hue_indices: [u16; 8]) -> WorldState {
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
    let mut hue_ids = [0u16; 8];
    for (i, &(r, g, b)) in HUES.iter().enumerate() {
        hue_ids[i] = registry.register(r, g, b, 255).unwrap();
    }

    let world = cantor_dust_world(depth, hue_ids);

    // Far-diagonal pose (see `scripts/test-fractals.sh`): symmetric
    // body-diagonal vantage, so all 8 corner hues appear around the
    // frame (corners closest to camera dominate, opposite corners
    // shrink into the prism's vanishing point).
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
    fn cantor_dust_has_expected_depth() {
        let w = cantor_dust_world(5, [11, 12, 13, 14, 15, 16, 17, 18]);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn cantor_dust_dedup() {
        // 8 distinct block ids → leaf node unique. Every level above
        // reuses the same structure → one node per level.
        let w = cantor_dust_world(6, [11, 12, 13, 14, 15, 16, 17, 18]);
        assert_eq!(w.library.len(), 6);
    }
}
