//! Sierpinski pyramid — 4 base corners + 1 apex, recursively.
//!
//! # Structure
//!
//! A stepped Sierpinski tetrahedron in a more "architectural" form:
//! at every level we keep 4 corners of the y=0 base and the centre of
//! the y=2 top:
//!
//! ```text
//!   base (y=0):  (0,0,0)  (2,0,0)  (0,0,2)  (2,0,2)
//!   apex (y=2):  (1,2,1)
//! ```
//!
//! Recursing yields a ziggurat-like self-similar pyramid — the apex
//! of every sub-pyramid has its own 4 corners + apex at the next
//! level. 5 / 27 cells filled per level → quite sparse.
//!
//! # Source
//!
//! PySpace's `sierpinski_tetrahedron` places 4 tetrahedral corners
//! at every level; here we reshape those into a pyramid stacked on
//! one axis, which reads better in a voxel world with a defined
//! "up" direction. The silhouette is a 4-sided stepped pyramid.
//!
//! # Coloring
//!
//! Egyptian-sandstone palette: base blocks are a warm limestone, the
//! apex is a brighter gold. Together with the self-similar stacking,
//! each zoom level shows the same "gold-tipped stone pyramid" motif
//! — the closest voxel equivalent to PySpace's `mausoleum` scene
//! (which uses `OrbitMax((0.42, 0.38, 0.19))` for ochre highlights
//! on a Menger-derived ziggurat).

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// 4 base corners at y=0, 1 apex at y=2 centre. `(x, y, z)` slots.
const PYRAMID: [(u8, u8, u8); 5] = [
    (0, 0, 0),
    (2, 0, 0),
    (0, 0, 2),
    (2, 0, 2),
    (1, 2, 1),
];

fn sierpinski_pyramid_world(depth: u8, base: u16, apex: u16) -> WorldState {
    let slots: Vec<Slot> = PYRAMID
        .iter()
        .map(|&(x, y, z)| {
            let block = if y == 2 { apex } else { base };
            (x, y, z, block)
        })
        .collect();
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_sierpinski_pyramid_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    let mut registry = ColorRegistry::new();
    // Limestone base, bright gilded apex.
    let base = registry.register(190, 160, 95, 255).unwrap();
    let apex = registry.register(235, 195, 85, 255).unwrap();

    let world = sierpinski_pyramid_world(depth, base, apex);

    // Far-diagonal pose (see `scripts/test-fractals.sh`): body-diagonal
    // vantage shows the pyramid stack (base corners + apex) recursed
    // at every level in a single frame.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [2.8, 2.8, 2.8],
        2,
        &world.library,
        world.root,
    )
    .deepened_to(8, &world.library, world.root);
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
    fn pyramid_has_expected_depth() {
        let w = sierpinski_pyramid_world(5, 11, 12);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn pyramid_dedup() {
        let w = sierpinski_pyramid_world(6, 11, 12);
        assert_eq!(w.library.len(), 6);
    }
}
