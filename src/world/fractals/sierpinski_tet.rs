//! Sierpinski tetrahedron — binary octree adaptation.
//!
//! # Source
//!
//! PySpace's `sierpinski_tetrahedron` scene:
//!
//! ```text
//! obj.add(OrbitInitZero())
//! for _ in range(9):
//!     obj.add(FoldSierpinski())
//!     obj.add(FoldScaleTranslate(2, -1))
//! obj.add(Tetrahedron(color=(0.8, 0.8, 0.5)))
//! ```
//!
//! `FoldSierpinski` folds at the three planes (1,1,0), (1,0,1),
//! (0,1,1), then `FoldScale(2)` re-scales — binary self-similarity
//! placing 4 sub-tetrahedra at the cube's tetrahedral corners.
//!
//! # Binary adaptation
//!
//! In a base-2 octree the 4 tetrahedral corners map directly to:
//!
//! ```text
//!   (0, 0, 0)  (1, 1, 0)  (1, 0, 1)  (0, 1, 1)
//! ```
//!
//! Each pair differs in exactly 2 coordinates — a regular tetrahedron
//! inscribed in the unit cube. 4/8 = 50% occupancy per level.
//!
//! # Coloring
//!
//! PySpace uses a solid cream `(0.8, 0.8, 0.5)`. We keep that as the
//! primary body colour and add a second accent (a warmer gold) at the
//! "apex" vertex `(0, 1, 1)` so one corner pops at every zoom level —
//! like the orbit-trap highlight on mausoleum-family fractals.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// Tetrahedral-corner pattern in the 2×2×2 subdivision. Four cells
/// chosen so every pair differs in exactly two coordinates.
const TET_CORNERS: [(u8, u8, u8); 4] = [
    (0, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
];

fn sierpinski_tet_world(depth: u8, body: u16, apex: u16) -> WorldState {
    // Apex = (0, 1, 1). All others body.
    let slots: Vec<Slot> = TET_CORNERS
        .iter()
        .map(|&(x, y, z)| {
            let block = if (x, y, z) == (0, 1, 1) { apex } else { body };
            (x, y, z, block)
        })
        .collect();
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_sierpinski_tet_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    let mut registry = ColorRegistry::new();
    // PySpace's tet colour: (0.8, 0.8, 0.5) -> cream.
    let body = registry.register(204, 204, 128, 255).unwrap();
    // Warm gold accent for the apex vertex.
    let apex = registry.register(230, 170, 60, 255).unwrap();

    let world = sierpinski_tet_world(depth, body, apex);

    // Far-diagonal pose: body-diagonal vantage point reveals the
    // tetrahedron's four corners as four distinct clusters arranged
    // around the centre of frame.
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
    fn dedup_one_node_per_level() {
        let w = sierpinski_tet_world(6, 11, 12);
        // 4 distinct block IDs (body + apex) at level 1, but the
        // whole tetrahedral pattern dedups to one node per deeper
        // level — so total = exactly depth unique nodes.
        assert_eq!(w.library.len(), 6);
    }

    #[test]
    fn correct_depth() {
        let w = sierpinski_tet_world(5, 11, 12);
        assert_eq!(w.tree_depth(), 5);
    }
}
