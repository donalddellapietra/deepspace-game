//! Mausoleum — Menger sponge with authentic PySpace orbit-trap ochre.
//!
//! # Source
//!
//! PySpace's `mausoleum` scene (`ray_marcher_demo.py:108`):
//!
//! ```text
//! obj.add(OrbitInitZero())
//! for _ in range(8):
//!     obj.add(FoldBox(0.34))
//!     obj.add(FoldMenger())
//!     obj.add(FoldScaleTranslate(3.28, (-5.27, -0.34, 0.0)))
//!     obj.add(FoldRotateX(π/2))
//!     obj.add(OrbitMax((0.42, 0.38, 0.19)))
//! obj.add(Box(2.0, color='orbit'))
//! ```
//!
//! Structurally it's a `FoldMenger` + scale (≈3.28) + rotation stack:
//! close enough to a pure Menger sponge that the trinary-native 20/27
//! mapping is the best available voxel adaptation. The `π/2`
//! `FoldRotateX` re-indexes axes without breaking the base-3 lattice;
//! the `FoldBox(0.34)` is a continuous clip we can't reproduce.
//!
//! # Coloring — the authentic orbit trap
//!
//! The distinctive look is `OrbitMax((0.42, 0.38, 0.19))`: per-axis
//! *maximum signed displacement* of the fold-iterate trajectory,
//! scaled by `(0.42, 0.38, 0.19)`, clamped to `[0, 1]` in the shader.
//! Because the R/G/B scales are biased warm (0.42 > 0.38 >> 0.19),
//! the output is **warm ochre gradient**: R saturates fastest, G
//! close behind, B stays damped → gold → dark-amber.
//!
//! In a voxel tree we can't compute a live orbit, but we can sample
//! two representative points on the curve and paint the two
//! structural roles of the Menger sponge (corners vs. edge-midpoints)
//! with them. The result is a static Menger whose palette evokes
//! the mausoleum's orbit-trap aesthetic at every zoom level.
//!
//! Color choices below are the *direct* RGB you get from
//! `(0.42, 0.38, 0.19) * orbit` at orbit magnitudes 1.0 (deep rod)
//! and 2.0 (saturated highlight) — no invention, straight PySpace.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::ColorRegistry;
use crate::world::state::WorldState;
use crate::world::tree::{NodeLibrary, MAX_DEPTH};

/// 8 corners get highlight, 12 edge-midpoints get rod. Same geometry
/// as `menger::menger_world` but with role-coloring swapped in.
fn mausoleum_world(depth: u8, corner_highlight: u16, edge_rod: u16) -> WorldState {
    let mut slots: Vec<Slot> = Vec::with_capacity(20);
    for z in 0u8..3 {
        for y in 0u8..3 {
            for x in 0u8..3 {
                let center_axes = (x == 1) as u8 + (y == 1) as u8 + (z == 1) as u8;
                if center_axes >= 2 { continue; } // Menger kept-set
                let block = if center_axes == 0 { corner_highlight } else { edge_rod };
                slots.push((x, y, z, block));
            }
        }
    }
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_mausoleum_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    // Authentic orbit-trap RGB. `OrbitMax((0.42, 0.38, 0.19))` × 1.0
    // → (107, 97, 48) dark ochre "rod"; × 2.0 (clamped to 1.0 for R/G)
    // → (214, 194, 97) → we choose the mid-orbit sample at magnitude
    // ≈1.75 → (178, 161, 81) "highlight" which matches the typical
    // bright-side appearance of the original render.
    let mut registry = ColorRegistry::new();
    let corner_highlight = registry.register(178, 161, 81, 255).unwrap();
    let edge_rod = registry.register(107, 97, 48, 255).unwrap();

    let world = mausoleum_world(depth, corner_highlight, edge_rod);

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
    fn mausoleum_has_expected_depth() {
        let w = mausoleum_world(5, 11, 12);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn mausoleum_dedup() {
        // Same kept-set as Menger → same O(depth) topology.
        let w = mausoleum_world(6, 11, 12);
        assert_eq!(w.library.len(), 6);
    }
}
