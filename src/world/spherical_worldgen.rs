//! Declarative setup for spherical-planet worldgen.
//!
//! This module is a thin, data-first wrapper around
//! [`super::cubesphere::generate_spherical_planet`]. Callers construct
//! (or start from [`demo_planet`]) a [`PlanetSetup`] value and hand it
//! to [`build`], which materializes the 6 face subtrees into the
//! provided [`NodeLibrary`] and returns a [`SphericalPlanet`] handle.
//!
//! The split exists so `main.rs` (and tests, and future editors) can
//! describe planets as plain data instead of open-coding the setup
//! every time one is needed.
//!
//! Tuning the starter scene is done by editing [`demo_planet`] — all
//! fields are `pub` so callers can also clone-and-modify.
//!
//! # Example
//! ```ignore
//! use deepspace_game::world::{spherical_worldgen, tree::NodeLibrary};
//! let mut lib = NodeLibrary::default();
//! let setup = spherical_worldgen::demo_planet();
//! let planet = spherical_worldgen::build(&mut lib, &setup);
//! ```

use super::cubesphere::{generate_spherical_planet, SphericalPlanet};
use super::palette::block;
use super::sdf::{Planet, Vec3};
use super::tree::NodeLibrary;

/// Declarative description of a planet to build. All units are in
/// world space. `depth` is the face-subtree recursion depth.
#[derive(Clone, Debug)]
pub struct PlanetSetup {
    pub center: Vec3,
    pub inner_r: f32,
    pub outer_r: f32,
    pub depth: u32,
    pub sdf: Planet,
}

/// The demo/starter planet used by the default world. Edit this
/// function to tune the starting scene.
pub fn demo_planet() -> PlanetSetup {
    // Centered at the world origin (root cell midpoint) so the
    // shell + gravity influence (2 × outer_r) fits inside the root
    // cell `[0, WORLD_SIZE)^3` and spawn / fly-space don't need
    // clamping. With WORLD_SIZE = 3.0, center = 1.5 and
    // influence_radius = 1.04 gives a hard floor/ceiling at
    // y ∈ [0.46, 2.54] for gravity — well inside root.
    let center: Vec3 = [1.5, 1.5, 1.5];
    let inner_r = 0.12_f32;
    let outer_r = 0.52_f32;
    PlanetSetup {
        center,
        inner_r,
        outer_r,
        depth: 20,
        sdf: Planet {
            center,
            radius: 0.32,
            noise_scale: 0.015,
            noise_freq: 8.0,
            noise_seed: 2024,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        },
    }
}

/// Generate the planet's 6 face subtrees into `lib` and return a
/// [`SphericalPlanet`] handle.
pub fn build(lib: &mut NodeLibrary, setup: &PlanetSetup) -> SphericalPlanet {
    generate_spherical_planet(
        lib,
        setup.center,
        setup.inner_r,
        setup.outer_r,
        setup.depth,
        &setup.sdf,
    )
}

// ───────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_planet_has_valid_parameters() {
        let s = demo_planet();
        // Radii form a valid shell.
        assert!(s.inner_r > 0.0);
        assert!(s.outer_r > s.inner_r);
        // Planet surface radius lies inside the shell.
        assert!(s.sdf.radius > s.inner_r);
        assert!(s.sdf.radius < s.outer_r);
        // SDF is anchored to the same center as the shell.
        assert_eq!(s.sdf.center, s.center);
        // Depth matches the original main.rs setup (extraction, not
        // redesign — bump deliberately if tuning the demo).
        assert_eq!(s.depth, 20);
        // Influence radius extends beyond the shell for gravity in space.
        assert!(s.sdf.influence_radius >= s.outer_r);
        // Blocks are sane.
        assert_eq!(s.sdf.surface_block, block::GRASS);
        assert_eq!(s.sdf.core_block, block::STONE);
    }

    #[test]
    fn build_produces_six_face_roots_in_library() {
        let mut lib = NodeLibrary::default();
        let setup = demo_planet();
        let planet = build(&mut lib, &setup);
        assert_eq!(planet.face_roots.len(), 6);
        for &id in &planet.face_roots {
            assert!(
                lib.get(id).is_some(),
                "face root {id} missing from NodeLibrary",
            );
        }
        assert_eq!(planet.center, setup.center);
        assert_eq!(planet.inner_r, setup.inner_r);
        assert_eq!(planet.outer_r, setup.outer_r);
        assert_eq!(planet.depth, setup.depth);
    }
}
