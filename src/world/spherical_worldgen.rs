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
use super::tree::{
    empty_children, slot_index, Child, NodeId, NodeKind, NodeLibrary,
};

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

/// The demo/starter planet. Center sits at the middle of the root
/// cell `[0, 3)³` so the planet's body-cell is the root-center slot
/// (path `[13]`), a 1×1×1 world cube at `[1, 2)³`. That keeps the
/// sphere strictly inside one parent cell, a hard requirement of the
/// `CubedSphereBody` node kind (doc §3).
pub fn demo_planet() -> PlanetSetup {
    let center: Vec3 = [1.5, 1.5, 1.5];
    let inner_r = 0.12_f32;
    // Body cell extent is 1.0 (root's child at slot 13), so world
    // outer_r == body-local outer_r. The hard cap is 0.5 (§3) so the
    // sphere doesn't poke past the body cell; pick 0.5 exactly.
    let outer_r = 0.5_f32;
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

/// Result of building a planet: the `SphericalPlanet` handle (center,
/// radii, face-root ids — kept for the legacy uniform upload path)
/// plus the body node that owns the face subtrees in the tree. Step
/// C deletes the handle once the shader dispatches on `NodeKind`.
pub struct BuiltPlanet {
    pub planet: SphericalPlanet,
    pub body_node: NodeId,
    pub body_path: Vec<u8>,
}

/// Generate the planet's 6 face subtrees into `lib`, wrap them in a
/// `CubedSphereBody` node with the face subtrees attached at the
/// six face-center child slots, and return the body. The caller is
/// expected to install the body at `root_children[slot_index(1,1,1)]`
/// and reinsert the tree root.
pub fn build(lib: &mut NodeLibrary, setup: &PlanetSetup) -> BuiltPlanet {
    let planet = generate_spherical_planet(
        lib,
        setup.center,
        setup.inner_r,
        setup.outer_r,
        setup.depth,
        &setup.sdf,
    );

    // Interior filler: a uniform subtree of core blocks of depth
    // matching the face subtrees, so the body's inner cavity looks
    // right when the player tunnels through.
    let filler_child = lib.build_uniform_subtree(setup.sdf.core_block, setup.depth);

    // Assemble the body's 27 children. Six face-center slots hold the
    // face subtrees (order: +X, −X, +Y, −Y, +Z, −Z). Center slot is
    // the interior filler; the other 20 slots are Empty.
    let mut body_children = empty_children();
    body_children[slot_index(2, 1, 1)] = Child::Node(planet.face_roots[0]); // +X
    body_children[slot_index(0, 1, 1)] = Child::Node(planet.face_roots[1]); // −X
    body_children[slot_index(1, 2, 1)] = Child::Node(planet.face_roots[2]); // +Y
    body_children[slot_index(1, 0, 1)] = Child::Node(planet.face_roots[3]); // −Y
    body_children[slot_index(1, 1, 2)] = Child::Node(planet.face_roots[4]); // +Z
    body_children[slot_index(1, 1, 0)] = Child::Node(planet.face_roots[5]); // −Z
    body_children[slot_index(1, 1, 1)] = filler_child;

    // Body radii in body-cell local frame `[0, 1)³`. Cell extent is
    // 1.0 (root's slot 13) so local == world for this demo.
    let body_node = lib.insert_with_kind(
        body_children,
        NodeKind::CubedSphereBody {
            inner_r: setup.inner_r,
            outer_r: setup.outer_r,
        },
    );

    BuiltPlanet {
        planet,
        body_node,
        body_path: vec![slot_index(1, 1, 1) as u8],
    }
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
        let built = build(&mut lib, &setup);
        let planet = &built.planet;
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

    #[test]
    fn build_produces_body_node_with_cubed_sphere_body_kind() {
        let mut lib = NodeLibrary::default();
        let setup = demo_planet();
        let built = build(&mut lib, &setup);
        let body = lib.get(built.body_node).expect("body node present");
        match body.kind {
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                assert!((inner_r - setup.inner_r).abs() < 1e-6);
                assert!((outer_r - setup.outer_r).abs() < 1e-6);
            }
            _ => panic!("body node has non-body kind: {:?}", body.kind),
        }
        // 6 face-center slots should hold the face subtrees.
        let face_slots = [
            (slot_index(2, 1, 1), 0), // +X
            (slot_index(0, 1, 1), 1), // −X
            (slot_index(1, 2, 1), 2), // +Y
            (slot_index(1, 0, 1), 3), // −Y
            (slot_index(1, 1, 2), 4), // +Z
            (slot_index(1, 1, 0), 5), // −Z
        ];
        for (slot, face_idx) in face_slots {
            match body.children[slot] {
                Child::Node(id) => assert_eq!(id, built.planet.face_roots[face_idx]),
                other => panic!("slot {} not a Node: {:?}", slot, other),
            }
        }
        // Path is just the root-center slot.
        assert_eq!(built.body_path, vec![slot_index(1, 1, 1) as u8]);
    }
}
