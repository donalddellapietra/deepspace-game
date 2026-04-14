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

use super::coords::Path;
use super::cubesphere::{generate_spherical_planet, SphericalPlanet};
use super::palette::block;
use super::sdf::{Planet, Vec3};
use super::state::WorldState;
use super::tree::{
    empty_children, slot_index, Child, NodeLibrary,
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

/// The demo/starter planet used by the default world. Edit this
/// function to tune the starting scene.
pub fn demo_planet() -> PlanetSetup {
    // Anchor slot (1, 2, 1) in the `[0, 3)` root frame spans
    // [1, 2) × [2, 3) × [1, 2) — its cell center is (1.5, 2.5, 1.5).
    // The body's local (0.5, 0.5, 0.5) maps to that world point, so
    // both the SDF sampling (during worldgen) and the shader render
    // must agree on this center. `inner_r` / `outer_r` are in the
    // body cell's local `[0, 1)` frame; per decisions §3 they must
    // satisfy `0 < inner_r < outer_r ≤ 0.5` so the shell fits
    // strictly inside one parent cell.
    let center: Vec3 = [1.5, 2.5, 1.5];
    let inner_r = 0.12_f32;
    let outer_r = 0.48_f32;
    let surface_r = 0.5 * (inner_r + outer_r);
    PlanetSetup {
        center,
        inner_r,
        outer_r,
        depth: 20,
        sdf: Planet {
            center,
            radius: surface_r,
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

/// One-shot scene bootstrap: build the planet's body node (with its
/// 6 face subtrees) and anchor it at slot (1, 2, 1) of a fresh
/// world root. Returns a [`Scene`] carrying the world, the body's
/// anchor [`Path`], and the [`SphericalPlanet`] handle with its
/// cached shell geometry.
///
/// The anchor slot is chosen so the body cell spans `[1, 2) × [2, 3)
/// × [1, 2)` in the root's `[0, 3)³` frame — the body's local center
/// at world `(1.5, 2.5, 1.5)`, with the full `outer_r ≤ 0.5` shell
/// fitting inside that single depth-1 cell.
///
/// The other 26 root-child slots hold uniform-empty subtrees of the
/// same depth as the body, so the tree is uniformly `1 +
/// face_subtree_depth` levels deep.
pub fn build(setup: &PlanetSetup) -> Scene {
    let mut lib = NodeLibrary::default();

    // 1. Build the body (body_node + 6 face subtrees) in `lib`.
    let planet = generate_spherical_planet(
        &mut lib,
        setup.center,
        setup.inner_r,
        setup.outer_r,
        setup.depth,
        &setup.sdf,
    );

    // 2. Build a uniform-empty subtree at the same depth as the
    //    body node, so all root children are equi-depth.
    let empty_sub = build_uniform_empty(&mut lib, setup.depth);

    // 3. Assemble the root: 27 children, the body at slot (1, 2, 1),
    //    the rest uniform empty. Slot ordering is
    //    `slot_index(x, y, z) = z*9 + y*3 + x`, so (1, 2, 1) = 16.
    let body_slot = slot_index(1, 2, 1);
    let mut root_children = empty_children();
    for i in 0..27 {
        root_children[i] = if i == body_slot {
            Child::Node(planet.body_node)
        } else {
            Child::Node(empty_sub)
        };
    }
    let root = lib.insert(root_children);
    lib.ref_inc(root);

    let mut body_anchor = Path::root();
    body_anchor.push(body_slot as u8);

    Scene {
        world: WorldState { root, library: lib },
        body_anchor,
        planet,
    }
}

/// Returned by [`build`]: the constructed world tree, the body's
/// anchor [`Path`] within it, and the [`SphericalPlanet`] cache.
pub struct Scene {
    pub world: WorldState,
    pub body_anchor: Path,
    pub planet: SphericalPlanet,
}

fn build_uniform_empty(lib: &mut NodeLibrary, depth: u32) -> super::tree::NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(super::tree::uniform_children(Child::Node(id)));
    }
    id
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
    fn build_scene_embeds_body_at_anchor_slot() {
        let setup = demo_planet();
        let scene = build(&setup);
        // Body anchor is at depth 1, slot_index(1, 2, 1).
        assert_eq!(scene.body_anchor.depth(), 1);
        assert_eq!(
            scene.body_anchor.slots()[0],
            super::super::tree::slot_index(1, 2, 1) as u8,
        );
        // Body node is reachable from world root via the anchor.
        let root = scene.world.library.get(scene.world.root).unwrap();
        match root.children[scene.body_anchor.slots()[0] as usize] {
            Child::Node(id) => assert_eq!(id, scene.planet.body_node),
            other => panic!("expected body at anchor, got {:?}", other),
        }
        // Face roots are still reachable.
        // Face roots are reachable through the body node's children.
        use super::super::cubesphere::{face_root_of, Face};
        for &face in &Face::ALL {
            let id = face_root_of(&scene.world.library, scene.planet.body_node, face)
                .expect("face root must resolve on a freshly built scene");
            assert!(scene.world.library.get(id).is_some());
        }
    }
}
