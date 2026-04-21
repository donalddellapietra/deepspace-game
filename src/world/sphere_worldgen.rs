//! World bootstrap for the cube-IS-a-sphere architecture.
//!
//! A "planet" is a uniform stone cube flagged `NodeKind::SphereBody`.
//! Storage dedups trivially: a cube of N layers = N + 2 unique library
//! entries (N uniform-stone subtrees + the SphereBody node + the
//! Cartesian wrapper). Zero voxel surface cells.
//!
//! Rendering is handled by the shader's ray-sphere analytic path —
//! when the DDA descends into a SphereBody cube, it runs a ray vs
//! the cube's inscribed sphere instead of traversing interior voxels.
//! Hits return with a radial world-space normal. See `march.wgsl`.
//!
//! The uniform-stone interior is kept (not collapsed to a single
//! `Child::Block`) so future edits (`break`, `place`) can carve into
//! Cartesian cells. The shader ignores the interior content until the
//! user actually digs.

use super::anchor::{Path, WorldPos};
use super::palette::block;
use super::state::WorldState;
use super::tree::{
    empty_children, uniform_children, Child, NodeKind, NodeLibrary, CENTER_SLOT,
};

/// Default SphereBody subtree depth. The cube is uniform stone at
/// every level so depth doesn't affect storage cost (O(layers)
/// unique nodes); it only sets the max depth available for future
/// editing.
pub const DEFAULT_SPHERE_LAYERS: u8 = 8;

/// Build a world containing a single sphere-body planet. The world
/// root is a Cartesian wrapper with one SphereBody subtree at the
/// center slot; every other slot is empty sky so the camera can
/// stand outside the cube and view the planet.
pub fn bootstrap_sphere_body_world(
    layers: u8,
) -> crate::world::bootstrap::WorldBootstrap {
    let world = build_sphere_body_world(layers);
    crate::world::bootstrap::WorldBootstrap {
        default_spawn_pos: default_spawn(),
        // World (2.5, 1.5, 1.5) is in the wrapper's slot 14 (= (2, 1, 1)),
        // an empty sky cell +x of the planet. Look toward −x to frame the
        // planet. Engine basis: yaw=0 gives fwd=(0,0,−1); positive yaw
        // rotates LEFT around +y; so fwd=(−1,0,0) requires yaw=π/2.
        default_spawn_yaw: std::f32::consts::FRAC_PI_2,
        default_spawn_pitch: 0.0,
        plain_layers: layers,
        color_registry: crate::world::palette::ColorRegistry::new(),
        world,
    }
}

fn default_spawn() -> WorldPos {
    // Wrapper root spans world [0, 3)³. Planet cube at slot 13 spans
    // world [1, 2)³. Camera at world (2.5, 1.5, 1.5): outside the
    // cube along +x, centered in y/z, looking back at the planet.
    // Anchor depth 1 puts the camera's cell at wrapper slot 14 — an
    // empty sky slot, not inside the SphereBody.
    WorldPos::from_frame_local(&Path::root(), [2.5, 1.5, 1.5], 1)
}

fn build_sphere_body_world(layers: u8) -> WorldState {
    assert!(layers > 0, "sphere world must have at least one layer");
    let mut lib = NodeLibrary::default();

    // Uniform-stone subtree of `layers` levels, built bottom-up. Each
    // `insert(uniform_children(Child::Node(prev)))` adds exactly ONE
    // library entry (content-addressed dedup — all 27 children point
    // at the same child NodeId, and that parent is unique per depth).
    let mut current: Child = Child::Block(block::STONE);
    for _ in 1..=layers {
        let node_id = lib.insert(uniform_children(current));
        current = Child::Node(node_id);
    }

    // Wrap the stone chain in a SphereBody-flagged node. All 27
    // children share the same deep-stone NodeId ⇒ adds exactly one
    // more library entry. The shader reads the `node_kinds[child_bfs]`
    // entry when deciding whether to descend or to do the analytic
    // ray-sphere test.
    let sphere_body = lib.insert_with_kind(
        uniform_children(current),
        NodeKind::SphereBody,
    );

    // Cartesian wrapper around the SphereBody. Center slot holds the
    // planet; the other 26 slots stay empty so the camera has sky to
    // stand in without ending up under a SphereBody ancestor.
    let mut root_children = empty_children();
    root_children[CENTER_SLOT] = Child::Node(sphere_body);
    let root = lib.insert(root_children);
    lib.ref_inc(root);

    let world = WorldState { root, library: lib };
    eprintln!(
        "Sphere body world: layers={}, library_entries={}, depth={}",
        layers,
        world.library.len(),
        world.tree_depth(),
    );
    world
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_root_wraps_sphere_body_at_center_slot() {
        let bs = bootstrap_sphere_body_world(1);
        let root = bs.world.library.get(bs.world.root).expect("root");
        assert_eq!(root.kind, NodeKind::Cartesian);
        match root.children[CENTER_SLOT] {
            Child::Node(sphere_id) => {
                let sphere = bs.world.library.get(sphere_id).expect("sphere body");
                assert_eq!(sphere.kind, NodeKind::SphereBody);
            }
            other => panic!("center slot = {:?}, expected SphereBody node", other),
        }
        for (i, c) in root.children.iter().enumerate() {
            if i == CENTER_SLOT {
                continue;
            }
            assert!(matches!(c, Child::Empty), "slot {i} = {:?}", c);
        }
    }

    #[test]
    fn storage_is_linear_in_depth() {
        // The whole point of the cube-is-a-sphere design: library size
        // scales linearly with `layers`, not with a surface voxel
        // count. 40 layers should produce ~40 entries.
        let bs = bootstrap_sphere_body_world(40);
        let lib_size = bs.world.library.len();
        // Expected: 40 uniform-stone nodes + SphereBody node +
        // Cartesian wrapper = 42. Allow a handful of headroom for any
        // intermediate bookkeeping nodes, but stay well under the
        // millions that a voxelized-ball approach would produce.
        assert!(
            lib_size < 64,
            "layers=40 produced {lib_size} library entries — storage not dedupping cleanly",
        );
    }

    #[test]
    fn sphere_body_subtree_is_uniform_stone() {
        let bs = bootstrap_sphere_body_world(3);
        let root = bs.world.library.get(bs.world.root).unwrap();
        let Child::Node(sphere_id) = root.children[CENTER_SLOT] else {
            panic!();
        };
        let sphere = bs.world.library.get(sphere_id).unwrap();
        let first = sphere.children[0];
        for c in sphere.children.iter() {
            assert_eq!(*c, first, "SphereBody children must all be the same stone subtree");
        }
    }
}
