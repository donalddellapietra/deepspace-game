//! Rhombic-dodecahedron primitive: 12 `TangentBlock` cubes at the
//! 12 edge-adjacent positions of root, each with a rotation that
//! orients its storage `+Y` axis toward the corresponding world
//! edge direction. Plus an unrotated centre cube.
//!
//! The rhombic dodecahedron is the 3D analog of a hexagon: it tiles
//! space (it's the Voronoi cell of the FCC lattice), and its 12
//! face-normals are the 12 edge-directions of a cube — the 12
//! nearest-neighbour directions in close-packing.
//!
//! Layout in the 3×3×3 root grid (slot indices `slot_index(x,y,z) = z*9 + y*3 + x`):
//!
//! ```text
//! XY-plane edges (z = 1):                YZ-plane edges (x = 1):
//!   (0, 0, 1) =  9  →  (-X, -Y, 0)         (1, 0, 0) =  1  →  (0, -Y, -Z)
//!   (2, 0, 1) = 11  →  (+X, -Y, 0)         (1, 2, 0) =  7  →  (0, +Y, -Z)
//!   (0, 2, 1) = 15  →  (-X, +Y, 0)         (1, 0, 2) = 19  →  (0, -Y, +Z)
//!   (2, 2, 1) = 17  →  (+X, +Y, 0)         (1, 2, 2) = 25  →  (0, +Y, +Z)
//!
//! XZ-plane edges (y = 1):
//!   (0, 1, 0) =  3  →  (-X, 0, -Z)
//!   (2, 1, 0) =  5  →  (+X, 0, -Z)
//!   (0, 1, 2) = 21  →  (-X, 0, +Z)
//!   (2, 1, 2) = 23  →  (+X, 0, +Z)
//!
//! Centre:
//!   (1, 1, 1) = 13  →  Cartesian uniform stone (unrotated)
//! ```
//!
//! Each rotation is a 45°-style rotation that maps storage `+Y` to
//! the outward edge direction. Without `tb_scale` (the inscribed-cube
//! shrink), the rotated content's corners poke beyond the slot
//! boundary by a factor of `√2/2 - 1/2 ≈ 0.207` per axis — a known
//! visual artifact, accepted while the runtime architecture commits
//! to `tb_scale = 1`. The architecture itself (rotation-aware
//! `renormalize_world` / `zoom_*_in_world`) handles arbitrary
//! rotations correctly regardless of the scale.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, matmul3x3, rotation_x, rotation_y, rotation_z, slot_index, uniform_children,
    Child, NodeKind, NodeLibrary,
};

/// Depth of each per-cube subtree — small for navigation testing.
pub const CUBE_SUBTREE_DEPTH: u8 = 6;

/// Empty Cartesian wrappings around the cluster. Each layer
/// triples the world's linear extent relative to the cluster, so
/// the camera has buffer space to fly around without hitting the
/// world boundary. With `BUFFER_LAYERS = 2`, the cluster occupies
/// `1/9 × 1/9 × 1/9` of the world cube — plenty of clear airspace.
pub const BUFFER_LAYERS: u8 = 2;

fn build_uniform_cartesian_subtree(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
) -> Child {
    if depth == 0 {
        return Child::Block(block_id);
    }
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert(uniform_children(inner)))
}

fn build_tangent_block_subtree(
    library: &mut NodeLibrary,
    block_id: u16,
    depth: u8,
    rotation: [[f32; 3]; 3],
) -> Child {
    assert!(depth >= 1, "TangentBlock subtree depth must be >= 1");
    let inner = build_uniform_cartesian_subtree(library, block_id, depth - 1);
    Child::Node(library.insert_with_kind(
        uniform_children(inner),
        NodeKind::TangentBlock { rotation },
    ))
}

/// Rotations that map storage `+Y` to each of the 12 cube-edge
/// directions.
///
/// XY-plane edges (`z = 0`, target `(±X, ±Y, 0) / √2`): single
/// `rotation_z(α)` since the rotation stays in the XY plane.
/// `rotation_z(α) · (0, 1, 0) = (-sin α, cos α, 0)`, so set
/// `α` such that `(-sin α, cos α) = (±1, ±1) / √2`.
///
/// YZ-plane edges (`x = 0`, target `(0, ±Y, ±Z) / √2`): single
/// `rotation_x(α)`.  `rotation_x(α) · (0, 1, 0) = (0, cos α, sin α)`.
///
/// XZ-plane edges (`y = 0`, target `(±X, 0, ±Z) / √2`): two-step.
/// First `rotation_x(π/2)` to send `+Y` → `+Z`, then `rotation_y(α)`
/// to swing `+Z` → `(sin α, 0, cos α)` in the XZ plane. Combined:
/// `rotation_y(α) · rotation_x(π/2)`.
fn edge_rotation_xy(sx: f32, sy: f32) -> [[f32; 3]; 3] {
    // `rotation_z(α) · (0, 1, 0) = (-sin α, cos α, 0)`. We want
    // (sx/√2, sy/√2, 0) → so `-sin α = sx/√2`, `cos α = sy/√2`.
    let alpha = (-sx).atan2(sy) - 0.0; // atan2(-sx, sy) gives the right α
    let _ = alpha;
    let alpha = f32::atan2(-sx, sy);
    rotation_z(alpha)
}

fn edge_rotation_yz(sy: f32, sz: f32) -> [[f32; 3]; 3] {
    // `rotation_x(α) · (0, 1, 0) = (0, cos α, sin α)`. We want
    // (0, sy/√2, sz/√2) → so `cos α = sy/√2`, `sin α = sz/√2`.
    let alpha = f32::atan2(sz, sy);
    rotation_x(alpha)
}

fn edge_rotation_xz(sx: f32, sz: f32) -> [[f32; 3]; 3] {
    // After `rotation_x(π/2)`, storage `+Y` lives at `+Z`. Then
    // `rotation_y(α) · (0, 0, 1) = (sin α, 0, cos α)`. We want
    // (sx/√2, 0, sz/√2) → `sin α = sx/√2`, `cos α = sz/√2`.
    let alpha = f32::atan2(sx, sz);
    matmul3x3(&rotation_y(alpha), &rotation_x(std::f32::consts::FRAC_PI_2))
}

/// Build the 27-slot cluster as a Cartesian node: centre + 12 edge
/// `TangentBlock`s + 14 empty slots. The cluster is a single
/// "world cell" that the bootstrap embeds inside `BUFFER_LAYERS`
/// empty Cartesian wrappings to give the camera surrounding airspace.
fn build_cluster(library: &mut NodeLibrary) -> Child {
    let centre = build_uniform_cartesian_subtree(
        library, block::STONE, CUBE_SUBTREE_DEPTH,
    );
    let edge_specs: [((usize, usize, usize), [[f32; 3]; 3]); 12] = [
        // XY-plane edges (z = 1):
        ((0, 0, 1), edge_rotation_xy(-1.0, -1.0)),
        ((2, 0, 1), edge_rotation_xy(1.0, -1.0)),
        ((0, 2, 1), edge_rotation_xy(-1.0, 1.0)),
        ((2, 2, 1), edge_rotation_xy(1.0, 1.0)),
        // YZ-plane edges (x = 1):
        ((1, 0, 0), edge_rotation_yz(-1.0, -1.0)),
        ((1, 2, 0), edge_rotation_yz(1.0, -1.0)),
        ((1, 0, 2), edge_rotation_yz(-1.0, 1.0)),
        ((1, 2, 2), edge_rotation_yz(1.0, 1.0)),
        // XZ-plane edges (y = 1):
        ((0, 1, 0), edge_rotation_xz(-1.0, -1.0)),
        ((2, 1, 0), edge_rotation_xz(1.0, -1.0)),
        ((0, 1, 2), edge_rotation_xz(-1.0, 1.0)),
        ((2, 1, 2), edge_rotation_xz(1.0, 1.0)),
    ];
    let mut children = empty_children();
    children[slot_index(1, 1, 1)] = centre;
    for &((sx, sy, sz), rotation) in &edge_specs {
        children[slot_index(sx, sy, sz)] = build_tangent_block_subtree(
            library, block::STONE, CUBE_SUBTREE_DEPTH, rotation,
        );
    }
    Child::Node(library.insert_with_kind(children, NodeKind::Cartesian))
}

pub fn rhombic_dodecahedron_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // The cluster as a single "world cell" Node.
    let mut cur = build_cluster(&mut library);

    // Wrap in `BUFFER_LAYERS` empty Cartesian layers, each placing
    // the previous wrap at slot 13 (centre) so the cluster sits in
    // the dead centre of the world. Every wrap triples the world
    // size relative to the cluster.
    for _ in 0..BUFFER_LAYERS {
        let mut children = empty_children();
        children[slot_index(1, 1, 1)] = cur;
        cur = Child::Node(library.insert_with_kind(children, NodeKind::Cartesian));
    }

    let root = match cur {
        Child::Node(id) => id,
        _ => unreachable!("cluster wrap always produces a Node"),
    };
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "rhombic_dodecahedron_test world: tree_depth={}, library_entries={}, buffer_layers={}",
        world.tree_depth(),
        world.library.len(),
        BUFFER_LAYERS,
    );
    world
}

/// Spawn just outside the cluster, in the empty cell adjacent to
/// the cluster's `(2, 2, 2)` corner. With `BUFFER_LAYERS = 2`, the
/// anchor reads `[13, 13, 26]` — slot 13 of root, slot 13 of buffer
/// layer 1, slot 26 (= the empty `(2,2,2)` corner of the cluster).
/// Camera's world position is just outside the cluster's `(+X+Y+Z)`
/// corner, with diagonal yaw + pitch aimed back at the centre.
pub fn rhombic_dodecahedron_test_spawn() -> WorldPos {
    let mut anchor = crate::world::anchor::Path::root();
    for _ in 0..BUFFER_LAYERS {
        anchor.push(slot_index(1, 1, 1) as u8);
    }
    anchor.push(slot_index(2, 2, 2) as u8);
    WorldPos::new(anchor, [0.5, 0.5, 0.5])
}

pub(super) fn bootstrap_rhombic_dodecahedron_test_world() -> WorldBootstrap {
    let world = rhombic_dodecahedron_test_world();
    let spawn_pos = rhombic_dodecahedron_test_spawn();
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // Camera spawns at the (+X+Y+Z) corner just outside the
        // cluster; aim back toward the centre with diagonal yaw +
        // pitch so the cluster reads as a 3D shape rather than a
        // single face. `-3π/4` yaw + `-π/4` pitch points the fwd
        // axis at roughly `(-X, -Y, -Z)`.
        default_spawn_yaw: -3.0 * std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: -std::f32::consts::FRAC_PI_4,
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity-check each of the 12 rotations: it must map storage
    /// `+Y` to its target edge direction.
    #[test]
    fn each_edge_rotation_aligns_storage_y_to_target() {
        // Apply `r · (0, 1, 0)` = the second column of `r` (since
        // column-major `r[col][row]` and the input has only the
        // y-component non-zero, contributing column 1).
        fn apply_to_y(r: &[[f32; 3]; 3]) -> [f32; 3] {
            [r[1][0], r[1][1], r[1][2]]
        }
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let cases: [(([f32; 3]), [[f32; 3]; 3]); 12] = [
            ([-s, -s, 0.0], edge_rotation_xy(-1.0, -1.0)),
            ([ s, -s, 0.0], edge_rotation_xy( 1.0, -1.0)),
            ([-s,  s, 0.0], edge_rotation_xy(-1.0,  1.0)),
            ([ s,  s, 0.0], edge_rotation_xy( 1.0,  1.0)),
            ([0.0, -s, -s], edge_rotation_yz(-1.0, -1.0)),
            ([0.0,  s, -s], edge_rotation_yz( 1.0, -1.0)),
            ([0.0, -s,  s], edge_rotation_yz(-1.0,  1.0)),
            ([0.0,  s,  s], edge_rotation_yz( 1.0,  1.0)),
            ([-s, 0.0, -s], edge_rotation_xz(-1.0, -1.0)),
            ([ s, 0.0, -s], edge_rotation_xz( 1.0, -1.0)),
            ([-s, 0.0,  s], edge_rotation_xz(-1.0,  1.0)),
            ([ s, 0.0,  s], edge_rotation_xz( 1.0,  1.0)),
        ];
        for (target, r) in cases {
            let got = apply_to_y(&r);
            for i in 0..3 {
                assert!((got[i] - target[i]).abs() < 1e-5,
                    "rotation maps +Y to {got:?}, want {target:?}");
            }
        }
    }

    #[test]
    fn cluster_is_nested_inside_buffer_layers() {
        let world = rhombic_dodecahedron_test_world();
        // Walk `BUFFER_LAYERS` empty Cartesian wrappings to reach
        // the cluster node. Each wrap is Cartesian with only slot
        // 13 occupied.
        let mut node_id = world.root;
        for _ in 0..BUFFER_LAYERS {
            let n = world.library.get(node_id).expect("buffer node exists");
            assert_eq!(n.kind, NodeKind::Cartesian, "buffer wrap must be Cartesian");
            // Only slot 13 is non-empty in buffer wrappings.
            for slot in 0..27 {
                let occupied = matches!(n.children[slot], Child::Node(_));
                assert_eq!(occupied, slot == slot_index(1, 1, 1),
                    "buffer wrap slot {slot} occupancy != (slot == 13)");
            }
            match n.children[slot_index(1, 1, 1)] {
                Child::Node(c) => node_id = c,
                other => panic!("buffer wrap centre must be a Node, got {other:?}"),
            }
        }
        let cluster = world.library.get(node_id).expect("cluster node exists");
        assert_eq!(cluster.kind, NodeKind::Cartesian);

        // Centre = Cartesian.
        match cluster.children[slot_index(1, 1, 1)] {
            Child::Node(id) => {
                let n = world.library.get(id).expect("centre subtree");
                assert_eq!(n.kind, NodeKind::Cartesian, "centre must be unrotated");
            }
            other => panic!("expected centre Node, got {other:?}"),
        }
        // 12 edge slots = TBs.
        let edge_slots: [(usize, usize, usize); 12] = [
            (0, 0, 1), (2, 0, 1), (0, 2, 1), (2, 2, 1),
            (1, 0, 0), (1, 2, 0), (1, 0, 2), (1, 2, 2),
            (0, 1, 0), (2, 1, 0), (0, 1, 2), (2, 1, 2),
        ];
        for (sx, sy, sz) in edge_slots {
            match cluster.children[slot_index(sx, sy, sz)] {
                Child::Node(id) => {
                    let n = world.library.get(id).expect("edge subtree");
                    assert!(n.kind.is_tangent_block(),
                        "edge slot ({sx},{sy},{sz}) must be a TangentBlock");
                }
                other => panic!("expected edge Node at ({sx},{sy},{sz}), got {other:?}"),
            }
        }
        // Other 14 cluster slots = empty (8 corners + 6 face-centres).
        let mut occupied = std::collections::HashSet::new();
        occupied.insert(slot_index(1, 1, 1));
        for (sx, sy, sz) in edge_slots {
            occupied.insert(slot_index(sx, sy, sz));
        }
        for slot in 0..27 {
            if occupied.contains(&slot) { continue; }
            assert_eq!(cluster.children[slot], Child::Empty,
                "cluster slot {slot} (not in dodecahedron) must be empty");
        }
    }

    /// `from_world_xyz` → `in_frame_rot` round-trip across all 12
    /// face-cubes proves the storage-frame slot derivation and the
    /// world-position derivation are exact inverses for every
    /// distinct rotation in the cluster.
    #[test]
    fn from_world_xyz_round_trips_through_every_edge_cube() {
        use crate::world::anchor::Path;
        let world = rhombic_dodecahedron_test_world();
        // With `BUFFER_LAYERS` empty Cartesian wraps around the
        // cluster, the cluster lives in world cube `[c-h, c+h]³`
        // where `c = 1.5` and `h = 0.5 / 3^BUFFER_LAYERS`. Each of
        // the cluster's 27 cells has world size `2h/3`. Probe each
        // cube near its centre (asymmetric within the cell to
        // exercise non-trivial offsets).
        let h = 0.5_f32 / 3.0_f32.powi(BUFFER_LAYERS as i32);
        let cluster_lo = 1.5 - h;
        let cell = 2.0 * h / 3.0; // cluster cell size in world units
        let centre_of = |sx: i32, sy: i32, sz: i32| -> [f32; 3] {
            [
                cluster_lo + (sx as f32 + 0.5) * cell,
                cluster_lo + (sy as f32 + 0.5) * cell,
                cluster_lo + (sz as f32 + 0.5) * cell,
            ]
        };
        let probes: [[f32; 3]; 13] = [
            centre_of(1, 1, 1),  // centre
            // XY-plane edges:
            centre_of(0, 0, 1),
            centre_of(2, 0, 1),
            centre_of(0, 2, 1),
            centre_of(2, 2, 1),
            // YZ-plane edges:
            centre_of(1, 0, 0),
            centre_of(1, 2, 0),
            centre_of(1, 0, 2),
            centre_of(1, 2, 2),
            // XZ-plane edges:
            centre_of(0, 1, 0),
            centre_of(2, 1, 0),
            centre_of(0, 1, 2),
            centre_of(2, 1, 2),
        ];
        // Anchor depth must reach the cluster level (= one slot
        // past the buffer wraps).
        let target_depth = BUFFER_LAYERS + 1;
        for &xyz in &probes {
            let pos = WorldPos::from_world_xyz(
                xyz, target_depth, &world.library, world.root,
            );
            let recovered = pos.in_frame_rot(
                &world.library, world.root, &Path::root(),
            );
            for i in 0..3 {
                assert!((recovered[i] - xyz[i]).abs() < 1e-4,
                    "round-trip drift at probe {xyz:?} axis {i}: got {} (anchor {:?}, offset {:?})",
                    recovered[i], pos.anchor.as_slice(), pos.offset);
            }
        }
    }
}
