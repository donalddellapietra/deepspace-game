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

pub fn rhombic_dodecahedron_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // Centre: plain Cartesian uniform stone.
    let centre = build_uniform_cartesian_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH,
    );

    // 12 edge-adjacent slots with their target outward edge
    // directions (sx, sy, sz) — only two of the three components
    // are non-zero per edge. Build a TB subtree per slot with the
    // appropriate rotation.
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

    let mut root_children = empty_children();
    root_children[slot_index(1, 1, 1)] = centre;
    for &((sx, sy, sz), rotation) in &edge_specs {
        root_children[slot_index(sx, sy, sz)] = build_tangent_block_subtree(
            &mut library, block::STONE, CUBE_SUBTREE_DEPTH, rotation,
        );
    }
    let root = library.insert_with_kind(root_children, NodeKind::Cartesian);
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "rhombic_dodecahedron_test world: tree_depth={}, library_entries={}",
        world.tree_depth(),
        world.library.len(),
    );
    world
}

/// Spawn at a vacant root corner (slot 0 = (0, 0, 0)) — the eight
/// root corners are not part of the dodecahedron and give a clean
/// outside-the-shape view. Diagonal yaw + pitch makes the cluster
/// read as a 3D shape.
pub fn rhombic_dodecahedron_test_spawn() -> WorldPos {
    WorldPos::uniform_column(slot_index(0, 0, 0) as u8, 1, [0.5, 0.5, 0.5])
}

pub(super) fn bootstrap_rhombic_dodecahedron_test_world() -> WorldBootstrap {
    let world = rhombic_dodecahedron_test_world();
    let spawn_pos = rhombic_dodecahedron_test_spawn();
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // Aim the camera diagonally toward the centre cluster from
        // the (0,0,0) corner: yaw = +π/4 (look toward +X+Z), pitch
        // = +π/4 (look up).
        default_spawn_yaw: std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: std::f32::consts::FRAC_PI_4,
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
    fn root_layout_has_centre_plus_12_edge_tbs() {
        let world = rhombic_dodecahedron_test_world();
        let root_node = world.library.get(world.root).expect("root exists");
        // Centre = Cartesian.
        match root_node.children[slot_index(1, 1, 1)] {
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
            match root_node.children[slot_index(sx, sy, sz)] {
                Child::Node(id) => {
                    let n = world.library.get(id).expect("edge subtree");
                    assert!(n.kind.is_tangent_block(),
                        "edge slot ({sx},{sy},{sz}) must be a TangentBlock");
                }
                other => panic!("expected edge Node at ({sx},{sy},{sz}), got {other:?}"),
            }
        }
        // Other 14 slots = empty (8 corners + 6 face-centres).
        let mut occupied = std::collections::HashSet::new();
        occupied.insert(slot_index(1, 1, 1));
        for (sx, sy, sz) in edge_slots {
            occupied.insert(slot_index(sx, sy, sz));
        }
        for slot in 0..27 {
            if occupied.contains(&slot) { continue; }
            assert_eq!(root_node.children[slot], Child::Empty,
                "slot {slot} (not in dodecahedron) must be empty");
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
        // Pick an asymmetric probe inside each edge cube. Choose
        // points near the cube's centre to stay within the slot's
        // [s, s+1] world extent (avoiding the corner-poke band).
        let probes: [[f32; 3]; 13] = [
            [1.5, 1.5, 1.5],   // centre
            // XY-plane edges:
            [0.5, 0.5, 1.5],   // ( 0, 0, 1)
            [2.5, 0.5, 1.5],   // ( 2, 0, 1)
            [0.5, 2.5, 1.5],   // ( 0, 2, 1)
            [2.5, 2.5, 1.5],   // ( 2, 2, 1)
            // YZ-plane edges:
            [1.5, 0.5, 0.5],   // ( 1, 0, 0)
            [1.5, 2.5, 0.5],   // ( 1, 2, 0)
            [1.5, 0.5, 2.5],   // ( 1, 0, 2)
            [1.5, 2.5, 2.5],   // ( 1, 2, 2)
            // XZ-plane edges:
            [0.5, 1.5, 0.5],   // ( 0, 1, 0)
            [2.5, 1.5, 0.5],   // ( 2, 1, 0)
            [0.5, 1.5, 2.5],   // ( 0, 1, 2)
            [2.5, 1.5, 2.5],   // ( 2, 1, 2)
        ];
        for &xyz in &probes {
            let pos = WorldPos::from_world_xyz(
                xyz, 3, &world.library, world.root,
            );
            let recovered = pos.in_frame_rot(
                &world.library, world.root, &Path::root(),
            );
            for i in 0..3 {
                assert!((recovered[i] - xyz[i]).abs() < 1e-3,
                    "round-trip drift at probe {xyz:?} axis {i}: got {} (anchor {:?}, offset {:?})",
                    recovered[i], pos.anchor.as_slice(), pos.offset);
            }
        }
    }
}
