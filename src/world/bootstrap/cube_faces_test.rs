//! 3D-plus / cross primitive: an unrotated centre cube surrounded by
//! six face-aligned `TangentBlock` cubes — one on each ±X, ±Y, ±Z
//! face. Each face TB carries the rotation that maps storage +Y to
//! the world outward direction of that face, modelling the per-face
//! tangent frame of a future cube-sphere layer.
//!
//! Tree shape (root, Cartesian, depth `CUBE_SUBTREE_DEPTH + 1`):
//!
//! ```text
//! root
//!   slot  4 (1,1,0) -> TB { R = rotation_x(-π/2) }   -Z face
//!   slot 10 (1,0,1) -> TB { R = rotation_x(  π  ) }  -Y face
//!   slot 12 (0,1,1) -> TB { R = rotation_z( π/2) }   -X face
//!   slot 13 (1,1,1) -> Cartesian uniform stone        centre (unrotated)
//!   slot 14 (2,1,1) -> TB { R = rotation_z(-π/2) }   +X face
//!   slot 16 (1,2,1) -> TB { R = identity         }   +Y face
//!   slot 22 (1,1,2) -> TB { R = rotation_x( π/2) }   +Z face
//! ```
//!
//! All six rotations are 90° axis-aligned (or identity / 180°), so the
//! rotated cube extents stay flush within the parent slot — no
//! inscribed-cube shrink needed. Validates the rotation-aware
//! `renormalize_world` against the multi-TB-at-one-level case: the
//! camera moving between two face TBs always passes through the
//! Cartesian centre, exercising the pop+redescend across two
//! different TB rotations on the way.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, rotation_x, rotation_z, slot_index, uniform_children, Child, NodeKind,
    NodeLibrary, IDENTITY_ROTATION,
};

/// Depth of each per-cube subtree (the centre and the six face
/// children of root). Small by default — this preset is for
/// navigation correctness across TB boundaries, not f32 precision
/// stress.
pub const CUBE_SUBTREE_DEPTH: u8 = 6;

/// Build a uniform-block recursive Cartesian subtree of `depth`
/// levels. Every level dedups against the same library entry.
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

/// Build a recursive subtree of `depth` levels whose OUTERMOST node
/// is a `TangentBlock` carrying `rotation`. Internal nodes below the
/// TB are plain Cartesian uniform stone.
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

pub fn cube_faces_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // Centre: plain Cartesian uniform stone.
    let centre = build_uniform_cartesian_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH,
    );

    // Six face TBs. Each rotation maps the TB's storage +Y axis to
    // its world outward direction. Verified by `R · (0, 1, 0)`:
    //   +Y face   identity            → (0,  1,  0)
    //   -Y face   rotation_x(π)       → (0, -1,  0)
    //   +X face   rotation_z(-π/2)    → (1,  0,  0)
    //   -X face   rotation_z( π/2)    → (-1, 0,  0)
    //   +Z face   rotation_x( π/2)    → (0,  0,  1)
    //   -Z face   rotation_x(-π/2)    → (0,  0, -1)
    let r_pos_y = IDENTITY_ROTATION;
    let r_neg_y = rotation_x(std::f32::consts::PI);
    let r_pos_x = rotation_z(-std::f32::consts::FRAC_PI_2);
    let r_neg_x = rotation_z(std::f32::consts::FRAC_PI_2);
    let r_pos_z = rotation_x(std::f32::consts::FRAC_PI_2);
    let r_neg_z = rotation_x(-std::f32::consts::FRAC_PI_2);

    let face_pos_y = build_tangent_block_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH, r_pos_y,
    );
    let face_neg_y = build_tangent_block_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH, r_neg_y,
    );
    let face_pos_x = build_tangent_block_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH, r_pos_x,
    );
    let face_neg_x = build_tangent_block_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH, r_neg_x,
    );
    let face_pos_z = build_tangent_block_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH, r_pos_z,
    );
    let face_neg_z = build_tangent_block_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH, r_neg_z,
    );

    let mut root_children = empty_children();
    root_children[slot_index(1, 1, 1)] = centre;
    root_children[slot_index(1, 2, 1)] = face_pos_y;   // slot 16
    root_children[slot_index(1, 0, 1)] = face_neg_y;   // slot 10
    root_children[slot_index(2, 1, 1)] = face_pos_x;   // slot 14
    root_children[slot_index(0, 1, 1)] = face_neg_x;   // slot 12
    root_children[slot_index(1, 1, 2)] = face_pos_z;   // slot 22
    root_children[slot_index(1, 1, 0)] = face_neg_z;   // slot 4
    let root = library.insert_with_kind(root_children, NodeKind::Cartesian);
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "cube_faces_test world: tree_depth={}, library_entries={}",
        world.tree_depth(),
        world.library.len(),
    );
    world
}

/// Camera spawn: at the corner slot 25 (= (1, 2, 2)) — vacant cell
/// adjacent to both the +Y arm (slot 16) and the +Z arm (slot 22).
/// Diagonal yaw + pitch puts the cross in view.
pub fn cube_faces_test_spawn() -> WorldPos {
    WorldPos::uniform_column(slot_index(1, 2, 2) as u8, 1, [0.5, 0.5, 0.5])
}

pub(super) fn bootstrap_cube_faces_test_world() -> WorldBootstrap {
    let world = cube_faces_test_world();
    let spawn_pos = cube_faces_test_spawn();
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // Diagonal off-axis view so the cross reads as a 3D plus
        // and the camera doesn't appear to "enter" any face frame
        // when rendering crosses TB boundaries.
        default_spawn_yaw: std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: -std::f32::consts::FRAC_PI_4,
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_layout_is_a_3d_plus() {
        let world = cube_faces_test_world();
        let root_node = world.library.get(world.root).expect("root exists");
        assert_eq!(root_node.kind, NodeKind::Cartesian);
        // Centre is Cartesian, not TB.
        match root_node.children[slot_index(1, 1, 1)] {
            Child::Node(id) => {
                let n = world.library.get(id).expect("centre subtree");
                assert_eq!(n.kind, NodeKind::Cartesian, "centre must be unrotated");
            }
            other => panic!("expected Node at centre, got {other:?}"),
        }
        // All six face slots are TBs.
        for slot_xyz in [
            (1, 2, 1), (1, 0, 1), (2, 1, 1), (0, 1, 1), (1, 1, 2), (1, 1, 0),
        ] {
            let slot = slot_index(slot_xyz.0, slot_xyz.1, slot_xyz.2);
            match root_node.children[slot] {
                Child::Node(id) => {
                    let n = world.library.get(id).expect("face subtree");
                    assert!(n.kind.is_tangent_block(),
                        "face slot {slot_xyz:?} must be a TangentBlock, got {:?}", n.kind);
                }
                other => panic!("expected Node at face slot {slot_xyz:?}, got {other:?}"),
            }
        }
        // Other slots are empty.
        for slot in 0..27 {
            let (sx, sy, sz) = crate::world::tree::slot_coords(slot);
            let in_cross = (sx, sy, sz) == (1, 1, 1)
                || (sx, sy, sz) == (1, 2, 1) || (sx, sy, sz) == (1, 0, 1)
                || (sx, sy, sz) == (2, 1, 1) || (sx, sy, sz) == (0, 1, 1)
                || (sx, sy, sz) == (1, 1, 2) || (sx, sy, sz) == (1, 1, 0);
            if in_cross { continue; }
            assert_eq!(root_node.children[slot], Child::Empty,
                "slot {} = ({},{},{}) outside the cross must be empty",
                slot, sx, sy, sz);
        }
    }

    /// Crossing from the centre into a face TB and back must
    /// preserve world position. Exercises pop+redescend with one
    /// non-trivial TB rotation per transition.
    #[test]
    fn crossing_centre_to_pos_x_face_preserves_world_pos() {
        use crate::world::anchor::Path;
        let world = cube_faces_test_world();
        // Spawn at world (1.99, 1.5, 1.5) — just inside the centre
        // cube near the +X face boundary.
        let mut pos = WorldPos::from_world_xyz(
            [1.99, 1.5, 1.5], 3, &world.library, world.root,
        );
        let world_before = pos.in_frame_rot(
            &world.library, world.root, &Path::root(),
        );
        // Move +X by enough to cross x=2 into the +X face TB.
        let cell_world = crate::world::anchor::WORLD_SIZE
            / 3.0_f32.powi(pos.anchor.depth() as i32);
        let dx_world = 0.10; // crosses x=2 into the +X face cube
        let dx_offset = dx_world / cell_world;
        pos.add_local([dx_offset, 0.0, 0.0], &world.library, world.root);
        let world_after = pos.in_frame_rot(
            &world.library, world.root, &Path::root(),
        );
        let tol = 1e-3;
        assert!((world_after[0] - (world_before[0] + dx_world)).abs() < tol,
            "X didn't track +X delta into face TB: before {} after {} dx {}",
            world_before[0], world_after[0], dx_world);
        assert!((world_after[1] - world_before[1]).abs() < tol,
            "Y teleported across +X face boundary: before {} after {}",
            world_before[1], world_after[1]);
        assert!((world_after[2] - world_before[2]).abs() < tol,
            "Z teleported across +X face boundary: before {} after {}",
            world_before[2], world_after[2]);
        // Anchor must now point through slot 14 (the +X face TB).
        assert_eq!(pos.anchor.slot(0), slot_index(2, 1, 1) as u8,
            "after crossing, anchor[0] should be the +X face slot");
    }

    /// Each face TB carries a distinct rotation. A round-trip through
    /// `from_world_xyz` → `in_frame_rot` for a world point inside any
    /// face must reproduce the input — proves that the storage-frame
    /// slot derivation (R^T at TB descend) and the world-position
    /// derivation (R at in_frame_rot accumulation) are exact inverses
    /// for every distinct rotation in the cross.
    #[test]
    fn from_world_xyz_round_trips_through_every_face() {
        use crate::world::anchor::Path;
        let world = cube_faces_test_world();
        // Pick a non-symmetric point inside each face cube. Y / Z
        // offsets break planar symmetries so any permutation /
        // sign-flip in the rotation matrix would surface.
        let probe_points: [[f32; 3]; 7] = [
            [1.5,  1.5,  1.5],   // centre  (Cartesian)
            [1.5,  2.27, 1.62],  // +Y face (slot 16)
            [1.5,  0.31, 1.62],  // -Y face (slot 10)
            [2.31, 1.62, 1.43],  // +X face (slot 14)
            [0.31, 1.62, 1.43],  // -X face (slot 12)
            [1.62, 1.43, 2.31],  // +Z face (slot 22)
            [1.62, 1.43, 0.31],  // -Z face (slot 4)
        ];
        for &xyz in &probe_points {
            let pos = WorldPos::from_world_xyz(
                xyz, 3, &world.library, world.root,
            );
            let recovered = pos.in_frame_rot(
                &world.library, world.root, &Path::root(),
            );
            for i in 0..3 {
                assert!((recovered[i] - xyz[i]).abs() < 1e-4,
                    "round-trip drift at probe {:?} axis {}: got {} (anchor {:?}, offset {:?})",
                    xyz, i, recovered[i], pos.anchor.as_slice(), pos.offset);
            }
        }
    }
}
