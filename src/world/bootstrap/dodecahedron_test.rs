//! Step-3 multi-TB primitive: a dodecahedral arrangement — twelve
//! `TangentBlock` cubes, one per face of a regular dodecahedron, plus
//! a Cartesian centre cube. Each face TB carries the rotation that
//! maps its storage +Y axis to the actual dodecahedron face normal —
//! a non-axis-aligned direction parametrised by the golden ratio
//! `φ = (1 + √5) / 2`.
//!
//! Where `cube_faces_test` validates `renormalize_world` against six
//! axis-aligned (90° / 180°) rotations, this preset stresses it
//! against twelve distinct *non*-axis-aligned rotations. The 24-bit
//! rotation hash means each TB is its own library entry; deduplication
//! only kicks in for the uniform-stone subtree below the rotated
//! head.
//!
//! Slot layout. The twelve dodecahedron face normals are the twelve
//! icosahedron vertices, namely cyclic permutations of `(0, ±1, ±φ)`.
//! Each is placed at the 3³ root slot whose grid direction
//! `(sx − 1, sy − 1, sz − 1) ∈ {−1, 0, +1}³` best approximates that
//! face normal — i.e. the twelve cube-edge slots:
//!
//! ```text
//!   (0,  ±1, ±φ)   →  YZ-plane edge slots: (1, 0|2, 0|2)
//!   (±1, ±φ,  0)   →  XY-plane edge slots: (0|2, 0|2, 1)
//!   (±φ,  0, ±1)   →  XZ-plane edge slots: (0|2, 1, 0|2)
//! ```
//!
//! The slot direction is only an approximate position — the *rotation*
//! stored in the TB is the exact dodecahedron face normal. A TB cube
//! therefore points its storage +Y at the true face normal, which is
//! tilted relative to its slot direction by ~32° (the offset between
//! `(0, 1, 1)/√2` and `(0, 1, φ)/√(1+φ²)`). That tilt is the whole
//! point: every transition between centre and any face exercises a
//! non-trivial, non-axis-aligned `R^T` at descend and `R` at world
//! recovery, with twelve distinct R's at the same tree level.
//!
//! Tree shape (root, Cartesian, depth `CUBE_SUBTREE_DEPTH + 1`):
//!
//! ```text
//! root
//!   slot 13 (1,1,1) -> Cartesian uniform stone        centre (unrotated)
//!   slot 25 (1,2,2) -> TB { face = (0,  1,  φ) }
//!   slot  7 (1,2,0) -> TB { face = (0,  1, -φ) }
//!   slot 19 (1,0,2) -> TB { face = (0, -1,  φ) }
//!   slot  1 (1,0,0) -> TB { face = (0, -1, -φ) }
//!   slot 17 (2,2,1) -> TB { face = (1,  φ,  0) }
//!   slot 11 (2,0,1) -> TB { face = (1, -φ,  0) }
//!   slot 15 (0,2,1) -> TB { face = (-1, φ,  0) }
//!   slot  9 (0,0,1) -> TB { face = (-1,-φ,  0) }
//!   slot 23 (2,1,2) -> TB { face = (φ,  0,  1) }
//!   slot  5 (2,1,0) -> TB { face = (φ,  0, -1) }
//!   slot 21 (0,1,2) -> TB { face = (-φ, 0,  1) }
//!   slot  3 (0,1,0) -> TB { face = (-φ, 0, -1) }
//! ```
//!
//! The remaining fourteen slots (six face-adjacent, eight corner) are
//! empty, leaving room for the camera to orbit the figure.
//!
//! Caveat — boundary crossings near face TBs are intentionally not
//! claimed by this preset. The slot direction `(sx − 1, sy − 1, sz −
//! 1)` is only a coarse grid placement, while the TB rotation stores
//! the true regular-dodecahedron face normal. The current architecture
//! makes the renderer/CPU raycast/anchor descent agree through
//! `TbBoundary`'s centred `R^T / tb_scale` transform, so this preset
//! validates the rotated content transform rather than importing an
//! older world-coordinate inverse API.

use super::WorldBootstrap;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary,
};

/// Depth of each per-cube subtree (the centre and the twelve face
/// children of root). Small by default — this preset is for
/// rotation-diversity stress on `renormalize_world`, not f32
/// precision stress.
const CUBE_SUBTREE_DEPTH: u8 = 6;

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
/// TB are plain Cartesian uniform stone — the rotation only attaches
/// to the TB head.
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

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / m, v[1] / m, v[2] / m]
}

/// Build a column-major rotation matrix `R` such that `R · (0, 1, 0) = n`,
/// using world `+Y` as a stable up-helper to fix the remaining yaw
/// degree of freedom. The dodecahedron face normals all satisfy
/// `|n.y| ≤ φ/√(1+φ²) ≈ 0.851 < 0.95`, so the helper is never
/// degenerate; the fallback to world `+X` exists for completeness.
///
/// Construction:
///
/// 1. `right = normalize(h × n)` — perpendicular to `n` in the helper
///    plane.
/// 2. `forward = right × n` — completes a right-handed basis with
///    `right × n = forward`, i.e. `col0 × col1 = col2`.
/// 3. `R = [right | n | forward]`, column-major.
fn rotation_align_y_to(face_normal: [f32; 3]) -> [[f32; 3]; 3] {
    let n = normalize(face_normal);
    let helper = if n[1].abs() > 0.95 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = normalize(cross(helper, n));
    let forward = cross(right, n);
    [right, n, forward]
}

/// `(slot_x, slot_y, slot_z, face_normal_unnormalized)` for each of
/// the twelve dodecahedron faces. The face normals are
/// `rotation_align_y_to`-normalised on use; the slot is the
/// closest cube-edge cell in the 3³ grid.
fn face_table() -> [(usize, usize, usize, [f32; 3]); 12] {
    // φ = (1 + √5) / 2. Computed at runtime — `f32::sqrt` is not const.
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    [
        // (0, ±1, ±φ) — YZ-plane normals → YZ-plane edge slots
        (1, 2, 2, [0.0,  1.0,  phi]),
        (1, 2, 0, [0.0,  1.0, -phi]),
        (1, 0, 2, [0.0, -1.0,  phi]),
        (1, 0, 0, [0.0, -1.0, -phi]),
        // (±1, ±φ, 0) — XY-plane normals → XY-plane edge slots
        (2, 2, 1, [ 1.0,  phi, 0.0]),
        (2, 0, 1, [ 1.0, -phi, 0.0]),
        (0, 2, 1, [-1.0,  phi, 0.0]),
        (0, 0, 1, [-1.0, -phi, 0.0]),
        // (±φ, 0, ±1) — XZ-plane normals → XZ-plane edge slots
        (2, 1, 2, [ phi, 0.0,  1.0]),
        (2, 1, 0, [ phi, 0.0, -1.0]),
        (0, 1, 2, [-phi, 0.0,  1.0]),
        (0, 1, 0, [-phi, 0.0, -1.0]),
    ]
}

pub(super) fn dodecahedron_test_world() -> WorldState {
    let mut library = NodeLibrary::default();

    // Centre: plain Cartesian uniform stone. Becomes the navigation
    // hub — every transition between two face TBs passes through it.
    let centre = build_uniform_cartesian_subtree(
        &mut library, block::STONE, CUBE_SUBTREE_DEPTH,
    );

    let mut root_children = empty_children();
    root_children[slot_index(1, 1, 1)] = centre;

    for (sx, sy, sz, face_normal) in face_table() {
        let rotation = rotation_align_y_to(face_normal);
        let child = build_tangent_block_subtree(
            &mut library, block::STONE, CUBE_SUBTREE_DEPTH, rotation,
        );
        root_children[slot_index(sx, sy, sz)] = child;
    }

    let root = library.insert_with_kind(root_children, NodeKind::Cartesian);
    library.ref_inc(root);

    let world = WorldState { root, library };
    eprintln!(
        "dodecahedron_test world: tree_depth={}, library_entries={}",
        world.tree_depth(),
        world.library.len(),
    );
    world
}

/// Camera spawn: corner slot 26 = (2, 2, 2), the empty +X+Y+Z corner
/// of the root cell. Diagonal yaw + pitch points back toward the
/// centre cube so the dodecahedral arrangement reads in 3D.
pub(super) fn dodecahedron_test_spawn() -> WorldPos {
    WorldPos::uniform_column(slot_index(2, 2, 2) as u8, 1, [0.5, 0.5, 0.5])
}

pub(super) fn bootstrap_dodecahedron_test_world() -> WorldBootstrap {
    let world = dodecahedron_test_world();
    let spawn_pos = dodecahedron_test_spawn();
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // Diagonal yaw 5π/4 looks from the +X+Y+Z corner back toward
        // the centre; pitch −π/6 lowers the gaze just enough to
        // catch the bottom faces without entering any TB's frame.
        default_spawn_yaw: 5.0 * std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: -std::f32::consts::FRAC_PI_6,
        plain_layers: 0,
        color_registry: crate::world::palette::ColorRegistry::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_has_expected_depth() {
        let world = dodecahedron_test_world();
        assert_eq!(world.tree_depth(), CUBE_SUBTREE_DEPTH as u32 + 1);
    }

    #[test]
    fn root_layout_has_centre_plus_twelve_face_tbs() {
        let world = dodecahedron_test_world();
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

        // All twelve face slots are TBs.
        for (sx, sy, sz, _) in face_table() {
            let slot = slot_index(sx, sy, sz);
            match root_node.children[slot] {
                Child::Node(id) => {
                    let n = world.library.get(id).expect("face subtree");
                    assert!(
                        n.kind.is_tangent_block(),
                        "face slot ({sx},{sy},{sz}) must be a TangentBlock, got {:?}",
                        n.kind,
                    );
                }
                other => panic!("expected Node at face slot ({sx},{sy},{sz}), got {other:?}"),
            }
        }

        // The remaining fourteen slots are empty.
        let face_slots: std::collections::HashSet<usize> = face_table()
            .iter()
            .map(|(sx, sy, sz, _)| slot_index(*sx, *sy, *sz))
            .chain(std::iter::once(slot_index(1, 1, 1)))
            .collect();
        for slot in 0..27 {
            if face_slots.contains(&slot) {
                continue;
            }
            assert_eq!(
                root_node.children[slot],
                Child::Empty,
                "slot {slot} outside the dodecahedral cross must be empty",
            );
        }
    }

    /// Each of the twelve face TBs carries a *distinct* rotation —
    /// proves that no two dodecahedron face normals collapse under
    /// the rotation-bit hash, and so each face is its own library
    /// entry.
    #[test]
    fn all_twelve_faces_have_distinct_rotations() {
        let world = dodecahedron_test_world();
        let root_node = world.library.get(world.root).expect("root exists");
        let mut tb_ids = Vec::new();
        for (sx, sy, sz, _) in face_table() {
            match root_node.children[slot_index(sx, sy, sz)] {
                Child::Node(id) => tb_ids.push(id),
                other => panic!("non-Node at face slot: {other:?}"),
            }
        }
        let unique: std::collections::HashSet<_> = tb_ids.iter().collect();
        assert_eq!(
            unique.len(),
            tb_ids.len(),
            "expected 12 distinct face-TB library entries, got {} unique of {}",
            unique.len(),
            tb_ids.len(),
        );
    }

    /// Verifies that `rotation_align_y_to` actually maps storage +Y
    /// to the requested face normal, for every dodecahedron face.
    /// The tolerance is loose because the input face normals are
    /// pre-normalisation — we compare to the *normalised* normal.
    #[test]
    fn rotation_aligns_storage_y_to_face_normal() {
        for (_, _, _, face_normal) in face_table() {
            let r = rotation_align_y_to(face_normal);
            // R · (0, 1, 0) = column 1 of R.
            let r_y = r[1];
            let expected = normalize(face_normal);
            for i in 0..3 {
                assert!(
                    (r_y[i] - expected[i]).abs() < 1e-5,
                    "R·ŷ mismatch on axis {i} for face {face_normal:?}: got {r_y:?}, want {expected:?}",
                );
            }
        }
    }

    /// Each rotation is a proper rotation: orthonormal columns, det = +1.
    #[test]
    fn every_rotation_is_orthonormal_and_right_handed() {
        for (_, _, _, face_normal) in face_table() {
            let r = rotation_align_y_to(face_normal);
            let c0 = r[0];
            let c1 = r[1];
            let c2 = r[2];
            // Unit columns
            for (i, c) in [c0, c1, c2].iter().enumerate() {
                let len2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                assert!(
                    (len2 - 1.0).abs() < 1e-5,
                    "column {i} not unit for face {face_normal:?}: |c|² = {len2}",
                );
            }
            // Orthogonal columns
            let dot01 = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
            let dot02 = c0[0] * c2[0] + c0[1] * c2[1] + c0[2] * c2[2];
            let dot12 = c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2];
            for (name, d) in [("c0·c1", dot01), ("c0·c2", dot02), ("c1·c2", dot12)] {
                assert!(
                    d.abs() < 1e-5,
                    "{name} not orthogonal for face {face_normal:?}: dot = {d}",
                );
            }
            // Right-handed: c0 × c1 = c2
            let cross01 = cross(c0, c1);
            for i in 0..3 {
                assert!(
                    (cross01[i] - c2[i]).abs() < 1e-5,
                    "c0 × c1 ≠ c2 on axis {i} for face {face_normal:?}: got {} want {}",
                    cross01[i],
                    c2[i],
                );
            }
        }
    }

    /// The current TangentBlock contract is a centred similarity
    /// transform: parent frame enters storage as `R^T / tb_scale`,
    /// storage exits parent as `R * tb_scale`. Verify that each
    /// dodecahedron face rotation participates in that inverse pair.
    #[test]
    fn tb_boundary_enter_exit_round_trips_for_every_face() {
        use crate::world::gpu::TbBoundary;

        for (_, _, _, face_normal) in face_table() {
            let boundary = TbBoundary::new(rotation_align_y_to(face_normal));
            assert!(
                boundary.tb_scale > 0.0 && boundary.tb_scale <= 1.0,
                "invalid tb_scale {} for face {:?}",
                boundary.tb_scale,
                face_normal,
            );

            let p = [1.87, 1.12, 1.41];
            let entered = boundary.enter_point(p, 1.5);
            let exited = boundary.exit_point(entered, 1.5);
            for i in 0..3 {
                assert!(
                    (exited[i] - p[i]).abs() < 1e-5,
                    "point round-trip drift on axis {i} for face {:?}: got {:?}, want {:?}",
                    face_normal,
                    exited,
                    p,
                );
            }

            let d = [0.23, -0.71, 0.66];
            let entered = boundary.enter_dir(d);
            let exited = boundary.exit_dir(entered);
            for i in 0..3 {
                assert!(
                    (exited[i] - d[i]).abs() < 1e-5,
                    "dir round-trip drift on axis {i} for face {:?}: got {:?}, want {:?}",
                    face_normal,
                    exited,
                    d,
                );
            }
        }
    }
}
