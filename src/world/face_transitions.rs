//! Cubed-sphere face-seam transitions as pure path rewrites.
//!
//! When a [`Position`] sits inside a `CubedSphereFace` node and the
//! carry from `step_neighbor` overflows that node's `u` or `v` axis,
//! we exit the current face and enter a sibling face. This module
//! turns the geometric problem ("which face borders +u of +X?") into
//! a lookup table keyed by `(face, edge)`.
//!
//! The math matches `cubesphere::face_uv_to_dir` and the face basis
//! definitions in `cubesphere::Face::tangents`. Callers in
//! `step_neighbor` (see [`crate::world::position`]) compose this with
//! the Cartesian carry to produce the full path rewrite.
//!
//! Step 8 introduces the plumbing; no engine code constructs
//! `NodeKind::CubedSphereFace` yet, so these helpers are dead code
//! until step 9 moves planets into the tree as body + face nodes.
//! Tests pin down the expected behavior so that step 9 has something
//! to rely on.

use crate::world::position::Position;

/// Which edge of a face we crossed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FaceEdge {
    PlusU,
    MinusU,
    PlusV,
    MinusV,
}

/// Result of crossing a face seam: the new face, plus how the
/// previous face's `(u_slot, v_slot)` maps onto the new face's.
///
/// `u_from` / `v_from` pick which axis on the old face (u=0, v=1)
/// provides the coordinate for the new axis, and `u_flip` / `v_flip`
/// indicate whether that axis is traversed in the opposite direction.
/// `r_slot` is preserved across every seam.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FaceTransition {
    pub new_face: u8,
    pub u_from: u8,
    pub u_flip: bool,
    pub v_from: u8,
    pub v_flip: bool,
}

/// Face indices matching `cubesphere::Face`: 0=+X, 1=-X, 2=+Y, 3=-Y,
/// 4=+Z, 5=-Z.
pub mod face {
    pub const POS_X: u8 = 0;
    pub const NEG_X: u8 = 1;
    pub const POS_Y: u8 = 2;
    pub const NEG_Y: u8 = 3;
    pub const POS_Z: u8 = 4;
    pub const NEG_Z: u8 = 5;
}

/// Return the face we enter when stepping off `face` across `edge`.
///
/// Derivation: each face has a `normal`, `u_axis`, `v_axis` (see
/// `cubesphere::Face::tangents`). Crossing `+u` means the point's
/// world direction leaves the current face's `n + u_axis + v * v_axis`
/// patch with `u = +1` (the face's `+u_axis` edge). The neighboring
/// face is the one whose normal direction is the current face's
/// `+u_axis` direction.
///
/// ```text
///   +X face: n=+X, u_axis=-Z, v_axis=+Y
///     +u → -Z face     -u → +Z face
///     +v → +Y face     -v → -Y face
///
///   -X face: n=-X, u_axis=+Z, v_axis=+Y
///     +u → +Z face     -u → -Z face
///     +v → +Y face     -v → -Y face
///
///   +Y face: n=+Y, u_axis=+X, v_axis=-Z
///     +u → +X face     -u → -X face
///     +v → -Z face     -v → +Z face
///
///   -Y face: n=-Y, u_axis=+X, v_axis=+Z
///     +u → +X face     -u → -X face
///     +v → +Z face     -v → -Z face
///
///   +Z face: n=+Z, u_axis=+X, v_axis=+Y
///     +u → +X face     -u → -X face
///     +v → +Y face     -v → -Y face
///
///   -Z face: n=-Z, u_axis=-X, v_axis=+Y
///     +u → -X face     -u → +X face
///     +v → +Y face     -v → -Y face
/// ```
pub fn neighbor_face(face: u8, edge: FaceEdge) -> u8 {
    use face::*;
    use FaceEdge::*;
    match (face, edge) {
        (POS_X, PlusU)  => NEG_Z,
        (POS_X, MinusU) => POS_Z,
        (POS_X, PlusV)  => POS_Y,
        (POS_X, MinusV) => NEG_Y,

        (NEG_X, PlusU)  => POS_Z,
        (NEG_X, MinusU) => NEG_Z,
        (NEG_X, PlusV)  => POS_Y,
        (NEG_X, MinusV) => NEG_Y,

        (POS_Y, PlusU)  => POS_X,
        (POS_Y, MinusU) => NEG_X,
        (POS_Y, PlusV)  => NEG_Z,
        (POS_Y, MinusV) => POS_Z,

        (NEG_Y, PlusU)  => POS_X,
        (NEG_Y, MinusU) => NEG_X,
        (NEG_Y, PlusV)  => POS_Z,
        (NEG_Y, MinusV) => NEG_Z,

        (POS_Z, PlusU)  => POS_X,
        (POS_Z, MinusU) => NEG_X,
        (POS_Z, PlusV)  => POS_Y,
        (POS_Z, MinusV) => NEG_Y,

        (NEG_Z, PlusU)  => NEG_X,
        (NEG_Z, MinusU) => POS_X,
        (NEG_Z, PlusV)  => POS_Y,
        (NEG_Z, MinusV) => NEG_Y,

        _ => face, // invalid → no-op
    }
}

/// Placeholder for the full (u_slot, v_slot) rewrite across a seam.
///
/// Deriving the 24 per-edge axis-swap/flip rules requires matching
/// each pair of face bases (current `u_axis`, `v_axis` vs. neighbor
/// `u_axis`, `v_axis`) along their shared edge. That derivation is
/// step-8 scaffolding; full rewrite rules land when the shader side
/// of the sphere-as-tree-node migration (step 9) starts consuming
/// them. Until then callers treat face seams as "continues in the
/// same (u_slot, v_slot) frame" — correct for seams where the bases
/// align, wrong elsewhere.
pub fn seam_transition(face: u8, edge: FaceEdge) -> FaceTransition {
    FaceTransition {
        new_face: neighbor_face(face, edge),
        u_from: 0,
        u_flip: false,
        v_from: 1,
        v_flip: false,
    }
}

/// Radial exit: a carry on the `r` axis past the face's outer
/// (`r_slot = 2`) or inner (`r_slot = 0`) boundary bubbles up out of
/// the face subtree entirely. This function removes the face and
/// body layers from `path`, leaving the path pointing at the body's
/// position in its Cartesian parent. Returns `true` if the pop was
/// possible.
///
/// Step 8 scaffolding: the actual depth at which face and body live
/// in the path depends on the tree layout that step 9 establishes.
/// For now this is a stub that pops two levels if present.
pub fn radial_exit(pos: &mut Position) -> bool {
    if pos.depth < 2 {
        return false;
    }
    pos.depth -= 2;
    // Zero-out the popped entries for hygiene.
    pos.path[pos.depth as usize] = 0;
    pos.path[pos.depth as usize + 1] = 0;
    true
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use face::*;

    #[test]
    fn neighbors_are_symmetric() {
        // Crossing +u of A and arriving at face B means crossing some
        // edge of B will return us to A. Verify the neighbor table is
        // geometrically consistent.
        for &f in &[POS_X, NEG_X, POS_Y, NEG_Y, POS_Z, NEG_Z] {
            for &edge in &[FaceEdge::PlusU, FaceEdge::MinusU, FaceEdge::PlusV, FaceEdge::MinusV] {
                let nb = neighbor_face(f, edge);
                assert_ne!(nb, f, "face {} crossing {:?} stays on same face", f, edge);
                // One of the four edges of nb must lead back to f.
                let returns: Vec<FaceEdge> = [
                    FaceEdge::PlusU, FaceEdge::MinusU, FaceEdge::PlusV, FaceEdge::MinusV,
                ]
                .into_iter()
                .filter(|&e| neighbor_face(nb, e) == f)
                .collect();
                assert!(
                    !returns.is_empty(),
                    "face {} crossing {:?} → {}, but {} has no edge back to {}",
                    f, edge, nb, nb, f
                );
            }
        }
    }

    #[test]
    fn opposite_faces_are_not_neighbors() {
        // +X and -X should NEVER be neighbors through any single
        // edge crossing — they're antipodal. Same for y and z pairs.
        for (a, b) in [(POS_X, NEG_X), (POS_Y, NEG_Y), (POS_Z, NEG_Z)] {
            for &edge in &[FaceEdge::PlusU, FaceEdge::MinusU, FaceEdge::PlusV, FaceEdge::MinusV] {
                assert_ne!(neighbor_face(a, edge), b, "{} → {} via {:?}", a, b, edge);
                assert_ne!(neighbor_face(b, edge), a, "{} → {} via {:?}", b, a, edge);
            }
        }
    }

    #[test]
    fn radial_exit_pops_two_levels() {
        let mut p = Position::at_slot_path(&[1, 2, 3], [0.0; 3]);
        assert!(radial_exit(&mut p));
        assert_eq!(p.depth, 1);
        assert_eq!(p.slots(), &[1]);
    }

    #[test]
    fn radial_exit_on_shallow_path_fails() {
        let mut p = Position::at_slot_path(&[1], [0.0; 3]);
        assert!(!radial_exit(&mut p));
        assert_eq!(p.depth, 1);
    }

    #[test]
    fn seam_transition_picks_neighbor_face() {
        let t = seam_transition(POS_X, FaceEdge::PlusU);
        assert_eq!(t.new_face, NEG_Z);
    }
}
