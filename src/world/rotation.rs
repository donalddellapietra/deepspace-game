//! Per-cell rotation primitives.
//!
//! Every cell in the tree has a rotation relative to its parent.
//! Most are identity (a normal Cartesian subdivision). Some are
//! `T` — the 45° Y rotation composed with a √2 XZ stretch that
//! maps a diamond-prism inscribed in the parent's AABB to a
//! standard `[0, 3)³` Cartesian frame.
//!
//! The renderer / walker accumulate rotations as they descend, and
//! identity rotations cost nothing at runtime — both `matmul` and
//! `matvec` reduce to no-ops when one operand is the identity.
//!
//! # The T transform
//!
//! `T(x, y, z) = (x - z, y, x + z)` is `R_y(-45°) ∘ scale_xz(√2)`.
//! It maps the inscribed prism `{|x|, |z| ≤ cs/(2√2)} × {|y| ≤ cs/2}`
//! to the cube `[-cs/2, cs/2]³`, so DDA inside a rotated subtree
//! runs in a standard `[0, 3)³` local frame.
//!
//! `T⁻¹(u, y, w) = ((u + w) / 2, y, (w − u) / 2)`.

use crate::world::tree::NodeKind;

/// 3×3 matrix in row-major layout. Row `i` is `m[i]`.
pub type Mat3 = [[f32; 3]; 3];

pub const IDENTITY: Mat3 = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// `T(x, y, z) = (x − z, y, x + z)` — 45° Y rotation + √2 XZ stretch.
/// Maps `Rotated45Y` local-frame coords to parent-frame coords' inverse;
/// applying T to a parent-frame vector yields the rotated-frame vector.
pub const T_MATRIX: Mat3 = [
    [1.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
];

/// `T⁻¹(u, y, w) = ((u + w)/2, y, (w − u)/2)`.
pub const T_INV_MATRIX: Mat3 = [
    [0.5, 0.0, 0.5],
    [0.0, 1.0, 0.0],
    [-0.5, 0.0, 0.5],
];

/// Rotation that maps a parent-frame vector into this kind's local
/// frame. For every kind except `Rotated45Y` this is identity, so
/// the walker accumulates rotations cheaply for the common case.
#[inline]
pub fn rotation_of(kind: NodeKind) -> Mat3 {
    match kind {
        NodeKind::Rotated45Y => T_MATRIX,
        NodeKind::Cartesian
        | NodeKind::WrappedPlane { .. }
        | NodeKind::TangentBlock => IDENTITY,
    }
}

#[inline]
pub fn matmul(a: Mat3, b: Mat3) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

#[inline]
pub fn matvec(m: Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
pub fn is_identity(m: Mat3) -> bool {
    m == IDENTITY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matvec() {
        assert_eq!(matvec(IDENTITY, [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_matmul() {
        assert_eq!(matmul(IDENTITY, T_MATRIX), T_MATRIX);
        assert_eq!(matmul(T_MATRIX, IDENTITY), T_MATRIX);
    }

    #[test]
    fn t_matvec_known_points() {
        // T(1, 0, 0) = (1, 0, 1)
        assert_eq!(matvec(T_MATRIX, [1.0, 0.0, 0.0]), [1.0, 0.0, 1.0]);
        // T(0, 0, 1) = (-1, 0, 1)
        assert_eq!(matvec(T_MATRIX, [0.0, 0.0, 1.0]), [-1.0, 0.0, 1.0]);
        // T(1, 0, 1) = (0, 0, 2) — diagonal stretches by √2
        assert_eq!(matvec(T_MATRIX, [1.0, 0.0, 1.0]), [0.0, 0.0, 2.0]);
    }

    #[test]
    fn t_inverse_round_trips() {
        let v = [1.7f32, 0.3, -0.8];
        let tv = matvec(T_MATRIX, v);
        let v2 = matvec(T_INV_MATRIX, tv);
        for i in 0..3 {
            assert!((v[i] - v2[i]).abs() < 1e-6, "axis {i}: {} vs {}", v[i], v2[i]);
        }
    }

    #[test]
    fn rotation_of_kind() {
        assert!(is_identity(rotation_of(NodeKind::Cartesian)));
        assert_eq!(rotation_of(NodeKind::Rotated45Y), T_MATRIX);
    }
}
