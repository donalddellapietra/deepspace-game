//! Cube-seam adjacency and face-local basis for cubed-sphere anchors.
//!
//! A [`Face`] subtree is laid out as a 27-ary tree whose local axes are
//! `(u, v, r)` — u and v are the face's tangent-plane coordinates,
//! r is radial distance from the planet center. Axes 0/1/2 inside a
//! `CubedSphereFace` subtree therefore map to `(u, v, r)`, not
//! `(x, y, z)`.
//!
//! This module answers two questions that come up once an anchor
//! descends into a face subtree:
//!
//! - What orthonormal basis expresses local `(u, v, r)` axes in
//!   world-space `(x, y, z)`? See [`face_basis`].
//!
//! - When a step overflows the face root on the `u` or `v` axis, which
//!   neighboring face do we enter, and how do the local axes remap?
//!   See [`seam_neighbor`] and [`Seam`]. The 24 cases are derived
//!   algorithmically from each face's `tangents()` and `normal()`, so
//!   the table can't drift from the geometric ground truth.
//!
//! The basis is an orthonormal approximation — the true Jacobian of
//! the equal-angle projection is non-linear, but this orthonormal
//! form is exact at the face center and a good approximation
//! elsewhere, which is all the player physics needs.

use super::cubesphere::Face;

type Vec3 = [f32; 3];

#[inline]
fn dot(a: Vec3, b: Vec3) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
#[inline]
fn neg(a: Vec3) -> Vec3 { [-a[0], -a[1], -a[2]] }

/// 3×3 orthonormal matrix columns: `[col_u, col_v, col_r]` expressed
/// in world `(x, y, z)`. Multiplying this matrix by a local-axis
/// delta `[du, dv, dr]` yields the corresponding world-space delta
/// (approximate, orthonormal).
#[derive(Copy, Clone, Debug)]
pub struct FaceBasis {
    pub u_axis: Vec3,
    pub v_axis: Vec3,
    pub r_axis: Vec3,
}

impl FaceBasis {
    /// World-space delta → face-local `(u, v, r)` delta. Since the
    /// basis is orthonormal, the inverse is the transpose.
    #[inline]
    pub fn world_to_local(&self, world: Vec3) -> Vec3 {
        [dot(world, self.u_axis), dot(world, self.v_axis), dot(world, self.r_axis)]
    }

    /// Face-local delta → world delta.
    #[inline]
    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        [
            local[0] * self.u_axis[0] + local[1] * self.v_axis[0] + local[2] * self.r_axis[0],
            local[0] * self.u_axis[1] + local[1] * self.v_axis[1] + local[2] * self.r_axis[1],
            local[0] * self.u_axis[2] + local[1] * self.v_axis[2] + local[2] * self.r_axis[2],
        ]
    }

    /// Identity basis — local axes equal world axes. Used when the
    /// anchor is not inside any face subtree (pure Cartesian nodes).
    pub const IDENTITY: FaceBasis = FaceBasis {
        u_axis: [1.0, 0.0, 0.0],
        v_axis: [0.0, 1.0, 0.0],
        r_axis: [0.0, 0.0, 1.0],
    };
}

/// Orthonormal face basis: `(tangent_u, tangent_v, normal)`.
pub fn face_basis(face: Face) -> FaceBasis {
    let (u, v) = face.tangents();
    let r = face.normal();
    FaceBasis { u_axis: u, v_axis: v, r_axis: r }
}

/// How local `(u, v, r)` axes on a from-face remap to the neighboring
/// to-face when a step crosses the seam.
///
/// `axis_map[i]` is the axis index on `to_face` that from-face axis
/// `i` maps to. `flip[i]` indicates whether the slot coordinate (and
/// offset) is mirrored (`x → 2-x` for slots, `v → 1-v` for offset).
///
/// The r axis (2) always maps to r (2) unchanged — both faces share
/// the same radial distance from the planet center.
///
/// `entering_sign` is the sign of the overflow-axis value on the new
/// face at the entry edge: `-1` means we enter at the `-1` edge
/// (slot 0), `+1` means we enter at the `+1` edge (slot 2). It
/// determines the offset remap on the overflow axis (see
/// `offset_remap_on_overflow_axis`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Seam {
    pub to_face: Face,
    pub axis_map: [u8; 3],
    pub flip: [bool; 3],
    pub entering_sign: i8,
    pub overflow_axis_old: u8,
}

/// Return the seam crossing for a step that exits `from_face` on
/// local axis `axis` (0=u, 1=v) in direction `dir` (+1 or -1).
///
/// Not valid for axis 2 (r): radial exits aren't seam crossings.
///
/// Derivation: the edge on `from_face` lies at points
/// `n_from + edge_dir + t * other_tangent` for `t ∈ [-1, 1]`, where
/// `edge_dir = dir * (tu or tv)` and `other_tangent` is the
/// perpendicular face tangent. The neighbor `to_face` is the cube
/// face with normal = `edge_dir`. Its own `(u', v')` parameterization
/// of the shared edge solves
/// `u' tu_to + v' tv_to = n_from + t * other_tangent`
/// (since both sides live at `n_to = edge_dir`), giving the axis-map
/// and sign-flip entries by taking dot products with `tu_to` and
/// `tv_to`.
pub fn seam_neighbor(from_face: Face, axis: u8, dir: i8) -> Seam {
    assert!(axis < 2, "seam only valid for u/v (axes 0, 1)");
    assert!(dir == 1 || dir == -1);

    let (tu_from, tv_from) = from_face.tangents();
    let n_from = from_face.normal();

    let edge_dir = match (axis, dir) {
        (0,  1) => tu_from,
        (0, -1) => neg(tu_from),
        (1,  1) => tv_from,
        (1, -1) => neg(tv_from),
        _ => unreachable!(),
    };
    let other_tangent_from = if axis == 0 { tv_from } else { tu_from };

    let to_face = face_from_normal(edge_dir);
    let (tu_to, tv_to) = to_face.tangents();

    // On to_face, solve point = n_to + u'*tu_to + v'*tv_to, where the
    // edge point is n_from + edge_dir + t*other_tangent_from.
    // Since n_to == edge_dir, this reduces to
    //   u'*tu_to + v'*tv_to = n_from + t*other_tangent_from.
    // Orthonormal tu_to/tv_to → dot with each to extract coefficients.
    let u_const = dot(n_from, tu_to);              // constant part of u'
    let u_t_coef = dot(other_tangent_from, tu_to); // t-coefficient of u'
    let v_const = dot(n_from, tv_to);
    let v_t_coef = dot(other_tangent_from, tv_to);

    // Exactly one of (u_const, v_const) is ±1 (the edge constant); the
    // other is 0. Likewise exactly one of (u_t_coef, v_t_coef) is ±1.
    let (entering_axis_to, entering_sign) = if u_const.abs() > 0.5 {
        (0u8, u_const.signum() as i8)
    } else {
        (1u8, v_const.signum() as i8)
    };
    let (other_axis_to, other_sign) = if u_t_coef.abs() > 0.5 {
        (0u8, u_t_coef.signum() as i8)
    } else {
        (1u8, v_t_coef.signum() as i8)
    };
    debug_assert_ne!(entering_axis_to, other_axis_to);

    let overflow_axis_old = axis as usize;
    let other_axis_old = 1 - overflow_axis_old;
    let mut axis_map = [0u8; 3];
    axis_map[overflow_axis_old] = entering_axis_to;
    axis_map[other_axis_old] = other_axis_to;
    axis_map[2] = 2;

    // flip for overflow axis: old slot was (2 if dir=+1 else 0); new
    // slot is (2 if entering_sign=+1 else 0). flip iff they differ.
    let flip_overflow = dir != entering_sign;
    // flip for other axis: old t ∈ [-1, 1] maps to t' = other_sign * t.
    let flip_other = other_sign == -1;

    let mut flip = [false; 3];
    flip[overflow_axis_old] = flip_overflow;
    flip[other_axis_old] = flip_other;
    flip[2] = false;

    Seam {
        to_face,
        axis_map,
        flip,
        entering_sign,
        overflow_axis_old: axis,
    }
}

/// Given the wrapped offset on the overflow axis (always in `[0, 1)`
/// after the `-= 1.0` or `+= 1.0` wrap), compute the offset value on
/// the new face's corresponding axis. See [`Seam::entering_sign`].
///
/// - `entering_sign == -1` → enter new cell from its -edge, moving
///   in +direction → new offset = wrapped.
/// - `entering_sign == +1` → enter from +edge moving in -direction
///   → new offset = 1 - wrapped.
pub fn offset_remap_on_overflow_axis(wrapped: f32, entering_sign: i8) -> f32 {
    if entering_sign == -1 { wrapped } else { 1.0 - wrapped }
}

fn face_from_normal(n: Vec3) -> Face {
    if n[0] > 0.5 { Face::PosX }
    else if n[0] < -0.5 { Face::NegX }
    else if n[1] > 0.5 { Face::PosY }
    else if n[1] < -0.5 { Face::NegY }
    else if n[2] > 0.5 { Face::PosZ }
    else { Face::NegZ }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edge_has_valid_seam() {
        for &f in &Face::ALL {
            for a in 0..2u8 {
                for d in &[-1i8, 1] {
                    let s = seam_neighbor(f, a, *d);
                    assert_eq!(s.axis_map[2], 2);
                    assert!(!s.flip[2]);
                    let (u_to, v_to) = (s.axis_map[0], s.axis_map[1]);
                    assert!(u_to < 2 && v_to < 2);
                    assert_ne!(u_to, v_to);
                    assert_ne!(s.to_face, f);
                    assert!(s.entering_sign == 1 || s.entering_sign == -1);
                }
            }
        }
    }

    /// Spot-check the +X/+u → -Z seam. Derived above:
    /// `u' = -1, v' = v`, so axis_map=[0,1,2] and flip=[true, false, false]
    /// (old us=2 → new us'=0; v preserved).
    #[test]
    fn spot_check_posx_plus_u() {
        let s = seam_neighbor(Face::PosX, 0, 1);
        assert_eq!(s.to_face, Face::NegZ);
        assert_eq!(s.axis_map, [0, 1, 2]);
        assert_eq!(s.flip, [true, false, false]);
        assert_eq!(s.entering_sign, -1);
    }

    /// Spot-check the +X/+v → +Y seam. Derived above:
    /// `u' = 1, v' = u`, so axis_map=[1,0,2] and flip=[false, false, false]
    /// (old us → new vs'=us; old vs=2 → new us'=2).
    #[test]
    fn spot_check_posx_plus_v() {
        let s = seam_neighbor(Face::PosX, 1, 1);
        assert_eq!(s.to_face, Face::PosY);
        assert_eq!(s.axis_map, [1, 0, 2]);
        assert_eq!(s.flip, [false, false, false]);
        assert_eq!(s.entering_sign, 1);
    }

    /// Spot-check +X/-v → -Y: `u'=1, v'=-u`.
    /// axis_map=[1,0,2], flip=[true (v'=-u mirrors u), true (old v=0 → new u=2), false].
    #[test]
    fn spot_check_posx_minus_v() {
        let s = seam_neighbor(Face::PosX, 1, -1);
        assert_eq!(s.to_face, Face::NegY);
        assert_eq!(s.axis_map, [1, 0, 2]);
        assert_eq!(s.flip, [true, true, false]);
        assert_eq!(s.entering_sign, 1);
    }

    /// Seams should be involutive under direction reversal: going from
    /// face F across +u to G, then from G back across the matching
    /// edge, must return to F.
    #[test]
    fn involutive_across_edge() {
        for &f in &Face::ALL {
            for a in 0..2u8 {
                for d in &[-1i8, 1] {
                    let forward = seam_neighbor(f, a, *d);
                    // To come back: on to_face, step on entering axis in
                    // direction that takes us back through the same edge.
                    // The "entering sign" of the forward seam gave the
                    // new-face slot-edge we entered at; reversing means
                    // stepping out that same edge, i.e., dir == entering_sign.
                    let back = seam_neighbor(
                        forward.to_face,
                        forward.axis_map[a as usize],
                        forward.entering_sign,
                    );
                    assert_eq!(back.to_face, f, "round-trip {:?} axis {} dir {}", f, a, d);
                }
            }
        }
    }

    #[test]
    fn face_basis_is_orthonormal() {
        for &f in &Face::ALL {
            let b = face_basis(f);
            assert!((dot(b.u_axis, b.u_axis) - 1.0).abs() < 1e-6);
            assert!((dot(b.v_axis, b.v_axis) - 1.0).abs() < 1e-6);
            assert!((dot(b.r_axis, b.r_axis) - 1.0).abs() < 1e-6);
            assert!(dot(b.u_axis, b.v_axis).abs() < 1e-6);
            assert!(dot(b.u_axis, b.r_axis).abs() < 1e-6);
            assert!(dot(b.v_axis, b.r_axis).abs() < 1e-6);
        }
    }

    #[test]
    fn world_to_local_round_trip() {
        let b = face_basis(Face::PosX);
        let w = [0.3, -0.7, 0.2];
        let l = b.world_to_local(w);
        let w2 = b.local_to_world(l);
        for i in 0..3 {
            assert!((w[i] - w2[i]).abs() < 1e-5);
        }
    }
}
