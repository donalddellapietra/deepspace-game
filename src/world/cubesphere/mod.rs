//! Cubed-sphere geometry primitives.
//!
//! Stage 0a of the unified-DDA rewrite — CPU-side scaffolding only.
//! No renderer, shader, or tree-kind changes here. This module
//! exposes:
//!
//! - [`Face`] — the six cube faces and their orthonormal basis.
//! - [`FACE_SLOTS`], [`CORE_SLOT`] — indices into a 27-slot body cell.
//! - [`ea_to_cube`] / [`cube_to_ea`] — equal-angle warp and its inverse.
//! - [`face_uv_to_dir`] — unit direction from `(face, u, v)`.
//! - [`FacePoint`] — a point in face-normalized `(un, vn, rn) ∈ [0, 1]³`.
//! - [`body_point_to_face_space`] / [`face_space_to_body_point`] —
//!   round-trip between the body's local `[0, body_size)³` frame and
//!   face-normalized coordinates on a sphere-shell cell.
//! - [`seams`] submodule — the 24-entry seam-crossing rotation table.
//!
//! All per-point computation stays O(1) in magnitude: no absolute
//! coordinates at deep depth, no unbounded scales. Inputs that happen
//! to be `f32` (sampled ray positions) compose in `f64` internally
//! and cast back on return so tiny face-geometry ops don't lose
//! precision from repeated round-trips.

use super::sdf::Vec3;
use super::tree::slot_index;

pub mod seams;

pub use seams::{seam_rotation, SeamTransition, SEAM_TABLE};

// ──────────────────────────────────────────────────────────── Face

/// One of the six cube faces.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];

    /// Construct a face from its `as usize` discriminant. Panics
    /// on out-of-range input — callers are expected to pass a
    /// known-good index (e.g. an iteration variable over `0..6`).
    #[inline]
    pub fn from_index(i: u8) -> Face {
        match i {
            0 => Face::PosX,
            1 => Face::NegX,
            2 => Face::PosY,
            3 => Face::NegY,
            4 => Face::PosZ,
            5 => Face::NegZ,
            _ => panic!("invalid face index {i}"),
        }
    }

    /// Map a 27-slot body-cell index back to the face whose subtree
    /// lives at that slot, if any. The 6 face slots are at
    /// [`FACE_SLOTS`]; the remaining 21 slots (core + 20 edge/corner
    /// fillers) return `None`.
    #[inline]
    pub fn from_body_slot(slot: usize) -> Option<Face> {
        FACE_SLOTS
            .iter()
            .position(|&s| s == slot)
            .map(|i| Face::from_index(i as u8))
    }

    /// Outward body-space normal of the face (unit vector).
    #[inline]
    pub fn normal(self) -> Vec3 {
        match self {
            Face::PosX => [1.0, 0.0, 0.0],
            Face::NegX => [-1.0, 0.0, 0.0],
            Face::PosY => [0.0, 1.0, 0.0],
            Face::NegY => [0.0, -1.0, 0.0],
            Face::PosZ => [0.0, 0.0, 1.0],
            Face::NegZ => [0.0, 0.0, -1.0],
        }
    }

    /// Returns `(u_axis, v_axis)` — the two tangent unit vectors
    /// forming the face's orthonormal 2-frame. `u_axis × v_axis =
    /// n_axis`, i.e. `(u, v, n)` is a right-handed basis.
    #[inline]
    pub fn tangents(self) -> (Vec3, Vec3) {
        match self {
            Face::PosX => ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            Face::NegX => ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            Face::PosY => ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            Face::NegY => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            Face::PosZ => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            Face::NegZ => ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        }
    }

    /// The 3×3 rotation from body-XYZ to this face's `(u, v, n)`
    /// basis. Rows are `u_axis`, `v_axis`, `n_axis` — so
    /// `R · body_vec = face_vec`.
    #[inline]
    pub fn basis(self) -> [[f32; 3]; 3] {
        let (u, v) = self.tangents();
        let n = self.normal();
        [u, v, n]
    }
}

/// Slot in a body cell's 27-grid that holds each face's subtree,
/// indexed by `Face as usize`. The faces attach at the center of
/// each outer 3×3 face of the body cube.
pub const FACE_SLOTS: [usize; 6] = [
    slot_index(2, 1, 1), // PosX
    slot_index(0, 1, 1), // NegX
    slot_index(1, 2, 1), // PosY
    slot_index(1, 0, 1), // NegY
    slot_index(1, 1, 2), // PosZ
    slot_index(1, 1, 0), // NegZ
];

/// Slot in a body cell's 27-grid that holds the interior core
/// subtree (below `inner_r`). The cell center at `(1, 1, 1)`.
pub const CORE_SLOT: usize = slot_index(1, 1, 1);

// ─────────────────────────────────────── Equal-angle face mapping

/// Equal-angle → cube warp. Input `x ∈ [-1, 1]` is an equal-angle
/// face coordinate (so that equal `Δx` steps span equal angular
/// wedges on the sphere); output is the un-warped cube-face
/// coordinate in `[-1, 1]`.
///
/// `ea_to_cube(x) = tan(x · π/4)`.
#[inline]
pub fn ea_to_cube(x: f32) -> f32 {
    (x as f64 * std::f64::consts::FRAC_PI_4).tan() as f32
}

/// Inverse of [`ea_to_cube`]: cube-face coord `[-1, 1]` → equal-angle
/// coord `[-1, 1]`. `cube_to_ea(c) = atan(c) · 4/π`.
#[inline]
pub fn cube_to_ea(c: f32) -> f32 {
    ((c as f64).atan() * (4.0 / std::f64::consts::PI)) as f32
}

/// Convert `(face, u, v)` with equal-angle face coords `u, v ∈
/// [-1, 1]` into a unit direction vector from the sphere centre.
#[inline]
pub fn face_uv_to_dir(face: Face, u: f32, v: f32) -> Vec3 {
    let cu = ea_to_cube(u) as f64;
    let cv = ea_to_cube(v) as f64;
    let n = face.normal();
    let (ua, va) = face.tangents();
    let px = n[0] as f64 + cu * ua[0] as f64 + cv * va[0] as f64;
    let py = n[1] as f64 + cu * ua[1] as f64 + cv * va[1] as f64;
    let pz = n[2] as f64 + cu * ua[2] as f64 + cv * va[2] as f64;
    let len = (px * px + py * py + pz * pz).sqrt();
    // len is ≥ 1 (equal exactly at face centre), never near zero.
    let inv = 1.0 / len;
    [(px * inv) as f32, (py * inv) as f32, (pz * inv) as f32]
}

// ───────────────────────────────────────────────── Face-space point

/// A point on a sphere-shell cell expressed in face-normalized
/// coordinates. All three components lie in `[0, 1]`:
/// - `un, vn`: equal-angle face coords mapped to `[0, 1]`
///   (`un = 0.5 · (u_ea + 1)`).
/// - `rn`: radial coordinate, `0` at inner shell, `1` at outer shell.
///
/// These are the face-frame coordinates used by the unified DDA
/// when it's descending a face subtree; multiplying by `3` gives
/// the `[0, 3)` residual the DDA operates on.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FacePoint {
    pub face: Face,
    pub un: f32,
    pub vn: f32,
    pub rn: f32,
}

/// Project a body-local `point ∈ [0, body_size)³` onto the
/// cubed-sphere shell `[inner_r, outer_r]` and return its
/// face-normalized coordinates.
///
/// The body cell is assumed centered at `(0.5 · body_size,) * 3`.
/// Returns `None` if the point is at the center (zero radius) or
/// degenerate in any other way that prevents a stable projection.
///
/// The returned `un, vn, rn` are clamped to `[0, 1]` — callers
/// that need to detect out-of-shell points should check `rn` at
/// their own tolerance.
pub fn body_point_to_face_space(
    point: Vec3,
    inner_r: f32,
    outer_r: f32,
    body_size: f32,
) -> Option<FacePoint> {
    debug_assert!(
        inner_r >= 0.0 && inner_r < outer_r && body_size > 0.0,
        "expected 0 ≤ inner_r < outer_r, body_size > 0 (got {inner_r}, {outer_r}, {body_size})"
    );

    let half = 0.5 * body_size as f64;
    let dx = point[0] as f64 - half;
    let dy = point[1] as f64 - half;
    let dz = point[2] as f64 - half;
    let r2 = dx * dx + dy * dy + dz * dz;
    if r2 < 1.0e-24 {
        return None;
    }
    let r = r2.sqrt();
    let inv = 1.0 / r;
    let nx = dx * inv;
    let ny = dy * inv;
    let nz = dz * inv;

    let ax = nx.abs();
    let ay = ny.abs();
    let az = nz.abs();
    let (face, cube_u, cube_v) = if ax >= ay && ax >= az {
        if nx > 0.0 {
            (Face::PosX, -nz / ax, ny / ax)
        } else {
            (Face::NegX, nz / ax, ny / ax)
        }
    } else if ay >= az {
        if ny > 0.0 {
            (Face::PosY, nx / ay, -nz / ay)
        } else {
            (Face::NegY, nx / ay, nz / ay)
        }
    } else if nz > 0.0 {
        (Face::PosZ, nx / az, ny / az)
    } else {
        (Face::NegZ, -nx / az, ny / az)
    };

    let u_ea = (cube_u.atan() * (4.0 / std::f64::consts::PI)) as f32;
    let v_ea = (cube_v.atan() * (4.0 / std::f64::consts::PI)) as f32;
    let un = (0.5 * (u_ea + 1.0)).clamp(0.0, 1.0);
    let vn = (0.5 * (v_ea + 1.0)).clamp(0.0, 1.0);

    let inner = inner_r as f64;
    let outer = outer_r as f64;
    let rn = ((r - inner) / (outer - inner)) as f32;
    let rn = rn.clamp(0.0, 1.0);

    Some(FacePoint { face, un, vn, rn })
}

/// Inverse of [`body_point_to_face_space`]: take face-normalized
/// coordinates and return the body-local point `∈ [0, body_size)³`.
pub fn face_space_to_body_point(
    face: Face,
    un: f32,
    vn: f32,
    rn: f32,
    inner_r: f32,
    outer_r: f32,
    body_size: f32,
) -> Vec3 {
    let u_ea = 2.0 * un - 1.0;
    let v_ea = 2.0 * vn - 1.0;
    let cu = (u_ea as f64 * std::f64::consts::FRAC_PI_4).tan();
    let cv = (v_ea as f64 * std::f64::consts::FRAC_PI_4).tan();
    let n = face.normal();
    let (ua, va) = face.tangents();
    let px = n[0] as f64 + cu * ua[0] as f64 + cv * va[0] as f64;
    let py = n[1] as f64 + cu * ua[1] as f64 + cv * va[1] as f64;
    let pz = n[2] as f64 + cu * ua[2] as f64 + cv * va[2] as f64;
    let len = (px * px + py * py + pz * pz).sqrt();
    let inv = 1.0 / len;
    let dirx = px * inv;
    let diry = py * inv;
    let dirz = pz * inv;

    let inner = inner_r as f64;
    let outer = outer_r as f64;
    let r = inner + (outer - inner) * rn as f64;

    let half = 0.5 * body_size as f64;
    [
        (half + r * dirx) as f32,
        (half + r * diry) as f32,
        (half + r * dirz) as f32,
    ]
}

// ────────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }
    fn approx_v(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx(a[0], b[0], eps) && approx(a[1], b[1], eps) && approx(a[2], b[2], eps)
    }

    #[test]
    fn face_center_matches_normal() {
        for &f in &Face::ALL {
            let dir = face_uv_to_dir(f, 0.0, 0.0);
            assert!(
                approx_v(dir, f.normal(), 1e-6),
                "face {f:?}: dir {:?} != normal {:?}",
                dir,
                f.normal()
            );
        }
    }

    #[test]
    fn face_corners_are_unit_vectors() {
        for &f in &Face::ALL {
            for &(u, v) in &[
                (1.0, 1.0),
                (-1.0, 1.0),
                (1.0, -1.0),
                (-1.0, -1.0),
                (0.5, -0.3),
                (0.0, 0.0),
            ] {
                let d = face_uv_to_dir(f, u, v);
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!(
                    approx(len, 1.0, 1e-6),
                    "face {f:?} at ({u}, {v}): len {len}"
                );
            }
        }
    }

    #[test]
    fn face_from_body_slot_round_trip() {
        for &f in &Face::ALL {
            let slot = FACE_SLOTS[f as usize];
            assert_eq!(Face::from_body_slot(slot), Some(f));
        }
        // Core + the 20 non-face/non-core slots should all be None.
        assert_eq!(Face::from_body_slot(CORE_SLOT), None);
        for slot in 0..27 {
            if FACE_SLOTS.contains(&slot) {
                continue;
            }
            assert_eq!(Face::from_body_slot(slot), None, "slot {slot}");
        }
    }

    #[test]
    fn ea_cube_round_trip() {
        let mut x = -0.9f32;
        while x <= 0.9 + 1.0e-6 {
            let c = ea_to_cube(x);
            let back = cube_to_ea(c);
            assert!(
                approx(back, x, 1e-6),
                "ea_to_cube({x}) = {c}, cube_to_ea(c) = {back}"
            );
            x += 0.1;
        }
    }

    #[test]
    fn face_basis_is_right_handed() {
        for &f in &Face::ALL {
            let (u, v) = f.tangents();
            let n = f.normal();
            // u × v must equal n (right-handed).
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            assert!(
                approx_v(cross, n, 1e-6),
                "face {f:?}: u × v = {cross:?}, expected n = {n:?}"
            );
        }
    }

    #[test]
    fn body_face_space_round_trip() {
        let body_size = 3.0f32;
        let inner_r = 0.4f32;
        let outer_r = 1.2f32;
        // Interior samples on each face, spread across (un, vn, rn).
        let samples: &[(f32, f32, f32)] = &[
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 1.0),
            (0.5, 0.5, 0.5),
            (0.1, 0.9, 0.25),
            (0.9, 0.1, 0.75),
            (0.3, 0.7, 0.5),
        ];
        for &f in &Face::ALL {
            for &(un, vn, rn) in samples {
                let p = face_space_to_body_point(f, un, vn, rn, inner_r, outer_r, body_size);
                let back = body_point_to_face_space(p, inner_r, outer_r, body_size)
                    .expect("round-trip point is non-degenerate");
                assert_eq!(back.face, f, "face {f:?} at ({un}, {vn}, {rn}) → {:?}", back.face);
                assert!(
                    approx(back.un, un, 1e-5),
                    "face {f:?} un: {un} → {}",
                    back.un
                );
                assert!(
                    approx(back.vn, vn, 1e-5),
                    "face {f:?} vn: {vn} → {}",
                    back.vn
                );
                assert!(
                    approx(back.rn, rn, 1e-5),
                    "face {f:?} rn: {rn} → {}",
                    back.rn
                );
            }
        }
    }
}
