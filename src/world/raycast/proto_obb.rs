//! CPU port of `assets/shaders/uvsphere/proto_block.wgsl`. The Rust
//! implementation must match the WGSL bit-for-bit; if a unit test
//! here passes but the screen is blank, the bug is in WGSL
//! (compilation, dispatch, or callsite) — not in the OBB math.
//!
//! Use this to:
//! - Verify a target cell's world bounds (centre, basis, half-
//!   extents) before placing an OBB.
//! - Confirm a given camera position + ray direction actually
//!   intersects the target OBB at all (catches the "cube floats
//!   in invisible space" failure mode).
//! - Pin numerical baselines that the WGSL must reproduce.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbHit {
    pub t: f32,
    pub axis: u32, // 0 = φ̂, 1 = θ̂, 2 = r̂
    pub side: u32, // 0 = `-h` face, 1 = `+h` face
}

/// Body-frame UV bounds of the prototype's hardcoded target cell.
/// Mirrors `PROTO_TARGET_*` constants in
/// `assets/shaders/uvsphere/proto_block.wgsl`. Used by the CPU-side
/// raycast intercept (`cpu_raycast_uv_body`) so cursor breaks land on
/// the same volume the shader paints — without depending on the
/// edit-depth budget being deep enough to walk all the way to body
/// path `[14, 21, 23]`.
pub const PROTO_PHI_LO: f32 = 4.654;
pub const PROTO_PHI_HI: f32 = 4.887;
pub const PROTO_THETA_LO: f32 = -0.052;
pub const PROTO_THETA_HI: f32 = 0.052;
pub const PROTO_R_LO: f32 = 0.4333;
pub const PROTO_R_HI: f32 = 0.45;

/// Slot path from a `UvSphereBody` root down to the prototype's
/// target cell (= body-tree depth 3, top of the grass shell).
pub const PROTO_BODY_PATH: [usize; 3] = [14, 21, 23];

/// World-frame OBB derived from a UV cell's spherical bounds.
#[derive(Clone, Copy, Debug)]
pub struct CellObb {
    pub center: [f32; 3],
    pub r_hat: [f32; 3],
    pub theta_hat: [f32; 3],
    pub phi_hat: [f32; 3],
    pub half_phi: f32,
    pub half_th: f32,
    pub half_r: f32,
}

/// Build an OBB matching the WGSL's `proto_ray_vs_obb` setup. `body_center`
/// is `(1.5, 1.5, 1.5)` for the demo body. `phi_lo`/`hi` etc. are the
/// cell's UV bounds.
pub fn cell_obb(
    body_center: [f32; 3],
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> CellObb {
    let phi_c = 0.5 * (phi_lo + phi_hi);
    let theta_c = 0.5 * (theta_lo + theta_hi);
    let r_c = 0.5 * (r_lo + r_hi);
    let cos_p = phi_c.cos();
    let sin_p = phi_c.sin();
    let cos_t = theta_c.cos();
    let sin_t = theta_c.sin();

    let r_hat = [cos_t * cos_p, sin_t, cos_t * sin_p];
    let theta_hat = [-sin_t * cos_p, cos_t, -sin_t * sin_p];
    let phi_hat = [-sin_p, 0.0, cos_p];

    let center = [
        body_center[0] + r_hat[0] * r_c,
        body_center[1] + r_hat[1] * r_c,
        body_center[2] + r_hat[2] * r_c,
    ];

    let half_phi = 0.5 * (phi_hi - phi_lo) * r_c * cos_t;
    let half_th = 0.5 * (theta_hi - theta_lo) * r_c;
    let half_r = 0.5 * (r_hi - r_lo);

    CellObb {
        center,
        r_hat,
        theta_hat,
        phi_hat,
        half_phi,
        half_th,
        half_r,
    }
}

/// Slab method ray-vs-OBB. Mirrors `proto_ray_vs_obb` in WGSL.
/// Returns `None` on miss. Returned `t` is in the same units as
/// `ray_dir` (world distance per unit ray-dir).
pub fn ray_vs_obb(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    obb: &CellObb,
) -> Option<ObbHit> {
    let to_origin = [
        ray_origin[0] - obb.center[0],
        ray_origin[1] - obb.center[1],
        ray_origin[2] - obb.center[2],
    ];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let q0 = dot(to_origin, obb.phi_hat);
    let q1 = dot(to_origin, obb.theta_hat);
    let q2 = dot(to_origin, obb.r_hat);
    let d0 = dot(ray_dir, obb.phi_hat);
    let d1 = dot(ray_dir, obb.theta_hat);
    let d2 = dot(ray_dir, obb.r_hat);

    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    let mut enter_axis = 0u32;
    let mut enter_side = 0u32;

    let slab = |q: f32, d: f32, h: f32, axis: u32,
                t_min: &mut f32, t_max: &mut f32,
                enter_axis: &mut u32, enter_side: &mut u32| -> bool {
        if d.abs() < 1e-12 {
            if q.abs() > h {
                return false;
            }
            return true;
        }
        let inv_d = 1.0 / d;
        let mut t_a = (-h - q) * inv_d;
        let mut t_b = (h - q) * inv_d;
        let mut sa = 0u32;
        if t_a > t_b {
            std::mem::swap(&mut t_a, &mut t_b);
            sa = 1u32;
        }
        if t_a > *t_min {
            *t_min = t_a;
            *enter_axis = axis;
            *enter_side = sa;
        }
        if t_b < *t_max {
            *t_max = t_b;
        }
        *t_min <= *t_max
    };

    if !slab(q0, d0, obb.half_phi, 0, &mut t_min, &mut t_max, &mut enter_axis, &mut enter_side)
        || !slab(q1, d1, obb.half_th, 1, &mut t_min, &mut t_max, &mut enter_axis, &mut enter_side)
        || !slab(q2, d2, obb.half_r, 2, &mut t_min, &mut t_max, &mut enter_axis, &mut enter_side)
    {
        return None;
    }
    if t_max < 1e-4 {
        return None;
    }
    Some(ObbHit {
        t: t_min.max(1e-4),
        axis: enter_axis,
        side: enter_side,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAU: f32 = std::f32::consts::TAU;

    /// Body-frame centre (matches `body_size = 3.0` shader convention).
    const BC: [f32; 3] = [1.5, 1.5, 1.5];

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn vec_approx_eq(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        approx_eq(a[0], b[0], eps) && approx_eq(a[1], b[1], eps) && approx_eq(a[2], b[2], eps)
    }

    fn norm(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Target = the prototype's hardcoded cell. Bounds reproduced
    /// from `proto_block.wgsl` (path [14, 21, 23] — depth-3 cell at
    /// the outermost r-slot, lying on the grass band so the OBB is
    /// visible on the body's silhouette).
    fn proto_target_obb() -> CellObb {
        cell_obb(
            BC,
            4.654, 4.887,
            -0.052, 0.052,
            0.4333, 0.45,
        )
    }

    #[test]
    fn obb_center_in_body_frame() {
        let obb = proto_target_obb();
        // φ_c ≈ 4.7705 — south face slightly east. World position:
        //   x = 1.5 + 0.4417 · cos(0) · cos(4.7705)
        //   z = 1.5 + 0.4417 · cos(0) · sin(4.7705)
        let r_c: f32 = 0.5 * (0.4333 + 0.45);
        let phi_c: f32 = 0.5 * (4.654 + 4.887);
        let expected_x = 1.5 + r_c * phi_c.cos();
        let expected_z = 1.5 + r_c * phi_c.sin();
        assert!(approx_eq(obb.center[0], expected_x, 1e-5),
            "x: expected {}, got {}", expected_x, obb.center[0]);
        assert!(approx_eq(obb.center[1], 1.5, 1e-5));
        assert!(approx_eq(obb.center[2], expected_z, 1e-5),
            "z: expected {}, got {}", expected_z, obb.center[2]);
        // Sanity: the cell's centre lies INSIDE the body shell, and
        // specifically in the grass band r ∈ [0.4275, 0.45].
        let off = [
            obb.center[0] - BC[0],
            obb.center[1] - BC[1],
            obb.center[2] - BC[2],
        ];
        let dist = norm(off);
        assert!(dist > 0.4275 - 1e-3 && dist < 0.45 + 1e-3,
            "centre should be in the grass band r ∈ [0.4275, 0.45], got dist {}", dist);
    }

    #[test]
    fn obb_basis_is_orthonormal() {
        let obb = proto_target_obb();
        for v in [obb.r_hat, obb.theta_hat, obb.phi_hat] {
            assert!(approx_eq(norm(v), 1.0, 1e-5),
                "basis vector not unit-length: {:?}", v);
        }
        assert!(approx_eq(dot(obb.r_hat, obb.theta_hat), 0.0, 1e-5));
        assert!(approx_eq(dot(obb.r_hat, obb.phi_hat), 0.0, 1e-5));
        assert!(approx_eq(dot(obb.theta_hat, obb.phi_hat), 0.0, 1e-5));
    }

    #[test]
    fn obb_extents_match_cell_arc_length() {
        let obb = proto_target_obb();
        let r_c = 0.5 * (0.4333 + 0.45);
        // half_phi: arc length r·cos(θ_c)·Δφ/2
        let expected_half_phi = 0.5 * (4.887 - 4.654) * r_c * 0.0_f32.cos();
        let expected_half_th = 0.5 * (0.052 - (-0.052)) * r_c;
        let expected_half_r = 0.5 * (0.45 - 0.4333);
        assert!(approx_eq(obb.half_phi, expected_half_phi, 1e-5));
        assert!(approx_eq(obb.half_th, expected_half_th, 1e-5));
        assert!(approx_eq(obb.half_r, expected_half_r, 1e-5));
    }

    #[test]
    fn ray_aimed_at_center_hits() {
        let obb = proto_target_obb();
        // Camera far back, ray pointed exactly at OBB centre.
        let cam = [obb.center[0], obb.center[1], obb.center[2] - 5.0];
        let dir = [0.0, 0.0, 1.0];
        let hit = ray_vs_obb(cam, dir, &obb).expect("aim-at-centre must hit");
        assert!(hit.t > 0.0);
        assert!(hit.t < 5.0);
    }

    #[test]
    fn ray_aimed_far_off_misses() {
        let obb = proto_target_obb();
        // Camera at body centre, ray pointing far above the body.
        let cam = BC;
        let dir = [0.0, 1.0, 0.0];
        let hit = ray_vs_obb(cam, dir, &obb);
        // OBB is at south face — ray pointing +y misses it.
        assert!(hit.is_none(), "ray pointing away should miss, got {:?}", hit);
    }

    /// Regression: the screenshot-test camera at body-marcher local
    /// `(1.5, 1.5, -1.5)` going `+z` (= world `(1.5, 1.5, 0.5)` →
    /// `+z` after the `march_cartesian → march_uv_sphere` body-frame
    /// rescale by 3.0) actually intersects the prototype OBB.
    #[test]
    fn proto_obb_visible_from_screenshot_camera() {
        let obb = proto_target_obb();
        // After `march_cartesian` rescales the ray into the body's
        // local [0, 3]^3 frame: world (1.5, 1.5, 0.5) → body-local
        // (1.5, 1.5, -1.5); world dir (0, 0, 1) → body-local (0, 0, 3).
        let cam = [1.5_f32, 1.5, -1.5];
        let dir = [0.0_f32, 0.0, 3.0];
        let hit = ray_vs_obb(cam, dir, &obb);
        assert!(
            hit.is_some(),
            "OBB invisible from screenshot camera:\n\
             camera_body_local={:?}\n\
             dir_body_local={:?}\n\
             OBB centre={:?}, half_extents=({:.3}, {:.3}, {:.3})",
            cam, dir,
            obb.center, obb.half_phi, obb.half_th, obb.half_r,
        );
    }
}
