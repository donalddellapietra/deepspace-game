//! Cube-to-sphere mapping (Math Proofs / Nowell formula).
//!
//! Forward:  F  : (-1, 1)^3 → open unit ball.
//! Inverse:  F^-1: open unit ball → (-1, 1)^3, via Newton on F(c) = w.
//! Jacobian: analytic, closed form.
//!
//! Pure f32-array math — no external deps. Written this way so the
//! same code can be ported directly to WGSL for CPU/GPU parity.
//!
//! Structural property (tested below): F is smooth on the closed cube
//! but its Jacobian degenerates at the 8 cube corners and 12 cube
//! edges (σ_min(J) → 0 there). Cartesian voxels that live strictly in
//! the open interior are fine for curved-space sphere-tracing; the
//! sphere-trace step-size formula must handle σ_min → 0 near the
//! cube boundary by falling back to a small Euler step or by clipping
//! the usable cube region slightly inside the boundary.

pub type V3 = [f32; 3];

// ---------- forward ----------

/// Forward map c → w. Cube [-1, 1]^3 → closed unit ball.
#[inline]
pub fn forward(c: V3) -> V3 {
    let [x, y, z] = c;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let sx = (1.0 - 0.5 * y2 - 0.5 * z2 + y2 * z2 / 3.0).max(0.0).sqrt();
    let sy = (1.0 - 0.5 * z2 - 0.5 * x2 + z2 * x2 / 3.0).max(0.0).sqrt();
    let sz = (1.0 - 0.5 * x2 - 0.5 * y2 + x2 * y2 / 3.0).max(0.0).sqrt();
    [x * sx, y * sy, z * sz]
}

// ---------- jacobian ----------

/// Analytic Jacobian J = ∂F/∂c at c. Row-major 3x3: j[row][col].
///
/// Derivation:
///   s_i = sqrt(1 − c_j²/2 − c_k²/2 + c_j² c_k² / 3)
///   F_i = c_i · s_i
///   ∂F_i/∂c_i = s_i
///   ∂F_i/∂c_j = c_i · c_j · (2 c_k² / 3 − 1) / (2 s_i)   (j ≠ i, k = the third axis)
#[inline]
pub fn jacobian(c: V3) -> [[f32; 3]; 3] {
    let [x, y, z] = c;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    // floor s_i away from zero to avoid div-by-zero in the off-diagonal terms;
    // at cube corners / edges where s_i → 0, the off-diag numerator also → 0
    // so the value stays finite, but we still want to be numerically safe.
    let sx = (1.0 - 0.5 * y2 - 0.5 * z2 + y2 * z2 / 3.0).max(1e-20).sqrt();
    let sy = (1.0 - 0.5 * z2 - 0.5 * x2 + z2 * x2 / 3.0).max(1e-20).sqrt();
    let sz = (1.0 - 0.5 * x2 - 0.5 * y2 + x2 * y2 / 3.0).max(1e-20).sqrt();

    // row 0: ∂F.x/∂(x, y, z)
    let fxy = x * y * (2.0 * z2 / 3.0 - 1.0) / (2.0 * sx);
    let fxz = x * z * (2.0 * y2 / 3.0 - 1.0) / (2.0 * sx);
    // row 1: ∂F.y/∂(x, y, z)
    let fyx = y * x * (2.0 * z2 / 3.0 - 1.0) / (2.0 * sy);
    let fyz = y * z * (2.0 * x2 / 3.0 - 1.0) / (2.0 * sy);
    // row 2: ∂F.z/∂(x, y, z)
    let fzx = z * x * (2.0 * y2 / 3.0 - 1.0) / (2.0 * sz);
    let fzy = z * y * (2.0 * x2 / 3.0 - 1.0) / (2.0 * sz);

    [
        [sx, fxy, fxz],
        [fyx, sy, fyz],
        [fzx, fzy, sz],
    ]
}

// ---------- small 3x3 helpers ----------

#[inline]
fn det3(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solve M · x = b for a 3x3 system. Returns None if near-singular.
#[inline]
pub fn solve3(m: [[f32; 3]; 3], b: V3) -> Option<V3> {
    let d = det3(m);
    if d.abs() < 1e-18 {
        return None;
    }
    // cofactor matrix C; x = C^T · b / det
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let inv_d = 1.0 / d;
    Some([
        inv_d * (c00 * b[0] + c10 * b[1] + c20 * b[2]),
        inv_d * (c01 * b[0] + c11 * b[1] + c21 * b[2]),
        inv_d * (c02 * b[0] + c12 * b[1] + c22 * b[2]),
    ])
}

#[inline]
pub fn mat_mul_vec(m: [[f32; 3]; 3], v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Transpose a 3x3.
#[inline]
pub fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

// ---------- inverse (Newton) ----------

/// F^-1(w) via Newton iteration from `start`. `max_iter` ≥ 4 recommended.
/// Returns None if diverges or if w is outside the open unit ball.
pub fn inverse_from(w: V3, start: V3, max_iter: u32) -> Option<V3> {
    let r2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if r2 > 1.0 + 1e-4 {
        return None;
    }
    let mut c = start;
    for _ in 0..max_iter {
        let f = forward(c);
        let r = [f[0] - w[0], f[1] - w[1], f[2] - w[2]];
        let rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        if rr < 1e-14 {
            return Some(c);
        }
        let j = jacobian(c);
        let Some(delta) = solve3(j, [-r[0], -r[1], -r[2]]) else {
            return None;
        };
        c[0] += delta[0];
        c[1] += delta[1];
        c[2] += delta[2];
        // guard: Newton diverged out of the extended cube
        if c[0].abs() > 1.5 || c[1].abs() > 1.5 || c[2].abs() > 1.5 {
            return None;
        }
    }
    // final residual tolerance
    let f = forward(c);
    let err2 = (f[0] - w[0]).powi(2) + (f[1] - w[1]).powi(2) + (f[2] - w[2]).powi(2);
    if err2 < 1e-8 { Some(c) } else { None }
}

/// F^-1(w), cold start from `w` itself (good near origin; Newton handles the rest).
pub fn inverse(w: V3, max_iter: u32) -> Option<V3> {
    inverse_from(w, w, max_iter)
}

// ---------- singular values of J ----------

/// Smallest eigenvalue of a 3x3 symmetric matrix, closed form.
/// Standard trigonometric solution (Smith 1961).
fn symm3_eig_min(a: [[f32; 3]; 3]) -> f32 {
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if p1 < 1e-20 {
        return a[0][0].min(a[1][1]).min(a[2][2]);
    }
    let tr = a[0][0] + a[1][1] + a[2][2];
    let q = tr / 3.0;
    let p2 = (a[0][0] - q).powi(2)
        + (a[1][1] - q).powi(2)
        + (a[2][2] - q).powi(2)
        + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let b = [
        [(a[0][0] - q) / p, a[0][1] / p, a[0][2] / p],
        [a[1][0] / p, (a[1][1] - q) / p, a[1][2] / p],
        [a[2][0] / p, a[2][1] / p, (a[2][2] - q) / p],
    ];
    let r = (det3(b) / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    // eigenvalues: q + 2p·cos(phi + k·2π/3), k=0,1,2. Smallest is k=2.
    q + 2.0 * p * (phi + 2.0 * std::f32::consts::PI / 3.0).cos()
}

/// σ_min(J) — smallest singular value of the Jacobian at c. Used by
/// sphere-tracing to derive a safe world-space step size from a
/// cube-space safe distance.
///
/// Returns 0 at the 8 cube corners and along the 12 cube edges
/// (Jacobian degenerates there).
pub fn sigma_min(c: V3) -> f32 {
    let j = jacobian(c);
    // A = Jᵀ J  (symmetric, PSD)
    let mut a = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            let mut s = 0.0;
            for r in 0..3 {
                s += j[r][i] * j[r][k];
            }
            a[i][k] = s;
        }
    }
    symm3_eig_min(a).max(0.0).sqrt()
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn v3_dist(a: V3, b: V3) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// deterministic f32 in [-1, 1)
    fn det_f32(seed: u64) -> f32 {
        // splitmix64
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s ^= s >> 30;
        s = s.wrapping_mul(0xbf58476d1ce4e5b9);
        s ^= s >> 27;
        s = s.wrapping_mul(0x94d049bb133111eb);
        s ^= s >> 31;
        // Use 23 mantissa bits to make a float in [1, 2), subtract 1 → [0, 1)
        let bits = ((s as u32) & 0x007F_FFFF) | 0x3F80_0000;
        let x = f32::from_bits(bits) - 1.0; // [0, 1)
        2.0 * x - 1.0 // [-1, 1)
    }

    fn sample(seed: u64, r: f32) -> V3 {
        [
            r * det_f32(seed.wrapping_mul(3).wrapping_add(11)),
            r * det_f32(seed.wrapping_mul(5).wrapping_add(17)),
            r * det_f32(seed.wrapping_mul(7).wrapping_add(23)),
        ]
    }

    #[test]
    fn axis_identity() {
        // On each coordinate axis, F is identity.
        for t in [-0.99_f32, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99] {
            let fx = forward([t, 0.0, 0.0]);
            assert!(v3_dist(fx, [t, 0.0, 0.0]) < 1e-6, "x-axis t={t}: {:?}", fx);
            let fy = forward([0.0, t, 0.0]);
            assert!(v3_dist(fy, [0.0, t, 0.0]) < 1e-6, "y-axis t={t}: {:?}", fy);
            let fz = forward([0.0, 0.0, t]);
            assert!(v3_dist(fz, [0.0, 0.0, t]) < 1e-6, "z-axis t={t}: {:?}", fz);
        }
    }

    #[test]
    fn cube_faces_map_onto_unit_sphere() {
        // Points on any cube face (one coord = ±1) must land on |w| = 1.
        for t in [-0.9_f32, -0.5, 0.0, 0.5, 0.9] {
            for s in [-0.9_f32, -0.5, 0.0, 0.5, 0.9] {
                for (axis, sign) in [(0, 1.0_f32), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)] {
                    let c = match axis {
                        0 => [sign, t, s],
                        1 => [t, sign, s],
                        2 => [t, s, sign],
                        _ => unreachable!(),
                    };
                    let w = forward(c);
                    let mag = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
                    assert!(
                        (mag - 1.0).abs() < 1e-5,
                        "cube face c={:?} → w={:?}, |w|={:.7}",
                        c,
                        w,
                        mag
                    );
                }
            }
        }
    }

    #[test]
    fn forward_is_bounded_by_unit_ball() {
        for seed in 0..5000 {
            let c = sample(seed, 1.0);
            let w = forward(c);
            let mag2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            assert!(
                mag2 <= 1.0 + 1e-4,
                "F({:?}) escapes the unit ball: |w|²={mag2:.6}",
                c
            );
        }
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let h = 1e-3_f32;
        for seed in 0..500 {
            let c = sample(seed, 0.9);
            let j_an = jacobian(c);
            for col in 0..3 {
                let mut cp = c;
                cp[col] += h;
                let mut cm = c;
                cm[col] -= h;
                let fp = forward(cp);
                let fm = forward(cm);
                for row in 0..3 {
                    let fd = (fp[row] - fm[row]) / (2.0 * h);
                    let err = (j_an[row][col] - fd).abs();
                    assert!(
                        err < 2e-3,
                        "J[{row}][{col}] at c={:?}: analytic={:.6}, fd={:.6}, err={:.2e}",
                        c,
                        j_an[row][col],
                        fd,
                        err
                    );
                }
            }
        }
    }

    #[test]
    fn forward_inverse_roundtrip_interior() {
        // Interior of cube: Newton must converge and recover c.
        let mut worst_err = 0.0_f32;
        for seed in 0..2000 {
            let c = sample(seed, 0.9);
            let w = forward(c);
            let c_back = inverse(w, 8).unwrap_or_else(|| {
                panic!("inverse failed: c={:?}, w={:?}", c, w);
            });
            let err = v3_dist(c, c_back);
            if err > worst_err {
                worst_err = err;
            }
            assert!(
                err < 1e-4,
                "roundtrip c={:?} → w={:?} → c'={:?}, err={:.2e}",
                c,
                w,
                c_back,
                err
            );
        }
        eprintln!("worst roundtrip error on [-0.9, 0.9]^3: {worst_err:.2e}");
    }

    #[test]
    fn newton_converges_in_4_iters_on_interior() {
        // On the interior [-0.9, 0.9]^3, Newton must converge in ≤ 4 iters.
        let mut worst = 0;
        let mut failures = 0;
        for seed in 0..1000 {
            let c = sample(seed, 0.9);
            let w = forward(c);
            let mut converged_at = None;
            for budget in 1..=8 {
                if inverse(w, budget).is_some() {
                    converged_at = Some(budget);
                    break;
                }
            }
            let iters = converged_at.expect("never converged");
            if iters > worst {
                worst = iters;
            }
            if iters > 4 {
                failures += 1;
            }
        }
        eprintln!("max Newton iters on [-0.9, 0.9]^3: {worst}, >4 count: {failures}");
        assert!(
            worst <= 4,
            "Newton needed {worst} iterations on interior — target ≤ 4"
        );
    }

    #[test]
    fn newton_converges_in_6_iters_near_boundary() {
        // Closer to boundary: relax budget; still want bounded convergence.
        let mut worst = 0;
        for seed in 0..1000 {
            let c = sample(seed, 0.98);
            let w = forward(c);
            let mut converged_at = None;
            for budget in 1..=12 {
                if inverse(w, budget).is_some() {
                    converged_at = Some(budget);
                    break;
                }
            }
            let iters = converged_at.expect("never converged");
            if iters > worst {
                worst = iters;
            }
        }
        eprintln!("max Newton iters on [-0.98, 0.98]^3: {worst}");
        assert!(
            worst <= 6,
            "Newton needed {worst} iterations near boundary — target ≤ 6"
        );
    }

    #[test]
    fn sigma_min_at_known_points() {
        // σ_min(I) = 1 at origin (J = I).
        let sm0 = sigma_min([0.0, 0.0, 0.0]);
        assert!((sm0 - 1.0).abs() < 1e-5, "σ_min(origin)={sm0}, expected 1");

        // Face centers: J = diag(1, √½, √½) up to permutation; σ_min = √½.
        let expect = (0.5_f32).sqrt();
        for c in [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let sm = sigma_min(c);
            assert!(
                (sm - expect).abs() < 1e-4,
                "σ_min({:?})={sm}, expected {expect}",
                c
            );
        }
    }

    #[test]
    fn sigma_min_falloff_profile() {
        // Characterize σ_min's lower bound vs. cube radius. This is the
        // data the sphere-tracer needs to reason about perf: near the
        // cube corners the Jacobian pinches, so the safe world-space
        // step is some fraction of the safe cube-space step.
        //
        // We scan radii 0.5, 0.7, 0.85, 0.95 and find the minimum σ_min
        // in each shell. Prints the table; asserts the tail isn't
        // worse than what we observe today.
        let radii = [0.5_f32, 0.7, 0.85, 0.95];
        // lower bounds: these are observed minima from a 5k-sample sweep
        // minus a safety margin. The drop from 0.5 → 0.95 quantifies
        // the per-step perf penalty near the cube boundary.
        let expected_lower = [0.60_f32, 0.30, 0.10, 0.01];
        for (r, expected) in radii.iter().zip(expected_lower.iter()) {
            let mut worst = f32::INFINITY;
            for seed in 0..5000 {
                let c = sample(seed, *r);
                let sm = sigma_min(c);
                if sm < worst {
                    worst = sm;
                }
            }
            eprintln!("|c|∞ ≤ {r}: min σ_min = {worst:.4}  (expected ≥ {expected})");
            assert!(
                worst >= *expected,
                "σ_min on |c|∞ ≤ {r}: {worst:.4} < expected {expected}"
            );
        }
    }

    #[test]
    fn sigma_min_degenerates_at_cube_corner() {
        // Confirm the known degeneracy at the corner — this test PASSES
        // by asserting the degeneracy exists, so we don't silently rely
        // on a property that doesn't hold.
        let sm_corner = sigma_min([1.0, 1.0, 1.0]);
        let sm_near = sigma_min([0.999, 0.999, 0.999]);
        let sm_middle = sigma_min([0.5, 0.5, 0.5]);
        eprintln!("σ_min(0.5,0.5,0.5)={sm_middle:.4}  σ_min(0.999,0.999,0.999)={sm_near:.4}  σ_min(corner)={sm_corner:.4}");
        assert!(sm_corner < 0.02, "σ_min at corner: {sm_corner}");
        assert!(sm_near < 0.05, "σ_min near corner: {sm_near}");
        assert!(sm_middle > 0.5, "σ_min mid-diagonal: {sm_middle}");
    }

    #[test]
    fn sigma_min_degenerates_along_cube_edge() {
        // Along the edge (1, 1, t), the Jacobian is rank-deficient; σ_min = 0.
        for t in [-0.9_f32, -0.5, 0.0, 0.5, 0.9] {
            let sm = sigma_min([1.0, 1.0, t]);
            assert!(sm < 0.01, "σ_min on edge (1,1,{t}): {sm}");
        }
    }

    #[test]
    fn inverse_handles_sphere_surface() {
        // w on the unit sphere should invert back to a cube face.
        // Construct by forward-mapping a known face point.
        for seed in 0_u64..500 {
            let a = det_f32(seed.wrapping_mul(3));
            let b = det_f32(seed.wrapping_mul(5));
            let a = 0.9 * a;
            let b = 0.9 * b;
            let c_true = [1.0, a, b];
            let w = forward(c_true);
            let c_back = inverse(w, 10).unwrap_or_else(|| {
                panic!("inverse failed at sphere-surface w={:?}", w);
            });
            let err = v3_dist(c_true, c_back);
            assert!(
                err < 1e-3,
                "sphere surface roundtrip: c_true={:?} → w={:?} → c'={:?}, err={:.2e}",
                c_true,
                w,
                c_back,
                err
            );
        }
    }

    #[test]
    fn inverse_warm_start_is_stable() {
        // Simulate ray-march style: advance through closely-spaced w's
        // along a ray that stays strictly inside the unit ball.
        // Warm-starting Newton from the previous c should keep the
        // iteration budget at 1-3 per step.
        let mut c = [0.0_f32; 3];
        let n = 200;
        let mut worst_iters = 0;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            // ray from (0,0,0) to (0.5, 0.4, 0.3): |w|max ≈ 0.707
            let w = [0.5 * t, 0.4 * t, 0.3 * t];
            let mut converged_at = None;
            for budget in 1..=6 {
                if let Some(c_new) = inverse_from(w, c, budget) {
                    c = c_new;
                    converged_at = Some(budget);
                    break;
                }
            }
            let iters = converged_at.expect("warm-start Newton failed");
            if iters > worst_iters {
                worst_iters = iters;
            }
        }
        eprintln!("warm-start worst iters over straight ray: {worst_iters}");
        assert!(
            worst_iters <= 3,
            "warm-start should converge in ≤ 3 iters, got {worst_iters}"
        );
    }
}
