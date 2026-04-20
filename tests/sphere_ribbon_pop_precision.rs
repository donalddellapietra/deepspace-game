//! Numeric verification of the `(n_base, n_delta)` ribbon-pop precision
//! scheme proposed in `docs/design/sphere-ribbon-pop-proposal.md`.
//!
//! At face-subtree depth N the absolute EA-coord delta between a
//! frame's adjacent child-cell u-planes is `2·size_ea/3 = 2/3^(N+1)`.
//! For N ≥ ~20 this is below f32 relative eps (~1.2e-7) when summed
//! with `u_corner_ea ≈ O(1)`. The naive form computes the plane
//! normal as `u_axis − ea_to_cube(u_corner_ea + K·d_ea)·n_axis`; at
//! deep N the three child-plane normals become numerically identical.
//!
//! The proposal factors the plane normal as
//! `n(K) = n_base + K·n_delta`, where `n_delta` captures the warp's
//! derivative. Ray-plane intersection
//! `t(K) = −(A + K·a)/(B + K·b)` with
//! `A = n_base·ro, B = n_base·rd, a = n_delta·ro, b = n_delta·rd`.
//!
//! This test compares three forms at depths 5, 10, 20, 25, 30, 35:
//!   * naive body-XYZ form (baseline — expected to collapse at deep N)
//!   * factored form `(A + K·a)/(B + K·b)` in body-XYZ
//!   * relative form `t(K) − t(0) = K·(A·b − B·a) / (B·(B + K·b))`
//!
//! and against an f64 ground-truth reference.

use std::f32::consts::FRAC_PI_4;

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn ea_to_cube_f32(e: f32) -> f32 {
    (e * FRAC_PI_4).tan()
}

#[inline]
fn ea_to_cube_deriv_f32(e: f32) -> f32 {
    let cs = (e * FRAC_PI_4).cos();
    FRAC_PI_4 / (cs * cs)
}

#[inline]
fn ea_to_cube_f64(e: f64) -> f64 {
    (e * std::f64::consts::FRAC_PI_4).tan()
}

/// Face: PosX. u_axis = (0,0,-1), v_axis = (0,1,0), n_axis = (1,0,0).
/// Plane normal at local K inside a frame with `u_corner_ea` and
/// per-unit-K EA-step `d_ea`:
/// `n(K) = u_axis − ea_to_cube(u_corner_ea + K·d_ea) · n_axis`.
///
/// Planes pass through body center (`oc = ro - body_center`).
fn naive_t(oc: [f32; 3], rd: [f32; 3], u_corner_ea: f32, d_ea: f32, k: f32) -> f32 {
    let u_axis = [0.0_f32, 0.0, -1.0];
    let n_axis = [1.0_f32, 0.0, 0.0];
    let e_k = u_corner_ea + k * d_ea;
    let c = ea_to_cube_f32(e_k);
    let n = [
        u_axis[0] - c * n_axis[0],
        u_axis[1] - c * n_axis[1],
        u_axis[2] - c * n_axis[2],
    ];
    -dot3(n, oc) / dot3(n, rd)
}

struct Factored {
    a_big: f32,
    b_big: f32,
    a_small: f32,
    b_small: f32,
}

fn factored_coeffs(oc: [f32; 3], rd: [f32; 3], u_corner_ea: f32, d_ea: f32) -> Factored {
    let u_axis = [0.0_f32, 0.0, -1.0];
    let n_axis = [1.0_f32, 0.0, 0.0];
    let c0 = ea_to_cube_f32(u_corner_ea);
    let slope = ea_to_cube_deriv_f32(u_corner_ea) * d_ea;
    let n_base = [
        u_axis[0] - c0 * n_axis[0],
        u_axis[1] - c0 * n_axis[1],
        u_axis[2] - c0 * n_axis[2],
    ];
    let n_delta = [-slope * n_axis[0], -slope * n_axis[1], -slope * n_axis[2]];
    Factored {
        a_big: dot3(n_base, oc),
        b_big: dot3(n_base, rd),
        a_small: dot3(n_delta, oc),
        b_small: dot3(n_delta, rd),
    }
}

fn factored_t(f: &Factored, k: f32) -> f32 {
    let num = -(f.a_big + k * f.a_small);
    let den = f.b_big + k * f.b_small;
    num / den
}

fn relative_t(f: &Factored, k: f32) -> f32 {
    let cross = f.a_big * f.b_small - f.b_big * f.a_small;
    let den = f.b_big * (f.b_big + k * f.b_small);
    k * cross / den
}

fn reference_t_f64(oc: [f64; 3], rd: [f64; 3], u_corner_ea: f64, d_ea: f64, k: f64) -> f64 {
    let u_axis = [0.0_f64, 0.0, -1.0];
    let n_axis = [1.0_f64, 0.0, 0.0];
    let e_k = u_corner_ea + k * d_ea;
    let c = ea_to_cube_f64(e_k);
    let n = [
        u_axis[0] - c * n_axis[0],
        u_axis[1] - c * n_axis[1],
        u_axis[2] - c * n_axis[2],
    ];
    let num = -(n[0] * oc[0] + n[1] * oc[1] + n[2] * oc[2]);
    let den = n[0] * rd[0] + n[1] * rd[1] + n[2] * rd[2];
    num / den
}

fn distinct_count<F: Fn(f32) -> f32>(f: F, ks: &[f32]) -> usize {
    let vals: Vec<f32> = ks.iter().map(|&k| f(k)).collect();
    let mut bits: Vec<u32> = vals.iter().map(|v| v.to_bits()).collect();
    bits.sort_unstable();
    bits.dedup();
    bits.len()
}

#[derive(Clone, Copy, Debug)]
struct DepthResult {
    depth: u32,
    size_ea: f64,
    reference_deltas: [f64; 4],
    naive_deltas: [f32; 4],
    factored_deltas: [f32; 4],
    relative_values: [f32; 4],
    naive_distinct: usize,
    factored_distinct: usize,
    relative_distinct: usize,
}

fn run_depth(depth: u32) -> DepthResult {
    // Face PosX. Cell roughly centered at (un=0.5, vn=0.5, rn=0.5).
    // Body sphere: center = 0, outer_r = 1, inner_r = 0.5.
    // At depth N we sit inside a frame of face-normalized size
    // 1/3^N centered on (0.5, 0.5, 0.5); frame u_corner_abs = 0.5 −
    // size_ea/2, so u_corner_ea = 2·u_corner_abs − 1 = −size_ea.
    let size_ea_f64: f64 = 3.0_f64.powi(-(depth as i32));
    let size_ea: f32 = size_ea_f64 as f32;
    let u_corner_ea: f32 = -size_ea;
    let u_corner_ea_f64: f64 = -size_ea_f64;
    // Per-local-unit-K EA step (local K spans [0, 3] over the frame).
    let d_ea: f32 = 2.0 * size_ea / 3.0;
    let d_ea_f64: f64 = 2.0 * size_ea_f64 / 3.0;

    // Ray from outside the sphere toward the cell center, with a
    // small off-axis tilt so plane-ray geometry isn't degenerate.
    let ro: [f32; 3] = [3.0, 0.10, 0.05];
    let body_pos_approx: [f32; 3] = [0.75, 0.0, 0.0];
    let diff = [
        body_pos_approx[0] - ro[0],
        body_pos_approx[1] - ro[1],
        body_pos_approx[2] - ro[2],
    ];
    let norm = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    let rd: [f32; 3] = [diff[0] / norm, diff[1] / norm, diff[2] / norm];

    // `oc = ro - body_center = ro` (body center at origin).
    let oc = ro;

    let ks = [0.0_f32, 1.0, 2.0, 3.0];

    let naive: Vec<f32> = ks.iter().map(|&k| naive_t(oc, rd, u_corner_ea, d_ea, k)).collect();
    let fact = factored_coeffs(oc, rd, u_corner_ea, d_ea);
    let factored: Vec<f32> = ks.iter().map(|&k| factored_t(&fact, k)).collect();
    let relative: Vec<f32> = ks.iter().map(|&k| relative_t(&fact, k)).collect();

    let oc_f64 = [oc[0] as f64, oc[1] as f64, oc[2] as f64];
    let rd_f64 = [rd[0] as f64, rd[1] as f64, rd[2] as f64];
    let reference: Vec<f64> = ks
        .iter()
        .map(|&k| reference_t_f64(oc_f64, rd_f64, u_corner_ea_f64, d_ea_f64, k as f64))
        .collect();

    let mut reference_deltas = [0.0_f64; 4];
    let mut naive_deltas = [0.0_f32; 4];
    let mut factored_deltas = [0.0_f32; 4];
    let mut relative_values = [0.0_f32; 4];
    for i in 0..4 {
        reference_deltas[i] = reference[i] - reference[0];
        naive_deltas[i] = naive[i] - naive[0];
        factored_deltas[i] = factored[i] - factored[0];
        relative_values[i] = relative[i];
    }

    let naive_distinct = distinct_count(|k| naive_t(oc, rd, u_corner_ea, d_ea, k), &ks);
    let factored_distinct = distinct_count(|k| factored_t(&fact, k), &ks);
    let relative_distinct = distinct_count(|k| relative_t(&fact, k), &ks);

    DepthResult {
        depth,
        size_ea: size_ea_f64,
        reference_deltas,
        naive_deltas,
        factored_deltas,
        relative_values,
        naive_distinct,
        factored_distinct,
        relative_distinct,
    }
}

fn print_result(r: &DepthResult) {
    println!("\n── depth {} (size_ea = {:.3e}) ───────────────────", r.depth, r.size_ea);
    println!(
        "   distinct counts  naive={}/4  factored={}/4  relative={}/4",
        r.naive_distinct, r.factored_distinct, r.relative_distinct
    );
    println!("   K |      ref Δt (f64)    |   naive Δt (f32)   | factored Δt (f32)  | relative t (f32)");
    for i in 0..4 {
        println!(
            "   {} | {:+20.12e} | {:+18.10e} | {:+18.10e} | {:+18.10e}",
            i, r.reference_deltas[i], r.naive_deltas[i], r.factored_deltas[i], r.relative_values[i]
        );
    }
}

#[test]
fn ribbon_pop_precision_at_depth() {
    let depths = [5u32, 10, 15, 20, 25, 30, 35];
    let mut results: Vec<DepthResult> = Vec::new();
    for &d in &depths {
        let r = run_depth(d);
        print_result(&r);
        results.push(r);
    }

    // The proposal's claim: the relative form keeps all four K values
    // distinguishable at arbitrary depth. Assert this for every
    // tested depth. The naive form is expected to collapse past
    // depth ~20; we do NOT assert on it (it is the baseline we're
    // trying to beat).
    println!("\nSummary:");
    for r in &results {
        println!(
            "  depth {:>2}  naive={}/4  factored={}/4  relative={}/4",
            r.depth, r.naive_distinct, r.factored_distinct, r.relative_distinct
        );
    }

    for r in &results {
        assert!(
            r.relative_distinct == 4,
            "relative form collapsed at depth {} (distinct={}/4)",
            r.depth, r.relative_distinct
        );
    }

    // The factored form SHOULD survive deep too — it's just the
    // relative form's numerator/denominator written differently, but
    // the subtraction that kills precision is hidden in the final
    // division rather than in intermediate sums. Document where it
    // breaks.
    let factored_breaks = results.iter().find(|r| r.factored_distinct < 4);
    if let Some(r) = factored_breaks {
        println!(
            "\n⚠ factored form collapses starting depth {} — relative form required",
            r.depth
        );
    }
}
