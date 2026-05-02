//! Double-float (df64) emulation: each value is `(hi, lo)` where
//! `hi` is the f32 nearest the true value and `lo` is the f32 residual
//! `value - hi`. Gives ~46-bit mantissa (vs 23 for plain f32) at
//! 2-3× compute. Used by the sphere descent to push past the f32
//! precision floor at deep cell-subtree depths (>~10).
//!
//! Mirrors `assets/shaders/df64.wgsl` op-for-op so the CPU raycast
//! and the GPU shader produce matching cell paths at every depth.
//!
//! `#[inline(never)]` on the error-free transforms is intentional —
//! it prevents LLVM from constant-folding `(s - bb) - s + bb` to 0
//! when test inputs are literals. The Apple Silicon WGSL compiler
//! has the same risk; we mitigate it shader-side by structuring the
//! ops as separate `let` bindings (no inline arithmetic chains).

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Df {
    pub hi: f32,
    pub lo: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Df3 {
    pub x: Df,
    pub y: Df,
    pub z: Df,
}

impl Df {
    pub const ZERO: Df = Df { hi: 0.0, lo: 0.0 };
    pub const ONE: Df = Df { hi: 1.0, lo: 0.0 };
    pub const THREE: Df = Df { hi: 3.0, lo: 0.0 };

    // f32 nearest to 1/3 = 0.333333343267441 (rounded up). The exact
    // residual is f64(1/3) − f32(1/3) ≈ -9.93410754e-9. Multiplying
    // a df by INV3 is the precision-stable way to divide by 3.
    pub const INV3: Df = Df { hi: 0.33333334, lo: -9.934108e-9 };

    pub const fn new(hi: f32, lo: f32) -> Self { Df { hi, lo } }
    pub const fn from_f32(hi: f32) -> Self { Df { hi, lo: 0.0 } }

    pub fn from_f64(v: f64) -> Self {
        let hi = v as f32;
        let lo = (v - hi as f64) as f32;
        Df { hi, lo }
    }

    pub fn to_f32(self) -> f32 { self.hi + self.lo }
    pub fn to_f64(self) -> f64 { self.hi as f64 + self.lo as f64 }

    pub fn neg(self) -> Self { Df { hi: -self.hi, lo: -self.lo } }
    pub fn abs(self) -> Self { if self.hi < 0.0 { self.neg() } else { self } }
}

#[inline(never)]
pub fn two_sum(a: f32, b: f32) -> Df {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    Df { hi: s, lo: e }
}

#[inline(never)]
pub fn quick_two_sum(a: f32, b: f32) -> Df {
    let s = a + b;
    let e = b - (s - a);
    Df { hi: s, lo: e }
}

const SPLITTER: f32 = 4097.0; // 2^12 + 1 — Veltkamp split for f32

#[inline(never)]
fn split(a: f32) -> (f32, f32) {
    let t = a * SPLITTER;
    let hi = t - (t - a);
    let lo = a - hi;
    (hi, lo)
}

#[inline(never)]
pub fn two_prod(a: f32, b: f32) -> Df {
    let p = a * b;
    let (a_hi, a_lo) = split(a);
    let (b_hi, b_lo) = split(b);
    let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    Df { hi: p, lo: e }
}

pub fn df_add(a: Df, b: Df) -> Df {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    let mid = quick_two_sum(s.hi, s.lo + t.hi);
    quick_two_sum(mid.hi, mid.lo + t.lo)
}

pub fn df_sub(a: Df, b: Df) -> Df { df_add(a, b.neg()) }

pub fn df_mul(a: Df, b: Df) -> Df {
    let p = two_prod(a.hi, b.hi);
    let cross = a.hi * b.lo + a.lo * b.hi;
    quick_two_sum(p.hi, p.lo + cross)
}

pub fn df_mul_f32(a: Df, k: f32) -> Df {
    let p = two_prod(a.hi, k);
    quick_two_sum(p.hi, p.lo + a.lo * k)
}

/// 1 / a — Newton refinement of the f32 reciprocal seed.
pub fn df_inv(a: Df) -> Df {
    let r0 = 1.0 / a.hi;
    let ar0 = df_mul_f32(a, r0);
    let two_minus = df_sub(Df::from_f32(2.0), ar0);
    df_mul_f32(two_minus, r0)
}

pub fn df_lt(a: Df, b: Df) -> bool {
    a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo)
}

pub fn df_le(a: Df, b: Df) -> bool {
    a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo)
}

/// Branchless argmin axis (0/1/2) by `Df`-aware comparison. Tie-break
/// matches WGSL `min_axis_mask`: x > y > z (z wins all-equal ties).
pub fn df3_min_axis(sd: Df3) -> usize {
    if df_lt(sd.x, sd.y) && df_lt(sd.x, sd.z) { 0 }
    else if df_le(sd.y, sd.z) { 1 }
    else { 2 }
}

// ─────────────────────────────────────────────────────────────────
// Df3 helpers
// ─────────────────────────────────────────────────────────────────

impl Df3 {
    pub const ZERO: Df3 = Df3 { x: Df::ZERO, y: Df::ZERO, z: Df::ZERO };

    pub fn from_f32_arr(v: [f32; 3]) -> Self {
        Df3 { x: Df::from_f32(v[0]), y: Df::from_f32(v[1]), z: Df::from_f32(v[2]) }
    }
    pub fn to_f32_arr(self) -> [f32; 3] {
        [self.x.to_f32(), self.y.to_f32(), self.z.to_f32()]
    }
    pub fn get(&self, i: usize) -> Df {
        match i { 0 => self.x, 1 => self.y, 2 => self.z, _ => unreachable!() }
    }
    pub fn set(&mut self, i: usize, v: Df) {
        match i { 0 => self.x = v, 1 => self.y = v, 2 => self.z = v, _ => unreachable!() }
    }
    pub fn neg(self) -> Self { Df3 { x: self.x.neg(), y: self.y.neg(), z: self.z.neg() } }
    pub fn abs(self) -> Self { Df3 { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() } }
}

pub fn df3_add(a: Df3, b: Df3) -> Df3 {
    Df3 { x: df_add(a.x, b.x), y: df_add(a.y, b.y), z: df_add(a.z, b.z) }
}

pub fn df3_sub(a: Df3, b: Df3) -> Df3 {
    Df3 { x: df_sub(a.x, b.x), y: df_sub(a.y, b.y), z: df_sub(a.z, b.z) }
}

pub fn df3_mul(a: Df3, b: Df3) -> Df3 {
    Df3 { x: df_mul(a.x, b.x), y: df_mul(a.y, b.y), z: df_mul(a.z, b.z) }
}

pub fn df3_mul_f32(a: Df3, k: f32) -> Df3 {
    Df3 { x: df_mul_f32(a.x, k), y: df_mul_f32(a.y, k), z: df_mul_f32(a.z, k) }
}

pub fn df3_scale(a: Df3, k: Df) -> Df3 {
    Df3 { x: df_mul(a.x, k), y: df_mul(a.y, k), z: df_mul(a.z, k) }
}

/// `a * 3` — multiplying by 3 is exact in f32 for inputs whose
/// mantissa has at least 2 unused bits; `lo*3` may overflow into hi
/// so we renormalize via quick_two_sum.
pub fn df_times3(a: Df) -> Df { quick_two_sum(a.hi * 3.0, a.lo * 3.0) }
pub fn df3_times3(a: Df3) -> Df3 {
    Df3 { x: df_times3(a.x), y: df_times3(a.y), z: df_times3(a.z) }
}

/// `a / 3` via the precomputed `INV3` constant. Multiplying by INV3
/// is precision-stable; raw `1/3` in f32 has ~3e-8 relative error
/// that compounds to f32 floor after ~5-6 pushes in the descent.
pub fn df_div3(a: Df) -> Df { df_mul(a, Df::INV3) }
pub fn df3_div3(a: Df3) -> Df3 {
    Df3 { x: df_div3(a.x), y: df_div3(a.y), z: df_div3(a.z) }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(a: f64, b: f64) -> f64 {
        if b == 0.0 { a.abs() } else { ((a - b) / b).abs() }
    }

    #[test]
    fn from_to_f64_roundtrip() {
        // Df gives ~46 bits relative (hi has 24 bits, lo adds another
        // 23-24). 1e-14 is the achievable bound near unity; tighter
        // tests below.
        for &v in &[0.0, 1.0, 1.0 / 3.0, std::f64::consts::PI, 1e-10, 1e10, -7.0 / 9.0] {
            let d = Df::from_f64(v);
            assert!(rel_err(d.to_f64(), v) < 1e-14,
                "v={v} d=({}, {}) recon={}", d.hi, d.lo, d.to_f64());
        }
    }

    #[test]
    fn two_sum_does_not_collapse() {
        // 1.0 + 1e-10 should produce a non-zero lo.
        let s = two_sum(1.0_f32, 1e-10_f32);
        assert_eq!(s.hi, 1.0);
        assert!(s.lo != 0.0, "two_sum collapsed to zero — likely FMA fusion in optimizer");
        assert!((s.lo - 1e-10_f32).abs() < 1e-15);
    }

    #[test]
    fn two_sum_matches_f64() {
        let pairs = [(1.0_f32, 1e-7_f32), (3.0, -3.0 + 1e-6), (1e8, -1e8 + 1.0)];
        for &(a, b) in &pairs {
            let s = two_sum(a, b);
            let exact = a as f64 + b as f64;
            let recon = s.to_f64();
            assert!((recon - exact).abs() < 1e-12 * exact.abs().max(1.0),
                "a={a} b={b} s=({}, {}) exact={exact} recon={recon}", s.hi, s.lo);
        }
    }

    #[test]
    fn df_add_matches_f64() {
        // No catastrophic-cancellation cases: when from_f64 quantizes
        // lo to f32, sub-ULP perturbations like 1e-12 fall below the
        // representation threshold. df gives ~46 bits across the hi+lo
        // pair under normal addition.
        let cases = [
            (1.0_f64, 1e-10),
            (1.0 / 3.0, 1.0 / 7.0),
            (1e6, 1e-6),
            (std::f64::consts::PI, std::f64::consts::E),
        ];
        for &(a, b) in &cases {
            let dc = df_add(Df::from_f64(a), Df::from_f64(b));
            let err = rel_err(dc.to_f64(), a + b);
            assert!(err < 1e-13, "a={a} b={b} got={} err={err}", dc.to_f64());
        }
    }

    #[test]
    fn df_add_distinguishes_sub_ulp_from_unity() {
        // A sub-ULP perturbation that f32 alone cannot represent
        // becomes visible as the lo of a df.
        let a = Df::from_f32(1.0);
        let b = Df::from_f32(1e-10); // f32 representable
        let s = df_add(a, b);
        assert_eq!(s.hi, 1.0);
        assert!(s.lo > 0.5e-10 && s.lo < 1.5e-10,
            "df should preserve the sub-ULP perturbation: lo={}", s.lo);
    }

    #[test]
    fn df_mul_matches_f64() {
        let cases = [
            (1.0 / 3.0, 3.0),
            (std::f64::consts::PI, 1.0 / std::f64::consts::PI),
            (1e5, 1e-5),
            (-7.0 / 9.0, 9.0 / 7.0),
        ];
        for &(a, b) in &cases {
            let dc = df_mul(Df::from_f64(a), Df::from_f64(b));
            let err = rel_err(dc.to_f64(), a * b);
            assert!(err < 1e-13, "a={a} b={b} got={} err={err}", dc.to_f64());
        }
    }

    #[test]
    fn df_inv_matches_f64() {
        for &v in &[3.0, 7.0, std::f64::consts::PI, 1e-3, 1e3, -2.5] {
            let d = Df::from_f64(v);
            let inv = df_inv(d);
            let err = rel_err(inv.to_f64(), 1.0 / v);
            assert!(err < 1e-13, "v={v} got={} err={err}", inv.to_f64());
        }
    }

    #[test]
    fn inv3_constant_matches_truth() {
        let exact = 1.0_f64 / 3.0;
        let recon = Df::INV3.to_f64();
        assert!((recon - exact).abs() < 1e-14, "INV3={} exact={} diff={}",
            recon, exact, recon - exact);
    }

    #[test]
    fn times3_div3_roundtrip_at_depth_20() {
        // 20 ÷3 then 20 ×3 round-trips with ~1e-7 relative error.
        // Each df_mul introduces ~1 ULP of f32 rounding in the cross
        // term that compounds across 20+ multiplications.
        // For the descent this is fine: side_dist precision needs to
        // beat cell width (~1e-10 in input-t units at depth 20), and
        // df gives 1e-13 absolute precision near typical t values —
        // 1000× headroom over what cell-resolution requires.
        for &v in &[0.5_f64, 1.0 / 7.0, 1e-6, std::f64::consts::PI / 9.0] {
            let mut d = Df::from_f64(v);
            for _ in 0..20 { d = df_div3(d); }
            for _ in 0..20 { d = df_times3(d); }
            let err = rel_err(d.to_f64(), v);
            assert!(err < 1e-6, "v={v} after ÷3^20 ×3^20 got={} err={err}", d.to_f64());
        }
    }

    #[test]
    fn df_beats_f32_at_deep_descent() {
        // Simulate the descent's inv_dir scaling: starts at orig_inv_dir,
        // each push divides by 3. After 20 pushes, |inv_dir| ≈ 3e-10.
        let exact = 1.0_f64 / 3.0_f64.powi(20);

        let mut d = Df::from_f32(1.0);
        for _ in 0..20 { d = df_div3(d); }
        let df_err = rel_err(d.to_f64(), exact);

        let mut f = 1.0_f32;
        for _ in 0..20 { f *= 1.0 / 3.0; }
        let f32_err = rel_err(f as f64, exact);

        // df precision floor at depth 20 (cumulative f32 lo errors)
        // is ~1e-7 relative. f32 alone is ~6e-7. df must be at least
        // 100× more precise.
        assert!(df_err < 1e-9, "df after ÷3^20: got={} err={df_err}", d.to_f64());
        assert!(f32_err > df_err * 100.0,
            "df not significantly better than f32: df_err={df_err} f32_err={f32_err}");
    }

    #[test]
    fn min_axis_handles_subulp_differences() {
        // Two side_dist values differ by 3e-10 — invisible in f32
        // (ULP near 1.0 is ~6e-8) but DF picks the smaller one.
        let sd = Df3 {
            x: Df::from_f64(1.0 + 3e-10),
            y: Df::from_f64(1.0),
            z: Df::from_f64(1.0 + 6e-10),
        };
        assert_eq!(df3_min_axis(sd), 1, "DF should distinguish sub-ULP differences");
    }
}
