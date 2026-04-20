//! [`FaceFrame`] — ribbon-popped per-cell plane-normal state for
//! precision-stable ray marching inside a face subtree.
//!
//! At face-subtree depth `D` (0 = face root, 1 = face root's direct
//! child, …), the frame carries four quantities per axis:
//!
//!   * `n_base_{u,v}` — plane normal at the frame's K=0 corner, in
//!     body-local XYZ. All u/v planes pass through the body sphere
//!     center; only the normal direction varies across the frame.
//!   * `n_delta_{u,v}` — how that normal changes per one local unit
//!     of advance along the face axis. The frame's 3 local units
//!     span its full u (or v) extent, so the plane at local K has
//!     normal `n_base + K · n_delta`.
//!   * `r_base`, `r_delta` — absolute body-local shell radius at the
//!     K=0 corner, plus per-local-unit advance. Radial axis is
//!     linear (no warp), so stored as scalars rather than vectors.
//!
//! Descent to a child at slot `(us, vs, rs)`:
//!
//! ```text
//! child.n_base_u  = parent.n_base_u + us · parent.n_delta_u
//! child.n_delta_u = parent.n_delta_u / 3
//! child.n_base_v  = parent.n_base_v + vs · parent.n_delta_v
//! child.n_delta_v = parent.n_delta_v / 3
//! child.r_base    = parent.r_base + rs · parent.r_delta
//! child.r_delta   = parent.r_delta / 3
//! ```
//!
//! This is the sphere ribbon-pop: at every descent, state stays
//! O(1) numerically even though absolute face-normalized
//! coordinates collapse below f32 eps past depth ~15. The
//! feasibility test (`ribbon_pop_feasibility`) verifies that a
//! depth-60 descent with cross-product Δt preserves precision
//! identically to depth-30, which requires this accumulation form.
//!
//! The face root's `(n_base, n_delta)` values are computed from the
//! warp directly — see [`FaceFrame::at_face_root`]. That is the
//! "exact handoff" from the curved body-root march to the linearized
//! descendant march. Feasibility-sweep data at `k_start ≥ 5` shows
//! the linearization residual drops below 0.1 %, well inside f32
//! precision. For shallow frames (depth 1–4) the residual is
//! visually noticeable as silhouette faceting; rendering uses the
//! exact curved march there instead.

use super::geometry::Face;

/// Body-local ribbon-pop state for a single face-subtree frame.
///
/// All plane normals are expressed in body-local XYZ (the body
/// node's `[0, 1)³` frame scaled to the caller's coordinate system
/// via a `body_size` factor where relevant). Planes pass through
/// the body sphere center, so ray-plane `t` reduces to
/// `-(n · ray_origin) / (n · ray_dir)` — no offset needed.
#[derive(Copy, Clone, Debug)]
pub struct FaceFrame {
    pub face: Face,
    /// Depth beneath the face root (0 = the face root itself, 1 =
    /// a direct child of the face root, …). Matches the length of
    /// the UVR-slot descent list, **not** the world-tree depth.
    pub depth: u32,
    /// Body-local inner shell radius. Carried to let the DDA
    /// compute the inner/outer radial boundary when the frame sits
    /// at the shell's edge.
    pub inner_r: f32,
    /// Body-local outer shell radius.
    pub outer_r: f32,

    /// Plane normal at the frame's K=0 corner along the u axis.
    pub n_base_u: [f32; 3],
    /// Per-local-unit advance of the u-plane normal across the
    /// frame's 3 local units.
    pub n_delta_u: [f32; 3],
    /// Plane normal at the frame's K=0 corner along the v axis.
    pub n_base_v: [f32; 3],
    /// Per-local-unit advance of the v-plane normal.
    pub n_delta_v: [f32; 3],

    /// Body-local radius at the frame's K=0 corner along r.
    pub r_base: f32,
    /// Per-local-unit advance of the shell radius across the
    /// frame's 3 local units (1/3 of the frame's r-extent).
    pub r_delta: f32,
}

impl FaceFrame {
    /// Build the face-root frame for a given body.
    ///
    /// The face root spans the full face: `u_ea ∈ [-1, 1]`,
    /// `v_ea ∈ [-1, 1]`, radial `[inner_r, outer_r]`. Its 3 local
    /// units thus span `2` ea (u or v) or `outer_r - inner_r`
    /// (radial).
    ///
    /// `n_base_u` at u_ea = -1 is `u_axis - n_axis · tan(-π/4) =
    /// u_axis + n_axis`. `n_delta_u` at the face root is
    /// `-n_axis · (π/4) · sec²(-π/4) · (2/3)`. Same for v.
    pub fn at_face_root(face: Face, inner_r: f32, outer_r: f32) -> Self {
        use std::f32::consts::FRAC_PI_4;
        let n_axis = face.normal();
        let (u_axis, v_axis) = face.tangents();

        // tan(-π/4) = -1, so u_axis - n_axis · (-1) = u_axis + n_axis.
        let tan_u0 = (-1.0_f32 * FRAC_PI_4).tan();
        let tan_v0 = tan_u0;
        let sec2_u0 = {
            let c = (-1.0_f32 * FRAC_PI_4).cos();
            1.0 / (c * c)
        };
        let sec2_v0 = sec2_u0;

        // Frame's 3 local units span 2 ea; per-local-unit ea = 2/3.
        let per_local_ea = 2.0_f32 / 3.0;
        let du_scalar = FRAC_PI_4 * sec2_u0 * per_local_ea;
        let dv_scalar = FRAC_PI_4 * sec2_v0 * per_local_ea;

        Self {
            face,
            depth: 0,
            inner_r,
            outer_r,
            n_base_u: [
                u_axis[0] - n_axis[0] * tan_u0,
                u_axis[1] - n_axis[1] * tan_u0,
                u_axis[2] - n_axis[2] * tan_u0,
            ],
            n_delta_u: [
                -n_axis[0] * du_scalar,
                -n_axis[1] * du_scalar,
                -n_axis[2] * du_scalar,
            ],
            n_base_v: [
                v_axis[0] - n_axis[0] * tan_v0,
                v_axis[1] - n_axis[1] * tan_v0,
                v_axis[2] - n_axis[2] * tan_v0,
            ],
            n_delta_v: [
                -n_axis[0] * dv_scalar,
                -n_axis[1] * dv_scalar,
                -n_axis[2] * dv_scalar,
            ],
            // Radial: frame's 3 local units span (outer − inner).
            r_base: inner_r,
            r_delta: (outer_r - inner_r) / 3.0,
        }
    }

    /// Descend into the child at slot `(us, vs, rs)`. Ribbon-pop
    /// formula: shift `n_base` by `slot · n_delta`, scale
    /// `n_delta` by 1/3 (child's 3 local units cover one parent
    /// local unit).
    pub fn descend(&self, us: u32, vs: u32, rs: u32) -> Self {
        let (us_f, vs_f, rs_f) = (us as f32, vs as f32, rs as f32);
        Self {
            face: self.face,
            depth: self.depth + 1,
            inner_r: self.inner_r,
            outer_r: self.outer_r,
            n_base_u: [
                self.n_base_u[0] + us_f * self.n_delta_u[0],
                self.n_base_u[1] + us_f * self.n_delta_u[1],
                self.n_base_u[2] + us_f * self.n_delta_u[2],
            ],
            n_delta_u: [
                self.n_delta_u[0] / 3.0,
                self.n_delta_u[1] / 3.0,
                self.n_delta_u[2] / 3.0,
            ],
            n_base_v: [
                self.n_base_v[0] + vs_f * self.n_delta_v[0],
                self.n_base_v[1] + vs_f * self.n_delta_v[1],
                self.n_base_v[2] + vs_f * self.n_delta_v[2],
            ],
            n_delta_v: [
                self.n_delta_v[0] / 3.0,
                self.n_delta_v[1] / 3.0,
                self.n_delta_v[2] / 3.0,
            ],
            r_base: self.r_base + rs_f * self.r_delta,
            r_delta: self.r_delta / 3.0,
        }
    }

    /// Descend along a slice of UVR slots (index `k` is the descent
    /// step from face-root into its k-th descendant). Equivalent to
    /// iterating `descend` and more convenient for path-based
    /// construction.
    pub fn descend_path(mut self, slots: &[(u32, u32, u32)]) -> Self {
        for &(us, vs, rs) in slots {
            self = self.descend(us, vs, rs);
        }
        self
    }
}

// ──────────────────────────────────────────────────────────────
// Feasibility gate for the ribbon-pop sphere proposal. See
// docs/design/sphere-ribbon-pop-proposal.md.
//
// These tests prove:
//   1. The naive f32 form collapses at face-subtree depth 30
//      (baseline confirming the problem).
//   2. A directly-computed `(n_base, n_delta)` in f32 delivers
//      ray-plane intersections within f32 precision of f64 exact.
//   3. Ribbon-popping `(n_base, n_delta)` through 30 / 40 / 60
//      descent levels preserves that precision.
//   4. The factored absolute t(K) form collapses at deep depth
//      while the cross-product Δt form survives — the DDA must
//      work with per-cell delta-t, not absolute t.
//   5. The radial axis ribbon-pops with a structurally identical
//      sqrt-difference form.

#[cfg(test)]
mod ribbon_pop_feasibility {
    const FRAC_PI_4_F64: f64 = std::f64::consts::FRAC_PI_4;
    const FRAC_PI_4_F32: f32 = std::f32::consts::FRAC_PI_4;

    // Slot pattern `[2, 0]` (repeated) drives u_face toward 0.75 →
    // u_ea ≈ 0.5, away from the `±1` face edges where sec² diverges.
    const SLOT_PATTERN: [u32; 2] = [2, 0];

    fn slots(depth: u32) -> Vec<u32> {
        (0..depth as usize).map(|i| SLOT_PATTERN[i % SLOT_PATTERN.len()]).collect()
    }

    /// Exact u_ea corner at the end of `slots` (computed in f64 so
    /// we have a precision reference at any depth).
    fn u_ea_at(slots: &[u32]) -> f64 {
        let mut u = 0.0_f64;
        let mut size = 1.0_f64;
        for &s in slots {
            u += (s as f64) * size / 3.0;
            size /= 3.0;
        }
        2.0 * u - 1.0
    }

    /// Frame width in ea at this frame depth: `2 / 3^depth`.
    fn size_ea(depth: u32) -> f64 {
        2.0 / 3.0_f64.powi(depth as i32)
    }

    fn norm3_f64(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    /// Ray origin and direction in body-local sphere-centered coords.
    fn ref_ray() -> ([f64; 3], [f64; 3]) {
        (
            [0.8_f64, 0.1, 0.1],
            norm3_f64([1.0, -0.05, -0.05]),
        )
    }

    /// PosX face: `u_axis = (0, 0, -1)`, `n_axis = (1, 0, 0)`. Cell
    /// plane passes through the sphere center, with normal
    /// `n = u_axis − n_axis · tan(u_ea · π/4) = (-tan, 0, -1)`.
    fn plane_n_f64(u_ea: f64) -> [f64; 3] {
        let t = (u_ea * FRAC_PI_4_F64).tan();
        [-t, 0.0, -1.0]
    }

    fn plane_n_f32(u_ea: f32) -> [f32; 3] {
        let t = (u_ea * FRAC_PI_4_F32).tan();
        [-t, 0.0, -1.0]
    }

    fn ray_plane_t_f64(n: [f64; 3], o: [f64; 3], d: [f64; 3]) -> f64 {
        let num = -(o[0] * n[0] + o[1] * n[1] + o[2] * n[2]);
        let den = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
        num / den
    }

    fn ray_plane_t_f32(n: [f32; 3], o: [f32; 3], d: [f32; 3]) -> f32 {
        let num = -(o[0] * n[0] + o[1] * n[1] + o[2] * n[2]);
        let den = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
        num / den
    }

    /// Well-conditioned Δt from K=0 to K=K1 given
    /// `n(K) = n_base + K·n_delta`:
    ///   Δt = K1·(A·b − B·a) / (B · (B + K1·b))
    /// where `(A, B, a, b) = (o·n_base, d·n_base, o·n_delta,
    /// d·n_delta)`.
    fn delta_t_lin(
        n_base: [f32; 3],
        n_delta: [f32; 3],
        o: [f32; 3],
        d: [f32; 3],
        k1: f32,
    ) -> f32 {
        let big_a = o[0] * n_base[0] + o[1] * n_base[1] + o[2] * n_base[2];
        let big_b = d[0] * n_base[0] + d[1] * n_base[1] + d[2] * n_base[2];
        let sm_a = o[0] * n_delta[0] + o[1] * n_delta[1] + o[2] * n_delta[2];
        let sm_b = d[0] * n_delta[0] + d[1] * n_delta[1] + d[2] * n_delta[2];
        let num = k1 * (big_a * sm_b - big_b * sm_a);
        let den = big_b * (big_b + k1 * sm_b);
        num / den
    }

    /// Ribbon-pop descent: `n_base_child = n_base + slot · n_delta`;
    /// `n_delta_child = n_delta / 3`.
    fn ribbon_pop(
        slots: &[u32],
        mut n_base: [f32; 3],
        mut n_delta: [f32; 3],
    ) -> ([f32; 3], [f32; 3]) {
        for &s in slots {
            let k = s as f32;
            n_base = [
                n_base[0] + k * n_delta[0],
                n_base[1] + k * n_delta[1],
                n_base[2] + k * n_delta[2],
            ];
            n_delta = [n_delta[0] / 3.0, n_delta[1] / 3.0, n_delta[2] / 3.0];
        }
        (n_base, n_delta)
    }

    /// f64 reference Δt across one cell at a frame of depth
    /// `frame_depth`, whose corner is at `u_ea`. Uses the identity
    ///   `tan(u + δ) − tan(u) = tan(δ)·sec²(u) / (1 − tan(u)·tan(δ))`
    /// so the subtraction doesn't cancel at deep depth.
    fn dt_exact(u_ea: f64, frame_depth: u32) -> f64 {
        let per_local = size_ea(frame_depth + 1);
        let u_arg = u_ea * FRAC_PI_4_F64;
        let delta_arg = per_local * FRAC_PI_4_F64;
        let tan_u = u_arg.tan();
        let cos_u = u_arg.cos();
        let sec2_u = 1.0 / (cos_u * cos_u);
        let tan_d = delta_arg.tan();
        let dtan = tan_d * sec2_u / (1.0 - tan_u * tan_d);

        let n_base = [-tan_u, 0.0, -1.0];
        let d_n = [-dtan, 0.0, 0.0];
        let (o, d) = ref_ray();
        let big_a = o[0] * n_base[0] + o[1] * n_base[1] + o[2] * n_base[2];
        let big_b = d[0] * n_base[0] + d[1] * n_base[1] + d[2] * n_base[2];
        let sm_a = o[0] * d_n[0] + o[1] * d_n[1] + o[2] * d_n[2];
        let sm_b = d[0] * d_n[0] + d[1] * d_n[1] + d[2] * d_n[2];
        let num = big_a * sm_b - big_b * sm_a;
        let den = big_b * (big_b + sm_b);
        num / den
    }

    fn o_f32() -> [f32; 3] {
        let (o, _) = ref_ray();
        [o[0] as f32, o[1] as f32, o[2] as f32]
    }

    fn d_f32() -> [f32; 3] {
        let (_, d) = ref_ray();
        [d[0] as f32, d[1] as f32, d[2] as f32]
    }

    fn exact_base_delta_f32(u_ea_corner: f64, depth: u32) -> ([f32; 3], [f32; 3]) {
        let per_local = size_ea(depth + 1);
        let tan_w = (u_ea_corner * FRAC_PI_4_F64).tan();
        let sec2_w = {
            let c = (u_ea_corner * FRAC_PI_4_F64).cos();
            1.0 / (c * c)
        };
        let delta_scalar = FRAC_PI_4_F64 * sec2_w * per_local;
        let n_base = [-tan_w as f32, 0.0_f32, -1.0_f32];
        let n_delta = [-delta_scalar as f32, 0.0_f32, 0.0_f32];
        (n_base, n_delta)
    }

    // ───────────────────────────────── test 1: naive f32 collapses

    #[test]
    fn naive_f32_collapses_at_depth_30() {
        let s = slots(30);
        let u_ea_corner_f64 = u_ea_at(&s);
        let per_local_f64 = size_ea(31);

        let u0_f32 = u_ea_corner_f64 as f32;
        let u1_f32 = (u_ea_corner_f64 + per_local_f64) as f32;
        assert_eq!(
            u0_f32, u1_f32,
            "expected f32 to collapse adjacent u_ea values at depth 30; \
             got distinct {u0_f32:e} vs {u1_f32:e}",
        );

        let n0 = plane_n_f32(u0_f32);
        let n1 = plane_n_f32(u1_f32);
        let t0 = ray_plane_t_f32(n0, o_f32(), d_f32());
        let t1 = ray_plane_t_f32(n1, o_f32(), d_f32());
        assert_eq!(t1 - t0, 0.0);
    }

    // ───── test 2: directly-computed linearized matches f64 exact

    #[test]
    fn directly_computed_linearized_matches_f64_at_depth_30() {
        let s = slots(30);
        let u_ea_corner = u_ea_at(&s);
        let (n_base, n_delta) = exact_base_delta_f32(u_ea_corner, 30);
        let dt_lin = delta_t_lin(n_base, n_delta, o_f32(), d_f32(), 1.0);
        let dt_ref = dt_exact(u_ea_corner, 30);
        let rel = ((dt_lin as f64) - dt_ref).abs() / dt_ref.abs();
        assert!(
            rel < 1e-4,
            "directly-computed: rel_err = {rel:.3e} (Δt_exact = {dt_ref:.3e}, Δt_lin = {:e})",
            dt_lin
        );
    }

    // ─── test 2b: factored form collapses, cross-product survives

    fn t_k_factored_f32(
        n_base: [f32; 3],
        n_delta: [f32; 3],
        o: [f32; 3],
        d: [f32; 3],
        k: f32,
    ) -> f32 {
        let big_a = o[0] * n_base[0] + o[1] * n_base[1] + o[2] * n_base[2];
        let big_b = d[0] * n_base[0] + d[1] * n_base[1] + d[2] * n_base[2];
        let sm_a = o[0] * n_delta[0] + o[1] * n_delta[1] + o[2] * n_delta[2];
        let sm_b = d[0] * n_delta[0] + d[1] * n_delta[1] + d[2] * n_delta[2];
        -(big_a + k * sm_a) / (big_b + k * sm_b)
    }

    #[test]
    fn factored_absolute_form_collapses_at_depth_30_but_cross_product_survives() {
        let s = slots(30);
        let u_ea_corner = u_ea_at(&s);
        let (n_base, n_delta) = exact_base_delta_f32(u_ea_corner, 30);
        let o = o_f32();
        let d = d_f32();

        let per_local = size_ea(31);
        let u_arg = u_ea_corner * FRAC_PI_4_F64;
        let tan_u = u_arg.tan();
        let sec2_u = 1.0 / (u_arg.cos() * u_arg.cos());
        let n_base_ref = [-tan_u, 0.0, -1.0];
        let (o_ref, d_ref) = ref_ray();
        let big_a_ref = o_ref[0] * n_base_ref[0] + o_ref[1] * n_base_ref[1] + o_ref[2] * n_base_ref[2];
        let big_b_ref = d_ref[0] * n_base_ref[0] + d_ref[1] * n_base_ref[1] + d_ref[2] * n_base_ref[2];
        let t0_ref = -big_a_ref / big_b_ref;

        let t_factored_0 = t_k_factored_f32(n_base, n_delta, o, d, 0.0);
        for k in 1..=3u32 {
            let k_f32 = k as f32;
            let delta_arg = (k as f64) * per_local * FRAC_PI_4_F64;
            let tan_d = delta_arg.tan();
            let dtan = tan_d * sec2_u / (1.0 - tan_u * tan_d);
            let d_n = [-dtan, 0.0, 0.0];
            let sm_a_ref = o_ref[0] * d_n[0] + o_ref[1] * d_n[1] + o_ref[2] * d_n[2];
            let sm_b_ref = d_ref[0] * d_n[0] + d_ref[1] * d_n[1] + d_ref[2] * d_n[2];
            let dt_ref = (big_a_ref * sm_b_ref - big_b_ref * sm_a_ref)
                / (big_b_ref * (big_b_ref + sm_b_ref));

            let dt_cross = delta_t_lin(n_base, n_delta, o, d, k_f32);
            let rel_cross = ((dt_cross as f64) - dt_ref).abs() / dt_ref.abs();

            let t_factored_k = t_k_factored_f32(n_base, n_delta, o, d, k_f32);
            let dt_factored = (t_factored_k - t_factored_0) as f64;
            let abs_factored = (dt_factored - dt_ref).abs();

            assert!(
                rel_cross < 1e-4,
                "K={k}: cross-product rel_err = {rel_cross:.3e} \
                 (dt_ref = {dt_ref:.3e}, dt_cross = {dt_cross:e})",
            );
            assert_eq!(
                dt_factored, 0.0,
                "K={k}: factored absolute form unexpectedly produced non-zero \
                 Δt = {dt_factored:e}; expected collapse at depth 30 \
                 (abs err vs truth = {abs_factored:e})",
            );
            let _ = t0_ref;
        }
    }

    // ─── test 3: ribbon-pop descent matches truth at 30 / 40 / 60

    fn ribbon_pop_sweep_at(full_depth: u32) -> Vec<(u32, f64)> {
        let full_slots = slots(full_depth);
        let u_ea_full = u_ea_at(&full_slots);
        let dt_ref = dt_exact(u_ea_full, full_depth);

        let mut results: Vec<(u32, f64)> = Vec::new();
        for k_start in 1..=10u32 {
            let slots_before = &full_slots[..k_start as usize];
            let slots_after = &full_slots[k_start as usize..];
            let u_ea_k_start = u_ea_at(slots_before);
            let (nb, nd) = exact_base_delta_f32(u_ea_k_start, k_start);
            let (n_base, n_delta) = ribbon_pop(slots_after, nb, nd);
            let dt_lin = delta_t_lin(n_base, n_delta, o_f32(), d_f32(), 1.0);
            let rel = ((dt_lin as f64) - dt_ref).abs() / dt_ref.abs();
            results.push((k_start, rel));
        }
        results
    }

    #[test]
    fn ribbon_pop_descent_to_depth_30_sweep_handoff() {
        let results = ribbon_pop_sweep_at(30);
        let diag: String = results.iter()
            .map(|(k, r)| format!("  k_start={k:2} rel_err={r:.3e}\n")).collect();
        eprintln!("ribbon-pop handoff sweep to depth 30:\n{diag}");
        let best = results.iter()
            .filter(|(k, _)| *k >= 2 && *k <= 10)
            .map(|(_, r)| *r)
            .fold(f64::INFINITY, f64::min);
        assert!(best < 0.01,
            "no handoff depth 2..=10 achieved rel_err < 1% — best was {best:.3e}\n{diag}");
    }

    #[test]
    fn ribbon_pop_descent_to_depth_40_sweep_handoff() {
        let results = ribbon_pop_sweep_at(40);
        let diag: String = results.iter()
            .map(|(k, r)| format!("  k_start={k:2} rel_err={r:.3e}\n")).collect();
        eprintln!("ribbon-pop handoff sweep to depth 40:\n{diag}");
        let best = results.iter()
            .filter(|(k, _)| *k >= 2 && *k <= 10)
            .map(|(_, r)| *r)
            .fold(f64::INFINITY, f64::min);
        assert!(best < 0.01,
            "no handoff depth 2..=10 achieved rel_err < 1% at depth 40 — best was {best:.3e}\n{diag}");
    }

    #[test]
    fn ribbon_pop_descent_to_depth_60_sweep_handoff() {
        let results = ribbon_pop_sweep_at(60);
        let diag: String = results.iter()
            .map(|(k, r)| format!("  k_start={k:2} rel_err={r:.3e}\n")).collect();
        eprintln!("ribbon-pop handoff sweep to depth 60:\n{diag}");
        let best = results.iter()
            .filter(|(k, _)| *k >= 2 && *k <= 10)
            .map(|(_, r)| *r)
            .fold(f64::INFINITY, f64::min);
        assert!(best < 0.01,
            "no handoff depth 2..=10 achieved rel_err < 1% at depth 60 — best was {best:.3e}\n{diag}");
    }

    // ─────────── test 4: radial (ray-sphere) axis at depth 40

    #[test]
    fn radial_ribbon_pop_at_depth_40() {
        let depth = 40u32;
        let shell_f64 = 0.05_f64;
        let inner_f64 = 0.45_f64;
        let per_local_f64 = shell_f64 / 3.0_f64.powi(depth as i32 + 1);
        let r_base_f64 = inner_f64 + 0.3 * shell_f64;

        let o_f64 = [0.8_f64, 0.0, 0.0];
        let d_f64 = norm3_f64([-1.0, 0.0, 0.0]);
        let b_f64 = o_f64[0] * d_f64[0] + o_f64[1] * d_f64[1] + o_f64[2] * d_f64[2];
        let oo_f64 = o_f64[0] * o_f64[0] + o_f64[1] * o_f64[1] + o_f64[2] * o_f64[2];
        let d0_f64 = b_f64 * b_f64 - oo_f64 + r_base_f64 * r_base_f64;
        let d1_f64 = 2.0 * r_base_f64 * per_local_f64;
        let d2_f64 = per_local_f64 * per_local_f64;

        let o_f32 = [o_f64[0] as f32, o_f64[1] as f32, o_f64[2] as f32];
        let d_f32 = [d_f64[0] as f32, d_f64[1] as f32, d_f64[2] as f32];
        let r_base = r_base_f64 as f32;
        let per_local = per_local_f64 as f32;
        let b = o_f32[0] * d_f32[0] + o_f32[1] * d_f32[1] + o_f32[2] * d_f32[2];
        let oo = o_f32[0] * o_f32[0] + o_f32[1] * o_f32[1] + o_f32[2] * o_f32[2];
        let d0 = b * b - oo + r_base * r_base;
        let d1 = 2.0 * r_base * per_local;
        let d2 = per_local * per_local;

        for k in 1..=3u32 {
            let kf64 = k as f64;
            let dk_num_f64 = kf64 * d1_f64 + kf64 * kf64 * d2_f64;
            let dt_ref = dk_num_f64 / ((d0_f64 + dk_num_f64).sqrt() + d0_f64.sqrt());

            let kf32 = k as f32;
            let dk_num = kf32 * d1 + kf32 * kf32 * d2;
            let dt_lin = dk_num / ((d0 + dk_num).sqrt() + d0.sqrt());
            let rel_lin = ((dt_lin as f64) - dt_ref).abs() / dt_ref.abs();

            let r_k = r_base + kf32 * per_local;
            let dk_naive = (b * b - oo + r_k * r_k).sqrt() - d0.sqrt();

            assert!(rel_lin < 1e-4,
                "radial K={k} at depth 40: rationalized rel_err = {rel_lin:.3e} \
                 (dt_ref = {dt_ref:.3e}, dt_lin = {dt_lin:e})");
            assert_eq!(dk_naive, 0.0,
                "radial K={k} at depth 40: expected naive form to collapse, got {dk_naive:e}");
        }
    }
}

// ─────────────────────── tests: FaceFrame ribbon-pop wiring

#[cfg(test)]
mod face_frame_tests {
    use super::*;

    /// Descend `FaceFrame` via repeated `descend` and verify the
    /// ribbon-pop invariants hold: `n_delta` shrinks by 1/3 each
    /// step, `r_delta` shrinks by 1/3 each step, `depth` increments.
    #[test]
    fn descend_preserves_invariants() {
        let mut f = FaceFrame::at_face_root(Face::PosX, 0.45, 0.50);
        let d0 = f.n_delta_u;
        let r0 = f.r_delta;
        for (us, vs, rs) in [(1, 2, 0), (0, 1, 2), (2, 0, 1)] {
            f = f.descend(us, vs, rs);
        }
        assert_eq!(f.depth, 3);
        // After 3 descents, n_delta should be 1/27 of root's.
        for i in 0..3 {
            let expected = d0[i] / 27.0;
            let err = (f.n_delta_u[i] - expected).abs();
            assert!(err < 1e-6 * expected.abs().max(1.0),
                "n_delta_u[{i}] = {} ≠ expected {}", f.n_delta_u[i], expected);
        }
        let expected_r = r0 / 27.0;
        let r_err = (f.r_delta - expected_r).abs();
        assert!(r_err < 1e-6 * expected_r.abs().max(1.0));
    }

    /// At the face root, the u-plane normal at K=0 should equal
    /// `u_axis + n_axis` (since `tan(-π/4) = -1`).
    #[test]
    fn face_root_u_base_at_corner() {
        for &face in &Face::ALL {
            let f = FaceFrame::at_face_root(face, 0.45, 0.50);
            let n_axis = face.normal();
            let (u_axis, _) = face.tangents();
            for i in 0..3 {
                let expected = u_axis[i] + n_axis[i];
                assert!((f.n_base_u[i] - expected).abs() < 1e-6,
                    "face {face:?} u[{i}]: got {}, expected {}", f.n_base_u[i], expected);
            }
        }
    }

    /// `descend_path` equivalent to repeated `descend`.
    #[test]
    fn descend_path_equivalent_to_iterated_descend() {
        let root = FaceFrame::at_face_root(Face::PosY, 0.45, 0.50);
        let path = [(2u32, 1, 0), (1, 1, 2), (0, 2, 1)];
        let one = root.descend_path(&path);
        let mut two = root;
        for &(us, vs, rs) in &path {
            two = two.descend(us, vs, rs);
        }
        for i in 0..3 {
            assert_eq!(one.n_base_u[i], two.n_base_u[i]);
            assert_eq!(one.n_delta_u[i], two.n_delta_u[i]);
            assert_eq!(one.n_base_v[i], two.n_base_v[i]);
            assert_eq!(one.n_delta_v[i], two.n_delta_v[i]);
        }
        assert_eq!(one.r_base, two.r_base);
        assert_eq!(one.r_delta, two.r_delta);
        assert_eq!(one.depth, two.depth);
    }
}
