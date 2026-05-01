//! Phase 3 / Phase 5 verification scaffold for the wrapped-Cartesian
//! planet's curvature silhouette analyzer.
//!
//! Owner: Phase 3 implements the math; Phase 5 tunes the
//! `k(altitude)` ramp against this same analyzer. See
//! `docs/sphere-mercator/plan.md` §3 and §7 / §9.
//!
//! Eventually (per the plan):
//! - Render the planet from orbit at altitudes that exercise k=0,
//!   k=0.5, k=1.0.
//! - For each screenshot, sample the planet's bottom horizon row
//!   (bottom 10% of frame where the planet meets the sky), fit a
//!   parabola to that row, extract the second derivative.
//! - Assert d²y/dx² > threshold for k=1.0 (must look spherical),
//!   within tolerance of 0 for k=0.0 (must look flat), and
//!   monotonically increasing for intermediate k.
//!
//! This is the architectural quality gate. A `k(altitude)` ramp that
//! produces "horizon pop" between adjacent altitudes fails this test
//! (Phase 5's tuning loop iterates against it).

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn planet_silhouette_phase3_placeholder() {
    // Phase 3 fills this with the silhouette curvature analyzer
    // described above.
}
