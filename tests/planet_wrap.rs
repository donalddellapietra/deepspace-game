//! Phase 2 verification scaffold for the wrapped-Cartesian planet.
//!
//! Owner: Phase 2 will implement. See `docs/sphere-mercator/plan.md` §6.
//!
//! Eventually:
//! - Walk-east-around test: spawn at planet-x = 0; run a script that
//!   nudges the camera east by exactly `x_extent_cells` cells; the
//!   resulting screenshot must be pixel-identical to the spawn
//!   screenshot (same view of the same column).
//! - Place-from-the-other-side test: place a block at planet-x = 5;
//!   spawn camera at planet-x = `x_extent_cells - 5 + 1` looking east.
//!   The placed block must appear in the screenshot.
//! - CPU raycast wraps: `cpu_raycast_in_frame` from
//!   `(planet_x = x_extent - 1, y = surface + 5)` looking east hits
//!   the cell at planet-x = 0; assert the hit Path's last slot.
//!
//! Until Phase 2 wires the wrap inside `march_cartesian` and
//! `cartesian::cpu_raycast_with_face_depth`, this is a placeholder.

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn planet_wrap_phase2_placeholder() {
    // Phase 2 fills this with the walk-around + place-from-other-side
    // verification described above.
}
