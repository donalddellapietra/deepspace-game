//! Phase 1 verification scaffold for the wrapped-Cartesian planet.
//!
//! Owner: Phase 1 will implement. See `docs/sphere-mercator/plan.md` §5.
//!
//! Eventually:
//! - Spawn the camera at the planet's surface at depth 8 with
//!   `--planet-world` (added in Phase 1).
//! - Capture a screenshot showing a flat slab of grass/dirt/stone
//!   rendered identically to a plain-world subtree of the same
//!   dimensions.
//! - Run a CPU `cpu_raycast_in_frame` from a known camera position
//!   against the planet — assert the path's first slot is `(1,1,1)`
//!   (= world centre) and depth ≥ 6.
//! - Use the harness `--script "place,wait:10,screenshot:tmp/p1-edit.png"`
//!   to verify a single placed block lands on the slab surface via
//!   the unmodified `cpu_raycast_in_frame` + `place_block` path.
//!
//! Until Phase 1 lands the planet preset, this test compiles to a
//! placeholder. The placeholder is NOT `#[ignore]`d because it
//! asserts nothing — `feedback_no_ignore_hide_bugs` says "don't
//! `#[ignore]` real failing tests"; an empty placeholder isn't
//! one.

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn planet_basic_phase1_placeholder() {
    // Phase 1 fills this with the slab-render verification described
    // above. Until then, this test exists only to reserve the file
    // path the implementing agent will edit.
}
