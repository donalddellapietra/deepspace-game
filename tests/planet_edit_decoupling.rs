//! Phase 6 verification scaffold: edit ray hits the UNBENT cell while
//! the GPU shows a curved planet.
//!
//! Owner: Phase 6 will implement. See
//! `docs/sphere-mercator/plan.md` §10 ("Edit decoupling at all
//! altitudes").
//!
//! Eventually:
//! - At orbital altitude (camera at depth ~4 looking through the
//!   planet's centre), `cpu_raycast_in_frame` returns the FLAT (un-
//!   bent) ray's hit. The GPU render shows the planet curved.
//! - A click on the planet edits the UNBENT cell — the placed block
//!   appears at the unbent location, not at the visible curved
//!   location. Verify by comparing the edit-ray's Path to the GPU
//!   walker probe's terminal Path: they should differ at orbital
//!   altitudes when curvature is on.
//!
//! This is the function-boundary regression test for the rule
//! "gameplay rays are straight, presentation rays are bent" — see
//! the proposal §"Decoupling simulation from presentation".

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn planet_edit_decoupling_phase6_placeholder() {
    // Phase 6 fills this with the edit-vs-render decoupling test
    // described above.
}
