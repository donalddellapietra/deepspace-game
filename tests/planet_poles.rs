//! Phase 4 verification scaffold for the wrapped-Cartesian planet's
//! polar-cap impostor.
//!
//! Owner: Phase 4 will implement. See `docs/sphere-mercator/plan.md` §8.
//!
//! Eventually:
//! - Spawn at the equator, look up at 80° pitch. Screenshot must show
//!   the polar cap impostor (not sky) at the top of the view.
//! - Spawn from orbit, look at the planet. The top and bottom of the
//!   disk silhouette must be solid white-ish, not black-with-a-hole.
//! - `place_block` whose Path falls in the polar Y rows must return
//!   `false` (gameplay-side validation; the polar band is banned at
//!   the edit boundary, not in the data model).
//!
//! Until Phase 4 wires the impostor pass + the `place_block` polar
//! guard, this is a placeholder.

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn planet_poles_phase4_placeholder() {
    // Phase 4 fills this with the cap-visibility + place-rejection
    // verification described above.
}
