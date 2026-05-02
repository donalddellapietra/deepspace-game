//! Player physics — intentionally empty.
//!
//! All motion is debug-only now: WASD teleports one child chunk at
//! the current anchor depth, Space/Shift move along world Y, and
//! `App::debug_teleport` jumps to an arbitrary path. Gravity,
//! velocity integration, and flight thrust are gone — they'll come
//! back later under a different input mode once the coordinate
//! refactor's behavior is stabilized.

use crate::camera::Camera;

/// No-op placeholder so `App::update` keeps its dispatch shape.
/// The camera's orientation is now driven purely by mouse look and
/// the debug-only "up" stays at world +Y.
pub fn update(camera: &mut Camera, dt: f32) {
    let _ = dt;
    camera.update_up([0.0, 1.0, 0.0], 0.0);
}
