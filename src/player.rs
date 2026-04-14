//! Per-frame player physics.
//!
//! Testing mode: gravity is disabled and flight speed is a fixed
//! world-units-per-second constant, independent of the camera's
//! anchor depth. Re-enable the body-pull + depth-proportional speed
//! once the refactor is visually verified.

use crate::camera::Camera;
use crate::input::Keys;
use crate::world::coords::{Path, ROOT_EXTENT};
use crate::world::sdf;
use crate::world::tree::{NodeId, NodeLibrary};

/// Fixed world-space flight speed (units per second). Chosen so
/// crossing a root-extent cell (3 units) takes ~1s.
const FLIGHT_SPEED: f32 = 3.0;

/// Step the camera forward one frame. Gravity disabled for testing;
/// flight thrust is direct per-frame displacement in world units.
/// The final step is converted to offset units at the camera's
/// current anchor depth and handed to `WorldPos::add_local`.
pub fn update(
    camera: &mut Camera,
    velocity: &mut [f32; 3],
    keys: &Keys,
    _cell_size: f32,
    _body_anchor: &Path,
    library: &NodeLibrary,
    world_root: NodeId,
    dt: f32,
) {
    // Clear any residual velocity from a previous frame — gravity off.
    *velocity = [0.0, 0.0, 0.0];
    camera.update_up([0.0, 1.0, 0.0], dt);

    let (fwd, right, up) = camera.basis();
    let mut d = [0.0f32; 3];
    if keys.w { d = sdf::add(d, fwd); }
    if keys.s { d = sdf::sub(d, fwd); }
    if keys.d { d = sdf::add(d, right); }
    if keys.a { d = sdf::sub(d, right); }
    if keys.space { d = sdf::add(d, up); }
    if keys.shift { d = sdf::sub(d, up); }
    let l = sdf::length(d);
    let step_world = if l > 1e-4 {
        sdf::scale(d, FLIGHT_SPEED * dt / l)
    } else {
        return;
    };

    // Convert the world-space step into offset units at the camera's
    // anchor depth; `add_local` handles cell crossings and bubble-up.
    let depth = camera.position.anchor.depth();
    let cell_world = ROOT_EXTENT / 3f32.powi(depth as i32);
    let delta_local = [
        step_world[0] / cell_world,
        step_world[1] / cell_world,
        step_world[2] / cell_world,
    ];
    let _transition = camera.position.add_local(delta_local, library, world_root);
}
