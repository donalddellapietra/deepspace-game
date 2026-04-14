//! Per-frame player physics: world-space flight with constant-Y gravity.

use crate::camera::Camera;
use crate::input::Keys;
use crate::world::sdf;
use crate::world::tree::NodeLibrary;

/// Step the camera forward one frame.
///
/// Velocity is stored in world-units-per-second. `add_local` expects
/// the delta in cell-local units, so each per-frame world-step is
/// divided by `cell_size()` before being applied. Gravity and flight
/// thrust scale with `cell_size` so holding W at any zoom feels the
/// same on screen.
pub fn update(
    camera: &mut Camera,
    velocity: &mut [f32; 3],
    keys: &Keys,
    lib: &NodeLibrary,
    dt: f32,
) {
    let cell_size = camera.cell_size();
    let world_up = [0.0f32, 1.0, 0.0];

    // Simple world-aligned up — no sphere gravity until sphere
    // rendering is revived via NodeKind dispatch.
    camera.update_up(world_up, dt);
    let gravity_acc = [0.0f32, -8.0 * cell_size, 0.0];
    *velocity = sdf::add(*velocity, sdf::scale(gravity_acc, dt));
    let damp = (-2.5_f32 * dt).exp();
    *velocity = sdf::scale(*velocity, damp);

    let speed = 5.0 * cell_size;
    let (fwd, right, up) = camera.basis();
    let mut d = [0.0f32; 3];
    if keys.w { d = sdf::add(d, fwd); }
    if keys.s { d = sdf::sub(d, fwd); }
    if keys.d { d = sdf::add(d, right); }
    if keys.a { d = sdf::sub(d, right); }
    if keys.space { d = sdf::add(d, up); }
    if keys.shift { d = sdf::sub(d, up); }
    let l = sdf::length(d);
    let thrust = if l > 1e-4 { sdf::scale(d, speed / l) } else { [0.0; 3] };

    let step_world = sdf::add(sdf::scale(*velocity, dt), sdf::scale(thrust, dt));
    let inv = 1.0 / cell_size;
    let delta = [step_world[0] * inv, step_world[1] * inv, step_world[2] * inv];
    camera.position.add_local(delta, lib);
}
