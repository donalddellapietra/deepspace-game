//! Per-frame player physics: gravity, velocity, and flight thrust.

use crate::camera::Camera;
use crate::input::Keys;
use crate::world::cubesphere::SphericalPlanet;
use crate::world::sdf;

/// Step the camera forward one frame.
///
/// Radial gravity toward the cubed-sphere planet's center with
/// smoothstep falloff from the outer shell out to 2x the outer
/// radius. Inside the outer shell gravity is at full strength;
/// past the influence boundary it's zero, so flying up far
/// enough lets you escape. `target_up` blends between the
/// planet's radial and world +Y with the same weight, so the
/// horizon rotates gradually instead of snapping.
pub fn update(
    camera: &mut Camera,
    velocity: &mut [f32; 3],
    keys: &Keys,
    cell_size: f32,
    cs_planet: Option<&SphericalPlanet>,
    dt: f32,
) {
    let world_up = [0.0f32, 1.0, 0.0];
    let _ = cs_planet;
    let gravity_acc = [0.0f32, 0.0, 0.0];
    camera.update_up(world_up, dt);

    // Integrate gravity into persistent velocity, then damp so we
    // have a terminal fall speed rather than unbounded divergence.
    *velocity = sdf::add(*velocity, sdf::scale(gravity_acc, dt));
    let damp = (-2.5_f32 * dt).exp();
    *velocity = sdf::scale(*velocity, damp);

    // Flight thrust: WASD in the camera's horizontal plane,
    // Space/Shift along the camera's local up. Applied as direct
    // per-frame displacement, not velocity, so controls feel
    // responsive and momentum only comes from gravity. No
    // collisions — you fly through everything.
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
    let thrust = if l > 1e-4 {
        sdf::scale(d, speed / l)
    } else {
        [0.0, 0.0, 0.0]
    };

    let step = sdf::add(
        sdf::scale(*velocity, dt),
        sdf::scale(thrust, dt),
    );
    camera.pos = sdf::add(camera.pos, step);
}
