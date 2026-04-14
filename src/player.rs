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
    let (target_up, gravity_acc) = if let Some(p) = cs_planet {
        let to_player = sdf::sub(camera.position.pos_in_ancestor_frame(0), p.center);
        let r = sdf::length(to_player);
        let surface_r = p.outer_r;
        let influence_r = surface_r * 2.0;
        let weight = if r <= surface_r {
            1.0
        } else if r >= influence_r {
            0.0
        } else {
            let t = (r - surface_r) / (influence_r - surface_r);
            let s = 1.0 - t;
            s * s * (3.0 - 2.0 * s)
        };
        let radial_up = if r > 1e-6 {
            sdf::scale(to_player, 1.0 / r)
        } else {
            world_up
        };
        let up_blend = sdf::normalize(sdf::add(
            sdf::scale(radial_up, weight),
            sdf::scale(world_up, 1.0 - weight),
        ));
        // Gravity magnitude scales with cell_size so fall speed
        // is visible at every zoom. Tuned so holding Space
        // (thrust = 5 * cell_size) noticeably overpowers gravity
        // and you can fly off the planet.
        let g_mag = 8.0 * cell_size * weight;
        let grav = if r > 1e-6 {
            sdf::scale(radial_up, -g_mag)
        } else {
            [0.0, 0.0, 0.0]
        };
        (up_blend, grav)
    } else {
        (world_up, [0.0, 0.0, 0.0])
    };
    camera.update_up(target_up, dt);

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
    // Integrate motion path-based. `step` is a world-space delta;
    // at the camera's anchoring depth each cell is 3^(1 - depth)
    // wide in world units, so offset delta = world delta / cell_size
    // = world delta * 3^(depth - 1). add_offset handles carry.
    let depth = camera.position.depth as i32;
    let inv_cell = 3.0f32.powi(depth - 1);
    camera.position.add_offset([
        step[0] * inv_cell,
        step[1] * inv_cell,
        step[2] * inv_cell,
    ]);
}
