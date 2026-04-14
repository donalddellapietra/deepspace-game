//! Per-frame player physics: gravity, velocity, and flight thrust.

use crate::camera::Camera;
use crate::input::Keys;
use crate::world::coords::ROOT_EXTENT;
use crate::world::cubesphere::SphericalPlanet;
use crate::world::sdf;
use crate::world::tree::NodeLibrary;

/// Step the camera forward one frame.
///
/// Radial gravity toward the cubed-sphere planet's center with
/// smoothstep falloff from the outer shell out to 2x the outer
/// radius. Inside the outer shell gravity is at full strength;
/// past the influence boundary it's zero, so flying up far
/// enough lets you escape. `target_up` blends between the
/// planet's radial and world +Y with the same weight, so the
/// horizon rotates gradually instead of snapping.
///
/// Movement mutates `camera.position` via `WorldPos::add_local`, so
/// the player crosses cell boundaries exactly (anchor re-anchors
/// on overflow, offset stays in `[0, 1)³`). Gravity/thrust math still
/// reads the legacy `[f32; 3]` via the `world_pos_f32` shim — the
/// gravity center of the demo planet is a world-space point, so
/// there's no benefit yet from expressing it anchor-relative. That
/// migration comes when the cubed-sphere body becomes a tree node.
pub fn update(
    camera: &mut Camera,
    velocity: &mut [f32; 3],
    keys: &Keys,
    cell_size: f32,
    cs_planet: Option<&SphericalPlanet>,
    library: &NodeLibrary,
    dt: f32,
) {
    let world_up = [0.0f32, 1.0, 0.0];
    let (target_up, gravity_acc) = if let Some(p) = cs_planet {
        let to_player = sdf::sub(camera.world_pos_f32(), p.center);
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

    let step_world = sdf::add(
        sdf::scale(*velocity, dt),
        sdf::scale(thrust, dt),
    );

    // Convert the world-space step into offset-units at the camera's
    // current anchor depth, then let `add_local` handle cell crossings
    // and bubble-up. One offset unit = one cell width at anchor depth,
    // which in world units is `ROOT_EXTENT / 3^depth`.
    let depth = camera.position.anchor.depth();
    let cell_world = ROOT_EXTENT / 3f32.powi(depth as i32);
    let delta_local = [
        step_world[0] / cell_world,
        step_world[1] / cell_world,
        step_world[2] / cell_world,
    ];
    let _transition = camera.position.add_local(delta_local, library);
    // Sphere entry/exit/seam transitions will be dispatched here once
    // the cubed-sphere body is represented as a tree node (step 8).
}
