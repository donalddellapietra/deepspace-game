//! Per-frame player physics: gravity, velocity, and flight thrust.

use crate::camera::Camera;
use crate::input::Keys;
use crate::world::cubesphere::SphericalPlanet;
use crate::world::sdf;
use crate::world::tree::NodeLibrary;

/// Step the camera forward one frame.
///
/// Radial gravity toward the cubed-sphere planet's center with
/// smoothstep falloff from the outer shell out to 2x the outer
/// radius. Motion integrates through `WorldPos::add_local` so the
/// camera's anchor advances as the player crosses cell boundaries.
pub fn update(
    camera: &mut Camera,
    velocity: &mut [f32; 3],
    keys: &Keys,
    cs_planet: Option<&SphericalPlanet>,
    lib: &NodeLibrary,
    dt: f32,
) {
    let cell_size = camera.cell_size();
    let world_up = [0.0f32, 1.0, 0.0];
    let cam_world = camera.world_pos_f32();

    let (target_up, gravity_acc) = if let Some(p) = cs_planet {
        let to_player = sdf::sub(cam_world, p.center);
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
