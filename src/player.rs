//! Per-frame player physics: continuous thrust + radial gravity,
//! integrated through the anchor-based coordinate primitives.
//!
//! Everything here is in "offset units / sec" at the camera's
//! current anchor depth. One offset unit = one cell at that depth,
//! so the player always crosses `WALK_SPEED` cells per second
//! regardless of zoom — the Minecraft-scaling intuition where a
//! 10-block-tall NPC covers 10× the ground.

use crate::camera::Camera;
use crate::input::Keys;
use crate::world::coords::{self, Path, Transition, WorldPos, ROOT_EXTENT};
use crate::world::sdf;
use crate::world::tree::{NodeId, NodeLibrary};

/// Walk speed in offset-units/sec (i.e. cells/sec at the anchor depth).
const WALK_SPEED: f32 = 5.0;
/// Gravity strength at the surface, in offset-units/sec².
const GRAVITY_STRENGTH: f32 = 20.0;
/// Velocity damp per second (multiplicative, `e^(-k·dt)`). Higher =
/// tighter stop-on-release.
const VELOCITY_DAMP: f32 = 1.5;
/// Maximum distance (in world units) beyond the body's outer shell
/// within which gravity applies. Outside this, the player flies free.
const GRAVITY_INFLUENCE_MULT: f32 = 2.0;

/// Per-frame update. Reads `keys` for thrust, the `body_anchor` for
/// gravity, and mutates camera.position + velocity + camera.smoothed_up.
///
/// Thrust is direct displacement (release WASD, stop instantly).
/// Gravity integrates into `velocity` so falling accelerates.
pub fn update(
    camera: &mut Camera,
    velocity: &mut [f32; 3],
    keys: &Keys,
    body_anchor: Path,
    lib: &NodeLibrary,
    world_root: NodeId,
    dt: f32,
) {
    // 1. Thrust from input. WASD in camera's horizontal basis;
    //    Space/Shift along world up. Returns a world-axis vector
    //    whose magnitude is WALK_SPEED (or 0 if no keys).
    let thrust = compute_thrust(camera, keys);

    // 2. Gravity: world-space acceleration pulling camera toward
    //    body center, with linear falloff past the shell.
    let (gravity_vec, has_gravity) = compute_gravity(camera, body_anchor);

    // 3. Integrate: velocity += gravity * dt, then damp, then apply.
    for i in 0..3 {
        velocity[i] += gravity_vec[i] * dt;
    }
    let damp = (-VELOCITY_DAMP * dt).exp();
    for i in 0..3 {
        velocity[i] *= damp;
    }

    // 4. Total offset-space step. Works because Cartesian offset axes
    //    align with world axes, so a world-direction scaled by a
    //    magnitude-per-second IS an offset-space delta of the same
    //    magnitude (measured in cells at the current anchor depth).
    let mut step = [0.0f32; 3];
    for i in 0..3 {
        step[i] = (velocity[i] + thrust[i]) * dt;
    }

    // 5. Move through the coordinate primitive; observe any
    //    coordinate-meaning transition that was crossed.
    let transition = camera.position.add_local(step, lib, world_root);
    handle_transition(transition, velocity);

    // 6. Target up-vector: radial out from body when gravity is
    //    active, world +Y otherwise. smoothed_up lerps toward it.
    let target_up = compute_target_up(camera, body_anchor, has_gravity);
    camera.update_up(target_up, dt);
}

fn compute_thrust(camera: &Camera, keys: &Keys) -> [f32; 3] {
    let (fwd, right, up) = camera.basis();
    let mut t = [0.0f32; 3];
    if keys.w { t = sdf::add(t, fwd); }
    if keys.s { t = sdf::sub(t, fwd); }
    if keys.d { t = sdf::add(t, right); }
    if keys.a { t = sdf::sub(t, right); }
    if keys.space { t = sdf::add(t, up); }
    if keys.shift { t = sdf::sub(t, up); }
    let len = sdf::length(t);
    if len < 1e-6 {
        [0.0; 3]
    } else {
        let k = WALK_SPEED / len;
        [t[0] * k, t[1] * k, t[2] * k]
    }
}

/// Returns `(gravity_acc, has_gravity)`. Gravity is in offset-units/sec²
/// (equivalently world-units/sec² because axes align for Cartesian).
fn compute_gravity(camera: &Camera, body_anchor: Path) -> ([f32; 3], bool) {
    // Body center in world space = anchor-cell center.
    let body_center = coords::world_pos_to_f32(&WorldPos {
        anchor: body_anchor,
        offset: [0.5, 0.5, 0.5],
    });
    let cam_world = coords::world_pos_to_f32(&camera.position);
    let to_body = sdf::sub(body_center, cam_world);
    let dist = sdf::length(to_body);

    // Body cell world size; gravity extends out GRAVITY_INFLUENCE_MULT ×
    // that much beyond the cell center.
    let body_cell_world = ROOT_EXTENT / 3f32.powi(body_anchor.depth() as i32);
    let influence_dist = body_cell_world * GRAVITY_INFLUENCE_MULT;

    if dist < 1e-6 || dist >= influence_dist {
        return ([0.0; 3], false);
    }
    // Smooth falloff: full at center, zero at edge.
    let t = dist / influence_dist;
    let mag = GRAVITY_STRENGTH * (1.0 - t) * (1.0 - t);
    let dir = [to_body[0] / dist, to_body[1] / dist, to_body[2] / dist];
    ([dir[0] * mag, dir[1] * mag, dir[2] * mag], true)
}

fn compute_target_up(
    camera: &Camera,
    body_anchor: Path,
    has_gravity: bool,
) -> [f32; 3] {
    if !has_gravity {
        return [0.0, 1.0, 0.0];
    }
    let body_center = coords::world_pos_to_f32(&WorldPos {
        anchor: body_anchor,
        offset: [0.5, 0.5, 0.5],
    });
    let cam_world = coords::world_pos_to_f32(&camera.position);
    // Up = away from body center = radial out.
    let d = sdf::sub(cam_world, body_center);
    let l = sdf::length(d);
    if l < 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [d[0] / l, d[1] / l, d[2] / l]
    }
}

/// React to a coordinate-frame change emitted by `add_local`. Today we
/// just log and zero velocity on sphere entry/exit to prevent the
/// integrated-gravity vector from carrying stale direction across a
/// frame transition. More nuanced handling (basis re-expression,
/// smooth orientation blending) can layer on later.
fn handle_transition(t: Transition, velocity: &mut [f32; 3]) {
    match t {
        Transition::None => {}
        Transition::SphereEntry { .. } => {
            log::info!("Transition: sphere entry");
            *velocity = [0.0; 3];
        }
        Transition::SphereExit { .. } => {
            log::info!("Transition: sphere exit");
            *velocity = [0.0; 3];
        }
        Transition::FaceEntry { face } => {
            log::info!("Transition: face entry ({:?})", face);
            *velocity = [0.0; 3];
        }
        Transition::FaceExit { face } => {
            log::info!("Transition: face exit ({:?})", face);
            *velocity = [0.0; 3];
        }
        Transition::CubeSeam { from_face, to_face } => {
            log::info!("Transition: cube seam {:?} → {:?}", from_face, to_face);
            // Axis-remapping across the seam will land here once
            // `face_transitions::seam_neighbor` is wired into
            // `add_local`. For now the anchor was stepped in the old
            // face's axes; the visible effect is a slight slide in
            // the wrong u/v direction. Zero velocity to avoid
            // compounding.
            *velocity = [0.0; 3];
        }
    }
}
