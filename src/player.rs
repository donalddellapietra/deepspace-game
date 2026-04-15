//! Per-frame player physics built on `WorldPos::add_local`.
//!
//! Movement is continuous (held-key WASD + Space/Shift) and runs
//! through the anchor primitive, so cell crossings + face seams +
//! sphere transitions are all handled by the coordinate layer.
//!
//! Gravity pulls toward the body's world-space center (resolved
//! from the body's anchor path + `NodeKind::CubedSphereBody`
//! payload), with a smoothstep falloff out to `2 × outer_r`. The
//! camera's `smoothed_up` blends from world `+Y` toward the radial
//! direction over the same falloff so the horizon settles smoothly
//! as the player approaches the surface.
//!
//! Speeds scale with the camera's anchor cell width (`ROOT_EXTENT /
//! 3^depth`): one second of WASD moves you about half a cell at
//! every layer, so a zoom step doesn't change how movement *feels*.

use crate::camera::Camera;
use crate::input::Keys;
use crate::world::coords::{self, Path, ROOT_EXTENT, WorldPos};
use crate::world::sdf;
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

/// Cell widths per second for held-WASD flight thrust.
const FLIGHT_CELLS_PER_SEC: f32 = 0.5;
/// Maximum gravity velocity in cell-widths per second.
const TERMINAL_FALL_CELLS_PER_SEC: f32 = 4.0;
/// Gravity acceleration in cell-widths per second² inside the
/// influence radius. Set to `0.0` for pure-flight debug mode;
/// re-enable when surface gameplay is being verified.
const GRAVITY_CELLS_PER_SEC2: f32 = 0.0;
/// Velocity damping rate (1/s); higher = more drag.
const VELOCITY_DAMP_PER_SEC: f32 = 2.5;

pub fn update(
    camera: &mut Camera,
    velocity: &mut [f32; 3],
    keys: &Keys,
    body_anchor: &Path,
    library: &NodeLibrary,
    world_root: NodeId,
    dt: f32,
) {
    // ---- 1. Resolve body footprint in world units (if present). ----
    let body = resolve_body(library, world_root, body_anchor);

    // ---- 2. Up-vector + gravity acceleration. ----
    let world_up = [0.0f32, 1.0, 0.0];
    let cell_world = ROOT_EXTENT / 3f32.powi(camera.position.anchor.depth() as i32);
    let (target_up, gravity_world) = if let Some(b) = body {
        let player_world = coords::world_pos_to_f32(&camera.position);
        let to_player = sdf::sub(player_world, b.center);
        let r = sdf::length(to_player);
        let surface_r = b.outer_r_world;
        let influence_r = surface_r * 2.0;
        let weight = if r <= surface_r {
            1.0
        } else if r >= influence_r {
            0.0
        } else {
            let t = (r - surface_r) / (influence_r - surface_r);
            let s = 1.0 - t;
            s * s * (3.0 - 2.0 * s) // smoothstep
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
        // Gravity points toward body center (= -radial_up).
        let g_mag = GRAVITY_CELLS_PER_SEC2 * cell_world * weight;
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

    // ---- 3. Integrate gravity into persistent velocity. ----
    *velocity = sdf::add(*velocity, sdf::scale(gravity_world, dt));
    let damp = (-VELOCITY_DAMP_PER_SEC * dt).exp();
    *velocity = sdf::scale(*velocity, damp);
    // Cap fall speed so deep gravity wells don't tunnel.
    let v_mag = sdf::length(*velocity);
    let v_cap = TERMINAL_FALL_CELLS_PER_SEC * cell_world;
    if v_mag > v_cap {
        *velocity = sdf::scale(*velocity, v_cap / v_mag);
    }

    // ---- 4. Held-key WASD / Space / Shift thrust. ----
    let speed = FLIGHT_CELLS_PER_SEC * cell_world;
    let (fwd, right, up) = camera.basis();
    let mut d = [0.0f32; 3];
    if keys.w     { d = sdf::add(d, fwd); }
    if keys.s     { d = sdf::sub(d, fwd); }
    if keys.d     { d = sdf::add(d, right); }
    if keys.a     { d = sdf::sub(d, right); }
    if keys.space { d = sdf::add(d, up); }
    if keys.shift { d = sdf::sub(d, up); }
    let l = sdf::length(d);
    let thrust = if l > 1e-4 {
        sdf::scale(d, speed / l)
    } else {
        [0.0; 3]
    };

    // ---- 5. Per-frame world-space step → offset units → add_local. ----
    let step_world = sdf::add(
        sdf::scale(*velocity, dt),
        sdf::scale(thrust, dt),
    );
    let delta_local = [
        step_world[0] / cell_world,
        step_world[1] / cell_world,
        step_world[2] / cell_world,
    ];
    let _transition = camera.position.add_local(delta_local, library, world_root);
    // Sphere entry/exit/seam transitions surface here as `_transition`;
    // currently consumed silently — UI hooks land in a follow-up.
}

// ----------------------------------------- body resolution helper

struct ResolvedBody {
    center: [f32; 3],
    outer_r_world: f32,
}

fn resolve_body(
    lib: &NodeLibrary,
    world_root: NodeId,
    body_anchor: &Path,
) -> Option<ResolvedBody> {
    let mut id = world_root;
    for &slot in body_anchor.slots() {
        let node = lib.get(id)?;
        match node.children[slot as usize] {
            Child::Node(child_id) => id = child_id,
            _ => return None,
        }
    }
    let body_node = lib.get(id)?;
    let outer_r = match body_node.kind {
        NodeKind::CubedSphereBody { outer_r, .. } => outer_r,
        _ => return None,
    };
    let center = coords::world_pos_to_f32(
        &WorldPos { anchor: *body_anchor, offset: [0.5, 0.5, 0.5] },
    );
    let cell_size = ROOT_EXTENT / 3f32.powi(body_anchor.depth() as i32);
    Some(ResolvedBody { center, outer_r_world: outer_r * cell_size })
}

// ----------------------------------------- debug-mode teleport helpers
//
// Kept available even though continuous physics is back, so the debug
// hotkeys (`T` for body-teleport, `[`/`]` for one-cell nudges if you
// ever want them) can still call into `add_local` without going
// through the physics integrator.

/// Teleport the camera by exactly one cell at its current anchor
/// depth, along world axis `axis` (0=X, 1=Y, 2=Z) in direction `±1`.
pub fn teleport_one_cell(
    camera: &mut Camera,
    axis: u8,
    dir: i8,
    library: &NodeLibrary,
    world_root: NodeId,
) {
    debug_assert!(axis < 3 && (dir == 1 || dir == -1));
    let mut delta = [0.0f32; 3];
    delta[axis as usize] = dir as f32;
    let _ = camera.position.add_local(delta, library, world_root);
}

/// Teleport along the camera's horizontal forward (snapped to the
/// nearest world axis). Used by `T`-style debug shortcuts.
pub fn teleport_along_camera(
    camera: &mut Camera,
    component: CameraDir,
    library: &NodeLibrary,
    world_root: NodeId,
) {
    let (fwd, right, _up) = camera.basis();
    let v = match component {
        CameraDir::Forward  => fwd,
        CameraDir::Backward => [-fwd[0], -fwd[1], -fwd[2]],
        CameraDir::Right    => right,
        CameraDir::Left     => [-right[0], -right[1], -right[2]],
    };
    let (axis, dir) = snap_to_cardinal(v);
    teleport_one_cell(camera, axis, dir, library, world_root);
}

#[derive(Clone, Copy, Debug)]
pub enum CameraDir { Forward, Backward, Right, Left }

fn snap_to_cardinal(v: [f32; 3]) -> (u8, i8) {
    let ax = v[0].abs();
    let ay = v[1].abs();
    let az = v[2].abs();
    let axis = if ax >= ay && ax >= az { 0 }
               else if ay >= az { 1 }
               else { 2 };
    let dir = if v[axis as usize] >= 0.0 { 1 } else { -1 };
    (axis, dir)
}
