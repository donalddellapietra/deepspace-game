//! Debug-only camera control. Physics + continuous movement are
//! OFF — every position change is a one-shot teleport fired from
//! `apply_key`. Re-enable when surface gameplay is being verified.

use crate::camera::Camera;
use crate::world::tree::{NodeId, NodeLibrary};

/// Per-frame update. With debug movement on, this is just an
/// up-vector re-blend toward world `+Y`; nothing physical happens.
pub fn update(camera: &mut Camera, _velocity: &mut [f32; 3], dt: f32) {
    camera.update_up([0.0, 1.0, 0.0], dt);
}

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
/// nearest world axis). WASD movement: always cardinal, "regardless
/// of the sphere."
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
