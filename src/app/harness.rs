//! Deterministic spawn/view construction for the render harness.
//!
//! The harness should not rely on memorized deep coordinates or a
//! fixed pitch that only happens to work at one layer. This module
//! derives a stable deep spawn from the bootstrap position and then
//! synthesizes a canonical center-ray view from the local scene.

use crate::camera::Camera;
use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldPreset;
#[cfg(test)]
use crate::world::bootstrap::PLAIN_SURFACE_Y;
use crate::world::edit;
use crate::world::sdf;
use crate::world::state::WorldState;
use crate::world::tree::{slot_coords, slot_index};

use super::{ActiveFrame, ActiveFrameKind, RENDER_FRAME_K};

const TEST_UP: [f32; 3] = [0.0, 1.0, 0.0];
const BOUNDARY_EPS: f32 = 1e-6;
const STABLE_OFFSET: f32 = 0.375;
const PLAIN_SURFACE_OFFSET: [f32; 3] = [0.5, 0.65, 0.5];
const YAW_OFFSETS: [f32; 7] = [0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45];
const PITCH_OFFSETS: [f32; 7] = [0.0, -0.12, 0.12, -0.24, 0.24, -0.36, 0.36];

pub(crate) fn spawn_position(
    world_preset: WorldPreset,
    spawn_xyz: [f32; 3],
    target_depth: u8,
    reference_depth: u8,
) -> WorldPos {
    if matches!(world_preset, WorldPreset::PlainTest) && target_depth > 0 {
        return plain_surface_spawn(spawn_xyz, target_depth, reference_depth);
    }
    let base_depth = target_depth.min(reference_depth);
    let base = WorldPos::from_root_local(spawn_xyz, base_depth);
    let position = if target_depth <= base_depth {
        base
    } else {
        base.deepened_to(target_depth)
    };
    stabilize_spawn(position)
}

fn plain_surface_spawn(spawn_xyz: [f32; 3], target_depth: u8, reference_depth: u8) -> WorldPos {
    let base_depth = target_depth.min(reference_depth).max(1);
    let base = WorldPos::from_root_local(spawn_xyz, base_depth).deepened_to(target_depth);
    let mut anchor = Path::root();
    for depth in 0..target_depth as usize {
        let (sx, _, sz) = slot_coords(base.anchor.slot(depth) as usize);
        // The flat plain surface sits at y = 1.5, whose ternary
        // expansion is 1.11111... . Keeping every deep y-slot at 1
        // makes the camera converge to the surface from above.
        let sy = 1;
        anchor.push(slot_index(sx, sy, sz) as u8);
    }
    WorldPos::new(anchor, PLAIN_SURFACE_OFFSET)
}

pub(crate) fn stabilize_spawn(mut position: WorldPos) -> WorldPos {
    for axis in 0..3 {
        if position.offset[axis] <= BOUNDARY_EPS || position.offset[axis] >= 1.0 - BOUNDARY_EPS {
            position.offset[axis] = STABLE_OFFSET;
        }
    }
    position
}

pub(crate) fn derive_view_angles(
    world: &WorldState,
    position: WorldPos,
    _world_preset: WorldPreset,
    reference_depth: u8,
    fallback_yaw: f32,
    fallback_pitch: f32,
    explicit_yaw: Option<f32>,
    explicit_pitch: Option<f32>,
) -> (f32, f32) {
    let mut logical_path = position.anchor;
    logical_path.truncate(reference_depth.saturating_sub(RENDER_FRAME_K));
    let frame = crate::app::App::frame_for_logical_path(&world.library, world.root, &logical_path);
    let base_yaw = fallback_yaw;
    let base_pitch = fallback_pitch;
    let seed_yaw = explicit_yaw.unwrap_or(base_yaw);
    let seed_pitch = explicit_pitch.unwrap_or(base_pitch);

    let mut best = (seed_yaw, seed_pitch);
    for &pitch_offset in &PITCH_OFFSETS {
        for &yaw_offset in &YAW_OFFSETS {
            let yaw = seed_yaw + yaw_offset;
            let pitch = (seed_pitch + pitch_offset).clamp(-1.55, 1.55);
            if center_frame_raycast_hit(world, &frame, position, yaw, pitch) {
                return (yaw, pitch);
            }
            if yaw_offset == 0.0 && pitch_offset == 0.0 {
                best = (yaw, pitch);
            }
        }
    }
    best
}

fn ray_dir_in_frame(forward_world: [f32; 3], frame_path: &Path) -> [f32; 3] {
    let _ = frame_path;
    sdf::normalize(forward_world)
}

pub(crate) fn center_frame_raycast_hit(
    world: &WorldState,
    frame: &ActiveFrame,
    position: WorldPos,
    yaw: f32,
    pitch: f32,
) -> bool {
    let camera = Camera {
        position,
        smoothed_up: TEST_UP,
        yaw,
        pitch,
    };
    let edit_depth = position.anchor.depth().saturating_sub(1).max(1) as u32;
    match frame.kind {
        ActiveFrameKind::Sphere(sphere) => {
            let cam_body = position.in_frame(&sphere.body_path);
            let ray_dir_local = ray_dir_in_frame(camera.forward(), &sphere.body_path);
            edit::cpu_raycast_in_sphere_frame(
                &world.library,
                world.root,
                sphere.body_path.as_slice(),
                cam_body,
                cam_body,
                ray_dir_local,
                edit_depth,
                sphere.face as u32,
                sphere.face_u_min,
                sphere.face_v_min,
                sphere.face_r_min,
                sphere.face_size,
                sphere.inner_r,
                sphere.outer_r,
                sphere.face_depth,
            )
            .is_some()
        }
        ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
            let cam_local = position.in_frame(&frame.render_path);
            let ray_dir = ray_dir_in_frame(camera.forward(), &frame.render_path);
            let min_depth = 1;
            let mut depth = edit_depth;
            while depth >= min_depth {
                if edit::cpu_raycast_in_frame(
                    &world.library,
                    world.root,
                    frame.render_path.as_slice(),
                    cam_local,
                    ray_dir,
                    depth,
                    edit_depth,
                )
                .is_some()
                {
                    return true;
                }
                if depth == min_depth {
                    break;
                }
                depth -= 1;
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::{bootstrap_world, WorldPreset};

    #[test]
    fn derived_plain_harness_view_hits_representative_layers() {
        let bootstrap = bootstrap_world(WorldPreset::PlainTest, Some(40));
        for depth in [39u8, 36, 34, 32, 22, 20, 18, 16] {
            let position = spawn_position(
                WorldPreset::PlainTest,
                bootstrap.default_spawn_pos.in_frame(&Path::root()),
                depth,
                bootstrap.default_spawn_pos.anchor.depth(),
            );
            let mut logical_path = position.anchor;
            let desired_depth = depth.saturating_sub(crate::app::RENDER_FRAME_K);
            logical_path.truncate(desired_depth);
            let frame = crate::app::App::frame_for_logical_path(
                &bootstrap.world.library,
                bootstrap.world.root,
                &logical_path,
            );
            let (yaw, pitch) = derive_view_angles(
                &bootstrap.world,
                position,
                WorldPreset::PlainTest,
                bootstrap.default_spawn_pos.anchor.depth(),
                bootstrap.default_spawn_yaw,
                bootstrap.default_spawn_pitch,
                None,
                None,
            );
            assert!(
                center_frame_raycast_hit(&bootstrap.world, &frame, position, yaw, pitch),
                "derived harness view should produce a stable frame-local hit at depth {depth}; yaw={yaw} pitch={pitch} frame={:?}",
                frame.render_path.as_slice()
            );
        }
    }

    #[test]
    fn plain_surface_spawn_hugs_surface_at_representative_layers() {
        let bootstrap = bootstrap_world(WorldPreset::PlainTest, Some(40));
        for depth in [39u8, 36, 34, 32, 22, 20, 18, 16] {
            let position = spawn_position(
                WorldPreset::PlainTest,
                bootstrap.default_spawn_pos.in_frame(&Path::root()),
                depth,
                bootstrap.default_spawn_pos.anchor.depth(),
            );
            let y0 = slot_coords(position.anchor.slot(0) as usize).1;
            assert_eq!(y0, 1, "plain spawn must stay in the air layer at depth {depth}");
            for level in 1..position.anchor.depth() as usize {
                let y = slot_coords(position.anchor.slot(level) as usize).1;
                assert_eq!(
                    y, 1,
                    "plain spawn should track the flat y=1.5 surface ternary prefix at depth {depth}, level {level}"
                );
            }
            assert!(
                position.offset[1] > 0.5,
                "plain spawn should sit above the surface boundary at depth {depth}: {:?}",
                position.offset,
            );
            let root_y = position.in_frame(&Path::root())[1];
            assert!(
                root_y > PLAIN_SURFACE_Y,
                "plain spawn must be above the surface at depth {depth}: y={root_y}",
            );
        }
    }
}
