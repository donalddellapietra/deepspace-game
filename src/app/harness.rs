//! Deterministic spawn/view construction for the render harness.
//!
//! The harness should not rely on memorized deep coordinates or a
//! fixed pitch that only happens to work at one layer. This module
//! derives a stable deep spawn from the bootstrap position and then
//! synthesizes a canonical center-ray view from the local scene.

use crate::camera::Camera;
use crate::world::anchor::{Path, WORLD_SIZE, WorldPos};
use crate::world::bootstrap::WorldPreset;
use crate::world::edit;
use crate::world::sdf;
use crate::world::state::WorldState;
use crate::world::tree::MAX_DEPTH;

use super::{frame_origin_size_world, ActiveFrame, ActiveFrameKind};

const TEST_UP: [f32; 3] = [0.0, 1.0, 0.0];
const BOUNDARY_EPS: f32 = 1e-6;
const STABLE_OFFSET: f32 = 0.375;
const YAW_OFFSETS: [f32; 7] = [0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45];
const PITCH_OFFSETS: [f32; 7] = [0.0, -0.12, 0.12, -0.24, 0.24, -0.36, 0.36];

pub(crate) fn spawn_position(
    spawn_xyz: [f32; 3],
    target_depth: u8,
    reference_depth: u8,
) -> WorldPos {
    let base_depth = target_depth.min(reference_depth);
    let base = WorldPos::from_world_xyz(spawn_xyz, base_depth);
    let position = if target_depth <= base_depth {
        base
    } else {
        base.deepened_to(target_depth)
    };
    stabilize_spawn(position)
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
    world_preset: WorldPreset,
    reference_depth: u8,
    fallback_yaw: f32,
    fallback_pitch: f32,
    explicit_yaw: Option<f32>,
    explicit_pitch: Option<f32>,
) -> (f32, f32) {
    let (base_yaw, base_pitch) = canonical_view_angles(
        world,
        position,
        world_preset,
        reference_depth,
        fallback_yaw,
        fallback_pitch,
    );
    let seed_yaw = explicit_yaw.unwrap_or(base_yaw);
    let seed_pitch = explicit_pitch.unwrap_or(base_pitch);

    let mut best = (seed_yaw, seed_pitch);
    let world_probe_depth = reference_depth.clamp(1, MAX_DEPTH as u8) as u32;
    for &pitch_offset in &PITCH_OFFSETS {
        for &yaw_offset in &YAW_OFFSETS {
            let yaw = seed_yaw + yaw_offset;
            let pitch = (seed_pitch + pitch_offset).clamp(-1.55, 1.55);
            if center_world_raycast_hit(world, position, yaw, pitch, world_probe_depth) {
                return (yaw, pitch);
            }
            if yaw_offset == 0.0 && pitch_offset == 0.0 {
                best = (yaw, pitch);
            }
        }
    }
    best
}

fn canonical_view_angles(
    world: &WorldState,
    position: WorldPos,
    world_preset: WorldPreset,
    reference_depth: u8,
    fallback_yaw: f32,
    fallback_pitch: f32,
) -> (f32, f32) {
    let cam_world = position.to_world_xyz();
    let up = TEST_UP;
    let down = match world_preset {
        WorldPreset::PlainTest => [0.0, -1.0, 0.0],
        WorldPreset::DemoSphere => [0.0, -1.0, 0.0],
    };
    let probe_depth = reference_depth.clamp(1, MAX_DEPTH as u8) as u32;
    let Some(surface_hit) = edit::cpu_raycast(
        &world.library,
        world.root,
        cam_world,
        down,
        probe_depth,
    ) else {
        return (fallback_yaw, fallback_pitch);
    };
    let (aabb_min, aabb_max) = edit::hit_aabb(&world.library, &surface_hit);
    let center = [
        (aabb_min[0] + aabb_max[0]) * 0.5,
        (aabb_min[1] + aabb_max[1]) * 0.5,
        (aabb_min[2] + aabb_max[2]) * 0.5,
    ];
    let cell_size = (aabb_max[0] - aabb_min[0]).max(1e-5);
    let (_, tangent_forward) = sdf::tangent_basis(up);
    let height_above_surface = sdf::dot(sdf::sub(cam_world, center), up).abs();
    let tangent_offset = (height_above_surface * 1.25).max(cell_size * 3.0);
    let target = sdf::add(center, sdf::scale(tangent_forward, tangent_offset));
    let forward = sdf::normalize(sdf::sub(target, cam_world));
    if sdf::length(forward) < 1e-5 {
        return (fallback_yaw, fallback_pitch);
    }
    yaw_pitch_for_forward(up, forward)
}

fn yaw_pitch_for_forward(up: [f32; 3], forward: [f32; 3]) -> (f32, f32) {
    let forward = sdf::normalize(forward);
    let pitch = sdf::dot(forward, up).asin().clamp(-1.55, 1.55);
    let horiz = sdf::normalize(sdf::sub(forward, sdf::scale(up, sdf::dot(forward, up))));
    let (tangent_right, tangent_forward) = sdf::tangent_basis(up);
    let sin_yaw = -sdf::dot(horiz, tangent_right);
    let cos_yaw = sdf::dot(horiz, tangent_forward);
    let yaw = sin_yaw.atan2(cos_yaw);
    (yaw, pitch)
}

fn ray_dir_in_frame(forward_world: [f32; 3], frame_path: &Path) -> [f32; 3] {
    let (_, frame_size_world) = frame_origin_size_world(frame_path);
    let scale = WORLD_SIZE / frame_size_world.max(1e-30);
    sdf::scale(sdf::normalize(forward_world), scale)
}

pub(crate) fn center_world_raycast_hit(
    world: &WorldState,
    position: WorldPos,
    yaw: f32,
    pitch: f32,
    probe_depth: u32,
) -> bool {
    let camera = Camera {
        position,
        smoothed_up: TEST_UP,
        yaw,
        pitch,
    };
    edit::cpu_raycast(
        &world.library,
        world.root,
        position.to_world_xyz(),
        sdf::normalize(camera.forward()),
        probe_depth,
    )
    .is_some()
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
                if let Some(hit) = edit::cpu_raycast_in_frame(
                    &world.library,
                    world.root,
                    frame.render_path.as_slice(),
                    cam_local,
                    ray_dir,
                    depth,
                    edit_depth,
                ) {
                    if hit.path.len() < edit_depth as usize {
                        let target_world = edit::hit_point_world(
                            &world.library,
                            &hit,
                            position.to_world_xyz(),
                            sdf::normalize(camera.forward()),
                        );
                        let _ = edit::refine_cartesian_hit_to_depth(
                            &world.library,
                            &hit,
                            target_world,
                            edit_depth as usize,
                        );
                    }
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
        for depth in [32u8, 34, 36, 39] {
            let position = spawn_position(
                bootstrap.default_spawn_xyz,
                depth,
                bootstrap.default_spawn_depth,
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
                bootstrap.default_spawn_depth,
                bootstrap.default_spawn_yaw,
                bootstrap.default_spawn_pitch,
                None,
                None,
            );
            assert!(
                center_world_raycast_hit(
                    &bootstrap.world,
                    position,
                    yaw,
                    pitch,
                    bootstrap.default_spawn_depth as u32,
                ),
                "derived harness view should produce a stable world hit at depth {depth}; yaw={yaw} pitch={pitch} frame={:?}",
                frame.render_path.as_slice()
            );
        }
    }
}
