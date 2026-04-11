//! Block targeting via voxel raycast.
//!
//! The world is addressed at the leaf layer — `1 Bevy unit = 1 leaf
//! voxel`, so we can run a plain DDA walk over `IVec3` grid cells
//! asking `solid_at_integer` for each step. The targeted hit is kept
//! both as an `IVec3` (for the wireframe gizmo) and as a `Position`
//! (for handing to `edit_leaf`).

use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::world::collision::{position_from_bevy, solid_at_integer};
use crate::world::position::Position;
use crate::world::render::{cell_size_at_layer, ROOT_ORIGIN};
use crate::world::{CameraZoom, WorldState};

const MAX_REACH: f32 = 20.0;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetedBlock>()
            .add_systems(Update, (update_target, draw_highlight));
    }
}

/// The block the crosshair is pointing at.
#[derive(Resource, Default)]
pub struct TargetedBlock {
    /// Bevy integer voxel coordinate of the hit, if any.
    pub hit: Option<IVec3>,
    /// Face normal at the hit, pointing away from the solid cell.
    pub normal: Option<IVec3>,
    /// The leaf `Position` corresponding to `hit`, for edit_leaf().
    pub hit_position: Option<Position>,
}

impl TargetedBlock {
    fn clear(&mut self) {
        self.hit = None;
        self.normal = None;
        self.hit_position = None;
    }
}

fn update_target(
    cam_q: Query<&GlobalTransform, With<FpsCam>>,
    world: Res<WorldState>,
    mut targeted: ResMut<TargetedBlock>,
) {
    targeted.clear();

    let Ok(cam) = cam_q.single() else {
        return;
    };
    let origin = cam.translation();
    let dir = cam.forward().as_vec3();

    let (hit, normal) = dda_world(&world, origin, dir, MAX_REACH);
    if let Some(h) = hit {
        let center = Vec3::new(h.x as f32 + 0.5, h.y as f32 + 0.5, h.z as f32 + 0.5);
        targeted.hit = Some(h);
        targeted.normal = normal;
        targeted.hit_position = position_from_bevy(center);
    }
}

/// Voxel DDA from `origin` in direction `dir`, stopping at the first
/// solid cell (via `solid_at_integer`) within `max_dist` leaf voxels.
fn dda_world(
    world: &WorldState,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> (Option<IVec3>, Option<IVec3>) {
    let mut pos = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );
    let step = IVec3::new(
        if dir.x >= 0.0 { 1 } else { -1 },
        if dir.y >= 0.0 { 1 } else { -1 },
        if dir.z >= 0.0 { 1 } else { -1 },
    );
    let inv = Vec3::new(
        if dir.x.abs() > 1e-10 { 1.0 / dir.x } else { f32::MAX },
        if dir.y.abs() > 1e-10 { 1.0 / dir.y } else { f32::MAX },
        if dir.z.abs() > 1e-10 { 1.0 / dir.z } else { f32::MAX },
    );
    let mut t_max = Vec3::new(
        ((if step.x > 0 { pos.x + 1 } else { pos.x }) as f32 - origin.x) * inv.x,
        ((if step.y > 0 { pos.y + 1 } else { pos.y }) as f32 - origin.y) * inv.y,
        ((if step.z > 0 { pos.z + 1 } else { pos.z }) as f32 - origin.z) * inv.z,
    );
    let t_delta = Vec3::new(
        (step.x as f32 * inv.x).abs(),
        (step.y as f32 * inv.y).abs(),
        (step.z as f32 * inv.z).abs(),
    );

    let mut normal = IVec3::ZERO;
    let mut dist: f32 = 0.0;
    let mut first = true;

    while dist < max_dist {
        if !first && solid_at_integer(world, pos) {
            return (Some(pos), Some(normal));
        }
        first = false;
        if t_max.x < t_max.y && t_max.x < t_max.z {
            dist = t_max.x;
            pos.x += step.x;
            t_max.x += t_delta.x;
            normal = IVec3::new(-step.x, 0, 0);
        } else if t_max.y < t_max.z {
            dist = t_max.y;
            pos.y += step.y;
            t_max.y += t_delta.y;
            normal = IVec3::new(0, -step.y, 0);
        } else {
            dist = t_max.z;
            pos.z += step.z;
            t_max.z += t_delta.z;
            normal = IVec3::new(0, 0, -step.z);
        }
    }
    (None, None)
}

fn draw_highlight(
    mut gizmos: Gizmos,
    targeted: Res<TargetedBlock>,
    zoom: Res<CameraZoom>,
) {
    let Some(hit) = targeted.hit else {
        return;
    };
    // `hit` is the leaf-voxel integer cell the raycast bottomed out
    // on. We need to draw an outline around the **view-layer cell**
    // that contains `hit`, sized `cell_size` Bevy units per axis.
    //
    // Critically, the voxel grid is aligned to `ROOT_ORIGIN`, NOT to
    // integer Bevy zero. With `ROOT_ORIGIN.{x,z} = -13` and
    // `cell_size = 5` (view layer 11), the view cells lie at
    // x = -13, -8, -3, 2, 7, ... — a different lattice from
    // `..., -10, -5, 0, 5, ...`. So we have to snap to the
    // root-aligned lattice by computing the cell index *relative to
    // `ROOT_ORIGIN`*.
    let cell_size = cell_size_at_layer(zoom.layer);
    let hit_center = hit.as_vec3() + Vec3::splat(0.5);
    let local = hit_center - ROOT_ORIGIN;
    let cell_idx = Vec3::new(
        (local.x / cell_size).floor(),
        (local.y / cell_size).floor(),
        (local.z / cell_size).floor(),
    );
    let cell_min = ROOT_ORIGIN + cell_idx * cell_size;
    let center = cell_min + Vec3::splat(cell_size * 0.5);
    gizmos.cube(
        Transform::from_translation(center).with_scale(Vec3::splat(cell_size * 1.02)),
        Color::WHITE,
    );
}
