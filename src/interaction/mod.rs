//! Block targeting via a view-layer-aware voxel raycast.
//!
//! At view layer L the world is rendered as cells of
//! `cell_size_at_layer(L)` Bevy units; the player picks and edits at
//! that same granularity. We DDA-walk one view-cell per iteration
//! (16 steps regardless of L) and store the hit as a [`LayerPos`] —
//! a path-based representation with no globally-growing integers, so
//! the same code works at `MAX_LAYER = 12` today and at 15+ later.

use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::world::collision::{
    bevy_origin_of_layer_pos, is_layer_pos_solid, layer_pos_from_bevy,
};
use crate::world::position::LayerPos;
use crate::world::render::{cell_size_at_layer, ROOT_ORIGIN};
use crate::world::{CameraZoom, WorldState};

/// Maximum reach in **cells** at the current view layer. The cell
/// DDA takes at most this many steps, so picking and placement work
/// the same way at every zoom level: 16 cells in front, regardless
/// of whether one cell is 1 leaf voxel (L=12) or 625 leaf voxels (L=8).
const MAX_REACH_CELLS: i32 = 16;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetedBlock>()
            .add_systems(Update, (update_target, draw_highlight));
    }
}

/// The view-layer cell the crosshair is pointing at.
#[derive(Resource, Default)]
pub struct TargetedBlock {
    /// The hit cell, as a [`LayerPos`] at the current view layer.
    /// `path.len() == lp.layer`, `cell` in `0..25`.
    pub hit_layer_pos: Option<LayerPos>,
    /// Face normal at the hit, in CELL units. Add `normal *
    /// cell_size_at_layer(lp.layer)` to the cell's centre to get
    /// the placement cell's centre.
    pub normal: Option<IVec3>,
}

impl TargetedBlock {
    fn clear(&mut self) {
        self.hit_layer_pos = None;
        self.normal = None;
    }
}

fn update_target(
    cam_q: Query<&GlobalTransform, With<FpsCam>>,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    mut targeted: ResMut<TargetedBlock>,
) {
    targeted.clear();

    let Ok(cam) = cam_q.single() else {
        return;
    };
    let origin = cam.translation();
    let dir = cam.forward().as_vec3();

    if let Some((hit, normal)) = dda_view_cells(&world, zoom.layer, origin, dir) {
        targeted.hit_layer_pos = Some(hit);
        targeted.normal = Some(normal);
    }
}

/// Cell-level DDA. Walks **view cells** at `view_layer` cell size,
/// not leaf voxels — so the cost is independent of zoom (16 steps
/// max, with each step doing one path walk in the tree).
///
/// Returns the hit `LayerPos` and the face normal in cell units.
fn dda_view_cells(
    world: &WorldState,
    view_layer: u8,
    origin: Vec3,
    dir: Vec3,
) -> Option<(LayerPos, IVec3)> {
    let cell_size = cell_size_at_layer(view_layer);
    let cell_origin = ROOT_ORIGIN;

    // Compute the starting cell index relative to ROOT_ORIGIN.
    // These IVec3 values are local to the camera (max ~MAX_REACH_CELLS
    // away from the camera's own cell), so they never overflow `i32`
    // even at large `MAX_LAYER` values.
    let local = origin - cell_origin;
    let mut pos = IVec3::new(
        (local.x / cell_size).floor() as i32,
        (local.y / cell_size).floor() as i32,
        (local.z / cell_size).floor() as i32,
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

    // First boundary on each axis in the direction of the ray.
    let next_x = if step.x > 0 {
        cell_origin.x + (pos.x + 1) as f32 * cell_size
    } else {
        cell_origin.x + pos.x as f32 * cell_size
    };
    let next_y = if step.y > 0 {
        cell_origin.y + (pos.y + 1) as f32 * cell_size
    } else {
        cell_origin.y + pos.y as f32 * cell_size
    };
    let next_z = if step.z > 0 {
        cell_origin.z + (pos.z + 1) as f32 * cell_size
    } else {
        cell_origin.z + pos.z as f32 * cell_size
    };

    let mut t_max = Vec3::new(
        (next_x - origin.x) * inv.x,
        (next_y - origin.y) * inv.y,
        (next_z - origin.z) * inv.z,
    );
    let t_delta = Vec3::new(
        (cell_size * inv.x).abs(),
        (cell_size * inv.y).abs(),
        (cell_size * inv.z).abs(),
    );

    let mut normal = IVec3::ZERO;
    let mut first = true;

    for _ in 0..MAX_REACH_CELLS {
        if !first {
            let lp = cell_index_to_layer_pos(pos, view_layer, cell_size, cell_origin)?;
            if is_layer_pos_solid(world, &lp) {
                return Some((lp, normal));
            }
        }
        first = false;

        if t_max.x < t_max.y && t_max.x < t_max.z {
            pos.x += step.x;
            t_max.x += t_delta.x;
            normal = IVec3::new(-step.x, 0, 0);
        } else if t_max.y < t_max.z {
            pos.y += step.y;
            t_max.y += t_delta.y;
            normal = IVec3::new(0, -step.y, 0);
        } else {
            pos.z += step.z;
            t_max.z += t_delta.z;
            normal = IVec3::new(0, 0, -step.z);
        }
    }
    None
}

/// Convert a cell IVec3 (relative to `ROOT_ORIGIN`, in cell-size
/// units) to a path-based [`LayerPos`] by routing through the
/// existing `Vec3 → LayerPos` converter. The cell's centre is the
/// canonical sample point.
fn cell_index_to_layer_pos(
    cell: IVec3,
    view_layer: u8,
    cell_size: f32,
    cell_origin: Vec3,
) -> Option<LayerPos> {
    let center = cell_origin
        + Vec3::new(
            (cell.x as f32 + 0.5) * cell_size,
            (cell.y as f32 + 0.5) * cell_size,
            (cell.z as f32 + 0.5) * cell_size,
        );
    layer_pos_from_bevy(center, view_layer)
}

fn draw_highlight(
    mut gizmos: Gizmos,
    targeted: Res<TargetedBlock>,
    zoom: Res<CameraZoom>,
) {
    let Some(lp) = targeted.hit_layer_pos.as_ref() else {
        return;
    };
    let cell_size = cell_size_at_layer(zoom.layer);
    let cell_min = bevy_origin_of_layer_pos(lp);
    let center = cell_min + Vec3::splat(cell_size * 0.5);
    gizmos.cube(
        Transform::from_translation(center).with_scale(Vec3::splat(cell_size * 1.02)),
        Color::WHITE,
    );
}
