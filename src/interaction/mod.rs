//! Block targeting via a view-layer voxel raycast.
//!
//! At view layer `L` the crosshair names one **view cell** —
//! precisely one voxel of the layer-`L` node's `25³` grid, with
//! Bevy-space extent `cell_size_at_layer(L)`. Clicks place or remove
//! a full layer-`L` block, so zooming out (pressing Q) lets the player
//! place bigger blocks without any editor mode switch; that is the
//! whole point of the layer hierarchy.
//!
//! The DDA walks view cells, sampling solidity via
//! [`is_layer_pos_solid`] at the view layer. The tree's downsample
//! rule (see `world::tree::downsample`) is presence-preserving — any
//! non-empty child voxel surfaces its value at the parent — so a
//! layer-`L` voxel correctly reports "solid" whenever the subtree
//! below it contains anything visible. That invariant is what makes
//! this single solidity query work at every view layer.

use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::editor::save_mode::{save_mode_eligible, SaveMode};
use crate::world::position::LayerPos;
use crate::world::view::{
    bevy_origin_of_layer_pos, cell_origin_for_anchor, cell_size_at_layer,
    is_layer_pos_solid, layer_pos_from_bevy, WorldAnchor,
};
use crate::world::{CameraZoom, WorldState};

/// Maximum reach in cells at the current view layer. The cell DDA
/// takes at most this many steps, so picking works the same at every
/// zoom level: 16 cells in front, regardless of whether one cell is
/// 1 leaf voxel (L=12) or 625 leaf voxels (L=8).
const MAX_REACH_CELLS: i32 = 16;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetedBlock>().add_systems(
            Update,
            (
                update_target.after(crate::player::sync_anchor_to_player),
                draw_highlight.after(crate::player::sync_anchor_to_player),
            ),
        );
    }
}

/// The view cell the crosshair is pointing at.
///
/// `hit_layer_pos.layer == zoom.layer` — one voxel in the layer-`L`
/// node's `25³` grid. Editing consumers drive `cell_size` and
/// re-projection off `zoom.layer` (equivalently `hit_layer_pos.layer`).
#[derive(Resource, Default)]
pub struct TargetedBlock {
    pub hit_layer_pos: Option<LayerPos>,
    /// Face normal at the hit, in view-cell units. Add
    /// `normal * cell_size_at_layer(hit_layer_pos.layer)` to the
    /// cell's centre to get the placement cell's centre.
    pub normal: Option<IVec3>,
}

impl TargetedBlock {
    fn clear(&mut self) {
        self.hit_layer_pos = None;
        self.normal = None;
    }
}

pub fn update_target(
    cam_q: Query<&GlobalTransform, With<FpsCam>>,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    mut targeted: ResMut<TargetedBlock>,
) {
    targeted.clear();

    let Ok(cam) = cam_q.single() else {
        return;
    };
    let origin = cam.translation();
    let dir = cam.forward().as_vec3();

    if let Some((hit, normal)) =
        dda_view_cells(&world, zoom.layer, origin, dir, &anchor)
    {
        targeted.hit_layer_pos = Some(hit);
        targeted.normal = Some(normal);
    }
}

/// View-cell DDA. Walks one view cell per iteration, sampling
/// solidity at the view layer. Returns the hit `LayerPos` at
/// `view_layer` and the face normal in view-cell units.
///
/// The cell grid is anchored to the current [`WorldAnchor`]: block
/// index 0 is the view cell that contains the anchor's leaf coord,
/// and `cell_origin` is the small `(-cell_size, 0]` offset from the
/// anchor to that cell's min corner.
fn dda_view_cells(
    world: &WorldState,
    view_layer: u8,
    origin: Vec3,
    dir: Vec3,
    anchor: &WorldAnchor,
) -> Option<(LayerPos, IVec3)> {
    let cell_size = cell_size_at_layer(view_layer);
    let cell_size_i64 = cell_size as i64;
    let cell_origin = cell_origin_for_anchor(anchor, cell_size_i64);

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
            let lp = cell_index_to_layer_pos(
                pos,
                view_layer,
                cell_size,
                cell_origin,
                anchor,
            )?;
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

/// Convert an anchor-local cell `IVec3` (in `cell_size` strides from
/// `cell_origin`) to a path-based [`LayerPos`] by going through
/// [`layer_pos_from_bevy`], which handles the `i64` leaf-coord
/// math against the anchor internally.
fn cell_index_to_layer_pos(
    cell: IVec3,
    layer: u8,
    cell_size: f32,
    cell_origin: Vec3,
    anchor: &WorldAnchor,
) -> Option<LayerPos> {
    let center = cell_origin
        + Vec3::new(
            (cell.x as f32 + 0.5) * cell_size,
            (cell.y as f32 + 0.5) * cell_size,
            (cell.z as f32 + 0.5) * cell_size,
        );
    layer_pos_from_bevy(center, layer, anchor)
}

fn draw_highlight(
    mut gizmos: Gizmos,
    targeted: Res<TargetedBlock>,
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    save_mode: Res<SaveMode>,
) {
    let Some(lp) = targeted.hit_layer_pos.as_ref() else {
        return;
    };
    // In save mode the hovered block is recoloured blue in place
    // (see `editor::save_mode::update_save_tint`); paint the outline
    // blue to match so the "save target" is visually cohesive.
    let save_tinted = save_mode.active && save_mode_eligible(zoom.layer);
    let color = if save_tinted {
        Color::srgb(0.3, 0.65, 1.0)
    } else {
        Color::WHITE
    };
    let cell_size = cell_size_at_layer(zoom.layer);
    let cell_min = bevy_origin_of_layer_pos(lp, &anchor);
    let center = cell_min + Vec3::splat(cell_size * 0.5);
    gizmos.cube(
        Transform::from_translation(center).with_scale(Vec3::splat(cell_size * 1.02)),
        color,
    );
}
