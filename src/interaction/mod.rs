use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::camera::FpsCam;
use crate::layer::{EditingContext, GameLayer};
use crate::world::Layer1World;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetedBlock>()
            .init_resource::<TargetedCell>()
            .add_systems(Update, update_targeted_block.run_if(in_state(GameLayer::Editing)))
            .add_systems(Update, draw_block_highlight.run_if(in_state(GameLayer::Editing)))
            .add_systems(Update, draw_cell_highlight.run_if(in_state(GameLayer::World)));
    }
}

/// Which block the crosshair is pointing at (layer 0 editing).
#[derive(Resource, Default)]
pub struct TargetedBlock {
    pub hit: Option<IVec3>,
    pub normal: Option<IVec3>,
}

/// Which cell the crosshair is pointing at (layer 1 world).
#[derive(Resource, Default)]
pub struct TargetedCell {
    pub coord: Option<IVec3>,
}

/// DDA voxel raycast through the 5x5x5 editing grid.
fn update_targeted_block(
    camera_q: Query<&GlobalTransform, With<FpsCam>>,
    context: Option<Res<EditingContext>>,
    world: Res<Layer1World>,
    mut targeted: ResMut<TargetedBlock>,
) {
    targeted.hit = None;
    targeted.normal = None;

    let Some(ctx) = context else { return };
    let Ok(cam_gtf) = camera_q.single() else { return };
    let Some(cell_data) = world.cells.get(&ctx.cell_coord) else { return };

    let origin = cam_gtf.translation();
    let dir = cam_gtf.forward().as_vec3();

    let cell_origin = ctx.cell_coord.as_vec3() * MODEL_SIZE as f32;
    let local_origin = origin - cell_origin;

    if let Some((hit, normal)) = dda_raycast(local_origin, dir, &cell_data.blocks, 20.0) {
        targeted.hit = Some(hit);
        targeted.normal = Some(normal);
    }
}

/// Simple DDA raycast through a voxel grid. Returns (hit_pos, face_normal).
fn dda_raycast(
    origin: Vec3,
    dir: Vec3,
    blocks: &[[[Option<crate::block::BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
    max_dist: f32,
) -> Option<(IVec3, IVec3)> {
    let s = MODEL_SIZE as i32;
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

    let inv_dir = Vec3::new(
        if dir.x.abs() > 1e-10 { 1.0 / dir.x } else { f32::MAX },
        if dir.y.abs() > 1e-10 { 1.0 / dir.y } else { f32::MAX },
        if dir.z.abs() > 1e-10 { 1.0 / dir.z } else { f32::MAX },
    );

    let mut t_max = Vec3::new(
        ((if step.x > 0 { pos.x + 1 } else { pos.x }) as f32 - origin.x) * inv_dir.x,
        ((if step.y > 0 { pos.y + 1 } else { pos.y }) as f32 - origin.y) * inv_dir.y,
        ((if step.z > 0 { pos.z + 1 } else { pos.z }) as f32 - origin.z) * inv_dir.z,
    );

    let t_delta = Vec3::new(
        (step.x as f32 * inv_dir.x).abs(),
        (step.y as f32 * inv_dir.y).abs(),
        (step.z as f32 * inv_dir.z).abs(),
    );

    let mut normal = IVec3::ZERO;
    let mut dist = 0.0f32;

    while dist < max_dist {
        if pos.x >= 0 && pos.x < s && pos.y >= 0 && pos.y < s && pos.z >= 0 && pos.z < s {
            if blocks[pos.y as usize][pos.z as usize][pos.x as usize].is_some() {
                return Some((pos, normal));
            }
        }

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

    None
}

fn draw_block_highlight(
    mut gizmos: Gizmos,
    targeted: Res<TargetedBlock>,
    context: Option<Res<EditingContext>>,
) {
    let Some(ctx) = context else { return };

    let cell_origin = Vec3::new(
        ctx.cell_coord.x as f32 * MODEL_SIZE as f32,
        ctx.cell_coord.y as f32 * MODEL_SIZE as f32,
        ctx.cell_coord.z as f32 * MODEL_SIZE as f32,
    );

    // Always draw the cell boundary box
    let cell_center = cell_origin + Vec3::splat(MODEL_SIZE as f32 / 2.0);
    gizmos.cube(
        Transform::from_translation(cell_center).with_scale(Vec3::splat(MODEL_SIZE as f32)),
        Color::srgba(0.3, 1.0, 0.3, 0.4),
    );

    // Highlight targeted block
    let Some(hit) = targeted.hit else { return };
    let center = cell_origin + hit.as_vec3() + Vec3::splat(0.5);

    gizmos.cube(
        Transform::from_translation(center).with_scale(Vec3::splat(1.02)),
        Color::WHITE,
    );
}

fn draw_cell_highlight(
    mut gizmos: Gizmos,
    targeted: Res<TargetedCell>,
) {
    let Some(coord) = targeted.coord else { return };
    let center = coord.as_vec3() * MODEL_SIZE as f32 + Vec3::splat(MODEL_SIZE as f32 / 2.0);

    gizmos.cube(
        Transform::from_translation(center).with_scale(Vec3::splat(MODEL_SIZE as f32 * 1.01)),
        Color::srgba(1.0, 1.0, 1.0, 0.3),
    );
}
