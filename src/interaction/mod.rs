use bevy::prelude::*;

use crate::block::MODEL_SIZE;
use crate::camera::FpsCam;
use crate::layer::ActiveLayer;
use crate::world::{CellSlot, VoxelWorld};

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetedBlock>()
            .add_systems(Update, (update_target, draw_highlights));
    }
}

/// The block/cell the crosshair is pointing at.
#[derive(Resource, Default)]
pub struct TargetedBlock {
    pub hit: Option<IVec3>,
    pub normal: Option<IVec3>,
}

fn update_target(
    cam_q: Query<&GlobalTransform, With<FpsCam>>,
    active: Res<ActiveLayer>,
    world: Res<VoxelWorld>,
    mut targeted: ResMut<TargetedBlock>,
) {
    targeted.hit = None;
    targeted.normal = None;

    let Ok(cam) = cam_q.single() else { return };
    let origin = cam.translation();
    let dir = cam.forward().as_vec3();

    if active.is_top_layer() {
        // Top layer: raycast against sparse HashMap. Each cell = 1 world unit.
        *targeted = dda_top_layer(&world, origin, dir, 20.0);
    } else {
        // Inside a grid: raycast against the bounded MODEL_SIZE^3 array.
        let Some(grid) = world.get_grid(&active.nav_stack) else { return };
        *targeted = dda_grid(grid, origin, dir, 20.0);
    }
}

/// DDA raycast against the top-layer sparse HashMap.
pub fn dda_top_layer(world: &VoxelWorld, origin: Vec3, dir: Vec3, max_dist: f32) -> TargetedBlock {
    let (step, inv, mut t_max, t_delta, mut pos) = dda_setup(origin, dir);
    let mut normal = IVec3::ZERO;
    let mut dist = 0.0f32;

    // Skip the starting cell (camera may be inside it looking outward)
    let mut first = true;

    while dist < max_dist {
        if !first && world.cells.contains_key(&pos) {
            return TargetedBlock { hit: Some(pos), normal: Some(normal) };
        }
        first = false;
        dda_step(&mut pos, &step, &mut t_max, &t_delta, &mut normal, &mut dist);
    }
    TargetedBlock::default()
}

/// DDA raycast against a bounded VoxelGrid.
pub fn dda_grid(grid: &crate::world::VoxelGrid, origin: Vec3, dir: Vec3, max_dist: f32) -> TargetedBlock {
    let (step, inv, mut t_max, t_delta, mut pos) = dda_setup(origin, dir);
    let mut normal = IVec3::ZERO;
    let mut dist = 0.0f32;
    let s = MODEL_SIZE as i32;
    let mut first = true;

    while dist < max_dist {
        if !first && pos.x >= 0 && pos.x < s && pos.y >= 0 && pos.y < s && pos.z >= 0 && pos.z < s {
            if grid.slots[pos.y as usize][pos.z as usize][pos.x as usize].is_solid() {
                return TargetedBlock { hit: Some(pos), normal: Some(normal) };
            }
        }
        first = false;
        dda_step(&mut pos, &step, &mut t_max, &t_delta, &mut normal, &mut dist);
    }
    TargetedBlock::default()
}

// --- DDA helpers ---

fn dda_setup(origin: Vec3, dir: Vec3) -> (IVec3, Vec3, Vec3, Vec3, IVec3) {
    let pos = IVec3::new(origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32);
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
    let t_max = Vec3::new(
        ((if step.x > 0 { pos.x + 1 } else { pos.x }) as f32 - origin.x) * inv.x,
        ((if step.y > 0 { pos.y + 1 } else { pos.y }) as f32 - origin.y) * inv.y,
        ((if step.z > 0 { pos.z + 1 } else { pos.z }) as f32 - origin.z) * inv.z,
    );
    let t_delta = Vec3::new(
        (step.x as f32 * inv.x).abs(),
        (step.y as f32 * inv.y).abs(),
        (step.z as f32 * inv.z).abs(),
    );
    (step, inv, t_max, t_delta, pos)
}

fn dda_step(pos: &mut IVec3, step: &IVec3, t_max: &mut Vec3, t_delta: &Vec3, normal: &mut IVec3, dist: &mut f32) {
    if t_max.x < t_max.y && t_max.x < t_max.z {
        *dist = t_max.x; pos.x += step.x; t_max.x += t_delta.x;
        *normal = IVec3::new(-step.x, 0, 0);
    } else if t_max.y < t_max.z {
        *dist = t_max.y; pos.y += step.y; t_max.y += t_delta.y;
        *normal = IVec3::new(0, -step.y, 0);
    } else {
        *dist = t_max.z; pos.z += step.z; t_max.z += t_delta.z;
        *normal = IVec3::new(0, 0, -step.z);
    }
}

fn draw_highlights(mut gizmos: Gizmos, targeted: Res<TargetedBlock>, active: Res<ActiveLayer>) {
    let Some(hit) = targeted.hit else { return };

    if active.is_top_layer() {
        // Highlight whole cell (1 world unit)
        gizmos.cube(
            Transform::from_translation(hit.as_vec3() + Vec3::splat(0.5)).with_scale(Vec3::splat(1.02)),
            Color::WHITE,
        );
    } else {
        // Cell boundary
        let s = MODEL_SIZE as f32;
        gizmos.cube(
            Transform::from_translation(Vec3::splat(s / 2.0)).with_scale(Vec3::splat(s)),
            Color::srgba(0.3, 1.0, 0.3, 0.2),
        );
        // Block highlight
        gizmos.cube(
            Transform::from_translation(hit.as_vec3() + Vec3::splat(0.5)).with_scale(Vec3::splat(1.02)),
            Color::WHITE,
        );
    }
}
