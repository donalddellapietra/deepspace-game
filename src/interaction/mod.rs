use bevy::prelude::*;

use crate::camera::FpsCam;
use crate::world::collision::SolidQuery;
use crate::world::WorldState;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TargetedBlock>()
            .add_systems(Update, (update_target, draw_highlight));
    }
}

#[derive(Resource, Default)]
pub struct TargetedBlock {
    pub hit: Option<IVec3>,
    pub normal: Option<IVec3>,
}

fn update_target(
    cam_q: Query<&GlobalTransform, With<FpsCam>>,
    state: Res<WorldState>,
    mut targeted: ResMut<TargetedBlock>,
) {
    targeted.hit = None;
    targeted.normal = None;

    let Ok(cam) = cam_q.single() else { return };
    let origin = cam.translation();
    let dir = cam.forward().as_vec3();

    // One raycast. world.is_solid handles everything.
    let mut pos = IVec3::new(origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32);
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
    let mut dist = 0.0f32;
    let mut first = true;

    while dist < 20.0 {
        if !first && state.is_solid(pos) {
            targeted.hit = Some(pos);
            targeted.normal = Some(normal);
            return;
        }
        first = false;

        if t_max.x < t_max.y && t_max.x < t_max.z {
            dist = t_max.x; pos.x += step.x; t_max.x += t_delta.x;
            normal = IVec3::new(-step.x, 0, 0);
        } else if t_max.y < t_max.z {
            dist = t_max.y; pos.y += step.y; t_max.y += t_delta.y;
            normal = IVec3::new(0, -step.y, 0);
        } else {
            dist = t_max.z; pos.z += step.z; t_max.z += t_delta.z;
            normal = IVec3::new(0, 0, -step.z);
        }
    }
}

fn draw_highlight(mut gizmos: Gizmos, targeted: Res<TargetedBlock>) {
    let Some(hit) = targeted.hit else { return };
    gizmos.cube(
        Transform::from_translation(hit.as_vec3() + Vec3::splat(0.5)).with_scale(Vec3::splat(1.02)),
        Color::WHITE,
    );
}
