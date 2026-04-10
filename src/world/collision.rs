//! AABB collision against anything that can answer "is this unit cube solid?".
//! At the top layer the cube is a whole chunk; deeper, it's an individual block.

use bevy::prelude::*;

use super::FlatWorld;

pub const PLAYER_HW: f32 = 0.3;
pub const PLAYER_H: f32 = 1.7;

/// Anything that can answer "is the unit cube at this integer coord solid?"
/// FlatWorld answers per-block; WorldState dispatches per-chunk at the top layer.
pub trait SolidQuery {
    fn is_solid(&self, coord: IVec3) -> bool;
}

impl SolidQuery for FlatWorld {
    fn is_solid(&self, coord: IVec3) -> bool {
        FlatWorld::is_solid(self, coord)
    }
}

pub fn block_solid<W: SolidQuery>(world: &W, coord: IVec3) -> bool {
    world.is_solid(coord)
}

#[derive(Clone, Copy)]
struct Aabb { min: Vec3, max: Vec3 }

impl Aabb {
    fn from_feet(pos: Vec3) -> Self {
        Self {
            min: Vec3::new(pos.x - PLAYER_HW, pos.y, pos.z - PLAYER_HW),
            max: Vec3::new(pos.x + PLAYER_HW, pos.y + PLAYER_H, pos.z + PLAYER_HW),
        }
    }
    fn expanded(&self, axis: usize, delta: f32) -> Self {
        let mut r = *self;
        if delta > 0.0 { r.max[axis] += delta; } else { r.min[axis] += delta; }
        r
    }
    fn block_range(&self) -> (IVec3, IVec3) {
        (
            IVec3::new(self.min.x.floor() as i32, self.min.y.floor() as i32, self.min.z.floor() as i32),
            IVec3::new(
                (self.max.x - 1e-5).floor() as i32,
                (self.max.y - 1e-5).floor() as i32,
                (self.max.z - 1e-5).floor() as i32,
            ),
        )
    }
}

fn clip_axis(player: &Aabb, delta: f32, axis: usize, bx: i32, by: i32, bz: i32) -> f32 {
    let (a1, a2) = match axis { 0 => (1, 2), 1 => (0, 2), _ => (0, 1) };
    let b_min = [bx as f32, by as f32, bz as f32];
    let b_max = [(bx + 1) as f32, (by + 1) as f32, (bz + 1) as f32];
    let shrink = 0.02;
    if player.max[a1] - shrink <= b_min[a1] || player.min[a1] + shrink >= b_max[a1] { return delta; }
    if player.max[a2] - shrink <= b_min[a2] || player.min[a2] + shrink >= b_max[a2] { return delta; }
    if delta < 0.0 {
        let gap = b_max[axis] - player.min[axis];
        if gap <= 0.0 && gap > delta { return gap; }
    } else if delta > 0.0 {
        let gap = b_min[axis] - player.max[axis];
        if gap >= 0.0 && gap < delta { return gap; }
    }
    delta
}

pub fn move_and_collide<W: SolidQuery>(
    pos: &mut Vec3, vel: &mut Vec3, horizontal_delta: Vec2, dt: f32, world: &W,
) {
    let solid = |coord: IVec3| world.is_solid(coord);
    let mut dy = vel.y * dt;
    let dx = horizontal_delta.x;
    let dz = horizontal_delta.y;

    let player = Aabb::from_feet(*pos);
    let expanded = Aabb {
        min: Vec3::new(player.min.x + dx.min(0.0) - 1.0, player.min.y + dy.min(0.0) - 1.0, player.min.z + dz.min(0.0) - 1.0),
        max: Vec3::new(player.max.x + dx.max(0.0) + 1.0, player.max.y + dy.max(0.0) + 1.0, player.max.z + dz.max(0.0) + 1.0),
    };
    let (bmin, bmax) = expanded.block_range();
    let mut blocks = Vec::new();
    for by in bmin.y..=bmax.y { for bz in bmin.z..=bmax.z { for bx in bmin.x..=bmax.x {
        if solid(IVec3::new(bx, by, bz)) { blocks.push((bx, by, bz)); }
    }}}

    let mut pa = Aabb::from_feet(*pos);
    for &(bx, by, bz) in &blocks { dy = clip_axis(&pa, dy, 1, bx, by, bz); }
    pos.y += dy;
    if (dy - vel.y * dt).abs() > 1e-6 { vel.y = 0.0; }

    pa = Aabb::from_feet(*pos);
    let mut cdx = dx;
    for &(bx, by, bz) in &blocks { cdx = clip_axis(&pa, cdx, 0, bx, by, bz); }
    pos.x += cdx;

    pa = Aabb::from_feet(*pos);
    let mut cdz = dz;
    for &(bx, by, bz) in &blocks { cdz = clip_axis(&pa, cdz, 2, bx, by, bz); }
    pos.z += cdz;
}

pub fn on_ground<W: SolidQuery>(pos: Vec3, world: &W) -> bool {
    let player = Aabb::from_feet(pos);
    let mut test_dy = -0.05f32;
    let (bmin, bmax) = player.expanded(1, -0.1).block_range();
    for by in bmin.y..=bmax.y { for bz in bmin.z..=bmax.z { for bx in bmin.x..=bmax.x {
        if world.is_solid(IVec3::new(bx, by, bz)) {
            test_dy = clip_axis(&player, test_dy, 1, bx, by, bz);
        }
    }}}
    test_dy.abs() < 0.04
}
