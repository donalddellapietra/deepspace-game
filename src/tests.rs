#![cfg(test)]

use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};
use crate::model::mesher::bake_model;
use crate::world::collision::{block_solid, move_and_collide, on_ground};
use crate::world::*;

fn ground_chunk() -> Chunk {
    let mut c = Chunk::new_empty();
    for z in 0..MODEL_SIZE { for x in 0..MODEL_SIZE {
        c.blocks[0][z][x] = Some(BlockType::Stone);
        c.blocks[1][z][x] = Some(BlockType::Stone);
        c.blocks[2][z][x] = Some(BlockType::Dirt);
        c.blocks[3][z][x] = Some(BlockType::Dirt);
        c.blocks[4][z][x] = Some(BlockType::Grass);
    }}
    c
}

fn ground_world(extent: i32) -> FlatWorld {
    let mut w = FlatWorld::default();
    for z in -extent..extent { for x in -extent..extent {
        w.chunks.insert(IVec3::new(x, 0, z), ground_chunk());
    }}
    w
}

// ============================================================
// block_solid
// ============================================================

#[test]
fn solid_block_exists() {
    let w = ground_world(2);
    assert!(w.is_solid(IVec3::new(2, 4, 2))); // grass
    assert!(w.is_solid(IVec3::new(0, 0, 0))); // stone
}

#[test]
fn solid_empty_above() {
    let w = ground_world(2);
    assert!(!w.is_solid(IVec3::new(2, 5, 2))); // above grass
    assert!(!w.is_solid(IVec3::new(2, -1, 2))); // below stone
}

#[test]
fn solid_cross_chunk() {
    let w = ground_world(4);
    // Block at x=7 is in chunk (1,0,0), local (2,4,2) = Grass
    assert!(w.is_solid(IVec3::new(7, 4, 2)));
    assert!(!w.is_solid(IVec3::new(7, 5, 2)));
}

#[test]
fn solid_negative_coords() {
    let w = ground_world(4);
    assert!(w.is_solid(IVec3::new(-3, 4, 2)));
    assert!(!w.is_solid(IVec3::new(-3, 5, 2)));
}

#[test]
fn solid_void() {
    let w = ground_world(2);
    assert!(!w.is_solid(IVec3::new(100, 0, 100)));
}

// ============================================================
// Editing
// ============================================================

#[test]
fn set_and_get() {
    let mut w = FlatWorld::default();
    assert!(!w.is_solid(IVec3::new(5, 5, 5)));
    w.set(IVec3::new(5, 5, 5), Some(BlockType::Brick));
    assert!(w.is_solid(IVec3::new(5, 5, 5)));
    assert_eq!(w.get(IVec3::new(5, 5, 5)), Some(BlockType::Brick));
    w.set(IVec3::new(5, 5, 5), None);
    assert!(!w.is_solid(IVec3::new(5, 5, 5)));
}

#[test]
fn edit_cross_chunk() {
    let mut w = ground_world(4);
    assert!(w.is_solid(IVec3::new(7, 4, 2)));
    w.set(IVec3::new(7, 4, 2), None);
    assert!(!w.is_solid(IVec3::new(7, 4, 2)));
}

// ============================================================
// Collision
// ============================================================

#[test]
fn gravity_lands() {
    let w = ground_world(4);
    let mut pos = Vec3::new(2.5, 10.0, 2.5);
    let mut vel = Vec3::ZERO;
    for _ in 0..180 {
        vel.y -= 20.0 / 60.0;
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, 1.0 / 60.0, &w);
    }
    assert!((pos.y - 5.0).abs() < 0.1, "Should land on grass top at y=5. Got {}", pos.y);
}

#[test]
fn walk_across_chunk_boundary() {
    let w = ground_world(4);
    let mut pos = Vec3::new(3.0, 5.0, 2.5);
    let mut vel = Vec3::ZERO;
    // Walk +X for 2 seconds — should cross the chunk boundary at x=5
    for _ in 0..120 {
        vel.y -= 20.0 / 60.0;
        move_and_collide(&mut pos, &mut vel, Vec2::new(8.0 / 60.0, 0.0), 1.0 / 60.0, &w);
    }
    assert!(pos.x > 5.0, "Should have crossed chunk boundary. Got x={}", pos.x);
    assert!((pos.y - 5.0).abs() < 0.1, "Should still be on ground. Got y={}", pos.y);
}

#[test]
fn on_ground_check() {
    let w = ground_world(2);
    assert!(on_ground(Vec3::new(2.5, 5.0, 2.5), &w));
    assert!(!on_ground(Vec3::new(2.5, 6.0, 2.5), &w));
}

// ============================================================
// Drill (zoom only — never modifies world data)
// ============================================================

#[test]
fn drill_in_increments_depth_and_scales_pos() {
    let mut state = WorldState::default();
    state.world = ground_world(2);
    let chunk_count_before = state.world.chunks.len();

    let pos = Vec3::new(0.5, 2.0, 0.5);
    let new_pos = state.drill_in(pos).expect("drill_in should succeed at depth 0");

    assert_eq!(state.depth(), 1);
    // Player bevy pos scales by MODEL_SIZE on drill in.
    assert_eq!(new_pos, pos * crate::block::MODEL_SIZE as f32);
    // World data is untouched.
    assert_eq!(state.world.chunks.len(), chunk_count_before);
    assert!(state.world.is_solid(IVec3::new(2, 4, 2)));
}

#[test]
fn drill_out_decrements_depth_and_unscales_pos() {
    let mut state = WorldState::default();
    state.world = ground_world(2);

    let start = Vec3::new(0.5, 2.0, 0.5);
    let drilled = state.drill_in(start).unwrap();
    assert_eq!(state.depth(), 1);

    let returned = state.drill_out(drilled).expect("drill_out should succeed");
    assert_eq!(state.depth(), 0);
    assert_eq!(returned, start);

    // World data still intact.
    assert!(state.world.is_solid(IVec3::new(2, 4, 2)));
}

#[test]
fn drill_in_blocked_at_max_depth() {
    let mut state = WorldState::default();
    state.world = ground_world(2);

    state.drill_in(Vec3::ZERO).unwrap();
    state.drill_in(Vec3::ZERO).unwrap();
    assert_eq!(state.depth(), crate::world::MAX_DEPTH);
    // One past MAX_DEPTH must be refused.
    assert!(state.drill_in(Vec3::ZERO).is_none());
    assert_eq!(state.depth(), crate::world::MAX_DEPTH);
}

// ============================================================
// Model save
// ============================================================

#[test]
fn save_model() {
    use crate::model::{ModelRegistry, VoxelModel};
    let mut reg = ModelRegistry::default();
    let mut meshes = Assets::<Mesh>::default();
    let mut blocks = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    blocks[0][0][0] = Some(BlockType::Brick);
    let id = reg.register(VoxelModel { name: "Test".into(), blocks, baked: bake_model(&blocks, &mut meshes) });
    assert_eq!(reg.models[id.0].blocks[0][0][0], Some(BlockType::Brick));
}
