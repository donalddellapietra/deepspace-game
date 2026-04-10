#![cfg(test)]

use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};
use crate::layer::NavEntry;
use crate::world::collision::{block_solid, move_and_collide, on_ground, PLAYER_H};
use crate::world::*;

// ============================================================
// Helpers
// ============================================================

fn make_grid(fill: impl Fn(usize, usize, usize) -> Option<BlockType>) -> VoxelGrid {
    let mut grid = VoxelGrid::new_empty();
    for y in 0..MODEL_SIZE {
        for z in 0..MODEL_SIZE {
            for x in 0..MODEL_SIZE {
                if let Some(bt) = fill(x, y, z) {
                    grid.slots[y][z][x] = CellSlot::Block(bt);
                }
            }
        }
    }
    grid
}

fn ground_cell() -> VoxelGrid {
    make_grid(|_, y, _| match y {
        0 | 1 => Some(BlockType::Stone),
        2 | 3 => Some(BlockType::Dirt),
        4 => Some(BlockType::Grass),
        _ => None,
    })
}

fn ground_world(extent: i32) -> VoxelWorld {
    let mut world = VoxelWorld::default();
    for z in -extent..extent {
        for x in -extent..extent {
            world.cells.insert(IVec3::new(x, 0, z), ground_cell());
        }
    }
    world
}

fn nav1(coord: IVec3) -> Vec<NavEntry> {
    vec![NavEntry { cell_coord: coord, return_position: Vec3::ZERO }]
}

fn nav2(top: IVec3, inner: IVec3) -> Vec<NavEntry> {
    vec![
        NavEntry { cell_coord: top, return_position: Vec3::ZERO },
        NavEntry { cell_coord: inner, return_position: Vec3::ZERO },
    ]
}

// ============================================================
// block_solid — the foundation of all collision
// ============================================================

#[test]
fn solid_top_layer_cell_exists() {
    let world = ground_world(2);
    assert!(block_solid(&world, &[], IVec3::new(0, 0, 0)));
}

#[test]
fn solid_top_layer_no_cell() {
    let world = ground_world(2);
    assert!(!block_solid(&world, &[], IVec3::new(0, 1, 0))); // no cell at y=1
    assert!(!block_solid(&world, &[], IVec3::new(100, 0, 100))); // out of range
}

#[test]
fn solid_depth1_block() {
    let world = ground_world(2);
    let nav = nav1(IVec3::ZERO);
    assert!(block_solid(&world, &nav, IVec3::new(2, 4, 2))); // grass
    assert!(block_solid(&world, &nav, IVec3::new(0, 0, 0))); // stone
    assert!(!block_solid(&world, &nav, IVec3::new(2, 5, 2))); // above grid — empty
}

#[test]
fn solid_depth1_neighbor() {
    let world = ground_world(4);
    let nav = nav1(IVec3::ZERO);
    // Block at x=6 is inside neighbor cell (1,0,0), local x=1, y=4 = grass
    assert!(block_solid(&world, &nav, IVec3::new(6, 4, 2)));
    // Above neighbor's grid
    assert!(!block_solid(&world, &nav, IVec3::new(6, 5, 2)));
}

#[test]
fn solid_depth2_parent_block_sibling() {
    let mut world = ground_world(2);
    // Create child at (2,4,2) inside cell (0,0,0)
    let grid = world.cells.get_mut(&IVec3::ZERO).unwrap();
    grid.slots[4][2][2] = CellSlot::Child(Box::new(VoxelGrid::new_empty()));

    let nav = nav2(IVec3::ZERO, IVec3::new(2, 4, 2));
    // The parent slot at (2,3,2) = Dirt. It spans a 5x5x5 region below.
    // Block at (2, -1, 2) in inner space = inside the parent Dirt block (pdy=-1)
    assert!(block_solid(&world, &nav, IVec3::new(2, -1, 2)));
    // Block at (2, -6, 2) = inside parent Stone block at (2,2,2) (pdy=-2)
    // Hmm, pdy = (-6).div_euclid(5) = -2. local_y = (-6).rem_euclid(5) = 4.
    // sib_coord = (2,4,2) + (0,-2,0) = (2,2,2). Parent slot[2][2][2] = Dirt. Solid.
    assert!(block_solid(&world, &nav, IVec3::new(2, -6, 2)));
}

// ============================================================
// move_and_collide — swept AABB tests
// ============================================================

#[test]
fn gravity_lands_on_top_layer() {
    let world = ground_world(4);
    let mut pos = Vec3::new(0.5, 5.0, 0.5);
    let mut vel = Vec3::ZERO;

    // Simulate 120 frames of gravity (2 seconds)
    for _ in 0..120 {
        vel.y -= 20.0 * (1.0 / 60.0);
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, 1.0 / 60.0, &world, &[]);
    }

    assert!((pos.y - 1.0).abs() < 0.02, "Player should land at y=1.0 (top of cell). Got {}", pos.y);
    assert!(vel.y.abs() < 0.01, "Velocity should be zero after landing");
}

#[test]
fn gravity_lands_on_grass_depth1() {
    let world = ground_world(4);
    let nav = nav1(IVec3::ZERO);
    let mut pos = Vec3::new(2.5, 8.0, 2.5);
    let mut vel = Vec3::ZERO;

    for _ in 0..120 {
        vel.y -= 20.0 * (1.0 / 60.0);
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, 1.0 / 60.0, &world, &nav);
    }

    assert!((pos.y - 5.0).abs() < 0.02, "Should land on grass top at y=5. Got {}", pos.y);
}

#[test]
fn no_step_up() {
    let mut world = ground_world(2);
    // Grid with ground at y=0, and a wall block at y=1 in column (3,2)
    let custom = make_grid(|x, y, z| {
        if y == 0 { Some(BlockType::Stone) }
        else if y == 1 && x == 3 && z == 2 { Some(BlockType::Brick) }
        else { None }
    });
    world.cells.insert(IVec3::new(5, 0, 5), custom);
    let nav = nav1(IVec3::new(5, 0, 5));

    // Player standing on stone floor at y=1.0, trying to walk into the brick at x=3
    let mut pos = Vec3::new(2.5, 1.0, 2.5);
    let mut vel = Vec3::ZERO;

    // Walk in +X direction for 30 frames
    for _ in 0..30 {
        vel.y -= 20.0 * (1.0 / 60.0);
        let h_delta = Vec2::new(8.0 * (1.0 / 60.0), 0.0);
        move_and_collide(&mut pos, &mut vel, h_delta, 1.0 / 60.0, &world, &nav);
    }

    // Player should be blocked by the brick, NOT teleported on top
    assert!(pos.y < 1.5, "Player should NOT step up onto brick. Got y={}", pos.y);
    assert!(pos.x < 3.0, "Player should be blocked before x=3. Got x={}", pos.x);
}

#[test]
fn jump_onto_block() {
    let mut world = ground_world(2);
    // Full platform of bricks at y=1 covering x=3..5, all z
    let custom = make_grid(|x, y, _z| {
        if y == 0 { Some(BlockType::Stone) }
        else if y == 1 && x >= 3 { Some(BlockType::Brick) }
        else { None }
    });
    world.cells.insert(IVec3::new(5, 0, 5), custom);
    let nav = nav1(IVec3::new(5, 0, 5));

    // Player at x=2.5 on stone floor (y=1.0). Jump straight up, then drift right onto the brick platform.
    let mut pos = Vec3::new(2.5, 1.0, 2.5);
    let mut vel = Vec3::new(0.0, 10.0, 0.0); // jump

    // Phase 1: go up (no horizontal movement)
    for _ in 0..12 {
        vel.y -= 20.0 * (1.0 / 60.0);
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, 1.0 / 60.0, &world, &nav);
    }
    // Should be above the brick platform (y > 2.0)
    assert!(pos.y > 2.0, "Should rise above bricks. Got y={}", pos.y);

    // Phase 2: drift right and fall onto the platform
    for _ in 0..60 {
        vel.y -= 20.0 * (1.0 / 60.0);
        let h_delta = Vec2::new(6.0 * (1.0 / 60.0), 0.0);
        move_and_collide(&mut pos, &mut vel, h_delta, 1.0 / 60.0, &world, &nav);
    }

    // Player should be on top of the brick platform at y=2.0
    assert!((pos.y - 2.0).abs() < 0.1, "Should land on brick platform at y=2. Got y={}", pos.y);
    assert!(pos.x >= 3.0, "Should have moved onto the platform. Got x={}", pos.x);
}

#[test]
fn ceiling_collision() {
    let custom = make_grid(|_x, y, _z| {
        if y == 0 || y == 4 { Some(BlockType::Stone) } else { None }
    });
    let mut world = VoxelWorld::default();
    world.cells.insert(IVec3::new(0, 0, 0), custom);
    let nav = nav1(IVec3::ZERO);

    // Player on floor (y=1), jumps. Should hit ceiling at y=4 (block bottom).
    let mut pos = Vec3::new(2.5, 1.0, 2.5);
    let mut vel = Vec3::new(0.0, 15.0, 0.0); // strong jump

    for _ in 0..30 {
        vel.y -= 20.0 * (1.0 / 60.0);
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, 1.0 / 60.0, &world, &nav);
    }

    // Player head should never go above block bottom at y=4
    assert!(pos.y + PLAYER_H <= 4.01, "Head should not pass ceiling. Got head_y={}", pos.y + PLAYER_H);
}

#[test]
fn depth2_no_void_fall() {
    let mut world = ground_world(2);
    let grid = world.cells.get_mut(&IVec3::ZERO).unwrap();
    let inner = VoxelGrid::new_empty(); // totally empty child
    grid.slots[4][2][2] = CellSlot::Child(Box::new(inner));

    let nav = nav2(IVec3::ZERO, IVec3::new(2, 4, 2));

    // Player at y=2, should land on the parent block's top surface
    let mut pos = Vec3::new(2.5, 2.0, 2.5);
    let mut vel = Vec3::ZERO;

    for _ in 0..300 {
        vel.y -= 20.0 * (1.0 / 60.0);
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, 1.0 / 60.0, &world, &nav);
    }

    assert!(pos.y >= -0.1, "Should not fall to void. Parent block catches. Got y={}", pos.y);
}

#[test]
fn on_ground_check() {
    let world = ground_world(2);
    assert!(on_ground(Vec3::new(0.5, 1.0, 0.5), &world, &[]));
    assert!(!on_ground(Vec3::new(0.5, 2.0, 0.5), &world, &[]));
}

// ============================================================
// get_grid navigation
// ============================================================

#[test]
fn get_grid_depth1() {
    let world = ground_world(2);
    let nav = nav1(IVec3::ZERO);
    let grid = world.get_grid(&nav);
    assert!(grid.is_some());
    assert_eq!(grid.unwrap().solid_count(), 125);
}

#[test]
fn get_grid_depth2() {
    let mut world = ground_world(2);
    let grid = world.cells.get_mut(&IVec3::ZERO).unwrap();
    grid.slots[4][2][2] = CellSlot::Child(Box::new(VoxelGrid::new_empty()));
    let nav = nav2(IVec3::ZERO, IVec3::new(2, 4, 2));
    assert!(world.get_grid(&nav).is_some());
}

#[test]
fn get_grid_invalid() {
    let world = ground_world(2);
    // Nav into a Stone slot (not Child)
    let nav = nav2(IVec3::ZERO, IVec3::new(0, 0, 0));
    assert!(world.get_grid(&nav).is_none());
}

// ============================================================
// render_ancestor_transforms
// ============================================================

#[test]
fn ancestors_depth1() {
    let nav = nav1(IVec3::ZERO);
    let t = render_ancestor_transforms(&nav);
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].2, MODEL_SIZE as f32); // scale = 5
    assert_eq!(t[0].1, Vec3::ZERO); // offset = 0 for origin cell
}

#[test]
fn ancestors_depth2() {
    let nav = nav2(IVec3::ZERO, IVec3::new(2, 4, 2));
    let t = render_ancestor_transforms(&nav);
    assert_eq!(t.len(), 2);
    assert_eq!(t[0].2, 5.0);  // parent scale
    assert_eq!(t[1].2, 25.0); // grandparent scale
}
