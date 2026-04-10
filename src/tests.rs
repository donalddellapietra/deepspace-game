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

    // Phase 2: drift slightly right and fall onto the platform
    for _ in 0..60 {
        vel.y -= 20.0 * (1.0 / 60.0);
        let h_delta = Vec2::new(2.0 * (1.0 / 60.0), 0.0); // slow drift
        move_and_collide(&mut pos, &mut vel, h_delta, 1.0 / 60.0, &world, &nav);
    }

    // Player should have landed on the brick platform (x=3..5) at y=2.0
    assert!(pos.x >= 3.0 && pos.x <= 5.0, "Should be on platform. Got x={}", pos.x);
    assert!((pos.y - 2.0).abs() < 0.1, "Should land on brick at y=2. Got y={}", pos.y);
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
// Exact user scenario: depth 2 fall through ground
// ============================================================

#[test]
fn depth2_fall_from_height() {
    // Setup: ground at y=0, extra cell at y=1
    let mut world = ground_world(4);
    // Place a cell at y=1 (the block the user placed)
    world.cells.insert(IVec3::new(0, 1, 0), ground_cell());

    // Drill into the y=1 cell
    let nav1_stack = nav1(IVec3::new(0, 1, 0));

    // Convert a block inside to a Child (simulating drill_down again)
    {
        let grid = world.get_grid_mut(&nav1_stack).unwrap();
        let mut inner = VoxelGrid::new_empty();
        for y in 0..MODEL_SIZE {
            for z in 0..MODEL_SIZE {
                for x in 0..MODEL_SIZE {
                    inner.slots[y][z][x] = CellSlot::Block(BlockType::Grass);
                }
            }
        }
        grid.slots[4][2][2] = CellSlot::Child(Box::new(inner));
    }

    // Now at depth 2 inside (0,1,0) → (2,4,2)
    let nav = nav2(IVec3::new(0, 1, 0), IVec3::new(2, 4, 2));

    // Verify block_solid works for blocks below the child grid
    assert!(block_solid(&world, &nav, IVec3::new(2, -1, 2)),
        "Parent block below should be solid");
    assert!(block_solid(&world, &nav, IVec3::new(2, -5, 2)),
        "Parent block further below should be solid");

    // Player walks off the edge of the child grid to x=6 (outside 0..5).
    // The neighboring parent blocks are solid, so the player should stand on them.
    let mut pos = Vec3::new(6.5, 5.0, 2.5);
    let mut vel = Vec3::ZERO;

    // Verify block_solid works at various depths
    assert!(block_solid(&world, &nav, IVec3::new(6, 4, 2)), "neighbor should be solid");
    assert!(!block_solid(&world, &nav, IVec3::new(6, 5, 2)), "above neighbor should be empty");

    let dt = 1.0 / 60.0;
    for _ in 0..300 {
        vel.y -= 20.0 * dt;
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, dt, &world, &nav);
    }

    assert!(pos.y >= 4.9, "Player should stand on parent block top at y=5. Got y={}", pos.y);
}

/// The EXACT user scenario: place block at y=1 at depth 0, drill twice,
/// walk off the edge of the 25x25x25 block, fall to the ground 20 units below.
#[test]
fn depth2_fall_off_grandparent_edge() {
    let mut world = ground_world(4);
    // Place a cell at y=1 (the block the user placed at depth 0)
    world.cells.insert(IVec3::new(0, 1, 0), ground_cell());

    // Drill into (0,1,0) then into block (2,4,2) inside it
    let nav1_stack = nav1(IVec3::new(0, 1, 0));
    {
        let grid = world.get_grid_mut(&nav1_stack).unwrap();
        let mut inner = VoxelGrid::new_empty();
        for y in 0..MODEL_SIZE { for z in 0..MODEL_SIZE { for x in 0..MODEL_SIZE {
            inner.slots[y][z][x] = CellSlot::Block(BlockType::Grass);
        }}}
        grid.slots[4][2][2] = CellSlot::Child(Box::new(inner));
    }
    let nav = nav2(IVec3::new(0, 1, 0), IVec3::new(2, 4, 2));

    // At depth 2, the placed cell (0,1,0) spans 25 units in current space.
    // Its top is at y=5 (from the parent grid's top).
    // The ground cell (0,0,0) has its top at y=-20 in current space.
    // Walking to x=16 (outside the parent grid but inside the grandparent)
    // should still detect the placed cell's blocks.
    // Walking to x=26 (outside the grandparent cell) enters neighboring
    // top-layer cells like (1,0,0) or (1,1,0).

    // Test: x=12 is inside the grandparent cell (0,1,0) which spans x=-10..15.
    // It's outside the parent grid (0..5) but inside the top-layer cell.
    assert!(block_solid(&world, &nav, IVec3::new(12, 4, 2)),
        "Blocks inside grandparent cell should be detected as solid");

    // Test: position at x=20, y=5. Fall from the top of the placed cell.
    // Should land on... the placed cell extends all the way to y=5 at this position.
    // If they walk to x=26 (outside the grandparent), they're above the ground.
    // Ground at (1,0,0) has top at y = -45 + 0*25 + 25 = -20 in current space.
    // Wait, let me recalculate. Using render_ancestor_transforms:
    // Level 0: scale=5, offset = -(2,4,2)*5 = (-10,-20,-10)
    // Level 1: scale=25, offset = (-10,-20,-10) - (0,1,0)*25 = (-10,-45,-10)
    // Ground cell (0,0,0) at top layer: position = (-10,-45,-10) + (0,0,0)*25 = (-10,-45,-10)
    // Its top = (-10,-45,-10) + (25,25,25) → y_top = -45+25 = -20
    // Ground cell (1,0,0): position = (-10,-45,-10) + (1,0,0)*25 = (15,-45,-10)
    // Its top y = -45+25 = -20

    // Player walks off to x=30 (well outside the placed cell, above (1,0,0) ground)
    let mut pos = Vec3::new(30.0, 5.0, 2.5);
    let mut vel = Vec3::ZERO;

    // Verify there's ground far below
    assert!(block_solid(&world, &nav, IVec3::new(30, -21, 2)),
        "Ground cell should be solid at y=-21 (inside ground cell in current space)");

    let dt = 1.0 / 60.0;
    for _ in 0..600 { // 10 seconds of falling
        vel.y -= 20.0 * dt;
        move_and_collide(&mut pos, &mut vel, Vec2::ZERO, dt, &world, &nav);
    }

    assert!(pos.y > -25.0, "Should land on ground, not fall to void. Got y={}", pos.y);
    assert!(pos.y < 5.0, "Should have fallen from y=5. Got y={}", pos.y);
}

// ============================================================
// Model registry / save flow
// ============================================================

#[test]
fn save_model_to_registry() {
    use crate::model::{ModelRegistry, VoxelModel};
    use crate::model::mesher::bake_model;

    let mut registry = ModelRegistry::default();
    assert_eq!(registry.models.len(), 0);

    // Simulate saving: create a custom block pattern and register it
    let mut blocks = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    blocks[0][0][0] = Some(BlockType::Brick);
    blocks[1][0][0] = Some(BlockType::Brick);

    let mut meshes = Assets::<Mesh>::default();
    let baked = bake_model(&blocks, &mut meshes);

    let id = registry.register(VoxelModel {
        name: "Test Tower".into(),
        blocks,
        baked,
    });

    assert_eq!(registry.models.len(), 1, "Registry should have 1 model after save");
    assert_eq!(registry.models[id.0].name, "Test Tower");
    assert_eq!(registry.models[id.0].blocks[0][0][0], Some(BlockType::Brick));
    assert_eq!(registry.models[id.0].blocks[1][0][0], Some(BlockType::Brick));
    assert_eq!(registry.models[id.0].blocks[2][0][0], None);
}

#[test]
fn save_multiple_models() {
    use crate::model::{ModelRegistry, VoxelModel};
    use crate::model::mesher::bake_model;

    let mut registry = ModelRegistry::default();
    let mut meshes = Assets::<Mesh>::default();

    for i in 0..3 {
        let mut blocks = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
        blocks[0][0][0] = Some(BlockType::ALL[i]);
        let baked = bake_model(&blocks, &mut meshes);
        registry.register(VoxelModel {
            name: format!("Model {}", i),
            blocks,
            baked,
        });
    }

    assert_eq!(registry.models.len(), 3);
    assert_eq!(registry.models[0].name, "Model 0");
    assert_eq!(registry.models[1].name, "Model 1");
    assert_eq!(registry.models[2].name, "Model 2");
}

#[test]
fn saved_model_blocks_are_independent() {
    use crate::model::{ModelRegistry, VoxelModel};
    use crate::model::mesher::bake_model;

    let mut registry = ModelRegistry::default();
    let mut meshes = Assets::<Mesh>::default();

    // Save a model
    let mut blocks = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    blocks[0][0][0] = Some(BlockType::Stone);
    let baked = bake_model(&blocks, &mut meshes);
    registry.register(VoxelModel { name: "Original".into(), blocks, baked });

    // Modify the original blocks array — should NOT affect the saved model
    blocks[0][0][0] = Some(BlockType::Grass);

    assert_eq!(registry.models[0].blocks[0][0][0], Some(BlockType::Stone),
        "Saved model should be independent copy, not reference");
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
