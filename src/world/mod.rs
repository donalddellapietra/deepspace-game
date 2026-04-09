use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::layer::{EditingContext, GameLayer};
use crate::model::mesher::bake_model;
use crate::model::{ModelId, ModelRegistry, VoxelModel};
use crate::player::Player;

const RENDER_DISTANCE: i32 = 8;
const CELL_WORLD_SIZE: f32 = MODEL_SIZE as f32;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Layer1World>()
            .init_resource::<LoadedCells>()
            .add_systems(Startup, setup_world)
            .add_systems(Update, manage_visible_cells);
    }
}

/// Placement data for a cell in the world.
#[derive(Clone)]
pub struct CellData {
    pub model_id: ModelId,
}

/// The layer 1 world grid (data only, no rendering state).
#[derive(Resource, Default)]
pub struct Layer1World {
    pub cells: HashMap<IVec3, CellData>,
}

/// Tracks spawned entities for visible cells (rendering state).
#[derive(Resource, Default)]
pub struct LoadedCells {
    pub entities: HashMap<IVec3, Entity>,
}

/// Marker on the root entity of a rendered layer 1 cell.
#[derive(Component)]
pub struct Layer1Cell {
    pub coord: IVec3,
}

fn setup_world(
    mut meshes: ResMut<Assets<Mesh>>,
    mut registry: ResMut<ModelRegistry>,
    mut world: ResMut<Layer1World>,
) {
    // Ground model: 5x5 slab, 3 blocks tall (dirt + grass top)
    let mut ground = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for z in 0..MODEL_SIZE {
        for x in 0..MODEL_SIZE {
            ground[0][z][x] = Some(BlockType::Dirt);
            ground[1][z][x] = Some(BlockType::Dirt);
            ground[2][z][x] = Some(BlockType::Grass);
        }
    }
    let ground_id = registry.register(VoxelModel {
        name: "Ground".into(),
        blocks: ground,
        baked: bake_model(&ground, &mut meshes),
    });

    // Tree model
    let mut tree = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    tree[0][2][2] = Some(BlockType::Wood);
    tree[1][2][2] = Some(BlockType::Wood);
    tree[2][2][2] = Some(BlockType::Wood);
    for dy in 3..5 {
        for dz in 1..4 {
            for dx in 1..4 {
                tree[dy][dz][dx] = Some(BlockType::Leaf);
            }
        }
    }
    let tree_id = registry.register(VoxelModel {
        name: "Tree".into(),
        blocks: tree,
        baked: bake_model(&tree, &mut meshes),
    });

    // Place ground grid with scattered trees
    let extent = 12;
    for z in -extent..extent {
        for x in -extent..extent {
            world.cells.insert(IVec3::new(x, 0, z), CellData { model_id: ground_id });

            let hash = ((x.wrapping_mul(73856093)) ^ (z.wrapping_mul(19349663))).unsigned_abs();
            if hash % 7 == 0 {
                world.cells.insert(IVec3::new(x, 1, z), CellData { model_id: tree_id });
            }
        }
    }
}

fn manage_visible_cells(
    mut commands: Commands,
    registry: Res<ModelRegistry>,
    materials: Res<BlockMaterials>,
    world: Res<Layer1World>,
    mut loaded: ResMut<LoadedCells>,
    player_query: Query<&Transform, With<Player>>,
    editing: Option<Res<EditingContext>>,
) {
    let Ok(player_tf) = player_query.single() else { return };
    let pos = player_tf.translation;

    let player_cell = IVec2::new(
        (pos.x / CELL_WORLD_SIZE).floor() as i32,
        (pos.z / CELL_WORLD_SIZE).floor() as i32,
    );

    // Collect desired visible cells
    let mut desired = HashSet::new();
    let rd = RENDER_DISTANCE;
    for dz in -rd..=rd {
        for dx in -rd..=rd {
            if dx * dx + dz * dz > rd * rd { continue; }
            for y in -2..10 {
                let coord = IVec3::new(player_cell.x + dx, y, player_cell.y + dz);
                if world.cells.contains_key(&coord) {
                    // Skip the cell being edited (it renders as individual blocks)
                    if let Some(ref ctx) = editing {
                        if coord == ctx.cell_coord { continue; }
                    }
                    desired.insert(coord);
                }
            }
        }
    }

    // Spawn missing
    for &coord in &desired {
        if loaded.entities.contains_key(&coord) { continue; }
        let Some(cell_data) = world.cells.get(&coord) else { continue };
        let Some(model) = registry.get(cell_data.model_id) else { continue };

        let world_pos = Vec3::new(
            coord.x as f32 * CELL_WORLD_SIZE,
            coord.y as f32 * CELL_WORLD_SIZE,
            coord.z as f32 * CELL_WORLD_SIZE,
        );

        let root = commands.spawn((
            Layer1Cell { coord },
            Transform::from_translation(world_pos),
            Visibility::Inherited,
        )).id();

        // One child per sub-mesh (per block type)
        for sub in &model.baked {
            let child = commands.spawn((
                Mesh3d(sub.mesh.clone()),
                MeshMaterial3d(materials.get(sub.block_type)),
                Transform::default(),
            )).id();
            commands.entity(root).add_child(child);
        }

        loaded.entities.insert(coord, root);
    }

    // Despawn far / unneeded
    let to_remove: Vec<IVec3> = loaded.entities.keys()
        .filter(|c| !desired.contains(c))
        .copied()
        .collect();

    for coord in to_remove {
        if let Some(entity) = loaded.entities.remove(&coord) {
            commands.entity(entity).despawn();
        }
    }
}
