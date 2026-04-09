use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::layer::EditingContext;
use crate::model::mesher::bake_model;
use crate::model::{BakedSubMesh, ModelId, ModelRegistry, VoxelModel};
use crate::player::Player;

const RENDER_DISTANCE: i32 = 8;
const CELL_WORLD_SIZE: f32 = MODEL_SIZE as f32;

pub type BlockArray = [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Layer1World>()
            .init_resource::<LoadedCells>()
            .init_resource::<WorldHotbar>()
            .add_systems(Startup, setup_world)
            .add_systems(Update, manage_visible_cells);
    }
}

/// Each cell owns its block data + baked mesh. Editing a cell only affects that cell.
#[derive(Clone)]
pub struct CellData {
    pub template_id: ModelId,
    pub blocks: BlockArray,
    pub baked: Vec<BakedSubMesh>,
}

#[derive(Resource, Default)]
pub struct Layer1World {
    pub cells: HashMap<IVec3, CellData>,
}

#[derive(Resource, Default)]
pub struct LoadedCells {
    pub entities: HashMap<IVec3, Entity>,
}

#[derive(Component)]
pub struct Layer1Cell {
    pub coord: IVec3,
}

/// Which model template is selected for placement in world mode.
#[derive(Resource)]
pub struct WorldHotbar {
    pub selected_slot: usize,
}

impl Default for WorldHotbar {
    fn default() -> Self {
        Self { selected_slot: 0 }
    }
}

fn setup_world(
    mut meshes: ResMut<Assets<Mesh>>,
    mut registry: ResMut<ModelRegistry>,
    mut world: ResMut<Layer1World>,
) {
    // Ground template
    let mut ground: BlockArray = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for z in 0..MODEL_SIZE {
        for x in 0..MODEL_SIZE {
            ground[0][z][x] = Some(BlockType::Dirt);
            ground[1][z][x] = Some(BlockType::Dirt);
            ground[2][z][x] = Some(BlockType::Grass);
        }
    }
    registry.register(VoxelModel {
        name: "Ground".into(),
        blocks: ground,
        baked: bake_model(&ground, &mut meshes),
    });

    // Place ground — each cell gets its OWN copy of the block data
    let extent = 12;
    for z in -extent..extent {
        for x in -extent..extent {
            world.cells.insert(
                IVec3::new(x, 0, z),
                CellData {
                    template_id: ModelId(0),
                    blocks: ground,
                    baked: bake_model(&ground, &mut meshes),
                },
            );
        }
    }
}

fn manage_visible_cells(
    mut commands: Commands,
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

    let mut desired = HashSet::new();
    let rd = RENDER_DISTANCE;
    for dz in -rd..=rd {
        for dx in -rd..=rd {
            if dx * dx + dz * dz > rd * rd { continue; }
            for y in -2..10 {
                let coord = IVec3::new(player_cell.x + dx, y, player_cell.y + dz);
                if !world.cells.contains_key(&coord) { continue; }
                if let Some(ref ctx) = editing {
                    if coord == ctx.cell_coord { continue; }
                }
                desired.insert(coord);
            }
        }
    }

    for &coord in &desired {
        if loaded.entities.contains_key(&coord) { continue; }
        let Some(cell_data) = world.cells.get(&coord) else { continue };

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

        for sub in &cell_data.baked {
            let child = commands.spawn((
                Mesh3d(sub.mesh.clone()),
                MeshMaterial3d(materials.get(sub.block_type)),
                Transform::default(),
            )).id();
            commands.entity(root).add_child(child);
        }

        loaded.entities.insert(coord, root);
    }

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
