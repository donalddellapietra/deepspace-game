use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::MODEL_SIZE;
use crate::layer::EditingContext;
use crate::model::mesher::bake_model;
use crate::player::{Player, Velocity};
use crate::world::{Layer1World, LoadedCells};

use super::SharedCubeMesh;

#[derive(Component)]
pub struct EditEntity;

#[derive(Component)]
pub struct EditBlock {
    pub local_pos: IVec3,
}

pub fn spawn_edit_grid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    context: Res<EditingContext>,
    world: Res<Layer1World>,
    materials: Res<BlockMaterials>,
    loaded: Res<LoadedCells>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
) {
    let cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    commands.insert_resource(SharedCubeMesh(cube.clone()));

    let Some(cell_data) = world.cells.get(&context.cell_coord) else { return };

    let cell_origin = context.cell_coord.as_vec3() * MODEL_SIZE as f32;

    // Hide the baked cell entity
    if let Some(&entity) = loaded.entities.get(&context.cell_coord) {
        commands.entity(entity).insert(Visibility::Hidden);
    }

    // Spawn individual block entities from this cell's OWN data
    for y in 0..MODEL_SIZE {
        for z in 0..MODEL_SIZE {
            for x in 0..MODEL_SIZE {
                let Some(block_type) = cell_data.blocks[y][z][x] else { continue };
                let pos = cell_origin + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);

                commands.spawn((
                    EditEntity,
                    EditBlock { local_pos: IVec3::new(x as i32, y as i32, z as i32) },
                    Mesh3d(cube.clone()),
                    MeshMaterial3d(materials.get(block_type)),
                    Transform::from_translation(pos),
                ));
            }
        }
    }

    // Position player in front of cell looking in -Z (forward at yaw=0)
    if let Ok((mut tf, mut vel)) = player_q.single_mut() {
        vel.0 = Vec3::ZERO;
        tf.translation = cell_origin + Vec3::new(2.5, 5.5, 7.0);
    }
}

pub fn exit_edit_mode(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    context: Res<EditingContext>,
    mut world: ResMut<Layer1World>,
    mut loaded: ResMut<LoadedCells>,
    edit_entities: Query<Entity, With<EditEntity>>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
) {
    // Despawn edit entities
    for entity in &edit_entities {
        commands.entity(entity).despawn();
    }

    // Re-bake this cell's own block data
    if let Some(cell_data) = world.cells.get_mut(&context.cell_coord) {
        cell_data.baked = bake_model(&cell_data.blocks, &mut meshes);
    }

    // Despawn the old rendered entity + remove from loaded so it gets respawned
    if let Some(entity) = loaded.entities.remove(&context.cell_coord) {
        commands.entity(entity).despawn();
    }

    // Return player to world
    if let Ok((mut tf, mut vel)) = player_q.single_mut() {
        vel.0 = Vec3::ZERO;
        tf.translation = context.return_position;
    }

    commands.remove_resource::<EditingContext>();
    commands.remove_resource::<SharedCubeMesh>();
}
