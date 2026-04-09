use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::MODEL_SIZE;
use crate::layer::EditingContext;
use crate::model::ModelRegistry;
use crate::player::{Player, Velocity};
use crate::world::LoadedCells;

use super::SharedCubeMesh;

/// Marker for entities spawned during editing. Cleaned up on exit.
#[derive(Component)]
pub struct EditEntity;

/// Marker on individual block entities in the edit grid.
#[derive(Component)]
pub struct EditBlock {
    pub local_pos: IVec3,
}

pub fn spawn_edit_grid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    context: Res<EditingContext>,
    registry: Res<ModelRegistry>,
    materials: Res<BlockMaterials>,
    loaded: Res<LoadedCells>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
) {
    // Create or reuse shared cube mesh
    let cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    commands.insert_resource(SharedCubeMesh(cube.clone()));

    let Some(model) = registry.get(context.model_id) else { return };

    let cell_origin = Vec3::new(
        context.cell_coord.x as f32 * MODEL_SIZE as f32,
        context.cell_coord.y as f32 * MODEL_SIZE as f32,
        context.cell_coord.z as f32 * MODEL_SIZE as f32,
    );

    // Hide the baked cell entity
    if let Some(&entity) = loaded.entities.get(&context.cell_coord) {
        commands.entity(entity).insert(Visibility::Hidden);
    }

    // Spawn individual block entities
    for y in 0..MODEL_SIZE {
        for z in 0..MODEL_SIZE {
            for x in 0..MODEL_SIZE {
                let Some(block_type) = model.blocks[y][z][x] else { continue };

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

    // Move player inside the cell
    if let Ok((mut tf, mut vel)) = player_q.single_mut() {
        vel.0 = Vec3::ZERO;
        tf.translation = cell_origin + Vec3::new(2.5, 3.5, -1.0);
    }
}

pub fn exit_edit_mode(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    context: Res<EditingContext>,
    mut registry: ResMut<ModelRegistry>,
    loaded: Res<LoadedCells>,
    edit_entities: Query<Entity, With<EditEntity>>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
) {
    // Despawn all edit entities
    for entity in &edit_entities {
        commands.entity(entity).despawn();
    }

    // Rebake the model
    registry.rebake(context.model_id, &mut meshes);

    // Show the baked cell entity again
    if let Some(&entity) = loaded.entities.get(&context.cell_coord) {
        // The old children have stale meshes — despawn and let manage_visible_cells respawn it
        commands.entity(entity).despawn();
    }

    // Return player to world position
    if let Ok((mut tf, mut vel)) = player_q.single_mut() {
        vel.0 = Vec3::ZERO;
        tf.translation = context.return_position;
    }

    commands.remove_resource::<EditingContext>();
    commands.remove_resource::<SharedCubeMesh>();
}
