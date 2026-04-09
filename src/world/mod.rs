pub mod chunk;
pub mod terrain;

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use chunk::{Chunk, CHUNK_WORLD_SIZE, generate_chunk_mesh};
use terrain::TerrainGenerator;

use crate::player::Player;

const RENDER_DISTANCE: i32 = 8;
const MAX_CHUNKS_PER_FRAME: usize = 4;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(TerrainGenerator::new(42))
            .add_systems(Startup, setup_world)
            .add_systems(Update, manage_chunks);
    }
}

#[derive(Resource)]
struct ChunkManager {
    loaded: HashMap<IVec2, Entity>,
    material: Handle<StandardMaterial>,
}

fn setup_world(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.55, 0.2),
        perceptual_roughness: 0.95,
        ..default()
    });

    commands.insert_resource(ChunkManager {
        loaded: HashMap::new(),
        material,
    });
}

fn manage_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut manager: ResMut<ChunkManager>,
    terrain: Res<TerrainGenerator>,
    player_query: Query<&Transform, With<Player>>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let pos = player_tf.translation;

    let player_chunk = IVec2::new(
        (pos.x / CHUNK_WORLD_SIZE).floor() as i32,
        (pos.z / CHUNK_WORLD_SIZE).floor() as i32,
    );

    // Desired chunk set: circular region around player
    let mut desired = HashSet::new();
    let rd = RENDER_DISTANCE;
    for z in -rd..=rd {
        for x in -rd..=rd {
            if x * x + z * z <= rd * rd {
                desired.insert(player_chunk + IVec2::new(x, z));
            }
        }
    }

    // Spawn missing chunks, limited per frame to avoid hitching
    let mut spawned = 0;
    // Sort by distance so closest chunks load first
    let mut to_spawn: Vec<IVec2> = desired
        .iter()
        .filter(|c| !manager.loaded.contains_key(c))
        .copied()
        .collect();
    to_spawn.sort_by_key(|c| {
        let d = *c - player_chunk;
        d.x * d.x + d.y * d.y
    });

    for coord in to_spawn {
        if spawned >= MAX_CHUNKS_PER_FRAME {
            break;
        }
        let mesh = generate_chunk_mesh(coord, &terrain);
        let entity = commands
            .spawn((
                Chunk { coord },
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(manager.material.clone()),
                Transform::from_xyz(
                    coord.x as f32 * CHUNK_WORLD_SIZE,
                    0.0,
                    coord.y as f32 * CHUNK_WORLD_SIZE,
                ),
            ))
            .id();
        manager.loaded.insert(coord, entity);
        spawned += 1;
    }

    // Despawn chunks outside render distance
    let to_remove: Vec<IVec2> = manager
        .loaded
        .keys()
        .filter(|c| !desired.contains(c))
        .copied()
        .collect();

    for coord in to_remove {
        if let Some(entity) = manager.loaded.remove(&coord) {
            commands.entity(entity).despawn();
        }
    }
}
