//! Depth-aware rendering on top of the [`super::library::MeshLibrary`].
//!
//! Depth 0 renders super-super-chunks (level 3). Depth 1 renders super-
//! chunks (level 2). Depth 2 and 3 render chunks (level 1) at two different
//! visual scales. Each rendered entity stores the library id it currently
//! shows; on the next render pass we compare the "correct" id against the
//! stored one and respawn only if it changed.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::player::Player;

use super::chunk::{S, SUPER};
use super::library::{
    compute_level2_key, compute_level3_key, ensure_level1_id, MeshLibrary, EMPTY_ID,
};
use super::state::WorldState;
use super::render_distance_for_depth;

#[derive(Component)]
pub struct ChunkEntity;

#[derive(Resource, Default)]
pub struct RenderState {
    /// render key → (entity, library id it's currently showing).
    /// Render key is super-chunk key at depth 0, chunk key at depth 1/2.
    pub entities: HashMap<IVec3, (Entity, u64)>,
    pub rendered_depth: usize,
    pub needs_full_refresh: bool,
}

pub fn render_world(
    mut commands: Commands,
    materials: Res<BlockMaterials>,
    mut state: ResMut<WorldState>,
    mut rs: ResMut<RenderState>,
    mut library: ResMut<MeshLibrary>,
    mut meshes: ResMut<Assets<Mesh>>,
    player_q: Query<&Transform, With<Player>>,
) {
    let Ok(player_tf) = player_q.single() else { return };

    if rs.rendered_depth != state.depth || rs.needs_full_refresh {
        // Decrement the correct level's refcount based on which layer we
        // were rendering before the drain. Depth-1 entities hold level-2,
        // depth-0 entities hold level-3. Chunk-level render modes don't
        // refcount via entities.
        let drained_depth = rs.rendered_depth;
        for (_, (entity, id)) in rs.entities.drain() {
            commands.entity(entity).despawn();
            match drained_depth {
                0 => library.level3_decrement(id),
                1 => library.level2_decrement(id),
                _ => {}
            }
        }
        rs.rendered_depth = state.depth;
        rs.needs_full_refresh = false;
    }

    match state.depth {
        0 => render_super_super_chunks(
            &mut commands,
            &materials,
            &mut state,
            &mut rs,
            &mut library,
            &mut meshes,
            player_tf,
        ),
        1 => render_super_chunks(
            &mut commands,
            &materials,
            &mut state,
            &mut rs,
            &mut library,
            &mut meshes,
            player_tf,
        ),
        2 => render_chunks(
            &mut commands,
            &materials,
            &mut state,
            &mut rs,
            &mut library,
            &mut meshes,
            player_tf,
            1.0 / S as f32,
            1.0,
        ),
        _ => render_chunks(
            &mut commands,
            &materials,
            &mut state,
            &mut rs,
            &mut library,
            &mut meshes,
            player_tf,
            1.0,
            S as f32,
        ),
    }

    // Consume this frame's dirty set now that every render path has had a
    // chance to act on it. Anything inserted during the next frame's
    // PreUpdate / Update passes will be picked up on the next render.
    state.dirty_chunks.clear();
}

fn render_chunks(
    commands: &mut Commands,
    materials: &BlockMaterials,
    state: &mut WorldState,
    rs: &mut RenderState,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
    player_tf: &Transform,
    view_scale: f32,
    chunk_size_w: f32,
) {
    let player_key = IVec3::new(
        (player_tf.translation.x / chunk_size_w).floor() as i32,
        (player_tf.translation.y / chunk_size_w).floor() as i32,
        (player_tf.translation.z / chunk_size_w).floor() as i32,
    );

    let rd = render_distance_for_depth(state.depth);

    // Range query via the spatial index — work is O(|visible chunks|), not
    // O(|world.chunks|). Same linear-style "rebuild every frame" shape that
    // keeps frametime consistent, just massively cheaper per frame.
    let desired: HashSet<IVec3> = state
        .world
        .index
        .chunks_in_sphere(player_key, rd)
        .into_iter()
        .collect();

    for &key in &desired {
        let level1_id = match state.world.chunks.get_mut(&key) {
            Some(chunk) => ensure_level1_id(chunk, library, meshes),
            None => continue,
        };

        if let Some(&(entity, existing_id)) = rs.entities.get(&key) {
            if existing_id == level1_id {
                continue;
            }
            commands.entity(entity).despawn();
            rs.entities.remove(&key);
        }

        if level1_id == EMPTY_ID {
            continue;
        }

        let entry = library.level1_get(level1_id).expect("ensured above");
        let pos = key.as_vec3() * chunk_size_w;
        let root = commands
            .spawn((
                ChunkEntity,
                Transform::from_translation(pos).with_scale(Vec3::splat(view_scale)),
                Visibility::Inherited,
            ))
            .id();
        for sub in &entry.baked {
            let child = commands
                .spawn((
                    Mesh3d(sub.mesh.clone()),
                    MeshMaterial3d(materials.get(sub.block_type)),
                    Transform::default(),
                ))
                .id();
            commands.entity(root).add_child(child);
        }
        rs.entities.insert(key, (root, level1_id));
    }

    // Stale despawn. Level-1 refs are from chunks, not entities.
    let stale: Vec<_> = rs
        .entities
        .keys()
        .filter(|k| !desired.contains(k))
        .copied()
        .collect();
    for key in stale {
        if let Some((e, _)) = rs.entities.remove(&key) {
            commands.entity(e).despawn();
        }
    }
}

fn render_super_chunks(
    commands: &mut Commands,
    materials: &BlockMaterials,
    state: &mut WorldState,
    rs: &mut RenderState,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
    player_tf: &Transform,
) {
    let player_key = IVec3::new(
        player_tf.translation.x.floor() as i32,
        player_tf.translation.y.floor() as i32,
        player_tf.translation.z.floor() as i32,
    );

    // Range query via the spatial index. The index lives in chunk coords,
    // so we ask for chunks in a sphere large enough to cover every super-
    // chunk within `rd`, then map to super-chunks and filter at the super-
    // chunk level. Layer-agnostic: the index doesn't know what a super-
    // chunk is.
    let rd = render_distance_for_depth(state.depth);
    let rd_sq = rd * rd;
    let chunk_center = player_key * S + IVec3::splat(S / 2);
    let chunk_radius = (rd + 1) * S;
    let mut desired: HashSet<IVec3> = HashSet::new();
    for chunk_key in state.world.index.chunks_in_sphere(chunk_center, chunk_radius) {
        let super_key = IVec3::new(
            chunk_key.x.div_euclid(S),
            chunk_key.y.div_euclid(S),
            chunk_key.z.div_euclid(S),
        );
        if desired.contains(&super_key) {
            continue;
        }
        let d = super_key - player_key;
        if d.x * d.x + d.y * d.y + d.z * d.z > rd_sq {
            continue;
        }
        desired.insert(super_key);
    }

    let view_scale = 1.0 / SUPER as f32;

    // Derive per-level dirty set from the layer-agnostic dirty_chunks list.
    let dirty_supers: HashSet<IVec3> = state
        .dirty_chunks
        .iter()
        .map(|ck| IVec3::new(ck.x.div_euclid(S), ck.y.div_euclid(S), ck.z.div_euclid(S)))
        .collect();

    for &super_key in &desired {
        let is_dirty = dirty_supers.contains(&super_key);
        let already_rendered = rs.entities.contains_key(&super_key);
        if already_rendered && !is_dirty {
            continue;
        }

        let level2_key = compute_level2_key(&mut state.world, super_key, library, meshes);

        let all_empty = level2_key
            .iter()
            .flatten()
            .flatten()
            .all(|&id| id == EMPTY_ID);
        let level2_id = if all_empty {
            EMPTY_ID
        } else {
            library.level2_lookup_or_bake(level2_key, &state.world, super_key, meshes)
        };

        if let Some(&(entity, existing_id)) = rs.entities.get(&super_key) {
            if existing_id == level2_id {
                continue;
            }
            commands.entity(entity).despawn();
            rs.entities.remove(&super_key);
            library.level2_decrement(existing_id);
        }

        if level2_id == EMPTY_ID {
            continue;
        }

        library.level2_increment(level2_id);
        let entry = library.level2_get(level2_id).expect("just ensured");
        let pos = super_key.as_vec3();
        let root = commands
            .spawn((
                ChunkEntity,
                Transform::from_translation(pos).with_scale(Vec3::splat(view_scale)),
                Visibility::Inherited,
            ))
            .id();
        for sub in &entry.baked {
            let child = commands
                .spawn((
                    Mesh3d(sub.mesh.clone()),
                    MeshMaterial3d(materials.get(sub.block_type)),
                    Transform::default(),
                ))
                .id();
            commands.entity(root).add_child(child);
        }
        rs.entities.insert(super_key, (root, level2_id));
    }

    // Stale despawn — decrement each entity's level-2 refcount.
    let stale: Vec<_> = rs
        .entities
        .keys()
        .filter(|k| !desired.contains(k))
        .copied()
        .collect();
    for key in stale {
        if let Some((e, id)) = rs.entities.remove(&key) {
            commands.entity(e).despawn();
            library.level2_decrement(id);
        }
    }
}

fn render_super_super_chunks(
    commands: &mut Commands,
    materials: &BlockMaterials,
    state: &mut WorldState,
    rs: &mut RenderState,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
    player_tf: &Transform,
) {
    // Exact same shape as `render_super_chunks`, just one layer up:
    // query the spatial index in chunk coords, derive the super-super-chunk
    // key, dedupe, filter at the SSS level.
    let player_key = IVec3::new(
        player_tf.translation.x.floor() as i32,
        player_tf.translation.y.floor() as i32,
        player_tf.translation.z.floor() as i32,
    );

    let rd = render_distance_for_depth(state.depth);
    let rd_sq = rd * rd;
    // SSS spans SUPER chunks per axis. The index is in chunk coords; ask
    // for chunks in a sphere wide enough to cover every SSS within `rd`.
    let chunk_center = player_key * SUPER + IVec3::splat(SUPER / 2);
    let chunk_radius = (rd + 1) * SUPER;
    let mut desired: HashSet<IVec3> = HashSet::new();
    for chunk_key in state.world.index.chunks_in_sphere(chunk_center, chunk_radius) {
        let sss_key = IVec3::new(
            chunk_key.x.div_euclid(SUPER),
            chunk_key.y.div_euclid(SUPER),
            chunk_key.z.div_euclid(SUPER),
        );
        if desired.contains(&sss_key) {
            continue;
        }
        let d = sss_key - player_key;
        if d.x * d.x + d.y * d.y + d.z * d.z > rd_sq {
            continue;
        }
        desired.insert(sss_key);
    }

    let view_scale = 1.0 / (SUPER as f32 * S as f32);

    // Derive per-level dirty set from the layer-agnostic dirty_chunks list.
    let dirty_sss: HashSet<IVec3> = state
        .dirty_chunks
        .iter()
        .map(|ck| {
            IVec3::new(
                ck.x.div_euclid(SUPER),
                ck.y.div_euclid(SUPER),
                ck.z.div_euclid(SUPER),
            )
        })
        .collect();

    for &sss_key in &desired {
        let is_dirty = dirty_sss.contains(&sss_key);
        let already_rendered = rs.entities.contains_key(&sss_key);
        if already_rendered && !is_dirty {
            continue;
        }

        let level3_key = compute_level3_key(&mut state.world, sss_key, library, meshes);

        let all_empty = level3_key
            .iter()
            .flatten()
            .flatten()
            .all(|&id| id == EMPTY_ID);
        let level3_id = if all_empty {
            EMPTY_ID
        } else {
            library.level3_lookup_or_bake(level3_key, &state.world, sss_key, meshes)
        };

        if let Some(&(entity, existing_id)) = rs.entities.get(&sss_key) {
            if existing_id == level3_id {
                continue;
            }
            commands.entity(entity).despawn();
            rs.entities.remove(&sss_key);
            library.level3_decrement(existing_id);
        }

        if level3_id == EMPTY_ID {
            continue;
        }

        library.level3_increment(level3_id);
        let entry = library.level3_get(level3_id).expect("just ensured");
        let pos = sss_key.as_vec3();
        let root = commands
            .spawn((
                ChunkEntity,
                Transform::from_translation(pos).with_scale(Vec3::splat(view_scale)),
                Visibility::Inherited,
            ))
            .id();
        for sub in &entry.baked {
            let child = commands
                .spawn((
                    Mesh3d(sub.mesh.clone()),
                    MeshMaterial3d(materials.get(sub.block_type)),
                    Transform::default(),
                ))
                .id();
            commands.entity(root).add_child(child);
        }
        rs.entities.insert(sss_key, (root, level3_id));
    }

    let stale: Vec<_> = rs
        .entities
        .keys()
        .filter(|k| !desired.contains(k))
        .copied()
        .collect();
    for key in stale {
        if let Some((e, id)) = rs.entities.remove(&key) {
            commands.entity(e).despawn();
            library.level3_decrement(id);
        }
    }
}
