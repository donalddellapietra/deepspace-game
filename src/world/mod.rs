pub mod collision;

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::model::mesher::{bake_model, bake_volume};
use crate::model::BakedSubMesh;
use crate::player::Player;

const S: i32 = MODEL_SIZE as i32;
/// Side length of a "super-chunk" in integer blocks: MODEL_SIZE chunks per
/// axis × MODEL_SIZE blocks per chunk = 25.
pub const SUPER: i32 = S * S;

/// How many drill levels are supported. Depth 0 = most zoomed out
/// (super-chunks visible), depth MAX_DEPTH = most zoomed in (single integer
/// blocks visible). Each step is a 5× linear zoom.
pub const MAX_DEPTH: usize = 2;

/// How far the player can see, measured in *render entities* (super-chunks at
/// depth 0, chunks at depth 1/2). Same entity budget at every depth.
const RENDER_DISTANCE: i32 = 8;

// ============================================================
// Core data: Chunk + FlatWorld (one global integer-block grid)
// ============================================================

#[derive(Clone)]
pub struct Chunk {
    pub blocks: [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
    pub mesh_dirty: bool,
    pub baked: Vec<BakedSubMesh>,
    /// True iff a player edit has touched this chunk. The streaming generator
    /// must never overwrite a user_modified chunk. Removed cells are kept as
    /// empty `user_modified=true` chunks so the generator skips them too.
    pub user_modified: bool,
}

impl Chunk {
    pub fn new_empty() -> Self {
        Self {
            blocks: [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
            mesh_dirty: true,
            baked: vec![],
            user_modified: false,
        }
    }

    pub fn new_filled(bt: BlockType) -> Self {
        Self {
            blocks: [[[Some(bt); MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
            mesh_dirty: true,
            baked: vec![],
            user_modified: false,
        }
    }

    pub fn tombstone() -> Self {
        let mut c = Self::new_empty();
        c.user_modified = true;
        c
    }
}

/// All blocks in the game live here, indexed by integer-block coordinate
/// through the chunk grid. There is exactly one of these — drilling never
/// creates a new world, it just changes how we view this one.
#[derive(Clone, Default)]
pub struct FlatWorld {
    pub chunks: HashMap<IVec3, Chunk>,
}

impl FlatWorld {
    pub fn get(&self, coord: IVec3) -> Option<BlockType> {
        let (key, local) = Self::decompose(coord);
        self.chunks.get(&key)?.blocks[local.y as usize][local.z as usize][local.x as usize]
    }

    pub fn set(&mut self, coord: IVec3, block: Option<BlockType>) {
        let (key, local) = Self::decompose(coord);
        let chunk = self.chunks.entry(key).or_insert_with(Chunk::new_empty);
        chunk.blocks[local.y as usize][local.z as usize][local.x as usize] = block;
        chunk.mesh_dirty = true;
    }

    pub fn is_solid(&self, coord: IVec3) -> bool {
        self.get(coord).is_some()
    }

    /// True iff the chunk at `key` exists and contains at least one block.
    /// Used by depth-1 collision, where one chunk = one bevy unit cube.
    pub fn chunk_solid(&self, key: IVec3) -> bool {
        self.chunks
            .get(&key)
            .is_some_and(|c| c.blocks.iter().flatten().flatten().any(|b| b.is_some()))
    }

    /// True iff any of the 5×5×5 chunks beneath this super-chunk key has a
    /// block. Used by depth-0 collision, where one super-chunk = one bevy unit.
    pub fn super_chunk_solid(&self, super_key: IVec3) -> bool {
        for cz in 0..S {
            for cy in 0..S {
                for cx in 0..S {
                    let chunk_key = super_key * S + IVec3::new(cx, cy, cz);
                    if self.chunk_solid(chunk_key) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn decompose(coord: IVec3) -> (IVec3, IVec3) {
        let key = IVec3::new(
            coord.x.div_euclid(S),
            coord.y.div_euclid(S),
            coord.z.div_euclid(S),
        );
        let local = IVec3::new(
            coord.x.rem_euclid(S),
            coord.y.rem_euclid(S),
            coord.z.rem_euclid(S),
        );
        (key, local)
    }
}

// ============================================================
// World state: one global world + a current zoom depth
// ============================================================

#[derive(Resource, Default)]
pub struct WorldState {
    pub world: FlatWorld,
    /// Current zoom level. 0 = most zoomed out (super-chunks), MAX_DEPTH = most
    /// zoomed in (single integer blocks).
    pub depth: usize,
    /// Super-chunks whose cached bake is stale and must be rebuilt before the
    /// next render at depth 0. Editor systems insert into this set after edits.
    pub dirty_supers: HashSet<IVec3>,
}

impl collision::SolidQuery for WorldState {
    fn is_solid(&self, coord: IVec3) -> bool {
        // `coord` is in the current depth's bevy-block grid, i.e. one integer
        // unit equals one player-interactable cube at this depth.
        match self.depth {
            0 => self.world.super_chunk_solid(coord),
            1 => self.world.chunk_solid(coord),
            _ => self.world.is_solid(coord),
        }
    }
}

impl WorldState {
    pub fn is_top_layer(&self) -> bool {
        self.depth == 0
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Drill in: zoom by 5×. The world is unchanged; only the player's bevy
    /// position scales so they stay over the same integer-block location.
    pub fn drill_in(&mut self, player_pos: Vec3) -> Option<Vec3> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        Some(player_pos * S as f32)
    }

    /// Drill out: zoom by 1/5×.
    pub fn drill_out(&mut self, player_pos: Vec3) -> Option<Vec3> {
        if self.depth == 0 {
            return None;
        }
        self.depth -= 1;
        Some(player_pos / S as f32)
    }

    /// Mark the super-chunk that owns this integer block as needing a rebake.
    pub fn dirty_super_for_block(&mut self, block_coord: IVec3) {
        self.dirty_supers.insert(IVec3::new(
            block_coord.x.div_euclid(SUPER),
            block_coord.y.div_euclid(SUPER),
            block_coord.z.div_euclid(SUPER),
        ));
    }

    /// Mark the super-chunk that owns this chunk key as needing a rebake.
    pub fn dirty_super_for_chunk(&mut self, chunk_key: IVec3) {
        self.dirty_supers.insert(IVec3::new(
            chunk_key.x.div_euclid(S),
            chunk_key.y.div_euclid(S),
            chunk_key.z.div_euclid(S),
        ));
    }
}

// ============================================================
// Procedural generation
// ============================================================

const WORLD_SEED: u64 = 0xDEAD_BEEF_F00D_CAFE;

/// Deterministic per-chunk content. Returns None for chunk coords that should
/// not be materialized (above the rock layer or below bedrock). Never touches
/// world state — pure function of `coord`.
pub fn generate_chunk(coord: IVec3) -> Option<Chunk> {
    use BlockType::*;
    match coord.y {
        0 | 1 => Some(Chunk::new_filled(Stone)),
        2 | 3 => Some(Chunk::new_filled(Dirt)),
        4 => Some(Chunk::new_filled(Grass)),
        5 => {
            // Sparse layer above the grass: one rock at a deterministic
            // (rx, rz) per (cx, cz) column. local y = 0 puts the rock on top
            // of the grass below.
            let mut chunk = Chunk::new_empty();
            let (rx, rz) = rock_position(coord.x, coord.z);
            chunk.blocks[0][rz][rx] = Some(Stone);
            Some(chunk)
        }
        _ => None,
    }
}

fn rock_position(cx: i32, cz: i32) -> (usize, usize) {
    let h = WORLD_SEED
        ^ (cx as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (cz as i64 as u64).wrapping_mul(0xBF58476D1CE4E5B9);
    let rx = (h % MODEL_SIZE as u64) as usize;
    let rz = ((h / MODEL_SIZE as u64) % MODEL_SIZE as u64) as usize;
    (rx, rz)
}

// ============================================================
// Baking: super-chunks (depth 0) reuse the generic bake_volume
// ============================================================

/// Bake one super-chunk's 25×25×25 integer-block contents into per-block-type
/// sub-meshes. Mesh local coordinates are 0..SUPER on each axis.
pub fn bake_super_chunk(
    world: &FlatWorld,
    super_key: IVec3,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let origin = super_key * SUPER;
    bake_volume(
        SUPER,
        |x, y, z| {
            // Out-of-bounds means "no neighbor here", so border faces render.
            // (We don't cull across super-chunk borders — keeps the render
            // cache local to one super-chunk at a small visual cost.)
            if x < 0 || y < 0 || z < 0 || x >= SUPER || y >= SUPER || z >= SUPER {
                return None;
            }
            let ix = origin.x + x;
            let iy = origin.y + y;
            let iz = origin.z + z;
            let chunk_key = IVec3::new(ix.div_euclid(S), iy.div_euclid(S), iz.div_euclid(S));
            let lx = ix.rem_euclid(S) as usize;
            let ly = iy.rem_euclid(S) as usize;
            let lz = iz.rem_euclid(S) as usize;
            world.chunks.get(&chunk_key).and_then(|c| c.blocks[ly][lz][lx])
        },
        meshes,
    )
}

// ============================================================
// Rendering: depth-aware (super-chunks at 0, chunks at 1/2)
// ============================================================

#[derive(Component)]
pub struct ChunkEntity;

#[derive(Resource, Default)]
pub struct RenderState {
    /// Spawned root entities, keyed by their depth-specific render key
    /// (super_key at depth 0, chunk_key at depth 1/2).
    pub entities: HashMap<IVec3, Entity>,
    pub rendered_depth: usize,
    pub needs_full_refresh: bool,
}

/// State for the streaming generator. We only re-stream when the player has
/// crossed a chunk-column boundary, and we only generate the new edge.
#[derive(Resource, Default)]
pub struct StreamState {
    pub last_column: Option<IVec2>,
}

/// How wide a window of chunk columns to keep generated around the player,
/// measured in chunks per axis from the player's column. RENDER_DISTANCE is
/// in super-chunks at depth 0, so multiply by S to get chunks, plus a margin.
const STREAM_RADIUS_CHUNKS: i32 = RENDER_DISTANCE * S + S;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldState>()
            .init_resource::<RenderState>()
            .init_resource::<StreamState>()
            .add_systems(PreUpdate, generate_terrain)
            .add_systems(Update, render_world);
    }
}

/// Streaming terrain generator. Runs in PreUpdate so the world exists before
/// player physics or render see it. Only ever inserts chunks that are absent;
/// user_modified chunks (including tombstones) are never overwritten.
fn generate_terrain(
    mut stream: ResMut<StreamState>,
    mut state: ResMut<WorldState>,
    player_q: Query<&Transform, With<Player>>,
) {
    let Ok(tf) = player_q.single() else { return };

    // Convert the player's bevy position to integer-block coords using the
    // current depth's scale. At depth 0 one bevy unit = SUPER integer blocks,
    // at depth 1 one bevy = S, at depth 2 one bevy = 1.
    let scale = match state.depth {
        0 => SUPER as f32,
        1 => S as f32,
        _ => 1.0,
    };
    let int_x = (tf.translation.x * scale).floor() as i32;
    let int_z = (tf.translation.z * scale).floor() as i32;
    let cx_player = int_x.div_euclid(S);
    let cz_player = int_z.div_euclid(S);
    let new_column = IVec2::new(cx_player, cz_player);

    let prev = stream.last_column;
    if prev == Some(new_column) {
        return;
    }
    stream.last_column = Some(new_column);

    let r = STREAM_RADIUS_CHUNKS;
    let cy_min: i32 = 0;
    let cy_max: i32 = 5;

    for dz in -r..=r {
        for dx in -r..=r {
            let cx = new_column.x + dx;
            let cz = new_column.y + dz;

            // Delta optimization: skip cells that were already in the previous
            // window — they're either generated or tombstoned.
            if let Some(p) = prev {
                if (cx - p.x).abs() <= r && (cz - p.y).abs() <= r {
                    continue;
                }
            }

            for cy in cy_min..=cy_max {
                let key = IVec3::new(cx, cy, cz);
                if state.world.chunks.contains_key(&key) {
                    continue; // user_modified or already generated
                }
                let Some(chunk) = generate_chunk(key) else { continue };
                state.world.chunks.insert(key, chunk);
                state.dirty_super_for_chunk(key);
            }
        }
    }
}

fn render_world(
    mut commands: Commands,
    materials: Res<BlockMaterials>,
    mut state: ResMut<WorldState>,
    mut rs: ResMut<RenderState>,
    mut meshes: ResMut<Assets<Mesh>>,
    player_q: Query<&Transform, With<Player>>,
) {
    let Ok(player_tf) = player_q.single() else { return };

    if rs.rendered_depth != state.depth || rs.needs_full_refresh {
        for (_, entity) in rs.entities.drain() {
            commands.entity(entity).despawn();
        }
        rs.rendered_depth = state.depth;
        rs.needs_full_refresh = false;
    }

    match state.depth {
        0 => render_super_chunks(&mut commands, &materials, &mut state, &mut rs, &mut meshes, player_tf),
        1 => render_chunks(&mut commands, &materials, &mut state, &mut rs, &mut meshes, player_tf, 1.0 / S as f32, 1.0),
        _ => render_chunks(&mut commands, &materials, &mut state, &mut rs, &mut meshes, player_tf, 1.0, S as f32),
    }
}

fn render_super_chunks(
    commands: &mut Commands,
    materials: &BlockMaterials,
    state: &mut WorldState,
    rs: &mut RenderState,
    meshes: &mut Assets<Mesh>,
    player_tf: &Transform,
) {
    // 1 bevy unit = 1 super-chunk at depth 0. Player floor-divides into
    // super-chunk space directly.
    let player_key = IVec3::new(
        player_tf.translation.x.floor() as i32,
        player_tf.translation.y.floor() as i32,
        player_tf.translation.z.floor() as i32,
    );

    let rd = RENDER_DISTANCE;
    let mut desired: HashSet<IVec3> = HashSet::new();
    for dy in -rd..=rd {
        for dz in -rd..=rd {
            for dx in -rd..=rd {
                if dx * dx + dy * dy + dz * dz > rd * rd {
                    continue;
                }
                let key = player_key + IVec3::new(dx, dy, dz);
                if state.world.super_chunk_solid(key) {
                    desired.insert(key);
                }
            }
        }
    }

    // Mesh local size is SUPER, but each super-chunk should occupy 1 bevy
    // unit, so the entity scale is 1/SUPER.
    let view_scale = 1.0 / SUPER as f32;

    for &key in &desired {
        let dirty = state.dirty_supers.contains(&key);
        if rs.entities.contains_key(&key) && !dirty {
            continue;
        }

        if let Some(old) = rs.entities.remove(&key) {
            commands.entity(old).despawn();
        }

        let baked = bake_super_chunk(&state.world, key, meshes);
        state.dirty_supers.remove(&key);

        let pos = key.as_vec3();
        let root = commands
            .spawn((
                ChunkEntity,
                Transform::from_translation(pos).with_scale(Vec3::splat(view_scale)),
                Visibility::Inherited,
            ))
            .id();
        for sub in &baked {
            let child = commands
                .spawn((
                    Mesh3d(sub.mesh.clone()),
                    MeshMaterial3d(materials.get(sub.block_type)),
                    Transform::default(),
                ))
                .id();
            commands.entity(root).add_child(child);
        }
        rs.entities.insert(key, root);
    }

    despawn_stale(commands, rs, &desired);
}

fn render_chunks(
    commands: &mut Commands,
    materials: &BlockMaterials,
    state: &mut WorldState,
    rs: &mut RenderState,
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

    let rd = RENDER_DISTANCE;
    let mut desired: HashSet<IVec3> = HashSet::new();
    for dy in -rd..=rd {
        for dz in -rd..=rd {
            for dx in -rd..=rd {
                if dx * dx + dy * dy + dz * dz > rd * rd {
                    continue;
                }
                let key = player_key + IVec3::new(dx, dy, dz);
                if state.world.chunks.contains_key(&key) {
                    desired.insert(key);
                }
            }
        }
    }

    for &key in &desired {
        let chunk = state.world.chunks.get_mut(&key).unwrap();
        if chunk.mesh_dirty {
            chunk.baked = bake_model(&chunk.blocks, meshes);
            chunk.mesh_dirty = false;
            if let Some(old) = rs.entities.remove(&key) {
                commands.entity(old).despawn();
            }
        }
        if rs.entities.contains_key(&key) {
            continue;
        }

        let chunk = state.world.chunks.get(&key).unwrap();
        let pos = key.as_vec3() * chunk_size_w;
        let root = commands
            .spawn((
                ChunkEntity,
                Transform::from_translation(pos).with_scale(Vec3::splat(view_scale)),
                Visibility::Inherited,
            ))
            .id();
        for sub in &chunk.baked {
            let child = commands
                .spawn((
                    Mesh3d(sub.mesh.clone()),
                    MeshMaterial3d(materials.get(sub.block_type)),
                    Transform::default(),
                ))
                .id();
            commands.entity(root).add_child(child);
        }
        rs.entities.insert(key, root);
    }

    despawn_stale(commands, rs, &desired);
}

fn despawn_stale(commands: &mut Commands, rs: &mut RenderState, desired: &HashSet<IVec3>) {
    let stale: Vec<_> = rs
        .entities
        .keys()
        .filter(|k| !desired.contains(k))
        .copied()
        .collect();
    for key in stale {
        if let Some(e) = rs.entities.remove(&key) {
            commands.entity(e).despawn();
        }
    }
}
