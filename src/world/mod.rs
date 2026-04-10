pub mod collision;

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::model::mesher::{bake_model, bake_volume};
use crate::model::BakedSubMesh;
use crate::player::Player;

const S: i32 = MODEL_SIZE as i32;
/// Side length of one super-chunk in integer blocks.
pub const SUPER: i32 = S * S;

/// Number of supported zoom levels. Depth 0 = most zoomed out (super-chunks),
/// MAX_DEPTH = most zoomed in (per-block editing).
pub const MAX_DEPTH: usize = 2;

const RENDER_DISTANCE: i32 = 8;

// ============================================================
// Chunk + FlatWorld
// ============================================================

#[derive(Clone)]
pub struct Chunk {
    pub blocks: [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
    /// True iff a player edit has touched this chunk. The streaming generator
    /// must never overwrite a user_modified chunk. Removed cells are kept as
    /// empty `user_modified=true` chunks so the generator skips them too.
    pub user_modified: bool,
    /// Set to true whenever the content changes. Render will re-look-up the
    /// mesh library on the next pass.
    pub mesh_dirty: bool,
    /// Cached library entry ID. `None` = needs lookup. `Some(0)` = empty.
    pub level1_id: Option<u64>,
}

impl Chunk {
    pub fn new_empty() -> Self {
        Self {
            blocks: [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
            user_modified: false,
            mesh_dirty: true,
            level1_id: None,
        }
    }

    pub fn new_filled(bt: BlockType) -> Self {
        Self {
            blocks: [[[Some(bt); MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
            user_modified: false,
            mesh_dirty: true,
            level1_id: None,
        }
    }

    pub fn tombstone() -> Self {
        let mut c = Self::new_empty();
        c.user_modified = true;
        c
    }
}

/// The single integer-block grid. All blocks at every depth live here.
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
        chunk.level1_id = None;
    }

    pub fn is_solid(&self, coord: IVec3) -> bool {
        self.get(coord).is_some()
    }

    pub fn chunk_solid(&self, key: IVec3) -> bool {
        self.chunks
            .get(&key)
            .is_some_and(|c| c.blocks.iter().flatten().flatten().any(|b| b.is_some()))
    }

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
// Procedural generation
// ============================================================

const WORLD_SEED: u64 = 0xDEAD_BEEF_F00D_CAFE;

pub fn generate_chunk(coord: IVec3) -> Option<Chunk> {
    use BlockType::*;
    match coord.y {
        0 | 1 => Some(Chunk::new_filled(Stone)),
        2 | 3 => Some(Chunk::new_filled(Dirt)),
        4 => Some(Chunk::new_filled(Grass)),
        5 => {
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
// MeshLibrary: content-addressed mesh cache, one level per hierarchy step
// ============================================================

pub type Level1Key = [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
pub type Level2Key = [[[u64; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];

#[derive(Clone)]
pub struct LibraryEntry {
    pub id: u64,
    pub baked: Vec<BakedSubMesh>,
}

/// Content-addressed mesh cache. Entries are never evicted — for procedural
/// worlds the number of unique patterns is tiny, so this stays bounded even as
/// the world grows without limit.
///
/// * `level1` is keyed by a chunk's 5×5×5 block pattern.
/// * `level2` is keyed by the 5×5×5 layout of its children's level-1 IDs, so
///    two super-chunks containing the same 125 chunks (in the same layout)
///    share one baked mesh.
#[derive(Resource)]
pub struct MeshLibrary {
    pub level1: HashMap<Level1Key, LibraryEntry>,
    pub level2: HashMap<Level2Key, LibraryEntry>,
    pub next_id: u64,
}

impl Default for MeshLibrary {
    fn default() -> Self {
        Self {
            level1: HashMap::new(),
            level2: HashMap::new(),
            // 0 is reserved for "empty" (no mesh to render).
            next_id: 1,
        }
    }
}

pub const EMPTY_ID: u64 = 0;

/// Resolve a chunk's level-1 library ID, baking its mesh on a cache miss.
/// Clears `mesh_dirty` and stores the id back on the chunk as a cache.
fn ensure_level1_id(
    chunk: &mut Chunk,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
) -> u64 {
    if let Some(id) = chunk.level1_id {
        if !chunk.mesh_dirty {
            return id;
        }
    }

    // Empty chunk short-circuit.
    if chunk.blocks.iter().flatten().flatten().all(|b| b.is_none()) {
        chunk.level1_id = Some(EMPTY_ID);
        chunk.mesh_dirty = false;
        return EMPTY_ID;
    }

    // Library hit?
    if let Some(entry) = library.level1.get(&chunk.blocks) {
        chunk.level1_id = Some(entry.id);
        chunk.mesh_dirty = false;
        return entry.id;
    }

    // Miss: bake + insert.
    let id = library.next_id;
    library.next_id += 1;
    let baked = bake_model(&chunk.blocks, meshes);
    library.level1.insert(chunk.blocks, LibraryEntry { id, baked });
    chunk.level1_id = Some(id);
    chunk.mesh_dirty = false;
    id
}

/// Walk the 125 children of a super-chunk, ensuring each has a fresh level-1
/// id, and return the resulting level-2 key.
fn compute_level2_key(
    world: &mut FlatWorld,
    super_key: IVec3,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
) -> Level2Key {
    let origin = super_key * S;
    let mut child_ids: Level2Key = [[[EMPTY_ID; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for cz in 0..MODEL_SIZE {
        for cy in 0..MODEL_SIZE {
            for cx in 0..MODEL_SIZE {
                let chunk_key = origin + IVec3::new(cx as i32, cy as i32, cz as i32);
                if let Some(chunk) = world.chunks.get_mut(&chunk_key) {
                    child_ids[cz][cy][cx] = ensure_level1_id(chunk, library, meshes);
                }
            }
        }
    }
    child_ids
}

/// Bake a super-chunk's 25×25×25 integer grid into per-block-type sub-meshes.
/// Pre-fetches the 125 chunks into a local array so the inner closure is
/// array indexing instead of a HashMap lookup per cell.
fn bake_super_chunk(
    world: &FlatWorld,
    super_key: IVec3,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let origin = super_key * S;
    let mut chunks: [[[Option<&Chunk>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE] =
        [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for cz in 0..MODEL_SIZE {
        for cy in 0..MODEL_SIZE {
            for cx in 0..MODEL_SIZE {
                let key = origin + IVec3::new(cx as i32, cy as i32, cz as i32);
                chunks[cz][cy][cx] = world.chunks.get(&key);
            }
        }
    }

    bake_volume(
        SUPER,
        |x, y, z| {
            if x < 0 || y < 0 || z < 0 || x >= SUPER || y >= SUPER || z >= SUPER {
                return None;
            }
            let cx = (x / S) as usize;
            let cy = (y / S) as usize;
            let cz = (z / S) as usize;
            let lx = (x % S) as usize;
            let ly = (y % S) as usize;
            let lz = (z % S) as usize;
            chunks[cz][cy][cx].and_then(|c| c.blocks[ly][lz][lx])
        },
        meshes,
    )
}

// ============================================================
// World state + drill (single global world, depth is only a view state)
// ============================================================

#[derive(Resource, Default)]
pub struct WorldState {
    pub world: FlatWorld,
    pub depth: usize,
    /// Super-chunks whose cached level-2 entry may be stale. Populated by
    /// editor writes; consumed by `render_super_chunks`.
    pub dirty_supers: HashSet<IVec3>,
}

impl collision::SolidQuery for WorldState {
    fn is_solid(&self, coord: IVec3) -> bool {
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

    pub fn drill_in(&mut self, player_pos: Vec3) -> Option<Vec3> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        Some(player_pos * S as f32)
    }

    pub fn drill_out(&mut self, player_pos: Vec3) -> Option<Vec3> {
        if self.depth == 0 {
            return None;
        }
        self.depth -= 1;
        Some(player_pos / S as f32)
    }

    pub fn dirty_super_for_block(&mut self, block_coord: IVec3) {
        self.dirty_supers.insert(IVec3::new(
            block_coord.x.div_euclid(SUPER),
            block_coord.y.div_euclid(SUPER),
            block_coord.z.div_euclid(SUPER),
        ));
    }

    pub fn dirty_super_for_chunk(&mut self, chunk_key: IVec3) {
        self.dirty_supers.insert(IVec3::new(
            chunk_key.x.div_euclid(S),
            chunk_key.y.div_euclid(S),
            chunk_key.z.div_euclid(S),
        ));
    }
}

// ============================================================
// Rendering
// ============================================================

#[derive(Component)]
pub struct ChunkEntity;

#[derive(Resource, Default)]
pub struct RenderState {
    /// (entity, library id currently shown) per render key.
    pub entities: HashMap<IVec3, (Entity, u64)>,
    pub rendered_depth: usize,
    pub needs_full_refresh: bool,
}

#[derive(Resource, Default)]
pub struct StreamState {
    pub last_column: Option<IVec2>,
}

const STREAM_RADIUS_CHUNKS: i32 = RENDER_DISTANCE * S + S;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldState>()
            .init_resource::<RenderState>()
            .init_resource::<MeshLibrary>()
            .init_resource::<StreamState>()
            .add_systems(PreUpdate, generate_terrain)
            .add_systems(Update, render_world);
    }
}

fn generate_terrain(
    mut stream: ResMut<StreamState>,
    mut state: ResMut<WorldState>,
    player_q: Query<&Transform, With<Player>>,
) {
    let Ok(tf) = player_q.single() else { return };

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

    for dz in -r..=r {
        for dx in -r..=r {
            let cx = new_column.x + dx;
            let cz = new_column.y + dz;

            if let Some(p) = prev {
                if (cx - p.x).abs() <= r && (cz - p.y).abs() <= r {
                    continue;
                }
            }

            for cy in 0..=5 {
                let key = IVec3::new(cx, cy, cz);
                if state.world.chunks.contains_key(&key) {
                    continue;
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
    mut library: ResMut<MeshLibrary>,
    mut meshes: ResMut<Assets<Mesh>>,
    player_q: Query<&Transform, With<Player>>,
) {
    let Ok(player_tf) = player_q.single() else { return };

    if rs.rendered_depth != state.depth || rs.needs_full_refresh {
        for (_, (entity, _)) in rs.entities.drain() {
            commands.entity(entity).despawn();
        }
        rs.rendered_depth = state.depth;
        rs.needs_full_refresh = false;
    }

    match state.depth {
        0 => render_super_chunks(
            &mut commands,
            &materials,
            &mut state,
            &mut rs,
            &mut library,
            &mut meshes,
            player_tf,
        ),
        1 => render_chunks(
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
        // Resolve the current library id (may bake on miss).
        let level1_id = match state.world.chunks.get_mut(&key) {
            Some(chunk) => ensure_level1_id(chunk, library, meshes),
            None => continue,
        };

        // Freshness check.
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

        // Library lookup for spawning. Chunk content is Copy so this is cheap.
        let content = state.world.chunks.get(&key).unwrap().blocks;
        let entry = library.level1.get(&content).expect("ensured above");

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

    despawn_stale(commands, rs, &desired);
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

    let view_scale = 1.0 / SUPER as f32;

    for &super_key in &desired {
        let is_dirty = state.dirty_supers.contains(&super_key);
        let already_rendered = rs.entities.contains_key(&super_key);

        // Optimization: already-rendered and not dirty → nothing to do.
        if already_rendered && !is_dirty {
            continue;
        }

        // Compute the key (also ensures each child has a level-1 id).
        let level2_key = compute_level2_key(&mut state.world, super_key, library, meshes);
        state.dirty_supers.remove(&super_key);

        // Resolve level-2 id: cache hit, bake miss, or all-empty short-circuit.
        let all_empty = level2_key
            .iter()
            .flatten()
            .flatten()
            .all(|&id| id == EMPTY_ID);
        let level2_id = if all_empty {
            EMPTY_ID
        } else if let Some(entry) = library.level2.get(&level2_key) {
            entry.id
        } else {
            let id = library.next_id;
            library.next_id += 1;
            let baked = bake_super_chunk(&state.world, super_key, meshes);
            library.level2.insert(level2_key, LibraryEntry { id, baked });
            id
        };

        // Freshness check.
        if let Some(&(entity, existing_id)) = rs.entities.get(&super_key) {
            if existing_id == level2_id {
                continue;
            }
            commands.entity(entity).despawn();
            rs.entities.remove(&super_key);
        }

        if level2_id == EMPTY_ID {
            continue;
        }

        let entry = library.level2.get(&level2_key).expect("ensured above");
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
        if let Some((e, _)) = rs.entities.remove(&key) {
            commands.entity(e).despawn();
        }
    }
}
