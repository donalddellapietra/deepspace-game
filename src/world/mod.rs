pub mod collision;

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::layer::{ActiveLayer, NavEntry};
use crate::model::mesher::bake_model;
use crate::model::BakedSubMesh;
use crate::player::Player;

const CELL_SCALE: f32 = 1.0 / MODEL_SIZE as f32;
const RENDER_DISTANCE: i32 = 16;
const NEIGHBOR_RANGE: i32 = 4;

// ============================================================
// Data model
// ============================================================

#[derive(Clone)]
pub enum CellSlot {
    Empty,
    Block(BlockType),
    Child(Box<VoxelGrid>),
}

impl Default for CellSlot {
    fn default() -> Self { Self::Empty }
}

impl CellSlot {
    pub fn is_solid(&self) -> bool { !matches!(self, Self::Empty) }

    /// Representative block type for baking/display.
    pub fn block_type(&self) -> Option<BlockType> {
        match self {
            Self::Block(bt) => Some(*bt),
            Self::Child(child) => most_common_block(child),
            Self::Empty => None,
        }
    }
}

#[derive(Clone)]
pub struct VoxelGrid {
    pub slots: [[[CellSlot; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
    pub baked: Vec<BakedSubMesh>,
}

impl VoxelGrid {
    pub fn new_empty() -> Self {
        Self {
            slots: std::array::from_fn(|_| std::array::from_fn(|_| std::array::from_fn(|_| CellSlot::Empty))),
            baked: vec![],
        }
    }

    pub fn from_blocks(blocks: &[[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE], meshes: &mut Assets<Mesh>) -> Self {
        let mut grid = Self::new_empty();
        for y in 0..MODEL_SIZE {
            for z in 0..MODEL_SIZE {
                for x in 0..MODEL_SIZE {
                    if let Some(bt) = blocks[y][z][x] {
                        grid.slots[y][z][x] = CellSlot::Block(bt);
                    }
                }
            }
        }
        grid.rebake(meshes);
        grid
    }

    pub fn to_block_array(&self) -> [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE] {
        let mut arr = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
        for y in 0..MODEL_SIZE {
            for z in 0..MODEL_SIZE {
                for x in 0..MODEL_SIZE {
                    arr[y][z][x] = self.slots[y][z][x].block_type();
                }
            }
        }
        arr
    }

    pub fn rebake(&mut self, meshes: &mut Assets<Mesh>) {
        self.baked = bake_model(&self.to_block_array(), meshes);
    }

    pub fn column_top(&self, x: usize, z: usize) -> f32 {
        for y in (0..MODEL_SIZE).rev() {
            if self.slots[y][z][x].is_solid() { return (y + 1) as f32; }
        }
        0.0
    }

    /// Count solid slots.
    pub fn solid_count(&self) -> usize {
        self.slots.iter().flatten().flatten().filter(|s| s.is_solid()).count()
    }
}

fn most_common_block(grid: &VoxelGrid) -> Option<BlockType> {
    let mut counts = [0u32; 10];
    for slot in grid.slots.iter().flatten().flatten() {
        if let CellSlot::Block(bt) = slot { counts[*bt as usize] += 1; }
    }
    counts.iter().enumerate().max_by_key(|(_, c)| **c)
        .filter(|(_, c)| **c > 0).map(|(i, _)| BlockType::ALL[i])
}

// ============================================================
// World resource
// ============================================================

#[derive(Resource, Default)]
pub struct VoxelWorld {
    pub cells: HashMap<IVec3, VoxelGrid>,
}

impl VoxelWorld {
    /// Get the grid the player is currently inside (follow the nav stack).
    pub fn get_grid(&self, nav_stack: &[NavEntry]) -> Option<&VoxelGrid> {
        if nav_stack.is_empty() { return None; }
        let mut cur = self.cells.get(&nav_stack[0].cell_coord)?;
        for entry in &nav_stack[1..] {
            let c = entry.cell_coord;
            cur = match &cur.slots[c.y as usize][c.z as usize][c.x as usize] {
                CellSlot::Child(child) => child,
                _ => { warn!("get_grid: expected Child at {:?}, got non-Child", c); return None; }
            };
        }
        Some(cur)
    }

    pub fn get_grid_mut(&mut self, nav_stack: &[NavEntry]) -> Option<&mut VoxelGrid> {
        if nav_stack.is_empty() { return None; }
        let mut cur = self.cells.get_mut(&nav_stack[0].cell_coord)?;
        for entry in &nav_stack[1..] {
            let c = entry.cell_coord;
            cur = match &mut cur.slots[c.y as usize][c.z as usize][c.x as usize] {
                CellSlot::Child(child) => child,
                _ => { warn!("get_grid_mut: expected Child at {:?}", c); return None; }
            };
        }
        Some(cur)
    }

    /// Get a sibling grid in the parent layer. Works at any depth.
    /// `nav_stack` is the current full stack, `sibling_coord` is an absolute coord in the parent.
    /// At depth 1: parent is `self.cells` (unbounded HashMap).
    /// At depth 2+: parent is the grid at `nav_stack[..len-1]` (bounded 0..MODEL_SIZE).
    pub fn get_sibling(&self, nav_stack: &[NavEntry], sibling_coord: IVec3) -> Option<&VoxelGrid> {
        if nav_stack.is_empty() { return None; }
        let s = MODEL_SIZE as i32;

        if nav_stack.len() == 1 {
            // Parent is the top-layer HashMap (unbounded)
            self.cells.get(&sibling_coord)
        } else {
            // Parent is a grid (bounded)
            if sibling_coord.x < 0 || sibling_coord.x >= s
                || sibling_coord.y < 0 || sibling_coord.y >= s
                || sibling_coord.z < 0 || sibling_coord.z >= s { return None; }
            let parent = self.get_grid(&nav_stack[..nav_stack.len() - 1])?;
            match &parent.slots[sibling_coord.y as usize][sibling_coord.z as usize][sibling_coord.x as usize] {
                CellSlot::Child(child) => Some(child),
                _ => None,
            }
        }
    }

    /// Get the CellSlot of a sibling in the parent. Works at any depth.
    pub fn get_sibling_slot(&self, nav_stack: &[NavEntry], sibling_coord: IVec3) -> Option<&CellSlot> {
        if nav_stack.is_empty() { return None; }
        let s = MODEL_SIZE as i32;

        if nav_stack.len() == 1 {
            // Top-layer cells are always "solid" (they exist or don't)
            self.cells.get(&sibling_coord).map(|_| &CellSlot::Block(BlockType::Grass) as &CellSlot)
        } else {
            if sibling_coord.x < 0 || sibling_coord.x >= s
                || sibling_coord.y < 0 || sibling_coord.y >= s
                || sibling_coord.z < 0 || sibling_coord.z >= s { return None; }
            let parent = self.get_grid(&nav_stack[..nav_stack.len() - 1])?;
            Some(&parent.slots[sibling_coord.y as usize][sibling_coord.z as usize][sibling_coord.x as usize])
        }
    }
}

// ============================================================
// Rendering
// ============================================================

#[derive(Component)]
pub struct LayerEntity;

#[derive(Resource, Default)]
pub struct RenderState {
    pub entities: HashMap<IVec3, Entity>,
    pub rendered_depth: usize,
    pub needs_refresh: bool,
}

#[derive(Resource)]
pub struct SharedCubeMesh(pub Handle<Mesh>);

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VoxelWorld>()
            .init_resource::<RenderState>()
            .add_systems(Startup, setup_world)
            .add_systems(Update, render_layer);
    }
}

fn setup_world(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut world: ResMut<VoxelWorld>) {
    commands.insert_resource(SharedCubeMesh(meshes.add(Cuboid::new(1.0, 1.0, 1.0))));

    let mut ground = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for z in 0..MODEL_SIZE {
        for x in 0..MODEL_SIZE {
            ground[0][z][x] = Some(BlockType::Stone);
            ground[1][z][x] = Some(BlockType::Stone);
            ground[2][z][x] = Some(BlockType::Dirt);
            ground[3][z][x] = Some(BlockType::Dirt);
            ground[4][z][x] = Some(BlockType::Grass);
        }
    }

    let extent = 16;
    for z in -extent..extent {
        for x in -extent..extent {
            world.cells.insert(IVec3::new(x, 0, z), VoxelGrid::from_blocks(&ground, &mut meshes));
        }
    }
}

fn render_layer(
    mut commands: Commands,
    materials: Res<BlockMaterials>,
    world: Res<VoxelWorld>,
    active: Res<ActiveLayer>,
    cube: Option<Res<SharedCubeMesh>>,
    mut rs: ResMut<RenderState>,
    player_q: Query<&Transform, With<Player>>,
) {
    let Ok(player_tf) = player_q.single() else { return };
    let Some(cube) = cube else { return };

    let depth = active.nav_stack.len();
    if rs.rendered_depth != depth || rs.needs_refresh {
        for (_, entity) in rs.entities.drain() {
            commands.entity(entity).despawn();
        }
        rs.rendered_depth = depth;
        rs.needs_refresh = false;
    }

    if active.is_top_layer() {
        render_top(&mut commands, &materials, &world, &mut rs, player_tf.translation);
    } else {
        if rs.entities.is_empty() {
            render_inside(&mut commands, &materials, &world, &active, &cube, &mut rs);
        }
    }
}

/// Top layer: each cell = 1 world unit.
fn render_top(commands: &mut Commands, materials: &BlockMaterials, world: &VoxelWorld,
    rs: &mut RenderState, pos: Vec3,
) {
    let px = pos.x.floor() as i32;
    let pz = pos.z.floor() as i32;
    let mut desired = HashSet::new();
    let rd = RENDER_DISTANCE;
    for dz in -rd..=rd {
        for dx in -rd..=rd {
            if dx * dx + dz * dz > rd * rd { continue; }
            for y in -2..10 {
                let c = IVec3::new(px + dx, y, pz + dz);
                if world.cells.contains_key(&c) { desired.insert(c); }
            }
        }
    }

    for &coord in &desired {
        if rs.entities.contains_key(&coord) { continue; }
        let Some(grid) = world.cells.get(&coord) else { continue };
        let e = spawn_baked(commands, materials, &grid.baked, coord.as_vec3(), Vec3::splat(CELL_SCALE));
        rs.entities.insert(coord, e);
    }

    let stale: Vec<_> = rs.entities.keys().filter(|c| !desired.contains(c)).copied().collect();
    for c in stale {
        if let Some(e) = rs.entities.remove(&c) { commands.entity(e).despawn(); }
    }
}

/// Inside a grid at any depth. Renders:
/// 1. Current cell's individual blocks/children
/// 2. Neighboring parent-layer cells (GENERIC — works at any depth)
fn render_inside(
    commands: &mut Commands, materials: &BlockMaterials, world: &VoxelWorld,
    active: &ActiveLayer, cube: &SharedCubeMesh, rs: &mut RenderState,
) {
    let Some(grid) = world.get_grid(&active.nav_stack) else {
        warn!("render_inside: get_grid returned None at depth {}", active.nav_stack.len());
        return;
    };

    info!("render_inside: depth={}, solid_blocks={}", active.nav_stack.len(), grid.solid_count());

    let s = MODEL_SIZE as i32;

    // 1. Current cell's contents
    for y in 0..MODEL_SIZE {
        for z in 0..MODEL_SIZE {
            for x in 0..MODEL_SIZE {
                let coord = IVec3::new(x as i32, y as i32, z as i32);
                match &grid.slots[y][z][x] {
                    CellSlot::Empty => {}
                    CellSlot::Block(bt) => {
                        let e = commands.spawn((
                            LayerEntity, Mesh3d(cube.0.clone()), MeshMaterial3d(materials.get(*bt)),
                            Transform::from_translation(coord.as_vec3() + Vec3::splat(0.5)),
                        )).id();
                        rs.entities.insert(coord, e);
                    }
                    CellSlot::Child(child) => {
                        let e = spawn_baked(commands, materials, &child.baked,
                            coord.as_vec3(), Vec3::splat(CELL_SCALE));
                        rs.entities.insert(coord, e);
                    }
                }
            }
        }
    }

    // 2. Render the ENTIRE world at every ancestor layer.
    //    Walk up the nav stack. At each level, render all cells/slots in that layer
    //    EXCEPT the one we drilled into (that's already shown as expanded content).
    //    Each ancestor level is offset and scaled relative to the current coordinate space.
    //
    //    offset accumulates: each ancestor's content is shifted by the cell_coord * MODEL_SIZE
    //    at that level, and scaled by 1/MODEL_SIZE per level below.

    render_ancestors(commands, materials, world, &active.nav_stack, cube, rs);
}

/// Render all ancestor layers' content around the current cell.
/// This makes the full world visible at every depth.
fn render_ancestors(
    commands: &mut Commands, materials: &BlockMaterials, world: &VoxelWorld,
    nav_stack: &[NavEntry], cube: &SharedCubeMesh, rs: &mut RenderState,
) {
    let s = MODEL_SIZE as i32;
    let sf = MODEL_SIZE as f32;

    // Walk the stack from deepest to shallowest.
    // At each level, we compute the offset from the current coordinate space
    // to that ancestor's coordinate space.
    //
    // cumulative_scale tracks how many current-blocks one ancestor slot spans.
    // It is multiplied by MODEL_SIZE at the TOP of each iteration, so:
    //   level 0 (immediate parent): cumulative_scale = MODEL_SIZE
    //   level 1 (grandparent):      cumulative_scale = MODEL_SIZE^2
    //   etc.
    //
    // cumulative_offset is updated by subtracting skip_coord * cumulative_scale
    // at each level, so that the ancestor's coordinate origin aligns correctly
    // with the current block-space.
    //
    // Baked meshes have vertices in 0..MODEL_SIZE, so mesh_scale = cumulative_scale / MODEL_SIZE
    // to make them the right size in current block-space.
    // Individual blocks (cubes) are placed with scale = cumulative_scale.

    let depth = nav_stack.len();
    let mut cumulative_offset = Vec3::ZERO;
    let mut cumulative_scale = 1.0f32; // how many current-blocks one ancestor slot spans

    for level in 0..depth {
        let ancestor_idx = depth - 1 - level; // walk from immediate parent up
        let entry = &nav_stack[ancestor_idx];
        let skip_coord = entry.cell_coord;

        // Scale up FIRST: each slot at this ancestor level spans MODEL_SIZE
        // times as many current-blocks as the level below it.
        // At level 0 (immediate parent): each slot = MODEL_SIZE current blocks.
        // At level 1 (grandparent):      each slot = MODEL_SIZE^2 current blocks.
        cumulative_scale *= sf;

        // The cell we drilled into at this level is at skip_coord.
        // Its origin in current block-space is at cumulative_offset.
        // Each slot at this level spans cumulative_scale blocks.
        // Update offset: shift so this ancestor's (0,0,0) aligns correctly.
        cumulative_offset -= skip_coord.as_vec3() * cumulative_scale;

        // Determine which cells/slots exist at this ancestor level
        let key_base = (level as i32 + 1) * 10000;

        if ancestor_idx == 0 {
            // This ancestor is the top-layer HashMap
            let range = RENDER_DISTANCE;
            let anchor = skip_coord; // the top-layer cell we entered
            for dz in -range..=range {
                for dx in -range..=range {
                    if dx * dx + dz * dz > range * range { continue; }
                    for dy in -2..10 {
                        let coord = IVec3::new(anchor.x + dx, anchor.y + dy, anchor.z + dz);
                        if coord == skip_coord { continue; } // already expanded
                        let Some(grid) = world.cells.get(&coord) else { continue };

                        let pos = cumulative_offset + coord.as_vec3() * cumulative_scale;
                        let mesh_scale = Vec3::splat(cumulative_scale / sf); // baked mesh is 0..sf, scale to fit
                        let key = IVec3::new(key_base + dx, key_base + dy, key_base + dz);
                        let e = spawn_baked(commands, materials, &grid.baked, pos, mesh_scale);
                        rs.entities.insert(key, e);
                    }
                }
            }
        } else {
            // This ancestor is a grid inside the stack
            let parent_nav = &nav_stack[..ancestor_idx];
            let Some(parent_grid) = world.get_grid(parent_nav) else { continue };

            for y in 0..MODEL_SIZE {
                for z in 0..MODEL_SIZE {
                    for x in 0..MODEL_SIZE {
                        let coord = IVec3::new(x as i32, y as i32, z as i32);
                        if coord == skip_coord { continue; }

                        let slot = &parent_grid.slots[y][z][x];
                        if !slot.is_solid() { continue; }

                        let pos = cumulative_offset + coord.as_vec3() * cumulative_scale;
                        let key = IVec3::new(key_base + x as i32, key_base + y as i32, key_base + z as i32);

                        match slot {
                            CellSlot::Child(child) => {
                                let mesh_scale = Vec3::splat(cumulative_scale / sf);
                                let e = spawn_baked(commands, materials, &child.baked, pos, mesh_scale);
                                rs.entities.insert(key, e);
                            }
                            CellSlot::Block(bt) => {
                                let e = commands.spawn((
                                    LayerEntity, Mesh3d(cube.0.clone()),
                                    MeshMaterial3d(materials.get(*bt)),
                                    Transform::from_translation(pos + Vec3::splat(cumulative_scale / 2.0))
                                        .with_scale(Vec3::splat(cumulative_scale)),
                                )).id();
                                rs.entities.insert(key, e);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // cumulative_scale was already multiplied at the top of this iteration
    }
}

/// Pure math for render_ancestors: given a nav_stack, compute the
/// cumulative_offset and cumulative_scale at each ancestor level.
///
/// Returns a Vec of `(ancestor_idx, cumulative_offset, cumulative_scale)` tuples,
/// ordered from the immediate parent (level 0) up to the root.
pub fn render_ancestor_transforms(nav_stack: &[NavEntry]) -> Vec<(usize, Vec3, f32)> {
    let sf = MODEL_SIZE as f32;
    let depth = nav_stack.len();
    let mut result = Vec::new();
    let mut cumulative_offset = Vec3::ZERO;
    let mut cumulative_scale = 1.0f32;

    for level in 0..depth {
        let ancestor_idx = depth - 1 - level;
        let skip_coord = nav_stack[ancestor_idx].cell_coord;

        cumulative_scale *= sf;
        cumulative_offset -= skip_coord.as_vec3() * cumulative_scale;

        result.push((ancestor_idx, cumulative_offset, cumulative_scale));
    }
    result
}

fn spawn_baked(commands: &mut Commands, materials: &BlockMaterials, baked: &[BakedSubMesh],
    position: Vec3, scale: Vec3) -> Entity
{
    let root = commands.spawn((
        LayerEntity, Transform::from_translation(position).with_scale(scale), Visibility::Inherited,
    )).id();
    for sub in baked {
        let child = commands.spawn((
            Mesh3d(sub.mesh.clone()), MeshMaterial3d(materials.get(sub.block_type)), Transform::default(),
        )).id();
        commands.entity(root).add_child(child);
    }
    root
}

// Collision is in world/collision.rs
