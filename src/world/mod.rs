use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::layer::{ActiveLayer, NavEntry};
use crate::model::mesher::bake_model;
use crate::model::BakedSubMesh;
use crate::player::Player;

/// At top layer, each cell appears as 1 world unit. Baked mesh (0..MODEL_SIZE) scaled by this.
const CELL_SCALE: f32 = 1.0 / MODEL_SIZE as f32;

/// How many cells to render around the player at top layer.
const RENDER_DISTANCE: i32 = 16;

/// How many neighboring parent-cells to render when drilled in.
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

    pub fn from_blocks(
        blocks: &[[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
        meshes: &mut Assets<Mesh>,
    ) -> Self {
        let mut grid = Self::new_empty();
        for y in 0..MODEL_SIZE {
            for z in 0..MODEL_SIZE {
                for x in 0..MODEL_SIZE {
                    grid.slots[y][z][x] = match blocks[y][z][x] {
                        Some(bt) => CellSlot::Block(bt),
                        None => CellSlot::Empty,
                    };
                }
            }
        }
        grid.rebake(meshes);
        grid
    }

    /// Convert to block array for the mesher. Child slots use their dominant color.
    pub fn to_block_array(&self) -> [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE] {
        let mut arr = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
        for y in 0..MODEL_SIZE {
            for z in 0..MODEL_SIZE {
                for x in 0..MODEL_SIZE {
                    arr[y][z][x] = match &self.slots[y][z][x] {
                        CellSlot::Empty => None,
                        CellSlot::Block(bt) => Some(*bt),
                        CellSlot::Child(child) => most_common_block(child),
                    };
                }
            }
        }
        arr
    }

    pub fn rebake(&mut self, meshes: &mut Assets<Mesh>) {
        let blocks = self.to_block_array();
        self.baked = bake_model(&blocks, meshes);
    }

    /// Find the highest solid block in column (x, z). Returns y+1 (the top surface).
    pub fn column_top(&self, x: usize, z: usize) -> f32 {
        for y in (0..MODEL_SIZE).rev() {
            if self.slots[y][z][x].is_solid() {
                return (y + 1) as f32;
            }
        }
        0.0
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
// World resource + navigation
// ============================================================

#[derive(Resource, Default)]
pub struct VoxelWorld {
    pub cells: HashMap<IVec3, VoxelGrid>,
}

impl VoxelWorld {
    /// Follow the nav stack to reach the grid the player is inside.
    /// Returns None if nav_stack is empty (at top layer — use cells directly).
    pub fn get_grid(&self, nav_stack: &[NavEntry]) -> Option<&VoxelGrid> {
        if nav_stack.is_empty() { return None; }
        let mut cur = self.cells.get(&nav_stack[0].cell_coord)?;
        for entry in &nav_stack[1..] {
            let c = entry.cell_coord;
            match &cur.slots[c.y as usize][c.z as usize][c.x as usize] {
                CellSlot::Child(child) => cur = child,
                _ => return None,
            }
        }
        Some(cur)
    }

    pub fn get_grid_mut(&mut self, nav_stack: &[NavEntry]) -> Option<&mut VoxelGrid> {
        if nav_stack.is_empty() { return None; }
        let mut cur = self.cells.get_mut(&nav_stack[0].cell_coord)?;
        for entry in &nav_stack[1..] {
            let c = entry.cell_coord;
            match &mut cur.slots[c.y as usize][c.z as usize][c.x as usize] {
                CellSlot::Child(child) => cur = child,
                _ => return None,
            }
        }
        Some(cur)
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

    // Ground template: full 5-block column
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

    // If layer changed, wipe all rendered entities
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

/// Top layer: sparse grid, each cell = 1 world unit.
fn render_top(
    commands: &mut Commands,
    materials: &BlockMaterials,
    world: &VoxelWorld,
    rs: &mut RenderState,
    player_pos: Vec3,
) {
    let px = player_pos.x.floor() as i32;
    let pz = player_pos.z.floor() as i32;

    let mut desired = HashSet::new();
    let rd = RENDER_DISTANCE;
    for dz in -rd..=rd {
        for dx in -rd..=rd {
            if dx * dx + dz * dz > rd * rd { continue; }
            for y in -2..10 {
                let coord = IVec3::new(px + dx, y, pz + dz);
                if world.cells.contains_key(&coord) { desired.insert(coord); }
            }
        }
    }

    for &coord in &desired {
        if rs.entities.contains_key(&coord) { continue; }
        let Some(grid) = world.cells.get(&coord) else { continue };
        let entity = spawn_baked_mesh(commands, materials, &grid.baked,
            coord.as_vec3(), Vec3::splat(CELL_SCALE));
        rs.entities.insert(coord, entity);
    }

    let to_remove: Vec<_> = rs.entities.keys().filter(|c| !desired.contains(c)).copied().collect();
    for coord in to_remove {
        if let Some(e) = rs.entities.remove(&coord) { commands.entity(e).despawn(); }
    }
}

/// Inside a grid: render current cell's blocks + neighboring cells from parent.
fn render_inside(
    commands: &mut Commands,
    materials: &BlockMaterials,
    world: &VoxelWorld,
    active: &ActiveLayer,
    cube: &SharedCubeMesh,
    rs: &mut RenderState,
) {
    let Some(grid) = world.get_grid(& active.nav_stack) else { return };
    let s = MODEL_SIZE as i32;

    // Current cell's blocks at full scale (each block = 1 unit, positions 0..MODEL_SIZE)
    for y in 0..MODEL_SIZE {
        for z in 0..MODEL_SIZE {
            for x in 0..MODEL_SIZE {
                let coord = IVec3::new(x as i32, y as i32, z as i32);
                match &grid.slots[y][z][x] {
                    CellSlot::Empty => {}
                    CellSlot::Block(bt) => {
                        let e = commands.spawn((
                            LayerEntity,
                            Mesh3d(cube.0.clone()),
                            MeshMaterial3d(materials.get(*bt)),
                            Transform::from_translation(coord.as_vec3() + Vec3::splat(0.5)),
                        )).id();
                        rs.entities.insert(coord, e);
                    }
                    CellSlot::Child(child) => {
                        let e = spawn_baked_mesh(commands, materials, &child.baked,
                            coord.as_vec3(), Vec3::splat(CELL_SCALE));
                        rs.entities.insert(coord, e);
                    }
                }
            }
        }
    }

    // Neighboring cells from the parent layer.
    // Each neighbor is offset by MODEL_SIZE in the relevant direction.
    let current_coord = active.nav_stack.last().unwrap().cell_coord;

    if active.nav_stack.len() == 1 {
        // Parent is the top-layer HashMap
        for dz in -NEIGHBOR_RANGE..=NEIGHBOR_RANGE {
            for dy in -2..4 {
                for dx in -NEIGHBOR_RANGE..=NEIGHBOR_RANGE {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let nc = current_coord + IVec3::new(dx, dy, dz);
                    let Some(ng) = world.cells.get(&nc) else { continue };
                    let offset = Vec3::new((dx * s) as f32, (dy * s) as f32, (dz * s) as f32);
                    let key = IVec3::new(1000 + dx, 1000 + dy, 1000 + dz);
                    let e = spawn_baked_mesh(commands, materials, &ng.baked, offset, Vec3::ONE);
                    rs.entities.insert(key, e);
                }
            }
        }
    } else {
        // Deeper layers: render sibling slots from the parent grid
        let parent_nav = &active.nav_stack[..active.nav_stack.len() - 1];
        if let Some(parent_grid) = world.get_grid(parent_nav) {
            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                        let sib = current_coord + IVec3::new(dx, dy, dz);
                        if sib.x < 0 || sib.x >= s || sib.y < 0 || sib.y >= s
                            || sib.z < 0 || sib.z >= s { continue; }

                        let slot = &parent_grid.slots[sib.y as usize][sib.z as usize][sib.x as usize];
                        let baked = match slot {
                            CellSlot::Child(child) => &child.baked,
                            CellSlot::Block(bt) => {
                                // Render solid block as a cube
                                let offset = Vec3::new((dx * s) as f32, (dy * s) as f32, (dz * s) as f32);
                                let key = IVec3::new(2000 + dx, 2000 + dy, 2000 + dz);
                                // Fill the neighbor area with a scaled cube of the block color
                                let e = commands.spawn((
                                    LayerEntity,
                                    Mesh3d(cube.0.clone()),
                                    MeshMaterial3d(materials.get(*bt)),
                                    Transform::from_translation(offset + Vec3::splat(s as f32 / 2.0))
                                        .with_scale(Vec3::splat(s as f32)),
                                )).id();
                                rs.entities.insert(key, e);
                                continue;
                            }
                            CellSlot::Empty => continue,
                        };

                        let offset = Vec3::new((dx * s) as f32, (dy * s) as f32, (dz * s) as f32);
                        let key = IVec3::new(2000 + dx, 2000 + dy, 2000 + dz);
                        let e = spawn_baked_mesh(commands, materials, baked, offset, Vec3::ONE);
                        rs.entities.insert(key, e);
                    }
                }
            }
        }
    }
}

/// Helper: spawn a baked mesh as a parent entity with sub-mesh children.
fn spawn_baked_mesh(
    commands: &mut Commands,
    materials: &BlockMaterials,
    baked: &[BakedSubMesh],
    position: Vec3,
    scale: Vec3,
) -> Entity {
    let root = commands.spawn((
        LayerEntity,
        Transform::from_translation(position).with_scale(scale),
        Visibility::Inherited,
    )).id();
    for sub in baked {
        let child = commands.spawn((
            Mesh3d(sub.mesh.clone()),
            MeshMaterial3d(materials.get(sub.block_type)),
            Transform::default(),
        )).id();
        commands.entity(root).add_child(child);
    }
    root
}

// ============================================================
// Collision — simple, correct, no hacks
// ============================================================

/// Top layer: each cell = 1 world unit. Find highest cell top at or below player feet.
pub fn floor_top_layer(cells: &HashMap<IVec3, VoxelGrid>, pos: Vec3) -> f32 {
    let gx = pos.x.floor() as i32;
    let gz = pos.z.floor() as i32;

    // Search from one above feet downward. This catches the case where the player
    // has fallen slightly into a cell.
    let search_top = pos.y.floor() as i32 + 1;
    for cy in (-4..=search_top).rev() {
        if cells.contains_key(&IVec3::new(gx, cy, gz)) {
            return (cy + 1) as f32; // top surface of this cell
        }
    }
    f32::NEG_INFINITY
}

/// Inside a grid: find the floor. Handles both the current cell and neighboring parent cells.
pub fn floor_inner(world: &VoxelWorld, nav_stack: &[NavEntry], pos: Vec3) -> f32 {
    if nav_stack.is_empty() { return f32::NEG_INFINITY; }

    let current = nav_stack.last().unwrap().cell_coord;
    let sf = MODEL_SIZE as f32;
    let s = MODEL_SIZE as i32;

    // Which parent-layer cell does the player's XZ map to?
    let pdx = pos.x.div_euclid(sf) as i32;
    let pdz = pos.z.div_euclid(sf) as i32;

    // Local block coords within that cell
    let lx = (pos.x.rem_euclid(sf).floor() as i32).clamp(0, s - 1) as usize;
    let lz = (pos.z.rem_euclid(sf).floor() as i32).clamp(0, s - 1) as usize;

    let mut best = f32::NEG_INFINITY;

    // Scan multiple Y levels in the parent
    for pdy in -2..4 {
        let parent_coord = current + IVec3::new(pdx, pdy, pdz);

        let grid = if pdx == 0 && pdy == 0 && pdz == 0 {
            world.get_grid(nav_stack)
        } else if nav_stack.len() == 1 {
            world.cells.get(&parent_coord)
        } else {
            None
        };

        let Some(grid) = grid else { continue; };
        let base_y = pdy as f32 * sf;

        // Scan column top-down, find highest solid block below player feet
        let search_top = pos.y.floor() as i32 + 1;
        for y in (0..MODEL_SIZE).rev() {
            if grid.slots[y][lz][lx].is_solid() {
                let top = base_y + (y + 1) as f32;
                if top <= search_top as f32 && top > best {
                    best = top;
                }
            }
        }
    }

    best
}
