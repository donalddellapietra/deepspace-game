//! GPU-instanced uniform-layer tree-walk renderer.
//!
//! Every frame, walk the content-addressed tree from the root down to
//! the emit layer (`target_layer - 1`) and group visible nodes by
//! `(NodeId, voxel)`.  Each unique group is ONE Bevy entity with a
//! custom instance buffer; the GPU draws the baked mesh N times from a
//! single draw call.  Typical worlds have ~3-4 unique NodeIds and ~2
//! sub-meshes each, so the live entity count drops from ~2 200 to ~6-10.
//!
//! The custom render pipeline follows the Bevy 0.18
//! `custom_shader_instancing` example pattern:
//!   - `ExtractComponent` copies `InstanceMaterialData` into the render world
//!   - `prepare_instance_buffers` uploads instance data to the GPU
//!   - `queue_custom` inserts draw items into the `Transparent3d` phase
//!   - `DrawMeshInstanced` issues `draw_indexed` with the instance count
//!
//! Key invariants (unchanged from the scalar renderer):
//!
//! * One Bevy unit equals one leaf voxel *in the current
//!   [`WorldAnchor`] frame*.
//! * Node origins are `leaf-coord − anchor.leaf_coord`, keeping
//!   rendered positions small regardless of absolute world location.
//! * Frustum culling is disabled (`NoFrustumCulling`) — the walk
//!   already culls by a sphere of `RADIUS_VIEW_CELLS` view-cells.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::pbr::SetMeshViewBindingArrayBindGroup;
use bevy::{
    core_pipeline::core_3d::Transparent3d,
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout},
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup,
        SetMeshViewBindGroup,
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex,
            RenderCommand, RenderCommandResult, SetItemPipeline,
            TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
        view::ExtractedView,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};

use crate::block::Palette;
use crate::model::{mesher::bake_volume, BakedSubMesh};

use super::state::WorldState;
use super::tree::{
    slot_coords, slot_index, voxel_idx, NodeId, VoxelGrid,
    BRANCH_FACTOR, CHILDREN_PER_NODE, EMPTY_NODE, EMPTY_VOXEL, MAX_LAYER,
    NODE_VOXELS_PER_AXIS,
};
use super::view::{
    cell_size_at_layer, extent_for_layer, scale_for_layer, target_layer_for,
    WorldAnchor,
};

// ------------------------------------------------------- markers

/// Marker attached to each instanced entity, carrying the `NodeId`
/// the entity is representing. Save-mode tinting looks up entities
/// by this component.
#[derive(Component)]
pub struct WorldRenderedNode(pub NodeId);

/// Marker attached to each instanced entity, remembering its
/// canonical voxel index so callers can restore the original material
/// after tinting.
#[derive(Component)]
pub struct SubMeshBlock(pub u8);

// ------------------------------------------------------- shader path

const SHADER_ASSET_PATH: &str = "shaders/instanced_block.wgsl";

// -------------------------------------------------- instance data

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct InstanceData {
    pub position: Vec3,
    pub scale: f32,
    pub color: [f32; 4],
}

#[derive(Component, Deref)]
pub struct InstanceMaterialData(pub Vec<InstanceData>);

impl ExtractComponent for InstanceMaterialData {
    type QueryData = &'static InstanceMaterialData;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        item: QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self> {
        Some(InstanceMaterialData(item.0.clone()))
    }
}

// ---------------------------------------------- custom pipeline plugin

pub struct InstancedBlockPlugin;

impl Plugin for InstancedBlockPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            ExtractComponentPlugin::<InstanceMaterialData>::default(),
        );
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems(RenderStartup, init_custom_pipeline)
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSystems::QueueMeshes),
                    prepare_instance_buffers
                        .in_set(RenderSystems::PrepareResources),
                ),
            );
    }
}

#[derive(Resource)]
struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

fn init_custom_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(CustomPipeline {
        shader: asset_server.load(SHADER_ASSET_PATH),
        mesh_pipeline: mesh_pipeline.clone(),
    });
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                // i_pos_scale: vec4<f32> at location 3
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3,
                },
                // i_color: vec4<f32> at location 4
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader =
            self.shader.clone();
        Ok(descriptor)
    }
}

// --------------------------------------------- queue / prepare

fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    material_meshes: Query<
        (Entity, &MainEntity),
        With<InstanceMaterialData>,
    >,
    mut transparent_render_phases: ResMut<
        ViewSortedRenderPhases<Transparent3d>,
    >,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let draw_custom =
        transparent_3d_draw_functions.read().id::<DrawCustom>();

    for (view, msaa) in &views {
        let Some(transparent_phase) =
            transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();

        for (entity, main_entity) in &material_meshes {
            let Some(mesh_instance) =
                render_mesh_instances.render_mesh_queue_data(*main_entity)
            else {
                continue;
            };
            let Some(mesh) =
                meshes.get(mesh_instance.mesh_asset_id)
            else {
                continue;
            };
            let key = view_key
                | MeshPipelineKey::from_primitive_topology(
                    mesh.primitive_topology(),
                );
            let pipeline = pipelines
                .specialize(
                    &pipeline_cache,
                    &custom_pipeline,
                    key,
                    &mesh.layout,
                )
                .unwrap();
            transparent_phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_custom,
                distance: rangefinder
                    .distance(&mesh_instance.center),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

#[derive(Component)]
struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
    /// Number of instances the buffer can hold without reallocation.
    capacity: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &InstanceMaterialData, Option<&InstanceBuffer>)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    for (entity, instance_data, existing_buf) in &query {
        let new_len = instance_data.len();
        let contents = bytemuck::cast_slice(instance_data.as_slice());

        if let Some(existing) = existing_buf {
            if existing.capacity >= new_len {
                // Reuse existing buffer — just upload new data in place.
                render_queue.write_buffer(&existing.buffer, 0, contents);
                // Update length only (buffer & capacity stay the same).
                commands.entity(entity).insert(InstanceBuffer {
                    buffer: existing.buffer.clone(),
                    length: new_len,
                    capacity: existing.capacity,
                });
                continue;
            }
        }

        // Need a new (larger) buffer.
        let buffer =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("instance data buffer"),
                contents,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: new_len,
            capacity: new_len,
        });
    }
}

// -------------------------------------------- draw command

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    DrawMeshInstanced,
);

struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        (meshes, render_mesh_instances, mesh_allocator): SystemParamItem<
            'w,
            '_,
            Self::Param,
        >,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_allocator = mesh_allocator.into_inner();

        let Some(mesh_instance) =
            render_mesh_instances.render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) =
            meshes.into_inner().get(mesh_instance.mesh_asset_id)
        else {
            return RenderCommandResult::Skip;
        };
        let Some(instance_buffer) = instance_buffer else {
            return RenderCommandResult::Skip;
        };
        let Some(vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id)
        else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) = mesh_allocator
                    .mesh_index_slice(&mesh_instance.mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(
                    index_buffer_slice.buffer.slice(..),
                    *index_format,
                );
                pass.draw_indexed(
                    index_buffer_slice.range.start
                        ..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..instance_buffer.length as u32,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(
                    vertex_buffer_slice.range,
                    0..instance_buffer.length as u32,
                );
            }
        }
        RenderCommandResult::Success
    }
}

// --------------------------------------------------------------- camera zoom

#[derive(Resource)]
pub struct CameraZoom {
    /// Which tree layer the camera renders. Clamped to
    /// `MIN_ZOOM..=MAX_ZOOM`.
    pub layer: u8,
}

pub const MIN_ZOOM: u8 = 2;
pub const MAX_ZOOM: u8 = MAX_LAYER;

impl Default for CameraZoom {
    fn default() -> Self {
        Self { layer: MAX_LAYER }
    }
}

impl CameraZoom {
    pub fn zoom_in(&mut self) -> bool {
        if self.layer < MAX_ZOOM {
            self.layer += 1;
            true
        } else {
            false
        }
    }
    pub fn zoom_out(&mut self) -> bool {
        if self.layer > MIN_ZOOM {
            self.layer -= 1;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------- constants

/// How far a rendered node's centre may be from the camera, measured
/// in **cells at the current view layer**.
pub const RADIUS_VIEW_CELLS: f32 = 32.0;

// ----------------------------------------------------------------- state

/// Per-frame timing data exposed to the diagnostics HUD.
#[derive(Resource, Default)]
pub struct RenderTimings {
    pub walk_us: u64,
    pub reconcile_us: u64,
    pub visit_count: usize,
    pub group_count: usize,
    /// Set by the player movement system.
    pub collision_us: u64,
    pub collision_blocks: usize,
}

/// Entity entry for one `(NodeId, voxel)` group.
struct GroupEntry {
    entity: Entity,
}

/// Render state: caches baked meshes and maps live instanced entities.
#[derive(Resource, Default)]
pub struct RenderState {
    /// Cached per-`NodeId` baked sub-meshes.
    meshes: HashMap<NodeId, Vec<BakedSubMesh>>,
    /// Live entities, keyed by `(NodeId, voxel)`.
    entities: HashMap<(NodeId, u8), GroupEntry>,
    /// Emit layer the `entities` set was built for.
    last_zoom_layer: u8,
    /// Whether we have done at least one render pass.
    initialised: bool,
    /// Reusable DFS stack for `walk()`.
    walk_stack: Vec<WalkFrame>,
    /// Reusable buffer for the target-layer visits.
    visits: Vec<Visit>,
}

/// A compact identifier for a node's position in the tree during a
/// single-frame walk.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct SmallPath {
    depth: u8,
    slots: [u8; MAX_LAYER as usize],
}

impl SmallPath {
    fn empty() -> Self {
        Self {
            depth: 0,
            slots: [0; MAX_LAYER as usize],
        }
    }

    fn push(&self, slot: u8) -> Self {
        let mut out = *self;
        out.slots[out.depth as usize] = slot;
        out.depth += 1;
        out
    }
}

// ----------------------------------------------------------- mesh caching

fn get_or_bake_mesh<'a>(
    render_state: &'a mut RenderState,
    world: &WorldState,
    node_id: NodeId,
    meshes: &mut Assets<Mesh>,
) -> &'a [BakedSubMesh] {
    render_state.get_or_bake(world, node_id, meshes)
}

impl RenderState {
    fn get_or_bake<'a>(
        &'a mut self,
        world: &WorldState,
        node_id: NodeId,
        meshes: &mut Assets<Mesh>,
    ) -> &'a [BakedSubMesh] {
        if !self.meshes.contains_key(&node_id) {
            let node = world
                .library
                .get(node_id)
                .expect("render: node missing from library");
            let baked = if let Some(children) = &node.children {
                let child_voxels: Vec<Option<VoxelGrid>> = children
                    .iter()
                    .map(|&id| {
                        if id == EMPTY_NODE {
                            None
                        } else {
                            Some(
                                world
                                    .library
                                    .get(id)
                                    .expect("render: child missing from library")
                                    .voxels
                                    .clone(),
                            )
                        }
                    })
                    .collect();
                let size = (BRANCH_FACTOR * NODE_VOXELS_PER_AXIS) as i32;
                bake_volume(
                    size,
                    move |x, y, z| {
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= size
                            || y >= size
                            || z >= size
                        {
                            return None;
                        }
                        let xu = x as usize;
                        let yu = y as usize;
                        let zu = z as usize;
                        let slot = slot_index(
                            xu / NODE_VOXELS_PER_AXIS,
                            yu / NODE_VOXELS_PER_AXIS,
                            zu / NODE_VOXELS_PER_AXIS,
                        );
                        let voxels = child_voxels[slot].as_ref()?;
                        let v = voxels[voxel_idx(
                            xu % NODE_VOXELS_PER_AXIS,
                            yu % NODE_VOXELS_PER_AXIS,
                            zu % NODE_VOXELS_PER_AXIS,
                        )];
                        if v == EMPTY_VOXEL { None } else { Some(v) }
                    },
                    meshes,
                )
            } else {
                let voxels = node.voxels.clone();
                bake_volume(
                    NODE_VOXELS_PER_AXIS as i32,
                    move |x, y, z| {
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= NODE_VOXELS_PER_AXIS as i32
                            || y >= NODE_VOXELS_PER_AXIS as i32
                            || z >= NODE_VOXELS_PER_AXIS as i32
                        {
                            return None;
                        }
                        let v = voxels[voxel_idx(
                            x as usize,
                            y as usize,
                            z as usize,
                        )];
                        if v == EMPTY_VOXEL { None } else { Some(v) }
                    },
                    meshes,
                )
            };
            self.meshes.insert(node_id, baked);
        }
        self.meshes
            .get(&node_id)
            .expect("just inserted")
            .as_slice()
    }
}

// ------------------------------------------------------------- tree walk

struct Visit {
    #[allow(dead_code)]
    path: SmallPath,
    node_id: NodeId,
    origin: Vec3,
    scale: f32,
}

struct WalkFrame {
    node_id: NodeId,
    path: SmallPath,
    origin_leaves: [i64; 3],
    depth: u8,
}

fn walk(
    world: &WorldState,
    emit_layer: u8,
    target_layer: u8,
    camera_pos: Vec3,
    radius_bevy: f32,
    anchor: &WorldAnchor,
    stack: &mut Vec<WalkFrame>,
    out: &mut Vec<Visit>,
) {
    stack.clear();
    out.clear();
    if world.root == EMPTY_NODE {
        return;
    }
    let mut child_extent_leaves: [i64; MAX_LAYER as usize + 1] =
        [0; MAX_LAYER as usize + 1];
    {
        let mut ext: i64 = super::state::world_extent_voxels();
        child_extent_leaves[0] = ext;
        for layer in 1..=(MAX_LAYER as usize) {
            ext /= 5;
            child_extent_leaves[layer] = ext;
        }
    }

    stack.push(WalkFrame {
        node_id: world.root,
        path: SmallPath::empty(),
        origin_leaves: [0; 3],
        depth: 0,
    });

    let radius_sq = radius_bevy * radius_bevy;

    while let Some(frame) = stack.pop() {
        let WalkFrame { node_id, path, origin_leaves, depth } = frame;
        let origin_bevy = Vec3::new(
            (origin_leaves[0] - anchor.leaf_coord[0]) as f32,
            (origin_leaves[1] - anchor.leaf_coord[1]) as f32,
            (origin_leaves[2] - anchor.leaf_coord[2]) as f32,
        );
        let extent = extent_for_layer(depth);
        let aabb_min = origin_bevy;
        let aabb_max = origin_bevy + Vec3::splat(extent);

        let dx = (aabb_min.x - camera_pos.x)
            .max(0.0)
            .max(camera_pos.x - aabb_max.x);
        let dy = (aabb_min.y - camera_pos.y)
            .max(0.0)
            .max(camera_pos.y - aabb_max.y);
        let dz = (aabb_min.z - camera_pos.z)
            .max(0.0)
            .max(camera_pos.z - aabb_max.z);
        let min_dist_sq = dx * dx + dy * dy + dz * dz;
        if min_dist_sq > radius_sq {
            continue;
        }

        if depth == emit_layer {
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: scale_for_layer(target_layer),
            });
            continue;
        }

        let Some(node) = world.library.get(node_id) else { continue };
        let Some(children) = node.children.as_ref() else {
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: scale_for_layer(depth),
            });
            continue;
        };

        let child_extent_i64 = child_extent_leaves[(depth + 1) as usize];
        for slot in 0..CHILDREN_PER_NODE {
            let child_id = children[slot];
            if child_id == EMPTY_NODE {
                continue;
            }
            let (sx, sy, sz) = slot_coords(slot);
            let child_origin_leaves = [
                origin_leaves[0] + (sx as i64) * child_extent_i64,
                origin_leaves[1] + (sy as i64) * child_extent_i64,
                origin_leaves[2] + (sz as i64) * child_extent_i64,
            ];
            let child_path = path.push(slot as u8);
            stack.push(WalkFrame {
                node_id: child_id,
                path: child_path,
                origin_leaves: child_origin_leaves,
                depth: depth + 1,
            });
        }
    }
}

// ------------------------------------------------------ colour helper

fn voxel_color_linear(palette: &Palette, voxel: u8) -> [f32; 4] {
    let entry = palette.get(voxel);
    match entry {
        Some(e) => {
            let lin = e.color.to_linear();
            [lin.red, lin.green, lin.blue, lin.alpha]
        }
        None => [1.0, 0.0, 1.0, 1.0], // magenta fallback
    }
}

// ----------------------------------------------------------------- system

/// Bevy system: walk the tree, reconcile instanced entities.
pub fn render_world(
    mut commands: Commands,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    camera_q: Query<&Transform, With<Camera3d>>,
    palette: Option<Res<Palette>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_state: ResMut<RenderState>,
    mut timings: ResMut<RenderTimings>,
) {
    let Some(palette) = palette else {
        return;
    };
    let Ok(camera_tf) = camera_q.single() else {
        return;
    };
    let camera_pos = camera_tf.translation;

    let target_layer = target_layer_for(zoom.layer);
    let emit_layer = target_layer.saturating_sub(1);

    let radius_bevy = RADIUS_VIEW_CELLS * cell_size_at_layer(zoom.layer);

    // If emit layer changed, or first pass, drop everything.
    if !render_state.initialised || render_state.last_zoom_layer != emit_layer
    {
        for (_, entry) in render_state.entities.drain() {
            if let Ok(mut ec) = commands.get_entity(entry.entity) {
                ec.despawn();
            }
        }
        render_state.last_zoom_layer = emit_layer;
        render_state.initialised = true;
    }

    // Walk
    let walk_start = std::time::Instant::now();
    let mut walk_stack = std::mem::take(&mut render_state.walk_stack);
    let mut visits = std::mem::take(&mut render_state.visits);
    walk(
        &world,
        emit_layer,
        target_layer,
        camera_pos,
        radius_bevy,
        &anchor,
        &mut walk_stack,
        &mut visits,
    );
    timings.walk_us = walk_start.elapsed().as_micros() as u64;
    timings.visit_count = visits.len();

    // Ensure all visited NodeIds are baked before grouping.
    for v in visits.iter() {
        if !render_state.meshes.contains_key(&v.node_id) {
            get_or_bake_mesh(&mut render_state, &world, v.node_id, &mut meshes);
        }
    }

    // Group visits by (NodeId, voxel) → Vec<(origin, scale)>.
    let mut groups: HashMap<(NodeId, u8), Vec<(Vec3, f32)>> =
        HashMap::new();
    for visit in visits.drain(..) {
        if let Some(baked) = render_state.meshes.get(&visit.node_id) {
            for sub in baked {
                groups
                    .entry((visit.node_id, sub.voxel))
                    .or_default()
                    .push((visit.origin, visit.scale));
            }
        }
    }

    // Reconcile: update or spawn one entity per group.
    let reconcile_start = std::time::Instant::now();
    timings.group_count = groups.len();
    let mut alive: HashMap<(NodeId, u8), GroupEntry> =
        HashMap::with_capacity(groups.len());

    for ((node_id, voxel), origins) in groups {
        let color = voxel_color_linear(&palette, voxel);
        let instance_data: Vec<InstanceData> = origins
            .iter()
            .map(|&(origin, scale)| InstanceData {
                position: origin,
                scale,
                color,
            })
            .collect();

        match render_state.entities.remove(&(node_id, voxel)) {
            Some(entry) => {
                // Update instance data on existing entity.
                if let Ok(mut ec) = commands.get_entity(entry.entity) {
                    ec.insert(InstanceMaterialData(instance_data));
                    alive.insert((node_id, voxel), entry);
                }
            }
            None => {
                // Look up the baked mesh handle.
                let baked = render_state
                    .meshes
                    .get(&node_id)
                    .expect("just baked");
                let mesh_handle = baked
                    .iter()
                    .find(|s| s.voxel == voxel)
                    .expect("group key came from baked submeshes")
                    .mesh
                    .clone();

                let entity = commands
                    .spawn((
                        Mesh3d(mesh_handle),
                        InstanceMaterialData(instance_data),
                        WorldRenderedNode(node_id),
                        SubMeshBlock(voxel),
                        NoFrustumCulling,
                        Transform::default(),
                        Visibility::Visible,
                    ))
                    .id();
                alive.insert(
                    (node_id, voxel),
                    GroupEntry { entity },
                );
            }
        }
    }

    // Despawn stale groups.
    for (_, entry) in render_state.entities.drain() {
        if let Ok(mut ec) = commands.get_entity(entry.entity) {
            ec.despawn();
        }
    }
    render_state.entities = alive;

    timings.reconcile_us = reconcile_start.elapsed().as_micros() as u64;

    render_state.walk_stack = walk_stack;
    render_state.visits = visits;
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_path_push() {
        let p = SmallPath::empty().push(7).push(12);
        assert_eq!(p.depth, 2);
        assert_eq!(p.slots[0], 7);
        assert_eq!(p.slots[1], 12);
    }

    fn anchor_origin() -> WorldAnchor {
        WorldAnchor { leaf_coord: [0, 0, 0] }
    }

    #[test]
    fn walk_grassland_at_leaves_emits_at_least_one_visit() {
        let world = WorldState::new_grassland();
        let mut stack = Vec::new();
        let mut visits = Vec::new();
        let target = target_layer_for(MAX_LAYER);
        let emit = target.saturating_sub(1);
        walk(
            &world,
            emit,
            target,
            Vec3::ZERO,
            RADIUS_VIEW_CELLS * cell_size_at_layer(MAX_LAYER),
            &anchor_origin(),
            &mut stack,
            &mut visits,
        );
        assert!(
            !visits.is_empty(),
            "grassland walk at leaf layer should emit at least one visit"
        );
    }

    #[test]
    fn walk_radius_limits_emit_count() {
        let world = WorldState::new_grassland();
        let mut stack = Vec::new();
        let mut visits = Vec::new();
        let target = target_layer_for(MAX_LAYER);
        let emit = target.saturating_sub(1);
        walk(
            &world,
            emit,
            target,
            Vec3::ZERO,
            RADIUS_VIEW_CELLS * cell_size_at_layer(MAX_LAYER),
            &anchor_origin(),
            &mut stack,
            &mut visits,
        );
        assert!(
            visits.len() < 200_000,
            "walk emitted {} visits; expected far fewer",
            visits.len()
        );
    }

    #[test]
    fn walk_radius_scales_with_view_layer() {
        let world = WorldState::new_grassland();
        let anchor = anchor_origin();
        let mut counts = Vec::new();
        let mut stack = Vec::new();
        for view_layer in (MIN_ZOOM..=MAX_ZOOM).rev() {
            let target = target_layer_for(view_layer);
            let emit = target.saturating_sub(1);
            let radius = RADIUS_VIEW_CELLS * cell_size_at_layer(view_layer);
            let mut visits = Vec::new();
            walk(
                &world,
                emit,
                target,
                Vec3::ZERO,
                radius,
                &anchor,
                &mut stack,
                &mut visits,
            );
            counts.push((view_layer, visits.len()));
        }
        for &(view_layer, n) in &counts {
            assert!(
                n > 0,
                "view layer {view_layer}: walk emitted 0 visits — radius too small at this zoom?"
            );
        }
    }
}
