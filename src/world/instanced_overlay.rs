//! GPU-instanced overlay rendering for NPCs.
//!
//! Instead of spawning one Bevy entity per NPC body part per sub-mesh
//! (which creates thousands of entities and is the main CPU bottleneck),
//! this module groups all instances of each unique mesh into a single
//! entity with an instance buffer. One draw call per mesh type.
//!
//! Based on Bevy's `custom_shader_instancing` example.

use std::sync::OnceLock;

use bevy::{
    camera::visibility::NoFrustumCulling,
    core_pipeline::core_3d::Transparent3d,
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout},
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup,
        SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup,
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::{ExtractedView, NoIndirectDrawing},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::Shader,
};
use bytemuck::{Pod, Zeroable};

use crate::block::Palette;
use crate::model::mesher::bake_volume;
use crate::model::BakedSubMesh;

use super::overlay::OverlayList;
use super::state::WorldState;
use super::tree::{voxel_idx, NodeId, EMPTY_VOXEL, NODE_VOXELS_PER_AXIS};

// --------------------------------------------------------------- data

/// Per-instance data sent to the GPU.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct NpcInstanceData {
    /// Column-major 4x4 transform (position + rotation + scale).
    pub transform: [f32; 16],
    /// RGBA base color.
    pub color: [f32; 4],
}

/// Component holding instance data for one mesh group.
/// Extracted to the render world each frame.
#[derive(Component, Deref)]
pub struct NpcInstanceList(pub Vec<NpcInstanceData>);

impl ExtractComponent for NpcInstanceList {
    type QueryData = &'static NpcInstanceList;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self> {
        Some(NpcInstanceList(item.0.clone()))
    }
}

// --------------------------------------------------------------- state

/// Key for grouping instances: one entity per unique mesh.
#[derive(Clone, PartialEq, Eq, Hash)]
struct MeshGroupKey {
    node_id: NodeId,
    voxel: u8,
    sub_idx: usize,
}

/// Persistent state for the instanced overlay system.
#[derive(Default)]
pub struct InstancedOverlayState {
    /// Baked sub-meshes per NodeId.
    meshes: HashMap<NodeId, Vec<BakedSubMesh>>,
    /// Pivot per NodeId.
    pivots: HashMap<NodeId, Vec3>,
    /// Palette color cache: voxel -> linear RGBA. Avoids per-instance lookup.
    color_cache: HashMap<u8, [f32; 4]>,
    /// One Bevy entity per mesh group (reused across frames).
    group_entities: HashMap<MeshGroupKey, Entity>,
    /// Instance data per group, rebuilt each frame.
    group_instances: HashMap<MeshGroupKey, Vec<NpcInstanceData>>,
}

// --------------------------------------------------------------- reconcile

/// Build instance data from the overlay list, then update the
/// instanced entities. Called from `render_world` instead of
/// `reconcile_overlays`.
pub fn reconcile_instanced(
    commands: &mut Commands,
    world: &WorldState,
    palette: &Palette,
    meshes: &mut Assets<Mesh>,
    overlay_list: &OverlayList,
    state: &mut InstancedOverlayState,
) {
    // Pre-bake any new overlay leaf nodes.
    for entry in &overlay_list.entries {
        for part in &entry.parts {
            if !state.meshes.contains_key(&part.node_id) {
                let node = world
                    .library
                    .get(part.node_id)
                    .expect("overlay: node missing from library");
                let voxels = node.voxels.clone();
                let baked = bake_volume(
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
                        let v = voxels[voxel_idx(x as usize, y as usize, z as usize)];
                        if v == EMPTY_VOXEL { None } else { Some(v) }
                    },
                    meshes,
                );
                state.meshes.insert(part.node_id, baked);
            }
            state.pivots.entry(part.node_id).or_insert(part.pivot);
        }
    }

    // Clear instance lists (reuse allocations).
    for instances in state.group_instances.values_mut() {
        instances.clear();
    }

    // Cache palette colors (only ~10 entries, done once).
    if state.color_cache.is_empty() {
        for voxel in 1..=255u8 {
            if let Some(entry) = palette.get(voxel) {
                let c = entry.color.to_linear();
                state.color_cache.insert(voxel, [c.red, c.green, c.blue, 1.0]);
            }
        }
    }

    // Collect instances grouped by (mesh, material).
    for entry in &overlay_list.entries {
        for part in &entry.parts {
            let baked = match state.meshes.get(&part.node_id) {
                Some(b) => b,
                None => continue,
            };
            let pivot = state.pivots.get(&part.node_id).copied().unwrap_or(Vec3::ZERO);

            let part_origin =
                entry.bevy_pos + entry.rotation * (entry.scale * part.offset);
            let part_rotation = entry.rotation * part.rotation;

            // Combined transform: translate + rotate + scale, then pivot offset.
            let tf = Transform::from_translation(part_origin)
                .with_scale(Vec3::splat(entry.scale))
                .with_rotation(part_rotation);
            let pivot_tf = Transform::from_translation(-pivot);
            let combined = tf.to_matrix() * pivot_tf.to_matrix();

            for (sub_idx, sub) in baked.iter().enumerate() {
                let key = MeshGroupKey {
                    node_id: part.node_id,
                    voxel: sub.voxel,
                    sub_idx,
                };

                let color = state.color_cache
                    .get(&sub.voxel)
                    .copied()
                    .unwrap_or([1.0, 0.0, 1.0, 1.0]);

                state
                    .group_instances
                    .entry(key)
                    .or_default()
                    .push(NpcInstanceData {
                        transform: combined.to_cols_array(),
                        color,
                    });
            }
        }
    }

    // Collect keys to process (can't mutate group_entities while iterating group_instances).
    let keys: Vec<MeshGroupKey> = state.group_instances.keys().cloned().collect();

    for key in &keys {
        let instances = state.group_instances.get_mut(key).unwrap();

        if instances.is_empty() {
            if let Some(ent) = state.group_entities.remove(key) {
                if let Ok(mut ec) = commands.get_entity(ent) {
                    ec.despawn();
                }
            }
            continue;
        }

        let baked = match state.meshes.get(&key.node_id) {
            Some(b) => b,
            None => continue,
        };
        let sub = match baked.get(key.sub_idx) {
            Some(s) => s,
            None => continue,
        };

        let data = std::mem::take(instances);

        match state.group_entities.get(key) {
            Some(&ent) => {
                if let Ok(mut ec) = commands.get_entity(ent) {
                    ec.insert(NpcInstanceList(data));
                }
            }
            None => {
                let ent = commands
                    .spawn((
                        Mesh3d(sub.mesh.clone()),
                        NpcInstanceList(data),
                        Transform::IDENTITY,
                        Visibility::Visible,
                        NoFrustumCulling,
                        NoIndirectDrawing,
                    ))
                    .id();
                state.group_entities.insert(key.clone(), ent);
            }
        }
    }

    // Despawn groups not seen this frame (key not in group_instances).
    state.group_entities.retain(|key, ent| {
        if !state.group_instances.contains_key(key) {
            if let Ok(mut ec) = commands.get_entity(*ent) {
                ec.despawn();
            }
            return false;
        }
        true
    });
}

/// Clear all instanced overlay entities.
pub fn clear_instanced(commands: &mut Commands, state: &mut InstancedOverlayState) {
    for (_, ent) in state.group_entities.drain() {
        if let Ok(mut ec) = commands.get_entity(ent) {
            ec.despawn();
        }
    }
}

// --------------------------------------------------------------- render plugin

/// Global handle for the embedded NPC instancing shader.
static NPC_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

pub struct NpcInstancePlugin;

impl Plugin for NpcInstancePlugin {
    fn build(&self, app: &mut App) {
        // Embed the shader at compile time.
        let handle = {
            let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
            shaders.add(Shader::from_wgsl(
                include_str!("../../assets/shaders/npc_instanced.wgsl"),
                "npc_instanced.wgsl",
            ))
        };
        NPC_SHADER.set(handle).ok();

        app.add_plugins(ExtractComponentPlugin::<NpcInstanceList>::default());
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawNpcInstanced>()
            .init_resource::<SpecializedMeshPipelines<NpcPipeline>>()
            .add_systems(RenderStartup, init_npc_pipeline)
            .add_systems(
                Render,
                (
                    queue_npc_instances.in_set(RenderSystems::QueueMeshes),
                    prepare_npc_instance_buffers.in_set(RenderSystems::PrepareResources),
                ),
            );
    }
}

// --------------------------------------------------------------- pipeline

#[derive(Resource)]
struct NpcPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

fn init_npc_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
) {
    let shader = NPC_SHADER
        .get()
        .cloned()
        .expect("NPC shader not loaded");
    commands.insert_resource(NpcPipeline {
        shader,
        mesh_pipeline: mesh_pipeline.clone(),
    });
}

impl SpecializedMeshPipeline for NpcPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        descriptor.vertex.shader = self.shader.clone();

        // Add instance buffer layout: 4x4 matrix (4 x vec4) + color (vec4).
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: size_of::<NpcInstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                // Transform column 0
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 10,
                },
                // Transform column 1
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 11,
                },
                // Transform column 2
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 12,
                },
                // Transform column 3
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 13,
                },
                // Color
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 64,
                    shader_location: 14,
                },
            ],
        });

        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        Ok(descriptor)
    }
}

// --------------------------------------------------------------- queue

fn queue_npc_instances(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<NpcPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<NpcPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    material_meshes: Query<(Entity, &MainEntity), With<NpcInstanceList>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawNpcInstanced>();

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
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let key = view_key
                | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());
            let pipeline = pipelines
                .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                .unwrap();
            transparent_phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_custom,
                distance: rangefinder.distance(&mesh_instance.center),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

// --------------------------------------------------------------- prepare

#[derive(Component)]
struct NpcInstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_npc_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &NpcInstanceList)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("npc instance buffer"),
            contents: bytemuck::cast_slice(instance_data.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(NpcInstanceBuffer {
            buffer,
            length: instance_data.len(),
        });
    }
}

// --------------------------------------------------------------- draw

type DrawNpcInstanced = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    DrawNpcMeshInstanced,
);

struct DrawNpcMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawNpcMeshInstanced {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = Read<NpcInstanceBuffer>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: Option<&'w NpcInstanceBuffer>,
        (meshes, render_mesh_instances, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_allocator = mesh_allocator.into_inner();

        let Some(mesh_instance) =
            render_mesh_instances.render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.into_inner().get(mesh_instance.mesh_asset_id) else {
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
                let Some(index_buffer_slice) =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), *index_format);
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
