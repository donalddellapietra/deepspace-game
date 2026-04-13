//! Compute pass that builds instance data from NPC state.
//!
//! Replaces CPU-side collect_overlays + reconcile_instanced.
//! Reads NPC state buffer + animation data, writes instance transforms.

use std::borrow::Cow;
use std::sync::OnceLock;

use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_graph::{self, RenderGraph, RenderLabel},
    render_resource::{
        binding_types::{storage_buffer_sized, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy::shader::Shader;
use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};

use super::anim_data::{GpuAnimData, GpuKeyframe, GpuPartInfo, GPU_MAX_KEYFRAMES, GPU_MAX_PARTS};
use super::data::GpuNpcState;

// --------------------------------------------------------------- data

/// Uniforms for the instance builder compute shader.
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
pub struct BuildUniforms {
    pub npc_count: u32,
    pub num_parts: u32,
    pub num_keyframes: u32,
    pub _pad: u32,
    pub frame_duration: f32,
    pub total_duration: f32,
    pub scale: f32,
    pub _pad2: f32,
}

/// CPU-side resource holding all data needed for instance building.
#[derive(Resource, Clone, ExtractResource)]
pub struct InstanceBuilderData {
    /// NPC states (same as NpcGpuData.states).
    pub npc_states: Vec<GpuNpcState>,
    /// Part info (rest offsets, pivots).
    pub parts: [GpuPartInfo; GPU_MAX_PARTS],
    /// Keyframes: [keyframe][part].
    pub keyframes: Vec<GpuKeyframe>, // flattened: keyframes * parts
    /// Per-part colors (RGBA).
    pub colors: [[f32; 4]; GPU_MAX_PARTS],
    /// Build uniforms.
    pub uniforms: BuildUniforms,
    /// Whether data needs re-upload.
    pub dirty: bool,
}

impl Default for InstanceBuilderData {
    fn default() -> Self {
        Self {
            npc_states: Vec::new(),
            parts: [GpuPartInfo {
                rest_offset: [0.0; 3],
                _pad0: 0.0,
                pivot: [0.0; 3],
                _pad1: 0.0,
            }; GPU_MAX_PARTS],
            keyframes: Vec::new(),
            colors: [[1.0, 0.0, 1.0, 1.0]; GPU_MAX_PARTS],
            uniforms: BuildUniforms {
                npc_count: 0,
                num_parts: 0,
                num_keyframes: 0,
                _pad: 0,
                frame_duration: 0.2,
                total_duration: 0.8,
                scale: 1.0,
                _pad2: 0.0,
            },
            dirty: false,
        }
    }
}

// --------------------------------------------------------------- plugin

static BUILDER_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct InstanceBuilderLabel;

pub struct InstanceBuilderPlugin;

impl Plugin for InstanceBuilderPlugin {
    fn build(&self, app: &mut App) {
        let handle = {
            let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
            shaders.add(Shader::from_wgsl(
                include_str!("../../../assets/shaders/npc_build_instances.wgsl"),
                "npc_build_instances.wgsl",
            ))
        };
        BUILDER_SHADER.set(handle).ok();

        app.init_resource::<InstanceBuilderData>();
        app.add_plugins(ExtractResourcePlugin::<InstanceBuilderData>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_builder_pipeline)
            .add_systems(
                Render,
                prepare_builder_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(InstanceBuilderLabel, InstanceBuilderNode::default());
        render_graph.add_node_edge(
            InstanceBuilderLabel,
            bevy::render::graph::CameraDriverLabel,
        );
    }
}

// --------------------------------------------------------------- pipeline

#[derive(Resource)]
struct BuilderPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
}

fn init_builder_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "npc_instance_builder_bgl",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),  // @binding(0) npcs (read)
                uniform_buffer::<BuildUniforms>(false), // @binding(1) uniforms
                storage_buffer_sized(false, None),  // @binding(2) parts (read)
                storage_buffer_sized(false, None),  // @binding(3) keyframes (read)
                storage_buffer_sized(false, None),  // @binding(4) instances (read_write)
                storage_buffer_sized(false, None),  // @binding(5) colors (read)
            ),
        ),
    );

    let shader = BUILDER_SHADER.get().cloned().expect("Builder shader not loaded");
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("build_instances")),
        ..default()
    });

    commands.insert_resource(BuilderPipeline {
        bind_group_layout,
        pipeline_id,
    });
}

// --------------------------------------------------------------- bind group

#[derive(Resource)]
pub struct BuilderBindGroup {
    pub bind_group: BindGroup,
    pub instance_buffer: Buffer,
    pub total_instances: u32,
}

fn prepare_builder_bind_group(
    mut commands: Commands,
    pipeline: Res<BuilderPipeline>,
    pipeline_cache: Res<PipelineCache>,
    data: Res<InstanceBuilderData>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    if data.npc_states.is_empty() || data.uniforms.num_parts == 0 {
        return;
    }

    if !matches!(
        pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id),
        CachedPipelineState::Ok(_)
    ) {
        return;
    }

    let total_instances = data.uniforms.npc_count * data.uniforms.num_parts;

    // NPC state buffer.
    let npc_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("builder_npc_states"),
        contents: bytemuck::cast_slice(&data.npc_states),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Uniforms.
    let mut uniform_buf = UniformBuffer::from(data.uniforms);
    uniform_buf.write_buffer(&render_device, &queue);

    // Part info buffer.
    let parts_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("builder_parts"),
        contents: bytemuck::cast_slice(&data.parts),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Keyframes buffer.
    let kf_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("builder_keyframes"),
        contents: bytemuck::cast_slice(&data.keyframes),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Instance output buffer (written by compute, read by vertex shader).
    let instance_size = total_instances as u64 * 80; // 80 bytes per NpcInstanceData
    let instance_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("builder_instances"),
        size: instance_size.max(80), // at least one instance
        usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Colors buffer.
    let colors_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("builder_colors"),
        contents: bytemuck::cast_slice(&data.colors),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let bind_group = render_device.create_bind_group(
        Some("npc_instance_builder_bg"),
        &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
        &BindGroupEntries::sequential((
            npc_buffer.as_entire_buffer_binding(),
            &uniform_buf,
            parts_buffer.as_entire_buffer_binding(),
            kf_buffer.as_entire_buffer_binding(),
            instance_buffer.as_entire_buffer_binding(),
            colors_buffer.as_entire_buffer_binding(),
        )),
    );

    commands.insert_resource(BuilderBindGroup {
        bind_group,
        instance_buffer,
        total_instances,
    });
}

// --------------------------------------------------------------- render graph node

enum BuilderState {
    Loading,
    Ready,
}

struct InstanceBuilderNode {
    state: BuilderState,
}

impl Default for InstanceBuilderNode {
    fn default() -> Self {
        Self { state: BuilderState::Loading }
    }
}

impl render_graph::Node for InstanceBuilderNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<BuilderPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if let BuilderState::Loading = self.state {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id)
            {
                self.state = BuilderState::Ready;
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !matches!(self.state, BuilderState::Ready) {
            return Ok(());
        }

        let Some(bg) = world.get_resource::<BuilderBindGroup>() else {
            return Ok(());
        };
        if bg.total_instances == 0 {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<BuilderPipeline>();

        let Some(compute_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &bg.bind_group, &[]);
        pass.set_pipeline(compute_pipeline);

        let workgroups = (bg.total_instances + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }
}
