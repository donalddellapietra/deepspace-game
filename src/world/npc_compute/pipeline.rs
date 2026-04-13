//! Compute pipeline setup and per-frame bind group preparation.

use std::borrow::Cow;

use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{sampler, storage_buffer_sized, texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
};

use super::data::{NpcComputeUniforms, NpcGpuData};
use super::compute_shader_handle;

// --------------------------------------------------------------- pipeline resource

#[derive(Resource)]
pub struct NpcComputePipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub pipeline_id: CachedComputePipelineId,
}

pub fn init_npc_compute_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "npc_compute_bgl",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // @binding(0): NPC state storage buffer (read_write)
                storage_buffer_sized(false, None),
                // @binding(1): Uniforms
                uniform_buffer::<NpcComputeUniforms>(false),
                // @binding(2): Heightmap texture
                texture_2d(TextureSampleType::Float { filterable: true }),
                // @binding(3): Heightmap sampler
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    let shader = compute_shader_handle();
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("simulate")),
        ..default()
    });

    commands.insert_resource(NpcComputePipeline {
        bind_group_layout,
        pipeline_id,
    });
}

// --------------------------------------------------------------- bind group

/// Prepared bind group for the compute dispatch.
#[derive(Resource)]
pub struct NpcComputeBindGroup {
    pub bind_group: BindGroup,
    pub npc_count: u32,
}

/// Each frame: upload NPC state if dirty, update uniforms, create bind group.
pub fn prepare_npc_bind_group(
    mut commands: Commands,
    pipeline: Res<NpcComputePipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_data: Res<NpcGpuData>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    // TODO: pass the heightmap handle from the main world.
    // For now, we'll skip the bind group if no heightmap is available.
) {
    if gpu_data.states.is_empty() {
        return;
    }

    // Don't create bind group until the compute pipeline has compiled.
    if !matches!(
        pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id),
        bevy::render::render_resource::CachedPipelineState::Ok(_)
    ) {
        return;
    }

    // Create/update the NPC state storage buffer.
    let npc_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("npc_state_buffer"),
        contents: bytemuck::cast_slice(&gpu_data.states),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });

    // Uniform buffer.
    let mut uniform_buf = UniformBuffer::from(gpu_data.uniforms);
    uniform_buf.write_buffer(&render_device, &queue);

    // TODO: bind the actual heightmap texture.
    // For now, create a 1x1 placeholder so the pipeline compiles.
    let placeholder_tex = render_device.create_texture(&TextureDescriptor {
        label: Some("heightmap_placeholder"),
        size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let placeholder_view = placeholder_tex.create_view(&TextureViewDescriptor::default());
    let placeholder_sampler = render_device.create_sampler(&SamplerDescriptor {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    let bind_group = render_device.create_bind_group(
        Some("npc_compute_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
        &BindGroupEntries::sequential((
            npc_buffer.as_entire_buffer_binding(),
            &uniform_buf,
            &placeholder_view,
            &placeholder_sampler,
        )),
    );

    commands.insert_resource(NpcComputeBindGroup {
        bind_group,
        npc_count: gpu_data.states.len() as u32,
    });
}
