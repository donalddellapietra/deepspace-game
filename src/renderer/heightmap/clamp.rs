//! Per-entity Y clamp compute pipeline.
//!
//! Reads a pre-generated heightmap texture and the raster
//! `InstanceData` buffer; for each live instance, samples the
//! heightmap at the instance's XZ and patches `translate.y` to
//! land the entity's bbox-min corner on the ground. Instances
//! outside the heightmap extent, or in a column the heightmap
//! flagged as `GROUND_NONE`, are left untouched.
//!
//! This keeps CPU-side motion authoritative for XZ (the existing
//! tick loop updates entity positions) and lets the GPU resolve Y
//! at render time — no persistent GPU entity state, no CPU
//! readback.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{HeightmapTexture, GROUND_NONE};

/// Uniforms matching `ClampUniforms` in `entity_heightmap_clamp.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ClampUniforms {
    pub entity_count: u32,
    pub heightmap_side: u32,
    pub no_ground_threshold: f32,
    pub frame_xz_origin_x: f32,
    pub frame_xz_origin_z: f32,
    pub frame_xz_size: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl ClampUniforms {
    pub fn new(entity_count: u32, heightmap_side: u32, frame_xz_size: f32) -> Self {
        Self {
            entity_count,
            heightmap_side,
            // Any value more negative than a plausible world Y is a
            // "no ground" marker. `GROUND_NONE` is -1e30; use -1e20
            // as the threshold to absorb float round-trip error.
            no_ground_threshold: -1.0e20,
            frame_xz_origin_x: 0.0,
            frame_xz_origin_z: 0.0,
            frame_xz_size,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

const _ASSERT_GROUND_NONE_BELOW_THRESHOLD: () = {
    // Keep the sentinel ↔ threshold invariant locked in at compile
    // time. If someone changes `GROUND_NONE` without updating the
    // clamp threshold, the shader would start treating real ground
    // as "no ground" and vice versa.
    assert!(GROUND_NONE < -1.0e20);
};

fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entity_heightmap_clamp"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                // Sampled texture — read-only storage textures
                // require a native-only feature that wgpu's WebGPU
                // backend rejects. `textureLoad` on a `texture_2d`
                // with integer coords does what we need without
                // the feature gate.
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

pub const CLAMP_WORKGROUP_SIZE: u32 = 64;

pub struct EntityHeightmapClamp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl EntityHeightmapClamp {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("entity_heightmap_clamp"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shader_compose::compose("entity_heightmap_clamp.wgsl").into(),
            ),
        });
        let bind_group_layout = bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("entity_heightmap_clamp"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("entity_heightmap_clamp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }

    /// Build uniforms buffer from a `ClampUniforms` value. Callers
    /// typically recreate this every frame (`entity_count` changes
    /// on spawn/despawn); the buffer is tiny and creation is cheap.
    pub fn make_uniforms_buffer(&self, device: &wgpu::Device, u: &ClampUniforms) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entity_heightmap_clamp_uniforms"),
            contents: bytemuck::bytes_of(u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        instances: &wgpu::Buffer,
        uniforms: &wgpu::Buffer,
        heightmap: &HeightmapTexture,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entity_heightmap_clamp"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: instances.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&heightmap.view),
                },
            ],
        })
    }

    /// Dispatch the clamp pass. `entity_count` must match the
    /// `entity_count` baked into the uniform buffer bound to
    /// `bind_group`.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        entity_count: u32,
    ) {
        if entity_count == 0 {
            return;
        }
        let wg = entity_count.div_ceil(CLAMP_WORKGROUP_SIZE);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entity_heightmap_clamp"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
