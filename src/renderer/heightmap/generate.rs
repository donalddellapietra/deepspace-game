//! Heightmap generation compute pipeline.
//!
//! One compute dispatch writes the entire heightmap texture: every
//! texel's thread walks the tree rooted at `frame_root_bfs` and
//! finds the top-Y of the highest solid collision cell in its
//! column. See `heightmap_gen.wgsl` for the per-thread algorithm.
//!
//! The pipeline's bind group layout is intentionally minimal —
//! only `tree`, `node_offsets`, the heightmap uniforms, and the
//! output texture. Camera / palette / ribbon etc. from the render
//! pass are irrelevant here and deliberately excluded so the
//! binding surface stays small.

use super::{HeightmapTexture, GEN_WORKGROUP_SIDE};

/// Bind group layout for the heightmap gen shader.
/// - binding 0: `tree` storage buffer (read-only) — the packed
///   world tree, same buffer the render shader uses.
/// - binding 1: `node_offsets` storage buffer (read-only) — BFS
///   index → u32 offset into `tree[]`.
/// - binding 2: heightmap uniforms (HeightmapUniforms).
/// - binding 3: heightmap storage texture (R32F, write-only).
fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("heightmap_gen"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: super::HEIGHTMAP_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

/// Heightmap generation pipeline.
pub struct HeightmapGen {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl HeightmapGen {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("heightmap_gen"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shader_compose::compose("heightmap_gen.wgsl").into(),
            ),
        });
        let bind_group_layout = bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("heightmap_gen"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("heightmap_gen"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }

    /// Build a one-shot bind group for a given (tree, offsets,
    /// heightmap) triple. Lightweight — keep per-frame if the
    /// underlying buffers are stable, or rebuild on resource
    /// changes.
    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        tree: &wgpu::Buffer,
        node_offsets: &wgpu::Buffer,
        heightmap: &HeightmapTexture,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("heightmap_gen"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tree.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: node_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: heightmap.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&heightmap.view),
                },
            ],
        })
    }

    /// Record a full-heightmap dispatch into `encoder`. Workgroup
    /// count per axis is ceil(side / 8). Every base-2 size in the
    /// shipped range [2, 128] is an exact multiple of 8 (except 2
    /// and 4), so the ceiling is only nontrivial for very small
    /// sides — in which case the excess threads bail via the
    /// `u >= side` check at the top of `cs_main`.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        heightmap: &HeightmapTexture,
    ) {
        let wg_per_axis = heightmap.side.div_ceil(GEN_WORKGROUP_SIDE);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("heightmap_gen"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_per_axis, wg_per_axis, 1);
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
