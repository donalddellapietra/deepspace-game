//! Entity raster pass: instanced triangle rendering for entity
//! subtrees, running AFTER the ray-march pass and depth-testing
//! against the same depth buffer the ray-march writes via
//! `@builtin(frag_depth)`.
//!
//! One GPU mesh per unique subtree NodeId, cached lifetime-of-run
//! (content-addressing makes this straightforward: same NodeId =
//! same mesh). A crowd of 10k identical soldiers = 1 vertex/index
//! buffer + 10k rows in the instance buffer + 1 draw call.
//!
//! Design constraints kept:
//! - Every entity is a subtree NodeId, identical to how terrain
//!   subtrees are identified. The mesh cache is a content-addressed
//!   satellite of the NodeLibrary — no entity-specific tree hacks.
//! - Zooming into an entity: when the ribbon descends so the entity
//!   IS the frame root, the ray-march handles it natively. The
//!   raster pass only runs against entities that are siblings of
//!   the frame, not against the frame itself.
//! - Toggle: when off, `EntityRasterState` is `None` on the
//!   renderer; scene builder routes entities through the ray-march
//!   tag=3 path exactly as before. Zero dead-code cost.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::world::tree::{NodeId, NodeLibrary};

pub mod extract;

use self::extract::{unit_cube_mesh, MeshVertex};

/// Sentinel NodeId for the LOD-terminal cube impostor. Real
/// library NodeIds start at 1, so 0 is unambiguous.
pub const LOD_CUBE_NODE: NodeId = 0;

/// Per-instance data written to the raster pass's instance buffer.
/// One row per live entity whose mesh is drawn this frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct InstanceData {
    pub translate: [f32; 3],
    /// Entity's anchor-cell edge length in frame-local units
    /// (same value the ray-march uses as `bbox_max - bbox_min`).
    pub scale: f32,
    pub tint: [f32; 4],
}

/// Uniforms for the raster pass: just the view+projection matrix.
/// Shared with the ray-march pass's `@builtin(frag_depth)` write so
/// depth values are pixel-accurate across the two.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RasterUniforms {
    pub view_proj: [[f32; 4]; 4],
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

/// One batch of entities that share the same mesh — emitted as a
/// single instanced draw call.
pub struct DrawBatch {
    pub node_id: NodeId,
    pub instance_start: u32,
    pub instance_count: u32,
}

pub struct EntityRasterState {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// NodeId → GpuMesh. Entries never evict during a run; edits
    /// create NEW NodeIds (content-addressing), so the old mesh is
    /// simply unused rather than invalid. Memory grows with unique
    /// subtree variants, not entity count.
    mesh_cache: HashMap<NodeId, GpuMesh>,
    /// Per-frame instance buffer. Rebuilt each frame from the
    /// current entity set (entities move, so transforms change).
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
    batches: Vec<DrawBatch>,
    /// Dedup staging so we don't repeat work every frame when
    /// entity set is stable.
    scratch_bucket: HashMap<NodeId, Vec<InstanceData>>,
}

impl EntityRasterState {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("entity_raster"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shader_compose::compose("entity_raster.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("entity_raster"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entity_raster_uniforms"),
            contents: bytemuck::bytes_of(&RasterUniforms {
                view_proj: identity4x4(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entity_raster"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("entity_raster"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_stride = std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress;
        let instance_stride = std::mem::size_of::<InstanceData>() as wgpu::BufferAddress;
        let vertex_attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
        ];
        let instance_attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 3 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 4 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 5 },
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("entity_raster"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: vertex_stride,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: instance_stride,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let initial_instance_bytes: u64 = 1024;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entity_raster_instances"),
            size: initial_instance_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Always-resident LOD-terminal cube mesh. Every entity whose
        // projected screen size falls below the LOD threshold gets
        // routed here, tinted by its subtree's representative block
        // color. One batch, one draw call, 12 triangles per instance.
        let cube = unit_cube_mesh();
        let cube_vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entity_lod_cube_vertices"),
            contents: bytemuck::cast_slice(&cube.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entity_lod_cube_indices"),
            contents: bytemuck::cast_slice(&cube.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let mut mesh_cache = HashMap::new();
        mesh_cache.insert(LOD_CUBE_NODE, GpuMesh {
            vertex_buffer: cube_vertex,
            index_buffer: cube_index,
            index_count: cube.indices.len() as u32,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniforms_buffer,
            bind_group,
            mesh_cache,
            instance_buffer,
            instance_capacity: initial_instance_bytes,
            batches: Vec::new(),
            scratch_bucket: HashMap::new(),
        }
    }

    /// Ensure a GPU mesh exists for `node_id`. Extracts + uploads on
    /// first call; subsequent calls are HashMap lookups. Returns
    /// `true` when a mesh is available (non-empty subtree).
    pub fn ensure_mesh(
        &mut self,
        device: &wgpu::Device,
        library: &NodeLibrary,
        node_id: NodeId,
        palette: &[[f32; 4]; 256],
    ) -> bool {
        if self.mesh_cache.contains_key(&node_id) {
            return true;
        }
        let Some(mesh) = extract::extract(library, node_id, palette) else {
            return false;
        };
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entity_mesh_vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entity_mesh_indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.mesh_cache.insert(node_id, GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        });
        true
    }

    /// Upload `view_proj` to the raster uniform buffer. Must be
    /// called each frame with the same matrix the ray-march shader
    /// uses to derive `frag_depth` — otherwise depth comparisons are
    /// off and entities z-fight or disappear behind terrain.
    pub fn update_view_proj(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
    ) {
        queue.write_buffer(
            &self.uniforms_buffer, 0,
            bytemuck::bytes_of(&RasterUniforms { view_proj }),
        );
    }

    /// Rebuild the instance buffer from `per_entity` (one entry per
    /// live entity) and assemble draw batches grouped by NodeId.
    /// Entities whose mesh isn't in the cache yet are skipped —
    /// callers should call `ensure_mesh` first for every unique
    /// NodeId in the entity set.
    pub fn update_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        per_entity: &[(NodeId, InstanceData)],
    ) {
        self.batches.clear();
        if per_entity.is_empty() {
            return;
        }
        // Sort a scratch copy by NodeId so instances of the same
        // mesh land contiguously. HashMap bucketing was 6–8× slower
        // at 10k entities because each insert hashes + probes, and
        // the resulting Vec allocations fragment. A radix of 2 values
        // (cube vs full mesh) doesn't need a real histogram, but sort
        // covers the general multi-mesh case too.
        let mut flat: Vec<(NodeId, InstanceData)> = per_entity
            .iter()
            .copied()
            .filter(|(id, _)| self.mesh_cache.contains_key(id))
            .collect();
        flat.sort_unstable_by_key(|(id, _)| *id);
        let mut flat_instances: Vec<InstanceData> = Vec::with_capacity(flat.len());
        let mut i = 0;
        while i < flat.len() {
            let node_id = flat[i].0;
            let start = flat_instances.len() as u32;
            while i < flat.len() && flat[i].0 == node_id {
                flat_instances.push(flat[i].1);
                i += 1;
            }
            let count = flat_instances.len() as u32 - start;
            self.batches.push(DrawBatch {
                node_id,
                instance_start: start,
                instance_count: count,
            });
        }
        let flat = flat_instances;
        let total = flat.len();
        if total == 0 {
            return;
        }

        // Grow the instance buffer if needed.
        let needed_bytes = std::mem::size_of_val(flat.as_slice()) as u64;
        if needed_bytes > self.instance_capacity {
            let new_cap = (needed_bytes * 3 / 2).max(needed_bytes + 1024);
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entity_raster_instances"),
                size: new_cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&flat));
    }

    /// Record the raster pass into `encoder`. Runs AFTER the ray-
    /// march pass — loads the existing color + depth attachments so
    /// the ray-march's output is preserved and z-tested against.
    pub fn record_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        if self.batches.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("entity_raster"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        for batch in &self.batches {
            let Some(mesh) = self.mesh_cache.get(&batch.node_id) else { continue };
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(
                mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            let end = batch.instance_start + batch.instance_count;
            pass.draw_indexed(
                0..mesh.index_count,
                0,
                batch.instance_start..end,
            );
        }
    }

    pub fn batch_count(&self) -> usize { self.batches.len() }
    pub fn total_instances(&self) -> u32 {
        self.batches.iter().map(|b| b.instance_count).sum()
    }
    pub fn cached_meshes(&self) -> usize { self.mesh_cache.len() }

    #[allow(dead_code)]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

/// Column-major 4x4 identity.
#[inline]
fn identity4x4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Build a column-major view×projection matrix consistent with the
/// ray-march shader's `camera.forward/right/up/fov` convention AND
/// the entity raster pass.
///
/// The ray-march writes `@builtin(frag_depth)` by multiplying the
/// world-space hit position by this same matrix — so as long as both
/// passes read the same uniform and both frag shaders compute
/// `clip.z / clip.w`, the depth tests compare apples to apples.
///
/// Assumes `forward` / `right` / `up` are unit and mutually
/// orthogonal (the app owns that invariant).
///
/// Near / far clip are hardcoded to 0.001 / 100.0 — frame-local
/// coordinates always fit in that range since the frame cell is
/// 3 units wide and the camera sits inside it.
pub fn compute_view_proj(
    pos: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    fov: f32,
    aspect: f32,
) -> [[f32; 4]; 4] {
    // Camera-space basis: camera looks down -Z, so view.z = -forward.
    let nx = right;
    let ny = up;
    let nz = [-forward[0], -forward[1], -forward[2]];
    let tx = -dot(nx, pos);
    let ty = -dot(ny, pos);
    let tz = -dot(nz, pos);
    // Column-major. WGSL / wgpu is column-major: column i, row j
    // is mat[i][j]. Build view so view * world_pos transforms into
    // camera space.
    let view = [
        [nx[0], ny[0], nz[0], 0.0],
        [nx[1], ny[1], nz[1], 0.0],
        [nx[2], ny[2], nz[2], 0.0],
        [tx,    ty,    tz,    1.0],
    ];
    // Perspective, depth range [0, 1] (wgpu convention).
    let near = 0.001_f32;
    let far = 100.0_f32;
    let t = (fov * 0.5).tan();
    let sy = 1.0 / t;
    let sx = sy / aspect;
    let a = far / (near - far);          // clip.z = a * view.z + b*1
    let b = (near * far) / (near - far);
    let proj = [
        [sx,  0.0, 0.0,  0.0],
        [0.0, sy,  0.0,  0.0],
        [0.0, 0.0, a,   -1.0],
        [0.0, 0.0, b,    0.0],
    ];
    mat_mul(proj, view)
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Column-major 4x4 matrix multiply: `out = a * b`.
fn mat_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[0][j] * b[i][0] +
                a[1][j] * b[i][1] +
                a[2][j] * b[i][2] +
                a[3][j] * b[i][3];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_proj_identity_at_origin() {
        // Camera at origin, looking -Z, ensures a point directly in
        // front lands within NDC x/y = 0 and a plausible depth.
        let vp = compute_view_proj(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.2, 1.0,
        );
        // Point 1 unit in front.
        let p = [0.0_f32, 0.0, -1.0, 1.0];
        let clip = [
            vp[0][0] * p[0] + vp[1][0] * p[1] + vp[2][0] * p[2] + vp[3][0] * p[3],
            vp[0][1] * p[0] + vp[1][1] * p[1] + vp[2][1] * p[2] + vp[3][1] * p[3],
            vp[0][2] * p[0] + vp[1][2] * p[1] + vp[2][2] * p[2] + vp[3][2] * p[3],
            vp[0][3] * p[0] + vp[1][3] * p[1] + vp[2][3] * p[2] + vp[3][3] * p[3],
        ];
        // NDC x/y should be 0 for on-axis point.
        assert!(clip[0].abs() < 1e-5);
        assert!(clip[1].abs() < 1e-5);
        // NDC z should be in [0, 1].
        let z = clip[2] / clip[3];
        assert!(z >= 0.0 && z <= 1.0, "ndc_z out of range: {z}");
    }
}
