//! Per-frame buffer uploads. Every upload goes through
//! `upload_or_recreate` — when the new payload outgrows the existing
//! buffer we allocate a fresh one and return `true`, and the caller
//! rebuilds the bind group.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuChild, GpuNodeKind, GpuPalette, GpuRibbonEntry};

use super::{GpuUniforms, Renderer, MAX_RIBBON_LEN};

impl Renderer {
    /// Re-upload the tree + node_kinds buffers after an edit or
    /// re-pack. Recreates the GPU buffers and bind group when the
    /// data outgrew the previous allocation.
    pub fn update_tree(
        &mut self,
        tree_data: &[GpuChild],
        node_kinds: &[GpuNodeKind],
        root_index: u32,
    ) {
        self.root_index = root_index;
        self.node_count = (tree_data.len() / 27) as u32;

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let write_start = std::time::Instant::now();
        let tree_grew = upload_or_recreate(
            &mut self.tree_buffer, &self.device, &self.queue,
            "tree", tree_data, storage,
        );
        let kinds_grew = upload_or_recreate(
            &mut self.node_kinds_buffer, &self.device, &self.queue,
            "node_kinds", node_kinds, storage,
        );
        self.last_tree_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;

        self.last_bind_group_rebuild_ms = 0.0;
        if tree_grew || kinds_grew {
            let rebuild_start = std::time::Instant::now();
            self.bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer,
            );
            self.last_bind_group_rebuild_ms = rebuild_start.elapsed().as_secs_f64() * 1000.0;
        }

        self.write_uniforms();
    }

    /// Upload the ancestor ribbon (pop chain from frame's direct
    /// parent up to the absolute root). Resizes the ribbon buffer
    /// if needed and recreates the bind group.
    pub fn update_ribbon(&mut self, ribbon: &[GpuRibbonEntry]) {
        let truncated = if ribbon.len() > MAX_RIBBON_LEN {
            &ribbon[..MAX_RIBBON_LEN]
        } else {
            ribbon
        };
        // Always upload at least one entry — empty storage buffers
        // break the bind group.
        let stub_storage = [GpuRibbonEntry { node_idx: 0, slot_bits: 0 }];
        let payload: &[GpuRibbonEntry] = if truncated.is_empty() {
            &stub_storage
        } else {
            truncated
        };

        self.ribbon_count = truncated.len() as u32;

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let write_start = std::time::Instant::now();
        let grew = upload_or_recreate(
            &mut self.ribbon_buffer, &self.device, &self.queue,
            "ribbon", payload, storage,
        );
        self.last_ribbon_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
        if grew {
            let rebuild_start = std::time::Instant::now();
            self.bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer,
            );
            self.last_bind_group_rebuild_ms += rebuild_start.elapsed().as_secs_f64() * 1000.0;
        }
        self.write_uniforms();
    }

    pub fn update_palette(&self, palette: &GpuPalette) {
        self.queue.write_buffer(&self.palette_buffer, 0, bytemuck::bytes_of(palette));
    }

    pub fn update_camera(&mut self, camera: &GpuCamera) {
        let write_start = std::time::Instant::now();
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        self.last_camera_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    }

    pub(super) fn write_uniforms(&self) {
        let uniforms = GpuUniforms {
            root_index: self.root_index,
            node_count: self.node_count,
            screen_width: self.config.width as f32,
            screen_height: self.config.height as f32,
            max_depth: self.max_depth,
            highlight_active: self.highlight_active,
            root_kind: self.root_kind,
            ribbon_count: self.ribbon_count,
            highlight_min: self.highlight_min,
            highlight_max: self.highlight_max,
            root_radii: self.root_radii,
            root_face_meta: self.root_face_meta,
            root_face_bounds: self.root_face_bounds,
            root_face_pop_pos: self.root_face_pop_pos,
        };
        self.queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

/// Upload `data` into `buffer`. If the new payload is larger than
/// the existing allocation, create a fresh buffer (and signal the
/// caller via the returned `bool` that the bind group needs to be
/// rebuilt). Otherwise patch in place with `queue.write_buffer`.
pub(super) fn upload_or_recreate<T: bytemuck::Pod>(
    buffer: &mut wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> bool {
    let needed = std::mem::size_of_val(data) as u64;
    if needed > buffer.size() {
        *buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });
        true
    } else {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        false
    }
}

pub(super) fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    tree: &wgpu::Buffer,
    camera: &wgpu::Buffer,
    palette: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
    node_kinds: &wgpu::Buffer,
    ribbon: &wgpu::Buffer,
    shader_stats: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ray_march"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tree.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: camera.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: palette.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: node_kinds.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: ribbon.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: shader_stats.as_entire_binding() },
        ],
    })
}
