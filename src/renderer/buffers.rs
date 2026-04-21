//! Per-frame buffer uploads. Every upload goes through
//! `upload_or_recreate` — when the new payload outgrows the existing
//! buffer we allocate a fresh one and return `true`, and the caller
//! rebuilds the bind group.

use wgpu::util::DeviceExt;

use crate::world::gpu::{GpuCamera, GpuEntity, GpuNodeKind, GpuRibbonEntry};

use super::{GpuUniforms, Renderer, MAX_RIBBON_LEN};

impl Renderer {
    /// Sync the GPU tree buffers to match the latest packed state.
    ///
    /// The pack buffer is append-only: every edit grows `tree`,
    /// `node_kinds`, and `node_offsets` with a handful of new
    /// entries. This method writes ONLY the appended tail via
    /// `queue.write_buffer` instead of re-uploading the whole thing,
    /// so edits don't stall the shader behind an 8 MB write barrier.
    ///
    /// Falls back to full allocate + bind-group-rebuild only when an
    /// append would overflow the current GPU buffer allocation.
    pub fn update_tree(
        &mut self,
        tree: &[u32],
        node_kinds: &[GpuNodeKind],
        node_offsets: &[u32],
        aabbs: &[u32],
        root_bfs_index: u32,
    ) {
        self.root_index = root_bfs_index;
        self.node_count = node_kinds.len() as u32;

        // Storage buffers cannot be zero-sized. Supply stubs for the
        // pathological "empty pack" case so the bind group stays valid.
        let stub_tree = [0u32, 2u32];
        let stub_offsets = [0u32];
        let stub_aabbs = [0u32];
        let tree_payload: &[u32] = if tree.is_empty() { &stub_tree } else { tree };
        let offsets_payload: &[u32] = if node_offsets.is_empty() { &stub_offsets } else { node_offsets };
        let aabbs_payload: &[u32] = if aabbs.is_empty() { &stub_aabbs } else { aabbs };

        let tree_byte_size = std::mem::size_of_val(tree_payload) as u64;
        let max_binding_size = self.device.limits().max_storage_buffer_binding_size as u64;
        if tree_byte_size > max_binding_size {
            eprintln!(
                "tree_buffer_size_warning packed={} bytes, limit={} bytes",
                tree_byte_size, max_binding_size,
            );
        }

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let write_start = web_time::Instant::now();
        let prev_tree_u32s = self.uploaded_tree_u32s;
        let tree_grew = append_or_recreate_u32(
            &mut self.tree_buffer, &mut self.uploaded_tree_u32s,
            &self.device, &self.queue, "tree", tree_payload, storage,
        );
        let kinds_grew = append_or_recreate(
            &mut self.node_kinds_buffer, &mut self.uploaded_kinds_count,
            &self.device, &self.queue, "node_kinds", node_kinds, storage,
        );
        let offsets_grew = append_or_recreate_u32(
            &mut self.node_offsets_buffer, &mut self.uploaded_offsets_count,
            &self.device, &self.queue, "node_offsets", offsets_payload, storage,
        );
        let aabbs_grew = append_or_recreate_u32(
            &mut self.aabbs_buffer, &mut self.uploaded_aabbs_count,
            &self.device, &self.queue, "aabbs", aabbs_payload, storage,
        );
        self.last_tree_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
        let _ = prev_tree_u32s;

        self.last_bind_group_rebuild_ms = 0.0;
        if tree_grew || kinds_grew || offsets_grew || aabbs_grew {
            let rebuild_start = web_time::Instant::now();
            self.bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer, &self.node_offsets_buffer,
                &self.aabbs_buffer,
                &self.mask_view,
                &self.entity_buffer,
            );
            self.coarse_bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer, &self.node_offsets_buffer,
                &self.aabbs_buffer,
                &self.dummy_mask_view,
                &self.entity_buffer,
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
        let write_start = web_time::Instant::now();
        let needed = std::mem::size_of_val(payload) as u64;
        let grew = if needed > self.ribbon_buffer.size() {
            self.ribbon_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ribbon"),
                contents: bytemuck::cast_slice(payload),
                usage: storage,
            });
            true
        } else {
            self.queue.write_buffer(&self.ribbon_buffer, 0, bytemuck::cast_slice(payload));
            false
        };
        self.last_ribbon_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
        if grew {
            let rebuild_start = web_time::Instant::now();
            self.bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer, &self.node_offsets_buffer,
                &self.aabbs_buffer,
                &self.mask_view,
                &self.entity_buffer,
            );
            self.coarse_bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer, &self.node_offsets_buffer,
                &self.aabbs_buffer,
                &self.dummy_mask_view,
                &self.entity_buffer,
            );
            self.last_bind_group_rebuild_ms += rebuild_start.elapsed().as_secs_f64() * 1000.0;
        }
        self.write_uniforms();
    }

    /// Upload the current palette. The buffer is a read-only
    /// storage buffer sized to `registry.len() * 16 bytes`; if the
    /// new palette is longer than what the buffer can hold, we
    /// recreate it (and rebuild both bind groups) so it fits.
    pub fn update_palette(&mut self, palette: &[[f32; 4]]) {
        // Every color is 16 bytes; the minimum buffer size is 16
        // (one entry) to keep the storage binding valid.
        let stub = [[0.0f32; 4]];
        let payload: &[[f32; 4]] = if palette.is_empty() { &stub } else { palette };
        let needed = std::mem::size_of_val(payload) as u64;
        if needed > self.palette_buffer.size() {
            // Grow with 1.5× headroom, rounded up to a multiple of 16.
            let raw = (needed.max(16) * 3 / 2).max(16);
            let new_size = raw.div_ceil(16) * 16;
            self.palette_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("palette"),
                size: new_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&self.palette_buffer, 0, bytemuck::cast_slice(payload));
            // Rebuild both bind groups: palette buffer identity changed.
            self.bind_group = make_bind_group(
                &self.device,
                &self.bind_group_layout,
                &self.tree_buffer,
                &self.camera_buffer,
                &self.palette_buffer,
                &self.uniforms_buffer,
                &self.node_kinds_buffer,
                &self.ribbon_buffer,
                &self.shader_stats_buffer,
                &self.node_offsets_buffer,
                &self.aabbs_buffer,
                &self.mask_view,
                &self.entity_buffer,
            );
            self.coarse_bind_group = make_bind_group(
                &self.device,
                &self.bind_group_layout,
                &self.tree_buffer,
                &self.camera_buffer,
                &self.palette_buffer,
                &self.uniforms_buffer,
                &self.node_kinds_buffer,
                &self.ribbon_buffer,
                &self.shader_stats_buffer,
                &self.node_offsets_buffer,
                &self.aabbs_buffer,
                &self.dummy_mask_view,
                &self.entity_buffer,
            );
        } else {
            self.queue
                .write_buffer(&self.palette_buffer, 0, bytemuck::cast_slice(payload));
        }
    }

    /// Upload the entity list. Always overwrites the buffer from
    /// offset 0 — entity positions change every frame under motion,
    /// so a tail-only write would miss updates. Recreates the
    /// buffer with 1.5× headroom on overflow; rebuilds both
    /// bind groups when the buffer identity changes.
    ///
    /// `entity_count` on the uniforms gates shader iteration, so
    /// the one-entry stub allocated at init is never read when no
    /// entities are live.
    pub fn update_entities(&mut self, entities: &[GpuEntity]) {
        self.entity_count = entities.len() as u32;
        let stub = [GpuEntity::default()];
        let payload: &[GpuEntity] = if entities.is_empty() { &stub } else { entities };
        let needed = std::mem::size_of_val(payload) as u64;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let mut grew = false;
        if needed > self.entity_buffer.size() {
            let raw = (needed * 3 / 2).max(needed + 256);
            let new_size = raw.div_ceil(16) * 16;
            self.entity_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("entities"),
                size: new_size,
                usage,
                mapped_at_creation: false,
            });
            grew = true;
        }
        self.queue
            .write_buffer(&self.entity_buffer, 0, bytemuck::cast_slice(payload));
        self.uploaded_entities_count = entities.len() as u64;

        if grew {
            self.bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer, &self.node_offsets_buffer,
                &self.aabbs_buffer, &self.mask_view, &self.entity_buffer,
            );
            self.coarse_bind_group = make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.tree_buffer, &self.camera_buffer, &self.palette_buffer,
                &self.uniforms_buffer, &self.node_kinds_buffer, &self.ribbon_buffer,
                &self.shader_stats_buffer, &self.node_offsets_buffer,
                &self.aabbs_buffer, &self.dummy_mask_view, &self.entity_buffer,
            );
        }
        self.write_uniforms();
    }

    pub fn update_camera(&mut self, camera: &GpuCamera) {
        let write_start = web_time::Instant::now();
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        self.last_camera_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
        // Keep a CPU-side copy with jitter cleared — the TAA resolve
        // path stashes this as `prev_camera` and re-derives ray
        // directions at pixel centers, so the stored form must NOT
        // carry whatever jitter was overlaid on the GPU buffer below
        // by the TAA path (see `render()`).
        let mut mirror = *camera;
        mirror.jitter_x_px = 0.0;
        mirror.jitter_y_px = 0.0;
        self.last_camera = mirror;
    }

    pub(super) fn write_uniforms(&self) {
        // When TAAU is on the ray-march pipeline writes into a
        // half-res target; feeding full-res dimensions here would
        // mis-scale the jitter NDC and shrink the crosshair 2×
        // under upscale. Use the march-pass dimensions instead.
        let (sw, sh) = self.march_dims();
        let uniforms = GpuUniforms {
            root_index: self.root_index,
            node_count: self.node_count,
            screen_width: sw as f32,
            screen_height: sh as f32,
            max_depth: self.max_depth,
            highlight_active: self.highlight_active,
            ribbon_count: self.ribbon_count,
            entity_count: self.entity_count,
            highlight_min: self.highlight_min,
            highlight_max: self.highlight_max,
            sphere_body_active: self.sphere_body_active,
            sphere_body_root_bfs: self.sphere_body_root_bfs,
            _pad_sphere: [0; 3],
        };
        self.queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

/// Append-style buffer sync for generic Pod slices indexed by element
/// count. When `data.len() > *uploaded_count`, writes only the new
/// tail via `queue.write_buffer`. If the buffer can't hold `data` at
/// its current size, recreates it with 1.5× headroom (so the next
/// few edits patch in-place without another rebuild) and returns
/// `true` so the caller knows to rebuild the bind group.
pub(super) fn append_or_recreate<T: bytemuck::Pod>(
    buffer: &mut wgpu::Buffer,
    uploaded_count: &mut u64,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> bool {
    let elem_size = std::mem::size_of::<T>() as u64;
    let needed = data.len() as u64 * elem_size;
    if needed > buffer.size() {
        // Overflow: recreate with 1.5× headroom so the next several
        // edits fit without another grow. Round UP to a multiple of
        // `elem_size` so WebGPU's strict binding-size validation
        // accepts the buffer (it requires storage-binding sizes to be
        // a whole number of elements). Native Metal silently tolerates
        // this; WebGPU does not.
        let raw = (needed.max(1) * 3 / 2).max(elem_size);
        let new_size = raw.div_ceil(elem_size) * elem_size;
        *buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: new_size,
            usage,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        }
        *uploaded_count = data.len() as u64;
        return true;
    }
    if data.len() as u64 > *uploaded_count {
        let start_elem = *uploaded_count as usize;
        let tail = &data[start_elem..];
        let byte_offset = *uploaded_count * elem_size;
        queue.write_buffer(buffer, byte_offset, bytemuck::cast_slice(tail));
        *uploaded_count = data.len() as u64;
    } else if (data.len() as u64) < *uploaded_count {
        // Pack shouldn't shrink, but if it ever does, rewrite from 0
        // so we don't leak stale tail content.
        if !data.is_empty() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        }
        *uploaded_count = data.len() as u64;
    }
    false
}

/// Same as `append_or_recreate` but specialized to `&[u32]`. Useful
/// because `tree: Vec<u32>` and `node_offsets: Vec<u32>` use u32
/// counts rather than element counts of a larger struct.
pub(super) fn append_or_recreate_u32(
    buffer: &mut wgpu::Buffer,
    uploaded_u32s: &mut u64,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    data: &[u32],
    usage: wgpu::BufferUsages,
) -> bool {
    append_or_recreate(buffer, uploaded_u32s, device, queue, label, data, usage)
}

#[allow(clippy::too_many_arguments)]
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
    node_offsets: &wgpu::Buffer,
    aabbs: &wgpu::Buffer,
    mask_view: &wgpu::TextureView,
    entities: &wgpu::Buffer,
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
            wgpu::BindGroupEntry { binding: 7, resource: node_offsets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(mask_view) },
            wgpu::BindGroupEntry { binding: 9, resource: aabbs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: entities.as_entire_binding() },
        ],
    })
}
