//! Cursor-probe compute pipeline: re-runs `march()` once per frame
//! for the crosshair ray and writes the hit to a CPU-readable
//! staging buffer. Replaces the old CPU raycast module entirely —
//! the GPU is the single source of truth for ray-march hits, so
//! there is no CPU/GPU algorithm pair that can drift.
//!
//! Flow:
//! 1. `dispatch_cursor_probe()` — encodes a `@workgroup_size(1, 1, 1)`
//!    compute dispatch that reads the same uniforms / tree / ribbon
//!    the fragment shader uses, then writes
//!    `{ hit, depth, t, face, path[16 u32s] }` to a storage buffer.
//! 2. `queue_cursor_probe_readback()` — copies the storage buffer to
//!    a `MAP_READ` staging buffer, returns a future that resolves
//!    once the copy has completed.
//! 3. `read_cursor_probe_sync()` — polls the device until the
//!    staging buffer maps, reads the packed result, unmaps. Called
//!    from the harness probe commands and from break/place.
//!
//! Live frames don't need to block on the readback — they consume
//! the previous frame's result for the highlight uniform. Scripted
//! probes and edits pay the ~sub-ms sync to get "this frame's" hit.

use wgpu::util::DeviceExt;

/// Raw packed layout emitted by `assets/shaders/cursor_probe.wgsl`.
/// Must match `CursorProbeOut` in the shader byte-for-byte. The
/// `path` is a 16-u32 (64-byte) packed slot array identical to
/// `Uniforms.highlight_path` / `render_path` in `bindings.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct CursorProbeRaw {
    pub hit: u32,
    pub depth: u32,
    pub t: f32,
    pub face: u32,
    pub path: [[u32; 4]; 4],
}

pub const CURSOR_PROBE_BYTES: u64 = std::mem::size_of::<CursorProbeRaw>() as u64;

/// Decoded cursor-probe result the CPU consumes.
#[derive(Clone, Debug, Default)]
pub struct CursorProbe {
    pub hit: bool,
    pub depth: u32,
    pub t: f32,
    /// Normal-axis face id (0/1 = ±X, 2/3 = ±Y, 4/5 = ±Z) matching
    /// the Cartesian DDA convention — computed from the hit normal
    /// by the shader so the CPU doesn't have to.
    pub face: u8,
    /// Hit cell's slot path from the world root (length == `depth`).
    /// Empty vec on miss.
    pub slots: Vec<u8>,
}

impl CursorProbe {
    pub fn decode(raw: &CursorProbeRaw) -> Self {
        let hit = raw.hit != 0;
        let depth = raw.depth.min(64);
        let mut slots = Vec::with_capacity(depth as usize);
        if hit {
            for i in 0..depth {
                let word = (i / 16) as usize;
                let lane = ((i / 4) % 4) as usize;
                let byte = (i % 4) as usize;
                let lane_u32 = raw.path[word][lane];
                slots.push(((lane_u32 >> (byte * 8)) & 0xFF) as u8);
            }
        }
        CursorProbe {
            hit,
            depth,
            t: raw.t,
            face: (raw.face & 0xFF) as u8,
            slots,
        }
    }
}

/// GPU-side cursor-probe resources: pipeline, output buffer, staging
/// buffer, and bind group. Owned by the `Renderer` and torn down
/// with the device.
pub struct CursorProbe_Gpu {
    pub pipeline: wgpu::ComputePipeline,
    /// Written by the compute shader (storage, `read_write`).
    pub output_buffer: wgpu::Buffer,
    /// `COPY_DST | MAP_READ` companion — populated via
    /// `encoder.copy_buffer_to_buffer` after the compute dispatch
    /// so the CPU can map it without needing STORAGE on the same
    /// buffer (Metal / WebGPU forbid that combo).
    pub staging_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl CursorProbe_Gpu {
    pub fn new(
        device: &wgpu::Device,
        shared_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let zero: CursorProbeRaw = bytemuck::Zeroable::zeroed();
        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cursor_probe_out"),
            contents: bytemuck::bytes_of(&zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cursor_probe_staging"),
            size: CURSOR_PROBE_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let cp_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cursor_probe_out_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cursor_probe_out"),
            layout: &cp_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cursor_probe"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shader_compose::compose("cursor_probe.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cursor_probe"),
            bind_group_layouts: &[shared_bind_group_layout, &cp_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Probe-only overrides: disable the fragment shader's LOD
        // pixel cutoff so the walker always descends to the library
        // leaf, regardless of ray distance. Fragment rendering keeps
        // Nyquist LOD for performance; the probe is a single ray per
        // frame, so making it deep-walk costs ~microseconds and lets
        // edits operate at the user's anchor depth even when the
        // visible surface LOD terminates at a coarser cell.
        let override_constants: [(&str, f64); 2] = [
            ("LOD_PIXEL_THRESHOLD", 0.0),
            ("ENABLE_STATS", 0.0),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cursor_probe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_cursor_probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &override_constants,
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });

        Self {
            pipeline,
            output_buffer,
            staging_buffer,
            bind_group,
        }
    }
}
