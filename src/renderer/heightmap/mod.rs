//! GPU heightmap for entity ground-collision.
//!
//! Stage 1 of the heightmap plan (see
//! `docs/design/heightmap-collisions.md`):
//!
//! - A compute shader (`heightmap_gen.wgsl`) walks the voxel tree
//!   and writes the top-of-ground Y into a 2D R32F texture.
//!   One texel per collision cell (base-3 aligned to the tree).
//! - Resolution = `3^(collision_depth - frame_depth)` per axis,
//!   where `collision_depth = entity_anchor_depth + 1` (one layer
//!   finer than the entity's own anchor).
//!
//! Entity-side consumption lives in `clamp.rs` — a second compute
//! shader that samples this texture per instance and patches the
//! Y component of the raster instance buffer.
//!
//! The heightmap is rebuilt lazily:
//!
//! - On frame-root change (zoom, teleport, ribbon pop) — full
//!   rebuild.
//! - On tree edit inside the current frame — partial rebuild (not
//!   implemented in stage 1; stage 1 does a full rebuild on any
//!   edit).
//! - Otherwise persists across frames.
//!
//! The texture lives on the renderer and outlives individual
//! frames; the compute pipeline itself is stateless.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub mod clamp;
pub mod generate;

pub use clamp::{ClampUniforms, EntityHeightmapClamp, CLAMP_WORKGROUP_SIZE};
pub use generate::HeightmapGen;

/// Texel format for the heightmap texture. R32F gives us enough
/// precision for world-Y values in the `[0, 3)` frame range and
/// below (we use a large-negative sentinel for "no ground").
pub const HEIGHTMAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Sentinel value the shader writes when a column has no solid
/// content anywhere. Matches `GROUND_NONE` in `heightmap_gen.wgsl`.
/// Use `is_no_ground(value)` instead of direct equality to absorb
/// any tiny round-trip error.
pub const GROUND_NONE: f32 = -1.0e30;

#[inline]
pub fn is_no_ground(y: f32) -> bool {
    y < -1.0e20
}

/// CPU mirror of `HeightmapUniforms` in `heightmap_gen.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct HeightmapUniforms {
    pub frame_root_bfs: u32,
    pub frame_depth: u32,
    pub side: u32,
    pub delta: u32,
    pub y_origin: f32,
    pub y_size: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl HeightmapUniforms {
    /// Build uniforms for a heightmap of `side = 3^delta` texels
    /// per axis, rooted at the given frame. `y_origin` + `y_size`
    /// describe the frame cell's extent in world-Y (for a Cartesian
    /// frame that's `[0, WORLD_SIZE)` = `[0, 3)`).
    pub fn new(
        frame_root_bfs: u32,
        frame_depth: u32,
        delta: u32,
        y_origin: f32,
        y_size: f32,
    ) -> Self {
        let side = 3u32.pow(delta);
        Self {
            frame_root_bfs,
            frame_depth,
            side,
            delta,
            y_origin,
            y_size,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Owned GPU heightmap resources. Holds the texture, its view, and
/// a tiny uniforms buffer. Recreated when `delta` changes (a new
/// resolution = a new texture allocation); in-place updates when
/// only the frame root / Y range shifts.
pub struct HeightmapTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub uniforms: wgpu::Buffer,
    pub side: u32,
    pub delta: u32,
}

impl HeightmapTexture {
    /// Allocate a fresh heightmap texture at `side = 3^delta`
    /// resolution. Call when delta changes or for the initial
    /// creation. The uniforms buffer is created initialized to
    /// a valid-but-trivial configuration so a dispatch before any
    /// `write_uniforms` call is still well-defined.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, delta: u32) -> Self {
        let side = 3u32.pow(delta);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("heightmap"),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HEIGHTMAP_FORMAT,
            // STORAGE_BINDING: written by heightmap_gen.
            // TEXTURE_BINDING: read by entity_heightmap_clamp (as a
            //   sampled texture, since WebGPU baseline doesn't
            //   allow read-only storage textures).
            // COPY_SRC: readback in tests + dump paths.
            // COPY_DST: synthetic uploads in tests.
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let default_uniforms = HeightmapUniforms::new(0, 0, delta, 0.0, 3.0);
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("heightmap_uniforms"),
            contents: bytemuck::bytes_of(&default_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let _ = queue;
        Self { texture, view, uniforms, side, delta }
    }

    /// Push fresh uniforms to the GPU. Validates that `u.delta`
    /// still matches the allocated texture — callers should
    /// reallocate via `new` on delta changes.
    pub fn write_uniforms(&self, queue: &wgpu::Queue, u: &HeightmapUniforms) {
        debug_assert_eq!(u.delta, self.delta, "delta mismatch — reallocate texture");
        debug_assert_eq!(u.side, self.side, "side mismatch — reallocate texture");
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(u));
    }
}

/// Block size of the gen compute shader's workgroup. Matches the
/// `@workgroup_size(9, 9, 1)` in `heightmap_gen.wgsl`.
pub const GEN_WORKGROUP_SIDE: u32 = 9;
