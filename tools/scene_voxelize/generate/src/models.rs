//! GLB → `.vxs` voxelization.
//!
//! Modified from James Catania's voxel-raymarching (MIT). The upstream
//! pipeline voxelizes into a 4³ sparse brickmap; this fork keeps the
//! voxelize compute pass and k-means palette, but drops the brickmap
//! packer — instead, we read back the raw voxel grid and write the
//! game's `.vxs` sparse format (see `src/import/vxs.rs` for the reader
//! and `tools/scene_voxelize/ATTRIBUTION.md` for provenance).

use std::io::{BufRead, BufWriter, Seek, Write};
use std::path::Path;

use crate::{
    gltf::{self, Gltf, Scene},
    palette::{Palette, linear_rgb_to_oklab, srgb_to_linear},
};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use scene_voxelize_utils::{
    layout::{
        DeviceUtils, sampled_texture, sampler, storage_buffer, storage_texture, uniform_buffer,
    },
    pipeline::PipelineUtils,
};
use wgpu::util::DeviceExt;

const RAW_CHUNK_SIZE: u32 = 64;

/// Voxelize a GLB into a palette-indexed sparse voxel grid.
///
/// `voxels_per_unit` is the voxelization density — for a ~1 m unit
/// glTF this is voxels per meter.
pub fn voxelize<R: BufRead + Seek>(
    src: &mut R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: Option<String>,
    voxels_per_unit: u32,
) -> Result<VoxelModel> {
    let gltf = Gltf::parse(src)?;
    let scene = Scene::from_gltf(&gltf)?;
    voxelize_gltf(device, queue, name, gltf, scene, voxels_per_unit)
}

/// A palette-indexed sparse voxel grid, in memory.
///
/// `voxels` stores only the non-empty cells, one (x, y, z, palette_idx)
/// per entry. `palette` carries sRGB RGBA8 per-palette-entry data,
/// ready to write out to a `.vxs` file.
pub struct VoxelModel {
    pub name: String,
    pub voxels_per_unit: u32,
    pub size: glam::UVec3,
    /// sRGB RGBA8 per palette index. `len() <= Palette::SIZE`.
    pub palette: Vec<[u8; 4]>,
    /// Non-empty voxels as (x, y, z, palette_idx). palette_idx is a
    /// 0-based index into `palette`.
    pub voxels: Vec<(u32, u32, u32, u32)>,
}

impl VoxelModel {
    /// Write this model as `.vxs` (see `src/import/vxs.rs` for the
    /// reader). Returns the number of bytes written.
    pub fn write_vxs(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        path: &Path,
    ) -> Result<usize> {
        let file = std::fs::File::create(path)
            .with_context(|| format!("open {:?} for writing", path))?;
        let mut w = BufWriter::new(&file);

        w.write_all(b"DSVX")?;
        w.write_u32::<LittleEndian>(1)?; // version
        w.write_u32::<LittleEndian>(self.size.x)?;
        w.write_u32::<LittleEndian>(self.size.y)?;
        w.write_u32::<LittleEndian>(self.size.z)?;

        w.write_u32::<LittleEndian>(self.palette.len() as u32)?;
        for rgba in &self.palette {
            w.write_all(rgba)?;
        }

        w.write_u32::<LittleEndian>(self.voxels.len() as u32)?;
        for &(x, y, z, pi) in &self.voxels {
            w.write_u32::<LittleEndian>(x)?;
            w.write_u32::<LittleEndian>(y)?;
            w.write_u32::<LittleEndian>(z)?;
            w.write_u32::<LittleEndian>(pi)?;
        }

        w.flush()?;
        let bytes = file.metadata()?.len() as usize;
        Ok(bytes)
    }
}

struct VoxelizeCtx<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    bg_layouts: &'a BindGroupLayouts,
    scene: &'a Scene,
    gltf: &'a Gltf,
    voxels_per_unit: u32,
}
struct Pipelines {
    voxelize: wgpu::ComputePipeline,
}
struct BindGroupLayouts {
    voxelize_shared: wgpu::BindGroupLayout,
    voxelize_scene_textures: wgpu::BindGroupLayout,
    voxelize_per_primitive: wgpu::BindGroupLayout,
}

fn layouts(device: &wgpu::Device, scene_texture_count: u32) -> (Pipelines, BindGroupLayouts) {
    let bg_layouts = BindGroupLayouts {
        voxelize_shared: device.layout(
            "voxelize_shared",
            wgpu::ShaderStages::COMPUTE,
            (
                uniform_buffer(),
                storage_buffer().read_only(),
                sampler().filtering(),
                storage_texture().r32uint().dimension_3d().read_only(),
                storage_texture().r32uint().dimension_3d().write_only(),
                storage_texture().r32uint().dimension_3d().read_only(),
            ),
        ),
        voxelize_scene_textures: device.layout(
            "voxelize_scene_textures",
            wgpu::ShaderStages::COMPUTE,
            sampled_texture()
                .float()
                .dimension_2d()
                .count(std::num::NonZeroU32::new(scene_texture_count.max(1)).unwrap()),
        ),
        voxelize_per_primitive: device.layout(
            "voxelize_per_primitive",
            wgpu::ShaderStages::COMPUTE,
            (
                uniform_buffer(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
            ),
        ),
    };
    let pipelines = Pipelines {
        voxelize: device
            .compute_pipeline(
                "voxelize",
                &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("voxelize"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("shaders/voxelize.wgsl").into(),
                    ),
                }),
            )
            .layout(&[
                &bg_layouts.voxelize_shared,
                &bg_layouts.voxelize_scene_textures,
                &bg_layouts.voxelize_per_primitive,
            ]),
    };
    (pipelines, bg_layouts)
}

fn voxelize_gltf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: Option<String>,
    gltf: Gltf,
    scene: Scene,
    voxels_per_unit: u32,
) -> Result<VoxelModel> {
    let name = name.unwrap_or_else(|| "model".to_string());

    let (pipelines, bg_layouts) = layouts(device, scene.textures.len() as u32);

    let ctx = VoxelizeCtx {
        device,
        queue,
        bg_layouts: &bg_layouts,
        scene: &scene,
        gltf: &gltf,
        voxels_per_unit,
    };

    let size_unscaled = (scene.max.ceil() - scene.min.floor()).ceil().as_uvec3();
    let base_unscaled = ctx.scene.min.floor();
    let size = size_unscaled * voxels_per_unit;

    // --- pre-pass on CPU: which 64³ chunks of the bounding volume
    // contain any triangle? --------------------------------------------
    let raw_chunk_size = size.map(|x| x.div_ceil(RAW_CHUNK_SIZE));
    let mut raw_chunk_indices = vec![0u32; raw_chunk_size.element_product() as usize];
    let mut raw_chunk_count = 0u32;

    let start = std::time::Instant::now();
    println!("chunk estimation..");
    for node in &ctx.scene.nodes {
        let Some(mesh) = ctx.scene.meshes.get(node.mesh_id) else {
            continue;
        };
        for primitive in &mesh.primitives {
            let indices: &[[u16; 3]] =
                bytemuck::cast_slice(&gltf.bin[primitive.indices.start..primitive.indices.end]);
            // Stride-aware position lookup. glTF allows interleaved
            // vertex attributes (POSITION + NORMAL + … in one buffer);
            // Sponza uses a 48-byte stride. Reading with a raw cast as
            // &[[f32; 3]] on byte-stride-48 data would pull garbage
            // (actually the next attribute's bytes) for every vertex
            // after the first — producing triangles with random
            // positions that rasterize as radial streaks outside the
            // real scene bounds.
            let pos_stride = if primitive.positions.byte_stride == 0 {
                12
            } else {
                primitive.positions.byte_stride as usize
            };
            let pos_base = primitive.positions.start;
            let read_pos = |i: usize| -> glam::Vec3 {
                let off = pos_base + i * pos_stride;
                let bytes: &[u8] = &gltf.bin[off..off + 12];
                let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
                let z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
                glam::vec3(x, y, z)
            };

            for tri in indices {
                let pos = tri
                    .map(|i| read_pos(i as usize))
                    .map(|p| {
                        (node.transform.transform_point3(p) - base_unscaled)
                            * (voxels_per_unit as f32)
                    });
                let mn = pos[0].min(pos[1]).min(pos[2]);
                let mx = pos[0].max(pos[1]).max(pos[2]);

                let mn = mn.floor().max(glam::Vec3::ZERO).as_uvec3();
                let mx = mx.ceil().as_uvec3().min(size);

                let min_chunk = mn / RAW_CHUNK_SIZE;
                let max_chunk = (mx / RAW_CHUNK_SIZE + 1).min(raw_chunk_size);

                for z in min_chunk.z..max_chunk.z {
                    for y in min_chunk.y..max_chunk.y {
                        for x in min_chunk.x..max_chunk.x {
                            let i = (x
                                + y * raw_chunk_size.x
                                + z * raw_chunk_size.x * raw_chunk_size.y)
                                as usize;
                            if raw_chunk_indices[i] == 0 {
                                raw_chunk_indices[i] = (raw_chunk_count << 1) | 1;
                                raw_chunk_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "chunk estimation: {:.2?} ({} chunks occupied)",
        start.elapsed(),
        raw_chunk_count
    );

    // --- allocate GPU textures for voxelize pass ----------------------
    let tex_raw_chunk_indices = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raw_chunk_indices"),
        size: wgpu::Extent3d {
            width: raw_chunk_size.x,
            height: raw_chunk_size.y,
            depth_or_array_layers: raw_chunk_size.z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_raw_chunk_indices,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&raw_chunk_indices),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * raw_chunk_size.x),
            rows_per_image: Some(raw_chunk_size.y),
        },
        wgpu::Extent3d {
            width: raw_chunk_size.x,
            height: raw_chunk_size.y,
            depth_or_array_layers: raw_chunk_size.z,
        },
    );

    // Pack all occupied 64³ chunks into a near-cubic texture. The
    // minor axes form a square whose side is ≈ cbrt(raw_voxel_count);
    // the major axis takes whatever's needed to fit the rest.
    let raw_voxel_count = raw_chunk_count * RAW_CHUNK_SIZE.pow(3);
    let raw_minor_size = ((raw_voxel_count as f64).powf(1.0 / 3.0).ceil() as u32)
        .max(RAW_CHUNK_SIZE)
        .next_multiple_of(RAW_CHUNK_SIZE);
    let raw_major_size = raw_voxel_count
        .div_ceil(raw_minor_size * raw_minor_size)
        .next_multiple_of(RAW_CHUNK_SIZE);

    let tex_raw_voxels = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raw_voxels"),
        size: wgpu::Extent3d {
            width: raw_minor_size,
            height: raw_minor_size,
            depth_or_array_layers: raw_major_size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let raw_voxels_bytes =
        (raw_minor_size as u64) * (raw_minor_size as u64) * (raw_major_size as u64) * 4;

    println!(
        "raw_chunk_indices: {:.2} MiB",
        (raw_chunk_size.element_product() as f64) * 4.0 / (1024.0 * 1024.0)
    );
    println!(
        "raw_voxels:        {:.2} MiB  ({} × {} × {})",
        raw_voxels_bytes as f64 / (1024.0 * 1024.0),
        raw_minor_size,
        raw_minor_size,
        raw_major_size
    );

    // --- palette + LUT (CPU k-means in Oklab; LUT uploaded as 256³) ---
    let palette = build_palette(&ctx);

    // --- bind groups for the voxelize compute pass --------------------
    let bg_voxelize_shared =
        create_bg_voxelize_shared(&ctx, &tex_raw_voxels, &tex_raw_chunk_indices, &palette.lut);
    let bg_voxelize_scene_textures = create_bg_voxelize_scene_textures(&ctx);
    let bg_voxelize_per_primitive = create_bg_voxelize_per_primitive(&ctx);

    // --- run the voxelize pass ----------------------------------------
    let mut encoder = device.create_command_encoder(&Default::default());

    // Zero the voxel texture so unwritten voxels are guaranteed to
    // read back as 0 (our "empty" sentinel).
    encoder.clear_texture(
        &tex_raw_voxels,
        &wgpu::ImageSubresourceRange {
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxelize"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxelize);
        pass.set_bind_group(0, Some(&bg_voxelize_shared), &[]);
        pass.set_bind_group(1, Some(&bg_voxelize_scene_textures), &[]);

        for primitive in &bg_voxelize_per_primitive {
            pass.set_bind_group(2, Some(&primitive.bg), &[]);
            let tris = primitive.index_count / 3;
            pass.dispatch_workgroups(tris.div_ceil(64), 1, 1);
        }
    }

    // --- copy the raw voxels texture to a MAP_READ buffer --------------
    //
    // `raw_chunk_indices` is already on CPU from the pre-pass — the
    // shader only reads that texture, so no need to round-trip it.
    let voxels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raw_voxels_readback"),
        size: raw_voxels_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_raw_voxels,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &voxels_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * raw_minor_size),
                rows_per_image: Some(raw_minor_size),
            },
        },
        wgpu::Extent3d {
            width: raw_minor_size,
            height: raw_minor_size,
            depth_or_array_layers: raw_major_size,
        },
    );

    queue.submit([encoder.finish()]);

    // --- await mapping, scan, emit sparse voxels ----------------------
    voxels_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let voxels_raw: Vec<u32> = {
        let data = voxels_buffer.slice(..).get_mapped_range();
        bytemuck::cast_slice::<u8, u32>(&data).to_vec()
    };

    let raw_chunks_bds = glam::uvec3(
        raw_minor_size / RAW_CHUNK_SIZE,
        raw_minor_size / RAW_CHUNK_SIZE,
        raw_major_size / RAW_CHUNK_SIZE,
    );

    let scan_start = std::time::Instant::now();
    let voxels = scan_voxels(
        &raw_chunk_indices,
        raw_chunk_size,
        &voxels_raw,
        raw_minor_size,
        raw_chunks_bds,
        size,
    );
    println!(
        "voxel scan: {:.2?} ({} non-empty voxels)",
        scan_start.elapsed(),
        voxels.len()
    );

    // --- palette to sRGB RGBA8 ----------------------------------------
    let palette_rgba8: Vec<[u8; 4]> = palette
        .rgba
        .iter()
        .map(|c| {
            let srgb = linear_to_srgb(glam::vec3(c.x, c.y, c.z));
            let r = (srgb.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let g = (srgb.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let b = (srgb.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let a = (c.w.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            [r, g, b, a]
        })
        .collect();

    Ok(VoxelModel {
        name,
        voxels_per_unit,
        size,
        palette: palette_rgba8,
        voxels,
    })
}

/// Walk every occupied chunk in `raw_chunk_indices` and pull its 64³
/// voxels out of the packed `raw_voxels` buffer. Empty voxels (`== 0`)
/// and voxels past the model bounding box are skipped.
fn scan_voxels(
    chunk_indices: &[u32],
    raw_chunk_size: glam::UVec3,
    voxels_raw: &[u32],
    raw_minor_size: u32,
    raw_chunks_bds: glam::UVec3,
    size: glam::UVec3,
) -> Vec<(u32, u32, u32, u32)> {
    let mut out = Vec::with_capacity(1 << 20);

    for cz in 0..raw_chunk_size.z {
        for cy in 0..raw_chunk_size.y {
            for cx in 0..raw_chunk_size.x {
                let idx = (cx + cy * raw_chunk_size.x + cz * raw_chunk_size.x * raw_chunk_size.y)
                    as usize;
                let entry = chunk_indices[idx];
                if (entry & 1) == 0 {
                    continue;
                }
                let ci = entry >> 1;

                let chunk_offset = glam::uvec3(
                    ci % raw_chunks_bds.x,
                    (ci / raw_chunks_bds.x) % raw_chunks_bds.y,
                    ci / (raw_chunks_bds.x * raw_chunks_bds.y),
                ) * RAW_CHUNK_SIZE;

                let chunk_world = glam::uvec3(cx, cy, cz) * RAW_CHUNK_SIZE;

                for lz in 0..RAW_CHUNK_SIZE {
                    for ly in 0..RAW_CHUNK_SIZE {
                        for lx in 0..RAW_CHUNK_SIZE {
                            let wx = chunk_world.x + lx;
                            let wy = chunk_world.y + ly;
                            let wz = chunk_world.z + lz;
                            if wx >= size.x || wy >= size.y || wz >= size.z {
                                continue;
                            }
                            let tx = chunk_offset.x + lx;
                            let ty = chunk_offset.y + ly;
                            let tz = chunk_offset.z + lz;
                            let ti = (tx
                                + ty * raw_minor_size
                                + tz * raw_minor_size * raw_minor_size)
                                as usize;
                            let packed = voxels_raw[ti];
                            if packed == 0 {
                                continue;
                            }
                            // voxelize.wgsl packs the voxel as
                            // `(1 << 31) | palette_index`. The high
                            // bit is the presence flag (we cleared
                            // the texture to 0 pre-pass, so 0 means
                            // "nothing written"); palette lives in
                            // bits 0-30.
                            let pi = packed & 0x7FFF_FFFF;
                            out.push((wx, wy, wz, pi));
                        }
                    }
                }
            }
        }
    }

    out
}

fn linear_to_srgb(linear: glam::Vec3) -> glam::Vec3 {
    linear.map(|x| {
        if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    })
}

fn create_bg_voxelize_shared(
    ctx: &'_ VoxelizeCtx<'_>,
    tex_raw_voxels: &wgpu::Texture,
    tex_raw_chunk_indices: &wgpu::Texture,
    tex_palette_lut: &wgpu::Texture,
) -> wgpu::BindGroup {
    let base_unscaled = ctx.scene.min.floor().as_ivec3();
    let size_unscaled = (ctx.scene.max.ceil() - ctx.scene.min.floor())
        .ceil()
        .as_ivec3();

    #[repr(C)]
    #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SceneBufferEntry {
        base: glam::Vec4,
        size: glam::Vec3,
        scale: f32,
    }
    let buffer_scene = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer_scene"),
            contents: bytemuck::cast_slice(&[SceneBufferEntry {
                base: base_unscaled.as_vec3().extend(0.0),
                size: size_unscaled.as_vec3(),
                scale: ctx.voxels_per_unit as f32,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    #[repr(C)]
    #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct MaterialBufferEntry {
        base_albedo: glam::Vec4,
        base_metallic: f32,
        base_roughness: f32,
        normal_scale: f32,
        albedo_index: i32,
        normal_index: i32,
        metallic_roughness_index: i32,
        emissive_index: i32,
        double_sided: u32,
        is_emissive: u32,
        _pad: [f32; 3],
        emissive_factor: glam::Vec3,
        emissive_intensity: f32,
    }
    let buffer_materials = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material data storage buffer"),
            contents: bytemuck::cast_slice(
                &ctx.scene
                    .materials
                    .iter()
                    .map(|mat| MaterialBufferEntry {
                        base_albedo: mat.base_albedo,
                        base_metallic: mat.base_metallic,
                        base_roughness: mat.base_roughness,
                        normal_scale: mat.normal_scale,
                        albedo_index: mat.albedo_index,
                        normal_index: mat.normal_index,
                        metallic_roughness_index: mat.metallic_roughness_index,
                        emissive_index: mat.emissive_index,
                        double_sided: mat.double_sided as u32,
                        is_emissive: mat.is_emissive as u32,
                        _pad: [0.0; 3],
                        emissive_factor: mat.emissive_factor,
                        emissive_intensity: mat.emissive_intensity,
                    })
                    .collect::<Vec<MaterialBufferEntry>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxelize_shared"),
        layout: &ctx.bg_layouts.voxelize_shared,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_scene.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_materials.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_chunk_indices.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_voxels.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(
                    &tex_palette_lut.create_view(&Default::default()),
                ),
            },
        ],
    })
}

fn create_bg_voxelize_scene_textures(ctx: &'_ VoxelizeCtx<'_>) -> wgpu::BindGroup {
    let texture_views = ctx
        .scene
        .textures
        .iter()
        .map(|tex| {
            let (width, height) = tex.data.dimensions();

            // Note that the textures are are all created as Rgba8Unorm, despite the albedo ones being in Srgb per gltf spec
            //
            // Since in the voxelize pipeline I use the palette LUT to grab the palette index, I can just index the LUT with Srgb
            // instead of linear, which lets us keep best-possible precision without inflating the palette beyond 256^3
            //
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: match tex.encoding {
                    gltf::scene::TextureEncoding::Linear => wgpu::TextureFormat::Rgba8Unorm,
                    gltf::scene::TextureEncoding::Srgb => wgpu::TextureFormat::Rgba8Unorm,
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &tex.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        })
        .collect::<Vec<wgpu::TextureView>>();

    // binding_array wants at least one element even if the scene has
    // no textures — create a 1×1 white placeholder if empty.
    if texture_views.is_empty() {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("placeholder_white"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        return ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene textures bind group (placeholder)"),
            layout: &ctx.bg_layouts.voxelize_scene_textures,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&[&view]),
            }],
        });
    }

    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene textures bind group"),
        layout: &ctx.bg_layouts.voxelize_scene_textures,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureViewArray(
                &texture_views.iter().collect::<Vec<&wgpu::TextureView>>(),
            ),
        }],
    })
}

struct PrimitiveGroup {
    bg: wgpu::BindGroup,
    index_count: u32,
}

fn create_bg_voxelize_per_primitive(ctx: &'_ VoxelizeCtx<'_>) -> Vec<PrimitiveGroup> {
    let mut res = Vec::new();
    for node in &ctx.scene.nodes {
        let Some(mesh) = ctx.scene.meshes.get(node.mesh_id) else {
            continue;
        };
        for primitive in &mesh.primitives {
            // each (object, primitive) pair in the scene gets its own bind group
            // inefficient but idc its just a generation step for now

            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PrimitiveBufferEntry {
                matrix: glam::Mat4,
                normal_matrix: [[f32; 4]; 3],
                material_id: u32,
                index_count: u32,
                _pad: [f32; 2],
            }
            let buffer_primitive =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive data uniform buffer"),
                        contents: bytemuck::cast_slice(&[PrimitiveBufferEntry {
                            matrix: node.transform,
                            normal_matrix: glam::Mat3::from_mat4(node.transform.inverse())
                                .transpose()
                                .to_cols_array_2d()
                                .map(|v| [v[0], v[1], v[2], 0.0]),
                            material_id: primitive.material_id,
                            index_count: primitive.indices.count,
                            _pad: [0.0; 2],
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

            let indices_u32 =
                bytemuck::cast_slice(&ctx.gltf.bin[primitive.indices.start..primitive.indices.end])
                    .iter()
                    .map(|idx: &u16| *idx as u32)
                    .collect::<Vec<u32>>();
            let buffer_indices = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive indices data"),
                    contents: match primitive.indices.component_type {
                        gltf::schema::ComponentType::UnsignedShort => {
                            &bytemuck::cast_slice(&indices_u32)
                        }
                        _ => &ctx.gltf.bin[primitive.indices.start..primitive.indices.end],
                    },
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            // De-interleave vertex attributes if the source has a
            // `byteStride` set. The voxelize shader reads each
            // attribute as `array<f32>` indexed by `vertex*N + k` (so
            // it expects tight-packed data); feeding it a strided
            // source buffer would read garbage (the adjacent
            // attribute's bytes) for every vertex after the first.
            let gather_tight = |desc: &gltf::scene::PrimitiveBufferDescriptor,
                                element_size: usize|
             -> Vec<u8> {
                let stride = if desc.byte_stride == 0 {
                    element_size
                } else {
                    desc.byte_stride as usize
                };
                if stride == element_size {
                    // Already tight-packed — zero-copy slice.
                    ctx.gltf.bin[desc.start..desc.start + element_size * desc.count as usize]
                        .to_vec()
                } else {
                    let mut out = Vec::with_capacity(element_size * desc.count as usize);
                    for v in 0..desc.count as usize {
                        let off = desc.start + v * stride;
                        out.extend_from_slice(&ctx.gltf.bin[off..off + element_size]);
                    }
                    out
                }
            };
            let positions_tight = gather_tight(&primitive.positions, 12); // VEC3 f32
            let normals_tight = gather_tight(&primitive.normals, 12); // VEC3 f32
            let tangents_tight = gather_tight(&primitive.tangents, 16); // VEC4 f32
            let uv_tight = gather_tight(&primitive.uv, 8); // VEC2 f32

            let buffer_positions =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex position data"),
                        contents: &positions_tight,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
            let buffer_normals = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex normal data"),
                    contents: &normals_tight,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            let buffer_tangents =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex tangent data"),
                        contents: &tangents_tight,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
            let buffer_uv = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex uv data"),
                    contents: &uv_tight,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

            res.push(PrimitiveGroup {
                bg: ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("per primitive voxelize bind group"),
                    layout: &ctx.bg_layouts.voxelize_per_primitive,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer_primitive.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffer_indices.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffer_positions.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buffer_normals.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: buffer_tangents.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: buffer_uv.as_entire_binding(),
                        },
                    ],
                }),
                index_count: primitive.indices.count,
            });
        }
    }
    res
}

struct PaletteData {
    rgba: Vec<glam::Vec4>,
    lut: wgpu::Texture,
}

fn build_palette(ctx: &'_ VoxelizeCtx<'_>) -> PaletteData {
    const TARGET_SAMPLE_COUNT: usize = 2_000_000;

    // Random sampling from scene textures in Oklab space. Weighted
    // equally across textures.
    let per_texture = if ctx.scene.textures.is_empty() {
        0
    } else {
        TARGET_SAMPLE_COUNT / ctx.scene.textures.len()
    };

    let mut samples =
        Vec::with_capacity(per_texture * ctx.scene.textures.len().max(1));
    let mut rng = SmallRng::seed_from_u64(0);

    for texture in &ctx.scene.textures {
        for _ in 0..per_texture {
            let x = rng.next_u32() % texture.data.width();
            let y = rng.next_u32() % texture.data.height();
            let sample = texture.data.get_pixel(x, y).0;
            let srgb = glam::u8vec3(sample[0], sample[1], sample[2]).as_vec3() / 255.0;
            let rgb = srgb_to_linear(srgb);
            let lab = linear_rgb_to_oklab(rgb);
            samples.push(lab);
        }
    }

    // If the scene has no textures, seed from material base albedos so
    // k-means has something to cluster on.
    if samples.is_empty() {
        for mat in &ctx.scene.materials {
            let srgb =
                glam::vec3(mat.base_albedo.x, mat.base_albedo.y, mat.base_albedo.z);
            let rgb = srgb_to_linear(srgb);
            let lab = linear_rgb_to_oklab(rgb);
            for _ in 0..128 {
                samples.push(lab);
            }
        }
    }

    if samples.is_empty() {
        samples.push(linear_rgb_to_oklab(glam::Vec3::splat(0.5)));
    }

    let palette = Palette::from_samples(&mut samples);

    let lut = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("palette_lut"),
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 256,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &lut,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&palette.lut),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * 256),
            rows_per_image: Some(256),
        },
        wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 256,
        },
    );

    PaletteData {
        rgba: palette.rgba,
        lut,
    }
}
