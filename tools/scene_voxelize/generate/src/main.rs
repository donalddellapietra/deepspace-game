//! GLB -> `.vxs` voxelizer.
//!
//! Reads glTF/GLB files from `assets/scenes/` (populated by
//! `scripts/fetch-glb-presets.sh`) and writes palette-quantized sparse
//! voxel grids to `assets/scenes/`. The output format is the game's
//! existing `.vxs` (see `src/import/vxs.rs`).
//!
//! Modified from James Catania's voxel-raymarching (MIT); see
//! `tools/scene_voxelize/ATTRIBUTION.md`.

use anyhow::{Context, Result};
use clap::Parser;
use scene_voxelize::{MAX_STORAGE_BUFFER_BINDING_SIZE, MODEL_FILE_EXT, voxelize};
use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

/// Voxelize glTF scenes into the game's `.vxs` sparse voxel format.
#[derive(Parser, Debug)]
#[command()]
struct Args {
    /// Model basenames to voxelize (e.g. `sponza bistro`). If omitted,
    /// voxelizes every `.glb` / `.gltf` found in `--sources-dir`.
    #[arg(short, long)]
    models: Option<Vec<String>>,

    /// Input directory containing source `.glb` / `.gltf` files.
    /// Matches the dir populated by `scripts/fetch-glb-presets.sh`.
    #[arg(long, default_value = "assets/scenes")]
    sources_dir: PathBuf,

    /// Output directory for generated `.vxs` files. Defaults to the
    /// same dir as `--sources-dir` — both `.glb` and `.vxs` live
    /// side-by-side in `assets/scenes/` (gitignored).
    #[arg(long, default_value = "assets/scenes")]
    out_dir: PathBuf,

    /// Voxel scale, in voxels/meter. Larger values generate more
    /// voxels and larger output files.
    #[arg(short, long, default_value_t = 16)]
    scale: u32,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    fs::create_dir_all(&args.out_dir).with_context(|| {
        format!("failed to create output dir {:?}", args.out_dir)
    })?;

    let (device, queue) = init_device().context("failed to initialize GPU context")?;

    let mut sources = walk_asset_sources(&args.sources_dir)
        .with_context(|| format!("error reading sources from {:?}", args.sources_dir))?;

    if let Some(filter) = &args.models {
        sources.retain(|src| filter.iter().any(|n| n.eq_ignore_ascii_case(&src.name)));
    }

    if sources.is_empty() {
        anyhow::bail!(
            "no source GLBs found in {:?} (after filter). Run \
             `scripts/fetch-glb-presets.sh` to download them.",
            args.sources_dir
        );
    }

    eprintln!("Voxelizing {} model(s):", sources.len());
    for src in &sources {
        eprintln!("-    {} {:?}", src.name, src.path);
    }

    for src in &sources {
        voxelize_model(&device, &queue, src, &args.out_dir, args.scale)
            .with_context(|| format!("error voxelizing model {}", src.name))?;
    }

    Ok(())
}

fn init_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

    let mut features = wgpu::Features::default();
    features |= wgpu::Features::FLOAT32_FILTERABLE;
    features |= wgpu::Features::TEXTURE_BINDING_ARRAY;
    features |= wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    features |= wgpu::Features::CLEAR_TEXTURE;
    features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;

    // Start from the adapter's actual caps so we don't ask for more
    // than the device can provide (e.g. Metal caps
    // `max_sampled_textures_per_shader_stage` at 128 while the
    // upstream pipeline optimistically requested 460). Then clamp the
    // couple of limits the voxelize pipeline genuinely needs up to
    // whatever the adapter offers.
    let adapter_limits = adapter.limits();
    let mut limits = adapter_limits.clone();
    limits.max_buffer_size = limits
        .max_buffer_size
        .max(MAX_STORAGE_BUFFER_BINDING_SIZE as u64);
    limits.max_storage_buffer_binding_size = limits
        .max_storage_buffer_binding_size
        .max(MAX_STORAGE_BUFFER_BINDING_SIZE);
    limits.max_compute_invocations_per_workgroup =
        limits.max_compute_invocations_per_workgroup.max(256);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: limits,
        ..Default::default()
    }))?;

    Ok((device, queue))
}

#[derive(Debug)]
struct AssetSource {
    name: String,
    path: PathBuf,
}

fn walk_asset_sources(path: &Path) -> Result<Vec<AssetSource>> {
    let mut sources = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut name_fallback_counter = 0;

    if !path.exists() {
        return Ok(sources);
    }

    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                let e = e.to_ascii_lowercase();
                e == "glb" || e == "gltf"
            })
            .unwrap_or(false);
        if !ext_ok {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|stem| {
                let mut name = stem.to_string_lossy().to_string();
                name = name.to_ascii_lowercase();
                name = name.replace(|c: char| !c.is_alphanumeric(), "_");
                while name.contains("__") {
                    name = name.replace("__", "_");
                }
                if name.chars().next().map_or(false, |c| c.is_numeric()) {
                    name.insert(0, '_');
                }
                if names.contains(&name) {
                    None
                } else {
                    Some(name)
                }
            })
            .unwrap_or_else(|| loop {
                let name = format!("model_{}", name_fallback_counter);
                name_fallback_counter += 1;
                if !names.contains(&name) {
                    names.insert(name.clone());
                    return name;
                }
            });

        names.insert(name.clone());
        sources.push(AssetSource { name, path });
    }

    Ok(sources)
}

fn voxelize_model(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &AssetSource,
    out_dir: &Path,
    voxels_per_unit: u32,
) -> Result<()> {
    let glb = fs::File::open(&src.path)?;
    let mut reader = io::BufReader::new(&glb);
    let model = voxelize(
        &mut reader,
        device,
        queue,
        Some(src.name.clone()),
        voxels_per_unit,
    )
    .context("failed to voxelize model")?;

    let out = out_dir.join(format!("{}.{}", &src.name, MODEL_FILE_EXT));
    let written = model.write_vxs(device, queue, &out)?;

    eprintln!(
        "Completed {} → {:?} ({:.2} MB)",
        src.name,
        out,
        (written as f64) / (1024.0 * 1024.0)
    );

    Ok(())
}
