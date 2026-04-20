//! Minecraft → VXS converter.
//!
//! Reads a `.litematic` file, maps each block to an RGBA via the
//! canonical Minecraft map-color palette, and writes a `.vxs` file
//! consumable by `src/import/vxs.rs::load_sparse`.
//!
//! Usage:
//! ```text
//! mc_voxelize <input.litematic> <output.vxs>
//! ```
//!
//! Format: the output is the exact DSVX v1 layout documented at the
//! top of `src/import/vxs.rs` — magic + dims + local RGBA palette +
//! sparse (x,y,z,palette_idx) records. Coordinates are Y-up in both
//! formats, so no axis swap is needed.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rustmatica::Litematic;

mod block_map;
use block_map::{AIR, Rgba, UNKNOWN};

/// One non-empty voxel in the output grid (coords shifted to 0-origin,
/// palette index already resolved).
struct Voxel {
    x: u32,
    y: u32,
    z: u32,
    pal_idx: u32,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: mc_voxelize <input.litematic> <output.vxs>");
        std::process::exit(2);
    }
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);

    let lit: Litematic = Litematic::read_file(input)
        .with_context(|| format!("reading litematic {:?}", input))?;

    if lit.regions.is_empty() {
        return Err(anyhow!("litematic has no regions"));
    }

    // Pass 1: gather all occupied voxels in global coords (the union
    // across regions), tracking the overall bounding box.
    let mut occupied: Vec<(i32, i32, i32, Rgba)> = Vec::new();
    let mut unknown_names: HashMap<String, u64> = HashMap::new();
    let mut known_count: u64 = 0;
    let mut air_count: u64 = 0;

    let (mut min_x, mut min_y, mut min_z) = (i32::MAX, i32::MAX, i32::MAX);
    let (mut max_x, mut max_y, mut max_z) = (i32::MIN, i32::MIN, i32::MIN);

    for region in &lit.regions {
        // `size` can be negative — it encodes a direction. `blocks()`
        // already returns local coords in [0, |size|), so we just
        // translate by `region.position` (the region's anchor in
        // global space). For negative size axes, the region's anchor
        // is at the "max" corner; correcting that to the min corner
        // is what BlockPos::min_global_{x,y,z} does, so use those.
        let gx0 = region.min_global_x();
        let gy0 = region.min_global_y();
        let gz0 = region.min_global_z();

        for (pos, block) in region.blocks() {
            let name: &str = &block.name;
            let rgba = block_map::block_rgba(name);
            if rgba == AIR {
                air_count += 1;
                continue;
            }
            if rgba == UNKNOWN {
                *unknown_names.entry(name.to_string()).or_insert(0) += 1;
            } else {
                known_count += 1;
            }
            let gx = gx0 + pos.x;
            let gy = gy0 + pos.y;
            let gz = gz0 + pos.z;
            min_x = min_x.min(gx);
            min_y = min_y.min(gy);
            min_z = min_z.min(gz);
            max_x = max_x.max(gx);
            max_y = max_y.max(gy);
            max_z = max_z.max(gz);
            occupied.push((gx, gy, gz, rgba));
        }
    }

    if occupied.is_empty() {
        return Err(anyhow!("no non-air blocks found in litematic"));
    }

    // Pass 2: shift to 0-origin, dedup palette.
    let size_x = (max_x - min_x + 1) as u32;
    let size_y = (max_y - min_y + 1) as u32;
    let size_z = (max_z - min_z + 1) as u32;

    let mut palette: Vec<Rgba> = Vec::new();
    let mut palette_index: HashMap<Rgba, u32> = HashMap::new();
    let mut voxels: Vec<Voxel> = Vec::with_capacity(occupied.len());

    for (gx, gy, gz, rgba) in occupied {
        let pal_idx = *palette_index.entry(rgba).or_insert_with(|| {
            let i = palette.len() as u32;
            palette.push(rgba);
            i
        });
        voxels.push(Voxel {
            x: (gx - min_x) as u32,
            y: (gy - min_y) as u32,
            z: (gz - min_z) as u32,
            pal_idx,
        });
    }

    // Report stats so the user can see what happened.
    let unknown_total: u64 = unknown_names.values().sum();
    println!(
        "converted {:?} → {:?}",
        input, output
    );
    println!(
        "  bounds: {}×{}×{} (min=({},{},{}) max=({},{},{}))",
        size_x, size_y, size_z,
        min_x, min_y, min_z,
        max_x, max_y, max_z
    );
    println!(
        "  voxels: {} emitted ({} mapped, {} unknown→magenta), {} air skipped",
        voxels.len(), known_count, unknown_total, air_count
    );
    println!("  palette: {} unique RGBA entries", palette.len());
    if !unknown_names.is_empty() {
        let mut entries: Vec<_> = unknown_names.iter().collect();
        entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let show = entries.len().min(10);
        println!("  top unknowns:");
        for (name, count) in entries.iter().take(show) {
            println!("    {:>8}  {}", count, name);
        }
        if entries.len() > show {
            println!("    ... ({} more distinct unknowns)", entries.len() - show);
        }
    }

    // Pass 3: write DSVX v1 binary.
    write_dsvx(output, size_x, size_y, size_z, &palette, &voxels)
        .with_context(|| format!("writing {:?}", output))?;

    Ok(())
}

/// Emit the exact layout documented in `src/import/vxs.rs`:
///
/// ```text
/// magic:    b"DSVX"           (4 B)
/// version:  u32 = 1
/// size:     u32 × 3           (x, y, z)
/// pal_n:    u32
/// palette:  [u8; 4] × pal_n
/// vox_n:    u32
/// voxels:   (u32 x, u32 y, u32 z, u32 pal_idx) × vox_n
/// ```
fn write_dsvx(
    out: &Path,
    size_x: u32,
    size_y: u32,
    size_z: u32,
    palette: &[Rgba],
    voxels: &[Voxel],
) -> Result<()> {
    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).ok();
        }
    }
    let f = File::create(out)?;
    let mut w = BufWriter::new(f);
    w.write_all(b"DSVX")?;
    w.write_all(&1u32.to_le_bytes())?;
    w.write_all(&size_x.to_le_bytes())?;
    w.write_all(&size_y.to_le_bytes())?;
    w.write_all(&size_z.to_le_bytes())?;
    w.write_all(&(palette.len() as u32).to_le_bytes())?;
    for rgba in palette {
        w.write_all(rgba)?;
    }
    w.write_all(&(voxels.len() as u32).to_le_bytes())?;
    for v in voxels {
        w.write_all(&v.x.to_le_bytes())?;
        w.write_all(&v.y.to_le_bytes())?;
        w.write_all(&v.z.to_le_bytes())?;
        w.write_all(&v.pal_idx.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
