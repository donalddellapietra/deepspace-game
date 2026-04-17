//! Parse the custom sparse voxel format (`.vxs`) produced by
//! `tools/glb_to_vox.py` for models exceeding `.vox`'s 256-per-axis
//! limit.
//!
//! Layout:
//!
//! ```text
//! magic:        b"DSVX"              (4 bytes)
//! version:      u32 (=1)
//! size_x, y, z: u32 × 3
//! palette_n:    u32
//! palette:      [u8; 4] × palette_n
//! voxel_n:      u32
//! voxels:       (u32 x, u32 y, u32 z, u32 palette_idx) × voxel_n
//! ```
//!
//! Palette indices are 0-based; our internal `VoxelModel::data` is
//! 1-based (0 = empty), so we add 1 to each index when building the
//! dense grid. The palette is registered into `ColorRegistry` and an
//! index LUT maps `.vxs` palette → registry index.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::world::palette::ColorRegistry;
use super::VoxelModel;

pub fn load(path: &Path, registry: &mut ColorRegistry) -> Result<VoxelModel, String> {
    let mut file = File::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"DSVX" {
        return Err(format!(".vxs bad magic: {:?}", magic));
    }
    let version = read_u32(&mut file)?;
    if version != 1 {
        return Err(format!(".vxs unsupported version: {}", version));
    }
    let size_x = read_u32(&mut file)? as usize;
    let size_y = read_u32(&mut file)? as usize;
    let size_z = read_u32(&mut file)? as usize;

    let palette_n = read_u32(&mut file)? as usize;
    let mut lut = vec![0u8; palette_n];
    for i in 0..palette_n {
        let mut rgba = [0u8; 4];
        file.read_exact(&mut rgba).map_err(|e| e.to_string())?;
        if rgba[3] == 0 {
            lut[i] = 0;
        } else if let Some(idx) = registry.register(rgba[0], rgba[1], rgba[2], rgba[3]) {
            lut[i] = idx;
        }
    }

    let voxel_n = read_u32(&mut file)? as usize;
    let total_cells = size_x
        .checked_mul(size_y)
        .and_then(|v| v.checked_mul(size_z))
        .ok_or_else(|| format!(".vxs dims overflow: {}x{}x{}", size_x, size_y, size_z))?;
    let mut data = vec![0u8; total_cells];

    // Read voxels in a single bulk read (16 bytes each) and dispatch.
    let bytes_needed = voxel_n
        .checked_mul(16)
        .ok_or_else(|| format!(".vxs voxel count overflow: {}", voxel_n))?;
    let mut buf = vec![0u8; bytes_needed];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    for i in 0..voxel_n {
        let off = i * 16;
        let x = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) as usize;
        let y = u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]) as usize;
        let z = u32::from_le_bytes([buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11]]) as usize;
        let pi = u32::from_le_bytes([buf[off + 12], buf[off + 13], buf[off + 14], buf[off + 15]]) as usize;
        if x >= size_x || y >= size_y || z >= size_z { continue; }
        let reg = if pi < palette_n { lut[pi] } else { 0 };
        if reg == 0 { continue; }
        let idx = (z * size_y + y) * size_x + x;
        data[idx] = reg;
    }

    Ok(VoxelModel { size_x, size_y, size_z, data })
}

fn read_u32(f: &mut File) -> Result<u32, String> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

// Suppress unused-import warnings when this module is tested in isolation.
#[allow(dead_code)]
fn _seek_unused(_f: &mut File) { let _ = SeekFrom::Start(0); }
