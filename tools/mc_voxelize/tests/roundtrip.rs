//! End-to-end: run the built `mc_voxelize` binary on the bundled
//! sample litematics, then re-parse the emitted DSVX bytes and
//! assert the header + voxel count match the stats we print.
//!
//! This catches any drift between the writer and the format
//! documented in `src/import/vxs.rs` — the main-game loader reads
//! exactly these bytes, so if the roundtrip passes here it parses
//! cleanly in the game.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run_bin(input: &Path, output: &Path) {
    let bin = env!("CARGO_BIN_EXE_mc_voxelize");
    let status = Command::new(bin)
        .arg(input)
        .arg(output)
        .status()
        .expect("spawn mc_voxelize");
    assert!(status.success(), "mc_voxelize failed on {:?}", input);
}

fn parse_dsvx(path: &Path) -> (u32, u32, u32, u32, u32, Vec<u8>, Vec<(u32, u32, u32, u32)>) {
    let data = fs::read(path).expect("read .vxs");
    assert!(data.len() >= 24, "short file");
    assert_eq!(&data[0..4], b"DSVX", "bad magic");
    let u32_at = |off: usize| {
        u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
    };
    let version = u32_at(4);
    let sx = u32_at(8);
    let sy = u32_at(12);
    let sz = u32_at(16);
    let pal_n = u32_at(20);
    let pal_bytes = (pal_n as usize) * 4;
    let palette = data[24..24 + pal_bytes].to_vec();
    let vox_n = u32_at(24 + pal_bytes);
    let voxels_start = 24 + pal_bytes + 4;
    let mut voxels = Vec::with_capacity(vox_n as usize);
    for i in 0..vox_n as usize {
        let o = voxels_start + i * 16;
        let x = u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        let y = u32::from_le_bytes([data[o + 4], data[o + 5], data[o + 6], data[o + 7]]);
        let z = u32::from_le_bytes([data[o + 8], data[o + 9], data[o + 10], data[o + 11]]);
        let p = u32::from_le_bytes([data[o + 12], data[o + 13], data[o + 14], data[o + 15]]);
        voxels.push((x, y, z, p));
    }
    let expected_size = voxels_start + vox_n as usize * 16;
    assert_eq!(
        data.len(),
        expected_size,
        "file size {} doesn't match header (expected {})",
        data.len(),
        expected_size
    );
    assert_eq!(version, 1, "version");
    (sx, sy, sz, pal_n, vox_n, palette, voxels)
}

fn assert_voxels_in_bounds(
    sx: u32,
    sy: u32,
    sz: u32,
    pal_n: u32,
    voxels: &[(u32, u32, u32, u32)],
) {
    for (x, y, z, p) in voxels {
        assert!(*x < sx, "x {} >= size_x {}", x, sx);
        assert!(*y < sy, "y {} >= size_y {}", y, sy);
        assert!(*z < sz, "z {} >= size_z {}", z, sz);
        assert!(*p < pal_n, "palette index {} >= pal_n {}", p, pal_n);
    }
}

fn assert_no_dup_coords(voxels: &[(u32, u32, u32, u32)]) {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for (x, y, z, _) in voxels {
        assert!(seen.insert((*x, *y, *z)), "duplicate voxel at {:?}", (x, y, z));
    }
}

#[test]
fn axolotl_roundtrip() {
    let dir = crate_dir();
    let input = dir.join("test_data/axolotl.litematic");
    let output = dir.join("tmp/axolotl_test.vxs");
    fs::create_dir_all(output.parent().unwrap()).ok();
    run_bin(&input, &output);

    let (sx, sy, sz, pal_n, vox_n, palette, voxels) = parse_dsvx(&output);
    // axolotl is the rustmatica sample: a 3x2x3 block including water
    // and 2 unknowns (chest + oak_wall_sign).
    assert_eq!((sx, sy, sz), (3, 2, 3));
    assert_eq!(vox_n, 18, "18 non-air voxels expected");
    assert!(pal_n >= 2, "at least stone + water palette entries");
    assert_eq!(palette.len(), (pal_n as usize) * 4);
    assert_voxels_in_bounds(sx, sy, sz, pal_n, &voxels);
    assert_no_dup_coords(&voxels);
}

#[test]
fn donut_roundtrip() {
    let dir = crate_dir();
    let input = dir.join("test_data/donut.litematic");
    let output = dir.join("tmp/donut_test.vxs");
    fs::create_dir_all(output.parent().unwrap()).ok();
    run_bin(&input, &output);

    let (sx, sy, sz, pal_n, vox_n, _palette, voxels) = parse_dsvx(&output);
    assert_eq!((sx, sy, sz), (3, 3, 3));
    assert_eq!(vox_n, 24, "24 non-air voxels expected");
    // donut is made of 7 distinct ore/metal blocks + waxed_copper
    // (unknown → magenta), so ≥ 7 palette entries.
    assert!(pal_n >= 7, "got {} palette entries", pal_n);
    assert_voxels_in_bounds(sx, sy, sz, pal_n, &voxels);
    assert_no_dup_coords(&voxels);
}
