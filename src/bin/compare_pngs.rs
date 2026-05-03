//! Pixel-level PNG comparison tool.
//!
//! Usage: compare_pngs <a.png> <b.png>
//!
//! Reads two PNGs, expects same dimensions, reports:
//! - identical: byte-for-byte pixel match
//! - pixel_diff_count: number of pixels whose RGB differs
//! - pixel_diff_fraction: percentage of pixels that differ
//! - max_channel_diff: largest per-channel absolute difference
//! - mean_channel_diff: average per-channel absolute difference over
//!   differing pixels
//!
//! Exit code:
//!   0 = identical
//!   1 = differ
//!   2 = error (can't read, size mismatch, etc.)

use std::fs::File;
use std::io::BufReader;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: compare_pngs <a.png> <b.png>");
        std::process::exit(2);
    }
    let a = match read_rgba(&args[0]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error reading {}: {e}", args[0]);
            std::process::exit(2);
        }
    };
    let b = match read_rgba(&args[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error reading {}: {e}", args[1]);
            std::process::exit(2);
        }
    };
    if a.w != b.w || a.h != b.h {
        eprintln!(
            "size mismatch: {} is {}x{}, {} is {}x{}",
            args[0], a.w, a.h, args[1], b.w, b.h,
        );
        std::process::exit(2);
    }
    let total = (a.w as usize) * (a.h as usize);
    let mut diff_pixels = 0usize;
    let mut max_ch = 0u8;
    let mut sum_ch: u64 = 0;
    let mut sum_ch_count: u64 = 0;
    for i in 0..total {
        let off = i * 4;
        let r_diff = (a.px[off] as i32 - b.px[off] as i32).unsigned_abs() as u8;
        let g_diff = (a.px[off + 1] as i32 - b.px[off + 1] as i32).unsigned_abs() as u8;
        let b_diff = (a.px[off + 2] as i32 - b.px[off + 2] as i32).unsigned_abs() as u8;
        let any = r_diff | g_diff | b_diff;
        if any != 0 {
            diff_pixels += 1;
            max_ch = max_ch.max(r_diff).max(g_diff).max(b_diff);
            sum_ch += r_diff as u64 + g_diff as u64 + b_diff as u64;
            sum_ch_count += 3;
        }
    }
    let frac = diff_pixels as f64 / total as f64;
    let mean_ch = if sum_ch_count > 0 {
        sum_ch as f64 / sum_ch_count as f64
    } else {
        0.0
    };
    println!(
        "pixel_diff_count={} pixel_diff_fraction={:.6} max_channel_diff={} mean_channel_diff={:.3}",
        diff_pixels, frac, max_ch, mean_ch,
    );
    if diff_pixels == 0 {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

struct Rgba {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

fn read_rgba(path: &str) -> Result<Rgba, Box<dyn std::error::Error>> {
    let decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());
    // Normalize to RGBA8. Our harness writes RGBA8, so this is the
    // expected path; anything else falls through with an error.
    match info.color_type {
        png::ColorType::Rgba => Ok(Rgba { w: info.width, h: info.height, px: buf }),
        other => Err(format!("unexpected color type {other:?}").into()),
    }
}
