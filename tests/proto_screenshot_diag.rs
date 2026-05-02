//! Diagnostic: dump pixel colour histogram from the harness
//! screenshot. Helps identify what IS being rendered when the
//! magenta detector fails.

#![cfg(test)]

use std::process::Command;

#[test]
#[ignore]
fn dump_pixel_histogram() {
    // Build + run.
    let _ = Command::new("cargo")
        .args(["build", "--bin", "deepspace-game"])
        .status();
    std::fs::create_dir_all("tmp").unwrap();
    let png_path = "tmp/proto_diag.png";
    let _ = std::fs::remove_file(png_path);
    let pi = std::f32::consts::PI.to_string();
    let _ = Command::new("./target/debug/deepspace-game")
        .args([
            "--uv-sphere",
            "--render-harness",
            "--headless",
            "--frames", "5",
            "--spawn-xyz", "1.5", "1.5", "0.5",
            "--spawn-yaw", &pi,
            "--spawn-pitch", "0.0",
            "--screenshot", png_path,
            "--disable-overlay",
            "--disable-highlight",
        ])
        .status()
        .unwrap();

    let png_data = std::fs::read(png_path).unwrap();
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info().clone();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut buf).unwrap();
    let bpp = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        c => panic!("color {:?}", c),
    };

    // Histogram of dominant-channel buckets.
    let mut bins = std::collections::HashMap::<(u8, u8, u8), usize>::new();
    let mut total = 0usize;
    for chunk in buf.chunks(bpp) {
        let r = chunk[0] / 32; // bucket to 8 bins per channel
        let g = chunk[1] / 32;
        let b = chunk[2] / 32;
        *bins.entry((r, g, b)).or_insert(0) += 1;
        total += 1;
    }
    let mut sorted: Vec<_> = bins.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\nTotal pixels: {} ({}x{})", total, info.width, info.height);
    println!("Top colour buckets (R/G/B in 0..7, count, %):");
    for ((r, g, b), c) in sorted.iter().take(10) {
        println!(
            "  R={} G={} B={}  →  {:>8} ({:.1}%)",
            r,
            g,
            b,
            c,
            100.0 * (*c as f32) / (total as f32)
        );
    }

    // Sample centre pixel and a few corners to gauge what the user
    // actually sees.
    let sample = |x: u32, y: u32| {
        let i = ((y * info.width + x) as usize) * bpp;
        (buf[i], buf[i + 1], buf[i + 2])
    };
    let cx = info.width / 2;
    let cy = info.height / 2;
    println!("\nSamples:");
    println!("  centre ({}, {}): {:?}", cx, cy, sample(cx, cy));
    println!("  top-left:         {:?}", sample(10, 10));
    println!("  top-right:        {:?}", sample(info.width - 11, 10));
    println!("  bottom-centre:    {:?}", sample(cx, info.height - 11));

    // Probe a 5×3 grid around the centre to find the OBB's water
    // pixels (the proto subtree should land directly under the
    // crosshair).
    println!("\nCentre grid (rows = y offsets, cols = x offsets):");
    for dy in [-6_i32, -3, 0, 3, 6] {
        let y = (cy as i32 + dy).max(0) as u32;
        let mut row = String::new();
        for dx in [-12_i32, -6, 0, 6, 12] {
            let x = (cx as i32 + dx).max(0) as u32;
            let p = sample(x, y);
            row.push_str(&format!(" ({},{},{})", p.0, p.1, p.2));
        }
        println!("  dy={:+}:{}", dy, row);
    }
}
