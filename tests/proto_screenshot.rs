//! Integration test: run the headless harness, capture a screenshot,
//! and verify the prototype's magenta OBB shows up. Bridges
//! "the WGSL code path is exercised" with "the unit tests pass."
//!
//! Run with: `cargo test --test proto_screenshot -- --ignored`
//! (Marked ignored because it builds + runs the binary, which is
//! too slow for the default test suite.)
//!
//! Camera placement: `--spawn-xyz 1.5 1.5 0.5 --spawn-yaw π
//! --spawn-pitch 0`. With `smoothed_up = (0, 1, 0)` (its initial
//! value, never updated in the code path), yaw=π, pitch=0 yields
//! `forward = (0, 0, +1)`. Camera at `(1.5, 1.5, 0.5)` looking +Z
//! lands the body (centre `(1.5, 1.5, 1.5)`, outer radius 0.60) and
//! the prototype OBB (centred at `~(1.524, 1.5, 1.093)`) directly
//! ahead. The OBB is ~0.6 m away and ~2° off forward — well inside
//! the 69° FOV.

#![cfg(test)]

use std::path::Path;
use std::process::Command;

fn run_harness(png_path: &str) -> (u32, u32, Vec<u8>) {
    // Build first so cargo's test runner doesn't race the binary.
    let build = Command::new("cargo")
        .args(["build", "--bin", "deepspace-game"])
        .output()
        .expect("cargo build invocation");
    assert!(
        build.status.success(),
        "cargo build failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    std::fs::create_dir_all("tmp").expect("create tmp/");
    let _ = std::fs::remove_file(png_path);

    let pi = std::f32::consts::PI.to_string();
    let run = Command::new("./target/debug/deepspace-game")
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
        .output()
        .expect("harness invocation");
    assert!(
        run.status.success(),
        "harness exit: {:?}\nstderr:\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        Path::new(png_path).exists(),
        "screenshot not written to {}\nstderr:\n{}",
        png_path,
        String::from_utf8_lossy(&run.stderr)
    );

    let png_data = std::fs::read(png_path).expect("read screenshot");
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().expect("png decode");
    let info = reader.info().clone();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut buf).expect("png frame");

    let bpp = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        c => panic!("unexpected png color type: {:?}", c),
    };
    let mut rgb = Vec::with_capacity((info.width * info.height) as usize * 3);
    for chunk in buf.chunks(bpp) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    (info.width, info.height, rgb)
}

fn count_magenta(rgb: &[u8]) -> usize {
    // Magenta = high R, low G, high B. The prototype paints
    // `(1.0, 0.0, 1.0) * (0.7 + 0.3 * bevel)` which gives R, B in
    // [178, 255] and G ≤ 76 (the bevel only attenuates, never adds
    // green). Tolerate gamma + sRGB tonemapping by using loose
    // thresholds.
    rgb.chunks_exact(3)
        .filter(|c| c[0] > 150 && c[1] < 100 && c[2] > 150 && (c[0] as i16 - c[2] as i16).abs() < 80)
        .count()
}

#[test]
#[ignore]
fn proto_obb_visible_in_screenshot() {
    let (w, h, rgb) = run_harness("tmp/proto_obb_default.png");
    let total = (w as usize) * (h as usize);
    let magenta = count_magenta(&rgb);

    println!("screenshot {}x{} ({} pixels)", w, h, total);
    println!(
        "magenta = {} ({:.4}%)",
        magenta,
        100.0 * magenta as f32 / total as f32
    );

    // OBB at distance ~0.6 m, half-extent ~0.05 m. Angular size
    // ~2·atan(0.05/0.6) ≈ 9.5°. At 1280×720 with FOV 1.2 rad, one
    // degree ≈ 19 px → ~180×180 px footprint = ~32 000 magenta px.
    // Threshold is conservative (50 px) to allow tonemap losses or
    // partial occlusion by the body shell.
    assert!(
        magenta > 50,
        "no magenta cube visible — expected >50 px, got {}. \
         Either the OBB is being placed outside the camera's FOV, \
         the t-comparison vs UV march is wrong, or the WGSL doesn't \
         match the CPU port at `src/world/raycast/proto_obb.rs`. \
         Screenshot saved to tmp/proto_obb_default.png — inspect \
         to debug.",
        magenta
    );
}

#[test]
#[ignore]
fn proto_obb_dominates_or_renders_correctly() {
    // Sanity: the screenshot is non-trivial (sky + something
    // rendered). Catches the case where the harness produces an
    // all-black or all-sky frame and the magenta count happens to
    // be zero just because nothing rendered.
    let (w, h, rgb) = run_harness("tmp/proto_obb_default.png");
    let total = (w as usize) * (h as usize);
    let mut variant_pixels = 0usize;
    for chunk in rgb.chunks_exact(3) {
        let r = chunk[0] as i16;
        let g = chunk[1] as i16;
        let b = chunk[2] as i16;
        let is_dark = r < 10 && g < 10 && b < 10;
        if !is_dark {
            variant_pixels += 1;
        }
    }
    assert!(
        variant_pixels > total / 4,
        "screenshot {}x{} has {} non-dark pixels — harness produced \
         a mostly-black frame, can't trust magenta detection",
        w, h, variant_pixels
    );
}
