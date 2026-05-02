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

fn count_water_blue(rgb: &[u8]) -> usize {
    // Water (palette block 6) = `(0.20, 0.40, 0.80)`. After bevel
    // (0.7..1.0) and sRGB encode that gamma-corrects roughly 1.4×,
    // we expect R ≲ 130, G ∈ [120, 200], B ∈ [180, 230]. The body's
    // sky band has high values across all channels (R≈160, G≈200,
    // B≈240); the green grass has G > R, B not dominant. Demand
    // B > R+30 AND B > G+10 to discriminate water from sky/grass.
    rgb.chunks_exact(3)
        .filter(|c| {
            let r = c[0] as i16;
            let g = c[1] as i16;
            let b = c[2] as i16;
            b > 150 && b > r + 30 && b > g + 10 && r < 180
        })
        .count()
}

#[test]
#[ignore]
fn proto_obb_visible_in_screenshot() {
    let (w, h, rgb) = run_harness("tmp/proto_obb_default.png");
    let total = (w as usize) * (h as usize);
    let blue = count_water_blue(&rgb);

    println!("screenshot {}x{} ({} pixels)", w, h, total);
    println!(
        "water_blue = {} ({:.4}%)",
        blue,
        100.0 * blue as f32 / total as f32
    );

    // OBB at distance ~0.6 m, half-extent ~0.05 m → ~180×180 px on
    // a 1280×720 frame with FOV 1.2 rad. Threshold conservative.
    assert!(
        blue > 50,
        "no water (blue) OBB visible — expected >50 px, got {}. \
         Likely causes: OBB is outside camera FOV, the WATER subtree \
         BFS isn't reaching the shader, or `march_entity_subtree` \
         isn't returning hits in OBB-local. \
         Screenshot saved to tmp/proto_obb_default.png — inspect \
         to debug.",
        blue
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
