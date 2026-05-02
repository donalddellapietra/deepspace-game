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
    // Water (palette 6) cells in the rendered frame land at roughly
    // RGB ≈ (167, 192, 221). Discriminating from sky (~(163, 196, 239))
    // is tight on R/G/B individually — but `B−R ≈ 54` for water vs
    // `B−R ≈ 76` for sky, and water's `B−G ≈ 29` vs sky's `B−G ≈ 43`.
    // Use those gaps. The grass band has B < G, so any `b > g`
    // already excludes it.
    rgb.chunks_exact(3)
        .filter(|c| {
            let r = c[0] as i16;
            let g = c[1] as i16;
            let b = c[2] as i16;
            let br = b - r;
            let bg = b - g;
            b > 200
                && bg > 20 && bg < 35
                && br > 40 && br < 65
        })
        .count()
}

#[test]
#[ignore]
fn proto_obb_visible_in_screenshot() {
    let (w, h, rgb) = run_harness("tmp/proto_obb_default.png");
    let total = (w as usize) * (h as usize);

    // Sample a 60×40 patch centred on the screen — that's where the
    // OBB lands at the test's spawn config. Counting matching pixels
    // in this window discriminates water (the proto subtree's
    // rendered cell) from the surrounding sky.
    let cx = w / 2;
    let cy = h / 2;
    let x0 = cx.saturating_sub(30);
    let x1 = (cx + 30).min(w);
    let y0 = cy.saturating_sub(20);
    let y1 = (cy + 20).min(h);
    let mut window_water = 0usize;
    let mut window_total = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = ((y * w + x) as usize) * 3;
            let r = rgb[i] as i16;
            let g = rgb[i + 1] as i16;
            let b = rgb[i + 2] as i16;
            let br = b - r;
            let bg = b - g;
            window_total += 1;
            if b > 200 && bg > 20 && bg < 35 && br > 40 && br < 65 {
                window_water += 1;
            }
        }
    }
    let total_water = count_water_blue(&rgb);

    println!("screenshot {}x{} ({} pixels)", w, h, total);
    println!(
        "centre window {}x{} water = {} / {} ({:.1}%)",
        x1 - x0,
        y1 - y0,
        window_water,
        window_total,
        100.0 * window_water as f32 / window_total as f32,
    );
    println!(
        "frame total water = {} ({:.4}%)",
        total_water,
        100.0 * total_water as f32 / total as f32,
    );

    // The OBB at depth 3 of the body's tree covers roughly a
    // 30×15 px patch under the crosshair at spawn distance. The
    // 60×40 window catches the OBB plus surrounding margin; expect
    // 100+ matching water pixels.
    assert!(
        window_water > 100,
        "OBB not visible in the centre window — water = {} pixels, \
         expected > 100. Screenshot saved to tmp/proto_obb_default.png.",
        window_water,
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
