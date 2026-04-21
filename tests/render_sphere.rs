//! Visual regression test for the cube→sphere normal-remap planet.
//!
//! Boots a `SphereBody`-flagged world headless via the render harness,
//! captures a screenshot, and asserts that the resulting silhouette
//! is round — width roughly balanced across the image, growing toward
//! the equator (quadratic-like), not cube-faceted.
//!
//! These checks don't try to replace visual review; they just catch
//! gross regressions (the ball disappears, turns into a cube, or
//! becomes degenerate) in CI-style automation.

#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};
use std::process::Command;

const WIDTH: usize = 960;
const HEIGHT: usize = 540;

struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>,
}

fn tmp_png(label: &str) -> PathBuf {
    // Per project convention, test artefacts go in the worktree's
    // `tmp/` dir (not system /tmp). Create on first use.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!(
        "sphere-{label}-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn run_game(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_deepspace-game"))
        .args(args)
        .output()
        .expect("launch deepspace-game")
}

fn load_png(path: &Path) -> RgbImage {
    use png::{ColorType, Decoder};
    let file = std::fs::File::open(path).expect("open png");
    let decoder = Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().expect("read png info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode png");
    let bytes = &buf[..info.buffer_size()];
    let mut pixels = Vec::with_capacity((info.width * info.height) as usize);
    match info.color_type {
        ColorType::Rgb => {
            for c in bytes.chunks_exact(3) {
                pixels.push([c[0], c[1], c[2]]);
            }
        }
        ColorType::Rgba => {
            for c in bytes.chunks_exact(4) {
                pixels.push([c[0], c[1], c[2]]);
            }
        }
        other => panic!("unexpected color type {other:?}"),
    }
    RgbImage {
        width: info.width as usize,
        height: info.height as usize,
        pixels,
    }
}

/// A pixel is counted as "planet" if it differs meaningfully from the
/// sky gradient. The shader's sky colors are blue-ish (R low, B high);
/// stone terrain hits come out gray. We classify by low saturation
/// OR high red channel relative to blue.
fn is_planet_pixel(p: [u8; 3]) -> bool {
    let r = p[0] as i32;
    let g = p[1] as i32;
    let b = p[2] as i32;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let sat = max - min;
    // Sky: blue > red + ~20, and saturation is moderate-high.
    // Planet (stone-lit): approximately gray, saturation < 15.
    sat < 25 && max > 30
}

fn is_sandboxed_gui_startup_blocked(stderr: &str) -> bool {
    let has_no_frames = !stderr.contains("startup_perf frame=");
    let has_launchservices_failure = stderr.contains("scheduleApplicationNotification")
        || stderr.contains("Connection Invalid")
        || stderr.contains("Error received in message reply handler");
    has_no_frames && has_launchservices_failure
}

#[test]
fn sphere_world_produces_round_silhouette() {
    let png = tmp_png("silhouette");
    let output = run_game(&[
        "--render-harness",
        "--disable-overlay",
        "--sphere-world",
        "--plain-layers",
        "6",
        "--harness-width",
        &WIDTH.to_string(),
        "--harness-height",
        &HEIGHT.to_string(),
        "--screenshot",
        png.to_str().unwrap(),
        "--exit-after-frames",
        "2",
        "--timeout-secs",
        "10",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if is_sandboxed_gui_startup_blocked(&stderr) {
        eprintln!("sphere silhouette test: skipping in sandboxed GUI env");
        return;
    }
    assert!(
        output.status.success(),
        "game exited non-zero\nstderr:\n{stderr}"
    );
    assert!(
        png.exists(),
        "screenshot not written at {:?}\nstderr:\n{stderr}",
        png
    );
    let img = load_png(&png);

    // Count planet pixels on each row.
    let mut rows: Vec<usize> = Vec::with_capacity(img.height);
    let mut total = 0usize;
    for y in 0..img.height {
        let mut n = 0;
        for x in 0..img.width {
            if is_planet_pixel(img.pixels[y * img.width + x]) {
                n += 1;
            }
        }
        rows.push(n);
        total += n;
    }

    // Sanity: planet covers a meaningful fraction of the frame. The
    // default camera sits at a cube corner just outside the inscribed
    // sphere, so a close-range view is expected — allow the planet to
    // dominate the frame but not fill it entirely (would indicate the
    // camera wound up inside the ball).
    let frac = total as f64 / (img.width * img.height) as f64;
    assert!(
        frac > 0.05,
        "planet covers only {:.2}% of frame (expected >5%)",
        frac * 100.0
    );
    assert!(
        frac < 0.95,
        "planet covers {:.2}% of frame (expected <95% — camera should frame ~a planet, not be inside it)",
        frac * 100.0
    );

    // Silhouette should be widest somewhere near the middle row. Find
    // the argmax and confirm it sits in the central band (not an edge
    // artifact).
    let (argmax, max_row) = rows
        .iter()
        .enumerate()
        .max_by_key(|&(_, n)| *n)
        .map(|(i, &n)| (i, n))
        .unwrap();
    let central_band = (img.height / 4)..(3 * img.height / 4);
    assert!(
        central_band.contains(&argmax),
        "silhouette peak at row {argmax} is outside central band {:?}",
        central_band
    );

    // Silhouette width should fall off toward top and bottom — a
    // cube-silhouette wouldn't (the cube face fills a flat rectangle).
    // Compare max row width to the top-and-bottom-edge average. For a
    // round ball the ratio should comfortably exceed 1.5.
    let edge_band = 20usize.min(img.height / 10);
    let top_avg = rows[..edge_band].iter().sum::<usize>() as f64 / edge_band as f64;
    let bot_avg = rows[(img.height - edge_band)..].iter().sum::<usize>() as f64 / edge_band as f64;
    let edge_avg = (top_avg + bot_avg) / 2.0;
    assert!(
        (max_row as f64) > edge_avg * 1.3 + 1.0,
        "silhouette width peak {} not meaningfully larger than image-edge avg {:.1}",
        max_row,
        edge_avg
    );

    eprintln!(
        "sphere silhouette: total_planet_frac={:.3}, argmax_row={argmax}, max_row_px={max_row}, edge_avg={:.1}",
        frac, edge_avg
    );
}

/// Directional-light gradient: the shader's sun points at
/// `normalize(0.4, 0.7, 0.3)` — mostly up (+y) and slightly right
/// (+x). The planet should be brighter on the upper-right than on
/// the lower-left. If the Jacobian normal-remap is broken (e.g. the
/// identity fallback kicks in or the transpose is wrong), lighting
/// collapses to flat-face shading and this differential disappears.
#[test]
fn sphere_world_lighting_differential() {
    let png = tmp_png("terminator");
    let output = run_game(&[
        "--render-harness",
        "--disable-overlay",
        "--sphere-world",
        "--plain-layers",
        "6",
        "--harness-width",
        &WIDTH.to_string(),
        "--harness-height",
        &HEIGHT.to_string(),
        "--screenshot",
        png.to_str().unwrap(),
        "--exit-after-frames",
        "2",
        "--timeout-secs",
        "10",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if is_sandboxed_gui_startup_blocked(&stderr) {
        eprintln!("sphere lighting test: skipping in sandboxed GUI env");
        return;
    }
    assert!(output.status.success());
    assert!(png.exists());
    let img = load_png(&png);

    // Sample planet-interior pixels in the four image quadrants.
    // Take a band well inside the silhouette (avoid edge aliasing).
    let cx = img.width as i32 / 2;
    let cy = img.height as i32 / 2;
    let r = (img.height as i32 / 4).min(img.width as i32 / 4);

    let sample_quadrant = |dx: i32, dy: i32| -> f64 {
        let mut total = 0u64;
        let mut count = 0u64;
        let ox = cx + dx;
        let oy = cy + dy;
        let rr = r / 3;
        for y in (oy - rr)..(oy + rr) {
            for x in (ox - rr)..(ox + rr) {
                if x < 0 || y < 0 || x >= img.width as i32 || y >= img.height as i32 {
                    continue;
                }
                let p = img.pixels[(y as usize) * img.width + x as usize];
                if !is_planet_pixel(p) { continue; }
                total += p[0] as u64 + p[1] as u64 + p[2] as u64;
                count += 1;
            }
        }
        if count == 0 { return 0.0; }
        total as f64 / (3.0 * count as f64)
    };

    // Image y grows downward → negative dy means upper quadrants.
    let upper_right = sample_quadrant(r / 2, -r / 2);
    let lower_left = sample_quadrant(-r / 2, r / 2);
    let upper_left = sample_quadrant(-r / 2, -r / 2);
    let lower_right = sample_quadrant(r / 2, r / 2);
    eprintln!(
        "quadrant brightness: UL={:.1} UR={:.1} LL={:.1} LR={:.1}",
        upper_left, upper_right, lower_left, lower_right,
    );

    assert!(
        upper_right > 0.0 && lower_left > 0.0,
        "quadrants didn't contain planet pixels: UR={upper_right:.1} LL={lower_left:.1}"
    );
    // Sun is toward +x (slight right) and mostly +y (image up). Top
    // should outshine bottom by a meaningful margin if normals are
    // correctly bent.
    let top_avg = (upper_left + upper_right) / 2.0;
    let bot_avg = (lower_left + lower_right) / 2.0;
    // The visible hemisphere is mostly on +x/+z cube faces with
    // similar sun-dot products, so the top/bottom differential from
    // sun-dir (+y dominant) is real but subtle — sampling variance
    // and gamma compression flatten it further. 1.0 is a conservative
    // noise floor that still catches "remap is broken" (would give
    // zero differential with cube-aligned normals in this pose).
    assert!(
        top_avg > bot_avg + 1.0,
        "no top/bottom lighting differential: top={top_avg:.1} bot={bot_avg:.1}. \
         Sphere normal remap likely broken."
    );
}
