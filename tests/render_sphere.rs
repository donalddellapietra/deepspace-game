//! Visual regression tests for the cube-IS-a-sphere architecture.
//!
//! Boots a `SphereBody` world (uniform stone cube wrapped in a
//! Cartesian shell) via the render harness, captures a screenshot,
//! and asserts:
//!
//! 1. The rendered planet has a round silhouette — produced by the
//!    shader's analytic ray-vs-inscribed-sphere test, NOT by
//!    voxelizing a ball shape into the tree.
//! 2. Shading shows a directional-light terminator — normals come
//!    from `normalize(hit − sphere_center)`, so the sun-facing side
//!    is meaningfully brighter than the shadowed side.
//! 3. Storage dedups to O(depth): a depth-40 planet has <1k library
//!    entries, not the millions that a voxelized-ball approach
//!    would produce.

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

fn is_planet_pixel(p: [u8; 3]) -> bool {
    let r = p[0] as i32;
    let g = p[1] as i32;
    let b = p[2] as i32;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let sat = max - min;
    sat < 25 && max > 30
}

fn is_sandboxed_gui_startup_blocked(stderr: &str) -> bool {
    let has_no_frames = !stderr.contains("startup_perf frame=");
    let has_launchservices_failure = stderr.contains("scheduleApplicationNotification")
        || stderr.contains("Connection Invalid")
        || stderr.contains("Error received in message reply handler");
    has_no_frames && has_launchservices_failure
}

fn run_sphere_screenshot(
    label: &str,
    extra_args: &[&str],
) -> Option<(RgbImage, String)> {
    let png = tmp_png(label);
    let mut args: Vec<String> = vec![
        "--render-harness".into(),
        "--disable-overlay".into(),
        // The harness's cursor highlight paints a yellow glow over
        // the cube-AABB of whatever the CPU raycast hits. For a
        // uniform-stone SphereBody that's a big yellow rectangle
        // swamping the sphere silhouette. Disable so pixel analysis
        // reads the actual shader output.
        "--disable-highlight".into(),
        "--sphere-world".into(),
        "--harness-width".into(),
        WIDTH.to_string(),
        "--harness-height".into(),
        HEIGHT.to_string(),
        "--screenshot".into(),
        png.to_string_lossy().into_owned(),
        "--exit-after-frames".into(),
        "2".into(),
        "--timeout-secs".into(),
        "10".into(),
    ];
    for a in extra_args {
        args.push((*a).to_string());
    }
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = run_game(&arg_refs);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if is_sandboxed_gui_startup_blocked(&stderr) {
        eprintln!("sphere {label}: skipping in sandboxed GUI env");
        return None;
    }
    assert!(
        output.status.success(),
        "game exited non-zero\nstderr:\n{stderr}"
    );
    assert!(png.exists(), "screenshot missing at {:?}", png);
    Some((load_png(&png), stderr))
}

#[test]
fn sphere_world_produces_round_silhouette() {
    let Some((img, _)) = run_sphere_screenshot("silhouette", &[]) else {
        return;
    };

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

    let frac = total as f64 / (img.width * img.height) as f64;
    assert!(
        (0.02..0.6).contains(&frac),
        "planet covers {:.2}% of frame (expected 2-60%)",
        frac * 100.0
    );

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

    let edge_band = 20usize.min(img.height / 10);
    let top_avg = rows[..edge_band].iter().sum::<usize>() as f64 / edge_band as f64;
    let bot_avg =
        rows[(img.height - edge_band)..].iter().sum::<usize>() as f64 / edge_band as f64;
    let edge_avg = (top_avg + bot_avg) / 2.0;
    assert!(
        max_row as f64 > edge_avg * 1.5 + 5.0,
        "silhouette peak {} not much wider than edges {:.1} — shape looks cubic",
        max_row,
        edge_avg,
    );

    eprintln!(
        "silhouette: frac={:.3} peak_row={argmax} max_row={max_row} edge_avg={:.1}",
        frac, edge_avg
    );
}

#[test]
fn sphere_world_lighting_differential() {
    let Some((img, _)) = run_sphere_screenshot("terminator", &[]) else {
        return;
    };

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
                if !is_planet_pixel(p) {
                    continue;
                }
                total += p[0] as u64 + p[1] as u64 + p[2] as u64;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        total as f64 / (3.0 * count as f64)
    };

    let upper_right = sample_quadrant(r / 2, -r / 2);
    let lower_left = sample_quadrant(-r / 2, r / 2);
    let upper_left = sample_quadrant(-r / 2, -r / 2);
    let lower_right = sample_quadrant(r / 2, r / 2);

    eprintln!(
        "quadrants: UL={:.1} UR={:.1} LL={:.1} LR={:.1}",
        upper_left, upper_right, lower_left, lower_right,
    );

    assert!(
        [upper_left, upper_right, lower_left, lower_right]
            .iter()
            .all(|&v| v > 0.0),
        "some quadrant had no planet pixels",
    );

    let top_avg = (upper_left + upper_right) / 2.0;
    let bot_avg = (lower_left + lower_right) / 2.0;
    assert!(
        top_avg > bot_avg + 5.0,
        "no top/bottom lighting differential: top={top_avg:.1} bot={bot_avg:.1}. \
         Under analytic radial normals the +y-heavy sun should light the top much \
         more than the bottom."
    );
}

#[test]
fn sphere_world_storage_is_o_depth() {
    let Some((_, stderr)) = run_sphere_screenshot("deep-40", &["--plain-layers", "40"]) else {
        return;
    };
    let lib_line = stderr
        .lines()
        .find(|l| l.contains("Sphere body world: layers=40"))
        .expect("worldgen banner missing");
    for part in lib_line.split(',') {
        if let Some(stripped) = part.trim().strip_prefix("library_entries=") {
            let n: u64 = stripped.parse().unwrap();
            assert!(
                n < 1_000,
                "layers=40 produced {n} library entries — dedup broken"
            );
            eprintln!("depth-40 library entries: {n}");
            return;
        }
    }
    panic!("could not parse library_entries from stderr: {lib_line}");
}
