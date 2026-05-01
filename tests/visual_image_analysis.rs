//! Smoke + unit tests for the generic image-analysis utilities in
//! `tests/e2e_layer_descent/image_analysis.rs`.
//!
//! Two flavors:
//!
//! 1. **Pure unit tests** (no rendering) — feed in synthetic
//!    DecodedImage buffers and assert the metrics return what they
//!    say on the tin. These guard against regressions in the math
//!    itself, independent of the renderer.
//!
//! 2. **Smoke tests** that boot the binary in `--render-harness`
//!    mode against the existing `--plain-world` (Cartesian) preset
//!    and run the analysis on the resulting PNG. These verify the
//!    full pipeline (CLI flag → frame → screenshot → load → metric)
//!    end-to-end against a sphere-AGNOSTIC scene, so they survive
//!    the cubed-sphere code removal.
//!
//! Phase 3 / Phase 4 will add the planet-specific tests (zoom-out
//! curvature transition, edge-on horizon must be a circle) on top
//! of these primitives. THIS file deliberately avoids naming a
//! particular planet geometry so the utilities can be validated
//! before the planet exists.
//!
//! Run with the standard `timeout 6 cargo test --release …` pattern
//! (per the 5s harness-timeout note in MEMORY.md).

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

#[path = "e2e_layer_descent/image_analysis.rs"]
mod image_analysis;

use harness::tmp_dir;
use image_analysis::{
    altitude_step_png, fit_circle_to_top_edge, image_delta, is_sky_pixel, load_png,
    silhouette_top_edge, solid_aspect_ratio, solid_bounding_box, solid_fraction,
    solid_pixels_per_col, solid_pixels_per_row, top_edge_curvature_residual, AltitudeSweep,
    DecodedImage,
};

// ─── synthetic-image unit tests ───────────────────────────────────────

fn fill_rgb(w: usize, h: usize, color: [u8; 3]) -> DecodedImage {
    let mut data = Vec::with_capacity(w * h * 3);
    for _ in 0..(w * h) {
        data.extend_from_slice(&color);
    }
    DecodedImage { width: w, height: h, channels: 3, data }
}

fn put_rect_rgb(img: &mut DecodedImage, x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 3]) {
    for y in y0..=y1.min(img.height - 1) {
        for x in x0..=x1.min(img.width - 1) {
            let i = (y * img.width + x) * img.channels;
            img.data[i] = color[0];
            img.data[i + 1] = color[1];
            img.data[i + 2] = color[2];
        }
    }
}

const SKY: [u8; 3] = [162, 196, 229]; // R<G<B → sky predicate true
const SOLID: [u8; 3] = [205, 225, 177]; // grass — G>R>B → not sky

#[test]
fn sky_predicate_matches_engine_palette() {
    assert!(is_sky_pixel(SKY[0], SKY[1], SKY[2]));
    assert!(!is_sky_pixel(SOLID[0], SOLID[1], SOLID[2]));
    // Warm-white star must register as solid (not sky).
    assert!(!is_sky_pixel(255, 240, 200));
    // Pure white is technically NOT sky (b is not > r,g — equal).
    assert!(!is_sky_pixel(255, 255, 255));
    // Pure black is NOT sky.
    assert!(!is_sky_pixel(0, 0, 0));
}

#[test]
fn solid_fraction_pure_sky_is_zero() {
    let img = fill_rgb(32, 16, SKY);
    assert_eq!(solid_fraction(&img), 0.0);
}

#[test]
fn solid_fraction_pure_solid_is_one() {
    let img = fill_rgb(32, 16, SOLID);
    assert_eq!(solid_fraction(&img), 1.0);
}

#[test]
fn solid_fraction_half_split_is_half() {
    let mut img = fill_rgb(32, 16, SKY);
    put_rect_rgb(&mut img, 0, 0, 31, 7, SOLID); // top half solid
    let f = solid_fraction(&img);
    assert!((f - 0.5).abs() < 1e-3, "got {f}");
}

#[test]
fn bounding_box_matches_rect() {
    let mut img = fill_rgb(64, 48, SKY);
    put_rect_rgb(&mut img, 10, 20, 30, 35, SOLID);
    let bbox = solid_bounding_box(&img).expect("non-empty");
    assert_eq!(bbox, (10, 20, 30, 35));
}

#[test]
fn aspect_ratio_of_square_is_one() {
    let mut img = fill_rgb(64, 64, SKY);
    put_rect_rgb(&mut img, 16, 16, 47, 47, SOLID); // 32×32
    let ar = solid_aspect_ratio(&img).expect("non-empty");
    assert!((ar - 1.0).abs() < 1e-3, "got {ar}");
}

#[test]
fn aspect_ratio_of_wide_slab() {
    let mut img = fill_rgb(64, 64, SKY);
    put_rect_rgb(&mut img, 0, 30, 63, 33, SOLID); // 64×4
    let ar = solid_aspect_ratio(&img).expect("non-empty");
    assert!(ar > 10.0, "expected wide slab to have ar >> 1, got {ar}");
}

#[test]
fn pure_sky_has_no_bounding_box() {
    let img = fill_rgb(8, 8, SKY);
    assert!(solid_bounding_box(&img).is_none());
}

#[test]
fn rows_and_cols_sum_to_solid_count() {
    let mut img = fill_rgb(64, 64, SKY);
    put_rect_rgb(&mut img, 16, 8, 47, 39, SOLID); // 32×32 → 1024 px
    let rows = solid_pixels_per_row(&img);
    let cols = solid_pixels_per_col(&img);
    let row_sum: u32 = rows.iter().sum();
    let col_sum: u32 = cols.iter().sum();
    assert_eq!(row_sum, 1024);
    assert_eq!(col_sum, 1024);
}

#[test]
fn flat_top_edge_has_near_zero_curvature() {
    let mut img = fill_rgb(64, 64, SKY);
    // A flat slab silhouette: solid from y=30 down, all columns.
    put_rect_rgb(&mut img, 0, 30, 63, 63, SOLID);
    let resid = top_edge_curvature_residual(&img).expect("has top edge");
    // Flat → exact zero residual (perfect line fit).
    assert!(resid < 1e-3, "expected ~0 residual for flat slab, got {resid}");
}

#[test]
fn arched_top_edge_has_nonzero_curvature() {
    // Build a parabolic top edge: y(x) = 32 - 0.5*((x-32)^2)/32 → bulges up at center.
    let mut img = fill_rgb(64, 64, SKY);
    for x in 0..64 {
        let dx = x as f32 - 32.0;
        let y_top = (32.0 - 0.5 * dx * dx / 32.0).max(0.0) as usize;
        for y in y_top..64 {
            put_rect_rgb(&mut img, x, y, x, y, SOLID);
        }
    }
    let resid = top_edge_curvature_residual(&img).expect("has top edge");
    assert!(resid > 1e-2, "expected nonzero curvature for arch, got {resid}");
}

#[test]
fn fit_circle_to_arc_recovers_radius() {
    // Generate the top half of a known circle: cx=32, cy=40, r=30.
    let cx = 32.0_f64;
    let cy = 40.0_f64;
    let r = 30.0_f64;
    let mut img = fill_rgb(64, 64, SKY);
    for x in 0..64 {
        let dx = x as f64 - cx;
        let inside = r * r - dx * dx;
        if inside <= 0.0 { continue; }
        let dy = inside.sqrt();
        let y_top = (cy - dy).round().clamp(0.0, 63.0) as usize;
        let y_bot = (cy + dy).round().clamp(0.0, 63.0) as usize;
        for y in y_top..=y_bot {
            put_rect_rgb(&mut img, x, y, x, y, SOLID);
        }
    }
    let (fx, fy, fr, mar) = fit_circle_to_top_edge(&img).expect("fit");
    assert!((fx - cx).abs() < 1.5, "cx off: {fx} vs {cx}");
    assert!((fy - cy).abs() < 1.5, "cy off: {fy} vs {cy}");
    assert!((fr - r).abs() < 1.5, "r off: {fr} vs {r}");
    // Mean residual must be small (we sampled an exact arc).
    assert!(mar < 1.5, "mar too high: {mar}");
}

#[test]
fn silhouette_top_edge_is_first_solid_row() {
    let mut img = fill_rgb(8, 8, SKY);
    put_rect_rgb(&mut img, 2, 3, 5, 7, SOLID);
    let top = silhouette_top_edge(&img);
    assert_eq!(top[0], None);
    assert_eq!(top[2], Some(3));
    assert_eq!(top[5], Some(3));
    assert_eq!(top[6], None);
}

#[test]
fn image_delta_zero_for_identical_images() {
    let a = fill_rgb(16, 16, SOLID);
    let b = fill_rgb(16, 16, SOLID);
    let d = image_delta(&a, &b);
    assert_eq!(d.changed_frac, 0.0);
    assert!(d.bbox.is_none());
}

#[test]
fn image_delta_nonzero_for_different_images() {
    let mut a = fill_rgb(16, 16, SKY);
    let mut b = fill_rgb(16, 16, SKY);
    put_rect_rgb(&mut a, 4, 4, 7, 7, SOLID);
    put_rect_rgb(&mut b, 6, 6, 9, 9, SOLID);
    let d = image_delta(&a, &b);
    assert!(d.changed_frac > 0.0);
    let bbox = d.bbox.expect("bbox");
    assert_eq!(bbox, (4, 4, 9, 9));
}

#[test]
fn altitude_sweep_endpoints_and_args() {
    let sweep = AltitudeSweep {
        xz: (1.5, 1.5),
        y_lo: 0.5,
        y_hi: 2.5,
        steps: 5,
        spawn_depth: 6,
        width: 320,
        height: 240,
        settle_frames: 30,
        timeout_secs: 10,
        pitch_override: None,
        yaw_override: None,
    };
    assert!((sweep.altitude_at(0) - 0.5).abs() < 1e-6);
    assert!((sweep.altitude_at(4) - 2.5).abs() < 1e-6);
    assert!((sweep.altitude_at(2) - 1.5).abs() < 1e-6);
    let path = altitude_step_png(std::path::Path::new("/tmp"), 3);
    assert!(path.ends_with("step_03.png"));
    let args = sweep.args_for_step(2, &path);
    // Spot-check a few key flags.
    assert!(args.iter().any(|s| s == "--render-harness"));
    assert!(args.iter().any(|s| s == "--spawn-xyz"));
    assert!(args.iter().any(|s| s == "--screenshot"));
    let depth_idx = args.iter().position(|s| s == "--spawn-depth").expect("depth flag");
    assert_eq!(args[depth_idx + 1], "6");
}

// ─── end-to-end smoke against the live renderer ───────────────────────
//
// One harness run, look down at the plain world, screenshot. Assert
// (a) the PNG loads, (b) it has at least SOME non-sky pixels (the
// terrain rendered), and (c) the analysis pipeline produces sensible
// numbers. Sphere-agnostic — uses `--plain-world`, the Cartesian
// reference preset that survives 0b's cubed-sphere removal.

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn smoke_plain_world_screenshot_analysis() {
    use std::process::Command;

    let dir = tmp_dir("visual_image_analysis_smoke");
    let png = dir.join("plain_lookdown.png");
    let _ = std::fs::remove_file(&png);
    let exe = env!("CARGO_BIN_EXE_deepspace-game");
    let out = Command::new(exe)
        .args([
            "--render-harness",
            "--disable-overlay",
            "--disable-highlight",
            "--plain-world",
            "--plain-layers", "8",
            "--spawn-depth", "6",
            "--spawn-pitch", "-1.5707",
            "--harness-width", "320",
            "--harness-height", "240",
            "--exit-after-frames", "8",
            "--timeout-secs", "5",
            "--suppress-startup-logs",
            "--screenshot", png.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn deepspace-game");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Skip in sandboxed-GUI environments where the binary can't even
    // open a swapchain — same heuristic as render_visibility.rs.
    let blocked = !stderr.contains("startup_perf frame=")
        && (stderr.contains("scheduleApplicationNotification")
            || stderr.contains("hiservices-xpcservice")
            || stderr.contains("Connection invalid"));
    if blocked {
        eprintln!("visual_image_analysis: skipping in sandboxed GUI environment");
        return;
    }
    assert!(
        out.status.success(),
        "harness binary failed: stderr=\n{stderr}"
    );
    assert!(png.exists(), "screenshot missing: {}", png.display());
    let img = load_png(&png);
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    let frac = solid_fraction(&img);
    eprintln!("plain_lookdown solid_fraction = {frac:.3}");
    // Looking straight down at a Cartesian terrain with 8 plain
    // layers must produce SOME non-sky pixels — otherwise either the
    // camera's pointed at empty space or the screenshot pipeline is
    // dropping pixels. Tolerant lower bound.
    assert!(
        frac > 0.05,
        "plain world look-down produced almost no solid pixels: frac={frac:.3}"
    );
    // The bounding box must cover most of the frame for a top-down
    // shot of a fully-tiled Cartesian region.
    let (x0, y0, x1, y1) = solid_bounding_box(&img).expect("non-empty silhouette");
    eprintln!("bbox = ({x0},{y0})-({x1},{y1})");
    let bbox_w = (x1 - x0 + 1) as f32 / img.width as f32;
    let bbox_h = (y1 - y0 + 1) as f32 / img.height as f32;
    assert!(bbox_w > 0.5 && bbox_h > 0.5, "bbox too small: {bbox_w:.2} x {bbox_h:.2}");
}
