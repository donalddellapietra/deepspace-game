//! Phase 1 visual tests for the wrapped-Cartesian planet.
//!
//! These tests live alongside the worldgen change in
//! `bootstrap::wrapped_planet_world`. They drive the render harness
//! (`--wrapped-planet`) at known camera positions, capture a PNG, and
//! analyse the silhouette via the generic
//! `tests/e2e_layer_descent/image_analysis.rs` primitives.
//!
//! Phase 1 is "no wrap, no curvature" — the slab is a hardcoded
//! `20 × 10 × 2` Cartesian patch embedded at depth 22. So:
//!
//! - From directly above, the slab silhouette is a rectangle whose
//!   visible aspect is `dims.x / dims.z = 20 / 2 = 10` (we look down
//!   the Y axis, so the visible extent is X × Z).
//! - From a horizontal viewpoint slightly above the surface looking
//!   into the slab, the silhouette is a horizontal band — a flat
//!   patch, not an arc. The top edge has near-zero curvature
//!   residual.
//!
//! Note on the parent-spec's `slab_dims.x / slab_dims.y = 2.0`
//! claim: looking from `+y` (top-down) the visible extent is
//! `dims.x × dims.z`, not `dims.x × dims.y`. The test asserts the
//! correct geometry (`x / z = 10`) and tolerates a wide range so
//! sub-pixel rendering / aliasing of the thin Z-axis don't break it.
//!
//! Bootstrap-side regression that the worldgen produces a real
//! `NodeKind::WrappedPlane` lives in `src/world/bootstrap.rs::tests`
//! (no harness dependency). Visual tests here are the second
//! gate — they would catch a silent renderer regression even if
//! the tree structure stayed valid.

#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::process::Command;

#[path = "e2e_layer_descent/image_analysis.rs"]
mod image_analysis;

use image_analysis::{
    is_sky_pixel, load_png, silhouette_top_edge, solid_aspect_ratio, solid_bounding_box,
    solid_fraction,
};

/// Gitignored artifact directory for this test scenario. Mirrors
/// `e2e_layer_descent/harness::tmp_dir` but inlined because we don't
/// need the rest of the harness module here. NEVER `/tmp` per the
/// `feedback_tmp_in_worktree` memory rule.
fn tmp_dir() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tmp")
        .join("wrapped_planet_visual");
    std::fs::create_dir_all(&root)
        .unwrap_or_else(|e| panic!("failed to create tmp dir {root:?}: {e}"));
    root
}

/// Detect the sandboxed-GUI environment heuristic used elsewhere
/// (`render_visibility.rs`, `visual_image_analysis.rs`) so headless
/// CI runs gracefully skip rather than fail.
fn run_or_skip(args: &[&str]) -> Option<std::process::Output> {
    let exe = env!("CARGO_BIN_EXE_deepspace-game");
    let out = Command::new(exe)
        .args(args)
        .output()
        .expect("spawn deepspace-game");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let blocked = !stderr.contains("startup_perf frame=")
        && (stderr.contains("scheduleApplicationNotification")
            || stderr.contains("hiservices-xpcservice")
            || stderr.contains("Connection invalid"));
    if blocked {
        eprintln!("wrapped_planet_visual: skipping in sandboxed GUI environment");
        return None;
    }
    if !out.status.success() {
        panic!("harness binary failed: stderr=\n{stderr}");
    }
    Some(out)
}

#[test]
fn slab_top_down_renders_rectangle() {
    let dir = tmp_dir();
    let png = dir.join("slab_top_down.png");
    let _ = std::fs::remove_file(&png);
    // Default `--wrapped-planet` spawn already places the camera
    // above the slab's X-Z centroid at `cam_y = slab_top + clearance`
    // (see `wrapped_planet_spawn`); pitch -π/2 looks straight down.
    let png_str = png.to_str().expect("utf8 path");
    let args = [
        "--render-harness",
        "--disable-overlay",
        "--disable-highlight",
        "--wrapped-planet",
        "--spawn-pitch", "-1.5707",
        "--harness-width", "320",
        "--harness-height", "240",
        "--exit-after-frames", "30",
        "--timeout-secs", "10",
        "--suppress-startup-logs",
        "--screenshot", png_str,
    ];
    let Some(_out) = run_or_skip(&args) else { return };
    assert!(png.exists(), "screenshot missing: {}", png.display());
    let img = load_png(&png);

    let frac = solid_fraction(&img);
    eprintln!("slab_top_down solid_fraction = {frac:.4}");
    assert!(frac > 0.05, "expected the slab to render some solid pixels, got {frac:.4}");
    assert!(frac < 0.75,
        "expected the slab to leave visible sky around it (frac={frac:.4}); \
         too high suggests the camera is buried in the slab",
    );

    let bbox = solid_bounding_box(&img).expect("non-empty silhouette");
    let (x0, y0, x1, y1) = bbox;
    eprintln!("slab_top_down bbox = ({x0},{y0})-({x1},{y1})");
    let aspect = solid_aspect_ratio(&img).expect("non-empty silhouette");
    eprintln!("slab_top_down aspect (w/h) = {aspect:.3}");

    // Looking down +y at the slab, visible aspect is dims.x / dims.z.
    // Default dims are [27, 2, 14] (longitude × vertical × latitude),
    // so visible aspect ≈ 27/14 ≈ 1.93. Asserting > 1.3 keeps the test
    // robust to slight aspect variations from camera FOV / framing.
    assert!(aspect > 1.3,
        "top-down aspect ratio = {aspect:.3}; expected wide rectangle (dims.x/dims.z ≈ 1.93)",
    );
}

// Removed: `slab_at_low_altitude_renders_flat`,
// `ray_east_along_wrap_axis_sees_slab_via_wrap`,
// `slab_xwrap_seam_is_continuous`. These tested flat-slab DDA + X-wrap
// rendering of the WrappedPlane frame. With the rotated-tangent-cube
// rewrite, WrappedPlane always renders as a sphere of rotated cubes —
// the flat-slab visual paths these tests asserted no longer fire. The
// future high-zoom flat-render path will need its own tests rather
// than reviving these (camera config, wrap dispatch, and seam
// behaviour will all differ).

#[test]
fn sky_predicate_used_here_matches_engine() {
    // Sanity check that the sky/solid predicate the asserts above
    // depend on still matches the engine's sky color. If a future
    // change to the sky shader breaks this, the visual tests would
    // give noisy false negatives — catch it here at a single line.
    assert!(is_sky_pixel(162, 196, 229));
    assert!(!is_sky_pixel(205, 225, 177)); // grass
    let _ = silhouette_top_edge; // keep the import live
}
