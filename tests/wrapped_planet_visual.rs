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
    solid_fraction, solid_pixels_per_row, top_edge_curvature_residual,
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
    assert!(frac < 0.5,
        "expected the slab to leave visible sky around it (frac={frac:.4}); \
         too high suggests the camera is buried in the slab",
    );

    let bbox = solid_bounding_box(&img).expect("non-empty silhouette");
    let (x0, y0, x1, y1) = bbox;
    eprintln!("slab_top_down bbox = ({x0},{y0})-({x1},{y1})");
    let aspect = solid_aspect_ratio(&img).expect("non-empty silhouette");
    eprintln!("slab_top_down aspect (w/h) = {aspect:.3}");

    // Looking down +y at a 20×2 slab (dims.x × dims.z), the visible
    // aspect is 20/2 = 10. Allow a wide tolerance because rendering
    // the 2-cell-thin Z dimension at this resolution is borderline
    // sub-pixel.
    assert!(aspect > 3.0,
        "top-down aspect ratio = {aspect:.3}; expected wide rectangle (dims.x/dims.z = 10)",
    );
}

#[test]
fn slab_at_low_altitude_renders_flat() {
    let dir = tmp_dir();
    let png = dir.join("slab_low_altitude.png");
    let _ = std::fs::remove_file(&png);
    // Phase 2 enables X-wrap, so any yaw aimed along the slab's
    // wrapped (X) axis from low altitude produces an infinite tunnel
    // — perspective curvature would trip this test's "flat" predicate.
    // We aim along the **non-wrap** axis (Z, slab's 2-cell thin face)
    // and steepen the pitch so the camera sees the slab top through
    // a downward-angled cone that exits via Y / Z, not via the wrap.
    // This guards the Phase 1 invariant (Cartesian DDA still flat
    // when the wrap branch isn't on the visible ray path) without
    // conflicting with Phase 2's wrap dispatch on the X axis.
    let png_str = png.to_str().expect("utf8 path");
    let args = [
        "--render-harness",
        "--disable-overlay",
        "--disable-highlight",
        "--wrapped-planet",
        "--spawn-pitch", "-1.0",  // ~57° down, so the slab top sits
                                  // in the lower frame as a flat band.
        "--spawn-yaw", "0.0",  // +z — slab's short axis (no wrap)
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
    eprintln!("slab_low_altitude solid_fraction = {frac:.4}");
    // The slab is thin (z-extent ≈ 7% of the embedding cell) and
    // the visible silhouette is small but non-zero.
    assert!(frac > 0.005, "expected some slab pixels, got frac={frac:.4}");
    assert!(frac < 0.5,
        "expected mostly-sky frame with the slab visible below the horizon, \
         got frac={frac:.4}",
    );

    let bbox = solid_bounding_box(&img).expect("non-empty silhouette");
    let (x0, y0, x1, y1) = bbox;
    eprintln!("slab_low_altitude bbox = ({x0},{y0})-({x1},{y1})");

    // The silhouette top edge of a flat patch is a (near-)horizontal
    // line — its best-fit straight-line residual is near zero. A
    // circular planet horizon viewed from low altitude would give
    // a non-trivially larger residual under the same scale.
    let resid = top_edge_curvature_residual(&img)
        .expect("expect a top edge for a non-empty silhouette");
    eprintln!("slab_low_altitude top_edge_curvature_residual = {resid:.5}");
    // Threshold 0.05 (= 5% of frame height) is generous; the
    // synthetic-flat test in `visual_image_analysis.rs` measures
    // ~1e-3, and a true sphere of similar bbox at low altitude
    // would give a much larger residual.
    assert!(resid < 0.05,
        "slab top edge looks curved (resid={resid:.5}); expected a flat band — \
         this test guards against accidentally enabling Phase 3 curvature in Phase 1",
    );

    // Sanity: the silhouette must have a defined TOP edge (not
    // touching y=0); a slab-from-above usually rests at the bottom
    // of the frame so y1 is allowed to reach `height-1`.
    assert!(y0 > 0,
        "silhouette top edge touches frame top (y0={y0}); the slab \
         should be below the camera, not filling the upper half",
    );

    let rows = solid_pixels_per_row(&img);
    let max_row = *rows.iter().max().unwrap_or(&0);
    eprintln!("slab_low_altitude max_row_count = {max_row}");
    assert!(max_row > 0, "no solid pixels in any row");
    let _ = (x0, x1, y1);
}

/// Phase 2: a ray cast east along the slab's wrapped axis from
/// above the slab, with a downward pitch shallow enough that the
/// ray would normally fly OVER the slab (no hit), should now hit
/// slab content via wrap re-entry. Without wrap, the slab is a
/// 27-cell finite wide patch; once the ray exits the WrappedPlane
/// node's east face it sails into empty embedding cells (sky).
/// With wrap, the ray re-enters from the WEST face and continues
/// east until it hits slab below. The visible artifact: solid slab
/// pixels appear in the LOWER half of the frame even though the
/// straight-line ray would never reach the slab.
///
/// We compare against a yaw=0 (along +z, NON-wrap axis) baseline —
/// the +z view at the same pitch should see roughly similar slab
/// pixel coverage (the slab below is the same; no wrap involved).
/// The yaw=π/2 view (along +x, wrap axis) MUST have at least
/// comparable solid pixels — i.e., wrap doesn't blank out the
/// view by mis-translating the ray off-screen.
#[test]
fn ray_east_along_wrap_axis_sees_slab_via_wrap() {
    let dir = tmp_dir();
    let png_x = dir.join("ray_east_wrap_x.png");
    let png_z = dir.join("ray_east_wrap_z.png");
    let _ = std::fs::remove_file(&png_x);
    let _ = std::fs::remove_file(&png_z);

    // Same pitch for both shots — the only difference is yaw axis.
    // pitch ≈ -0.25 puts the slab below the horizon; the camera
    // sees a band of slab in the lower frame.
    let pitch = "-0.25";
    let common = [
        "--render-harness",
        "--disable-overlay",
        "--disable-highlight",
        "--wrapped-planet",
        "--spawn-pitch", pitch,
        "--harness-width", "320",
        "--harness-height", "240",
        "--exit-after-frames", "30",
        "--timeout-secs", "10",
        "--suppress-startup-logs",
    ];

    let p_x = png_x.to_str().expect("utf8");
    let p_z = png_z.to_str().expect("utf8");
    let args_x: Vec<&str> = common
        .iter()
        .copied()
        .chain(["--spawn-yaw", "1.5707", "--screenshot", p_x])
        .collect();
    let args_z: Vec<&str> = common
        .iter()
        .copied()
        .chain(["--spawn-yaw", "0.0", "--screenshot", p_z])
        .collect();

    let Some(_) = run_or_skip(&args_x) else { return };
    let Some(_) = run_or_skip(&args_z) else { return };
    assert!(png_x.exists() && png_z.exists());
    let img_x = load_png(&png_x);
    let img_z = load_png(&png_z);

    let frac_x = image_analysis::solid_fraction(&img_x);
    let frac_z = image_analysis::solid_fraction(&img_z);
    eprintln!("wrap_axis_x solid_frac = {frac_x:.4}");
    eprintln!("wrap_axis_z solid_frac = {frac_z:.4}");
    // Phase 2 wrap: along +x, the ray loops back around the slab
    // and hits the slab from the wrapped side, even though the
    // straight-line ray-cast would only see slab in a narrow window.
    // Without wrap (Phase 1), an east-looking ray from above the
    // slab sails over the slab footprint and exits to empty
    // embedding cells (sky) → slab is barely / not visible.
    //
    // The +z reference confirms the camera setup: along +z the ray
    // exits the 2-cell-thin slab quickly into empty embedding —
    // that's the no-wrap baseline (frac_z ≈ 0). The +x view firing
    // its wrap branch is what makes frac_x distinctly positive.
    // The +x view sees more slab than the +z view by a meaningful
    // margin. This is robust to Phase 3 curvature: at high altitudes
    // the bent ray drops fast and clips the wrapped tail (so absolute
    // frac_x falls), but the asymmetry between wrap-axis and
    // non-wrap-axis remains — the wrap branch still opens a path
    // along +x that doesn't exist along +z.
    assert!(
        frac_x > frac_z + 0.002,
        "expected wrap-axis (+x) to see more slab than non-wrap-axis (+z) \
         by ≥ 0.002, got frac_x={frac_x:.4} frac_z={frac_z:.4}",
    );
    assert!(
        frac_x > 0.001,
        "expected slab visible in +x view at all (wrap branch firing or \
         direct sight under curvature), got frac_x={frac_x:.4}",
    );
}

/// Phase 2: a top-down screenshot at the slab must NOT show a missing
/// column at the X-wrap seam (the easternmost / westernmost slab
/// column). Without proper wrap dispatch the seam can show as a thin
/// vertical sky-stripe between the slab's two ends. Use
/// `solid_pixels_per_row` as a coarse proxy — every row that crosses
/// the slab footprint should report a non-zero solid count.
#[test]
fn slab_xwrap_seam_is_continuous() {
    use image_analysis::solid_bounding_box;
    let dir = tmp_dir();
    let png = dir.join("slab_xwrap_seam.png");
    let _ = std::fs::remove_file(&png);
    let png_str = png.to_str().expect("utf8 path");
    // Top-down (pitch ≈ -π/2). Default spawn yaw is whatever the
    // bootstrap chose; we override to a steady value so the slab is
    // axis-aligned in the frame.
    let args = [
        "--render-harness",
        "--disable-overlay",
        "--disable-highlight",
        "--wrapped-planet",
        "--spawn-pitch", "-1.5707",
        "--spawn-yaw", "0.0",
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

    let bbox = solid_bounding_box(&img).expect("non-empty silhouette");
    let (_x0, y0, _x1, y1) = bbox;
    // Within the slab's vertical span, every row should be solid all
    // the way across (no sky-coloured columns in the middle). Allow
    // a small slack at the bbox edges where antialiasing leaks.
    let rows_solid_count = image_analysis::solid_pixels_per_row(&img);
    let inset = 2usize;
    let y_lo = y0.saturating_add(inset).min(y1);
    let y_hi = y1.saturating_sub(inset).max(y0);
    let mut min_in_band: Option<u32> = None;
    for y in y_lo..=y_hi {
        let n = rows_solid_count[y];
        min_in_band = Some(min_in_band.map_or(n, |m| m.min(n)));
    }
    let min_in_band = min_in_band.expect("non-empty band");
    eprintln!("slab_xwrap min_row_in_band = {min_in_band}");
    // The slab occupies a wide rectangle (top-down); every row inside
    // it should have at least ~half the rectangle width solid. This
    // catches a wrap-axis seam (a thin sky stripe) which would drop
    // some row's solid count to ~width-of-stripe.
    let bbox_width = bbox.2.saturating_sub(bbox.0);
    let lower = (bbox_width as u32 / 2).max(1);
    assert!(
        min_in_band >= lower,
        "found a row with only {min_in_band} solid pixels inside the slab band \
         (bbox width {bbox_width}, expected >= {lower}); this looks like a \
         missing X-wrap seam column",
    );
}

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
