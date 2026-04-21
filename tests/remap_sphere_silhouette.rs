//! End-to-end visual regression for the cube→sphere remap renderer.
//!
//! Launches the game with `--remap-sphere-world --render-harness`,
//! captures a screenshot, and image-analyzes the silhouette:
//!
//! - The planet body must be visible (non-trivial pixel count).
//! - Its silhouette must be a circle (sub-pixel radial variance).
//! - No cube-face seams — a rotated viewpoint gives the same shape.
//!
//! This is the real validation of "sphere = cube + ray remap": the
//! claim is that nothing about the storage or tree knows the result
//! is a sphere, so the silhouette must be independent of cube
//! orientation. Any residual hexagonal or diamond artifact would
//! show up as a >1 px radial std-dev in the circle fit.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{run, tmp_dir, ScriptBuilder};

const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--remap-sphere-world",
    "--disable-highlight",
    "--harness-width",
    "512",
    "--harness-height",
    "512",
    "--exit-after-frames",
    "60",
    "--timeout-secs",
    "30",
];

/// Load a PNG and return (width, height, rgba buffer).
fn load_png(path: &std::path::Path) -> (usize, usize, Vec<u8>) {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read png header");
    let info = reader.info().clone();
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unsupported png color type {other:?}"),
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).expect("decode png frame");
    let data = &buf[..frame.buffer_size()];
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * channels;
            rgba.push(data[i]);
            rgba.push(data[i + 1]);
            rgba.push(data[i + 2]);
            rgba.push(if channels == 4 { data[i + 3] } else { 255 });
        }
    }
    (w, h, rgba)
}

/// Planet-pixel mask: non-sky pixels. The sky gradient is
/// `b > r`, so anything with `r ≥ b − 10` and `r ≥ 100` is
/// considered planet.
fn planet_mask(rgba: &[u8], w: usize, h: usize) -> Vec<bool> {
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let r = rgba[i] as i32;
            let g = rgba[i + 1] as i32;
            let b = rgba[i + 2] as i32;
            // Planet: warm-ish gray, r dominant or equal to b.
            if r >= b - 10 && r >= 80 && g >= 60 {
                mask[y * w + x] = true;
            }
        }
    }
    mask
}

fn fit_circle(mask: &[bool], w: usize, h: usize) -> (f64, f64, f64, f64, usize) {
    let mut n = 0.0_f64;
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                cx += x as f64 + 0.5;
                cy += y as f64 + 0.5;
                n += 1.0;
            }
        }
    }
    if n < 1.0 {
        return (0.0, 0.0, 0.0, f64::INFINITY, 0);
    }
    cx /= n;
    cy /= n;
    // Boundary pixels
    let mut rs = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            let mut boundary = false;
            for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    boundary = true;
                    break;
                }
                if !mask[ny as usize * w + nx as usize] {
                    boundary = true;
                    break;
                }
            }
            if boundary {
                let ex = x as f64 + 0.5 - cx;
                let ey = y as f64 + 0.5 - cy;
                rs.push((ex * ex + ey * ey).sqrt());
            }
        }
    }
    if rs.is_empty() {
        return (cx, cy, 0.0, f64::INFINITY, n as usize);
    }
    let mean: f64 = rs.iter().sum::<f64>() / rs.len() as f64;
    let var: f64 = rs.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rs.len() as f64;
    (cx, cy, mean, var.sqrt(), n as usize)
}

#[test]
fn remap_sphere_renders_a_circular_silhouette() {
    let dir = tmp_dir("remap_sphere_silhouette");
    let shot = dir.join("axis.png");
    let _ = std::fs::remove_file(&shot);

    let script = ScriptBuilder::new()
        .wait(5)
        .screenshot(shot.to_string_lossy().as_ref())
        .emit("axis");

    let trace = run(HARNESS_ARGS, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(80).collect::<Vec<_>>().join("\n"),
    );
    assert!(shot.exists(), "screenshot missing: {}", shot.display());

    let (w, h, rgba) = load_png(&shot);
    let mask = planet_mask(&rgba, w, h);
    let (cx, cy, r_mean, r_std, count) = fit_circle(&mask, w, h);

    eprintln!(
        "remap-sphere silhouette: {count} px, center=({cx:.2}, {cy:.2}), \
         r_mean={r_mean:.3}, r_std={r_std:.3}"
    );

    // Silhouette is non-trivial.
    assert!(
        count > 5_000,
        "planet pixel count {count} too small — the sphere may not be rendering at all. \
         Check shader dispatch or camera position."
    );
    // Centered within a few pixels of image center.
    let img_cx = w as f64 / 2.0;
    let img_cy = h as f64 / 2.0;
    assert!(
        (cx - img_cx).abs() < 10.0 && (cy - img_cy).abs() < 10.0,
        "planet off-center: ({cx}, {cy}), image center ({img_cx}, {img_cy})"
    );
    // Silhouette is circular — this is the architectural claim. Any
    // hexagonal / diamond / cube-face-seam artifact in the F-map
    // would show up as large radial variance.
    //
    // Budget: 1 px absolute. At r_mean ≈ 194, that's 0.5% — far
    // tighter than any faceted silhouette would be. Current
    // baseline on this commit is ~0.27 px.
    assert!(
        r_std < 1.0,
        "silhouette is not circular: r_std={r_std:.3} px. \
         This means cube-face seams are leaking through F — the \
         architectural claim is broken."
    );
}

/// Same test, rotated so a cube CORNER points at the camera. The
/// historical "seam detector": every cubed-sphere approach breaks at
/// the 8 corners where chart boundaries meet. With the F-map there
/// are no charts — the mapping is globally smooth — so the
/// silhouette must be identically circular at any camera orientation.
///
/// Places the camera on the (1,1,1) diagonal at distance 1.5 from
/// the ball center (1.5, 1.5, 1.5), looking back at center. This
/// keeps the sphere in the middle of the frame while aligning the
/// view straight at the (+x,+y,+z) cube corner.
///
/// yaw=π/4, pitch=−atan(1/√2)≈−0.6155 ⇒ forward = −(1,1,1)/√3.
#[test]
fn remap_sphere_corner_on_view_still_circular() {
    let dir = tmp_dir("remap_sphere_silhouette");
    let shot = dir.join("corner_on.png");
    let _ = std::fs::remove_file(&shot);

    // ball center (1.5, 1.5, 1.5) + 0.866 · (1,1,1) = (2.366)³
    let corner_args: &[&str] = &[
        "--render-harness",
        "--remap-sphere-world",
        "--disable-highlight",
        "--harness-width",
        "512",
        "--harness-height",
        "512",
        "--exit-after-frames",
        "60",
        "--timeout-secs",
        "30",
        "--spawn-xyz",
        "2.366",
        "2.366",
        "2.366",
        "--spawn-yaw",
        "0.7854", // π/4
        "--spawn-pitch",
        "-0.6155", // -atan(1/√2)
    ];

    let script = ScriptBuilder::new()
        .wait(5)
        .screenshot(shot.to_string_lossy().as_ref())
        .emit("corner_on");

    let trace = run(corner_args, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}",
        trace.stderr,
    );
    assert!(shot.exists(), "screenshot missing: {}", shot.display());

    let (w, h, rgba) = load_png(&shot);
    let mask = planet_mask(&rgba, w, h);
    let (cx, cy, r_mean, r_std, count) = fit_circle(&mask, w, h);

    eprintln!(
        "corner-on silhouette: {count} px, center=({cx:.2}, {cy:.2}), \
         r_mean={r_mean:.3}, r_std={r_std:.3}"
    );
    assert!(count > 30_000, "corner-on sphere not centered: {count} px");
    assert!(
        (cx - w as f64 / 2.0).abs() < 15.0 && (cy - h as f64 / 2.0).abs() < 15.0,
        "corner-on sphere off-center: ({cx}, {cy})"
    );
    // Same tolerance as face-on. Any chart boundary leaking through
    // F would show up here as ≥2 px radial stddev.
    assert!(
        r_std < 1.0,
        "corner-on silhouette: r_std={r_std:.3} px — cube-face seams leaked through F"
    );
}

/// Renders the sphere at `--remap-sphere-layers 3` — 3 levels of
/// subdivision gives voxel face patches ~25 pixels across at this
/// viewport. Under that resolution the cube_face_bevel smoothstep
/// should carve visible dark lines between neighboring cells.
///
/// Verifies two things:
/// - The silhouette stays a circle (lower layer count must not leak
///   a faceted outline — that would mean F isn't mapping the cube
///   boundary onto the sphere).
/// - Adjacent-pixel brightness deltas on the center row have high
///   stddev, i.e. the image has sharp block boundaries (bevels are
///   actually modulating), not just smooth diffuse gradient.
#[test]
fn remap_sphere_blocks_show_bevels_at_layers_3() {
    let dir = tmp_dir("remap_sphere_silhouette");
    let shot = dir.join("blocks_layers_3.png");
    let _ = std::fs::remove_file(&shot);

    let args: &[&str] = &[
        "--render-harness",
        "--remap-sphere-world",
        "--remap-sphere-layers",
        "3",
        "--disable-highlight",
        "--harness-width",
        "512",
        "--harness-height",
        "512",
        "--exit-after-frames",
        "60",
        "--timeout-secs",
        "30",
    ];

    let script = ScriptBuilder::new()
        .wait(5)
        .screenshot(shot.to_string_lossy().as_ref())
        .emit("blocks_layers_3");

    let trace = run(args, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}",
        trace.stderr,
    );
    assert!(shot.exists(), "screenshot missing: {}", shot.display());

    let (w, h, rgba) = load_png(&shot);
    let mask = planet_mask(&rgba, w, h);
    let (cx, cy, _r_mean, r_std, count) = fit_circle(&mask, w, h);
    eprintln!(
        "layers=3 silhouette: {count} px, center=({cx:.2}, {cy:.2}), r_std={r_std:.3}"
    );
    assert!(count > 30_000, "ball not rendered: {count} px");
    assert!(
        r_std < 2.0,
        "silhouette not circular at layers=3: r_std={r_std:.3} px"
    );

    // Bevel visibility: scan center row, collect on-ball adjacent-
    // pixel luminance deltas, report their stddev. Smooth diffuse
    // shading gives near-constant deltas (~1–2 per pixel). A
    // bevel-stamped row produces dark cliffs at each cell edge,
    // inflating the stddev well above that.
    let y = h / 2;
    let lum = |rgba: &[u8], idx: usize| -> f64 {
        0.2126 * rgba[idx] as f64
            + 0.7152 * rgba[idx + 1] as f64
            + 0.0722 * rgba[idx + 2] as f64
    };
    let mut deltas: Vec<f64> = Vec::new();
    for x in 1..w {
        if mask[y * w + x] && mask[y * w + x - 1] {
            let i_cur = (y * w + x) * 4;
            let i_prev = (y * w + x - 1) * 4;
            deltas.push((lum(&rgba, i_cur) - lum(&rgba, i_prev)).abs());
        }
    }
    assert!(
        deltas.len() > 100,
        "center row has too few on-ball pairs ({}) — is the ball off-center?",
        deltas.len()
    );
    let mean: f64 = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let var: f64 =
        deltas.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / deltas.len() as f64;
    let delta_std = var.sqrt();
    // Report max delta too — a single ≥30-intensity cliff is an
    // unmistakable bevel transition, impossible on a smooth sphere.
    let delta_max = deltas.iter().copied().fold(0.0_f64, f64::max);
    eprintln!(
        "layers=3 center-row bevel probe: {} pairs, mean |Δlum|={:.2}, \
         stddev={:.2}, max={:.2}",
        deltas.len(),
        mean,
        delta_std,
        delta_max,
    );
    // Discrimination criterion: a smooth-shaded sphere at this
    // viewport produces center-row adjacent-pixel deltas around
    // stddev=0.33 / max=1 (measured empirically before bevel wiring
    // landed). Once bevels modulate per-cell, transitions at cell
    // edges produce sharp cliffs that push max well past that noise
    // floor. Each cube_face_bevel smoothstep ramps over ~12% of a
    // cell face — at layers=3 / 512 px that's ~3 px, so transitions
    // are gradual rather than 1-px cliffs, and `max` is a better
    // indicator than `stddev` alone.
    assert!(
        delta_max > 2.5,
        "center-row max |Δlum| = {:.2} — bevels not visible. \
         Smooth-shaded baseline sits at max=1; a bevelled sphere \
         should easily exceed 2.5.",
        delta_max
    );
    assert!(
        delta_std > 0.5,
        "center-row stddev |Δlum| = {:.2} — even if max spikes, the \
         overall row is too flat. Bevels should at minimum double \
         the smooth-shaded baseline stddev of ~0.33.",
        delta_std
    );
}

/// Deep-zoom regression: spawn the camera at a deep anchor depth so
/// the render frame descends. The ball's world-space position is
/// invariant; the `remap_cube_xform` uniform tells the shader where
/// the ball sits in the CURRENT render frame's local coords. A
/// correct implementation renders the same ball silhouette at
/// spawn-depth N as at spawn-depth 0 (up to sub-pixel differences),
/// validating that deep zoom preserves unified geometry.
#[test]
fn remap_sphere_deep_zoom_preserves_silhouette() {
    let dir = tmp_dir("remap_sphere_silhouette");
    let depths = [1_u8, 8, 16];
    for d in depths {
        let _ = std::fs::remove_file(dir.join(format!("deep_zoom_{d}.png")));
    }
    let mut stats: Vec<(u8, f64, f64, usize)> = Vec::new();
    for d in depths {
        let shot = dir.join(format!("deep_zoom_{d}.png"));
        let dstr = d.to_string();
        let args: Vec<&str> = vec![
            "--render-harness",
            "--remap-sphere-world",
            "--disable-highlight",
            "--harness-width",
            "512",
            "--harness-height",
            "512",
            "--exit-after-frames",
            "60",
            "--timeout-secs",
            "30",
            "--spawn-depth",
            dstr.as_str(),
        ];
        let script = ScriptBuilder::new()
            .wait(5)
            .screenshot(shot.to_string_lossy().as_ref())
            .emit(format!("deep_zoom_{d}"));
        let trace = run(&args, &script);
        assert!(
            trace.exit_success,
            "binary failed at spawn_depth={d}: {}",
            trace.stderr
        );
        assert!(shot.exists(), "missing shot at depth={d}");
        let (w, h, rgba) = load_png(&shot);
        let mask = planet_mask(&rgba, w, h);
        let (_cx, _cy, r_mean, r_std, count) = fit_circle(&mask, w, h);
        eprintln!(
            "deep_zoom depth={d}: {count} px, r_mean={r_mean:.3}, r_std={r_std:.3}"
        );
        stats.push((d, r_mean, r_std, count));
    }
    // Every depth must render a circular silhouette. A deep-zoom
    // break would typically manifest as either total miss
    // (count ≈ 0), a distorted shape (r_std ≫ 1), or an off-center
    // pixel cluster.
    for (d, r_mean, r_std, count) in &stats {
        assert!(
            *count > 30_000,
            "at spawn_depth={d}, only {count} planet pixels — \
             render frame descended past the ball or cube_xform is \
             wrong. r_mean={r_mean:.3}, r_std={r_std:.3}"
        );
        assert!(
            *r_std < 2.0,
            "at spawn_depth={d}, silhouette is not circular \
             (r_std={r_std:.3}). Unified geometry broken at this depth."
        );
    }
    // All depths should render approximately the same ball size —
    // the world-space ball is invariant under zoom, so the
    // silhouette's radius should be the same within a few pixels
    // across all spawn depths.
    let r_ref = stats[0].1;
    for (d, r_mean, _, _) in &stats {
        let diff = (r_mean - r_ref).abs();
        assert!(
            diff < 5.0,
            "at spawn_depth={d}, silhouette radius {r_mean:.2} \
             differs from depth-1 reference {r_ref:.2} by {diff:.2} px. \
             Ball's apparent size should be invariant under zoom."
        );
    }
}

