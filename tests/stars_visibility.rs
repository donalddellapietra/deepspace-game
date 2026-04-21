//! Stars visibility test suite.
//!
//! Positive demonstration that distant occupants (root-level star
//! cubes 22 ribbon pops from the camera's depth-23 anchor) render
//! at full brightness in every cardinal direction.
//!
//! Note: this test is a *visibility* gate, not a precision
//! regression gate. The stars-world scene is sparse — DDA rays
//! have no competing near-camera geometry to mis-hit, so a
//! precision break doesn't cause mis-hits; rays either hit the
//! expected star or fall through to sky. For the precision
//! regression gate see `e2e_layer_descent::descent_sees_sky_and_
//! breaks_at_every_layer`, which exercises grass/dirt boundaries
//! at every depth 4..40 and caught the d21 precision failure
//! before the `ray_dir`-unscaled pop transform landed.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{ScriptBuilder, run, tmp_dir};

// Total tree depth. Chosen to force deep ribbons: the camera anchors
// at depth 23 (= total_depth - 1) with stars at ancestor levels
// 1, 5, 10, 15, 20, 24 — so a ray to the root-level star pops 22
// ribbon entries, a range that explicitly failed before the fix.
const TOTAL_DEPTH: &str = "24";

// Render harness args. `--spawn-depth` deliberately mirrors the
// bootstrap default (total_depth - 1 = 23), just being explicit.
const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--stars-world",
    "--plain-layers",
    TOTAL_DEPTH,
    "--spawn-depth",
    "23",
    "--spawn-xyz",
    "1.5",
    "1.35",
    "1.5",
    "--disable-highlight",
    "--harness-width",
    "640",
    "--harness-height",
    "360",
    "--exit-after-frames",
    "600",
    "--timeout-secs",
    "45",
];

/// Fraction of pixels in `path` that look like a star — warm
/// yellow. Matches `(r, g, b)` where `r >= g > b` and `r > 150`.
/// The star palette color is `(255, 220, 80)`; after shading it
/// renders around `(224, 224, 176)` with `r = g > b`. The sky
/// gradient is `b > g > r` (blue dominant) so it never matches.
/// Grass is `g > r` so it's excluded too.
fn star_pixel_fraction(path: impl AsRef<std::path::Path>) -> f32 {
    let file = std::fs::File::open(path.as_ref())
        .unwrap_or_else(|e| panic!("open {}: {e}", path.as_ref().display()));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read png header");
    let info = reader.info().clone();
    let (width, height) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unsupported png color type {other:?}"),
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).expect("decode png frame");
    let data = &buf[..frame.buffer_size()];

    let mut hits = 0usize;
    let mut total = 0usize;
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * channels;
            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];
            // Warm yellow: r ≥ g > b, with r bright enough to
            // exclude shadowed grass (which is desaturated).
            if r >= g && g > b && r > 150 {
                hits += 1;
            }
            total += 1;
        }
    }
    if total == 0 { 0.0 } else { hits as f32 / total as f32 }
}

/// Look in each of 5 cardinal "up-hemisphere" directions in turn
/// and assert stars are visible every time.
///
/// The stars-world places stars at root-slots `(1,2,1)`, `(0,1,1)`,
/// `(2,1,1)`, `(1,1,0)`, `(1,1,2)` — directly up, left, right,
/// back, front — plus smaller stars at deeper levels. Looking
/// yaw=θ / pitch=π/4 catches the horizon-ish bright stars in each
/// compass direction, while looking straight up catches the +Y
/// giant plus any deeper-level +Y stars.
#[test]
fn camera_sees_stars_in_every_direction() {
    const STAR_THRESHOLD: f32 = 0.02;

    let dir = tmp_dir("stars_visibility");
    let cases: &[(&str, f32, f32)] = &[
        ("up",     0.0,                              std::f32::consts::FRAC_PI_2 - 0.05),
        ("north",  0.0,                              0.0),
        ("east",   std::f32::consts::FRAC_PI_2,      0.0),
        ("south",  std::f32::consts::PI,             0.0),
        ("west",   -std::f32::consts::FRAC_PI_2,     0.0),
    ];

    let mut paths = Vec::<(String, String)>::new();
    let mut script = ScriptBuilder::new();
    for (label, yaw, pitch) in cases {
        let p = dir.join(format!("{label}.png")).to_string_lossy().into_owned();
        let _ = std::fs::remove_file(&p);
        script = script
            .yaw(*yaw)
            .pitch(*pitch)
            .wait(3)
            .screenshot(&p)
            .emit(*label);
        paths.push(((*label).to_string(), p));
    }

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(60).collect::<Vec<_>>().join("\n"),
    );

    for (label, path) in &paths {
        assert!(
            std::path::Path::new(path).exists(),
            "direction {label} screenshot {path} missing",
        );
        let frac = star_pixel_fraction(path);
        assert!(
            frac >= STAR_THRESHOLD,
            "direction {label} screenshot {path}: star-pixel fraction {:.4} below threshold {}",
            frac,
            STAR_THRESHOLD,
        );
    }
}

/// Per-direction regression: if this fails at any single depth
/// it tells you which star is misbehaving. Separate from the
/// combined test above so a localized regression gives a
/// single-line error message, not a truncated multi-assert trace.
#[test]
fn camera_sees_star_above() {
    const STAR_THRESHOLD: f32 = 0.05;

    let dir = tmp_dir("stars_above");
    let path = dir.join("up.png").to_string_lossy().into_owned();
    let _ = std::fs::remove_file(&path);

    let script = ScriptBuilder::new()
        .yaw(0.0)
        .pitch(std::f32::consts::FRAC_PI_2 - 0.05)
        .wait(3)
        .screenshot(&path)
        .emit("up");

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(60).collect::<Vec<_>>().join("\n"),
    );
    assert!(std::path::Path::new(&path).exists(), "screenshot {path} missing");

    let frac = star_pixel_fraction(&path);
    assert!(
        frac >= STAR_THRESHOLD,
        "looking straight up: star-pixel fraction {:.4} below threshold {} ({path})",
        frac,
        STAR_THRESHOLD,
    );
}
