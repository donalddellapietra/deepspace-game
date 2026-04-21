//! Stars visibility test suite.
//!
//! Positive demonstration that distant star cubes scattered across
//! the 26 non-center root slots render as visible dots against the
//! sky gradient, when viewed from the camera's depth-20 anchor at
//! world-center `(1.5, 1.5, 1.5)`. Every star takes 19 ribbon pops
//! to reach — the same deep-ribbon path that the d21 descent test
//! caught before the shader's `ray_dir`-unscaled pop transform
//! landed.
//!
//! This is a *visibility* test: it asserts that star-colored
//! pixels appear in each look direction. The d-sky descent suite
//! (`tests/e2e_layer_descent.rs`) remains the precision regression
//! gate with competing near-camera geometry.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{ScriptBuilder, run, tmp_dir};

// Tree depth 40 with camera anchor 20 gives 19 ribbon pops to
// reach any root-level sibling slot.
const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--stars-world",
    "--plain-layers",
    "40",
    "--spawn-depth",
    "20",
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

/// Fraction of pixels in `path` that look like a warm-white star.
/// The star palette color is `(255, 240, 200)` — after shading
/// the face-lit pixels cluster around `(192-224, 192-208, 176-208)`
/// with `r ≥ b` consistently. The sky gradient is `b > r` (blue
/// dominant) so sky never matches. Grass is `g > r`, so we gate
/// on `g ≤ r + 8` to exclude it too.
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
            if r >= b && r >= 170 && g <= r.saturating_add(8) {
                hits += 1;
            }
            total += 1;
        }
    }
    if total == 0 { 0.0 } else { hits as f32 / total as f32 }
}

/// Look in 6 directions spanning the full sphere. Every direction
/// should catch at least one star (>= 0.1% of the frame —
/// individual root-slot stars subtend tiny angular areas at this
/// camera depth).
#[test]
fn camera_sees_stars_in_every_direction() {
    const STAR_THRESHOLD: f32 = 0.001;

    let dir = tmp_dir("stars_visibility");
    // yaw, pitch pairs. Pitch > 0 = up, < 0 = down. Yaw: 0 is
    // default forward; +π/2 rotates right. Cover the +Y hemisphere
    // plus the four cardinals plus a slight-down pitch that should
    // catch the planet top (but not below the horizon where no
    // stars live).
    let cases: &[(&str, f32, f32)] = &[
        ("up",         0.0,                               1.3),
        ("up_forward", 0.0,                               0.7),
        ("east",       std::f32::consts::FRAC_PI_2,       0.3),
        ("north",      0.0,                               0.3),
        ("south",      std::f32::consts::PI,              0.3),
        ("west",       -std::f32::consts::FRAC_PI_2,      0.3),
    ];

    let mut paths = Vec::<(String, String, f32)>::new();
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
        paths.push(((*label).to_string(), p, *pitch));
    }

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(60).collect::<Vec<_>>().join("\n"),
    );

    for (label, path, _pitch) in &paths {
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

/// Single-direction regression: looking up should see the
/// `(1,2,1)` sky stars cluster. Fails with a single-line message
/// that names the direction.
#[test]
fn camera_sees_star_above() {
    const STAR_THRESHOLD: f32 = 0.003; // up-view catches ~0.8% star pixels

    let dir = tmp_dir("stars_above");
    let path = dir.join("up.png").to_string_lossy().into_owned();
    let _ = std::fs::remove_file(&path);

    let script = ScriptBuilder::new()
        .yaw(0.0)
        .pitch(1.3)
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
        "looking up: star-pixel fraction {:.4} below threshold {} ({path})",
        frac,
        STAR_THRESHOLD,
    );
}
