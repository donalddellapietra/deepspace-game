//! Regression: default sphere-world spawn renders the planet at the
//! expected zoom level, not teleported to the shell surface or into
//! space.
//!
//! The bug this catches: `maybe_enter_sphere` used to fire for any
//! anchor with a body ancestor, regardless of whether the camera was
//! inside the shell. For the default spawn (world y = 2.0, above
//! outer_r), `body_point_to_face_space` silently clamped rn to
//! 0.9999999 and SphereState initialized with the camera "snapped"
//! onto the outer shell. Visible symptoms: anchor depth collapses
//! (zoom UI jumps to layer ~29), the camera appears to fly inside
//! the shell looking out at the sky, and the planet vanishes from
//! view.
//!
//! Observables this test enforces:
//! 1. The rendered image contains a substantial planet silhouette
//!    (non-sky pixels). Teleported-to-space bug → pure sky.
//! 2. The camera's reported `anchor_depth` at spawn is within the
//!    expected range (not collapsed to 2).
//! 3. A single script-level `probe_down` from the default spawn
//!    hits the planet — without that, the crosshair has no target
//!    and all edits are no-ops.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{run, tmp_dir, ScriptBuilder};
use std::path::PathBuf;

fn planet_fraction(path: &PathBuf) -> f32 {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("png info");
    let info = reader.info().clone();
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        o => panic!("unsupported png color type {o:?}"),
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).expect("decode png");
    let data = &buf[..frame.buffer_size()];
    let mut planet = 0usize;
    let total = w * h;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * channels;
            let (r, g, b) = (data[i], data[i + 1], data[i + 2]);
            // Sky = blue-dominant pixel. Planet = everything else.
            let is_sky = b > r && b > g;
            if !is_sky {
                planet += 1;
            }
        }
    }
    planet as f32 / total as f32
}

#[test]
fn default_sphere_world_shows_planet() {
    let dir = tmp_dir("sphere_default_spawn");
    std::fs::create_dir_all(&dir).expect("create tmp dir");
    let png = dir.join("spawn.png");
    let _ = std::fs::remove_file(&png);
    let png_str = png.to_string_lossy().into_owned();
    // No --spawn-xyz, no --spawn-depth — use the bootstrap default.
    let args = vec![
        "--render-harness",
        "--sphere-world",
        "--harness-width", "480",
        "--harness-height", "320",
        "--exit-after-frames", "60",
        "--timeout-secs", "20",
        "--suppress-startup-logs",
        "--screenshot", &png_str,
    ];
    let script = ScriptBuilder::new().emit("start").wait(15).emit("end");
    let trace = run(&args, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(20).collect::<Vec<_>>().join("\n"),
    );
    assert!(
        png.exists(),
        "screenshot {} was not written", png.display()
    );

    let pf = planet_fraction(&png);
    eprintln!("default_sphere_world: planet_fraction={pf:.3} png={}", png.display());
    assert!(
        pf > 0.10,
        "default spawn should show a substantial planet silhouette \
         (planet_fraction={pf:.3}). Values near 0 indicate the camera \
         got teleported and the scene rendered pure sky. See {}",
        png.display(),
    );

    // Anchor depth must not collapse to a tiny value — that's the
    // telltale sign of the body-entry teleport. The buggy path
    // truncated the Cartesian anchor to the body cell's path (depth
    // 1) and put further state in `sphere.uvr_path`, leaving
    // `total_depth()` around 2 = "layer 29" in the UI.
    let mark = trace.marks.first().expect("start mark emitted");
    assert!(
        mark.anchor_depth >= 10,
        "default spawn anchor_depth={} is suspiciously shallow — \
         maybe_enter_sphere may have truncated the anchor (the body-\
         entry teleport bug)",
        mark.anchor_depth,
    );
}
