//! Phase 2 acceptance test: visual proof that the X-axis modular
//! wrap fires inside `WrappedPlanet` root frames.
//!
//! Strategy: stand on the default WrappedPlanet spawn (planet-frame
//! ~ (1.0, 1.5, 0.166)), aim along +X (yaw=π/2 — looking east along
//! the longitude axis) with pitch=-0.5 (~28° below horizon). The
//! camera is 1.17 planet-frame units above the grass top — far
//! enough that without the wrap, a ray going east hits the
//! planet's east frame edge and exits as sky long before reaching
//! the slab. With the wrap, rays exiting the active region's east
//! edge re-enter at the west face and continue over the slab; the
//! slab silhouette becomes visible at the bottom of the frame.
//!
//! Threshold: the "wrap fires" view must show >= 5% non-sky pixels
//! at 480x320. Phase 1 (no-wrap) showed near-pure sky from this
//! pose; 5% is well above that floor and below the empirically
//! observed Phase 2 fraction (~12% slab coverage when the wrap
//! lands the slab in the lower-left portion of the frame).

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
fn temp_png(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "deepspace-{label}-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos(),
    ));
    path
}

#[cfg(not(target_arch = "wasm32"))]
fn run_game(args: &[&str]) -> std::process::Output {
    use std::process::Command;
    let exe = env!("CARGO_BIN_EXE_deepspace-game");
    Command::new(exe)
        .args(args)
        .output()
        .expect("failed to launch deepspace-game binary")
}

/// Fraction of pixels in the screenshot that are NOT sky-blue.
/// Mirrors the convention of `tests/e2e_layer_descent::planet_fraction`:
/// pixels where `b > r && b > g` are sky, everything else is slab.
#[cfg(not(target_arch = "wasm32"))]
fn non_sky_fraction(path: &Path) -> f32 {
    use png::ColorType;
    use png::Decoder;

    let file = std::fs::File::open(path).expect("open png");
    let decoder = Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().expect("read png info");
    let info = reader.info().clone();
    let channels = match info.color_type {
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        other => panic!("unsupported png color type {other:?}"),
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).expect("decode png frame");
    let data = &buf[..frame.buffer_size()];

    let total = (info.width * info.height) as usize;
    let mut planet = 0usize;
    for y in 0..info.height as usize {
        for x in 0..info.width as usize {
            let i = (y * info.width as usize + x) * channels;
            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];
            let is_sky = b > r && b > g;
            if !is_sky {
                planet += 1;
            }
        }
    }
    if total == 0 { 0.0 } else { planet as f32 / total as f32 }
}

#[cfg(not(target_arch = "wasm32"))]
fn is_sandboxed_gui_startup_blocked(stderr: &str) -> bool {
    let has_no_frames = !stderr.contains("startup_perf frame=");
    let has_no_callbacks = !stderr.contains("startup_perf callback:");
    has_no_frames && has_no_callbacks && !stderr.contains("render_harness")
}

/// Phase 2 visual acceptance: with yaw=π/2 (looking east along the
/// longitude axis) and a slightly-down pitch, the X-wrap inside the
/// WrappedPlanet root makes the slab visible across a non-trivial
/// fraction of the frame. Without the wrap branch landing in the
/// shader, the same view would be near-pure sky beyond the thin
/// active extent.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn yaw_east_along_longitude_shows_wrapped_slab() {
    let png = temp_png("phase2-yaw90-wrap");
    let output = run_game(&[
        "--render-harness",
        "--disable-overlay",
        "--wrapped-planet-world",
        "--spawn-yaw",
        "1.5707",
        "--spawn-pitch",
        "-0.5",
        "--harness-width",
        "480",
        "--harness-height",
        "320",
        "--exit-after-frames",
        "30",
        "--timeout-secs",
        "8",
        "--suppress-startup-logs",
        "--screenshot",
        png.to_str().expect("utf8 path"),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if is_sandboxed_gui_startup_blocked(&stderr) {
        eprintln!("wrapped_planet_wrap: skipping in sandboxed GUI environment");
        return;
    }
    assert!(
        output.status.success(),
        "render harness failed; stderr:\n{stderr}",
    );
    assert!(
        png.exists(),
        "screenshot not produced at {}; stderr:\n{stderr}",
        png.display(),
    );

    let frac = non_sky_fraction(&png);
    assert!(
        frac >= 0.05,
        "expected >= 5% non-sky pixels (wrap fires) but got {:.3}; stderr:\n{}",
        frac,
        stderr,
    );
}
