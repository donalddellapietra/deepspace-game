//! Visual seamlessness sweep for the sphere-world across the
//! Body ⇄ SphereSub transition.
//!
//! Runs `--render-harness` at anchor depths straddling
//! `MIN_SPHERE_SUB_DEPTH` and writes a PNG per depth. For each image
//! we compute the "planet fraction" — pixels that aren't sky. A
//! seamless transition means adjacent depths produce very similar
//! planet fractions. A broken SphereSub (wrong sub-frame metadata,
//! ray transform bugs, shader bugs) produces a visibly different
//! silhouette at the transition depth.
//!
//! The test is **not** a golden-image diff — it's a behaviour-level
//! invariant that deliberately tolerates pixel-level wobble. If the
//! planet fraction jumps by > 10 % at any transition, something is
//! fundamentally wrong with either the Body or the SphereSub path.
//!
//! Also emits the raw PNGs into `tmp/sphere_zoom_seamless/` so you
//! can eyeball what each depth actually rendered.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{run, tmp_dir, ScriptBuilder};
use std::path::PathBuf;

/// Range of zoom depths we sweep. Body covers 3–4, transition lives
/// at 5 (first `SphereSub` per MIN_SPHERE_SUB_DEPTH=3: body + face
/// root + 3 inner = anchor depth 5), deep cases 6–8 exercise
/// SphereSub at non-trivial sub-frame depths.
const DEPTHS: &[u8] = &[3, 4, 5, 6, 7, 8];

/// Camera parked looking straight down at the planet. Gap shrinks
/// with depth so the interaction envelope always reaches the surface.
/// Sphere centered at 1.5, surface at r ≈ 1.80 world units.
fn harness_args(depth: u8, out_png: &str) -> Vec<String> {
    let anchor_cell = 3.0_f64 / (3.0_f64).powi(depth as i32);
    let gap = 12.0 * anchor_cell * 0.6;
    let cam_y = 1.80 + gap;
    vec![
        "--render-harness".to_string(),
        "--sphere-world".to_string(),
        "--spawn-depth".to_string(), depth.to_string(),
        "--spawn-xyz".to_string(), "1.5".to_string(), format!("{cam_y:.6}"), "1.5".to_string(),
        "--spawn-pitch".to_string(), "-1.5707".to_string(),
        "--spawn-yaw".to_string(), "0".to_string(),
        "--harness-width".to_string(), "480".to_string(),
        "--harness-height".to_string(), "320".to_string(),
        "--exit-after-frames".to_string(), "60".to_string(),
        "--timeout-secs".to_string(), "30".to_string(),
        "--suppress-startup-logs".to_string(),
        "--screenshot".to_string(), out_png.to_string(),
    ]
}

/// Read a PNG and return (planet_fraction, total_pixels). Planet
/// fraction = pixels where NOT (b > r && b > g) — anything
/// non-sky-blue. The sphere's grass surface has R,G > B so it counts
/// as planet. Sub-pixel anti-aliasing smears the silhouette but the
/// fraction is stable to a few percent.
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
            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];
            let is_sky = b > r && b > g;
            if !is_sky {
                planet += 1;
            }
        }
    }
    planet as f32 / total as f32
}

#[test]
fn sphere_zoom_seamless_across_sub_frame_transition() {
    let dir = tmp_dir("sphere_zoom_seamless");
    std::fs::create_dir_all(&dir).expect("create tmp dir");
    let mut fractions: Vec<(u8, f32, PathBuf)> = Vec::new();

    for &depth in DEPTHS {
        let png = dir.join(format!("depth_{depth:02}.png"));
        let _ = std::fs::remove_file(&png);
        let png_str = png.to_string_lossy().into_owned();
        let args = harness_args(depth, &png_str);
        let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let script = ScriptBuilder::new().wait(30);
        let trace = run(&args_refs, &script);
        assert!(
            trace.exit_success,
            "depth {depth}: binary did not exit 0\n--- stderr ---\n{}",
            trace.stderr
        );
        assert!(
            png.exists(),
            "depth {depth}: screenshot {} was not written",
            png.display()
        );
        let pf = planet_fraction(&png);
        eprintln!("depth={depth:2}  planet_fraction={pf:.3}  png={}", png.display());
        fractions.push((depth, pf, png));
    }

    // Invariant: adjacent depths should have planet fractions within
    // 10 % of each other. A bigger delta means the render silhouette
    // jumped — transition is not seamless.
    for pair in fractions.windows(2) {
        let (d0, f0, p0) = &pair[0];
        let (d1, f1, p1) = &pair[1];
        let delta = (f1 - f0).abs();
        assert!(
            delta < 0.10,
            "non-seamless transition between depth {d0} and {d1}: \
             planet fraction {f0:.3} → {f1:.3} (delta {delta:.3}). \
             See {} vs {}",
            p0.display(), p1.display(),
        );
    }

    // Sanity: every depth should actually render the planet (non-zero
    // non-sky pixels). Zero = pure-sky frame = render broke entirely.
    for (d, f, p) in &fractions {
        assert!(
            *f > 0.01,
            "depth {d}: planet_fraction = {f:.3} — render produced \
             no planet pixels. {}",
            p.display()
        );
    }
}
