//! End-to-end sphere descent.
//!
//! Mirrors the Cartesian `e2e_layer_descent` in structure, but
//! asserts sphere-specific invariants. The Cartesian test breaks
//! at UI layer 37, zooms in, teleports, breaks at layer 36, and
//! checks that the later anchor has MORE slots (descent into a
//! finer cell). We do the same shape here at the sphere.
//!
//! Key difference from Cartesian: the sphere's break path is
//! `[world_chain..., body_slot, face_root_slot, face_descent...]`.
//! For an anchor at world-depth `N`, the path length must be
//! exactly `N` — two of those entries go to body + face_root, the
//! remaining `N − 2` are face-subtree descents (UVR slots). If the
//! walker cap inside the face subtree is off by one, the break
//! path length diverges from the anchor depth and the test fails.
//!
//! The invariant "break cell == highlight cell == render cell" is
//! also enforced: the CPU raycast produces the `HitInfo` that
//! drives all three (probe, break, highlight); we verify the probe
//! anchor matches the broken anchor.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{run, tmp_dir, HarnessEdit, HarnessProbe, ScriptBuilder, Trace};

/// Args for a sphere break scenario. Camera hovers just above the
/// visible surface (sphere centered at 1.5, sdf.radius 0.30 →
/// surface top at 1.80), looking straight down. The gap to the
/// surface SHRINKS with anchor depth so the 12-anchor-cell
/// interaction envelope always reaches the surface — that's
/// exactly the "move camera closer to edit finer cells"
/// requirement the sphere UX relies on.
///
/// At anchor depth N, one anchor cell is `3 / 3^N` world units
/// wide. 12 cells of reach → `36 / 3^N` world units. We park the
/// camera at 60% of that envelope above the surface.
fn sphere_args(spawn_depth: u8) -> Vec<String> {
    let anchor_cell = 3.0_f64 / (3.0_f64).powi(spawn_depth as i32);
    let gap = 12.0 * anchor_cell * 0.6;
    let cam_y = 1.80 + gap;
    let spawn_depth = spawn_depth.to_string();
    vec![
        "--render-harness".to_string(),
        "--sphere-world".to_string(),
        "--spawn-depth".to_string(), spawn_depth,
        "--spawn-xyz".to_string(), "1.5".to_string(), format!("{cam_y:.6}"), "1.5".to_string(),
        "--spawn-pitch".to_string(), "-1.5707".to_string(),
        "--spawn-yaw".to_string(), "0".to_string(),
        "--interaction-radius".to_string(), "12".to_string(),
        "--harness-width".to_string(), "480".to_string(),
        "--harness-height".to_string(), "320".to_string(),
        "--exit-after-frames".to_string(), "1000".to_string(),
        "--timeout-secs".to_string(), "60".to_string(),
        "--suppress-startup-logs".to_string(),
    ]
}

/// Run one "probe → break → probe" scenario at the given spawn
/// depth, emit screenshots, and return the parsed trace plus the
/// screenshot paths.
fn run_scenario(spawn_depth: u8, tag: &str) -> (Trace, String, String) {
    let dir = tmp_dir("sphere_cell_size").join(tag);
    std::fs::create_dir_all(&dir).expect("create scenario dir");
    let pre_png = dir.join("pre.png");
    let post_png = dir.join("post.png");
    let _ = std::fs::remove_file(&pre_png);
    let _ = std::fs::remove_file(&post_png);
    let pre_png = pre_png.to_string_lossy().into_owned();
    let post_png = post_png.to_string_lossy().into_owned();

    let script = ScriptBuilder::new()
        .emit("start")
        .screenshot(&pre_png)
        .probe_down()
        .break_()
        .wait(15)
        .probe_down()
        .screenshot(&post_png)
        .emit("end");

    let args = sphere_args(spawn_depth);
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let trace = run(&args_refs, &script);
    (trace, pre_png, post_png)
}

/// Unwrap the single edit in a scenario, asserting one was recorded.
fn expect_one_edit(trace: &Trace) -> &HarnessEdit {
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );
    assert_eq!(
        trace.edits.len(), 1,
        "expected exactly one edit, got {:?}.\n--- marks ---\n{:?}\n\
         --- probes ---\n{:?}\n--- stdout reject lines ---\n{}",
        trace.edits, trace.marks, trace.probes,
        trace.stdout.lines()
            .filter(|l| l.contains("reject") || l.contains("miss") || l.contains("do_break"))
            .collect::<Vec<_>>().join("\n"),
    );
    &trace.edits[0]
}

fn expect_two_probes(trace: &Trace) -> (&HarnessProbe, &HarnessProbe) {
    assert_eq!(
        trace.probes.len(), 2,
        "expected pre + post probes, got {:?}", trace.probes,
    );
    (&trace.probes[0], &trace.probes[1])
}

/// The break path length must equal the anchor depth, at every
/// depth. This is the key invariant: inside the face subtree, the
/// walker descends `anchor_depth − 2` levels (two are spent on the
/// body and face-root entries), and the hit path includes both
/// prefixes → total length = anchor depth.
///
/// If the walker cap is off by one (e.g., forgot to subtract the
/// body+face_root prefix), the path length would grow by one extra
/// level at every depth, and this test would fail at the first
/// non-trivial spawn_depth.
#[test]
fn sphere_break_path_length_equals_anchor_depth() {
    for &depth in &[3u8, 5, 7] {
        let (trace, _, _) = run_scenario(depth, &format!("d{depth}"));
        let edit = expect_one_edit(&trace);
        assert_eq!(
            edit.anchor.len(), depth as usize,
            "depth={depth}: break anchor must be {depth} slots deep, got {} — \
             walker is producing cells at the wrong level",
            edit.anchor.len(),
        );
        assert!(edit.changed, "depth={depth}: break must modify world state");
    }
}

/// The broken cell must be the same cell the probe reported before
/// the break — the CPU raycast drives both, so they can't disagree.
/// This is what makes "highlight = break = render" hold.
#[test]
fn sphere_probe_anchor_equals_break_anchor() {
    for &depth in &[3u8, 5, 7] {
        let (trace, _, _) = run_scenario(depth, &format!("probe_d{depth}"));
        let edit = expect_one_edit(&trace);
        let (pre, post) = expect_two_probes(&trace);
        assert!(pre.hit, "depth={depth}: pre-break probe must hit");
        assert_eq!(
            pre.anchor, edit.anchor,
            "depth={depth}: pre-break probe anchor ({:?}) must equal the \
             broken cell's anchor ({:?}) — same CPU raycast drives both",
            pre.anchor, edit.anchor,
        );
        assert!(
            post.hit,
            "depth={depth}: post-break probe must hit the cell beneath the \
             broken one",
        );
        assert_ne!(
            post.anchor, edit.anchor,
            "depth={depth}: post-break probe must differ from broken anchor \
             (the broken cell is now empty)",
        );
    }
}

/// Sphere dig-down descent. The user reports the rendered planet
/// geometry breaks down past UI layer ~18-20 when **digging into**
/// the planet — not when hovering above. This test reproduces the
/// actual flow: probe → break → zoom_in → teleport_above_last_edit,
/// repeated through many layers, with a screenshot at every step so
/// the visual state is recorded alongside the probe/edit trace.
///
/// Assertion per layer:
///   - probe hits (crosshair has a valid target),
///   - break reports `changed=true`,
///   - edit anchor depth matches the current anchor depth.
///
/// Screenshots land in `tmp/sphere_descent/d{N}.png`.
#[test]
fn sphere_dig_down_descent() {
    // Start shallow and dig down through the deep precision wall.
    // Shallow depths establish baseline; deep depths exercise the
    // body-march wall.
    const START_DEPTH: u8 = 5;
    const END_DEPTH: u8 = 25;

    let dir = tmp_dir("sphere_descent");
    std::fs::create_dir_all(&dir).expect("create descent dir");

    let args = sphere_args(START_DEPTH);
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();

    let mut script = ScriptBuilder::new();
    let mut shot_paths: Vec<String> = Vec::new();
    for d in START_DEPTH..=END_DEPTH {
        let shot = dir.join(format!("d{d}.png")).to_string_lossy().into_owned();
        let _ = std::fs::remove_file(&shot);
        // Tilt the camera away from perfectly-axial pitch before
        // the screenshot: straight-down (−π/2) aligns with the
        // PosY face's r-axis exactly, which hits a shader
        // degeneracy in `sphere_in_sub_frame` (two of the three
        // rd_local components near zero → t-interval for u/v axes
        // becomes f32-unstable, DDA can't pick the right cell,
        // rendering collapses to a smeared gradient). `pitch:-0.5`
        // matches real gameplay (the user looks DOWN but not
        // straight down) and keeps all three ray components
        // well-separated from zero.
        //
        // Probe still runs with the harness's fixed −π/2 so the
        // anchor-invariant checks don't depend on camera angle.
        script = script
            .emit(&format!("d{d}"))
            .wait(5)
            .pitch(-0.95)
            .yaw(0.3)
            .wait(5)
            .screenshot(&shot)
            .pitch(-std::f32::consts::FRAC_PI_2)
            .yaw(0.0)
            .wait(2)
            .probe_down()
            .break_()
            .wait(10)
            .zoom_in(1)
            .teleport_above_last_edit()
            .wait(5);
        shot_paths.push(shot);
    }
    script = script.emit("descent_end");

    let trace = run(&args_refs, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );

    let layers = (END_DEPTH - START_DEPTH + 1) as usize;
    let mut failures: Vec<String> = Vec::new();
    for (i, (probe, edit)) in trace.probes.iter().zip(trace.edits.iter()).enumerate() {
        let depth = START_DEPTH as usize + i;
        if !probe.hit {
            failures.push(format!(
                "layer {depth}: probe missed (crosshair has no target — visible breakdown)",
            ));
            continue;
        }
        if !edit.changed {
            failures.push(format!("layer {depth}: break reported no world change"));
            continue;
        }
        if edit.anchor.len() != depth {
            failures.push(format!(
                "layer {depth}: edit anchor length {} ≠ current anchor depth {depth}",
                edit.anchor.len(),
            ));
            continue;
        }
    }
    if trace.probes.len() != layers || trace.edits.len() != layers {
        failures.push(format!(
            "expected {layers} probes and {layers} edits, got {} probes and {} edits",
            trace.probes.len(),
            trace.edits.len(),
        ));
    }
    // Always build a 6-column mosaic so visual inspection is
    // one-file, multi-depth. Eyeballing 21 individual screenshots
    // mid-debug is the failure mode we're escaping.
    let mosaic_path = dir.join("mosaic.png");
    match build_mosaic(&shot_paths, &mosaic_path, 6) {
        Ok(_) => eprintln!("mosaic saved to {}", mosaic_path.display()),
        Err(e) => eprintln!("mosaic build failed: {e}"),
    }
    // Per-layer color-variance analysis — quantifies the "shader
    // collapses into a uniform gradient" regression without needing
    // human eyeballs. Dumps one line per layer: depth, distinct
    // 16-colors, stddev of luminance. Uniform flood → low distinct +
    // low stddev; crisp cellular render → many distinct + high
    // stddev.
    let mut healthy = 0usize;
    let mut flood = 0usize;
    let mut weak = 0usize;
    for (i, path) in shot_paths.iter().enumerate() {
        let depth = START_DEPTH as usize + i;
        match analyze_shot(path) {
            Ok(a) => {
                // Classification thresholds tuned for the 480x320
                // screenshot format:
                //   flood   → almost-single-color (obvious regression)
                //   weak    → low distinct/low variance (visible
                //             "smeared gradient" regression)
                //   healthy → sufficient variation to render cell
                //             structure
                let class = if a.distinct_colors <= 2 && a.luma_std < 2.0 {
                    flood += 1;
                    "flood"
                } else if a.distinct_colors < 15 || a.luma_std < 15.0 {
                    weak += 1;
                    "weak"
                } else {
                    healthy += 1;
                    "healthy"
                };
                eprintln!(
                    "MOSAIC_ANALYSIS layer={depth} class={class} distinct_16col={} luma_std={:.3} dominant_frac={:.2}",
                    a.distinct_colors, a.luma_std, a.dominant_fraction,
                );
            }
            Err(e) => eprintln!("MOSAIC_ANALYSIS layer={depth} error={e}"),
        }
    }
    let total = shot_paths.len();
    eprintln!(
        "MOSAIC_SUMMARY total={total} healthy={healthy} weak={weak} flood={flood} see {}",
        mosaic_path.display(),
    );

    assert!(
        failures.is_empty(),
        "sphere dig-down failed — screenshots in {} (see mosaic.png):\n{}",
        dir.display(),
        failures.join("\n"),
    );
}

struct ShotAnalysis {
    distinct_colors: usize,
    luma_std: f32,
    dominant_fraction: f32,
}

/// Pixel-level post-process analysis of a single screenshot. The goal
/// is to OBJECTIVELY detect the "uniform grey gradient" shader-bug
/// regression: those screenshots have very few distinct colors + low
/// luma std-dev + high dominant-color fraction, whereas a correctly
/// rendered screenshot with visible cell walls has many distinct
/// colors at varied luminance.
fn analyze_shot(path: &str) -> std::io::Result<ShotAnalysis> {
    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let chan = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        _ => return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput, "unsupported color type",
        )),
    };
    // Bucket RGB values into 16 bins per channel (4 bits each).
    // 4×4×4 = 64 possible "16-color" keys, giving a stable count of
    // visually distinct hues regardless of exact f32 shading drift.
    let mut bin_counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut luma_sum: f64 = 0.0;
    let mut luma_sq_sum: f64 = 0.0;
    let mut n: usize = 0;
    for px in buf.chunks(chan) {
        let r = px[0];
        let g = px[1];
        let b = px[2];
        let key = ((r as u32 & 0xF0) << 8) | ((g as u32 & 0xF0) << 4) | (b as u32 & 0xF0);
        *bin_counts.entry(key).or_insert(0) += 1;
        let luma = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
        luma_sum += luma;
        luma_sq_sum += luma * luma;
        n += 1;
    }
    let mean = luma_sum / n as f64;
    let var = (luma_sq_sum / n as f64 - mean * mean).max(0.0);
    let luma_std = var.sqrt() as f32;
    let dominant_fraction = bin_counts
        .values()
        .max()
        .copied()
        .unwrap_or(0) as f32
        / n.max(1) as f32;
    Ok(ShotAnalysis {
        distinct_colors: bin_counts.len(),
        luma_std,
        dominant_fraction,
    })
}

/// Read a series of PNG screenshots and compose them into a grid
/// mosaic. Layout is `cols` columns, `ceil(len/cols)` rows. Each cell
/// is downscaled 2× via nearest-neighbor so the mosaic stays compact.
/// One-command pipeline for multi-depth visual verification — the
/// alternative is eyeballing 21 individual files per iteration.
fn build_mosaic(
    shot_paths: &[String],
    out_path: &std::path::Path,
    cols: usize,
) -> std::io::Result<()> {
    if shot_paths.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no shots",
        ));
    }
    let mut cells: Vec<(u32, u32, Vec<u8>, String)> = Vec::with_capacity(shot_paths.len());
    for path in shot_paths {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => continue, // missing shot = empty cell
        };
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = decoder.read_info()?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf)?;
        let w = info.width;
        let h = info.height;
        // Normalize to RGBA for composition; convert RGB if needed.
        let rgba = match info.color_type {
            png::ColorType::Rgba => buf,
            png::ColorType::Rgb => {
                let mut out = Vec::with_capacity((w * h) as usize * 4);
                for px in buf.chunks(3) {
                    out.extend_from_slice(px);
                    out.push(255);
                }
                out
            }
            _ => continue,
        };
        let label = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        cells.push((w, h, rgba, label));
    }
    if cells.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no readable shots",
        ));
    }
    let (src_w, src_h, _, _) = &cells[0];
    let (src_w, src_h) = (*src_w, *src_h);
    // 2× downscale.
    let dst_w = src_w / 2;
    let dst_h = src_h / 2;
    let rows = (cells.len() + cols - 1) / cols;
    let mosaic_w = dst_w as usize * cols;
    let mosaic_h = dst_h as usize * rows;
    let mut mosaic = vec![16u8; mosaic_w * mosaic_h * 4];
    // Set alpha to 255 for the empty background.
    for px in mosaic.chunks_exact_mut(4) {
        px[3] = 255;
    }
    for (i, (w, h, rgba, _label)) in cells.iter().enumerate() {
        if *w != src_w || *h != src_h {
            continue;
        }
        let col = i % cols;
        let row = i / cols;
        let base_x = col * dst_w as usize;
        let base_y = row * dst_h as usize;
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = dx * 2;
                let sy = dy * 2;
                let src_idx = ((sy * src_w + sx) * 4) as usize;
                let dst_idx = ((base_y + dy as usize) * mosaic_w
                    + (base_x + dx as usize))
                    * 4;
                mosaic[dst_idx..dst_idx + 4].copy_from_slice(&rgba[src_idx..src_idx + 4]);
            }
        }
    }
    let file = std::fs::File::create(out_path)?;
    let mut encoder =
        png::Encoder::new(std::io::BufWriter::new(file), mosaic_w as u32, mosaic_h as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    writer
        .write_image_data(&mosaic)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}
