//! End-to-end regression for the cube→sphere remap cursor edit path.
//!
//! Launches the game with `--remap-sphere-world`, captures a baseline
//! screenshot, issues a `break` command through the script harness,
//! captures a second screenshot, and asserts the pixels under the
//! crosshair changed. This validates the full cursor pipeline:
//! `frame_aware_raycast` → `cpu_raycast_in_remap_sphere_frame` →
//! `break_block` → `upload_tree`.
//!
//! Mirrors `remap_sphere_silhouette.rs` for harness setup; this one
//! is about edits, not silhouette shape.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{run, tmp_dir, ScriptBuilder};

const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--remap-sphere-world",
    "--disable-highlight",
    "--harness-width",
    "256",
    "--harness-height",
    "256",
    "--exit-after-frames",
    "180",
    "--timeout-secs",
    "45",
    "--force-edit-depth",
    "6",
];

/// Load a PNG and return (w, h, rgba buffer).
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

/// Number of pixels in a centered square window of side `2*radius+1`
/// whose RGB differs from the `before` image. A sensitive diff: any
/// per-channel delta ≥ 8 counts as "changed".
fn changed_pixels_in_window(
    before: &[u8],
    after: &[u8],
    w: usize,
    h: usize,
    radius: usize,
) -> usize {
    let cx = w / 2;
    let cy = h / 2;
    let x0 = cx.saturating_sub(radius);
    let x1 = (cx + radius + 1).min(w);
    let y0 = cy.saturating_sub(radius);
    let y1 = (cy + radius + 1).min(h);
    let mut changed = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 4;
            let dr = (before[i] as i32 - after[i] as i32).abs();
            let dg = (before[i + 1] as i32 - after[i + 1] as i32).abs();
            let db = (before[i + 2] as i32 - after[i + 2] as i32).abs();
            if dr.max(dg).max(db) >= 8 {
                changed += 1;
            }
        }
    }
    changed
}

#[test]
fn remap_sphere_break_modifies_pixels_under_crosshair() {
    let dir = tmp_dir("remap_sphere_edit");
    let before_path = dir.join("before.png");
    let after_path = dir.join("after.png");
    let _ = std::fs::remove_file(&before_path);
    let _ = std::fs::remove_file(&after_path);

    // Baseline screenshot, break, another screenshot. `wait` frames
    // give the upload + next render a chance to settle.
    let script = ScriptBuilder::new()
        .wait(15)
        .screenshot(before_path.to_string_lossy().as_ref())
        .wait(5)
        .break_()
        .wait(30)
        .screenshot(after_path.to_string_lossy().as_ref())
        .emit("edit_done");

    let trace = run(HARNESS_ARGS, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace
            .stdout
            .lines()
            .rev()
            .take(80)
            .collect::<Vec<_>>()
            .join("\n"),
    );
    assert!(
        before_path.exists(),
        "before screenshot missing: {}",
        before_path.display()
    );
    assert!(
        after_path.exists(),
        "after screenshot missing: {}",
        after_path.display()
    );

    let (w, h, before_rgba) = load_png(&before_path);
    let (w2, h2, after_rgba) = load_png(&after_path);
    assert_eq!((w, h), (w2, h2), "screenshot dimensions differ");

    // Require at least one harness edit event marking the break as
    // successful. If `changed=false` the raycast found no hit or the
    // edit was refused — neither counts as a cursor-path fix.
    let broke_ok = trace
        .edits
        .iter()
        .any(|e| e.action == "broke" && e.changed);
    assert!(
        broke_ok,
        "no successful `broke` edit in harness trace; edits = {:?}",
        trace.edits,
    );

    // The break should carve visible voxels from the crosshair region.
    // A 21×21 window around image center sees ~400 pixels of sphere
    // surface. Require at least ~10 pixels changed — generous floor
    // that a noise-only frame diff wouldn't cross.
    let changed = changed_pixels_in_window(&before_rgba, &after_rgba, w, h, 10);
    assert!(
        changed >= 10,
        "expected visible pixel change near crosshair after break; got {changed}"
    );
}
