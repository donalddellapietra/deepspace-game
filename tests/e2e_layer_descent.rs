//! End-to-end layer-descent test suite.
//!
//! Starting point: a single-layer test at UI layer 37 that exercises
//! the full verification chain:
//!
//! 1. `emit` a `start` marker.
//! 2. Screenshot the pre-break state.
//! 3. `probe_down` — CPU raycast straight down, records the cell the
//!    camera is about to remove.
//! 4. `break` — actual edit. Records the same cell via `HARNESS_EDIT`.
//! 5. `probe_down` again — should hit a DIFFERENT cell now that the
//!    first one is empty.
//! 6. Screenshot the post-break state.
//! 7. `emit` an `end` marker.
//!
//! The test then asserts the trace matches the expected shape.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{ScriptBuilder, run, sky_dominance_top_half, tmp_dir};

// Plain world with 40 layers. Spawn-depth 4 == UI layer 37
// per `docs/gotchas/layer-vs-depth.md` (zoom_level = tree_depth -
// anchor_depth + 1).
//
// `--spawn-xyz 1.5 1.01 1.5` bypasses `bootstrap::plain_surface_spawn`
// (which targets the dirt/grass boundary at y≈0.95 and carves an air
// pocket inside the ground) and places the camera one depth-4 cell
// above the grass surface at y=1.0. From there, looking up sees open
// sky; looking down sees grass.
const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--plain-world",
    "--plain-layers",
    "40",
    "--spawn-depth",
    "4",
    "--spawn-xyz",
    "1.5",
    "1.01",
    "1.5",
    "--spawn-pitch",
    "-1.5707",
    "--spawn-yaw",
    "0",
    "--disable-highlight",
    "--harness-width",
    "640",
    "--harness-height",
    "360",
    "--exit-after-frames",
    "1500",
    "--timeout-secs",
    "60",
];

#[test]
fn layer_37_break_below_is_registered_three_ways() {
    let dir = tmp_dir("layer_37_break_below");
    let pre_png = dir.join("pre.png");
    let post_png = dir.join("post.png");
    // Remove leftovers so "file exists" is a real signal.
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

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );

    // Marks: start + end.
    assert_eq!(trace.marks.len(), 2, "expected two marks, got {:?}", trace.marks);
    assert_eq!(trace.marks[0].label, "start");
    assert_eq!(trace.marks[0].ui_layer, 37, "must start at UI layer 37");
    assert_eq!(trace.marks[0].anchor_depth, 4);
    assert_eq!(trace.marks[1].label, "end");

    // Exactly one edit: the break.
    assert_eq!(trace.edits.len(), 1, "expected one edit, got {:?}", trace.edits);
    let edit = &trace.edits[0];
    assert_eq!(edit.action, "broke");
    assert!(edit.changed, "break must actually change world state");
    assert_eq!(edit.ui_layer, 37);
    assert_eq!(
        edit.anchor.len(),
        4,
        "anchor path depth must equal edit_depth at layer 37: {:?}",
        edit.anchor
    );

    // Two probes: one before break, one after.
    assert_eq!(trace.probes.len(), 2, "expected two probes, got {:?}", trace.probes);
    let pre_probe = &trace.probes[0];
    let post_probe = &trace.probes[1];
    assert!(pre_probe.hit, "pre-break probe must hit solid ground");
    assert!(post_probe.hit, "post-break probe must still hit something (the cell below)");
    assert_eq!(
        pre_probe.anchor, edit.anchor,
        "probe-before-break anchor must equal the broken cell's anchor"
    );
    assert_ne!(
        post_probe.anchor, edit.anchor,
        "probe-after-break should hit a different cell (the broken cell is now empty)"
    );

    // Screenshots landed on disk.
    assert!(
        std::path::Path::new(&pre_png).exists(),
        "pre-break screenshot {pre_png} missing"
    );
    assert!(
        std::path::Path::new(&post_png).exists(),
        "post-break screenshot {post_png} missing"
    );
}

#[test]
fn layers_37_to_36_descend_and_break() {
    let dir = tmp_dir("layers_37_to_36");
    let paths = [
        dir.join("layer37_pre.png"),
        dir.join("layer37_post.png"),
        dir.join("layer36_pre.png"),
        dir.join("layer36_post.png"),
    ];
    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
    let [pre37, post37, pre36, post36]: [String; 4] =
        paths.map(|p| p.to_string_lossy().into_owned());

    let script = ScriptBuilder::new()
        // Layer 37
        .emit("layer_37_start")
        .screenshot(&pre37)
        .probe_down()
        .break_()
        .wait(15)
        .screenshot(&post37)
        // Descend to layer 36: zoom_in changes depth only; teleport
        // positions the camera inside the bottom child of the cell
        // we just broke.
        .zoom_in(1)
        .teleport_above_last_edit()
        .wait(15)
        // Layer 36
        .emit("layer_36_start")
        .screenshot(&pre36)
        .probe_down()
        .break_()
        .wait(15)
        .screenshot(&post36)
        .emit("layer_36_end");

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
    );

    // Marks: 3 (layer_37_start, layer_36_start, layer_36_end).
    // Note: we assert on anchor_depth, not ui_layer. `ui_layer =
    // tree_depth - anchor_depth + 1`, and break_block currently
    // subdivides a uniform node which bumps tree_depth. That keeps
    // ui_layer visually stuck at 37 across the zoom-in. anchor_depth
    // is the deterministic truth.
    assert_eq!(trace.marks.len(), 3, "expected 3 marks, got {:?}", trace.marks);
    assert_eq!(trace.marks[0].label, "layer_37_start");
    assert_eq!(trace.marks[0].anchor_depth, 4);
    assert_eq!(trace.marks[1].label, "layer_36_start");
    assert_eq!(
        trace.marks[1].anchor_depth, 5,
        "after zoom_in:1 anchor_depth must go from 4 to 5"
    );
    assert_eq!(trace.marks[2].label, "layer_36_end");

    // Edits: exactly two breaks, one per layer.
    assert_eq!(trace.edits.len(), 2, "expected 2 edits, got {:?}", trace.edits);
    assert_eq!(trace.edits[0].action, "broke");
    assert!(trace.edits[0].changed, "layer-37 break must succeed");
    assert_eq!(trace.edits[0].anchor_depth, 4);
    assert_eq!(trace.edits[0].anchor.len(), 4, "layer-37 edit anchor depth");

    assert_eq!(trace.edits[1].action, "broke");
    assert!(trace.edits[1].changed, "layer-36 break must succeed");
    assert_eq!(trace.edits[1].anchor_depth, 5);
    assert_eq!(
        trace.edits[1].anchor.len(), 5,
        "layer-36 edit anchor must be one level deeper than layer-37's"
    );
    assert_ne!(
        trace.edits[1].anchor, trace.edits[0].anchor,
        "layer-36 break must target a different cell than layer-37's break"
    );

    // Probes: two (one before each break), both must hit and match the
    // corresponding edit's anchor.
    assert_eq!(trace.probes.len(), 2, "expected 2 probes, got {:?}", trace.probes);
    assert!(trace.probes[0].hit, "layer-37 probe must hit solid ground");
    assert_eq!(trace.probes[0].anchor_depth, 4);
    assert_eq!(
        trace.probes[0].anchor, trace.edits[0].anchor,
        "layer-37 probe anchor must match what break removed"
    );
    assert!(trace.probes[1].hit, "layer-36 probe must hit solid ground");
    assert_eq!(trace.probes[1].anchor_depth, 5);
    assert_eq!(
        trace.probes[1].anchor, trace.edits[1].anchor,
        "layer-36 probe anchor must match what break removed"
    );

    // Screenshots landed on disk.
    for p in [&pre37, &post37, &pre36, &post36] {
        assert!(
            std::path::Path::new(p).exists(),
            "screenshot {p} missing"
        );
    }
}

/// Per-layer verifications, running as one harness process: for each
/// of `N_LAYERS` iterations starting at `anchor_depth=4` (UI layer
/// 37), look up → screenshot → assert sky visible; look down → probe
/// → break → assert changed; zoom_in + teleport to the next layer.
///
/// Asserts, per layer:
///   - sky pixel-dominance above threshold on the look-up screenshot,
///   - exactly one break with `changed=true`,
///   - probe anchor matches edit anchor.
///
/// Failure message names the failing layer so a regression at
/// layer 35 says "layer 35", not "layer <unknown>".
#[test]
fn descent_sees_sky_and_breaks_at_every_layer() {
    // Full descent: start at anchor_depth=4 (UI layer 37), break +
    // zoom + teleport 37 times, ending at anchor_depth=40 (UI layer
    // 1) — one break per layer, all the way down.
    const N_LAYERS: u32 = 37;
    // A sky-dominant aperture must cover at least this fraction of
    // the top half. Permissive: at deep layers the nested aperture
    // subtends less of the view. Tighten once we have data.
    const SKY_THRESHOLD: f32 = 0.05;
    // Starting anchor_depth. Layer labels below are purely mnemonic.
    const START_ANCHOR_DEPTH: u32 = 4;

    let dir = tmp_dir("descent_sees_sky_and_breaks");
    let mut sky_paths = Vec::<String>::new();
    let mut labels = Vec::<String>::new();

    let mut script = ScriptBuilder::new();
    for i in 0..N_LAYERS {
        // Label is the nominal "UI layer" assuming monotonic descent
        // (see the 2-layer test doc about why actual ui_layer drifts).
        let anchor_depth_here = START_ANCHOR_DEPTH + i;
        let label = format!("d{anchor_depth_here}");
        let sky_path = dir
            .join(format!("{label}_sky.png"))
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_file(&sky_path);

        script = script
            .emit(&label)
            .look_up()
            .wait(5)
            .screenshot(&sky_path)
            .look_down()
            .wait(5)
            .probe_down()
            .break_()
            .wait(10)
            .zoom_in(1)
            .teleport_above_last_edit()
            .wait(5);

        sky_paths.push(sky_path);
        labels.push(label);
    }
    script = script.emit("descent_end");

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace
            .stdout
            .lines()
            .rev()
            .take(60)
            .collect::<Vec<_>>()
            .join("\n"),
    );

    // Marks: N per-layer + 1 descent_end.
    assert_eq!(
        trace.marks.len(),
        (N_LAYERS + 1) as usize,
        "expected {} marks, got labels {:?}",
        N_LAYERS + 1,
        trace.marks.iter().map(|m| &m.label).collect::<Vec<_>>()
    );
    for (i, mark) in trace.marks.iter().take(N_LAYERS as usize).enumerate() {
        assert_eq!(mark.label, labels[i], "mark {i} label mismatch");
        assert_eq!(
            mark.anchor_depth,
            START_ANCHOR_DEPTH + i as u32,
            "mark {} ({}) anchor_depth should be {}, got {}",
            i,
            labels[i],
            START_ANCHOR_DEPTH + i as u32,
            mark.anchor_depth,
        );
    }
    assert_eq!(trace.marks.last().unwrap().label, "descent_end");

    // Edits: one per layer, all changed=true, anchor_depth increasing.
    assert_eq!(
        trace.edits.len(),
        N_LAYERS as usize,
        "expected {} edits, got {:?}",
        N_LAYERS,
        trace.edits
    );
    for (i, edit) in trace.edits.iter().enumerate() {
        assert_eq!(edit.action, "broke", "edit {} ({}) action", i, labels[i]);
        assert!(
            edit.changed,
            "break at layer {} (anchor_depth {}) did not change world state",
            labels[i],
            START_ANCHOR_DEPTH + i as u32,
        );
        assert_eq!(
            edit.anchor_depth,
            START_ANCHOR_DEPTH + i as u32,
            "edit {} ({}) anchor_depth",
            i,
            labels[i],
        );
        assert_eq!(
            edit.anchor.len(),
            (START_ANCHOR_DEPTH + i as u32) as usize,
            "edit {} ({}) anchor path length",
            i,
            labels[i],
        );
    }

    // Probes: one per layer, all hit, matching corresponding edit.
    assert_eq!(
        trace.probes.len(),
        N_LAYERS as usize,
        "expected {} probes, got {:?}",
        N_LAYERS,
        trace.probes
    );
    for (i, probe) in trace.probes.iter().enumerate() {
        assert!(
            probe.hit,
            "probe at layer {} (anchor_depth {}) missed",
            labels[i],
            START_ANCHOR_DEPTH + i as u32,
        );
        assert_eq!(
            probe.anchor, trace.edits[i].anchor,
            "layer {} probe anchor doesn't match the cell break removed",
            labels[i],
        );
    }

    // Sky screenshots: each must be on disk and have above-threshold
    // sky-dominance in the top half.
    for (i, path) in sky_paths.iter().enumerate() {
        assert!(
            std::path::Path::new(path).exists(),
            "layer {} sky screenshot {} missing",
            labels[i],
            path,
        );
        let frac = sky_dominance_top_half(path);
        assert!(
            frac >= SKY_THRESHOLD,
            "layer {} sky screenshot {}: sky-dominance {:.3} below threshold {}",
            labels[i],
            path,
            frac,
            SKY_THRESHOLD,
        );
    }
}
