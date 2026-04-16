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

use harness::{ScriptBuilder, run, tmp_dir};

// Plain world with 40 layers. Spawn-depth 4 == UI layer 37
// per `docs/gotchas/layer-vs-depth.md` (zoom_level = tree_depth -
// anchor_depth + 1).
const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--plain-world",
    "--plain-layers",
    "40",
    "--spawn-depth",
    "4",
    "--spawn-pitch",
    "-1.5707",
    "--spawn-yaw",
    "0",
    "--harness-width",
    "640",
    "--harness-height",
    "360",
    "--exit-after-frames",
    "120",
    "--timeout-secs",
    "10",
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
