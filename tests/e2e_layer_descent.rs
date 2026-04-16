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
