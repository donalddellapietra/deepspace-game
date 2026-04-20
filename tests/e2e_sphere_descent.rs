//! End-to-end sphere descent.
//!
//! Verifies that on the cubed-sphere world, the cursor's highlighted
//! cell is the cell that breaks, AND the broken cell is the cell the
//! shader rendered at that position. A single truth across edit,
//! highlight, and render.
//!
//! The test scripts a probe-down (CPU raycast) + break (edit via same
//! raycast) + probe-down-again, and asserts the trace:
//!
//! - The probe before the break must hit a cell.
//! - The break must succeed with the same anchor path.
//! - The probe after the break must hit a DIFFERENT cell (the original
//!   is now empty — the ray falls through to the next non-empty cell
//!   along the same ray).

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{run, tmp_dir, ScriptBuilder};

// Sphere preset, camera just above the +Y pole looking straight down.
// Shallow spawn depth + large interaction radius so the ray lands at a
// visible LOD cell regardless of the depth the test runs at.
const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--sphere-world",
    "--spawn-depth", "5",
    // Sphere: center at 1.5, sdf.radius = 0.30 → the visible
    // surface top is at y = 1.80. We spawn 0.02 world-units above
    // it (well within the 12-anchor-cell interaction envelope at
    // depth 5, where one anchor cell is 3/243 ≈ 0.012 world wide).
    "--spawn-xyz", "1.5", "1.82", "1.5",
    "--spawn-pitch", "-1.5707",
    "--spawn-yaw", "0",
    // 12 anchor-sized cells of reach — matches the Cartesian
    // default interaction envelope. Cells are anchor-sized (via
    // `cs_edit_depth()` cap), so the projected hit cell is
    // always ≥ `pixel_density / 12` ≈ 20 px on screen; sub-pixel
    // edits are geometrically impossible within this range.
    "--interaction-radius", "12",
    "--harness-width", "640",
    "--harness-height", "480",
    "--exit-after-frames", "400",
    "--timeout-secs", "45",
    "--suppress-startup-logs",
];

#[test]
fn sphere_probe_hits_then_break_removes_same_cell() {
    let dir = tmp_dir("sphere_probe_break");
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

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr,
        trace.stdout.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
    );

    assert_eq!(trace.marks.len(), 2, "expected start + end marks, got {:?}", trace.marks);
    assert_eq!(trace.marks[0].label, "start");
    assert_eq!(trace.marks[1].label, "end");

    assert_eq!(trace.edits.len(), 1, "expected one edit, got {:?}", trace.edits);
    let edit = &trace.edits[0];
    assert_eq!(edit.action, "broke");
    assert!(edit.changed, "break must actually change world state");
    assert!(
        !edit.anchor.is_empty(),
        "sphere break anchor must have depth > 0: {:?}", edit.anchor,
    );

    assert_eq!(
        trace.probes.len(), 2,
        "expected pre + post probes, got {:?}", trace.probes,
    );
    let pre = &trace.probes[0];
    let post = &trace.probes[1];
    assert!(pre.hit, "pre-break probe must hit the sphere surface");
    assert_eq!(
        pre.anchor, edit.anchor,
        "pre-break probe anchor must equal the broken cell's anchor — \
         the same CPU raycast drives both",
    );
    assert!(
        post.hit,
        "post-break probe must still hit something (the cell beneath \
         the broken one)",
    );
    assert_ne!(
        post.anchor, edit.anchor,
        "post-break probe anchor must differ from the broken cell",
    );

    assert!(
        std::path::Path::new(&pre_png).exists(),
        "pre-break screenshot {pre_png} missing",
    );
    assert!(
        std::path::Path::new(&post_png).exists(),
        "post-break screenshot {post_png} missing",
    );
}
