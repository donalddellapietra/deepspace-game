//! End-to-end layer-descent test suite — cubed-sphere variant.
//!
//! Same protocol as `e2e_layer_descent` but rooted in the demo planet
//! (`--sphere-world`, `tree_depth = 30`). Verifies break + probe at
//! multiple `anchor_depth` levels on the PosY face of the sphere.
//!
//! See `docs/testing/e2e-layer-descent-sphere.md` for the protocol.

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{ScriptBuilder, run, tmp_dir, planet_pixel_count_at_row, highlight_glow_pixel_count};

/// Sphere world. `--spawn-on-surface` dispatches to
/// `demo_sphere_surface_spawn` which builds a path-based spawn that
/// tracks the SDF surface at any depth. Pitch is exactly `-π/2` so
/// camera-forward and `probe_down` are the same ray — at deep anchors
/// even a few degrees of tilt can land on a different face of the
/// cubed sphere, producing anchor paths that diverge on the face-slot
/// step and fail the "edit anchor == probe anchor" assertion.
const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--sphere-world",
    "--spawn-on-surface",
    "--spawn-depth",
    "5",
    "--spawn-pitch",
    "-1.5707963",
    "--spawn-yaw",
    "0",
    "--disable-highlight",
    // Default interaction reach (12 anchor cells) floors at the SDF
    // min cell size for sphere, yielding ~0.147 body-frame reach.
    // Each descent-break extends the next probe's path through the
    // cumulative tunnel, so at anchor_depth ≥ 22 the solid on the
    // far side of the accumulated holes sits just past the default
    // reach. Boost reach so the full 20-layer descent lands on
    // solid at every depth.
    "--interaction-radius",
    "36",
    "--harness-width",
    "640",
    "--harness-height",
    "360",
    "--exit-after-frames",
    "2000",
    "--timeout-secs",
    "90",
];

// UI layer = tree_depth − anchor_depth + 1. Sphere `tree_depth = 30`,
// so anchor_depth=5 ⇒ UI layer 26 (the user's requested starting frame).
const TREE_DEPTH: u32 = 30;
const START_ANCHOR_DEPTH: u32 = 5;

/// After breaking a cell directly below the camera, tilt the pitch
/// so the ray enters the new hole. At each tilt angle, the cursor
/// raycast either returns no hit OR returns a hit whose reported
/// hit_point lies **inside** the reported AABB.
///
/// Regression: when the body-frame sphere raycast missed (because
/// the break carved a hole through the surface cell), the pop-loop
/// fallback in `cpu_raycast_in_sphere_frame` re-ran the sphere
/// raycast at root-frame — whose `t` is in world units, but the
/// caller expected body-frame units. Downstream consumers
/// (interaction-radius gate, highlight AABB vs. hit_point check,
/// shader cursor glow) then compared mismatched units and the
/// cursor highlight landed in a cell far from where the next
/// break would edit.
///
/// Fix: `cpu_raycast_in_sphere_frame` now multiplies `hit.t` by
/// `3^pops` so the returned `t` is in the caller's frame (= deepest
/// frame = body-frame) regardless of which level of the frame chain
/// produced the hit.
#[test]
fn sphere_cursor_hit_point_is_inside_aabb_after_wall_dig() {
    let dir = tmp_dir("sphere_wall_cursor");
    let _ = dir;

    // Four tilts sweep through the cone where the ray enters the hole
    // and then either (a) still falls within reach of a wall cell, or
    // (b) goes past reach. Both outcomes are acceptable — the bug is
    // "cursor says hit is here but that 'here' is outside the
    // cell's AABB." So the assertion is per-hit, not per-tilt.
    let tilts = [-1.3f32, -1.1, -0.9, -0.7];
    let mut script = ScriptBuilder::new()
        .emit("spawn")
        .probe_cursor()
        .break_()
        .wait(10);
    for (i, pitch) in tilts.iter().enumerate() {
        script = script
            .pitch(*pitch)
            .wait(5)
            .emit(&format!("tilt_{i}"))
            .probe_cursor();
    }
    script = script.emit("end");

    let harness_args: &[&str] = &[
        "--render-harness",
        "--sphere-world",
        "--spawn-xyz", "1.5", "1.98", "1.5",
        "--spawn-depth", "5",
        "--spawn-pitch", "-1.5",
        "--spawn-yaw", "0",
        "--disable-highlight",
        "--harness-width", "320",
        "--harness-height", "180",
        "--exit-after-frames", "200",
        "--timeout-secs", "20",
        "--interaction-radius", "36",
    ];

    let trace = run(harness_args, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}",
        trace.stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );

    assert!(
        !trace.probe_aabbs.is_empty(),
        "expected at least one HARNESS_PROBE_AABB line; got probes={:?}",
        trace.probes,
    );
    for aabb in &trace.probe_aabbs {
        assert!(
            aabb.inside,
            "cursor hit_point {:?} not inside AABB ({:?} .. {:?}) for anchor {:?} — \
             this means the highlight box would render on a different cell than the \
             next break would edit",
            aabb.hit_point, aabb.aabb_min, aabb.aabb_max, aabb.anchor,
        );
    }
}

#[test]
fn sphere_layer_26_break_below_is_registered_three_ways() {
    let dir = tmp_dir("sphere_layer_26_break_below");
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
        "binary did not exit 0\n--- stderr tail ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
        trace.stdout.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );

    assert_eq!(trace.marks.len(), 2, "expected two marks, got {:?}", trace.marks);
    assert_eq!(trace.marks[0].label, "start");
    let expected_ui = TREE_DEPTH - START_ANCHOR_DEPTH + 1;
    assert_eq!(
        trace.marks[0].ui_layer, expected_ui,
        "must start at UI layer {expected_ui}"
    );
    assert_eq!(trace.marks[0].anchor_depth, START_ANCHOR_DEPTH);
    assert_eq!(trace.marks[1].label, "end");

    assert_eq!(trace.edits.len(), 1, "expected one edit, got {:?}", trace.edits);
    let edit = &trace.edits[0];
    assert_eq!(edit.action, "broke");
    assert!(edit.changed, "break must actually change world state");
    assert_eq!(edit.anchor_depth, START_ANCHOR_DEPTH);
    assert_eq!(
        edit.anchor.len() as u32,
        START_ANCHOR_DEPTH,
        "anchor path length must equal anchor_depth: {:?}",
        edit.anchor,
    );

    assert_eq!(trace.probes.len(), 2, "expected two probes, got {:?}", trace.probes);
    let pre_probe = &trace.probes[0];
    let post_probe = &trace.probes[1];
    assert!(pre_probe.hit, "pre-break probe must hit sphere surface");
    assert!(post_probe.hit, "post-break probe must still hit something");
    assert_eq!(
        pre_probe.anchor, edit.anchor,
        "probe-before-break anchor must equal the broken cell's anchor"
    );
    assert_ne!(
        post_probe.anchor, edit.anchor,
        "probe-after-break should hit a different cell (broken cell is now empty)"
    );

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
fn sphere_layers_26_to_25_descend_and_break() {
    let dir = tmp_dir("sphere_layers_26_to_25");
    let paths = [
        dir.join("layer26_pre.png"),
        dir.join("layer26_post.png"),
        dir.join("layer25_pre.png"),
        dir.join("layer25_post.png"),
    ];
    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
    let [pre26, post26, pre25, post25]: [String; 4] =
        paths.map(|p| p.to_string_lossy().into_owned());

    let script = ScriptBuilder::new()
        .emit("layer_26_start")
        .screenshot(&pre26)
        .probe_down()
        .break_()
        .wait(15)
        .screenshot(&post26)
        .zoom_in(1)
        .teleport_above_last_edit()
        .wait(15)
        .emit("layer_25_start")
        .screenshot(&pre25)
        .probe_down()
        .break_()
        .wait(15)
        .screenshot(&post25)
        .emit("layer_25_end");

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
        trace.stdout.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
    );

    assert_eq!(trace.marks.len(), 3, "expected 3 marks, got {:?}", trace.marks);
    assert_eq!(trace.marks[0].label, "layer_26_start");
    assert_eq!(trace.marks[0].anchor_depth, START_ANCHOR_DEPTH);
    assert_eq!(trace.marks[1].label, "layer_25_start");
    assert_eq!(
        trace.marks[1].anchor_depth, START_ANCHOR_DEPTH + 1,
        "after zoom_in:1 anchor_depth must increment"
    );
    assert_eq!(trace.marks[2].label, "layer_25_end");

    assert_eq!(trace.edits.len(), 2, "expected 2 edits, got {:?}", trace.edits);
    for (i, edit) in trace.edits.iter().enumerate() {
        assert_eq!(edit.action, "broke", "edit {i} action");
        assert!(edit.changed, "break at iteration {i} must succeed");
        let expected_d = START_ANCHOR_DEPTH + i as u32;
        assert_eq!(edit.anchor_depth, expected_d, "edit {i} depth");
        assert_eq!(edit.anchor.len() as u32, expected_d, "edit {i} anchor length");
    }
    assert_ne!(
        trace.edits[1].anchor, trace.edits[0].anchor,
        "second break must target a different cell than the first"
    );

    assert_eq!(trace.probes.len(), 2, "expected 2 probes, got {:?}", trace.probes);
    for (i, probe) in trace.probes.iter().enumerate() {
        assert!(probe.hit, "probe at iteration {i} missed");
        assert_eq!(
            probe.anchor, trace.edits[i].anchor,
            "iteration {i} probe anchor must match the edit anchor",
        );
    }

    for p in [&pre26, &post26, &pre25, &post25] {
        assert!(std::path::Path::new(p).exists(), "screenshot {p} missing");
    }
}

/// Full sphere-face descent: starting at `anchor_depth = 5` (UI layer
/// 26), break + zoom + respawn on surface at every level down to
/// `anchor_depth = 24` — 20 iterations.
///
/// Each iteration verifies:
/// - exactly one `HARNESS_EDIT action=broke changed=true` at this
///   depth,
/// - one `HARNESS_PROBE hit=true` with anchor matching the edit,
/// - a down-view screenshot is captured.
///
/// Note on sky-dominance: the Cartesian version checks that a
/// look-up screenshot is blue-dominant (nested-aperture line of
/// sight). On sphere the GPU render path currently produces a
/// uniform tan fill whenever the camera sits inside the outer
/// shell (a pre-existing issue seen in every sphere capture —
/// `sphere_zoom_invariance` works because it spawns at `y=2.0`,
/// above the shell). Skipped here; restore once that rendering
/// path is fixed.
#[test]
fn sphere_descent_breaks_at_every_layer() {
    const N_LAYERS: u32 = 20;

    let dir = tmp_dir("sphere_descent_breaks");
    let mut screenshot_paths = Vec::<String>::new();
    let mut labels = Vec::<String>::new();

    let mut script = ScriptBuilder::new();
    for i in 0..N_LAYERS {
        let anchor_depth_here = START_ANCHOR_DEPTH + i;
        let label = format!("d{anchor_depth_here}");
        let shot_path = dir
            .join(format!("{label}_down.png"))
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_file(&shot_path);

        script = script
            .emit(&label)
            .screenshot(&shot_path)
            .probe_down()
            .break_()
            .wait(10)
            .zoom_in(1)
            // Sphere faces index slots as (u, v, r) whereas `WorldPos`
            // is Cartesian, so `teleport_above_last_edit`'s slot_index
            // (1,0,1) descent drifts horizontally rather than radially.
            // `respawn_on_surface` re-targets the SDF surface at the
            // new anchor_depth instead.
            .respawn_on_surface()
            .wait(5);

        screenshot_paths.push(shot_path);
        labels.push(label);
    }
    script = script.emit("descent_end");

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}\n--- stdout tail ---\n{}",
        trace.stderr.lines().rev().take(60).collect::<Vec<_>>().join("\n"),
        trace.stdout.lines().rev().take(60).collect::<Vec<_>>().join("\n"),
    );

    assert_eq!(
        trace.marks.len(),
        (N_LAYERS + 1) as usize,
        "expected {} marks, got labels {:?}",
        N_LAYERS + 1,
        trace.marks.iter().map(|m| &m.label).collect::<Vec<_>>(),
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

    assert_eq!(
        trace.edits.len(),
        N_LAYERS as usize,
        "expected {} edits, got {:?}",
        N_LAYERS,
        trace.edits,
    );
    for (i, edit) in trace.edits.iter().enumerate() {
        assert_eq!(edit.action, "broke", "edit {} ({}) action", i, labels[i]);
        assert!(
            edit.changed,
            "break at {} (anchor_depth {}) did not change world state",
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
            edit.anchor.len() as u32,
            START_ANCHOR_DEPTH + i as u32,
            "edit {} ({}) anchor path length",
            i,
            labels[i],
        );
    }

    assert_eq!(
        trace.probes.len(),
        N_LAYERS as usize,
        "expected {} probes, got {:?}",
        N_LAYERS,
        trace.probes,
    );
    for (i, probe) in trace.probes.iter().enumerate() {
        assert!(
            probe.hit,
            "probe at {} (anchor_depth {}) missed",
            labels[i],
            START_ANCHOR_DEPTH + i as u32,
        );
        assert_eq!(
            probe.anchor, trace.edits[i].anchor,
            "layer {} probe anchor doesn't match the cell break removed",
            labels[i],
        );
    }

    for (i, path) in screenshot_paths.iter().enumerate() {
        assert!(
            std::path::Path::new(path).exists(),
            "layer {} screenshot {} missing",
            labels[i],
            path,
        );
    }
}

/// The planet must render with a CURVED (approximately circular)
/// silhouette when viewed from outside, not as a flat face / cube.
///
/// This is the visual contract the sphere locality refactor broke in
/// its first attempt: switching `set_root_kind_face` → `_cartesian`
/// for all sphere frames made the shader dispatch `march_cartesian`
/// on the face subtree, dropping the cube-to-equal-area warp and
/// rendering a flat rectangular terrain patch instead of a planet.
///
/// Test: capture the sphere from above with the full body in view
/// (spawn high above the body cell, looking straight down). Count
/// non-sky pixels in three rows — a band well above the equator, a
/// middle band, and a band well below. A sphere silhouette has
/// mostly sky in the top band (pole is small on screen) and far
/// more planet in the middle (equator fills the view). A flat face
/// renders either mostly sky (looking above it) or a constant-width
/// band — both of which fail this check.
#[test]
fn sphere_silhouette_is_curved_when_viewed_from_outside() {
    let dir = tmp_dir("sphere_silhouette");
    let shot_path = dir.join("planet.png");
    let _ = std::fs::remove_file(&shot_path);
    let shot = shot_path.to_string_lossy().into_owned();

    let script = ScriptBuilder::new().emit("start").screenshot(&shot);

    let harness_args: &[&str] = &[
        "--render-harness",
        "--sphere-world",
        // Place camera well above the body cell. Body at root slot 13
        // spans root [1, 2)³; outer shell at body-local 0.45 = root
        // radius 0.45 around (1.5, 1.5, 1.5), surface at y ≈ 1.95.
        // Spawn at y=2.7 — outside outer shell, looking down at the
        // whole body with sky above.
        "--spawn-xyz", "1.5", "2.7", "1.5",
        "--spawn-depth", "3",
        "--spawn-pitch", "-1.57",
        "--spawn-yaw", "0",
        "--disable-highlight",
        "--disable-overlay",
        "--harness-width", "320",
        "--harness-height", "240",
        "--exit-after-frames", "180",
        "--timeout-secs", "20",
    ];

    let trace = run(harness_args, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}",
        trace.stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );
    assert!(
        std::path::Path::new(&shot).exists(),
        "silhouette screenshot missing: {shot}",
    );

    let top = planet_pixel_count_at_row(&shot, 0.10);
    let mid = planet_pixel_count_at_row(&shot, 0.50);
    let bot = planet_pixel_count_at_row(&shot, 0.90);

    // Core assertion: the middle row must have substantially more
    // planet pixels than the top row. Sphere silhouette: mid ≈ 2-3×
    // top. Flat face (post-broken-refactor screenshot): mid ≈ top
    // (band across whole viewport) or mid < top (sky dominates).
    assert!(
        mid as f32 >= top as f32 * 1.5,
        "planet silhouette not curved at top — middle row {mid} must be ≥1.5× top row {top}. \
         Bottom row {bot}. If mid≈top the face subtree is rendering as an axis-aligned cube \
         instead of a curved sphere.",
    );
    assert!(
        mid > 0,
        "middle row has zero planet pixels — planet is not in view at all",
    );
}

/// At deep sphere anchors the cursor highlight glow must actually
/// render on screen.
///
/// Pre-refactor the sphere highlight used body-frame `vec3` AABB
/// uniforms. Cells past face-subtree depth ~20 collapsed below f32
/// ULP near `body_center ≈ 1.5`, so the shader's `hit_pos ∈ AABB`
/// check never fired — the yellow glow silently disappeared.
///
/// Post-refactor the highlight uses `(render_path, highlight_path)`
/// slot-path matching in the shader: every pixel's walker returns
/// its hit cell's slot sequence and the shader compares it prefix-
/// wise against the highlighted cell's world-root path. No f32
/// precision is involved, so the glow fires reliably at any anchor
/// depth.
///
/// Test: spawn with cursor_locked + highlight ENABLED at deep sphere
/// anchor, render a screenshot, assert yellow glow pixels are
/// present on the surface.
#[test]
fn sphere_highlight_glow_renders_at_deep_anchor() {
    let dir = tmp_dir("sphere_highlight_deep");
    let shot_path = dir.join("highlighted.png");
    let _ = std::fs::remove_file(&shot_path);
    let shot = shot_path.to_string_lossy().into_owned();

    let script = ScriptBuilder::new()
        .emit("start")
        // Wait a few frames for the highlight pipeline to settle —
        // the first frame's upload happens before the highlight raycast.
        .wait(5)
        .screenshot(&shot);

    let harness_args: &[&str] = &[
        "--render-harness",
        "--sphere-world",
        "--spawn-xyz", "1.5", "1.98", "1.5",
        "--spawn-depth", "22",
        "--spawn-pitch", "-1.5",
        "--spawn-yaw", "0",
        // Highlight enabled this time (no --disable-highlight).
        "--disable-overlay",
        "--harness-width", "320",
        "--harness-height", "180",
        "--exit-after-frames", "200",
        "--timeout-secs", "20",
        "--interaction-radius", "200",
    ];

    let trace = run(harness_args, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}",
        trace.stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );
    assert!(
        std::path::Path::new(&shot).exists(),
        "highlight screenshot missing: {shot}",
    );

    // Exclude the 20-pixel screen-center box so crosshair pixels
    // (always ~30 when hit registers) don't satisfy the assertion
    // on their own — we want to count the actual cell glow.
    let glow = highlight_glow_pixel_count(&shot, 20);
    assert!(
        glow > 0,
        "cursor highlight glow not visible at anchor_depth=22: 0 yellow-ish \
         pixels outside the crosshair. The shader's path-based match should \
         fire at deep anchor (no f32 precision wall in the path compare).",
    );
}
