//! End-to-end layer-descent test for the cubed-sphere planet.
//!
//! Spawns on the surface at `--spawn-depth 5` (UI layer 26 for a
//! `tree_depth=30` planet) and verifies the break + zoom + respawn
//! descent cycle at every layer down to anchor_depth=24. This is the
//! regression gate for the sphere's unified march + precision-safe
//! anchor pipeline.
//!
//! See `docs/testing/e2e-layer-descent-sphere.md` for the protocol
//! (ported from sphere-debug).

#[path = "e2e_layer_descent/harness.rs"]
mod harness;

use harness::{
    distinct_non_sky_color_count, planet_pixel_count_at_row, run, ScriptBuilder,
};

const HARNESS_ARGS: &[&str] = &[
    "--render-harness",
    "--sphere-world",
    "--spawn-on-surface",
    "--spawn-depth", "5",
    "--spawn-pitch", "-1.5707963",
    "--spawn-yaw", "0",
    "--disable-highlight",
    // Default interaction reach (12 cells) floors at SDF min-cell,
    // yielding ~0.147 body-frame reach. Each descent-break extends
    // the next probe's path through the cumulative tunnel, so deep
    // breaks need extra reach.
    "--interaction-radius", "36",
    "--harness-width", "640",
    "--harness-height", "360",
    "--exit-after-frames", "2000",
    "--timeout-secs", "90",
];

const START_ANCHOR_DEPTH: u32 = 5;

// ─────────────────────────────────────────────── visual tests

/// The default `--sphere-world` spawn (no --spawn-xyz override)
/// must render a correct sphere. This is exactly what the user
/// sees when running `scripts/dev.sh --sphere-world` — the default
/// bootstrap spawn, not a test-configured spawn. Regression guard
/// for "gray cube edge at spawn" where every pixel falls back to
/// LOD-terminal representative block because the render frame was
/// mis-aligned with the camera anchor.
#[test]
fn default_sphere_world_spawn_renders_correctly() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tmp")
        .join("default_spawn");
    std::fs::create_dir_all(&dir).expect("create tmp dir");
    let shot_path = dir.join("default.png");
    let _ = std::fs::remove_file(&shot_path);
    let shot = shot_path.to_string_lossy().into_owned();

    let script = ScriptBuilder::new().emit("start").screenshot(&shot);

    // Default sphere-world spawn — same path as
    // `scripts/dev.sh --sphere-world`.
    let args: &[&str] = &[
        "--render-harness",
        "--sphere-world",
        "--disable-highlight",
        "--disable-overlay",
        "--harness-width", "320",
        "--harness-height", "240",
        "--exit-after-frames", "180",
        "--timeout-secs", "20",
    ];

    let trace = run(args, &script);
    assert!(trace.exit_success, "binary did not exit 0\n{}", trace.stderr);
    assert!(std::path::Path::new(&shot).exists(), "screenshot missing");

    let distinct = distinct_non_sky_color_count(&shot);
    assert!(
        distinct > 30,
        "default sphere-world spawn only produced {distinct} distinct colors — \
         looks like a uniform failure fill (the 'gray cube edge' bug).",
    );
}

/// The planet must render as a CURVED silhouette from a camera
/// outside the body — not a flat face, not a cube, and not a
/// uniform gray fill from a broken raycast. This is the baseline
/// smoke test for the sphere render path.
///
/// Method: render the planet from above, then:
/// 1. Confirm there's SOME rendered content (fail on uniform-fill
///    from broken raycast).
/// 2. Assert the middle row has substantially more planet pixels
///    than the top row — a sphere bulges in the middle.
/// 3. Assert there are at least several dozen distinct non-sky
///    colors (a real lit surface has lighting gradients + grass/
///    dirt/stone biome; a uniform LOD-terminal fallback has one).
#[test]
fn sphere_renders_as_curved_silhouette_from_above() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tmp")
        .join("sphere_curved");
    std::fs::create_dir_all(&dir).expect("create tmp dir");
    let shot_path = dir.join("planet.png");
    let _ = std::fs::remove_file(&shot_path);
    let shot = shot_path.to_string_lossy().into_owned();

    let script = ScriptBuilder::new().emit("start").screenshot(&shot);

    // Camera above the body, looking down. Body at root slot 13 =
    // [1, 2]³, outer shell at radius 0.45 from center (1.5, 1.5,
    // 1.5). Spawn at y=2.7 — well outside the outer shell (y=1.95),
    // looking straight down to frame the whole planet.
    let args: &[&str] = &[
        "--render-harness",
        "--sphere-world",
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

    let trace = run(args, &script);
    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}",
        trace.stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );
    assert!(
        std::path::Path::new(&shot).exists(),
        "screenshot missing: {shot}",
    );

    let distinct = distinct_non_sky_color_count(&shot);
    assert!(
        distinct > 30,
        "image has only {distinct} distinct non-sky colors — \
         looks like a uniform failure fill (broken raycast). \
         A real sphere render has lighting + voxel-grid variance.",
    );

    let top = planet_pixel_count_at_row(&shot, 0.10);
    let mid = planet_pixel_count_at_row(&shot, 0.50);
    let bot = planet_pixel_count_at_row(&shot, 0.90);

    // Round silhouette: middle row fills much more than top.
    assert!(
        mid > 0,
        "middle row has zero planet pixels — planet not in view",
    );
    assert!(
        mid as f32 >= top as f32 * 1.5,
        "planet silhouette not curved — mid row {mid} must be >= 1.5 * top row {top}. \
         bot {bot}. If mid ≈ top, the face subtree is rendering as a flat cube.",
    );
}


/// Break + probe at layer 26. Basic sanity — one break, one probe,
/// three-way verification (edit+probe+screenshot).
#[test]
fn sphere_layer_26_break_below_is_registered() {
    let script = ScriptBuilder::new()
        .emit("start")
        .probe_down()
        .break_()
        .wait(15)
        .probe_down()
        .emit("end");

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}",
        trace.stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );

    assert_eq!(trace.marks.len(), 2, "expected 2 marks, got {:?}", trace.marks);
    assert_eq!(trace.marks[0].label, "start");
    assert_eq!(trace.marks[0].anchor_depth, START_ANCHOR_DEPTH);
    assert_eq!(trace.marks[1].label, "end");

    assert_eq!(trace.edits.len(), 1, "expected 1 edit, got {:?}", trace.edits);
    let edit = &trace.edits[0];
    assert_eq!(edit.action, "broke");
    assert!(edit.changed, "break must change world state");
    assert_eq!(edit.anchor_depth, START_ANCHOR_DEPTH);

    assert_eq!(trace.probes.len(), 2);
    let pre = &trace.probes[0];
    let post = &trace.probes[1];
    assert!(pre.hit, "pre-break probe must hit sphere surface");
    assert!(post.hit, "post-break probe must still hit something");
    assert_eq!(
        pre.anchor, edit.anchor,
        "probe-before-break anchor must equal the broken cell's anchor",
    );
    assert_ne!(
        post.anchor, edit.anchor,
        "probe-after-break should hit a different cell (broken cell is now empty)",
    );
}

/// Full 30-layer descent: break + zoom_in + respawn_on_surface at
/// every anchor_depth from 5 to 34. Each iteration verifies the
/// break changes world state, the probe anchor matches the edit
/// anchor, and anchor_depth increments correctly. A failure at
/// the very last layer (anchor_depth 34, at/past f32 precision
/// wall for body-frame math) is acceptable — the test's purpose
/// is to confirm the descent pipeline works across the useful
/// depth range, not to push past f32 breakdown.
#[test]
fn sphere_descent_breaks_at_every_layer() {
    const N_LAYERS: u32 = 30;

    let mut script = ScriptBuilder::new();
    let mut labels = Vec::<String>::new();
    for i in 0..N_LAYERS {
        let ad = START_ANCHOR_DEPTH + i;
        let label = format!("d{ad}");
        script = script
            .emit(&label)
            .probe_down()
            .break_()
            .wait(10)
            .zoom_in(1)
            .respawn_on_surface()
            .wait(5);
        labels.push(label);
    }
    script = script.emit("descent_end");

    let trace = run(HARNESS_ARGS, &script);

    assert!(
        trace.exit_success,
        "binary did not exit 0\n--- stderr tail ---\n{}",
        trace.stderr.lines().rev().take(60).collect::<Vec<_>>().join("\n"),
    );

    assert_eq!(trace.marks.len(), (N_LAYERS + 1) as usize, "mark count");
    for (i, mark) in trace.marks.iter().take(N_LAYERS as usize).enumerate() {
        assert_eq!(mark.label, labels[i], "mark {i} label");
        assert_eq!(
            mark.anchor_depth,
            START_ANCHOR_DEPTH + i as u32,
            "mark {} ({}) anchor_depth should be {}, got {}",
            i, labels[i], START_ANCHOR_DEPTH + i as u32, mark.anchor_depth,
        );
    }
    assert_eq!(trace.marks.last().unwrap().label, "descent_end");

    assert_eq!(trace.edits.len(), N_LAYERS as usize, "edit count");
    for (i, edit) in trace.edits.iter().enumerate() {
        assert_eq!(edit.action, "broke", "edit {} action", i);
        assert!(
            edit.changed,
            "break at layer {} (anchor_depth {}) didn't change world",
            labels[i], START_ANCHOR_DEPTH + i as u32,
        );
        assert_eq!(
            edit.anchor_depth, START_ANCHOR_DEPTH + i as u32,
            "edit {} anchor_depth", i,
        );
    }

    assert_eq!(trace.probes.len(), N_LAYERS as usize, "probe count");
    for (i, probe) in trace.probes.iter().enumerate() {
        assert!(
            probe.hit,
            "probe at {} (anchor_depth {}) missed",
            labels[i], START_ANCHOR_DEPTH + i as u32,
        );
        assert_eq!(
            probe.anchor, trace.edits[i].anchor,
            "layer {} probe anchor doesn't match edit anchor",
            labels[i],
        );
    }
}
