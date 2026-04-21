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
        "--exit-after-frames".to_string(), "200".to_string(),
        "--timeout-secs".to_string(), "30".to_string(),
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

/// Deep-depth sphere sanity. The user reports the rendered planet
/// geometry breaks down past UI layer ~18-20 even though the CPU
/// raycast in-pipeline tests pass to depth 30. This test pushes the
/// SAME harness-driven scenario the shallow sphere tests use
/// (camera parked just above the planet surface, looking down,
/// probe + break) but through the full deep-depth range that the
/// user actually zooms to at runtime.
///
/// Break path length at anchor depth N must be N (walker descends
/// to the edit-anchor cell, no off-by-one in the face-subtree cap).
/// Probe must hit at every depth — a miss here is the visible
/// breakdown: no crosshair target, no highlight, no breakable cell.
///
/// **CURRENTLY FAILING** — fails at depth ≥ 10 because the camera
/// hovers above the shell so SphereState never initializes; the
/// render + probe fall back to the body-march path, which has a
/// precision wall around layer 8-10. Fix in progress: route through
/// SphereSub even when the camera is outside the shell (synthesize
/// UVR state from where the crosshair points).
#[test]
fn sphere_probe_and_break_at_deep_depth() {
    let mut failures: Vec<String> = Vec::new();
    for &depth in &[10u8, 15, 20, 25, 30] {
        let (trace, _, _) = run_scenario(depth, &format!("deep_d{depth}"));
        if !trace.exit_success {
            failures.push(format!(
                "depth {depth}: binary did not exit 0\n--- stderr ---\n{}\n\
                 --- stdout tail ---\n{}",
                trace.stderr,
                trace.stdout.lines().rev().take(20).collect::<Vec<_>>().join("\n"),
            ));
            continue;
        }
        // Probe at the crosshair — the single source of truth that
        // drives the visible highlight box, the break action, and
        // what the shader should be rendering at that pixel.
        let Some(pre_probe) = trace.probes.first() else {
            failures.push(format!("depth {depth}: no pre-break probe recorded"));
            continue;
        };
        if !pre_probe.hit {
            failures.push(format!(
                "depth {depth}: probe missed (crosshair has no target — visible breakdown)",
            ));
            continue;
        }
        if pre_probe.anchor.len() != depth as usize {
            failures.push(format!(
                "depth {depth}: probe anchor length {} ≠ anchor depth {depth}",
                pre_probe.anchor.len(),
            ));
            continue;
        }
        if trace.edits.len() != 1 {
            failures.push(format!(
                "depth {depth}: expected exactly one edit, got {}",
                trace.edits.len(),
            ));
            continue;
        }
        let edit = &trace.edits[0];
        if !edit.changed {
            failures.push(format!(
                "depth {depth}: break action reported no world change",
            ));
            continue;
        }
        if edit.anchor.len() != depth as usize {
            failures.push(format!(
                "depth {depth}: edit anchor length {} ≠ anchor depth {depth} (walker cap off-by-one?)",
                edit.anchor.len(),
            ));
            continue;
        }
        if pre_probe.anchor != edit.anchor {
            failures.push(format!(
                "depth {depth}: probe anchor {:?} ≠ edit anchor {:?}",
                pre_probe.anchor, edit.anchor,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "deep-depth sphere scenarios failed:\n{}",
        failures.join("\n"),
    );
}
