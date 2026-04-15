#[cfg(not(target_arch = "wasm32"))]
#[test]
fn startup_render_stays_above_50_fps() {
    let output = run_game(&[
        "--disable-overlay",
        "--spawn-depth", "17",
        "--run-for-secs", "2",
        "--timeout-secs", "6",
        "--max-frame-gap-ms", "250",
        "--frame-gap-warmup-frames", "30",
        "--min-fps", "50",
        "--min-cadence-fps", "20",
        "--fps-warmup-frames", "4",
        "--cadence-warmup-frames", "4",
    ]);
    assert_perf_ok(&output);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn zoom_transition_to_layer_18_does_not_freeze() {
    let output = run_game(&[
        "--disable-overlay",
        "--spawn-depth", "17",
        "--script", "wait:8,zoom_out:12,wait:1000",
        "--run-for-secs", "6",
        "--timeout-secs", "10",
        "--max-frame-gap-ms", "250",
        "--frame-gap-warmup-frames", "30",
        "--min-fps", "50",
        "--min-cadence-fps", "20",
        "--fps-warmup-frames", "8",
        "--cadence-warmup-frames", "8",
    ]);
    assert_perf_ok(&output);
}

#[cfg(not(target_arch = "wasm32"))]
fn run_game(args: &[&str]) -> std::process::Output {
    use std::process::Command;

    let exe = env!("CARGO_BIN_EXE_deepspace-game");
    Command::new(exe)
        .args(args)
        .output()
        .expect("failed to launch deepspace-game binary")
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_perf_ok(output: &std::process::Output) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    if is_sandboxed_gui_startup_blocked(&stderr) {
        eprintln!("render_perf: skipping in sandboxed GUI environment");
        return;
    }
    assert!(
        stderr.contains("startup_perf frame="),
        "perf test did not reach measured frames; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("test_runner: perf summary"),
        "perf test did not print perf summary; stderr:\n{stderr}"
    );
    assert!(
        output.status.success(),
        "render perf test failed with status {:?}; stderr:\n{stderr}",
        output.status.code(),
    );
}

fn is_sandboxed_gui_startup_blocked(stderr: &str) -> bool {
    let has_no_frames = !stderr.contains("startup_perf frame=");
    let has_no_callbacks = !stderr.contains("startup_perf callback:");
    let has_launchservices_failure =
        stderr.contains("scheduleApplicationNotification")
        || stderr.contains("Connection Invalid error for service com.apple.hiservices-xpcservice")
        || stderr.contains("Error received in message reply handler: Connection invalid");
    let timed_out_before_perf =
        stderr.contains("wall-clock timeout hit before perf test completed")
        || stderr.contains("wall-clock timeout hit before min-fps test completed");
    let webview_never_created = stderr.contains("timed run ended without webview creation");
    has_no_frames && has_no_callbacks && has_launchservices_failure
        && (timed_out_before_perf || webview_never_created)
}
