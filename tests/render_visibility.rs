#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct ImageDiff {
    changed_frac: f64,
    bbox: Option<(usize, usize, usize, usize)>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>,
}

#[cfg(not(target_arch = "wasm32"))]
fn visibility_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static VISIBILITY_MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> =
        std::sync::OnceLock::new();
    VISIBILITY_MUTEX
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn deep_plain_center_raycast_has_positive_t_at_representative_depths() {
    let _guard = visibility_test_lock();
    for depth in [39u8, 36, 34, 32, 22, 20, 18, 16] {
        let png = temp_png(&format!("raycast-{depth}"));
        let output = run_game(&[
            "--render-harness",
            "--disable-overlay",
            "--plain-world",
            "--plain-layers",
            "40",
            "--spawn-depth",
            &depth.to_string(),
            "--harness-width",
            "960",
            "--harness-height",
            "540",
            "--screenshot",
            png.to_str().expect("utf8 path"),
            "--exit-after-frames",
            "2",
            "--timeout-secs",
            "4",
        ]);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if is_sandboxed_gui_startup_blocked(&stderr) {
            eprintln!("render_visibility: skipping in sandboxed GUI environment");
            return;
        }
        assert!(
            output.status.success(),
            "center raycast run failed at depth {depth}; stderr:\n{stderr}"
        );
        let t = parse_frame_raycast_t(&stderr)
            .unwrap_or_else(|| panic!("missing frame_raycast_hit t at depth {depth}; stderr:\n{stderr}"));
        assert!(t > 0.0, "non-positive center raycast t={t} at depth {depth}");
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn deep_plain_break_produces_visible_change_at_all_depths() {
    let _guard = visibility_test_lock();
    for depth in [32u8, 20, 11, 4] {
        let before = temp_png(&format!("break-before-{depth}"));
        let after = temp_png(&format!("break-after-{depth}"));
        let screenshot = run_game(&[
            "--render-harness",
            "--disable-overlay",
            "--disable-highlight",
            "--plain-world",
            "--plain-layers",
            "40",
            "--spawn-depth",
            &depth.to_string(),
            "--harness-width",
            "960",
            "--harness-height",
            "540",
            "--screenshot",
            before.to_str().expect("utf8 path"),
            "--exit-after-frames",
            "40",
            "--timeout-secs",
            "4",
        ]);
        let screenshot_stderr = String::from_utf8_lossy(&screenshot.stderr);
        if is_sandboxed_gui_startup_blocked(&screenshot_stderr) {
            eprintln!("render_visibility: skipping in sandboxed GUI environment");
            return;
        }
        assert!(
            screenshot.status.success(),
            "pre-break screenshot failed at depth {depth}; stderr:\n{screenshot_stderr}"
        );

        let break_output = run_game(&[
            "--render-harness",
            "--disable-overlay",
            "--disable-highlight",
            "--plain-world",
            "--plain-layers",
            "40",
            "--spawn-depth",
            &depth.to_string(),
            "--harness-width",
            "960",
            "--harness-height",
            "540",
            "--script",
            "break,wait:8",
            "--screenshot",
            after.to_str().expect("utf8 path"),
            "--exit-after-frames",
            "40",
            "--timeout-secs",
            "4",
        ]);
        let break_stderr = String::from_utf8_lossy(&break_output.stderr);
        assert!(
            break_output.status.success(),
            "break screenshot failed at depth {depth}; stderr:\n{break_stderr}"
        );
        assert!(
            break_stderr.contains("do_break: changed=true"),
            "break did not modify the world at depth {depth}; stderr:\n{break_stderr}"
        );

        let diff = image_diff(&load_png_rgb(&before), &load_png_rgb(&after));
        assert!(
            diff.changed_frac > 0.001,
            "break should produce a visible change at depth {depth}: {diff:?}"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn temp_png(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "deepspace-{label}-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos()
    ));
    path
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
fn parse_frame_raycast_t(stderr: &str) -> Option<f32> {
    let marker = "frame_raycast_hit";
    for line in stderr.lines() {
        if !line.contains(marker) {
            continue;
        }
        if let Some(start) = line.find(" t=") {
            let rest = &line[start + 3..];
            let end = rest.find(' ').unwrap_or(rest.len());
            if let Ok(value) = rest[..end].parse::<f32>() {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn load_png_rgb(path: &Path) -> RgbImage {
    use png::ColorType;
    use png::Decoder;

    let file = std::fs::File::open(path).expect("open png");
    let decoder = Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().expect("read png info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode png");
    let bytes = &buf[..info.buffer_size()];
    let mut pixels = Vec::with_capacity((info.width * info.height) as usize);

    match info.color_type {
        ColorType::Rgb => {
            for chunk in bytes.chunks_exact(3) {
                pixels.push([chunk[0], chunk[1], chunk[2]]);
            }
        }
        ColorType::Rgba => {
            for chunk in bytes.chunks_exact(4) {
                pixels.push([chunk[0], chunk[1], chunk[2]]);
            }
        }
        ColorType::Grayscale => {
            for &v in bytes {
                pixels.push([v, v, v]);
            }
        }
        ColorType::GrayscaleAlpha => {
            for chunk in bytes.chunks_exact(2) {
                pixels.push([chunk[0], chunk[0], chunk[0]]);
            }
        }
        ColorType::Indexed => panic!("indexed PNG output not supported for harness screenshots"),
    }

    RgbImage {
        width: info.width as usize,
        height: info.height as usize,
        pixels,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn image_diff(before: &RgbImage, after: &RgbImage) -> ImageDiff {
    assert_eq!(before.width, after.width);
    assert_eq!(before.height, after.height);
    let mut changed = 0usize;
    let mut min_x = before.width;
    let mut min_y = before.height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;

    for y in 0..before.height {
        for x in 0..before.width {
            if before.pixel(x, y) != after.pixel(x, y) {
                changed += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x + 1);
                max_y = max_y.max(y + 1);
            }
        }
    }

    ImageDiff {
        changed_frac: changed as f64 / (before.width * before.height) as f64,
        bbox: (changed > 0).then_some((min_x, min_y, max_x, max_y)),
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RgbImage {
    fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        self.pixels[y * self.width + x]
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn is_sandboxed_gui_startup_blocked(stderr: &str) -> bool {
    let has_no_frames = !stderr.contains("startup_perf frame=");
    let has_no_callbacks = !stderr.contains("startup_perf callback:");
    let has_launchservices_failure = stderr.contains("scheduleApplicationNotification")
        || stderr.contains("Connection Invalid error for service com.apple.hiservices-xpcservice")
        || stderr.contains("Error received in message reply handler: Connection invalid");
    let timed_out_before_perf = stderr
        .contains("wall-clock timeout hit before perf test completed")
        || stderr.contains("wall-clock timeout hit before min-fps test completed");
    let webview_never_created = stderr.contains("timed run ended without webview creation");
    has_no_frames
        && has_no_callbacks
        && has_launchservices_failure
        && (timed_out_before_perf || webview_never_created)
}
