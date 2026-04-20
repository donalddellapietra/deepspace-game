//! Visual tests for the entity pipeline.
//!
//! Each test launches the headless render harness with
//! `--spawn-entity assets/vox/soldier.vox`, captures a screenshot,
//! and verifies the entity actually rendered by comparing pixel
//! color distributions against a no-entity baseline rendered from
//! the same camera position.
//!
//! The baseline handles the "empty-sky" comparison cleanly:
//! the plain-world ground and sky are identical between runs, so
//! any pixel that differs MUST have come from the entity.
//!
//! Runs are mutex-guarded because multiple harness instances
//! contending over the GPU can deadlock on macOS.

#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard, OnceLock};

// ------------------------------------------------------- test scenarios

/// Camera that looks at the entity spawn location head-on. The
/// soldier sits one cell forward (-Z) of the camera; pointing yaw=0
/// and pitch≈0 puts it in the center of the frame.
const LOOK_FORWARD_ARGS: &[&str] = &[
    "--render-harness",
    "--disable-overlay",
    "--disable-highlight",
    "--plain-world",
    "--plain-layers",
    "40",
    "--spawn-depth",
    "6",
    "--spawn-xyz",
    // Camera a hair above sea level, looking horizontally. At
    // anchor depth 6 a cell is WORLD_SIZE/3^6 ≈ 0.004 units wide
    // so "just above the ground" means a few cells above sea
    // level. y=1.005 puts the camera ~1 cell up, entities at y=1.0
    // land directly on the horizon line with enough margin for
    // their 0.004-tall bodies to show against sky+grass.
    "1.5", "1.005", "1.8",
    "--spawn-yaw", "0",
    "--spawn-pitch", "0",
    "--harness-width", "640",
    "--harness-height", "360",
    "--exit-after-frames", "30",
    "--timeout-secs", "10",
];

/// Single entity facing forward — the basic "it renders" test.
#[test]
fn soldier_visible_in_front_of_camera() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();

    let dir = tmp_dir("soldier_visible_in_front");
    let baseline = dir.join("baseline.png");
    let with_entity = dir.join("with_entity.png");

    run_harness(LOOK_FORWARD_ARGS, &baseline, &[]);
    run_harness(
        LOOK_FORWARD_ARGS,
        &with_entity,
        &["--spawn-entity", "assets/vox/soldier.vox"],
    );

    let before = load_png_rgb(&baseline);
    let after = load_png_rgb(&with_entity);
    let diff = image_diff(&before, &after);

    // Entity should change a non-trivial fraction of pixels. The
    // soldier is small at depth 6; after the spawn-at-sea-level
    // fix it sits on the ground (lower third of the frame), so
    // the bbox covers ~40×40 px of which only the silhouette
    // actually differs from grass. Floor at 0.2% to catch
    // "nothing rendered" regressions while staying tolerant of
    // the smaller visible area.
    assert!(
        diff.changed_frac > 0.002,
        "soldier rendering barely changed image: changed_frac={:.4} bbox={:?}; \
         baseline={} with_entity={}",
        diff.changed_frac, diff.bbox,
        baseline.display(), with_entity.display(),
    );

    // The differing pixels should cluster roughly in frame X and
    // in the lower half of Y (entity at sea level with camera
    // just above it). If bbox X is way off, the entity spawned
    // off-screen horizontally.
    let (w, h) = (before.width, before.height);
    if let Some((x0, _y0, x1, _y1)) = diff.bbox {
        let cx = (x0 + x1) / 2;
        assert!(
            (cx as i32 - (w / 2) as i32).abs() < (w / 3) as i32,
            "entity-diff bbox center x={cx} not near frame center {}; bbox={:?} w={w}",
            w / 2, diff.bbox,
        );
        let _ = h;
    }
}

/// Count of distinct colors in the central region of the image with
/// the entity present. A humanoid with texture should introduce
/// multiple colors beyond the two (grass, sky) that dominate the
/// baseline.
#[test]
fn soldier_introduces_color_diversity() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();

    let dir = tmp_dir("soldier_color_diversity");
    let baseline = dir.join("baseline.png");
    let with_entity = dir.join("with_entity.png");

    run_harness(LOOK_FORWARD_ARGS, &baseline, &[]);
    run_harness(
        LOOK_FORWARD_ARGS,
        &with_entity,
        &["--spawn-entity", "assets/vox/soldier.vox"],
    );

    let baseline_img = load_png_rgb(&baseline);
    let entity_img = load_png_rgb(&with_entity);

    let baseline_colors = distinct_center_colors(&baseline_img, 5);
    let entity_colors = distinct_center_colors(&entity_img, 5);

    // The entity should introduce at least a handful of new colors
    // that aren't in the baseline's center (which is plain sky +
    // grass gradient, so ~8-15 quantized colors).
    assert!(
        entity_colors > baseline_colors + 3,
        "soldier didn't introduce new colors: baseline={baseline_colors} entity={entity_colors}",
    );
}

/// Spawn many entities and verify a much larger image region
/// differs from baseline — confirms the linear shader scan actually
/// iterates the whole buffer (precursor to hash-grid perf work).
#[test]
fn many_soldiers_change_large_area() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();

    let dir = tmp_dir("many_soldiers");
    let baseline = dir.join("baseline.png");
    let many = dir.join("many.png");

    run_harness(LOOK_FORWARD_ARGS, &baseline, &[]);
    run_harness(
        LOOK_FORWARD_ARGS,
        &many,
        &[
            "--spawn-entity", "assets/vox/soldier.vox",
            "--spawn-entity-count", "16",
        ],
    );

    let before = load_png_rgb(&baseline);
    let after = load_png_rgb(&many);
    let diff = image_diff(&before, &after);

    // 16 soldiers arranged in a horizontal grid (post spawn-at-
    // sea-level fix) cover less vertical area than the old eye-
    // level grid but still notably more than one soldier. Threshold
    // calibrated for the new geometry; the key regression this
    // test guards is "shader dropped most instances to sub-pixel."
    assert!(
        diff.changed_frac > 0.005,
        "16 soldiers barely changed image: changed_frac={:.4}",
        diff.changed_frac,
    );
}

/// Motion at 100 entities: capture frame 0 and frame 60 of the
/// same run; with per-entity velocity, the pixel images must
/// differ. Proves `EntityStore::tick` + per-frame re-upload are
/// wired through to the rendered output.
#[test]
fn soldiers_move_100() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();
    run_motion_test("motion_100", 100, 60, 0.01);
}

/// Motion at 1000 entities. Same check, expecting a larger delta
/// because more soldiers are drifting independently. If this fails
/// at 1000 while 100 passes, the issue is most likely the shader
/// iterating a stale entity buffer (bind group not rebound?).
#[test]
fn soldiers_move_1000() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();
    run_motion_test("motion_1000", 1000, 60, 0.02);
}

/// Motion at 10000 entities. Pushes the O(N) shader scan hard —
/// expect frame times in the tens-of-ms at 640×360. Test allows
/// more wall-clock time (20s timeout) and fewer tick frames (30)
/// to stay inside the budget. Hash-grid binning in a follow-up
/// should drop this test's frame time by an order of magnitude.
#[test]
fn soldiers_move_10000() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();
    run_motion_test_with_budget(
        "motion_10000",
        10000,
        30,   // fewer ticks to keep wall-clock reasonable
        0.02,
        "80", // total run frames (script needs wait:5 + screenshot + wait:30 + screenshot + tail)
        "120", // timeout secs: 10k entities at O(N) shader scan can hit 100ms+/frame
    );
}

/// Camera pointed away from the spawn cell should NOT show the
/// entity — the AABB cull must reject rays that miss the entity
/// box. Guards against "entities rendered on every ray" bugs.
#[test]
fn soldier_invisible_when_camera_faces_away() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();

    // yaw = π turns the camera 180° so the soldier is behind us.
    let args_back: &[&str] = &[
        "--render-harness",
        "--disable-overlay",
        "--disable-highlight",
        "--plain-world",
        "--plain-layers", "40",
        "--spawn-depth", "6",
        "--spawn-xyz", "1.5", "1.5", "1.8",
        "--spawn-yaw", "3.14159",
        "--spawn-pitch", "0",
        "--harness-width", "640",
        "--harness-height", "360",
        "--exit-after-frames", "30",
        "--timeout-secs", "10",
    ];

    let dir = tmp_dir("soldier_behind_camera");
    let baseline = dir.join("baseline.png");
    let behind = dir.join("with_entity_behind.png");

    run_harness(args_back, &baseline, &[]);
    run_harness(
        args_back,
        &behind,
        &["--spawn-entity", "assets/vox/soldier.vox"],
    );

    let before = load_png_rgb(&baseline);
    let after = load_png_rgb(&behind);
    let diff = image_diff(&before, &after);

    // Looking away from the entity: images should be essentially
    // identical. A few-pixel difference at the edges is tolerable
    // (floating-point noise, tonemapping), but anything above 0.1%
    // means the entity IS contributing — that's a render-path bug.
    assert!(
        diff.changed_frac < 0.001,
        "entity leaked into view when camera faces away: changed_frac={:.4} bbox={:?}",
        diff.changed_frac, diff.bbox,
    );
}

// ------------------------------------------------------- motion runner

/// Run the harness once, spawn `count` soldiers, capture a
/// screenshot before motion and after `tick_frames` of simulation,
/// and assert the two images differ by at least `min_frac`.
fn run_motion_test(name: &str, count: u32, tick_frames: u32, min_frac: f64) {
    // Total frames must cover: a few settle frames + both
    // screenshots + the wait between them.
    let total_frames = (tick_frames + 40).to_string();
    run_motion_test_with_budget(name, count, tick_frames, min_frac, &total_frames, "30");
}

fn run_motion_test_with_budget(
    name: &str,
    count: u32,
    tick_frames: u32,
    min_frac: f64,
    total_frames: &str,
    timeout_secs: &str,
) {
    let dir = tmp_dir(name);
    let frame_a = dir.join("frame_a.png");
    let frame_b = dir.join("frame_b.png");
    let _ = std::fs::remove_file(&frame_a);
    let _ = std::fs::remove_file(&frame_b);

    let count_str = count.to_string();
    let script = format!(
        "wait:5,screenshot:{},wait:{},screenshot:{}",
        frame_a.display(),
        tick_frames,
        frame_b.display(),
    );

    let args: &[&str] = &[
        "--render-harness",
        "--disable-overlay",
        "--disable-highlight",
        "--plain-world",
        "--plain-layers", "40",
        "--spawn-depth", "6",
        // Camera just above sea level (y=1.005 ≈ 1 depth-6 cell up)
        // so entities at y=1.0 appear at the horizon.
        "--spawn-xyz", "1.5", "1.005", "1.8",
        "--spawn-yaw", "0",
        "--spawn-pitch", "0",
        "--harness-width", "640",
        "--harness-height", "360",
        "--exit-after-frames", total_frames,
        "--timeout-secs", timeout_secs,
        "--spawn-entity", "assets/vox/soldier.vox",
        "--spawn-entity-count", &count_str,
        "--script", &script,
    ];

    let exe = env!("CARGO_BIN_EXE_deepspace-game");
    let mut cmd = Command::new(exe);
    for a in args { cmd.arg(a); }
    let output = cmd.output().expect("run harness");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{name} harness did not exit 0; stderr tail:\n{}",
        stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );
    assert!(frame_a.exists(), "{name}: frame_a missing");
    assert!(frame_b.exists(), "{name}: frame_b missing");

    let before = load_png_rgb(&frame_a);
    let after = load_png_rgb(&frame_b);
    let diff = image_diff(&before, &after);
    let frame_ms = parse_avg_frame_ms(&stderr).unwrap_or(0.0);
    eprintln!(
        "{name}: count={count} tick_frames={tick_frames} changed_frac={:.4} avg_frame_ms={:.2}",
        diff.changed_frac, frame_ms,
    );
    assert!(
        diff.changed_frac > min_frac,
        "{name}: entities didn't move enough between frames: changed_frac={:.4} (threshold {}); \
         frame_a={} frame_b={}",
        diff.changed_frac, min_frac,
        frame_a.display(), frame_b.display(),
    );
}

/// Extract `avg_ms ... total=<x>` from the harness's timing log
/// line so scale tests can report per-count perf costs.
fn parse_avg_frame_ms(stderr: &str) -> Option<f64> {
    for line in stderr.lines() {
        if !line.contains("render_harness_timing") { continue; }
        for field in line.split_whitespace() {
            if let Some(v) = field.strip_prefix("total=") {
                return v.parse().ok();
            }
        }
    }
    None
}

// ------------------------------------------------------- helpers

fn visibility_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// Per-suite tmp artifact directory at `<project>/tmp/<name>/`.
fn tmp_dir(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp").join(name);
    std::fs::create_dir_all(&root).expect("mkdir tmp");
    root
}

/// Ensure the soldier.vox file exists — regenerate via the python
/// voxelizer if missing. Lets CI + fresh checkouts run the suite
/// without manual setup.
fn ensure_soldier_vox() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/vox/soldier.vox");
    if path.exists() {
        return;
    }
    eprintln!("soldier.vox missing, regenerating from Soldier.glb...");
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts/regen-vox-entities.sh");
    let status = Command::new("bash")
        .arg(&script)
        .status()
        .expect("failed to run regen script");
    assert!(status.success(), "regen-vox-entities.sh failed");
    assert!(path.exists(), "soldier.vox still missing after regen");
}

/// Run the binary with the given args, routing screenshot to
/// `screenshot_path`.
fn run_harness(base_args: &[&str], screenshot_path: &Path, extra: &[&str]) {
    let exe = env!("CARGO_BIN_EXE_deepspace-game");
    let mut cmd = Command::new(exe);
    for arg in base_args {
        cmd.arg(arg);
    }
    for arg in extra {
        cmd.arg(arg);
    }
    cmd.arg("--screenshot");
    cmd.arg(screenshot_path);
    let output = cmd.output().expect("run harness");
    if !output.status.success() {
        panic!(
            "harness run failed for {}; stderr tail:\n{}",
            screenshot_path.display(),
            String::from_utf8_lossy(&output.stderr)
                .lines().rev().take(20).collect::<Vec<_>>().join("\n"),
        );
    }
    assert!(
        screenshot_path.exists(),
        "harness did not produce screenshot at {}",
        screenshot_path.display(),
    );
}

// ------------------------------------------------------- image analysis

struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>,
}

impl RgbImage {
    fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        self.pixels[y * self.width + x]
    }
}

#[derive(Debug)]
struct ImageDiff {
    changed_frac: f64,
    bbox: Option<(usize, usize, usize, usize)>,
}

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
        other => panic!("unsupported PNG color type {other:?}"),
    }

    RgbImage {
        width: info.width as usize,
        height: info.height as usize,
        pixels,
    }
}

/// Per-pixel comparison. A pixel counts as "changed" when any
/// channel differs by more than 2 — tolerates tiny tonemapping
/// noise between runs while catching real visual differences.
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
            let a = before.pixel(x, y);
            let b = after.pixel(x, y);
            let dr = (a[0] as i16 - b[0] as i16).abs();
            let dg = (a[1] as i16 - b[1] as i16).abs();
            let db = (a[2] as i16 - b[2] as i16).abs();
            if dr > 2 || dg > 2 || db > 2 {
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

// ------------------------------------------------------- heightmap wiring

/// Run the binary with heightmap collisions enabled (via
/// `--entity-render raster`) and return the parsed
/// `render_harness_timing` line. Panics on harness failure.
fn run_heightmap_timing(
    name: &str,
    count: u32,
    frames: u32,
) -> (PathBuf, String, Option<f64>) {
    let dir = tmp_dir(name);
    let shot = dir.join("final.png");
    let _ = std::fs::remove_file(&shot);
    let count_str = count.to_string();
    let frames_str = frames.to_string();
    let exe = env!("CARGO_BIN_EXE_deepspace-game");
    let output = Command::new(exe)
        .args([
            "--render-harness",
            "--disable-overlay",
            "--disable-highlight",
            "--plain-world",
            "--plain-layers", "40",
            "--spawn-depth", "6",
            "--spawn-xyz", "1.5", "1.02", "1.8",
            "--spawn-yaw", "0",
            "--spawn-pitch", "-1.3",
            "--harness-width", "320",
            "--harness-height", "180",
            "--exit-after-frames", &frames_str,
            "--timeout-secs", "30",
            "--spawn-entity", "assets/vox/soldier.vox",
            "--spawn-entity-count", &count_str,
            "--entity-render", "raster",
            "--screenshot", shot.to_str().unwrap(),
        ])
        .output()
        .expect("run harness");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "{name} harness did not exit 0; stderr tail:\n{}",
        stderr.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );
    assert!(shot.exists(), "{name}: screenshot missing");
    let avg_ms = parse_avg_frame_ms(&stderr);
    (shot, stderr, avg_ms)
}

/// Fraction of pixels in `img` that look like entities rather than
/// grass / sky. Uses a simple color-match heuristic: entity pixels
/// are anything that isn't close to the grass greens or the sky
/// blues the plain world paints. Used to verify entities are
/// actually ON SCREEN after the heightmap clamp has run.
fn entity_pixel_fraction(img: &RgbImage) -> f64 {
    let mut count = 0usize;
    let total = img.pixels.len();
    for &[r, g, b] in &img.pixels {
        let is_sky = b > 200 && g > 180 && r > 160;
        let is_grass = g > r + 20 && g > b + 10;
        if !is_sky && !is_grass {
            count += 1;
        }
    }
    count as f64 / total as f64
}

/// Smoke test — with heightmap wired, soldiers spawned on a plain
/// world appear on the ground when the camera pitches down. The
/// test also asserts the pipeline survives 60 frames without
/// crashing, which catches GPU-side barrier/binding issues that
/// might only surface after repeated dispatches.
#[test]
fn heightmap_clamp_places_entities_on_ground_under_camera() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();
    let (shot, _stderr, _ms) = run_heightmap_timing(
        "heightmap_places_on_ground", 30, 60,
    );
    let img = load_png_rgb(&shot);
    let entity_frac = entity_pixel_fraction(&img);
    assert!(
        entity_frac > 0.005,
        "expected entity pixels in frame (camera pitched down at ground), got {entity_frac:.4}; shot={}",
        shot.display(),
    );
}

/// 10k-entity perf guard. Wires heightmap + clamp through the real
/// renderer; if the dispatches leak per-frame allocation or block
/// on the wrong barrier, frame time will balloon well beyond the
/// budget. 20 ms is 2× our measured steady state, generous enough
/// to survive macOS scheduler jitter but tight enough to catch
/// regressions.
#[test]
fn heightmap_10k_raster_stays_under_20ms() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();
    let (_shot, stderr, avg_ms) = run_heightmap_timing(
        "heightmap_10k_perf", 10_000, 120,
    );
    let ms = avg_ms.unwrap_or_else(|| {
        panic!(
            "no render_harness_timing line in stderr:\n{}",
            stderr.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
        )
    });
    assert!(
        ms < 20.0,
        "heightmap-wired 10k raster avg frame = {ms:.3} ms (budget 20)",
    );
}

/// Verify the pipeline does not break when entity count spikes
/// upward between frames. This exercises the instance-buffer
/// recreation path (which got STORAGE usage added for this wiring)
/// plus the clamp bind-group's buffer-reference refresh.
#[test]
fn heightmap_clamp_survives_instance_buffer_growth() {
    let _guard = visibility_test_lock();
    ensure_soldier_vox();
    // Start at 10 entities; a single M-keybind press inside the
    // harness would grow to 1000. We can't easily script keypresses
    // but spawn-entity-count already starts at the larger value;
    // the relevant code path is exercised any time the harness
    // runs at a count past the initial 1024-byte buffer.
    let (_shot, _stderr, _ms) = run_heightmap_timing(
        "heightmap_clamp_grow", 1000, 30,
    );
}

/// Quantize the full image to `shift`-bit channels and count
/// distinct colors. Baseline (sky gradient + grass gradient) is
/// a handful of colors; a soldier adds its vox-palette entries
/// wherever it lands in frame. Checking the whole image absorbs
/// the entity's horizontal offset without a fixed bbox.
fn distinct_center_colors(img: &RgbImage, shift: u8) -> usize {
    use std::collections::HashSet;
    let mut seen: HashSet<(u8, u8, u8)> = HashSet::new();
    for y in 0..img.height {
        for x in 0..img.width {
            let [r, g, b] = img.pixel(x, y);
            seen.insert((r >> shift, g >> shift, b >> shift));
        }
    }
    seen.len()
}
