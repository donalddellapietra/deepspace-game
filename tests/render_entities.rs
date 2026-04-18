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
    "1.5",
    "1.5",
    "1.8",
    "--spawn-yaw",
    "0",
    "--spawn-pitch",
    "0",
    "--harness-width",
    "640",
    "--harness-height",
    "360",
    "--exit-after-frames",
    "30",
    "--timeout-secs",
    "10",
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
    // soldier is small at depth 6 but still clearly visible in the
    // center of the frame — ~1-5% of pixels differ in our smoke
    // test. Floor at 0.3% to catch "nothing rendered" regressions
    // while staying robust to LOD / bevel noise.
    assert!(
        diff.changed_frac > 0.003,
        "soldier rendering barely changed image: changed_frac={:.4} bbox={:?}; \
         baseline={} with_entity={}",
        diff.changed_frac, diff.bbox,
        baseline.display(), with_entity.display(),
    );

    // The differing pixels should cluster around the frame center
    // (that's where we pointed the camera). If the bbox is far from
    // center, something is spawning off-screen.
    let (w, h) = (before.width, before.height);
    if let Some((x0, y0, x1, y1)) = diff.bbox {
        let cx = (x0 + x1) / 2;
        let cy = (y0 + y1) / 2;
        assert!(
            (cx as i32 - (w / 2) as i32).abs() < (w / 3) as i32,
            "entity-diff bbox center x={cx} not near frame center {}; bbox={:?} w={w}",
            w / 2, diff.bbox,
        );
        assert!(
            (cy as i32 - (h / 2) as i32).abs() < (h / 3) as i32,
            "entity-diff bbox center y={cy} not near frame center {}; bbox={:?} h={h}",
            h / 2, diff.bbox,
        );
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

    // 16 soldiers spread in a row should cover much more area than
    // one.
    assert!(
        diff.changed_frac > 0.02,
        "16 soldiers barely changed image: changed_frac={:.4}",
        diff.changed_frac,
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

/// Quantize center-region pixels to `shift`-bit channels and count
/// distinct colors. Higher count = more color diversity (the
/// soldier texture vs. a flat sky/grass gradient).
fn distinct_center_colors(img: &RgbImage, shift: u8) -> usize {
    use std::collections::HashSet;
    let cx0 = img.width / 3;
    let cx1 = 2 * img.width / 3;
    let cy0 = img.height / 3;
    let cy1 = 2 * img.height / 3;
    let mut seen: HashSet<(u8, u8, u8)> = HashSet::new();
    for y in cy0..cy1 {
        for x in cx0..cx1 {
            let [r, g, b] = img.pixel(x, y);
            seen.insert((r >> shift, g >> shift, b >> shift));
        }
    }
    seen.len()
}
