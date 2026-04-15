//! Built-in test driver: drives the game through a scripted scenario
//! (zoom, click, screenshot, exit) and produces deterministic
//! artifacts on disk. Used by the agent to iterate on rendering
//! issues without external input synthesis.
//!
//! All behavior is opt-in via CLI flags parsed by [`TestConfig::from_args`].
//! When no flags are given the game runs interactively as normal.
//!
//! Recognized flags:
//!
//! ```text
//! --render-harness        Run in deterministic render-harness mode:
//!                         no overlay/webview, forced redraws, auto-exit.
//! --show-window           In render-harness mode, keep the native window
//!                         visible instead of hidden.
//! --disable-overlay       Keep the native window/surface path, but skip
//!                         WKWebView creation and overlay flushing.
//! --spawn-depth N         Camera anchor starts at depth N (default 4).
//! --screenshot PATH       Capture the rendered frame to PATH (PNG)
//!                         after the warm-up + script settle, then exit.
//! --exit-after-frames N   Exit after N rendered frames (default ~120).
//! --timeout-secs SECS     Wall-clock kill switch (default 5.0). Triggers
//!                         a screenshot + exit if the frame budget hasn't
//!                         already done so. Catches perf regressions:
//!                         a hung shader can't run a test forever.
//! --min-fps FPS           Fail the run if the measured FPS after warm-up
//!                         drops below this threshold. Default: disabled.
//! --fps-warmup-frames N   Ignore the first N rendered frames before the
//!                         min-fps gate becomes active. Default: 10.
//! --min-cadence-fps FPS   Fail the run if average present cadence after
//!                         warm-up drops below this threshold. This tracks
//!                         `dt` / overlay FPS, not pure frame work time.
//! --cadence-warmup-frames N
//!                         Ignore the first N presented frames before the
//!                         cadence gate becomes active. Default: 10.
//! --run-for-secs SECS     End the run after this much wall-clock time,
//!                         instead of relying on a rendered-frame budget.
//! --max-frame-gap-ms MS   Fail if the gap between rendered frames exceeds
//!                         this wall-clock budget. Detects real freezes.
//! --frame-gap-warmup-frames N
//!                         Ignore max-frame-gap checks until at least N
//!                         frames have rendered. Default: 30.
//! --require-webview       Fail if the native WKWebView overlay never
//!                         comes up during the test scenario.
//! --script CMDS           Comma-separated commands run in order:
//!                           break          left-click (break a block)
//!                           place          right-click (place a block)
//!                           wait:N         skip N frames
//!                           zoom_in:N      apply N zoom-in steps
//!                           zoom_out:N     apply N zoom-out steps
//!                           debug_overlay  toggle the debug overlay panel
//!                         (`screenshot` and exit are handled by the
//!                          flags above — no need to inline them.)
//! ```
//!
//! Example:
//!
//! ```bash
//! cargo run -- --spawn-depth 12 --screenshot /tmp/layer10.png \
//!     --script "wait:30,break,wait:30" --exit-after-frames 90
//! ```

#[derive(Default, Debug, Clone)]
pub struct TestConfig {
    pub render_harness: bool,
    pub show_window: bool,
    pub disable_overlay: bool,
    pub spawn_depth: Option<u8>,
    /// Explicit camera world-XYZ at spawn. Positions the camera
    /// at a specific point regardless of zoom level — since the
    /// in-game zoom function is broken, this is the way to put
    /// the camera near a feature (e.g., the planet surface) for
    /// screenshot-driven debugging.
    pub spawn_xyz: Option<[f32; 3]>,
    /// Camera yaw at spawn (radians). Default 0.
    pub spawn_yaw: Option<f32>,
    /// Camera pitch at spawn (radians). Default -1.2 (steep down).
    pub spawn_pitch: Option<f32>,
    pub screenshot: Option<String>,
    pub exit_after_frames: Option<u32>,
    /// Wall-clock kill switch in seconds. Defaults to 5.0 so a
    /// perf regression (hung shader, runaway DDA) can't block the
    /// test loop indefinitely. Override with `--timeout-secs N`
    /// for scenarios that genuinely need longer settle time.
    pub timeout_secs: Option<f32>,
    pub min_fps: Option<f32>,
    pub fps_warmup_frames: Option<u32>,
    pub min_cadence_fps: Option<f32>,
    pub cadence_warmup_frames: Option<u32>,
    pub run_for_secs: Option<f32>,
    pub max_frame_gap_ms: Option<f32>,
    pub frame_gap_warmup_frames: Option<u32>,
    pub require_webview: bool,
    pub script: Vec<ScriptCmd>,
}

#[derive(Debug, Clone)]
pub enum ScriptCmd {
    Break,
    Place,
    Wait(u32),
    ZoomIn(u32),
    ZoomOut(u32),
    ToggleDebugOverlay,
}

impl TestConfig {
    pub fn from_args() -> Self {
        let mut cfg = TestConfig::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--render-harness" => {
                    cfg.render_harness = true;
                }
                "--show-window" => {
                    cfg.show_window = true;
                }
                "--disable-overlay" => {
                    cfg.disable_overlay = true;
                }
                "--spawn-depth" => {
                    cfg.spawn_depth = args.next().and_then(|v| v.parse().ok());
                }
                "--spawn-xyz" => {
                    let x: Option<f32> = args.next().and_then(|v| v.parse().ok());
                    let y: Option<f32> = args.next().and_then(|v| v.parse().ok());
                    let z: Option<f32> = args.next().and_then(|v| v.parse().ok());
                    if let (Some(x), Some(y), Some(z)) = (x, y, z) {
                        cfg.spawn_xyz = Some([x, y, z]);
                    }
                }
                "--spawn-yaw" => {
                    cfg.spawn_yaw = args.next().and_then(|v| v.parse().ok());
                }
                "--spawn-pitch" => {
                    cfg.spawn_pitch = args.next().and_then(|v| v.parse().ok());
                }
                "--screenshot" => {
                    cfg.screenshot = args.next();
                }
                "--exit-after-frames" => {
                    cfg.exit_after_frames = args.next().and_then(|v| v.parse().ok());
                }
                "--timeout-secs" => {
                    cfg.timeout_secs = args.next().and_then(|v| v.parse().ok());
                }
                "--script" => {
                    if let Some(s) = args.next() {
                        cfg.script = parse_script(&s);
                    }
                }
                "--min-fps" => {
                    cfg.min_fps = args.next().and_then(|v| v.parse().ok());
                }
                "--fps-warmup-frames" => {
                    cfg.fps_warmup_frames = args.next().and_then(|v| v.parse().ok());
                }
                "--min-cadence-fps" => {
                    cfg.min_cadence_fps = args.next().and_then(|v| v.parse().ok());
                }
                "--cadence-warmup-frames" => {
                    cfg.cadence_warmup_frames = args.next().and_then(|v| v.parse().ok());
                }
                "--run-for-secs" => {
                    cfg.run_for_secs = args.next().and_then(|v| v.parse().ok());
                }
                "--max-frame-gap-ms" => {
                    cfg.max_frame_gap_ms = args.next().and_then(|v| v.parse().ok());
                }
                "--frame-gap-warmup-frames" => {
                    cfg.frame_gap_warmup_frames = args.next().and_then(|v| v.parse().ok());
                }
                "--require-webview" => {
                    cfg.require_webview = true;
                }
                _ => {}
            }
        }
        cfg
    }

    /// True if any flag asks the test runner to take action.
    pub fn is_active(&self) -> bool {
        self.render_harness
            || self.screenshot.is_some()
            || self.exit_after_frames.is_some()
            || !self.script.is_empty()
            || self.spawn_xyz.is_some()
            || self.spawn_yaw.is_some()
            || self.spawn_pitch.is_some()
            || self.min_fps.is_some()
            || self.min_cadence_fps.is_some()
            || self.run_for_secs.is_some()
            || self.max_frame_gap_ms.is_some()
            || self.frame_gap_warmup_frames.is_some()
            || self.require_webview
    }

    pub fn prefers_live_loop(&self) -> bool {
        self.screenshot.is_none()
            && (
                self.min_fps.is_some()
                    || self.min_cadence_fps.is_some()
                    || self.run_for_secs.is_some()
                    || self.max_frame_gap_ms.is_some()
                    || self.require_webview
            )
    }

    pub fn use_render_harness(&self) -> bool {
        (self.render_harness && !self.prefers_live_loop()) || self.screenshot.is_some()
    }
}

fn parse_script(s: &str) -> Vec<ScriptCmd> {
    s.split(',')
        .filter_map(|raw| {
            let raw = raw.trim();
            if raw.is_empty() { return None; }
            if raw == "break" { return Some(ScriptCmd::Break); }
            if raw == "place" { return Some(ScriptCmd::Place); }
            if raw == "debug_overlay" { return Some(ScriptCmd::ToggleDebugOverlay); }
            if let Some(n) = raw.strip_prefix("wait:") {
                if let Ok(frames) = n.parse() { return Some(ScriptCmd::Wait(frames)); }
            }
            if let Some(n) = raw.strip_prefix("zoom_in:") {
                if let Ok(steps) = n.parse() { return Some(ScriptCmd::ZoomIn(steps)); }
            }
            if let Some(n) = raw.strip_prefix("zoom_out:") {
                if let Ok(steps) = n.parse() { return Some(ScriptCmd::ZoomOut(steps)); }
            }
            eprintln!("test_runner: ignoring unknown script command {raw:?}");
            None
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PerfSamples {
    pub count: u32,
    pub total_frame_secs: f64,
    pub total_cadence_secs: f64,
    pub worst_frame_secs: f64,
    pub worst_cadence_secs: f64,
}

impl PerfSamples {
    pub fn record_frame(&mut self, frame_secs: f64) {
        self.count += 1;
        self.total_frame_secs += frame_secs;
        self.worst_frame_secs = self.worst_frame_secs.max(frame_secs);
    }

    pub fn record_cadence(&mut self, cadence_secs: f64) {
        self.total_cadence_secs += cadence_secs;
        self.worst_cadence_secs = self.worst_cadence_secs.max(cadence_secs);
    }

    pub fn avg_frame_fps(&self) -> Option<f32> {
        if self.count == 0 || self.total_frame_secs <= 0.0 {
            None
        } else {
            Some((self.count as f64 / self.total_frame_secs) as f32)
        }
    }

    pub fn avg_cadence_fps(&self) -> Option<f32> {
        if self.count == 0 || self.total_cadence_secs <= 0.0 {
            None
        } else {
            Some((self.count as f64 / self.total_cadence_secs) as f32)
        }
    }
}

pub struct TestMonitor {
    pub frames_rendered: std::sync::atomic::AtomicU32,
    pub last_frame_ms: std::sync::atomic::AtomicU64,
    pub perf_failed: std::sync::atomic::AtomicBool,
    pub webview_created: std::sync::atomic::AtomicBool,
    pub perf_samples: std::sync::Mutex<PerfSamples>,
}

impl TestMonitor {
    pub fn new() -> Self {
        Self {
            frames_rendered: std::sync::atomic::AtomicU32::new(0),
            last_frame_ms: std::sync::atomic::AtomicU64::new(0),
            perf_failed: std::sync::atomic::AtomicBool::new(false),
            webview_created: std::sync::atomic::AtomicBool::new(false),
            perf_samples: std::sync::Mutex::new(PerfSamples::default()),
        }
    }

    pub fn record_frame(
        &self,
        elapsed_since_start: std::time::Duration,
        frame_secs: Option<f64>,
        cadence_secs: Option<f64>,
    ) {
        use std::sync::atomic::Ordering;

        self.frames_rendered.fetch_add(1, Ordering::Relaxed);
        self.last_frame_ms.store(
            elapsed_since_start.as_millis().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        if let Ok(mut perf) = self.perf_samples.lock() {
            if let Some(frame_secs) = frame_secs {
                perf.record_frame(frame_secs);
            }
            if let Some(cadence_secs) = cadence_secs {
                perf.record_cadence(cadence_secs);
            }
        }
    }
}

/// Per-frame state for a running test scenario.
pub struct TestRunner {
    pub frame: u32,
    pub screenshot_path: Option<String>,
    pub screenshot_done: bool,
    pub exit_after_frames: u32,
    /// Pre-flattened action queue: each entry is `(at_frame, cmd)`.
    pub script: Vec<(u32, ScriptCmd)>,
    /// Wall-clock start; combined with `timeout_secs` to bail on
    /// perf hangs without depending on frame counting (a stuck
    /// renderer might never advance frames).
    pub started_at: std::time::Instant,
    pub timeout_secs: f32,
    pub min_fps: Option<f32>,
    pub fps_warmup_frames: u32,
    pub min_cadence_fps: Option<f32>,
    pub cadence_warmup_frames: u32,
    pub run_for_secs: Option<f32>,
    pub max_frame_gap_ms: Option<f32>,
    pub frame_gap_warmup_frames: u32,
    pub require_webview: bool,
    pub perf_samples: PerfSamples,
    pub monitor: std::sync::Arc<TestMonitor>,
}

impl TestRunner {
    pub fn from_config(cfg: TestConfig) -> Option<Self> {
        if !cfg.is_active() { return None; }
        // If the caller asked for a wall-clock run, do not silently
        // pre-empt it with the old default 120-frame budget.
        let exit_after = match (cfg.exit_after_frames, cfg.run_for_secs) {
            (Some(exit_after), _) => exit_after,
            (None, Some(_)) => u32::MAX,
            (None, None) => 120,
        };
        let timeout_secs = cfg.timeout_secs.unwrap_or(5.0);
        let min_fps = cfg.min_fps;
        let min_cadence_fps = cfg.min_cadence_fps;
        let run_for_secs = cfg.run_for_secs;
        let max_frame_gap_ms = cfg.max_frame_gap_ms;
        let frame_gap_warmup_frames = cfg.frame_gap_warmup_frames.unwrap_or(30);
        let require_webview = cfg.require_webview;
        let started_at = std::time::Instant::now();
        let monitor = std::sync::Arc::new(TestMonitor::new());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let monitor = std::sync::Arc::clone(&monitor);
            std::thread::spawn(move || {
                use std::sync::atomic::Ordering;

                loop {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    let elapsed = started_at.elapsed();
                    let elapsed_secs = elapsed.as_secs_f32();
                    let elapsed_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
                    let frame_count = monitor.frames_rendered.load(Ordering::Relaxed);
                    let last_frame_ms = monitor.last_frame_ms.load(Ordering::Relaxed);

                    if let Some(max_gap_ms) = max_frame_gap_ms {
                        if frame_count > frame_gap_warmup_frames {
                            let gap_ms = elapsed_ms.saturating_sub(last_frame_ms);
                            if gap_ms as f32 > max_gap_ms {
                                eprintln!(
                                    "test_runner: frame-gap freeze detected: gap_ms={} threshold_ms={:.2} frames={} warmup_frames={}",
                                    gap_ms, max_gap_ms, frame_count, frame_gap_warmup_frames,
                                );
                                print_monitor_summary(&monitor);
                                std::process::exit(1);
                            }
                        }
                    }

                    if let Some(run_for_secs) = run_for_secs {
                        if elapsed_secs >= run_for_secs {
                            print_monitor_summary(&monitor);
                            if monitor.perf_failed.load(Ordering::Relaxed) {
                                eprintln!("test_runner: perf threshold failed during timed run, quitting");
                                std::process::exit(1);
                            }
                            if require_webview && !monitor.webview_created.load(Ordering::Relaxed) {
                                eprintln!("test_runner: timed run ended without webview creation, quitting");
                                std::process::exit(1);
                            }
                            eprintln!(
                                "test_runner: run_for_secs={:.2} reached with {} rendered frames, quitting",
                                run_for_secs,
                                frame_count,
                            );
                            std::process::exit(0);
                        }
                    }

                    if elapsed_secs >= timeout_secs {
                        print_monitor_summary(&monitor);
                        if require_webview && !monitor.webview_created.load(Ordering::Relaxed) {
                            eprintln!("test_runner: wall-clock timeout hit before webview creation, quitting");
                            std::process::exit(1);
                        }
                        if min_fps.is_some() || min_cadence_fps.is_some() || run_for_secs.is_some() || max_frame_gap_ms.is_some() {
                            eprintln!("test_runner: wall-clock timeout hit before perf test completed, quitting");
                            std::process::exit(1);
                        }
                        eprintln!("test_runner: wall-clock timeout hit, quitting");
                        std::process::exit(0);
                    }
                }
            });
        }
        // Build the script schedule. Default: start running at frame
        // 30 so the first batch of GPU uploads has settled.
        let mut t = 30u32;
        let mut schedule = Vec::new();
        for cmd in cfg.script {
            match cmd {
                ScriptCmd::Wait(frames) => { t += frames; }
                other => { schedule.push((t, other)); }
            }
        }
        Some(TestRunner {
            frame: 0,
            screenshot_path: cfg.screenshot,
            screenshot_done: false,
            exit_after_frames: exit_after,
            script: schedule,
            started_at,
            timeout_secs,
            min_fps,
            fps_warmup_frames: cfg.fps_warmup_frames.unwrap_or(10),
            min_cadence_fps,
            cadence_warmup_frames: cfg.cadence_warmup_frames.unwrap_or(10),
            run_for_secs,
            max_frame_gap_ms,
            frame_gap_warmup_frames,
            require_webview,
            perf_samples: PerfSamples::default(),
            monitor,
        })
    }

    /// True once the wall-clock timeout has fired.
    pub fn timed_out(&self) -> bool {
        self.started_at.elapsed().as_secs_f32() >= self.timeout_secs
    }

    /// Pop any commands whose scheduled frame is `<= self.frame`.
    pub fn drain_due(&mut self) -> Vec<ScriptCmd> {
        let mut due = Vec::new();
        let frame = self.frame;
        self.script.retain(|(at, cmd)| {
            if *at <= frame { due.push(cmd.clone()); false } else { true }
        });
        due
    }
}

#[allow(deprecated)]
pub fn run_render_harness(cfg: TestConfig) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use crate::app::App;
    use crate::renderer::Renderer;
    use winit::event_loop::EventLoop;
    use winit::window::WindowAttributes;

    let mut app = App::with_test_config(cfg.clone());
    let event_loop = EventLoop::new()?;
    let window = Arc::new(event_loop.create_window(
        WindowAttributes::default()
            .with_title("Deep Space Render Harness")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_visible(cfg.show_window),
    )?);
    let (tree_data, node_kinds, root_index) =
        crate::world::gpu::pack_tree(&app.world.library, app.world.root);
    let renderer = pollster::block_on(Renderer::new(
        window.clone(), &tree_data, &node_kinds, root_index,
    ));
    app.window = Some(window);
    app.renderer = Some(renderer);
    app.apply_zoom();
    app.last_frame = std::time::Instant::now();

    let mut total_update = 0.0f64;
    let mut total_upload = 0.0f64;
    let mut total_highlight = 0.0f64;
    let mut total_render = 0.0f64;
    let mut frame_count = 0u32;

    loop {
        let t0 = std::time::Instant::now();
        app.update(1.0 / 60.0);
        let t_update = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        app.upload_tree_lod();
        let t_upload = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = std::time::Instant::now();
        app.update_highlight();
        let t_highlight = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = std::time::Instant::now();
        if let Some(renderer) = &app.renderer {
            renderer.render_offscreen();
        }
        let t_render = t3.elapsed().as_secs_f64() * 1000.0;

        total_update += t_update;
        total_upload += t_upload;
        total_highlight += t_highlight;
        total_render += t_render;
        frame_count += 1;

        let (due, frame, frame_budget_done, timed_out, exit_after, screenshot) = {
            let Some(test) = app.test.as_mut() else { break };
            test.frame += 1;
            let frame = test.frame;
            let due = test.drain_due();
            let frame_budget_done = frame + 1 >= test.exit_after_frames;
            let timed_out = test.timed_out();
            let exit_after = test.exit_after_frames;
            let screenshot = test.screenshot_path.clone();
            (due, frame, frame_budget_done, timed_out, exit_after, screenshot)
        };

        for cmd in due {
            match cmd {
                ScriptCmd::Break => app.do_break(),
                ScriptCmd::Place => app.do_place(),
                ScriptCmd::Wait(_) => {}
                ScriptCmd::ZoomIn(steps) => {
                    for _ in 0..steps {
                        app.zoom_anchor(1);
                    }
                }
                ScriptCmd::ZoomOut(steps) => {
                    for _ in 0..steps {
                        app.zoom_anchor(-1);
                    }
                }
                ScriptCmd::ToggleDebugOverlay => {
                    app.debug_overlay_visible = !app.debug_overlay_visible;
                }
            }
        }

        if let Some(path) = screenshot {
            let already_done = app.test.as_ref().is_some_and(|t| t.screenshot_done);
            if !already_done && (frame_budget_done || timed_out) {
                if let Some(renderer) = &mut app.renderer {
                    renderer.capture_to_png(&path)?;
                }
                if let Some(t) = app.test.as_mut() {
                    t.screenshot_done = true;
                }
            }
        }

        if timed_out {
            eprintln!("render_harness: timeout reached at frame {frame}, quitting");
            break;
        }
        if frame >= exit_after {
            eprintln!("render_harness: exit_after_frames={frame} reached, quitting");
            break;
        }
    }

    if frame_count > 0 {
        eprintln!(
            "render_harness_timing avg_ms update={:.3} upload={:.3} highlight={:.3} render={:.3} total={:.3}",
            total_update / frame_count as f64,
            total_upload / frame_count as f64,
            total_highlight / frame_count as f64,
            total_render / frame_count as f64,
            (total_update + total_upload + total_highlight + total_render) / frame_count as f64,
        );
    }

    Ok(())
}

fn print_monitor_summary(monitor: &std::sync::Arc<TestMonitor>) {
    if let Ok(perf) = monitor.perf_samples.lock() {
        eprintln!(
            "test_runner: perf summary samples={} avg_frame_fps={:.2} avg_cadence_fps={:.2} worst_frame_ms={:.2} worst_dt_ms={:.2}",
            perf.count,
            perf.avg_frame_fps().unwrap_or(0.0),
            perf.avg_cadence_fps().unwrap_or(0.0),
            perf.worst_frame_secs * 1000.0,
            perf.worst_cadence_secs * 1000.0,
        );
    }
}
