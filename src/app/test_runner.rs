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
//! --spawn-depth N         Camera anchor starts at depth N (default 4).
//! --screenshot PATH       Capture the rendered frame to PATH (PNG)
//!                         after the warm-up + script settle, then exit.
//! --exit-after-frames N   Exit after N rendered frames (default ~120).
//! --timeout-secs SECS     Wall-clock kill switch (default 5.0). Triggers
//!                         a screenshot + exit if the frame budget hasn't
//!                         already done so. Catches perf regressions:
//!                         a hung shader can't run a test forever.
//! --script CMDS           Comma-separated commands run in order:
//!                           break          left-click (break a block)
//!                           place          right-click (place a block)
//!                           wait:N         skip N frames
//!                         (`screenshot` and exit are handled by the
//!                          flags above — no need to inline them.)
//! ```
//!
//! Example:
//!
//! ```bash
//! cargo run -- --render-harness --spawn-depth 12 --screenshot /tmp/layer10.png \
//!     --script "wait:30,break,wait:30" --exit-after-frames 90
//! ```

use std::sync::Arc;

use crate::app::App;
use crate::renderer::Renderer;
use winit::event_loop::EventLoop;
use winit::window::WindowAttributes;

#[derive(Default, Debug, Clone)]
pub struct TestConfig {
    pub render_harness: bool,
    pub show_window: bool,
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
    pub script: Vec<ScriptCmd>,
}

#[derive(Debug, Clone)]
pub enum ScriptCmd {
    Break,
    Place,
    Wait(u32),
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
    }

    pub fn use_render_harness(&self) -> bool {
        self.render_harness
            || self.screenshot.is_some()
            || self.exit_after_frames.is_some()
            || !self.script.is_empty()
    }
}

fn parse_script(s: &str) -> Vec<ScriptCmd> {
    s.split(',')
        .filter_map(|raw| {
            let raw = raw.trim();
            if raw.is_empty() { return None; }
            if raw == "break" { return Some(ScriptCmd::Break); }
            if raw == "place" { return Some(ScriptCmd::Place); }
            if let Some(n) = raw.strip_prefix("wait:") {
                if let Ok(frames) = n.parse() { return Some(ScriptCmd::Wait(frames)); }
            }
            eprintln!("test_runner: ignoring unknown script command {raw:?}");
            None
        })
        .collect()
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
}

impl TestRunner {
    pub fn from_config(cfg: TestConfig) -> Option<Self> {
        if !cfg.is_active() { return None; }
        // Default: 120 frames warm-up before screenshot/exit.
        let exit_after = cfg.exit_after_frames.unwrap_or(120);
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
            started_at: std::time::Instant::now(),
            timeout_secs: cfg.timeout_secs.unwrap_or(5.0),
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
