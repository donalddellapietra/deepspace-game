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
//! cargo run -- --spawn-depth 12 --screenshot /tmp/layer10.png \
//!     --script "wait:30,break,wait:30" --exit-after-frames 90
//! ```

#[derive(Default, Debug, Clone)]
pub struct TestConfig {
    pub spawn_depth: Option<u8>,
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
                "--spawn-depth" => {
                    cfg.spawn_depth = args.next().and_then(|v| v.parse().ok());
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
        self.screenshot.is_some()
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
