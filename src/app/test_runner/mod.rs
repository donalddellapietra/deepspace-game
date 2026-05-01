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
//! --plain-world           Start in the Cartesian plain test world (default).
//! --plain-layers N        Layer count for the Cartesian plain preset.
//!                         Default: 40.
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

mod config;
mod monitor;
mod runner;

pub use config::{ScriptCmd, TestConfig};
pub use monitor::{PerfSamples, TestMonitor};
pub use runner::{run_render_harness, TestRunner};
