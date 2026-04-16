//! `TestConfig`, `ScriptCmd`, and the CLI arg parser.

use crate::world::bootstrap::{WorldPreset, DEFAULT_PLAIN_LAYERS};

#[derive(Default, Debug, Clone)]
pub struct TestConfig {
    pub render_harness: bool,
    pub show_window: bool,
    pub disable_overlay: bool,
    pub disable_highlight: bool,
    pub suppress_startup_logs: bool,
    pub force_visual_depth: Option<u32>,
    pub force_edit_depth: Option<u32>,
    pub harness_width: Option<u32>,
    pub harness_height: Option<u32>,
    pub world_preset: WorldPreset,
    pub plain_layers: Option<u8>,
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
    /// If set, the harness writes a per-frame CSV trace to this
    /// path. One row per rendered frame; header includes every
    /// phase the harness can see (CPU + GPU). Enables post-hoc
    /// analysis of worst-frame spikes, warm-up tails, and phase
    /// correlation across the run — things the `avg_ms` summary
    /// alone washes out.
    pub perf_trace: Option<String>,
    /// When `perf_trace` is set, skip the first N frames before
    /// starting to record. Defaults to 0 (record everything,
    /// including startup). Set this to skip warm-up frames.
    pub perf_trace_warmup: u32,
    /// Enable per-pixel DDA step-count atomics in the fragment
    /// shader. Adds ~0.5–1 ms per frame at 1280x720 from atomic
    /// contention, so it's off by default and only turned on for
    /// diagnostic runs. See `docs/testing/perf-isolation.md`.
    pub shader_stats: bool,
    /// Nyquist floor: pixels below this threshold get LOD-terminal.
    /// Default 1.0 = standard sub-pixel rejection. This is a
    /// FLOOR, not the primary LOD gate — the primary gate is
    /// `lod_base_depth` (ribbon-level cube shells).
    pub lod_pixels: Option<f32>,
    /// Detail budget inside the anchor cell. Each additional
    /// ribbon-pop shell gets one less level of detail. Default 4.
    /// Lower = faster + coarser; higher = slower + crisper distant
    /// content. Baked into the pipeline as the WGSL `override`
    /// `BASE_DETAIL_DEPTH`. See `docs/testing/perf-lod-diagnosis.md`.
    pub lod_base_depth: Option<u32>,
    /// Block-interaction radius, in anchor-cell units. The cursor
    /// raycast (highlight) and break/place only return hits at
    /// distances ≤ `interaction_radius × anchor_cell_size`. Default
    /// 6. At a high layer the anchor cell is physically huge so 6
    /// cells is a big world distance; at a deep layer the cell is
    /// tiny so 6 cells is a small world distance. This makes the
    /// interaction range scale with your current zoom, same as the
    /// LOD shells — symmetric cursor/interaction gate.
    pub interaction_radius: Option<u32>,
    /// When set and > 0, the live-surface render path emits a
    /// `render_live_sample` line every N frames (CPU-side phase
    /// timings only — no `device.poll(Wait)` stall). Lets us see
    /// the steady-state breakdown at 60 FPS without waiting for
    /// the `renderer_slow` 30 ms threshold. `None` or `Some(0)`
    /// disables.
    pub live_sample_every_frames: Option<u32>,
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
    /// Capture the current rendered frame to `PATH` (PNG). Fires after
    /// the current frame's render, so it reflects the state AS OF the
    /// scheduled frame — any mutations from commands later in the same
    /// tick only show up in the next frame's render.
    Screenshot(String),
    /// Set `camera.pitch` to an absolute value in radians.
    Pitch(f32),
    /// Set `camera.yaw` to an absolute value in radians.
    Yaw(f32),
    /// Run a CPU raycast straight down from the camera in world space
    /// and emit a `HARNESS_PROBE` line to stdout with the hit path.
    ProbeDown,
    /// Emit a `HARNESS_MARK` line to stdout with the given label plus
    /// the current ui_layer / anchor_depth / frame. Timeline marker
    /// for correlating screenshots to actions in a test trace.
    Emit(String),
    /// Teleport the camera to the horizontal center of the cell
    /// affected by the most recent break/place, positioned inside the
    /// bottom child of that cell at the current anchor depth.
    /// Intended use: after `zoom_in:1` following a break, this drops
    /// the camera to "one layer-N cell above the new ground" (where N
    /// is the current UI layer), matching the descent flow.
    TeleportAboveLastEdit,
}

impl TestConfig {
    pub fn from_args() -> Self {
        let mut cfg = TestConfig::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--render-harness" => { cfg.render_harness = true; }
                "--show-window" => { cfg.show_window = true; }
                "--disable-overlay" => { cfg.disable_overlay = true; }
                "--disable-highlight" => { cfg.disable_highlight = true; }
                "--suppress-startup-logs" => { cfg.suppress_startup_logs = true; }
                "--force-visual-depth" => {
                    cfg.force_visual_depth = args.next().and_then(|v| v.parse().ok());
                }
                "--force-edit-depth" => {
                    cfg.force_edit_depth = args.next().and_then(|v| v.parse().ok());
                }
                "--harness-width" => {
                    cfg.harness_width = args.next().and_then(|v| v.parse().ok());
                }
                "--harness-height" => {
                    cfg.harness_height = args.next().and_then(|v| v.parse().ok());
                }
                "--plain-world" => { cfg.world_preset = WorldPreset::PlainTest; }
                "--sphere-world" => { cfg.world_preset = WorldPreset::DemoSphere; }
                "--plain-layers" => {
                    cfg.plain_layers = args.next().and_then(|v| v.parse().ok());
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
                "--spawn-yaw" => { cfg.spawn_yaw = args.next().and_then(|v| v.parse().ok()); }
                "--spawn-pitch" => { cfg.spawn_pitch = args.next().and_then(|v| v.parse().ok()); }
                "--screenshot" => { cfg.screenshot = args.next(); }
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
                "--require-webview" => { cfg.require_webview = true; }
                "--perf-trace" => { cfg.perf_trace = args.next(); }
                "--perf-trace-warmup" => {
                    cfg.perf_trace_warmup = args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                }
                "--shader-stats" => { cfg.shader_stats = true; }
                "--lod-pixels" => {
                    cfg.lod_pixels = args.next().and_then(|v| v.parse().ok());
                }
                "--lod-base-depth" => {
                    cfg.lod_base_depth = args.next().and_then(|v| v.parse().ok());
                }
                "--interaction-radius" => {
                    cfg.interaction_radius = args.next().and_then(|v| v.parse().ok());
                }
                "--live-sample-every" => {
                    cfg.live_sample_every_frames = args.next().and_then(|v| v.parse().ok());
                }
                _ => {}
            }
        }
        cfg
    }

    pub fn plain_layers(&self) -> u8 {
        self.plain_layers.unwrap_or(DEFAULT_PLAIN_LAYERS)
    }

    pub fn harness_size(&self) -> (u32, u32) {
        (
            self.harness_width.unwrap_or(1280),
            self.harness_height.unwrap_or(720),
        )
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
            if raw == "probe_down" { return Some(ScriptCmd::ProbeDown); }
            if raw == "teleport_above_last_edit" { return Some(ScriptCmd::TeleportAboveLastEdit); }
            if let Some(n) = raw.strip_prefix("wait:") {
                if let Ok(frames) = n.parse() { return Some(ScriptCmd::Wait(frames)); }
            }
            if let Some(n) = raw.strip_prefix("zoom_in:") {
                if let Ok(steps) = n.parse() { return Some(ScriptCmd::ZoomIn(steps)); }
            }
            if let Some(n) = raw.strip_prefix("zoom_out:") {
                if let Ok(steps) = n.parse() { return Some(ScriptCmd::ZoomOut(steps)); }
            }
            if let Some(path) = raw.strip_prefix("screenshot:") {
                return Some(ScriptCmd::Screenshot(path.to_string()));
            }
            if let Some(r) = raw.strip_prefix("pitch:") {
                if let Ok(rad) = r.parse() { return Some(ScriptCmd::Pitch(rad)); }
            }
            if let Some(r) = raw.strip_prefix("yaw:") {
                if let Ok(rad) = r.parse() { return Some(ScriptCmd::Yaw(rad)); }
            }
            if let Some(label) = raw.strip_prefix("emit:") {
                return Some(ScriptCmd::Emit(label.to_string()));
            }
            eprintln!("test_runner: ignoring unknown script command {raw:?}");
            None
        })
        .collect()
}
