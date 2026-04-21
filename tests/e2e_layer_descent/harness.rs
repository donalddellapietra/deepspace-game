//! Test-side helpers for the e2e layer-descent suite.
//!
//! - `ScriptBuilder` composes the `--script` string.
//! - `run()` spawns the binary with the given args + script, captures
//!   stdout, parses every `HARNESS_*` line into a typed record, and
//!   returns them as a `Trace`.
//!
//! Grammar of parsed stdout records (one per line):
//!
//! ```text
//! HARNESS_MARK  label=<str> ui_layer=<u32> anchor_depth=<u32> frame=<u64>
//! HARNESS_EDIT  action=<broke|placed> anchor=[...] changed=<bool> ui_layer=<u32> anchor_depth=<u32>
//! HARNESS_PROBE direction=<str> hit=<bool> anchor=[...] ui_layer=<u32> anchor_depth=<u32>
//! ```
//!
//! Each line's fields are whitespace-separated `key=value` pairs.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// Gitignored artifact directory for a test scenario.
///
/// Returns `<project root>/tmp/<name>/`, creating it if missing. The
/// project root is `CARGO_MANIFEST_DIR` — Cargo's per-crate directory
/// — which is stable across machines and independent of CWD when the
/// test binary runs.
pub fn tmp_dir(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp").join(name);
    std::fs::create_dir_all(&root)
        .unwrap_or_else(|e| panic!("failed to create tmp dir {root:?}: {e}"));
    root
}

/// Builds a comma-separated `--script` argument.
pub struct ScriptBuilder {
    parts: Vec<String>,
}

impl ScriptBuilder {
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    pub fn wait(mut self, frames: u32) -> Self {
        self.parts.push(format!("wait:{frames}"));
        self
    }

    pub fn break_(mut self) -> Self {
        self.parts.push("break".to_string());
        self
    }

    pub fn place(mut self) -> Self {
        self.parts.push("place".to_string());
        self
    }

    pub fn zoom_in(mut self, steps: u32) -> Self {
        self.parts.push(format!("zoom_in:{steps}"));
        self
    }

    pub fn zoom_out(mut self, steps: u32) -> Self {
        self.parts.push(format!("zoom_out:{steps}"));
        self
    }

    pub fn screenshot(mut self, path: impl Into<String>) -> Self {
        self.parts.push(format!("screenshot:{}", path.into()));
        self
    }

    pub fn pitch(mut self, rad: f32) -> Self {
        self.parts.push(format!("pitch:{rad}"));
        self
    }

    pub fn yaw(mut self, rad: f32) -> Self {
        self.parts.push(format!("yaw:{rad}"));
        self
    }

    pub fn probe_down(mut self) -> Self {
        self.parts.push("probe_down".to_string());
        self
    }

    pub fn emit(mut self, label: impl Into<String>) -> Self {
        self.parts.push(format!("emit:{}", label.into()));
        self
    }

    pub fn teleport_above_last_edit(mut self) -> Self {
        self.parts.push("teleport_above_last_edit".to_string());
        self
    }

    /// Position the camera INSIDE the just-broken cell at its inner
    /// face (sphere: inner-r; Cartesian: -y). Unlike
    /// `teleport_above_last_edit`, this does NOT push an extra slot
    /// onto the anchor — the camera stays at the same total depth as
    /// the break. Used by the dig-a-hole descent flow: break a cell,
    /// step to its floor, probe+break the cell beneath, repeat.
    pub fn dig_step_down(mut self) -> Self {
        self.parts.push("dig_step_down".to_string());
        self
    }

    /// Shorthand for `pitch:-1.5707` (straight down, world frame).
    pub fn look_down(self) -> Self {
        self.pitch(-std::f32::consts::FRAC_PI_2)
    }

    /// Shorthand for `pitch:+1.5707` (straight up, world frame).
    pub fn look_up(self) -> Self {
        self.pitch(std::f32::consts::FRAC_PI_2)
    }

    pub fn compile(&self) -> String {
        self.parts.join(",")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessMark {
    pub label: String,
    pub ui_layer: u32,
    pub anchor_depth: u32,
    pub frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessEdit {
    pub action: String,
    pub anchor: Vec<u32>,
    pub changed: bool,
    pub ui_layer: u32,
    pub anchor_depth: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessProbe {
    pub direction: String,
    pub hit: bool,
    pub anchor: Vec<u32>,
    pub ui_layer: u32,
    pub anchor_depth: u32,
}

#[derive(Debug, Default, Clone)]
pub struct Trace {
    pub marks: Vec<HarnessMark>,
    pub edits: Vec<HarnessEdit>,
    pub probes: Vec<HarnessProbe>,
    pub stdout: String,
    pub stderr: String,
    pub exit_success: bool,
}

/// Run the game binary with the given args + script and return the
/// parsed harness trace.
///
/// The binary path is resolved at test-compile time via
/// `env!("CARGO_BIN_EXE_deepspace-game")`, which Cargo sets for every
/// integration test.
pub fn run(args: &[&str], script: &ScriptBuilder) -> Trace {
    let bin = env!("CARGO_BIN_EXE_deepspace-game");
    let script_arg = script.compile();
    let mut cmd = Command::new(bin);
    cmd.args(args);
    if !script_arg.is_empty() {
        cmd.arg("--script").arg(&script_arg);
    }
    let output = cmd.output().expect("failed to spawn deepspace-game");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    let mut trace = Trace {
        stdout: stdout.clone(),
        stderr,
        exit_success: output.status.success(),
        ..Default::default()
    };

    for line in stdout.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("HARNESS_MARK ") {
            if let Some(m) = parse_mark(rest) {
                trace.marks.push(m);
            }
        } else if let Some(rest) = line.strip_prefix("HARNESS_EDIT ") {
            if let Some(e) = parse_edit(rest) {
                trace.edits.push(e);
            }
        } else if let Some(rest) = line.strip_prefix("HARNESS_PROBE ") {
            if let Some(p) = parse_probe(rest) {
                trace.probes.push(p);
            }
        }
    }
    trace
}

fn parse_kv(s: &str) -> HashMap<String, String> {
    s.split_whitespace()
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect()
}

fn parse_anchor(s: &str) -> Option<Vec<u32>> {
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|t| t.trim().parse::<u32>().ok())
        .collect()
}

fn parse_mark(rest: &str) -> Option<HarnessMark> {
    let kv = parse_kv(rest);
    Some(HarnessMark {
        label: kv.get("label")?.clone(),
        ui_layer: kv.get("ui_layer")?.parse().ok()?,
        anchor_depth: kv.get("anchor_depth")?.parse().ok()?,
        frame: kv.get("frame")?.parse().ok()?,
    })
}

fn parse_edit(rest: &str) -> Option<HarnessEdit> {
    let kv = parse_kv(rest);
    Some(HarnessEdit {
        action: kv.get("action")?.clone(),
        anchor: parse_anchor(kv.get("anchor")?)?,
        changed: kv.get("changed")?.parse().ok()?,
        ui_layer: kv.get("ui_layer")?.parse().ok()?,
        anchor_depth: kv.get("anchor_depth")?.parse().ok()?,
    })
}

/// Fraction of pixels in the top half of `path` whose blue channel is
/// strictly greater than both red and green. Sky-blue in the engine
/// renders ~`(162, 196, 229)` (R<G<B), so pure sky gives `1.0`.
/// Grass renders ~`(205, 225, 177)` (B<R<G), so pure grass gives `0.0`.
/// A nested-aperture "sky at the end of the tunnel" frame scores
/// somewhere in between; the test picks a threshold.
pub fn sky_dominance_top_half(path: impl AsRef<std::path::Path>) -> f32 {
    let file = std::fs::File::open(path.as_ref())
        .unwrap_or_else(|e| panic!("open {}: {e}", path.as_ref().display()));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read png header");
    let info = reader.info().clone();
    let (width, height) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unsupported png color type {other:?}"),
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).expect("decode png frame");
    let data = &buf[..frame.buffer_size()];

    let half = height / 2;
    let mut sky = 0usize;
    let mut total = 0usize;
    for y in 0..half {
        for x in 0..width {
            let i = (y * width + x) * channels;
            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];
            if b > r && b > g {
                sky += 1;
            }
            total += 1;
        }
    }
    if total == 0 { 0.0 } else { sky as f32 / total as f32 }
}

fn parse_probe(rest: &str) -> Option<HarnessProbe> {
    let kv = parse_kv(rest);
    Some(HarnessProbe {
        direction: kv.get("direction")?.clone(),
        hit: kv.get("hit")?.parse().ok()?,
        anchor: parse_anchor(kv.get("anchor")?)?,
        ui_layer: kv.get("ui_layer")?.parse().ok()?,
        anchor_depth: kv.get("anchor_depth")?.parse().ok()?,
    })
}
