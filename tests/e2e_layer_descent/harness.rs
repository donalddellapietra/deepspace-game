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

    /// Raycast along the camera's CURRENT forward vector (pitch/yaw
    /// as set earlier in the script). Emits `HARNESS_PROBE
    /// direction=cursor` + `HARNESS_PROBE_AABB`. Used to check that
    /// the AABB used for the cursor highlight contains the hit
    /// point — i.e. that the highlight draws on top of the same
    /// cell the next break would edit.
    pub fn probe_cursor(mut self) -> Self {
        self.parts.push("probe_cursor".to_string());
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

    /// Sphere-only. Re-invokes `--spawn-on-surface` dispatch
    /// (`demo_sphere_surface_spawn`) for the current `anchor_depth`.
    /// See `ScriptCmd::RespawnOnSurface` for the rationale.
    pub fn respawn_on_surface(mut self) -> Self {
        self.parts.push("respawn_on_surface".to_string());
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

/// `HARNESS_PROBE_AABB`: for a cursor-direction raycast, the body-frame
/// AABB for the hit cell, the hit point, and whether the point lies
/// inside the AABB. Emitted alongside `HARNESS_PROBE direction=cursor`.
#[derive(Debug, Clone, PartialEq)]
pub struct HarnessProbeAabb {
    pub direction: String,
    pub anchor: Vec<u32>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub hit_point: [f32; 3],
    pub cam_frame: [f32; 3],
    pub inside: bool,
}

#[derive(Debug, Default, Clone)]
pub struct Trace {
    pub marks: Vec<HarnessMark>,
    pub edits: Vec<HarnessEdit>,
    pub probes: Vec<HarnessProbe>,
    pub probe_aabbs: Vec<HarnessProbeAabb>,
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
        } else if let Some(rest) = line.strip_prefix("HARNESS_PROBE_AABB ") {
            if let Some(a) = parse_probe_aabb(rest) {
                trace.probe_aabbs.push(a);
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

/// Count yellow-ish pixels OUTSIDE the screen-center crosshair
/// square. The crosshair renders ~30 yellow pixels on hit regardless
/// of whether the actual highlight glow fires, so we exclude a
/// generous center box (`exclude_radius_px` from center on each
/// axis) to make the count specific to the highlight-cell glow.
pub fn highlight_glow_pixel_count(
    path: impl AsRef<std::path::Path>,
    exclude_radius_px: u32,
) -> u32 {
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

    let cx = width as i64 / 2;
    let cy = height as i64 / 2;
    let r_excl = exclude_radius_px as i64;
    let mut glow = 0u32;
    for y in 0..height {
        for x in 0..width {
            // Skip the crosshair center box.
            if (x as i64 - cx).abs() <= r_excl && (y as i64 - cy).abs() <= r_excl {
                continue;
            }
            let i = (y * width + x) * channels;
            let r = data[i] as i32;
            let g = data[i + 1] as i32;
            let b = data[i + 2] as i32;
            // Yellow-ish: R ≥ 180, G ≥ 140, B well below both, with
            // R and G reasonably close. Tuned to catch the
            // `(1.0, 0.92, 0.18)` glow after color mixing.
            if r >= 180 && g >= 140 && b < 140 && (r - g).abs() <= 60 && r - b >= 60 {
                glow += 1;
            }
        }
    }
    glow
}

/// Count of non-sky (planet) pixels in a single horizontal row, where
/// "sky" matches the engine's sky-blue gradient (R < G < B). Useful
/// for measuring a planet's visible silhouette: a curved sphere seen
/// from a distance produces a roughly-circular outline whose width
/// peaks in the middle rows and tapers toward top/bottom.
pub fn planet_pixel_count_at_row(path: impl AsRef<std::path::Path>, row_frac: f32) -> u32 {
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

    let y = ((row_frac.clamp(0.0, 1.0)) * height as f32) as usize;
    let y = y.min(height.saturating_sub(1));
    let mut planet = 0u32;
    for x in 0..width {
        let i = (y * width + x) * channels;
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];
        if !(b > r && b > g) {
            planet += 1;
        }
    }
    planet
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

fn parse_vec3(s: &str) -> Option<[f32; 3]> {
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    let parts: Vec<f32> = inner.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    if parts.len() == 3 { Some([parts[0], parts[1], parts[2]]) } else { None }
}

fn parse_probe_aabb(rest: &str) -> Option<HarnessProbeAabb> {
    let kv = parse_kv(rest);
    Some(HarnessProbeAabb {
        direction: kv.get("direction")?.clone(),
        anchor: parse_anchor(kv.get("anchor")?)?,
        aabb_min: parse_vec3(kv.get("aabb_min")?)?,
        aabb_max: parse_vec3(kv.get("aabb_max")?)?,
        hit_point: parse_vec3(kv.get("hit_point")?)?,
        cam_frame: parse_vec3(kv.get("cam_frame")?)?,
        inside: kv.get("inside")?.parse().ok()?,
    })
}
