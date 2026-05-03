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
    /// Phase 3 Step 3.0 debug knob: constant curvature coefficient
    /// `A` for the per-step parabolic-drop bend. Set via
    /// `--curvature A`. None ⇒ flat path (default). Useful for
    /// validating the curvature math on `--plain-world` before
    /// wiring k(altitude) on the wrapped planet.
    pub curvature_a: Option<f32>,
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
    /// Default 1.0 = standard sub-pixel rejection. This is the
    /// sole visual LOD gate — the hard ceiling is MAX_STACK_DEPTH
    /// in the shader, which is driven by register pressure.
    pub lod_pixels: Option<f32>,
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
    /// Enable TAAU (temporal anti-aliasing + upscale). Ray-march
    /// pass runs at half per-axis (¼ pixel count); a resolve pass
    /// reprojects the previous frame's history, neighborhood-clamps
    /// against 3×3 of the new half-res samples, and blends. Recovers
    /// full-resolution detail after ~4 frames of camera stillness.
    /// Costs: one resolve pass (cheap), two full-res RGBA16F history
    /// textures, plus the 3×3 neighborhood loads in the resolve shader.
    /// See `docs/testing/proposed-perf-speedups.md` and the TAAU
    /// discussion in this session's chat log.
    pub taa: bool,
    /// How entities are rendered.
    /// `--entity-render ray-march` (default): entities live in the
    /// world tree as `Child::EntityRef` cells; the ray-march
    /// dispatches into their voxel subtrees. Decent to ~1k.
    /// `--entity-render raster`: instanced mesh raster pass (landed
    /// in a later commit on this branch). Incompatible with TAA.
    pub entity_render_mode: crate::renderer::EntityRenderMode,
    /// Compile-time disable for the shader's tag==3 (entity)
    /// dispatch. Default false = entities enabled everywhere. Flip
    /// to true via `--no-entities` to DCE the entity branch from
    /// the ray-march shader for pure-fractal perf runs that
    /// wouldn't use entities anyway (~2 ms/frame recovery on
    /// Jerusalem nucleus 2560x1440).
    pub disable_entities: bool,
    /// Load a `.vox` or `.vxs` file as a visual entity and spawn it
    /// one cell in front of the camera at startup. Used by the
    /// entity-visibility test suite to place a known-shape entity
    /// deterministically without scripted interaction.
    pub spawn_entity: Option<std::path::PathBuf>,
    /// Number of copies of `spawn_entity` to place. Defaults to 1;
    /// higher values arrange them in a grid in front of the camera.
    pub spawn_entity_count: u32,
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
    /// Nudge the camera along an axis (`0=x, 1=y, 2=z`) by `delta`
    /// units in the current anchor-cell's local frame. Cumulative;
    /// re-normalizes `WorldPos` so cell crossings update `anchor`.
    /// Used by perf repros that need to exercise the smooth-motion
    /// LOD/pack path without full player physics.
    Step { axis: u8, delta: f32 },
    /// Cast a ray straight down from the camera through the full
    /// tree depth and reposition the camera a couple anchor cells
    /// above whatever it hits. Bypasses the normal interaction
    /// radius cap — used by perf repros that need to land the
    /// camera on top of terrain regardless of spawn location.
    /// Following breaks/places then see real hits.
    FlyToSurface,
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
                "--help" | "-h" => { print_help_and_exit(); }
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
                // Step-1 unit primitive: a single rotated TangentBlock
                // at tree depth 3 in an otherwise empty world. View
                // from above, expect a diamond silhouette.
                "--rotated-cube-test" => {
                    cfg.world_preset = WorldPreset::RotatedCubeTest;
                }
                "--dodecahedron-test" => {
                    cfg.world_preset = WorldPreset::DodecahedronTest;
                }
                "--wrapped-planet" => {
                    cfg.world_preset = WorldPreset::WrappedPlanet {
                        embedding_depth: crate::world::bootstrap::DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH,
                        slab_dims: crate::world::bootstrap::DEFAULT_WRAPPED_PLANET_SLAB_DIMS,
                        slab_depth: crate::world::bootstrap::DEFAULT_WRAPPED_PLANET_SLAB_DEPTH,
                        cell_subtree_depth: crate::world::bootstrap::DEFAULT_WRAPPED_PLANET_CELL_SUBTREE_DEPTH,
                    };
                }
                // Override the wrapped-planet's total tree depth.
                // `--planet-layers N` sets total = N. We adjust
                // `cell_subtree_depth` to absorb the difference, so
                // each slab cell's recursive subtree gets shallower /
                // deeper. `embedding_depth` and `slab_depth` stay at
                // their defaults. The slab cells are STILL anchor
                // blocks (`Child::Node`), just with N - emb - slab
                // levels of content beneath them. MUST come AFTER
                // `--wrapped-planet` on the command line.
                "--planet-layers" => {
                    if let Some(layers) = args.next().and_then(|v| v.parse::<u8>().ok()) {
                        if let WorldPreset::WrappedPlanet {
                            embedding_depth,
                            slab_depth,
                            ref mut cell_subtree_depth,
                            ..
                        } = cfg.world_preset
                        {
                            let baseline = embedding_depth + slab_depth;
                            assert!(
                                layers >= baseline,
                                "--planet-layers {} must be >= embedding+slab ({})",
                                layers, baseline,
                            );
                            *cell_subtree_depth = layers - baseline;
                        }
                    }
                }
                "--menger-world" => { cfg.world_preset = WorldPreset::Menger; }
                "--sierpinski-tet-world" => { cfg.world_preset = WorldPreset::SierpinskiTet; }
                "--cantor-dust-world" => { cfg.world_preset = WorldPreset::CantorDust; }
                "--jerusalem-cross-world" => { cfg.world_preset = WorldPreset::JerusalemCross; }
                "--sierpinski-pyramid-world" => { cfg.world_preset = WorldPreset::SierpinskiPyramid; }
                "--mausoleum-world" => { cfg.world_preset = WorldPreset::Mausoleum; }
                "--edge-scaffold-world" => { cfg.world_preset = WorldPreset::EdgeScaffold; }
                "--hollow-cube-world" => { cfg.world_preset = WorldPreset::HollowCube; }
                "--stars-world" => { cfg.world_preset = WorldPreset::Stars; }
                "--sponza-world" => {
                    cfg.world_preset = WorldPreset::Scene {
                        id: crate::world::scenes::SceneId::Sponza,
                    };
                }
                "--san-miguel-world" => {
                    cfg.world_preset = WorldPreset::Scene {
                        id: crate::world::scenes::SceneId::SanMiguel,
                    };
                }
                "--bistro-world" => {
                    cfg.world_preset = WorldPreset::Scene {
                        id: crate::world::scenes::SceneId::Bistro,
                    };
                }
                "--vox-model" => {
                    if let Some(path_str) = args.next() {
                        // Interior depth may be set before or after this
                        // flag; capture the existing value if any.
                        let interior_depth = match &cfg.world_preset {
                            WorldPreset::VoxModel { interior_depth, .. } => *interior_depth,
                            _ => 0,
                        };
                        cfg.world_preset = WorldPreset::VoxModel {
                            path: path_str.into(),
                            interior_depth,
                        };
                    }
                }
                "--vox-interior-depth" => {
                    let n: u8 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                    // Update existing VoxModel if already set, otherwise
                    // stash as a zero-path VoxModel so a later
                    // --vox-model inherits it.
                    cfg.world_preset = match std::mem::take(&mut cfg.world_preset) {
                        WorldPreset::VoxModel { path, .. } => WorldPreset::VoxModel {
                            path,
                            interior_depth: n,
                        },
                        other => {
                            // No --vox-model yet; hold the interior_depth
                            // in a zero-path placeholder. If --vox-model
                            // arrives later it will pick this up.
                            cfg.world_preset = other;
                            WorldPreset::VoxModel {
                                path: std::path::PathBuf::new(),
                                interior_depth: n,
                            }
                        }
                    };
                }
                "--plain-layers" => {
                    cfg.plain_layers = args.next().and_then(|v| v.parse().ok());
                }
                "--spawn-depth" => {
                    cfg.spawn_depth = args.next().and_then(|v| v.parse().ok());
                }
                // Phase 3 Step 3.0: per-step parabolic-drop coefficient
                // for the curvature debug knob. `--curvature 0` (default)
                // = flat. Try `--curvature 0.2` on `--plain-world` to
                // see ground curve down past the horizon.
                "--curvature" => {
                    cfg.curvature_a = args.next().and_then(|v| v.parse().ok());
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
                "--interaction-radius" => {
                    cfg.interaction_radius = args.next().and_then(|v| v.parse().ok());
                }
                "--live-sample-every" => {
                    cfg.live_sample_every_frames = args.next().and_then(|v| v.parse().ok());
                }
                "--taa" => { cfg.taa = true; }
                "--spawn-entity" => {
                    if let Some(p) = args.next() {
                        cfg.spawn_entity = Some(std::path::PathBuf::from(p));
                        if cfg.spawn_entity_count == 0 {
                            cfg.spawn_entity_count = 1;
                        }
                    }
                }
                "--spawn-entity-count" => {
                    cfg.spawn_entity_count = args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1);
                }
                "--entity-render" => {
                    if let Some(v) = args.next() {
                        cfg.entity_render_mode = match v.as_str() {
                            "ray-march" | "raymarch" => {
                                crate::renderer::EntityRenderMode::RayMarch
                            }
                            "raster" => crate::renderer::EntityRenderMode::Raster,
                            other => {
                                eprintln!(
                                    "--entity-render: unknown value {other:?} (expected ray-march|raster)",
                                );
                                crate::renderer::EntityRenderMode::RayMarch
                            }
                        };
                    }
                }
                "--no-entities" => { cfg.disable_entities = true; }
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
            if raw == "fly_to_surface" {
                return Some(ScriptCmd::FlyToSurface);
            }
            if let Some(rest) = raw.strip_prefix("step:") {
                // "step:x+", "step:y-", "step:z+:0.25"
                let (dir, delta_str) = match rest.split_once(':') {
                    Some((d, s)) => (d, Some(s)),
                    None => (rest, None),
                };
                if dir.len() == 2 {
                    let axis = match &dir[0..1] {
                        "x" => 0u8, "y" => 1, "z" => 2, _ => return None,
                    };
                    let sign: f32 = match &dir[1..2] {
                        "+" => 1.0, "-" => -1.0, _ => return None,
                    };
                    let mag: f32 = delta_str
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.10);
                    return Some(ScriptCmd::Step { axis, delta: sign * mag });
                }
            }
            eprintln!("test_runner: ignoring unknown script command {raw:?}");
            None
        })
        .collect()
}

fn print_help_and_exit() -> ! {
    println!(r#"deepspace-game — ray-marched voxel engine

USAGE:
  scripts/dev.sh -- [FLAGS]
  ./target/debug/deepspace-game [FLAGS]

WORLD PRESETS (pick one; defaults to --plain-world):
  --plain-world               40-level test world with grass/dirt/stone
  --vox-model PATH            Load .vox / .vxs model (e.g. assets/vox/soldier_729.vxs)

FRACTAL PRESETS (default plain-layers = 8):
  --menger-world              Menger sponge, bronze corners + PySpace blue edges
  --mausoleum-world           Menger geometry, authentic PySpace orbit-ochre
  --sierpinski-tet-world      4 tetrahedral corners per level, cream + apex gold
  --sierpinski-pyramid-world  4 base + 1 apex per level, limestone + gilt
  --cantor-dust-world         8 cube corners per level, 8-hue prismatic
  --jerusalem-cross-world     7 axial cells (body + 6 faces), ochre two-tone
  --edge-scaffold-world       12 edge rods per level, cyan/magenta/yellow
  --hollow-cube-world         18 edges + faces (no corners/body), brass + steel

VISIBILITY TEST PRESETS:
  --stars-world               Planet cube + distant stars at varying ribbon
                              depths; validates precision across deep pops
  --rotated-cube-test         Single 45°-Y rotated TangentBlock; depth-30
                              precision stress for the unit primitive
  --dodecahedron-test         Centre cube + 12 TangentBlocks rotated to
                              the regular-dodecahedron face normals;
                              rotation-diversity stress on renormalize
  --wrapped-planet            Wrapped-Cartesian planet: 27x2x14
                              grass/dirt/stone slab with X-axis wrap
                              (longitude). Default total tree depth: 25
                              (embedding 22 + slab 3). No curvature yet.
  --planet-layers N           (After --wrapped-planet) Override total
                              tree depth. e.g. `--planet-layers 40`
                              puts slab leaves at world-tree depth 40
                              (embedding 37 + slab 3). Stress-tests the
                              wrap math at deep precision. Must come
                              after `--wrapped-planet`.
  --wrapped-planet-tangent    (After --wrapped-planet) Slab cell anchors
                              become NodeKind::TangentBlock. The shader
                              transforms each ray into the cell's local
                              tangent-cube frame and dispatches the
                              precision-stable Cartesian DDA below — so
                              40+ deeper layers retain precision the way
                              sphere-DDA descent cannot.

MESH SCENE PRESETS (voxelized offline via tools/scene_voxelize/; see
scripts/fetch-glb-presets.sh to download source GLBs):
  --sponza-world              Crytek Sponza atrium (Khronos glTF Sample Assets)
  --san-miguel-world          Morgan McGuire's San Miguel courtyard (~10.5M tris)
  --bistro-world              Amazon Lumberyard Bistro (NVIDIA ORCA, Paris street)

WORLD TUNING:
  --plain-layers N            Tree depth (default 40 for plain, 8 for fractals)
  --vox-interior-depth N      Subdivide each imported voxel N levels

SPAWN:
  --spawn-xyz X Y Z           Override spawn position (root-cell-local, 0..3)
  --spawn-depth N             Camera anchor depth
  --spawn-yaw RAD             Yaw (radians)
  --spawn-pitch RAD           Pitch (radians)

RENDERING:
  --lod-pixels N              Nyquist floor in screen pixels (default 1.0)
  --taa                       Enable temporal anti-aliasing (TAAU)
  --disable-overlay           Hide the wry WebView overlay UI
  --disable-highlight         Hide the cursor highlight reticle

HEADLESS HARNESS:
  --render-harness            Run without opening an interactive window
  --show-window               Open the harness window (for visual verify)
  --harness-width N           Framebuffer width  (default 1280)
  --harness-height N          Framebuffer height (default 720)
  --screenshot PATH           Save a PNG on exit
  --exit-after-frames N       Quit after N rendered frames
  --timeout-secs N            Hard-kill wall-clock cap (default 5)
  --script "cmd1,cmd2,..."    Scripted actions (see scripts/ for examples)
  --suppress-startup-logs     Silence the startup_perf / spawn log spam

PERF / DEBUG:
  --shader-stats              Emit per-ray DDA step atomics (adds ~1 ms)
  --perf-trace PATH           Per-frame CSV timings
  --perf-trace-warmup N       Skip first N frames in the trace
  --live-sample-every N       Render-loop phase timings every N frames
  --force-visual-depth N      Override computed visual_depth
  --force-edit-depth N        Override computed edit_depth
  --interaction-radius N      Cursor/break reach in anchor-cells (default 6)
  --min-fps N                 Assertion floor
  --fps-warmup-frames N       Exclude from the FPS assertion
  --min-cadence-fps N         Frame-cadence assertion floor
  --cadence-warmup-frames N   Exclude from the cadence assertion
  --run-for-secs N            Minimum wall-clock before exit
  --max-frame-gap-ms N        Assertion ceiling on per-frame gaps
  --frame-gap-warmup-frames N
  --require-webview           Fail if the wry overlay never comes up

HELP:
  --help, -h                  Show this message

EXAMPLES:
  scripts/dev.sh -- --menger-world
  scripts/dev.sh -- --mausoleum-world --plain-layers 10
  scripts/dev.sh -- --vox-model assets/vox/soldier_729.vxs --plain-layers 8
  scripts/test-fractals.sh                          # screenshot every fractal
"#);
    std::process::exit(0);
}
