//! Per-frame state machine for a scripted scenario + the off-screen
//! render harness main loop. The harness owns the event loop and
//! drives `App::update` / `App::upload_tree_lod` / `Renderer::render_offscreen`
//! directly, so it produces deterministic frames independent of winit
//! event delivery.

use super::perf::{print_monitor_summary, FrameSample, PerfAgg, PerfTraceWriter};
use super::{PerfSamples, ScriptCmd, TestConfig, TestMonitor};

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
    pub started_at: web_time::Instant,
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
        let started_at = web_time::Instant::now();
        let monitor = std::sync::Arc::new(TestMonitor::new());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let monitor = std::sync::Arc::clone(&monitor);
            std::thread::spawn(move || {
                use std::sync::atomic::Ordering;

                loop {
                    std::thread::sleep(web_time::Duration::from_millis(50));
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
                                run_for_secs, frame_count,
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

// winit 0.30 split EventLoop::create_window off into ActiveEventLoop;
// this harness drives the loop manually (no ApplicationHandler trait),
// so we keep the deprecated path deliberately.
#[allow(deprecated)]
pub fn run_render_harness(cfg: TestConfig) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use crate::app::{App, UserEvent};
    use crate::renderer::Renderer;
    use winit::event_loop::EventLoop;
    use winit::window::WindowAttributes;

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let mut app = App::with_test_config(cfg.clone(), event_loop.create_proxy());
    let window = Arc::new(event_loop.create_window(
        WindowAttributes::default()
            .with_title("Deep Space Render Harness")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_visible(cfg.show_window),
    )?);
    let (tree_packed, node_kinds, node_offsets, aabbs, _node_ids, root_index) =
        crate::world::gpu::pack_tree(&app.world.library, app.world.root);
    let renderer = pollster::block_on(Renderer::new(
        window.clone(),
        &tree_packed,
        &node_kinds,
        &node_offsets,
        &aabbs,
        root_index,
        wgpu::PresentMode::AutoNoVsync,
        cfg.shader_stats,
        cfg.lod_pixels.unwrap_or(1.0),
        cfg.live_sample_every_frames.unwrap_or(0),
        cfg.taa,
        // Default: entities enabled. `--no-entities` flips it off
        // so the shader's tag==3 path DCEs for pure fractal perf
        // runs that wouldn't use entities anyway.
        !cfg.disable_entities,
        cfg.entity_render_mode,
    ));
    let mut renderer = renderer;
    renderer.update_palette(&app.palette.to_gpu_palette());
    // Phase 3 Step 3.0: apply --curvature debug flag (test-harness
    // path bypasses App::finish_init, so wire the same setter here).
    if let Some(a) = cfg.curvature_a {
        eprintln!("render_harness: curvature A={a:.4}");
        renderer.set_curvature_a(a);
    }
    renderer.set_planet_lat_max(1.26);
    renderer.resize(app.harness_width, app.harness_height);
    eprintln!(
        "render_harness: resize width={} height={}",
        app.harness_width, app.harness_height,
    );
    app.window = Some(window);
    app.renderer = Some(renderer);
    app.apply_zoom();
    app.last_frame = web_time::Instant::now();

    let mut agg = PerfAgg::default();
    let mut trace_writer = cfg.perf_trace.as_ref().map(|path| {
        let w = PerfTraceWriter::new(path).expect("failed to open perf trace file");
        eprintln!("render_harness: perf trace -> {path} (warmup={})", cfg.perf_trace_warmup);
        w
    });
    let trace_warmup = cfg.perf_trace_warmup;

    loop {
        // Reset per-frame timings on the renderer. The upload-reuse
        // fast path skips update_tree/update_ribbon entirely; without
        // this reset, stale startup values would leak forward and
        // make every frame look like it uploaded a fresh tree.
        if let Some(r) = app.renderer.as_mut() {
            r.last_camera_write_ms = 0.0;
            r.last_tree_write_ms = 0.0;
            r.last_ribbon_write_ms = 0.0;
            r.last_bind_group_rebuild_ms = 0.0;
        }

        let t0 = web_time::Instant::now();
        app.update(1.0 / 60.0);
        let t_update = t0.elapsed().as_secs_f64() * 1000.0;
        let t_camera_write = app.renderer.as_ref().map(|r| r.last_camera_write_ms).unwrap_or(0.0);

        let t1 = web_time::Instant::now();
        app.upload_tree_lod();
        let t_upload_total = t1.elapsed().as_secs_f64() * 1000.0;
        let t_pack = app.last_pack_ms;
        let t_ribbon_build = app.last_ribbon_build_ms;
        let t_tree_write = app.renderer.as_ref().map(|r| r.last_tree_write_ms).unwrap_or(0.0);
        let t_ribbon_write = app.renderer.as_ref().map(|r| r.last_ribbon_write_ms).unwrap_or(0.0);
        let t_bind_group_rebuild = app.renderer.as_ref().map(|r| r.last_bind_group_rebuild_ms).unwrap_or(0.0);
        let packed_node_count = app.last_packed_node_count;
        let ribbon_len = app.last_ribbon_len;
        let effective_visual_depth = app.last_effective_visual_depth;
        let reused_gpu_tree = app.last_reused_gpu_tree;

        let t2 = web_time::Instant::now();
        app.update_highlight();
        let t_highlight = t2.elapsed().as_secs_f64() * 1000.0;
        let t_highlight_raycast = app.last_highlight_raycast_ms;
        let t_highlight_set = app.last_highlight_set_ms;

        let render_timing = if let Some(renderer) = &mut app.renderer {
            renderer.render_offscreen()
        } else {
            crate::renderer::OffscreenRenderTiming::default()
        };

        let sample = FrameSample {
            frame: agg.frame_count,
            wall_ms: app.last_frame.elapsed().as_secs_f64() * 1000.0,
            update_ms: t_update,
            camera_write_ms: t_camera_write,
            upload_total_ms: t_upload_total,
            pack_ms: t_pack,
            ribbon_build_ms: t_ribbon_build,
            tree_write_ms: t_tree_write,
            ribbon_write_ms: t_ribbon_write,
            bind_group_rebuild_ms: t_bind_group_rebuild,
            highlight_ms: t_highlight,
            highlight_raycast_ms: t_highlight_raycast,
            highlight_set_ms: t_highlight_set,
            render_total_ms: render_timing.total_ms,
            render_texture_alloc_ms: render_timing.texture_alloc_ms,
            render_view_ms: render_timing.view_ms,
            render_encode_ms: render_timing.encode_ms,
            render_submit_ms: render_timing.submit_ms,
            render_wait_ms: render_timing.wait_ms,
            submitted_done_ms: render_timing.submitted_done_ms,
            ray_count: render_timing.shader_stats.ray_count,
            hit_count: render_timing.shader_stats.hit_count,
            miss_count: render_timing.shader_stats.miss_count,
            max_iter_count: render_timing.shader_stats.max_iter_count,
            avg_steps: render_timing.shader_stats.avg_steps(),
            max_steps: render_timing.shader_stats.max_steps,
            avg_steps_oob: render_timing.shader_stats.avg_steps_oob(),
            avg_steps_empty: render_timing.shader_stats.avg_steps_empty(),
            avg_steps_descend: render_timing.shader_stats.avg_steps_descend(),
            avg_steps_lod_terminal: render_timing.shader_stats.avg_steps_lod_terminal(),
            avg_steps_would_cull: render_timing.shader_stats.avg_steps_would_cull(),
            avg_loads_tree: render_timing.shader_stats.avg_loads_tree(),
            avg_loads_offsets: render_timing.shader_stats.avg_loads_offsets(),
            avg_loads_kinds: render_timing.shader_stats.avg_loads_kinds(),
            avg_loads_ribbon: render_timing.shader_stats.avg_loads_ribbon(),
            avg_steps_per_hit: render_timing.shader_stats.avg_steps_per_hit(),
            avg_steps_per_miss: render_timing.shader_stats.avg_steps_per_miss(),
            packed_node_count,
            ribbon_len,
            effective_visual_depth,
            reused_gpu_tree,
        };
        agg.record(&sample);
        if let Some(w) = trace_writer.as_mut() {
            if sample.frame >= trace_warmup {
                w.write(&sample);
            }
        }

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
            app.handle_script_cmd(cmd, frame);
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

    agg.print_summary();
    if let Some(w) = trace_writer {
        if let Err(e) = w.finish() {
            eprintln!("perf_trace: failed to flush trace file: {e}");
        }
    }

    Ok(())
}

