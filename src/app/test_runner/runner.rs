//! Per-frame state machine for a scripted scenario + the off-screen
//! render harness main loop. The harness owns the event loop and
//! drives `App::update` / `App::upload_tree_lod` / `Renderer::render_offscreen`
//! directly, so it produces deterministic frames independent of winit
//! event delivery.

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
    let (tree_packed, node_kinds, node_offsets, _node_ids, root_index) =
        crate::world::gpu::pack_tree(&app.world.library, app.world.root);
    let renderer = pollster::block_on(Renderer::new(
        window.clone(),
        &tree_packed,
        &node_kinds,
        &node_offsets,
        root_index,
        wgpu::PresentMode::AutoNoVsync,
        cfg.shader_stats,
        cfg.lod_pixels.unwrap_or(1.0),
        cfg.lod_base_depth.unwrap_or(4),
        cfg.live_sample_every_frames.unwrap_or(0),
        cfg.taa,
    ));
    let mut renderer = renderer;
    renderer.update_palette(&app.palette.to_gpu_palette());
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
            gpu_pass_ms: render_timing.gpu_pass_ms,
            gpu_readback_ms: render_timing.gpu_readback_ms,
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

/// One row in the per-frame trace. Mirrors the CSV header.
#[derive(Debug, Clone, Copy)]
struct FrameSample {
    frame: u32,
    wall_ms: f64,
    update_ms: f64,
    camera_write_ms: f64,
    upload_total_ms: f64,
    pack_ms: f64,
    ribbon_build_ms: f64,
    tree_write_ms: f64,
    ribbon_write_ms: f64,
    bind_group_rebuild_ms: f64,
    highlight_ms: f64,
    highlight_raycast_ms: f64,
    highlight_set_ms: f64,
    render_total_ms: f64,
    render_texture_alloc_ms: f64,
    render_view_ms: f64,
    render_encode_ms: f64,
    render_submit_ms: f64,
    render_wait_ms: f64,
    gpu_pass_ms: Option<f64>,
    gpu_readback_ms: f64,
    submitted_done_ms: Option<f64>,
    ray_count: u32,
    hit_count: u32,
    miss_count: u32,
    max_iter_count: u32,
    avg_steps: f64,
    max_steps: u32,
    avg_steps_oob: f64,
    avg_steps_empty: f64,
    avg_steps_descend: f64,
    avg_steps_lod_terminal: f64,
    packed_node_count: u32,
    ribbon_len: u32,
    effective_visual_depth: u32,
    reused_gpu_tree: bool,
}

/// CSV writer for the per-frame trace. Buffered; flushed on finish.
struct PerfTraceWriter {
    path: String,
    writer: std::io::BufWriter<std::fs::File>,
}

impl PerfTraceWriter {
    fn new(path: &str) -> std::io::Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        use std::io::Write;
        writeln!(
            writer,
            "frame,wall_ms,update_ms,camera_write_ms,upload_total_ms,pack_ms,ribbon_build_ms,tree_write_ms,ribbon_write_ms,bind_group_rebuild_ms,highlight_ms,highlight_raycast_ms,highlight_set_ms,render_total_ms,render_texture_alloc_ms,render_view_ms,render_encode_ms,render_submit_ms,render_wait_ms,gpu_pass_ms,gpu_readback_ms,submitted_done_ms,ray_count,hit_count,miss_count,max_iter_count,avg_steps,max_steps,packed_node_count,ribbon_len,effective_visual_depth,reused_gpu_tree"
        )?;
        Ok(Self { path: path.to_string(), writer })
    }

    fn write(&mut self, s: &FrameSample) {
        use std::io::Write;
        let gpu = s.gpu_pass_ms.map(|v| format!("{v:.4}")).unwrap_or_else(|| String::new());
        let submitted_done = s.submitted_done_ms.map(|v| format!("{v:.4}")).unwrap_or_else(|| String::new());
        let _ = writeln!(
            self.writer,
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{},{},{},{},{},{:.2},{},{},{},{},{}",
            s.frame, s.wall_ms,
            s.update_ms, s.camera_write_ms,
            s.upload_total_ms, s.pack_ms, s.ribbon_build_ms, s.tree_write_ms, s.ribbon_write_ms, s.bind_group_rebuild_ms,
            s.highlight_ms, s.highlight_raycast_ms, s.highlight_set_ms,
            s.render_total_ms, s.render_texture_alloc_ms, s.render_view_ms, s.render_encode_ms, s.render_submit_ms, s.render_wait_ms,
            gpu, s.gpu_readback_ms, submitted_done,
            s.ray_count, s.hit_count, s.miss_count, s.max_iter_count, s.avg_steps, s.max_steps,
            s.packed_node_count, s.ribbon_len, s.effective_visual_depth,
            u32::from(s.reused_gpu_tree),
        );
    }

    fn finish(mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.writer.flush()?;
        eprintln!("perf_trace: flushed -> {}", self.path);
        Ok(())
    }
}

/// Running accumulator: sums, counts, worst-frame tracking, and
/// `gpu_pass_ms` fraction (only over frames where the GPU reported
/// a value). Prints a structured, single-line summary at the end.
#[derive(Default)]
struct PerfAgg {
    frame_count: u32,
    gpu_pass_count: u32,
    sum_update: f64,
    sum_camera_write: f64,
    sum_upload: f64,
    sum_pack: f64,
    sum_ribbon_build: f64,
    sum_tree_write: f64,
    sum_ribbon_write: f64,
    sum_bind_group_rebuild: f64,
    sum_highlight: f64,
    sum_hi_raycast: f64,
    sum_hi_set: f64,
    sum_render: f64,
    sum_render_alloc: f64,
    sum_render_view: f64,
    sum_render_encode: f64,
    sum_render_submit: f64,
    sum_render_wait: f64,
    sum_gpu_pass: f64,
    sum_gpu_readback: f64,
    sum_submitted_done: f64,
    submitted_done_count: u32,
    sum_ray_count: u64,
    sum_hit_count: u64,
    sum_miss_count: u64,
    sum_max_iter_count: u64,
    sum_avg_steps: f64,
    max_max_steps: u32,
    worst_avg_steps: f64,
    worst_avg_steps_frame: u32,
    sum_avg_steps_oob: f64,
    sum_avg_steps_empty: f64,
    sum_avg_steps_descend: f64,
    sum_avg_steps_lod_terminal: f64,
    sum_packed_node_count: u64,
    sum_ribbon_len: u64,
    worst_total_ms: f64,
    worst_total_frame: u32,
    worst_gpu_ms: f64,
    worst_gpu_frame: u32,
    worst_upload_ms: f64,
    worst_upload_frame: u32,
    max_packed_node_count: u32,
    max_ribbon_len: u32,
}

impl PerfAgg {
    fn record(&mut self, s: &FrameSample) {
        self.frame_count += 1;
        self.sum_update += s.update_ms;
        self.sum_camera_write += s.camera_write_ms;
        self.sum_upload += s.upload_total_ms;
        self.sum_pack += s.pack_ms;
        self.sum_ribbon_build += s.ribbon_build_ms;
        self.sum_tree_write += s.tree_write_ms;
        self.sum_ribbon_write += s.ribbon_write_ms;
        self.sum_bind_group_rebuild += s.bind_group_rebuild_ms;
        self.sum_highlight += s.highlight_ms;
        self.sum_hi_raycast += s.highlight_raycast_ms;
        self.sum_hi_set += s.highlight_set_ms;
        self.sum_render += s.render_total_ms;
        self.sum_render_alloc += s.render_texture_alloc_ms;
        self.sum_render_view += s.render_view_ms;
        self.sum_render_encode += s.render_encode_ms;
        self.sum_render_submit += s.render_submit_ms;
        self.sum_render_wait += s.render_wait_ms;
        self.sum_gpu_readback += s.gpu_readback_ms;
        self.sum_packed_node_count += s.packed_node_count as u64;
        self.sum_ribbon_len += s.ribbon_len as u64;
        if s.packed_node_count > self.max_packed_node_count {
            self.max_packed_node_count = s.packed_node_count;
        }
        if s.ribbon_len > self.max_ribbon_len {
            self.max_ribbon_len = s.ribbon_len;
        }
        if let Some(v) = s.gpu_pass_ms {
            self.sum_gpu_pass += v;
            self.gpu_pass_count += 1;
            if v > self.worst_gpu_ms {
                self.worst_gpu_ms = v;
                self.worst_gpu_frame = s.frame;
            }
        }
        if let Some(v) = s.submitted_done_ms {
            self.sum_submitted_done += v;
            self.submitted_done_count += 1;
        }
        self.sum_ray_count += s.ray_count as u64;
        self.sum_hit_count += s.hit_count as u64;
        self.sum_miss_count += s.miss_count as u64;
        self.sum_max_iter_count += s.max_iter_count as u64;
        self.sum_avg_steps += s.avg_steps;
        self.sum_avg_steps_oob += s.avg_steps_oob;
        self.sum_avg_steps_empty += s.avg_steps_empty;
        self.sum_avg_steps_descend += s.avg_steps_descend;
        self.sum_avg_steps_lod_terminal += s.avg_steps_lod_terminal;
        if s.max_steps > self.max_max_steps {
            self.max_max_steps = s.max_steps;
        }
        if s.avg_steps > self.worst_avg_steps {
            self.worst_avg_steps = s.avg_steps;
            self.worst_avg_steps_frame = s.frame;
        }
        let total = s.update_ms + s.upload_total_ms + s.highlight_ms + s.render_total_ms;
        if total > self.worst_total_ms {
            self.worst_total_ms = total;
            self.worst_total_frame = s.frame;
        }
        if s.upload_total_ms > self.worst_upload_ms {
            self.worst_upload_ms = s.upload_total_ms;
            self.worst_upload_frame = s.frame;
        }
    }

    fn print_summary(&self) {
        if self.frame_count == 0 {
            return;
        }
        let n = self.frame_count as f64;
        let gpu_avg = if self.gpu_pass_count > 0 {
            self.sum_gpu_pass / self.gpu_pass_count as f64
        } else {
            0.0
        };
        let submitted_done_avg = if self.submitted_done_count > 0 {
            self.sum_submitted_done / self.submitted_done_count as f64
        } else {
            0.0
        };
        let total_avg = (self.sum_update + self.sum_upload + self.sum_highlight + self.sum_render) / n;
        eprintln!(
            "render_harness_timing avg_ms update={:.3} camera_write={:.3} upload={:.3} pack={:.3} ribbon_build={:.3} tree_write={:.3} ribbon_write={:.3} bind_group_rebuild={:.3} highlight={:.3} highlight_raycast={:.3} highlight_set={:.3} render={:.3} render_texture_alloc={:.3} render_view={:.3} render_encode={:.3} render_submit={:.3} render_wait={:.3} gpu_pass={:.3} gpu_pass_samples={} gpu_readback={:.3} submitted_done={:.3} submitted_done_samples={} total={:.3}",
            self.sum_update / n,
            self.sum_camera_write / n,
            self.sum_upload / n,
            self.sum_pack / n,
            self.sum_ribbon_build / n,
            self.sum_tree_write / n,
            self.sum_ribbon_write / n,
            self.sum_bind_group_rebuild / n,
            self.sum_highlight / n,
            self.sum_hi_raycast / n,
            self.sum_hi_set / n,
            self.sum_render / n,
            self.sum_render_alloc / n,
            self.sum_render_view / n,
            self.sum_render_encode / n,
            self.sum_render_submit / n,
            self.sum_render_wait / n,
            gpu_avg,
            self.gpu_pass_count,
            self.sum_gpu_readback / n,
            submitted_done_avg,
            self.submitted_done_count,
            total_avg,
        );
        eprintln!(
            "render_harness_worst total_ms={:.3}@frame{} gpu_ms={:.3}@frame{} upload_ms={:.3}@frame{}",
            self.worst_total_ms, self.worst_total_frame,
            self.worst_gpu_ms, self.worst_gpu_frame,
            self.worst_upload_ms, self.worst_upload_frame,
        );
        eprintln!(
            "render_harness_workload frames={} avg_packed_nodes={} max_packed_nodes={} avg_ribbon_len={} max_ribbon_len={}",
            self.frame_count,
            self.sum_packed_node_count / self.frame_count as u64,
            self.max_packed_node_count,
            self.sum_ribbon_len / self.frame_count as u64,
            self.max_ribbon_len,
        );
        let avg_steps_overall = self.sum_avg_steps / n;
        let hit_frac = if self.sum_ray_count == 0 {
            0.0
        } else {
            self.sum_hit_count as f64 / self.sum_ray_count as f64
        };
        let max_iter_frac = if self.sum_ray_count == 0 {
            0.0
        } else {
            self.sum_max_iter_count as f64 / self.sum_ray_count as f64
        };
        eprintln!(
            "render_harness_shader frames={} avg_steps={:.2} worst_avg_steps={:.2}@frame{} max_steps={} hit_fraction={:.4} max_iter_fraction={:.6} avg_oob={:.2} avg_empty={:.2} avg_descend={:.2} avg_lod_terminal={:.2}",
            self.frame_count,
            avg_steps_overall,
            self.worst_avg_steps, self.worst_avg_steps_frame,
            self.max_max_steps,
            hit_frac,
            max_iter_frac,
            self.sum_avg_steps_oob / n,
            self.sum_avg_steps_empty / n,
            self.sum_avg_steps_descend / n,
            self.sum_avg_steps_lod_terminal / n,
        );
    }
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
