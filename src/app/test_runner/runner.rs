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
        window.clone(),
        &tree_data,
        &node_kinds,
        root_index,
        wgpu::PresentMode::AutoNoVsync,
    ));
    let mut renderer = renderer;
    renderer.resize(app.harness_width, app.harness_height);
    eprintln!(
        "render_harness: resize width={} height={}",
        app.harness_width, app.harness_height,
    );
    app.window = Some(window);
    app.renderer = Some(renderer);
    app.apply_zoom();
    app.last_frame = std::time::Instant::now();

    let mut total_update = 0.0f64;
    let mut total_upload = 0.0f64;
    let mut total_highlight = 0.0f64;
    let mut total_highlight_raycast = 0.0f64;
    let mut total_highlight_set = 0.0f64;
    let mut total_render = 0.0f64;
    let mut total_render_texture_alloc = 0.0f64;
    let mut total_render_view = 0.0f64;
    let mut total_render_encode = 0.0f64;
    let mut total_render_submit = 0.0f64;
    let mut total_render_wait = 0.0f64;
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
        let t_highlight_raycast = app.last_highlight_raycast_ms;
        let t_highlight_set = app.last_highlight_set_ms;

        let render_timing = if let Some(renderer) = &mut app.renderer {
            renderer.render_offscreen()
        } else {
            crate::renderer::OffscreenRenderTiming::default()
        };
        let t_render = render_timing.total_ms;

        total_update += t_update;
        total_upload += t_upload;
        total_highlight += t_highlight;
        total_highlight_raycast += t_highlight_raycast;
        total_highlight_set += t_highlight_set;
        total_render += t_render;
        total_render_texture_alloc += render_timing.texture_alloc_ms;
        total_render_view += render_timing.view_ms;
        total_render_encode += render_timing.encode_ms;
        total_render_submit += render_timing.submit_ms;
        total_render_wait += render_timing.wait_ms;
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

    if frame_count > 0 {
        eprintln!(
            "render_harness_timing avg_ms update={:.3} upload={:.3} highlight={:.3} highlight_raycast={:.3} highlight_set={:.3} render={:.3} render_texture_alloc={:.3} render_view={:.3} render_encode={:.3} render_submit={:.3} render_wait={:.3} total={:.3}",
            total_update / frame_count as f64,
            total_upload / frame_count as f64,
            total_highlight / frame_count as f64,
            total_highlight_raycast / frame_count as f64,
            total_highlight_set / frame_count as f64,
            total_render / frame_count as f64,
            total_render_texture_alloc / frame_count as f64,
            total_render_view / frame_count as f64,
            total_render_encode / frame_count as f64,
            total_render_submit / frame_count as f64,
            total_render_wait / frame_count as f64,
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
