//! Per-frame trace writer + running aggregator for the render
//! harness. `FrameSample` is the row schema; `PerfTraceWriter` writes
//! the CSV; `PerfAgg` accumulates sums / worsts and prints the
//! single-line summaries `print_summary` emits at the end.

use super::TestMonitor;

/// One row in the per-frame trace. Mirrors the CSV header.
#[derive(Debug, Clone, Copy)]
pub(super) struct FrameSample {
    pub frame: u32,
    pub wall_ms: f64,
    pub update_ms: f64,
    pub camera_write_ms: f64,
    pub upload_total_ms: f64,
    pub pack_ms: f64,
    pub ribbon_build_ms: f64,
    pub tree_write_ms: f64,
    pub ribbon_write_ms: f64,
    pub bind_group_rebuild_ms: f64,
    pub highlight_ms: f64,
    pub highlight_raycast_ms: f64,
    pub highlight_set_ms: f64,
    pub render_total_ms: f64,
    pub render_texture_alloc_ms: f64,
    pub render_view_ms: f64,
    pub render_encode_ms: f64,
    pub render_submit_ms: f64,
    pub render_wait_ms: f64,
    pub submitted_done_ms: Option<f64>,
    pub ray_count: u32,
    pub hit_count: u32,
    pub miss_count: u32,
    pub max_iter_count: u32,
    pub avg_steps: f64,
    pub max_steps: u32,
    pub avg_steps_oob: f64,
    pub avg_steps_empty: f64,
    pub avg_steps_descend: f64,
    pub avg_steps_lod_terminal: f64,
    pub avg_steps_would_cull: f64,
    pub avg_loads_tree: f64,
    pub avg_loads_offsets: f64,
    pub avg_loads_kinds: f64,
    pub avg_loads_ribbon: f64,
    pub avg_steps_per_hit: f64,
    pub avg_steps_per_miss: f64,
    pub packed_node_count: u32,
    pub ribbon_len: u32,
    pub effective_visual_depth: u32,
    pub reused_gpu_tree: bool,
}

/// CSV writer for the per-frame trace. Buffered; flushed on finish.
pub(super) struct PerfTraceWriter {
    path: String,
    writer: std::io::BufWriter<std::fs::File>,
}

impl PerfTraceWriter {
    pub(super) fn new(path: &str) -> std::io::Result<Self> {
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
            "frame,wall_ms,update_ms,camera_write_ms,upload_total_ms,pack_ms,ribbon_build_ms,tree_write_ms,ribbon_write_ms,bind_group_rebuild_ms,highlight_ms,highlight_raycast_ms,highlight_set_ms,render_total_ms,render_texture_alloc_ms,render_view_ms,render_encode_ms,render_submit_ms,render_wait_ms,submitted_done_ms,ray_count,hit_count,miss_count,max_iter_count,avg_steps,max_steps,packed_node_count,ribbon_len,effective_visual_depth,reused_gpu_tree"
        )?;
        Ok(Self { path: path.to_string(), writer })
    }

    pub(super) fn write(&mut self, s: &FrameSample) {
        use std::io::Write;
        let submitted_done = s.submitted_done_ms.map(|v| format!("{v:.4}")).unwrap_or_else(|| String::new());
        let _ = writeln!(
            self.writer,
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{:.2},{},{},{},{},{}",
            s.frame, s.wall_ms,
            s.update_ms, s.camera_write_ms,
            s.upload_total_ms, s.pack_ms, s.ribbon_build_ms, s.tree_write_ms, s.ribbon_write_ms, s.bind_group_rebuild_ms,
            s.highlight_ms, s.highlight_raycast_ms, s.highlight_set_ms,
            s.render_total_ms, s.render_texture_alloc_ms, s.render_view_ms, s.render_encode_ms, s.render_submit_ms, s.render_wait_ms,
            submitted_done,
            s.ray_count, s.hit_count, s.miss_count, s.max_iter_count, s.avg_steps, s.max_steps,
            s.packed_node_count, s.ribbon_len, s.effective_visual_depth,
            u32::from(s.reused_gpu_tree),
        );
    }

    pub(super) fn finish(mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.writer.flush()?;
        eprintln!("perf_trace: flushed -> {}", self.path);
        Ok(())
    }
}

/// Running accumulator: sums, counts, worst-frame tracking.
/// Prints a structured, single-line summary at the end.
#[derive(Default)]
pub(super) struct PerfAgg {
    pub(super) frame_count: u32,
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
    sum_avg_steps_would_cull: f64,
    sum_avg_loads_tree: f64,
    sum_avg_loads_offsets: f64,
    sum_avg_loads_kinds: f64,
    sum_avg_loads_ribbon: f64,
    sum_avg_steps_per_hit: f64,
    sum_avg_steps_per_miss: f64,
    sum_packed_node_count: u64,
    sum_ribbon_len: u64,
    worst_total_ms: f64,
    worst_total_frame: u32,
    /// Worst-frame GPU-bound time, sourced from `submitted_done_ms`
    /// (the authoritative Metal signal — queue.submit → on_submitted_work_done).
    worst_submitted_done_ms: f64,
    worst_submitted_done_frame: u32,
    worst_upload_ms: f64,
    worst_upload_frame: u32,
    max_packed_node_count: u32,
    max_ribbon_len: u32,
}

impl PerfAgg {
    pub(super) fn record(&mut self, s: &FrameSample) {
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
        self.sum_packed_node_count += s.packed_node_count as u64;
        self.sum_ribbon_len += s.ribbon_len as u64;
        if s.packed_node_count > self.max_packed_node_count {
            self.max_packed_node_count = s.packed_node_count;
        }
        if s.ribbon_len > self.max_ribbon_len {
            self.max_ribbon_len = s.ribbon_len;
        }
        if let Some(v) = s.submitted_done_ms {
            self.sum_submitted_done += v;
            self.submitted_done_count += 1;
            if v > self.worst_submitted_done_ms {
                self.worst_submitted_done_ms = v;
                self.worst_submitted_done_frame = s.frame;
            }
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
        self.sum_avg_steps_would_cull += s.avg_steps_would_cull;
        self.sum_avg_loads_tree += s.avg_loads_tree;
        self.sum_avg_loads_offsets += s.avg_loads_offsets;
        self.sum_avg_loads_kinds += s.avg_loads_kinds;
        self.sum_avg_loads_ribbon += s.avg_loads_ribbon;
        self.sum_avg_steps_per_hit += s.avg_steps_per_hit;
        self.sum_avg_steps_per_miss += s.avg_steps_per_miss;
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

    pub(super) fn print_summary(&self) {
        if self.frame_count == 0 {
            return;
        }
        let n = self.frame_count as f64;
        let submitted_done_avg = if self.submitted_done_count > 0 {
            self.sum_submitted_done / self.submitted_done_count as f64
        } else {
            0.0
        };
        let total_avg = (self.sum_update + self.sum_upload + self.sum_highlight + self.sum_render) / n;
        eprintln!(
            "render_harness_timing avg_ms update={:.3} camera_write={:.3} upload={:.3} pack={:.3} ribbon_build={:.3} tree_write={:.3} ribbon_write={:.3} bind_group_rebuild={:.3} highlight={:.3} highlight_raycast={:.3} highlight_set={:.3} render={:.3} render_texture_alloc={:.3} render_view={:.3} render_encode={:.3} render_submit={:.3} render_wait={:.3} submitted_done={:.3} submitted_done_samples={} total={:.3}",
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
            submitted_done_avg,
            self.submitted_done_count,
            total_avg,
        );
        eprintln!(
            "render_harness_worst total_ms={:.3}@frame{} submitted_done_ms={:.3}@frame{} upload_ms={:.3}@frame{}",
            self.worst_total_ms, self.worst_total_frame,
            self.worst_submitted_done_ms, self.worst_submitted_done_frame,
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
            "render_harness_shader frames={} avg_steps={:.2} worst_avg_steps={:.2}@frame{} max_steps={} hit_fraction={:.4} max_iter_fraction={:.6} avg_oob={:.2} avg_empty={:.2} avg_descend={:.2} avg_lod_terminal={:.2} avg_would_cull={:.2}",
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
            self.sum_avg_steps_would_cull / n,
        );
        let loads_tree = self.sum_avg_loads_tree / n;
        let loads_offsets = self.sum_avg_loads_offsets / n;
        let loads_kinds = self.sum_avg_loads_kinds / n;
        let loads_ribbon = self.sum_avg_loads_ribbon / n;
        let loads_total = loads_tree + loads_offsets + loads_kinds + loads_ribbon;
        eprintln!(
            "render_harness_loads frames={} avg_loads_total={:.2} tree={:.2} offsets={:.2} kinds={:.2} ribbon={:.2}",
            self.frame_count, loads_total, loads_tree, loads_offsets, loads_kinds, loads_ribbon,
        );
        eprintln!(
            "render_harness_hitmiss frames={} avg_steps_per_hit={:.2} avg_steps_per_miss={:.2}",
            self.frame_count,
            self.sum_avg_steps_per_hit / n,
            self.sum_avg_steps_per_miss / n,
        );
    }
}

pub(super) fn print_monitor_summary(monitor: &std::sync::Arc<TestMonitor>) {
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
