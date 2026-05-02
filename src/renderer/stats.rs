//! Per-frame readback structs: GPU timing, shader-side ray counters,
//! and the single-pixel walker probe. CPU mirrors of the SSBOs the
//! ray-march pipeline writes each frame.

#[derive(Debug, Clone, Copy, Default)]
pub struct OffscreenRenderTiming {
    pub texture_alloc_ms: f64,
    pub view_ms: f64,
    pub encode_ms: f64,
    pub submit_ms: f64,
    pub wait_ms: f64,
    pub total_ms: f64,
    /// Time from `queue.submit` to the `on_submitted_work_done`
    /// callback firing. Authoritative GPU-bound signal on Apple
    /// Silicon: captures all GPU work including TBDR tile resolve.
    /// (A prior render-pass-boundary `gpu_pass_ms` timestamp-query
    /// metric was removed — Metal's per-pass timestamps were
    /// non-monotonic for fast passes and gave nonsense values.)
    pub submitted_done_ms: Option<f64>,
    /// Shader-side per-ray counters for the frame. Populated by
    /// atomic writes in the fragment shader, read back by copy to
    /// the `shader_stats_readback` buffer + map_async.
    pub shader_stats: ShaderStatsFrame,
}

/// Decoded `walker_probe` buffer for one frame. The shader writes ONE
/// pixel's walker state per frame (the pixel set via
/// `Renderer::set_walker_probe_pixel`). `hit_flag == 0` means the
/// probe was inactive or the gate didn't match. f32 fields are
/// reconstructed via `f32::from_bits`.
#[derive(Debug, Clone, Copy, Default)]
pub struct WalkerProbeFrame {
    pub hit_flag: u32,
    pub ray_steps: u32,
    pub final_depth: u32,
    /// Packed terminal cell coords: x | (y << 2) | (z << 4).
    pub terminal_cell: u32,
    pub cur_node_origin: [f32; 3],
    pub cur_cell_size: f32,
    pub hit_t: f32,
    pub hit_face: u32,
    pub content_flag: u32,
    /// Phase 3 reserved: bitcast<u32>(Δy at hit). Zero until wired.
    pub curvature_offset: f32,
}

/// Decoded `shader_stats` buffer for one frame. `avg_steps` is
/// computed CPU-side as `sum_steps_div4 * 4 / ray_count` (the GPU
/// side stored div-by-4 to avoid u32 overflow).
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderStatsFrame {
    pub ray_count: u32,
    pub hit_count: u32,
    pub miss_count: u32,
    pub max_iter_count: u32,
    pub sum_steps_div4: u32,
    pub max_steps: u32,
    /// Per-branch breakdown of the ray_steps total. Each step lands
    /// in exactly one branch, so `sum_steps ≈ oob + empty + descend +
    /// lod_terminal` (the terminal-hit branch returns immediately
    /// without incrementing ray_steps for that iteration).
    pub sum_steps_oob_div4: u32,
    pub sum_steps_empty_div4: u32,
    pub sum_steps_node_descend_div4: u32,
    pub sum_steps_lod_terminal_div4: u32,
    /// Instrumentation counter: how many descent candidates would
    /// have been culled by `(child_occupancy & path_mask) == 0u`
    /// if the test ran BEFORE descending. Upper-bound on savings a
    /// real path-mask cull could deliver. Does not alter traversal.
    pub sum_steps_would_cull_div4: u32,
    /// Per-ray storage-buffer u32-load counters, split by which
    /// buffer is read. On Apple Silicon these are the dominant
    /// cost source (dependent chains stall L1); ALU counting on
    /// the same shader is not representative of real frame time.
    /// Populated only when ENABLE_STATS is true.
    pub sum_loads_tree_div4: u32,
    pub sum_loads_offsets_div4: u32,
    pub sum_loads_kinds_div4: u32,
    pub sum_loads_ribbon_div4: u32,
    /// Steps accumulated over rays that RETURNED a hit. Divided
    /// by 4 on the GPU side. Use with `hit_count` to compute avg
    /// steps per hit; `sum_steps_div4 - sum_steps_hits_div4` gives
    /// the per-miss total.
    pub sum_steps_hits_div4: u32,
}

impl ShaderStatsFrame {
    pub fn avg_steps(&self) -> f64 {
        if self.ray_count == 0 {
            0.0
        } else {
            (self.sum_steps_div4 as f64 * 4.0) / self.ray_count as f64
        }
    }

    pub fn avg_steps_oob(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_oob_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_steps_empty(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_empty_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_steps_descend(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_node_descend_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_steps_lod_terminal(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_lod_terminal_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_steps_would_cull(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_steps_would_cull_div4 as f64 * 4.0) / self.ray_count as f64 }
    }

    pub fn avg_loads_tree(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_tree_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_offsets(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_offsets_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_kinds(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_kinds_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_ribbon(&self) -> f64 {
        if self.ray_count == 0 { 0.0 }
        else { (self.sum_loads_ribbon_div4 as f64 * 4.0) / self.ray_count as f64 }
    }
    pub fn avg_loads_total(&self) -> f64 {
        self.avg_loads_tree() + self.avg_loads_offsets()
            + self.avg_loads_kinds() + self.avg_loads_ribbon()
    }

    pub fn avg_steps_per_hit(&self) -> f64 {
        if self.hit_count == 0 { 0.0 }
        else { (self.sum_steps_hits_div4 as f64 * 4.0) / self.hit_count as f64 }
    }

    pub fn avg_steps_per_miss(&self) -> f64 {
        let miss_count = self.miss_count;
        if miss_count == 0 { return 0.0; }
        let miss_sum_div4 = self.sum_steps_div4.saturating_sub(self.sum_steps_hits_div4);
        (miss_sum_div4 as f64 * 4.0) / miss_count as f64
    }

    pub fn hit_fraction(&self) -> f64 {
        if self.ray_count == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.ray_count as f64
        }
    }

    pub fn max_iter_fraction(&self) -> f64 {
        if self.ray_count == 0 {
            0.0
        } else {
            self.max_iter_count as f64 / self.ray_count as f64
        }
    }
}
