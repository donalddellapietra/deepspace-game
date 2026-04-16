//! Shared perf counters. `TestMonitor` lives in an `Arc` so the
//! background wall-clock thread in `runner.rs` can read frame counts
//! and bail when perf thresholds are violated.

#[derive(Debug, Clone, Copy, Default)]
pub struct PerfSamples {
    pub count: u32,
    pub total_frame_secs: f64,
    pub total_cadence_secs: f64,
    pub worst_frame_secs: f64,
    pub worst_cadence_secs: f64,
}

impl PerfSamples {
    pub fn record_frame(&mut self, frame_secs: f64) {
        self.count += 1;
        self.total_frame_secs += frame_secs;
        self.worst_frame_secs = self.worst_frame_secs.max(frame_secs);
    }

    pub fn record_cadence(&mut self, cadence_secs: f64) {
        self.total_cadence_secs += cadence_secs;
        self.worst_cadence_secs = self.worst_cadence_secs.max(cadence_secs);
    }

    pub fn avg_frame_fps(&self) -> Option<f32> {
        if self.count == 0 || self.total_frame_secs <= 0.0 {
            None
        } else {
            Some((self.count as f64 / self.total_frame_secs) as f32)
        }
    }

    pub fn avg_cadence_fps(&self) -> Option<f32> {
        if self.count == 0 || self.total_cadence_secs <= 0.0 {
            None
        } else {
            Some((self.count as f64 / self.total_cadence_secs) as f32)
        }
    }
}

pub struct TestMonitor {
    pub frames_rendered: std::sync::atomic::AtomicU32,
    pub last_frame_ms: std::sync::atomic::AtomicU64,
    pub perf_failed: std::sync::atomic::AtomicBool,
    pub webview_created: std::sync::atomic::AtomicBool,
    pub perf_samples: std::sync::Mutex<PerfSamples>,
}

impl TestMonitor {
    pub fn new() -> Self {
        Self {
            frames_rendered: std::sync::atomic::AtomicU32::new(0),
            last_frame_ms: std::sync::atomic::AtomicU64::new(0),
            perf_failed: std::sync::atomic::AtomicBool::new(false),
            webview_created: std::sync::atomic::AtomicBool::new(false),
            perf_samples: std::sync::Mutex::new(PerfSamples::default()),
        }
    }

    pub fn record_frame(
        &self,
        elapsed_since_start: std::time::Duration,
        frame_secs: Option<f64>,
        cadence_secs: Option<f64>,
    ) {
        use std::sync::atomic::Ordering;

        self.frames_rendered.fetch_add(1, Ordering::Relaxed);
        self.last_frame_ms.store(
            elapsed_since_start.as_millis().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        if let Ok(mut perf) = self.perf_samples.lock() {
            if let Some(frame_secs) = frame_secs {
                perf.record_frame(frame_secs);
            }
            if let Some(cadence_secs) = cadence_secs {
                perf.record_cadence(cadence_secs);
            }
        }
    }
}
