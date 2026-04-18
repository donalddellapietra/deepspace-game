//! Ping-pong history textures + validity tracking for TAAU.
//!
//! Two full-resolution RGBA16Float textures. Each frame the resolve
//! pass reads from `read` and writes to `write`; after submit we
//! swap, so next frame's `read` is this frame's output.
//!
//! RGBA16Float is the canonical TAA accumulation format: linear
//! values, enough precision for 10+ frames of stable blending, and
//! Apple Silicon handles it natively with no special path.
//!
//! Validity is tracked frame-to-frame by comparing frame-root
//! metadata (bfs index + kind + ribbon depth). A mismatch means the
//! ray-march is operating in a different coordinate system than last
//! frame, so reprojecting the previous history would sample garbage
//! — we signal the shader to skip the blend and seed history with the
//! current-frame color instead.

/// Format of both history textures. HDR-linear so the resolve pass's
/// clamp math stays in linear space, matching the ray-march pipeline's
/// (linear) output. RGBA16Float is natively filterable on Apple
/// Silicon, NVIDIA, and AMD — no feature flag needed.
pub const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Pair of same-sized RGBA16F textures plus a validity flag. The
/// resolve pass reads from `read_view()` and writes into the bound
/// render attachment; on `end_frame()` we swap so next frame's read
/// is this frame's write output.
pub struct HistoryPair {
    read: wgpu::Texture,
    write: wgpu::Texture,
    read_view: wgpu::TextureView,
    write_view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// Number of frames remaining before history is considered valid.
    /// Freshly-allocated history, frame-root changes, and explicit
    /// invalidations bump this to `WARMUP_FRAMES`; each resolve call
    /// decrements it. While > 0 the shader skips the reprojection
    /// blend and just writes the current-frame color into the new
    /// history texture (seeding it).
    warmup_countdown: u8,
}

/// Frames of forced history-skip after an invalidation. One frame is
/// enough for pure camera-jump cases, but two gives the history a
/// full round-trip to stabilize across a frame-root change.
const WARMUP_FRAMES: u8 = 2;

impl HistoryPair {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (read, read_view) = make_history_texture(device, width, height, "history_a");
        let (write, write_view) = make_history_texture(device, width, height, "history_b");
        Self {
            read,
            write,
            read_view,
            write_view,
            width,
            height,
            warmup_countdown: WARMUP_FRAMES,
        }
    }

    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }

    pub fn read_view(&self) -> &wgpu::TextureView { &self.read_view }
    pub fn write_view(&self) -> &wgpu::TextureView { &self.write_view }

    /// True when the previous frame's history is usable for blending.
    /// When false the resolve shader must seed history with the current
    /// color instead of reprojecting.
    pub fn is_valid(&self) -> bool { self.warmup_countdown == 0 }

    /// Mark history invalid for the next few frames. Cheap; safe to
    /// call on any frame where the world coordinate system shifted
    /// (zoom, ribbon pop, teleport, `set_root_kind_*`).
    pub fn invalidate(&mut self) {
        self.warmup_countdown = WARMUP_FRAMES;
    }

    /// Swap read/write textures and tick the warmup counter. Called
    /// after the resolve pass has been recorded into the encoder; the
    /// actual swap happens on the CPU side so next frame's resolve
    /// reads from what we just wrote.
    pub fn end_frame(&mut self) {
        std::mem::swap(&mut self.read, &mut self.write);
        std::mem::swap(&mut self.read_view, &mut self.write_view);
        if self.warmup_countdown > 0 {
            self.warmup_countdown -= 1;
        }
    }

    /// Reallocate both textures to a new size. Also invalidates
    /// history (the new textures have undefined contents).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height { return; }
        let (a, av) = make_history_texture(device, width, height, "history_a");
        let (b, bv) = make_history_texture(device, width, height, "history_b");
        self.read = a;
        self.read_view = av;
        self.write = b;
        self.write_view = bv;
        self.width = width;
        self.height = height;
        self.invalidate();
    }
}

fn make_history_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &'static str,
) -> (wgpu::Texture, wgpu::TextureView) {
    // Contents are undefined at allocation; `warmup_countdown` gates
    // reads until the shader has written at least one real frame
    // into this texture, so no init pass is needed.
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HISTORY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
