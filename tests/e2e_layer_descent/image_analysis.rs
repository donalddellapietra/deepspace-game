//! Generic image-analysis utilities for the visual harness.
//!
//! These helpers operate on PNGs written by the `--screenshot` flag.
//! They are sphere-AGNOSTIC: anything specialized to a particular
//! production geometry (cubed-sphere face math, EA → cube remap, etc.)
//! does NOT belong here.
//!
//! Functions intentionally avoid heavy dependencies — they parse PNGs
//! with the existing `png` crate (already in dev-dependencies via the
//! visibility tests) and compute scalar metrics suitable for assertion.
//!
//! ## Use cases
//!
//! * **Silhouette area / fraction** — what fraction of the frame is
//!   "solid" (non-sky)? Used by zoom-out / altitude sweeps to assert
//!   that the planet stays roughly the same apparent size as the
//!   curvature parameter changes.
//! * **Per-row pixel counts** — for each row, how many pixels are
//!   solid? The profile of a *flat* slab vs a *curved* horizon is
//!   geometrically distinct: a flat patch viewed edge-on has roughly
//!   uniform width across rows; a sphere has a strictly concave
//!   profile (max width at the centre row, zero at the poles).
//! * **Silhouette bounding box** — first/last row and column with any
//!   solid pixel. The box's aspect ratio is a quick "is it round?"
//!   metric.
//! * **Edge-on horizon circularity** — sweep the top edge of the
//!   silhouette and fit a circle; report the residual. Used by the
//!   Phase-3e "horizon must be a circle from orbit" gate.
//! * **Altitude sweep helper** — render N screenshots at evenly-spaced
//!   altitudes and apply a per-frame analyzer. Lets a single test
//!   express "sweep cam_y from h0 to h1 and assert frame[i].metric
//!   monotonically does X".
//!
//! All numeric outputs are `f32`/`f64` so callers can use ordinary
//! `assert!` with a tolerance literal — no special PartialEq dance.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Decoded RGB(A) image. RGBA is stored as four channels per pixel;
/// RGB is stored as three. The accessor helpers normalize so callers
/// see RGB only.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

impl DecodedImage {
    /// Read pixel `(x, y)` as `[r, g, b]`. Panics on out-of-bounds.
    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        debug_assert!(x < self.width && y < self.height);
        let i = (y * self.width + x) * self.channels;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

/// Decode a PNG file from `path`. Supports `Rgb` and `Rgba` color
/// types — these are the two the harness writes. Anything else
/// panics with a descriptive message.
pub fn load_png(path: impl AsRef<Path>) -> DecodedImage {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read png header");
    let info = reader.info().clone();
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unsupported png color type {other:?} at {}", path.display()),
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).expect("decode png frame");
    buf.truncate(frame.buffer_size());
    DecodedImage {
        width: info.width as usize,
        height: info.height as usize,
        channels,
        data: buf,
    }
}

/// Default sky predicate used across the engine: a pixel is sky when
/// its blue channel is *strictly* greater than both red and green.
/// The sky gradient renders ~`(162, 196, 229)` (R<G<B), so empty sky
/// always passes; grass `(205, 225, 177)` and warm-white stars
/// `(255, 240, 200)` both fail. Same predicate as
/// `harness::sky_dominance_top_half` and the legacy
/// `sphere_zoom_seamless::planet_fraction`.
#[inline]
pub fn is_sky_pixel(r: u8, g: u8, b: u8) -> bool {
    b > r && b > g
}

/// Fraction of pixels in `img` that are NOT sky. Mirror of the
/// legacy `planet_fraction` from the (skipped) sphere zoom test, but
/// generic over any solid geometry — a Cartesian cube floating in
/// space, a flat slab, the upcoming wrapped-Cartesian planet — anything
/// that renders against the sky-blue background.
pub fn solid_fraction(img: &DecodedImage) -> f32 {
    let mut solid = 0usize;
    for y in 0..img.height {
        for x in 0..img.width {
            let [r, g, b] = img.rgb(x, y);
            if !is_sky_pixel(r, g, b) {
                solid += 1;
            }
        }
    }
    let total = img.width * img.height;
    if total == 0 { 0.0 } else { solid as f32 / total as f32 }
}

/// Convenience wrapper: load + measure in one call.
pub fn solid_fraction_path(path: impl AsRef<Path>) -> f32 {
    solid_fraction(&load_png(path))
}

/// Per-row count of solid (non-sky) pixels. `result[y]` = number of
/// solid pixels in scanline `y`. Length == `img.height`. Used to
/// distinguish a flat slab profile (uniform across rows) from a
/// curved horizon (strictly concave).
pub fn solid_pixels_per_row(img: &DecodedImage) -> Vec<u32> {
    let mut rows = vec![0u32; img.height];
    for y in 0..img.height {
        let mut count = 0u32;
        for x in 0..img.width {
            let [r, g, b] = img.rgb(x, y);
            if !is_sky_pixel(r, g, b) {
                count += 1;
            }
        }
        rows[y] = count;
    }
    rows
}

/// Per-column count of solid (non-sky) pixels. Symmetric to
/// `solid_pixels_per_row`. A vertical wrap-test silhouette (the slab
/// looped around the X-axis on a planet) should have non-zero counts
/// uniformly across columns; a single pole-cap rendering as a single
/// blob should have a clear column-Gaussian.
pub fn solid_pixels_per_col(img: &DecodedImage) -> Vec<u32> {
    let mut cols = vec![0u32; img.width];
    for x in 0..img.width {
        let mut count = 0u32;
        for y in 0..img.height {
            let [r, g, b] = img.rgb(x, y);
            if !is_sky_pixel(r, g, b) {
                count += 1;
            }
        }
        cols[x] = count;
    }
    cols
}

/// Inclusive bounding box of all solid pixels: `(x_min, y_min,
/// x_max, y_max)`. Returns `None` for a pure-sky frame.
pub fn solid_bounding_box(img: &DecodedImage) -> Option<(usize, usize, usize, usize)> {
    let mut x_min = usize::MAX;
    let mut y_min = usize::MAX;
    let mut x_max = 0usize;
    let mut y_max = 0usize;
    let mut any = false;
    for y in 0..img.height {
        for x in 0..img.width {
            let [r, g, b] = img.rgb(x, y);
            if !is_sky_pixel(r, g, b) {
                any = true;
                if x < x_min { x_min = x; }
                if y < y_min { y_min = y; }
                if x > x_max { x_max = x; }
                if y > y_max { y_max = y; }
            }
        }
    }
    if any { Some((x_min, y_min, x_max, y_max)) } else { None }
}

/// Aspect ratio of the silhouette bounding box (`width / height`).
/// A perfectly round disk renders `1.0`; a wide flat slab renders
/// `>> 1`. Returns `None` for a pure-sky frame.
pub fn solid_aspect_ratio(img: &DecodedImage) -> Option<f32> {
    let (x0, y0, x1, y1) = solid_bounding_box(img)?;
    let w = (x1 - x0 + 1) as f32;
    let h = (y1 - y0 + 1) as f32;
    if h == 0.0 { None } else { Some(w / h) }
}

/// Sky→solid edge for each column: the smallest `y` where the column
/// transitions from sky (above) to solid (below). For columns that
/// are pure sky, the entry is `None`. This is the silhouette's *top
/// edge* — for a planet from orbit, it's the visible horizon arc;
/// for an edge-on flat slab, it's a horizontal line.
pub fn silhouette_top_edge(img: &DecodedImage) -> Vec<Option<usize>> {
    let mut top = vec![None; img.width];
    for x in 0..img.width {
        for y in 0..img.height {
            let [r, g, b] = img.rgb(x, y);
            if !is_sky_pixel(r, g, b) {
                top[x] = Some(y);
                break;
            }
        }
    }
    top
}

/// Score how flat-vs-curved the silhouette top edge is.
///
/// For each column with a defined top-edge `y(x)`, fit the best
/// straight line through all defined points and return the mean
/// absolute residual normalized by image height. A perfectly flat
/// (slab) silhouette returns `~0`. A perfect circular arc bulging
/// upward returns a positive value proportional to the curvature.
///
/// This is the metric the Phase-3e horizon test wants: from orbit,
/// the planet horizon must be a circle, so the residual against a
/// straight-line fit must be NON-trivial — assert `> some_threshold`.
/// At the surface, the horizon should be ~flat (very close to a
/// horizontal cap), so the residual should be small.
///
/// Returns `None` if fewer than 3 columns have a defined top edge.
pub fn top_edge_curvature_residual(img: &DecodedImage) -> Option<f32> {
    let top = silhouette_top_edge(img);
    let pts: Vec<(f32, f32)> = top
        .iter()
        .enumerate()
        .filter_map(|(x, y_opt)| y_opt.map(|y| (x as f32, y as f32)))
        .collect();
    if pts.len() < 3 { return None; }
    // Fit y = a*x + b (least squares).
    let n = pts.len() as f32;
    let sx: f32 = pts.iter().map(|p| p.0).sum();
    let sy: f32 = pts.iter().map(|p| p.1).sum();
    let sxx: f32 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f32 = pts.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-6 { return None; }
    let a = (n * sxy - sx * sy) / denom;
    let b = (sy - a * sx) / n;
    // Mean absolute residual normalized by image height.
    let mar: f32 = pts
        .iter()
        .map(|(x, y)| (y - (a * x + b)).abs())
        .sum::<f32>() / n;
    Some(mar / img.height.max(1) as f32)
}

/// Fit a circle to the silhouette top edge using a simple algebraic
/// least squares (Kasa's method). Returns `(cx, cy, r,
/// mean_abs_residual_px)`. The residual is in pixels — for a true
/// circular horizon, this should be small (< 1 % of image diagonal);
/// for a flat slab silhouette, the "best fit circle" has a huge
/// residual because no circle fits a line well at constrained scale.
///
/// Returns `None` if fewer than 3 columns have a defined top edge.
///
/// Used by the Phase-3e edge-on horizon test: at orbital altitude,
/// fit_circle_to_top_edge(img).residual / img_diag < epsilon.
pub fn fit_circle_to_top_edge(img: &DecodedImage) -> Option<(f64, f64, f64, f64)> {
    let top = silhouette_top_edge(img);
    let pts: Vec<(f64, f64)> = top
        .iter()
        .enumerate()
        .filter_map(|(x, y_opt)| y_opt.map(|y| (x as f64, y as f64)))
        .collect();
    if pts.len() < 3 { return None; }
    // Kasa's method: minimize Σ((x²+y²) − 2ax − 2by − c)² where
    // r² = a² + b² + c. Solve the 3×3 normal equations for (a, b, c).
    let n = pts.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut sxxx = 0.0;
    let mut syyy = 0.0;
    let mut sxyy = 0.0;
    let mut sxxy = 0.0;
    for (x, y) in &pts {
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sxxx += x * x * x;
        syyy += y * y * y;
        sxyy += x * y * y;
        sxxy += x * x * y;
    }
    // 2A·a + 2B·b + C·c = D, etc. — written as a 3x3 linear system.
    let m = [
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx,  sy,  n ],
    ];
    let rhs = [
        0.5 * (sxxx + sxyy),
        0.5 * (sxxy + syyy),
        0.5 * (sxx + syy),
    ];
    let (a, b, c) = solve_3x3(m, rhs)?;
    let r_sq = a * a + b * b + c;
    if r_sq < 0.0 { return None; }
    let r = r_sq.sqrt();
    let mar: f64 = pts
        .iter()
        .map(|(x, y)| {
            let dx = x - a;
            let dy = y - b;
            ((dx * dx + dy * dy).sqrt() - r).abs()
        })
        .sum::<f64>() / n;
    Some((a, b, r, mar))
}

/// Solve a 3x3 linear system `m * x = rhs` via Cramer's rule.
/// Returns `None` for a singular matrix.
fn solve_3x3(m: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<(f64, f64, f64)> {
    let det = det3(m);
    if det.abs() < 1e-12 { return None; }
    let mut mx = m;
    let mut my = m;
    let mut mz = m;
    for i in 0..3 {
        mx[i][0] = rhs[i];
        my[i][1] = rhs[i];
        mz[i][2] = rhs[i];
    }
    Some((det3(mx) / det, det3(my) / det, det3(mz) / det))
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Pixel-difference summary between two equally-sized RGB(A) images.
/// `changed_frac` is the fraction of pixels with any channel
/// differing; `bbox` is the bounding box of those pixels (or `None`
/// if identical). Useful for "did anything render differently
/// post-action" assertions in altitude sweeps and break/place edits.
#[derive(Debug, Clone)]
pub struct ImageDelta {
    pub changed_frac: f64,
    pub bbox: Option<(usize, usize, usize, usize)>,
    pub max_channel_diff: u8,
}

pub fn image_delta(a: &DecodedImage, b: &DecodedImage) -> ImageDelta {
    assert_eq!(a.width, b.width, "width mismatch in image_delta");
    assert_eq!(a.height, b.height, "height mismatch in image_delta");
    let mut changed = 0usize;
    let mut max_ch = 0u8;
    let mut x_min = usize::MAX;
    let mut y_min = usize::MAX;
    let mut x_max = 0usize;
    let mut y_max = 0usize;
    let mut any = false;
    for y in 0..a.height {
        for x in 0..a.width {
            let pa = a.rgb(x, y);
            let pb = b.rgb(x, y);
            let dr = pa[0].abs_diff(pb[0]);
            let dg = pa[1].abs_diff(pb[1]);
            let db = pa[2].abs_diff(pb[2]);
            if dr | dg | db != 0 {
                changed += 1;
                max_ch = max_ch.max(dr).max(dg).max(db);
                if x < x_min { x_min = x; }
                if y < y_min { y_min = y; }
                if x > x_max { x_max = x; }
                if y > y_max { y_max = y; }
                any = true;
            }
        }
    }
    let total = a.width * a.height;
    ImageDelta {
        changed_frac: if total == 0 { 0.0 } else { changed as f64 / total as f64 },
        bbox: if any { Some((x_min, y_min, x_max, y_max)) } else { None },
        max_channel_diff: max_ch,
    }
}

/// Description of an evenly-spaced altitude sweep along the
/// world-y axis. Camera looks straight down (`pitch = -π/2`).
/// `xy = (x, z)` is the lateral spawn position (root-cell-local,
/// 0..3). `y_lo`/`y_hi` are the world-y endpoints; `steps` is the
/// number of screenshots, inclusive on both ends (`steps >= 2`).
#[derive(Debug, Clone)]
pub struct AltitudeSweep {
    pub xz: (f32, f32),
    pub y_lo: f32,
    pub y_hi: f32,
    pub steps: u32,
    pub spawn_depth: u32,
    pub width: u32,
    pub height: u32,
    /// Frames to wait per step before screenshotting (lets the GPU
    /// settle after the spawn override applies). `60` is a safe
    /// default at 60 fps; bump for slower scenes.
    pub settle_frames: u32,
    /// Hard timeout per step in seconds. Applies to the per-process
    /// `--timeout-secs` flag so a hung step doesn't stall the suite.
    pub timeout_secs: u32,
    /// Pitch override (radians). `None` defaults to looking straight
    /// down (`-π/2`).
    pub pitch_override: Option<f32>,
    /// Yaw override (radians). `None` defaults to `0`.
    pub yaw_override: Option<f32>,
}

impl AltitudeSweep {
    /// World-y altitude for step `i` in `0..self.steps`.
    pub fn altitude_at(&self, i: u32) -> f32 {
        if self.steps <= 1 { return self.y_lo; }
        let t = i as f32 / (self.steps - 1) as f32;
        self.y_lo + t * (self.y_hi - self.y_lo)
    }

    /// Build the harness CLI args for step `i`, writing the
    /// screenshot to `out_png`.
    pub fn args_for_step(&self, i: u32, out_png: &Path) -> Vec<String> {
        let cam_y = self.altitude_at(i);
        let pitch = self.pitch_override.unwrap_or(-std::f32::consts::FRAC_PI_2);
        let yaw = self.yaw_override.unwrap_or(0.0);
        vec![
            "--render-harness".to_string(),
            "--spawn-depth".to_string(), self.spawn_depth.to_string(),
            "--spawn-xyz".to_string(),
                format!("{:.6}", self.xz.0),
                format!("{:.6}", cam_y),
                format!("{:.6}", self.xz.1),
            "--spawn-pitch".to_string(), format!("{pitch:.6}"),
            "--spawn-yaw".to_string(), format!("{yaw:.6}"),
            "--harness-width".to_string(), self.width.to_string(),
            "--harness-height".to_string(), self.height.to_string(),
            "--exit-after-frames".to_string(), (self.settle_frames + 1).to_string(),
            "--timeout-secs".to_string(), self.timeout_secs.to_string(),
            "--suppress-startup-logs".to_string(),
            "--screenshot".to_string(), out_png.to_string_lossy().into_owned(),
        ]
    }
}

/// Build a per-step screenshot path under `dir`, named `step_NN.png`.
pub fn altitude_step_png(dir: &Path, i: u32) -> PathBuf {
    dir.join(format!("step_{i:02}.png"))
}
