//! Halton(2, 3) sub-pixel jitter.
//!
//! Each frame shifts the camera projection by a deterministic
//! sub-pixel offset drawn from the Halton sequence. Over N frames
//! the jitter covers the pixel footprint uniformly, so TAAU's
//! history accumulates distinct sub-pixel samples — the core trick
//! that recovers full-res detail from a half-res render.
//!
//! Halton(2, 3) is the standard pick. 8 samples is enough coverage
//! for a 2× downscale (4 sub-pixel positions) with redundancy; longer
//! sequences help at larger scales. Symmetric around (0, 0) — we
//! subtract 0.5 so the offset is in `(-0.5, +0.5)` texels.
//!
//! Offsets are in "scaled-res texels" (i.e., the ray-march framebuffer's
//! pixel grid). The shader converts to NDC using the scaled `screen_width`
//! / `screen_height`, so the same Halton table works regardless of
//! output resolution.
//!
//! References: [Karis 2014, "High Quality Temporal Supersampling"](
//! http://advances.realtimerendering.com/s2014/).

/// Halton sequence length. Power of 2 isn't required but keeps the
/// distribution neat across a 2× downscale. 8 gives us 2 full rounds
/// of sub-pixel coverage per 2×2 footprint.
pub const JITTER_COUNT: u32 = 8;

/// Sub-pixel jitter offset for frame `index`, in scaled-res texels.
/// Range is `(-0.5, +0.5)` per axis; apply by shifting the NDC offset
/// by `jitter_px / screen_wh * 2 * half_fov_tan * aspect`.
pub fn jitter_offset(index: u32) -> [f32; 2] {
    let i = (index % JITTER_COUNT) + 1; // Halton is 1-indexed
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

/// `i`-th term of the Halton sequence with base `b`. Returns a value
/// in `[0, 1)`. This is the textbook loop — cheap enough for per-frame
/// CPU evaluation, no table needed.
fn halton(mut i: u32, b: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let bf = b as f32;
    while i > 0 {
        f /= bf;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_matches_known_prefix() {
        // Halton(2): 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, 1/16…
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        assert!((halton(4, 2) - 0.125).abs() < 1e-6);
        // Halton(3): 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, 8/9…
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
        assert!((halton(3, 3) - 1.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn jitter_in_half_pixel_range() {
        for i in 0..JITTER_COUNT * 4 {
            let [jx, jy] = jitter_offset(i);
            assert!(jx > -0.5 && jx < 0.5, "jitter x out of range: {jx}");
            assert!(jy > -0.5 && jy < 0.5, "jitter y out of range: {jy}");
        }
    }

    #[test]
    fn jitter_cycles_deterministically() {
        // Wrap-around produces the same offset.
        for i in 0..JITTER_COUNT {
            let a = jitter_offset(i);
            let b = jitter_offset(i + JITTER_COUNT);
            assert_eq!(a, b);
        }
    }
}
