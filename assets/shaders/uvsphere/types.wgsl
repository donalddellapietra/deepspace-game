// UV-sphere shared types and constants.
//
// Geometry: a body has spherical coords `(φ, θ, r)` with ranges
//   - `φ ∈ [0, 2π)`           (azimuth, wraps)
//   - `θ ∈ [-θ_cap, +θ_cap]`  (latitude, capped to avoid pole singularity)
//   - `r ∈ [inner_r, outer_r]` (radial shell)
// Each tree node has 27 children indexed by `slot = pt + tt*3 + rt*9`
// where `(pt, tt, rt) ∈ {0, 1, 2}³` are the per-axis tiers.

const UV_TWO_PI: f32 = 6.2831853;

// Hardware safety on the descent loop. The walker is pure data-driven —
// it terminates the moment it hits an empty slot, a Block leaf, or an
// EntityRef — so the cap only matters for fully-occupied chains, which
// the tree never produces in practice.
const UV_MAX_DEPTH: u32 = 63u;

// Per-ray iteration cap on the marcher. Each iteration crosses one
// cell boundary; in flat-shell gameplay rays terminate on the surface
// in a few dozen iterations.
const UV_MAX_ITER: u32 = 256u;

// Per-iteration cap on cell-local descent inside the sub-cell marcher.
//
// The cell-local fraction `un_*` is computed from the camera's
// `(φ, θ, r)` once per ray-march iteration with f32 precision
// `~1e-7 / frame_dphi` of frame `[0, 1]`. Each `un *= 3 - tier` step
// during descent multiplies that error by 3. Past 4 levels of
// in-frame descent the accumulated error consumes a leaf cell, the
// tier picked by `floor(un * 3)` becomes effectively random, and the
// rendered bevels start swarming as the camera moves sub-pixel.
//
// Capping descent here trades vertical resolution for stability:
// the deepest visual cells stop at `frame_depth + UV_SUBCELL_DESCENT`
// (typically 8 + 4 = 12). Going deeper requires propagating
// cell-local coords through a stack-based DDA — never deriving `un`
// from absolute body-frame `phi_w`. That's a future iteration; this
// rewrite stops at the precision cliff instead of falling off it.
const UV_SUBCELL_DESCENT: u32 = 4u;

// Outcome of descending from a frame to the deepest cell containing
// a point in `(φ, θ, r)` parameter-space. Returns ABSOLUTE bounds
// (so the marcher can step to the next cell boundary in body-frame
// coords without `3^K` cell-local amplification — see `cell.wgsl`)
// and the cell-axis sizes at the resolved depth.
struct UvDescend {
    found_block: bool,
    block_type: u32,
    dphi: f32,
    dth: f32,
    dr: f32,
    phi_lo: f32,
    phi_hi: f32,
    theta_lo: f32,
    theta_hi: f32,
    r_lo: f32,
    r_hi: f32,
    depth: u32,
}

// Hit-face descriptor. `axis` tags which cell-axis the ray was
// closest to crossing on hit (0=φ, 1=θ, 2=r), so callers can pick
// the in-face 2D pair for the bevel.
struct UvHitFace {
    normal: vec3<f32>,
    axis: u32,
}

// Closest-axis crossing returned by `uv_cell_step`. `t` is the
// world-distance to the next cell-local axis plane; `axis` tags it.
struct UvCellStep {
    t: f32,
    axis: u32,
}
