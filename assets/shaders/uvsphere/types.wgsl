// UV-sphere shared types and constants.
//
// Geometry: a body has spherical coords `(φ, θ, r)` with ranges
//   - `φ ∈ [0, 2π)`           (azimuth, wraps)
//   - `θ ∈ [-θ_cap, +θ_cap]`  (latitude, capped to avoid pole singularity)
//   - `r ∈ [inner_r, outer_r]` (radial shell)
// Each tree node has 27 children indexed by `slot = pt + tt*3 + rt*9`
// where `(pt, tt, rt) ∈ {0, 1, 2}³` are the per-axis tiers.

const UV_TWO_PI: f32 = 6.2831853;

// Hardware safety on the descent loop. The walker is data-driven —
// it terminates the moment it hits an empty slot, a Block leaf, or
// an EntityRef — so the cap only matters for fully-occupied chains.
const UV_MAX_DEPTH: u32 = 63u;

// Per-ray iteration cap on the marcher. Each iteration crosses one
// cell boundary; in flat-shell gameplay rays terminate on the
// surface in a few dozen iterations.
const UV_MAX_ITER: u32 = 256u;

// Outcome of descending from a frame to the deepest cell containing
// a point in `(φ, θ, r)` parameter-space. Carries absolute bounds —
// the marcher uses these to step the ray to the next cell-face
// crossing in body-frame world coords (no cell-local-fraction
// amplification of f32 ULPs).
struct UvDescend {
    found_block: bool,
    block_type: u32,
    /// BFS idx of a `CartesianTangent` Node the descent stopped at.
    /// 0 = no tangent dispatch needed. When non-zero, the caller
    /// must transform the ray into the cell's tangent-frame OBB and
    /// hand off to `march_entity_subtree(tangent_node_idx, ...)`.
    /// `found_block` is false when this is set.
    tangent_node_idx: u32,
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
