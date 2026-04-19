// Shared struct definitions, bind-group bindings, and core result
// types used across the ray-march shader modules.

struct Camera {
    pos: vec3<f32>,
    /// Sub-pixel jitter offset in scaled-res texel units. Zero when
    /// TAAU is off — the ray-march shader adds it to the NDC offset
    /// so each frame samples a different sub-pixel position within
    /// each output pixel, enabling temporal supersampling.
    jitter_x_px: f32,
    forward: vec3<f32>,
    jitter_y_px: f32,
    right: vec3<f32>,
    _pad2: f32,
    up: vec3<f32>,
    fov: f32,
}

struct Palette {
    colors: array<vec4<f32>, 256>,
}

struct Uniforms {
    root_index: u32,
    node_count: u32,
    screen_width: f32,
    screen_height: f32,
    max_depth: u32,
    highlight_active: u32,
    /// 0 = Cartesian, 1 = body root, 2 = face-space root.
    root_kind: u32,
    /// Number of ancestor ribbon entries available. When the ray
    /// exits the frame's [0, 3)³ bubble at depth 0, the shader
    /// pops upward, walking ribbon[0]..ribbon[ribbon_count-1].
    /// 0 = no ancestors (frame is at world root).
    ribbon_count: u32,
    highlight_min: vec4<f32>,
    highlight_max: vec4<f32>,
    /// xy = (inner_r, outer_r) in body cell's local [0, 1) frame.
    /// Used when root_kind == 1 or 2.
    root_radii: vec4<f32>,
    /// x = face id, y = how many generic UVR pops remain before the
    /// next pop crosses from face root to body.
    root_face_meta: vec4<u32>,
    /// Current face-frame cell bounds inside the full face:
    /// (u_lo, v_lo, r_lo, size) in normalized [0, 1]^3.
    root_face_bounds: vec4<f32>,
    root_face_pop_pos: vec4<f32>,
}

const ROOT_KIND_CARTESIAN: u32 = 0u;
const ROOT_KIND_BODY: u32 = 1u;
const ROOT_KIND_FACE: u32 = 2u;

/// One entry in the ancestor ribbon. `node_idx` is the buffer
/// index of the ancestor's node. `slot_bits` packs:
/// - low 5 bits: slot (0..27) of the child we're popping FROM
/// - bit 31: `siblings_all_empty` — when set, every other slot of
///   `node_idx` has tag=0 (Empty). The shader uses this to fast-
///   exit the whole shell with a single ray–box intersection,
///   bypassing the DDA across ~3–5 empty cells per shell that
///   otherwise compounds linearly with ribbon depth.
struct RibbonEntry {
    node_idx: u32,
    slot_bits: u32,
}

const RIBBON_SLOT_MASK: u32 = 0x1Fu;
const RIBBON_SIBLINGS_ALL_EMPTY: u32 = 0x80000000u;

struct NodeKindGpu {
    kind: u32,        // 0=Cartesian, 1=CubedSphereBody, 2=CubedSphereFace
    face: u32,
    inner_r: f32,
    outer_r: f32,
}

/// Per-frame shader-side counters. Reset to zero each frame by the
/// renderer via `encoder.clear_buffer`, then atomically accumulated
/// across all fragment invocations. Layout is fixed: the CPU reads
/// back these u32s in this exact order. `sum_steps_div4` divides by
/// 4 to keep the sum under u32::MAX at 1920x1080 (2.07M pixels *
/// 2048 cap = 4.24G, divided by 4 = 1.06G).
struct ShaderStats {
    ray_count: atomic<u32>,
    hit_count: atomic<u32>,
    miss_count: atomic<u32>,
    max_iter_count: atomic<u32>,
    sum_steps_div4: atomic<u32>,
    max_steps: atomic<u32>,
    // Per-branch step sums (div-4 to fit u32 at high res). Each
    // inner-DDA iteration lands in exactly one branch, so
    // sum_steps == sum_oob + sum_empty + sum_node + sum_lod_terminal.
    sum_steps_oob_div4: atomic<u32>,
    sum_steps_empty_div4: atomic<u32>,
    sum_steps_node_descend_div4: atomic<u32>,
    sum_steps_lod_terminal_div4: atomic<u32>,
    // Instrumentation: count of descent candidates that would have
    // been culled if we tested (child_occupancy & path_mask == 0)
    // BEFORE descending. Doesn't change behaviour — shader only
    // counts, then still descends as normal.
    sum_steps_would_cull_div4: atomic<u32>,
    // Per-ray memory load counters, split by buffer. Counts every
    // storage-buffer u32 load a ray performs in the march path.
    // On Apple Silicon these are the dominant cost source — ALU
    // is cheap, but dependent tree[]/node_offsets[] loads can
    // stall hundreds of cycles on L1 miss.
    sum_loads_tree_div4: atomic<u32>,
    sum_loads_offsets_div4: atomic<u32>,
    sum_loads_kinds_div4: atomic<u32>,
    sum_loads_ribbon_div4: atomic<u32>,
    /// Steps accumulated ONLY for rays that returned hit=true.
    /// Divided by 4 to fit in u32. Combined with `hit_count`
    /// this gives avg steps per hit; subtracted from sum_steps
    /// it gives the per-miss total.
    sum_steps_hits_div4: atomic<u32>,
}

/// Interleaved sparse-tree storage. Each node occupies
/// `2 + 2*popcount(occupancy)` contiguous u32s:
///
/// ```
/// tree[base + 0] = occupancy mask (27 bits)
/// tree[base + 1] = first_child_offset (absolute u32 index into tree[])
/// tree[first_child_offset + rank*2]     = packed (tag|block_type|pad)
/// tree[first_child_offset + rank*2 + 1] = child.node_index (BFS idx,
///                                         valid when tag==2)
/// ```
///
/// Header + first child share a 64-byte cache line, so the
/// popcount→child chain hits L1 on the second load.
@group(0) @binding(0) var<storage, read> tree: array<u32>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> palette: Palette;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> node_kinds: array<NodeKindGpu>;
@group(0) @binding(5) var<storage, read> ribbon: array<RibbonEntry>;
@group(0) @binding(6) var<storage, read_write> shader_stats: ShaderStats;
/// BFS index → header u32-offset in `tree[]`. Cold path only
/// (touched on descent and ribbon pops). The inner DDA loop never
/// reads this buffer.
@group(0) @binding(7) var<storage, read> node_offsets: array<u32>;

/// Per-fragment-thread counter; each DDA inner-loop iteration
/// increments it. Emitted to `shader_stats` at the end of fs_main.
var<private> ray_steps: u32 = 0u;
/// Per-fragment-thread counters for each branch of march_cartesian's
/// inner loop. At the end of fs_main they get atomicAdd'd into the
/// ShaderStats buffer. Off when ENABLE_STATS is false (compiled out).
var<private> ray_steps_oob: u32 = 0u;
var<private> ray_steps_empty: u32 = 0u;
var<private> ray_steps_node_descend: u32 = 0u;
var<private> ray_steps_lod_terminal: u32 = 0u;
var<private> ray_steps_would_cull: u32 = 0u;
var<private> ray_loads_tree: u32 = 0u;
var<private> ray_loads_offsets: u32 = 0u;
var<private> ray_loads_kinds: u32 = 0u;
var<private> ray_loads_ribbon: u32 = 0u;

/// Pipeline-override constant: when false, fs_main skips all
/// atomic writes to shader_stats and DDA loops skip the `ray_steps`
/// increment. Compile-time folded, so off-state has zero runtime
/// cost. Default off; the harness enables it via `--shader-stats`.
override ENABLE_STATS: bool = false;

/// Pipeline-override constant: Nyquist pixel floor. Acts as a
/// minimum — a Node child is treated as a LOD terminal when its
/// projected screen size is below this many pixels
/// (`cell_size / ray_dist * screen_height / (2 tan(fov/2))
///  < LOD_PIXEL_THRESHOLD`). Default 1.0 = classic sub-pixel
/// rejection. This is the ONLY visual LOD gate; the earlier
/// ribbon-shell budget was removed once Nyquist + empty-subtree
/// fast-path proved sufficient on their own.
override LOD_PIXEL_THRESHOLD: f32 = 1.0;

const MAX_FACE_DEPTH: u32 = 63u;

/// Cartesian DDA stack depth — the hard descent ceiling.
///
/// Sized to the Nyquist-limited descent depth, NOT tree depth.
/// When a ray ribbon-pops to an ancestor frame, the fresh
/// `march_cartesian` starts with `depth=0` and must descend
/// until `cell_size/ray_dist < LOD_PIXEL_THRESHOLD`. The number
/// of levels that takes is:
///
///     ceil(log₃(S_root · H / (2·tan(fov/2) · d_min · τ))) + 1
///
/// For 2560×1440 fov≈70° at `d_min ≈ 4.85` world units (the
/// fractal-presets canonical spawn) this comes out to 7. We
/// round up to 8 for headroom at closer zooms. Independent of
/// `tree_depth` — Nyquist prunes before the tree does, so a
/// `plain_layers = 40` world still only needs ~7 levels to
/// reach its effective visible horizon.
///
/// Per-invocation register cost scales linearly. At 8 the 5
/// per-fragment DDA stack arrays (~1 KB total) are just at the
/// Apple Silicon register-file boundary; larger values spill to
/// threadgroup memory, adding memory latency to every DDA
/// iteration. See `docs/testing/perf-lod-diagnosis.md`.
const MAX_STACK_DEPTH: u32 = 8u;

struct HitResult {
    hit: bool,
    color: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    /// Which ancestor-pop level the hit happened in. 0 = original
    /// camera frame; >0 = popped that many times into ancestors.
    frame_level: u32,
    frame_scale: f32,
    cell_min: vec3<f32>,
    cell_size: f32,
}
