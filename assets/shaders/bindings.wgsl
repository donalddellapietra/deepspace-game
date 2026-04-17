// Shared struct definitions, bind-group bindings, and core result
// types used across the ray-march shader modules.

struct Camera {
    pos: vec3<f32>,
    _pad0: f32,
    forward: vec3<f32>,
    _pad1: f32,
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
    _pad: u32,
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

/// Per-node parent pointer. Indexed by the child's BFS index — the
/// shader does `parent_info[current_idx]` on each pop instead of
/// reading a pre-built ribbon. `parent_node_idx == 0xFFFFFFFFu`
/// marks the world root (no parent — terminates the pop loop).
///
/// `slot_and_flags` packs:
/// - low 5 bits: slot (0..27) the child occupies in its parent
/// - bit 31: `siblings_all_empty` — when set, every other slot of
///   `parent_node_idx` is empty. The shader uses this to fast-exit
///   the whole shell with a single ray–box intersection, bypassing
///   the DDA across the ~3–5 empty cells per shell that otherwise
///   compound linearly with depth.
struct ParentInfo {
    parent_node_idx: u32,
    slot_and_flags: u32,
}

const PARENT_SLOT_MASK: u32 = 0x1Fu;
const PARENT_SIBLINGS_ALL_EMPTY: u32 = 0x80000000u;
const PARENT_NONE: u32 = 0xFFFFFFFFu;

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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
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
@group(0) @binding(5) var<storage, read> parent_info: array<ParentInfo>;
@group(0) @binding(6) var<storage, read_write> shader_stats: ShaderStats;
/// BFS index → header u32-offset in `tree[]`. Cold path only
/// (touched on descent and pop-up). The inner DDA loop never
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
/// rejection. This is a FLOOR: we never waste work on sub-pixel
/// detail. The CEILING is set by `BASE_DETAIL_DEPTH` below.
override LOD_PIXEL_THRESHOLD: f32 = 1.0;

/// Pipeline-override constant: detail budget inside the anchor
/// cell (pop_level=0). Each additional pop shell gets one less
/// level of detail — so a ray that's walked N ancestor shells
/// away from the camera is clamped to descend
/// `max(BASE_DETAIL_DEPTH - N, 1)` levels in its current frame.
///
/// This is the primary LOD gate. It's frame-local (uses the
/// tree's ancestor distance as the metric), so it's invariant
/// under zoom: zooming out adds one outer shell at budget=1 and
/// leaves everything else unchanged.
///
/// Default 4 gives detailed close content (4 levels under anchor)
/// while keeping far content cheap (1-level LOD terminal beyond
/// shell 3). Tune via `--lod-base-depth <N>`.
override BASE_DETAIL_DEPTH: u32 = 4u;

const MAX_FACE_DEPTH: u32 = 63u;
/// Cartesian DDA stack depth — must be ≥ `BASE_DETAIL_DEPTH + 1`.
/// 5 matches the default BASE_DETAIL_DEPTH=4 exactly. Previously 64,
/// which allocated 3.5 KB of per-fragment scratch and forced the
/// Apple Silicon register allocator to spill to local memory on
/// every DDA iteration. If you raise BASE_DETAIL_DEPTH, raise this
/// to match or the shader silently caps descent.
const MAX_STACK_DEPTH: u32 = 5u;

struct HitResult {
    hit: bool,
    color: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    /// Which ancestor-pop count the hit happened at. 0 = original
    /// camera frame; >0 = popped that many times into ancestors.
    frame_level: u32,
    frame_scale: f32,
    cell_min: vec3<f32>,
    cell_size: f32,
}
