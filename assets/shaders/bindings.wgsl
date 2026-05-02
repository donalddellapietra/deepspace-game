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
    /// World → clip-space matrix. Only used by `fs_main_depth`
    /// (entity-raster mode), which writes `@builtin(frag_depth)`
    /// from `view_proj * world_hit` so a subsequent raster pass can
    /// z-test against the ray-march output. Ignored by `fs_main`.
    view_proj: mat4x4<f32>,
}

// Palette lives in a read-only storage buffer of dynamic length
// (see binding 2 below). The `Palette` struct was removed when the
// palette index widened from u8 → u16 so CPU-side `ColorRegistry`
// can hold up to 65 535 entries.

struct Uniforms {
    root_index: u32,
    node_count: u32,
    screen_width: f32,
    screen_height: f32,
    max_depth: u32,
    highlight_active: u32,
    /// 0 = Cartesian. Field retained for layout compatibility with
    /// the CPU-side `GpuUniforms`; only Cartesian is dispatched.
    root_kind: u32,
    /// Number of ancestor ribbon entries available. When the ray
    /// exits the frame's [0, 3)³ bubble at depth 0, the shader
    /// pops upward, walking ribbon[0]..ribbon[ribbon_count-1].
    /// 0 = no ancestors (frame is at world root).
    ribbon_count: u32,
    /// Number of live entities in `entities[]`. The shader uses
    /// this only as a validity gate in the tag=3 dispatch branch —
    /// when zero, there are no EntityRef cells in the tree either,
    /// so the branch is never taken.
    entity_count: u32,
    // Pad to the next vec4 boundary with scalar u32s. WGSL's
    // `vec3<u32>` has 16-byte alignment that would force 12 bytes
    // of skew-pad BEFORE this field, which doesn't match the CPU-
    // side `[u32; 3]` layout we mirror; scalars are 4-byte aligned
    // and land byte-for-byte on top of the Rust struct.
    _pad_entities_0: u32,
    _pad_entities_1: u32,
    _pad_entities_2: u32,
    highlight_min: vec4<f32>,
    highlight_max: vec4<f32>,
    /// Step 5: Sphere sub-frame angular range. `(lat_lo, lat_hi,
    /// lon_lo, lon_hi)` in radians. Populated when `root_kind ==
    /// ROOT_KIND_SPHERE_SUBFRAME`; zero otherwise.
    subframe_lat_lon: vec4<f32>,
    /// `WrappedPlane` slab dimensions, populated when `root_kind ==
    /// ROOT_KIND_WRAPPED_PLANE` OR `ROOT_KIND_SPHERE_SUBFRAME`.
    /// `(dims_x, dims_y, dims_z, slab_depth)`. The X-wrap branch of
    /// `march_cartesian` reads `.x` + `.w`; sphere DDAs read all
    /// four lanes for the WP geometry context.
    slab_dims: vec4<u32>,
    /// Step 5: Sphere sub-frame radial range + center. `(r_lo, r_hi,
    /// r_c, _pad)`.
    subframe_r: vec4<f32>,
    /// Step 5: Sphere sub-frame's WP metadata mirror. `(wp_dims_x,
    /// wp_dims_y, wp_dims_z, wp_slab_depth)`. Allows ribbon-pop
    /// continuity to know the outer geometry.
    subframe_wp_dims: vec4<u32>,
    /// Node range — the (lat, lon, r) extent that the GPU node
    /// pointed to by `root_index` actually covers. Distinct from
    /// `subframe_lat_lon`: the SUB-FRAME range is the camera's deep
    /// virtual target (drives basis + camera projection), the NODE
    /// range is what the dispatched node literally partitions into
    /// 27 children. They differ when the GPU tree is shallower than
    /// the camera's logical depth (e.g. above the slab — node = WP =
    /// full sphere; sub-frame = a thin patch around the camera).
    /// `(lat_lo, lat_hi, lon_lo, lon_hi)` in radians.
    node_lat_lon: vec4<f32>,
    /// Node range radial. `(r_lo, r_hi, _, _)`.
    node_r: vec4<f32>,
    /// Visual debug paint mode. 0 = off (normal rendering); 1..=8
    /// replace the shaded colour with per-pixel diagnostic colors. See
    /// `march_debug.wgsl`. Lives in `.x`; `.yzw` reserved for future
    /// per-mode tuning. Modes 7 & 8 are reserved placeholders for the
    /// wrapped-planet phases (planet-frame indicator + curvature-offset
    /// magnitude); they paint a sentinel until Phase 2 / Phase 3 wires
    /// the underlying state.
    debug_mode: vec4<u32>,
    /// `xy` = screen-space pixel to probe walker state for;
    /// `z` = non-zero means probing is active (0 disables all
    /// writes to `walker_probe`). `w` reserved.
    probe_pixel: vec4<u32>,
    /// Render-time curvature parameters. Phase 3 Step 3.0 ships
    /// the simplest form: `.x = A`, the per-step parabolic-drop
    /// coefficient. The shader applies `child_entry.y -= A * dist²`
    /// at each descent. `A = 0` (default) disables curvature
    /// entirely, leaving the marcher bit-identical to the flat path.
    /// `.yzw` reserved for k(altitude) ramp + R_inv + slab_surface_y
    /// once Step 3.4 wires those.
    curvature: vec4<f32>,
    /// Phase 3 REVISED — UV-sphere render mode for the WrappedPlane
    /// frame. `.x = 0` (default): use the flat slab DDA (current
    /// behaviour). `.x = 1`: render the slab as a sphere (analytical
    /// ray-sphere intersect + (lon, lat) → slab cell lookup, with
    /// poles banned past `.y` radians of latitude). `.zw` reserved
    /// (lat_max for poles is `.y`; later: shell inner_radius for
    /// radial-depth marching).
    planet_render: vec4<f32>,
}

const ROOT_KIND_CARTESIAN: u32 = 0u;
/// WrappedPlane root kind. Phase 1: shader treats it identically to
/// Cartesian (the marcher does not branch on root_kind). Phase 2 will
/// hook X-wrap on this kind; Phase 3 will hook curvature.
const ROOT_KIND_WRAPPED_PLANE: u32 = 1u;
/// Step 5 of the sphere sub-frame architecture: shader sees a frame
/// rooted at a Cartesian Node INSIDE a `WrappedPlane` subtree, with
/// the sub-frame's spherical bounds + WP geometry passed via the
/// `subframe_*` uniform fields.
const ROOT_KIND_SPHERE_SUBFRAME: u32 = 2u;

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
    kind: u32,        // 0 = Cartesian, 1 = WrappedPlane
    /// Slab dims (cells/axis) for WrappedPlane; zero for Cartesian.
    /// Phase 2 reads these to compute X-wrap modulus; Phase 3 reads
    /// dims_x to derive the implied planet radius. Phase 1: carried
    /// but unused.
    dims_x: u32,
    dims_y: u32,
    dims_z: u32,
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
@group(0) @binding(2) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> node_kinds: array<NodeKindGpu>;
@group(0) @binding(5) var<storage, read> ribbon: array<RibbonEntry>;
@group(0) @binding(6) var<storage, read_write> shader_stats: ShaderStats;
/// BFS index → header u32-offset in `tree[]`. Cold path only
/// (touched on descent and ribbon pops). The inner DDA loop never
/// reads this buffer.
@group(0) @binding(7) var<storage, read> node_offsets: array<u32>;

/// BFS index → per-node content AABB (12 bits packed in the low 12
/// bits of each u32). See `content_aabb` in `gpu::pack`. The shader
/// reads `aabbs[child_idx]` when deciding to descend into a tag=2
/// child; a tight AABB lets rays skip subtrees whose occupied region
/// doesn't intersect the ray path, and trims the DDA entry point to
/// the first actually-populated cell.
@group(0) @binding(9) var<storage, read> aabbs: array<u32>;

/// Flat entity list. Each entity carries the BFS idx of its voxel
/// subtree root in the shared `tree[]` buffer plus a representative
/// block for LOD-terminal splats. Ray-march's tag=3 branch reads
/// this buffer when it hits an `EntityRef(idx)` cell — the idx is
/// packed into the child entry's node_index field.
struct EntityGpu {
    bbox_min: vec3<f32>,
    representative_block: u32,
    bbox_max: vec3<f32>,
    subtree_bfs: u32,
}
@group(0) @binding(10) var<storage, read> entities: array<EntityGpu>;

/// Per-pixel walker state probe. A tiny 16-u32 buffer that
/// `march_cartesian` writes into ONLY for the pixel matching
/// `uniforms.probe_pixel.xy`. Non-atomic stores are safe because
/// only one fragment invocation passes the pixel-match gate. Used
/// for in-situ debugging of walker behavior — the CPU reads back
/// the values after render and prints them.
///
/// Slot semantics (see `WalkerProbeFrame` in
/// `src/renderer/draw.rs` for the CPU-side decode order):
///
///   [ 0] hit_flag         — 1 = wrote a hit, 0 = no write
///   [ 1] ray_steps        — DDA iteration count at hit
///   [ 2] final_depth      — terminal walker depth
///   [ 3] terminal_cell    — packed (x|y<<2|z<<4) of the terminal cell
///   [ 4] cur_node_origin_x_bits  (bitcast<u32>(f32))
///   [ 5] cur_node_origin_y_bits
///   [ 6] cur_node_origin_z_bits
///   [ 7] cur_cell_size_bits      (bitcast<u32>(f32))
///   [ 8] hit_t_bits              (bitcast<u32>(f32))
///   [ 9] hit_face                — 0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z (or 7=unknown)
///   [10] content_flag            — 1 = block hit, 0 = empty/sky
///   [11] curvature_offset_bits   — Phase 3: bitcast<u32>(Δy at hit)
///   [12..16] reserved for future per-phase fields.
struct WalkerProbe {
    hit_flag: u32,
    ray_steps: u32,
    final_depth: u32,
    terminal_cell: u32,
    cur_node_origin_x_bits: u32,
    cur_node_origin_y_bits: u32,
    cur_node_origin_z_bits: u32,
    cur_cell_size_bits: u32,
    hit_t_bits: u32,
    hit_face: u32,
    content_flag: u32,
    curvature_offset_bits: u32,
    _reserved12: u32,
    _reserved13: u32,
    _reserved14: u32,
    _reserved15: u32,
}
@group(0) @binding(11) var<storage, read_write> walker_probe: WalkerProbe;

/// Coarse beam-prepass mask. The fine fragment shader samples a 5-tap
/// neighborhood at each pixel's tile: if every tap reads 0.0, the
/// pixel is definitively sky and we return the sky color without
/// running `march()`. Populated by `fs_coarse_mask` at
/// 1/BEAM_TILE_SIZE per-axis resolution; R8Unorm render target
/// (stored as f32 in the shader). The coarse pipeline uses a 1×1
/// dummy texture at this slot (it writes to the real mask as render
/// target, which can't be simultaneously sampled).
@group(0) @binding(8) var coarse_mask: texture_2d<f32>;

/// Tile size in output pixels. Finer tiles = more coarse rays but
/// fewer false positives near silhouettes; coarser tiles = cheaper
/// prepass but more rays leak through to the fine pass.
const BEAM_TILE_SIZE: u32 = 8u;

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
/// Current pixel coordinate. `fs_main` / `fs_main_taa` / `fs_main_depth`
/// stamp this from `@builtin(position).xy` before dispatching the
/// per-pixel march, so `march_cartesian`'s probe gate (and the
/// `march_debug.wgsl` paint modes) can read it without threading the
/// builtin through every call site.
var<private> current_pixel: vec2<u32> = vec2<u32>(0u, 0u);

/// Pipeline-override constant: when false, fs_main skips all
/// atomic writes to shader_stats and DDA loops skip the `ray_steps`
/// increment. Compile-time folded, so off-state has zero runtime
/// cost. Default off; the harness enables it via `--shader-stats`.
override ENABLE_STATS: bool = false;

/// Pipeline-override constant: compiles out the tag==3 (EntityRef)
/// dispatch entirely when `false`. Fractal / sphere preset worlds
/// never produce entity cells (the scene overlay collapses to
/// `world.root`), so the branch + the `march_entity_subtree` call
/// it guards are dead at compile time and the WGSL compiler
/// eliminates them. Skipped on Jerusalem nucleus 2560x1440 this
/// recovers ~1 ms/frame vs a runtime `entity_count > 0` gate.
///
/// Default off; the renderer turns it on when App has entities in
/// ray-march mode.
override ENABLE_ENTITIES: bool = false;

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
