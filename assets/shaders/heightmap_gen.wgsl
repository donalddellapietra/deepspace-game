// GPU heightmap generation.
//
// Per texel (u, v) in a 2^d × 2^d grid, walks the voxel tree rooted
// at `uniforms.frame_root_bfs` and writes the world-Y of the top of
// the highest solid collision cell into the heightmap texture.
//
// Collision depth = frame_depth + d. Every texel is EXACTLY one
// collision cell projected to the XZ plane (base-2 alignment), so
// there's no fractional mapping between texel coords and cell coords.
//
// Workgroup shape 9×9 (81 threads): divides every base-2 heightmap
// size (27, 81, 243, 729, 2187) with no idle threads. Well under
// the 1024-threads/workgroup limit on Apple silicon.
//
// Algorithm per texel:
//   start at frame_root. At each level, pick the highest Y slot with
//   solid content (Block, or Node whose representative != empty).
//   If it's a Block or we've reached collision_depth, return its
//   top-Y. Otherwise descend into it and repeat one level deeper.
//
// This is an O(d) tree walk per texel — total work is 2^(2d) × d
// DDA-like steps. For d=5 (243² = 59k texels) that's ~300k tree
// reads.

struct HeightmapUniforms {
    /// BFS index of the render frame's root node in `tree[]`.
    frame_root_bfs: u32,
    /// Depth of the frame root in the world tree. Documentation
    /// only — the compute doesn't consume it.
    frame_depth: u32,
    /// Heightmap resolution = 2^delta per axis.
    side: u32,
    /// Recursion depth = collision_depth - frame_depth.
    delta: u32,
    /// World-space Y origin + size of the frame-root cell.
    y_origin: f32,
    y_size: f32,
    _pad0: u32,
    _pad1: u32,
}

// Compute pipeline bind group — a subset of the render bindings:
// only tree + node_offsets are shared with the render pass; the
// uniforms and heightmap texture are compute-only.
@group(0) @binding(0) var<storage, read> tree: array<u32>;
@group(0) @binding(1) var<storage, read> node_offsets: array<u32>;
@group(0) @binding(2) var<uniform> hm_uniforms: HeightmapUniforms;
@group(0) @binding(3) var heightmap: texture_storage_2d<r32float, write>;

// Inline tree-access helpers. Kept local to this shader (rather
// than reusing `tree.wgsl`) because the render-pass bindings file
// declares 8 storage/uniform entries we don't need for compute;
// keeping compute's bind group focused avoids binding 6+ unused
// buffers per dispatch.

fn header_offset(node_idx: u32) -> u32 {
    return node_offsets[node_idx];
}

fn child_rank(occupancy: u32, slot: u32) -> u32 {
    return countOneBits(occupancy & ((1u << slot) - 1u));
}

fn child_packed(node_idx: u32, slot: u32) -> u32 {
    let h = header_offset(node_idx);
    let occupancy = tree[h];
    let bit = 1u << slot;
    if ((occupancy & bit) == 0u) {
        return 0u; // tag=0 (empty)
    }
    let first_child = tree[h + 1u];
    let rank = child_rank(occupancy, slot);
    return tree[first_child + rank * 2u];
}

fn child_node_index(node_idx: u32, slot: u32) -> u32 {
    let h = header_offset(node_idx);
    let occupancy = tree[h];
    let first_child = tree[h + 1u];
    let rank = child_rank(occupancy, slot);
    return tree[first_child + rank * 2u + 1u];
}

fn child_tag(packed: u32) -> u32 { return packed & 0xFFu; }
fn child_block_type(packed: u32) -> u32 { return (packed >> 8u) & 0xFFu; }

fn slot_from_xyz(x: u32, y: u32, z: u32) -> u32 {
    return z * 4u + y * 2u + x;
}

/// Empty-subtree sentinel. A Node whose `representative_block`
/// equals this has zero solid descendants and can be skipped.
const REP_EMPTY: u32 = 255u;

/// Sentinel written when a column has no solid content anywhere.
/// Chosen large-negative so later min/max logic in the clamp pass
/// never mistakes it for a valid ground height.
const GROUND_NONE: f32 = -1.0e30;

/// Walk the tree via DFS (iterative, explicit stack) to find the
/// top-Y of the highest solid collision cell in column (u, v).
///
/// A Node's `representative_block != empty` only tells us the
/// subtree has *some* solid content — not that the XZ column we
/// care about is solid. So we may descend into a Node, find every
/// Y slot inside empty, and need to **backtrack** to try the next
/// lower Y slot at the parent level.
///
/// Stack entry: `(node_idx, y_origin, cell_size, depth, next_y)`.
/// `next_y` is the Y slot to try on this frame's next visit.
/// `-1` means this frame is exhausted.
///
/// Max stack depth = `delta`. 16 is well clear of the `delta ≤ 6`
/// cap we ship.
fn find_column_top(u: u32, v: u32) -> f32 {
    var node_stack: array<u32, 16>;
    var y_origin_stack: array<f32, 16>;
    var cell_size_stack: array<f32, 16>;
    var next_y_stack: array<i32, 16>;
    // depth is implicit in sp — sp == depth-of-top-frame + 1.
    var sp: i32 = 0;

    // Push the root frame.
    node_stack[0] = hm_uniforms.frame_root_bfs;
    y_origin_stack[0] = hm_uniforms.y_origin;
    cell_size_stack[0] = hm_uniforms.y_size;
    next_y_stack[0] = 1;
    sp = 1;

    // Bounded loop — every iteration either descends (sp+1) or
    // advances/pops (next_y-1 or sp-1). The tree walk is O(delta),
    // this cap is just to keep the compiler comfortable.
    for (var _iter: u32 = 0u; _iter < 512u; _iter = _iter + 1u) {
        if sp <= 0 {
            return GROUND_NONE;
        }
        let top: i32 = sp - 1;
        let depth: u32 = u32(top);
        let y_slot: i32 = next_y_stack[top];
        if y_slot < 0 {
            sp = top;
            continue;
        }
        // Advance this frame's next-try to the lower Y so, if the
        // current y_slot's descent fails, we resume with y_slot - 1.
        next_y_stack[top] = y_slot - 1;

        // Extract x/z slots at this recursion level.
        let digit_idx: u32 = hm_uniforms.delta - depth - 1u;
        var divisor: u32 = 1u;
        for (var k: u32 = 0u; k < digit_idx; k = k + 1u) {
            divisor = divisor * 2u;
        }
        let x_slot: u32 = (u / divisor) % 2u;
        let z_slot: u32 = (v / divisor) % 2u;

        let slot = slot_from_xyz(x_slot, u32(y_slot), z_slot);
        let packed = child_packed(node_stack[top], slot);
        let tag = child_tag(packed);

        if tag == 0u {
            // Empty slot — fall through to the next y_slot on the
            // next iteration (we already decremented next_y_stack).
            continue;
        }

        let child_size = cell_size_stack[top] / 2.0;
        let child_y_origin = y_origin_stack[top] + f32(y_slot) * child_size;
        let child_top_y = child_y_origin + child_size;

        if tag == 1u {
            return child_top_y;
        }

        if tag == 2u {
            if child_block_type(packed) == REP_EMPTY {
                continue;
            }
            // At collision depth — the cell is the answer regardless
            // of what's inside, since we don't resolve finer than
            // one collision cell.
            if (depth + 1u) == hm_uniforms.delta {
                return child_top_y;
            }
            // Descend: push a new frame.
            let child_node = child_node_index(node_stack[top], slot);
            if sp >= 16 {
                // Stack full — shouldn't happen with delta ≤ 6;
                // fall back to the cell's top Y rather than crash.
                return child_top_y;
            }
            node_stack[sp] = child_node;
            y_origin_stack[sp] = child_y_origin;
            cell_size_stack[sp] = child_size;
            next_y_stack[sp] = 1;
            sp = sp + 1;
            continue;
        }
        // tag == 3 (EntityRef) — ignored, heightmap is terrain-only.
    }
    return GROUND_NONE;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let u = gid.x;
    let v = gid.y;
    if u >= hm_uniforms.side || v >= hm_uniforms.side {
        return;
    }
    let top = find_column_top(u, v);
    textureStore(heightmap, vec2<i32>(i32(u), i32(v)), vec4<f32>(top, 0.0, 0.0, 0.0));
}
