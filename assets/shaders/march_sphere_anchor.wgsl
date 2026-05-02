#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "march_helpers.wgsl"
#include "march_sphere_hit.wgsl"

// Hardware ceiling for the sphere anchor descent stack. Sized
// independently from `MAX_STACK_DEPTH` (the Cartesian frame stack,
// which is tuned at 8 for register pressure on Apple Silicon and
// works because Cartesian's frame-aware system pops between
// frames + LOD_PIXEL_THRESHOLD prunes descent). Sphere mode has
// neither — descent into a non-uniform anchor is one continuous
// stack — so this needs to cover any reasonable
// `cell_subtree_depth` (default 20) the user might edit into.
//
// Set to 24 to leave headroom above the 20-level default. Above
// this the descent splats representative (same shape as Cartesian's
// `at_max`). When edits exceed this, raise this constant; sphere
// mode is opt-in (--planet-render-sphere) so the extra register
// pressure only applies when sphere render is active.
const SPHERE_DESCENT_DEPTH: u32 = 24u;

// ─── Cartesian-local DDA inside an anchor block (Option A) ──────────
//
// The earlier sphere descent walked cells in spherical (lon, lat, r)
// coords. That tracked O(1) absolute coordinates with cell widths
// shrinking by 1/3 per push, so by descent depth ~12 cell widths hit
// the f32 ULP floor and bevels / cell traversal collapsed.
//
// New approach: build an orthonormal Cartesian basis at the slab
// cell's center (e_lon, e_lat, e_r tangents to the sphere) and
// transform the ray into a local frame where the slab cell occupies
// `[0, 3)³`. Inside that frame we run a standard Cartesian DDA —
// coordinates stay in `[0, 3)` at every descent depth (each push
// rescales [0, 3) → [0, 3) for the child), so f32 precision is
// preserved indefinitely. Same trick Cartesian rendering already uses
// across ribbon-pop frames; the only addition is the curved-→flat
// frame change at slab-cell entry. Sphere curvature within a sub-cell
// of width ε on a planet of radius r introduces deviation O(ε² / r)
// — for ε ≤ slab_step ≈ 0.04 and r ≈ 0.5, that's at most 1.6e-3 of
// cell extent at the slab cell, much smaller in sub-cells.
//
// Recipe — tree-structure descent (no LOD):
// * 27-children descent. Each push divides cell_size by 3.
// * tag=1 (uniform-flatten Block) → hit, terminate.
// * tag=2 (non-uniform Node) → push child frame. Coordinates rescale
//   so the child becomes the new [0, 3)³ frame.
// * tag=2 + empty representative → skip whole cell.
// * empty slot / unknown → advance ray to nearest cell face.
// * stack at SPHERE_DESCENT_DEPTH → splat representative.
//
// On exit (ray leaves the slab cell or stack underflows), returns
// hit=false so the caller's main slab DDA can continue.
//
// Bevel: standard cartesian face-edge from local in-cell position
// (the face axis matches the world-axis; bevel UVs are axis-aligned
// in local frame, and the local frame is orthonormal so bevels
// project to clean lines on the slab cell's faces). The result's
// normal is rotated back to world via the local basis.

fn sphere_descend_anchor(
    anchor_idx: u32,
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    cs_center: vec3<f32>, inv_norm: f32,
    cell_lon_center: f32, cell_lat_center: f32, cell_r_center: f32,
    cell_lon_step: f32, cell_lat_step: f32, cell_r_step: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // ── Local Cartesian frame at slab cell center ────────────────────
    let cos_lat = cos(cell_lat_center);
    let sin_lat = sin(cell_lat_center);
    let cos_lon = cos(cell_lon_center);
    let sin_lon = sin(cell_lon_center);
    // Sphere convention: lon = atan2(z, x), lat = asin(y / r).
    // Radial (e_r) points outward at the cell center; e_lon eastward
    // (∂P/∂lon, normalised), e_lat northward (∂P/∂lat, normalised).
    // The three are orthonormal.
    let e_r   = vec3<f32>(cos_lat * cos_lon, sin_lat,         cos_lat * sin_lon);
    let e_lon = vec3<f32>(-sin_lon,           0.0,             cos_lon);
    let e_lat = vec3<f32>(-sin_lat * cos_lon, cos_lat,        -sin_lat * sin_lon);
    // Slab cell extents in WORLD units along the three local axes
    // (anisotropic — lon arc length depends on cos(lat)).
    let ext_lon = cell_r_center * cos_lat * cell_lon_step;
    let ext_lat = cell_r_center * cell_lat_step;
    let ext_r   = cell_r_step;
    // Local-coord scale: slab cell occupies [0, 3) on each axis.
    let scale_lon = 3.0 / max(ext_lon, 1e-30);
    let scale_lat = 3.0 / max(ext_lat, 1e-30);
    let scale_r   = 3.0 / max(ext_r,   1e-30);
    // Slab cell's local-origin (where local = [0, 0, 0]) in world.
    let cell_corner = cs_center
        + cell_r_center * e_r
        - 0.5 * ext_lon * e_lon
        - 0.5 * ext_lat * e_lat
        - 0.5 * ext_r   * e_r;

    // Transform ray to local. Both origin (positions) and direction
    // (vectors) project onto basis with the same per-axis scale.
    let dv = ray_origin - cell_corner;
    let O_l = vec3<f32>(
        dot(dv,      e_lon) * scale_lon,
        dot(dv,      e_lat) * scale_lat,
        dot(dv,      e_r)   * scale_r,
    );
    let D_l = vec3<f32>(
        dot(ray_dir, e_lon) * scale_lon,
        dot(ray_dir, e_lat) * scale_lat,
        dot(ray_dir, e_r)   * scale_r,
    );

    // ── Cartesian DDA in local [0, 3)³ ───────────────────────────────
    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / D_l.x, abs(D_l.x) > 1e-8),
        select(1e10, 1.0 / D_l.y, abs(D_l.y) > 1e-8),
        select(1e10, 1.0 / D_l.z, abs(D_l.z) > 1e-8),
    );
    let step = vec3<i32>(
        select(-1, 1, D_l.x >= 0.0),
        select(-1, 1, D_l.y >= 0.0),
        select(-1, 1, D_l.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    // Slab-cell ray-box clip. If the local ray misses [0, 3)³ it can't
    // hit anything — bail.
    let slab_box = ray_box(O_l, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if slab_box.t_enter >= slab_box.t_exit || slab_box.t_exit < 0.0 {
        return result;
    }
    let t_start = max(slab_box.t_enter, 0.0) + 0.001;
    let entry_pos = O_l + D_l * t_start;
    let entry_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );

    var s_node_idx: array<u32, SPHERE_DESCENT_DEPTH>;
    var s_cell:     array<u32, SPHERE_DESCENT_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;
    s_cell[0] = pack_cell(entry_cell);

    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);

    let cell_f0 = vec3<f32>(entry_cell);
    var cur_side_dist = vec3<f32>(
        select((cell_f0.x       - O_l.x) * inv_dir.x,
               (cell_f0.x + 1.0 - O_l.x) * inv_dir.x, D_l.x >= 0.0),
        select((cell_f0.y       - O_l.y) * inv_dir.y,
               (cell_f0.y + 1.0 - O_l.y) * inv_dir.y, D_l.y >= 0.0),
        select((cell_f0.z       - O_l.z) * inv_dir.z,
               (cell_f0.z + 1.0 - O_l.z) * inv_dir.z, D_l.z >= 0.0),
    );
    var normal = vec3<f32>(0.0);

    let root_header_off = node_offsets[anchor_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    var iters: u32 = 0u;
    let max_iters: u32 = 4096u;
    var hit_local_t: f32 = 0.0;
    var hit_block_type: u32 = 0u;
    var hit_cell_min: vec3<f32> = vec3<f32>(0.0);
    var hit_cell_size: f32 = 1.0;
    var did_hit: bool = false;

    loop {
        if iters >= max_iters { break; }
        iters = iters + 1u;
        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            // OOB: pop, advance parent's cell on the OOB axis.
            if depth == 0u { break; }
            depth = depth - 1u;
            cur_cell_size = cur_cell_size * 3.0;
            let popped = unpack_cell(s_cell[depth]);
            cur_node_origin = cur_node_origin - vec3<f32>(popped) * cur_cell_size;
            let lc_pop = vec3<f32>(popped);
            cur_side_dist = vec3<f32>(
                select((cur_node_origin.x +  lc_pop.x        * cur_cell_size - O_l.x) * inv_dir.x,
                       (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - O_l.x) * inv_dir.x, D_l.x >= 0.0),
                select((cur_node_origin.y +  lc_pop.y        * cur_cell_size - O_l.y) * inv_dir.y,
                       (cur_node_origin.y + (lc_pop.y + 1.0) * cur_cell_size - O_l.y) * inv_dir.y, D_l.y >= 0.0),
                select((cur_node_origin.z +  lc_pop.z        * cur_cell_size - O_l.z) * inv_dir.z,
                       (cur_node_origin.z + (lc_pop.z + 1.0) * cur_cell_size - O_l.z) * inv_dir.z, D_l.z >= 0.0),
            );
            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];
            let m_oob = min_axis_mask(cur_side_dist);
            let advanced = popped + vec3<i32>(m_oob) * step;
            s_cell[depth] = pack_cell(advanced);
            cur_side_dist = cur_side_dist + m_oob * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let slot_bit = 1u << slot;

        if (cur_occupancy & slot_bit) == 0u {
            // Empty slot: step DDA along smallest side_dist.
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist = cur_side_dist + m_empty * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let child_bt = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            // Block leaf — hit at this cell.
            let cell_min_l = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let cell_max_l = cell_min_l + vec3<f32>(cur_cell_size);
            let cell_box_l = ray_box(O_l, inv_dir, cell_min_l, cell_max_l);
            hit_local_t = max(cell_box_l.t_enter, 0.0);
            hit_block_type = child_bt;
            hit_cell_min = cell_min_l;
            hit_cell_size = cur_cell_size;
            did_hit = true;
            break;
        }
        if tag != 2u {
            // EntityRef etc. — treat as miss inside anchor; advance.
            let m_skip = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
            cur_side_dist = cur_side_dist + m_skip * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_skip;
            continue;
        }
        let child_idx = tree[child_base + 1u];
        if child_bt == 0xFFFEu {
            // representative_empty — subtree contains no solid content.
            let m_rep = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_rep) * step);
            cur_side_dist = cur_side_dist + m_rep * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_rep;
            continue;
        }
        // tag=2 with non-empty representative: push child frame.
        let at_max = depth + 1u >= SPHERE_DESCENT_DEPTH;
        if at_max {
            // Splat representative at the current scale.
            let cell_min_l = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let cell_max_l = cell_min_l + vec3<f32>(cur_cell_size);
            let cell_box_l = ray_box(O_l, inv_dir, cell_min_l, cell_max_l);
            hit_local_t = max(cell_box_l.t_enter, 0.0);
            hit_block_type = child_bt;
            hit_cell_min = cell_min_l;
            hit_cell_size = cur_cell_size;
            did_hit = true;
            break;
        }
        let child_cell_size = cur_cell_size / 3.0;
        let child_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
        let ct_start = max(slab_box.t_enter, 0.0) + 0.0001 * child_cell_size;
        let child_entry = O_l + D_l * ct_start;
        let local_entry = (child_entry - child_origin) / child_cell_size;
        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_node_origin = child_origin;
        cur_cell_size = child_cell_size;
        let child_header_off = node_offsets[child_idx];
        cur_occupancy = tree[child_header_off];
        cur_first_child = tree[child_header_off + 1u];
        let child_cell_i = vec3<i32>(
            clamp(i32(floor(local_entry.x)), 0, 2),
            clamp(i32(floor(local_entry.y)), 0, 2),
            clamp(i32(floor(local_entry.z)), 0, 2),
        );
        s_cell[depth] = pack_cell(child_cell_i);
        let lc = vec3<f32>(child_cell_i);
        cur_side_dist = vec3<f32>(
            select((child_origin.x +  lc.x        * child_cell_size - O_l.x) * inv_dir.x,
                   (child_origin.x + (lc.x + 1.0) * child_cell_size - O_l.x) * inv_dir.x, D_l.x >= 0.0),
            select((child_origin.y +  lc.y        * child_cell_size - O_l.y) * inv_dir.y,
                   (child_origin.y + (lc.y + 1.0) * child_cell_size - O_l.y) * inv_dir.y, D_l.y >= 0.0),
            select((child_origin.z +  lc.z        * child_cell_size - O_l.z) * inv_dir.z,
                   (child_origin.z + (lc.z + 1.0) * child_cell_size - O_l.z) * inv_dir.z, D_l.z >= 0.0),
        );
    }

    if !did_hit { return result; }

    // ── Bevel + world-frame conversion ───────────────────────────────
    //
    // Cartesian-local bevel: local cell is axis-aligned, hit's normal
    // tells us the face axis, in-cell fractions on the OTHER two axes
    // form the UV used for face-edge falloff.
    let hit_pos_local = O_l + D_l * hit_local_t;
    let in_cell = clamp(
        (hit_pos_local - hit_cell_min) / hit_cell_size,
        vec3<f32>(0.0), vec3<f32>(1.0),
    );
    var u: f32;
    var v: f32;
    if abs(normal.x) > 0.5 {
        u = in_cell.y;
        v = in_cell.z;
    } else if abs(normal.y) > 0.5 {
        u = in_cell.x;
        v = in_cell.z;
    } else {
        u = in_cell.x;
        v = in_cell.y;
    }
    let face_edge = min(min(u, 1.0 - u), min(v, 1.0 - v));
    let shape = smoothstep(0.02, 0.14, face_edge);
    let bevel_strength = 0.7 + 0.3 * shape;

    // Convert local axis-aligned normal back to world via the local
    // basis. e_lon/e_lat/e_r are unit vectors in world Cartesian.
    let n_world = normal.x * e_lon + normal.y * e_lat + normal.z * e_r;

    // World-space hit position. The local-t and world-t are linearly
    // related (M * D_world = D_local), so `t_world = hit_local_t` at
    // the same parameterisation as ray_dir input. inv_norm scales to
    // camera-frame unit-direction t for the result struct.
    let world_hit = ray_origin + ray_dir * hit_local_t;

    var out: HitResult;
    out.hit = true;
    out.t = hit_local_t * inv_norm;
    out.color = palette[hit_block_type].rgb * bevel_strength;
    out.normal = n_world;
    out.frame_level = 0u;
    out.frame_scale = 1.0;
    // Neutralize main.wgsl::shade_pixel's cube_face_bevel — it picks a
    // cube face from result.normal and would double-darken using
    // world-axis-aligned bevels. We've already applied the bevel
    // computed in the local frame above, so set cell_min/size such
    // that local = (0.5, 0.5, 0.5) → edge=0.5 → smoothstep → 1.0.
    out.cell_min = world_hit - vec3<f32>(0.5);
    out.cell_size = 1.0;
    return out;
}
