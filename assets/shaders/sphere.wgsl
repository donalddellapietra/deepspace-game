#include "bindings.wgsl"
#include "tree.wgsl"
#include "face_math.wgsl"
#include "ray_prim.wgsl"
#include "face_walk.wgsl"

// Unified sphere march. ONE function — called from
// `march_cartesian`'s CubedSphereBody dispatch and from `march()`
// when the render root itself is a body cell. Both callers pass the
// body cell's origin and size in the CURRENT RENDER FRAME's
// coordinate system. Nothing in this file hardcodes a body-absolute
// constant; `body_origin` and `body_size` are the only geometric
// anchors, and both are guaranteed O(1) in render-frame units
// (frame rescaling keeps them bounded across any zoom depth).
//
// The render frame never roots at a face subtree. When the camera
// has zoomed deep into a face, `with_render_margin` keeps the
// render root at the containing body cell and the logical anchor
// path drives edit/highlight — the shader doesn't need a
// face-rooted entry point.

fn sphere_depth_tint(rn: f32) -> f32 {
    return 0.55 + 0.45 * clamp(rn, 0.0, 1.0);
}

// Simple single-level bevel used by the fragment shader for
// Cartesian hits (main.wgsl picks whichever two axes face the hit
// normal and computes their edge-distance smoothstep). Retained for
// backward compatibility with cartesian rendering.
fn face_uv_for_normal(local: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let an = abs(normal);
    if an.x >= an.y && an.x >= an.z {
        return local.yz;
    }
    if an.y >= an.z {
        return local.xz;
    }
    return local.xy;
}

fn cube_face_bevel(local: vec3<f32>, normal: vec3<f32>) -> f32 {
    let uv = face_uv_for_normal(local, normal);
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.02, 0.14, edge);
}

// Per-level bevel contribution. Draws a dark ~1px band at the cell
// edges; returns 1.0 when the cell is too small on screen for a
// visible band (so deep-sub-pixel grid lines don't darken everything).
fn bevel_level(
    un: f32, vn: f32,
    u_lo: f32, v_lo: f32, size: f32,
    cell_px: f32,
) -> f32 {
    if cell_px < 2.0 {
        return 1.0;
    }
    let cell_u = clamp((un - u_lo) / size, 0.0, 1.0);
    let cell_v = clamp((vn - v_lo) / size, 0.0, 1.0);
    let face_edge = min(
        min(cell_u, 1.0 - cell_u),
        min(cell_v, 1.0 - cell_v),
    );
    let band_end = clamp(1.0 / cell_px, 0.0, 0.25);
    let bevel = smoothstep(0.0, band_end, face_edge);
    return 0.78 + 0.22 * bevel;
}

// Single-level bevel overlay. Only the hit cell itself contributes
// a grid line — no ancestor or descendant overlays. This keeps the
// visible voxel grid in 1:1 correspondence with actual walker cells,
// so the cursor highlight AABB outlines the same cell the user sees.
// (A prior multi-level stack drew coarser ancestor grids that
// visually suggested bigger voxels than actually existed, making the
// depth-8 highlight appear misaligned with the depth-2 "bevel cell"
// the eye picked.)
fn sphere_bevel_stack(
    un: f32, vn: f32,
    u_lo: f32, v_lo: f32, size: f32,
    reference_scale: f32,
    ray_dist: f32,
    pixel_density: f32,
) -> f32 {
    let safe_dist = max(ray_dist, 1e-6);
    let base_px = size * reference_scale / safe_dist * pixel_density;
    return bevel_level(un, vn, u_lo, v_lo, size, base_px);
}

// Per-ray LOD depth cap. A face cell at depth `d` has radial extent
// `shell * (1/3)^(d-1)` in render-frame units; pick `d` so that
// extent projects to at least `LOD_PIXEL_THRESHOLD` pixels at the
// current ray distance. Matches Cartesian's Nyquist gate, which is
// what keeps zoom-invariant rendering working.
fn face_lod_cap(ray_dist: f32, shell_size: f32) -> u32 {
    let pixel_density = uniforms.screen_height
        / (2.0 * tan(camera.fov * 0.5));
    let safe_dist = max(ray_dist, 1e-6);
    let ratio = shell_size * pixel_density
        / (safe_dist * max(LOD_PIXEL_THRESHOLD, 1e-6));
    if ratio <= 1.0 { return 1u; }
    let log3_ratio = log2(ratio) * (1.0 / 1.5849625);
    let d_f = 1.0 + log3_ratio;
    return u32(clamp(d_f, 1.0, f32(MAX_FACE_DEPTH)));
}

// One curved-UVR DDA step through the sphere shell. All geometry
// expressed in the render frame's local coordinates via the
// caller-supplied body cell (`body_origin`, `body_size`).
//
// - The shell is an annulus: inner radius `cs_inner = inner_r *
//   body_size`, outer radius `cs_outer = outer_r * body_size`, both
//   in render-frame units.
// - `oc = ray_origin - cs_center` is the ray's position relative to
//   the body center, also in render-frame units. Bounded by
//   `body_size` in magnitude — no precision leak.
// - The DDA walks curved cells by picking the minimum t to the next
//   u/v radial plane crossing or r spherical shell crossing. All
//   plane normals and sphere centers are computed from `body_origin`
//   / `body_size`, never from a global body-absolute constant.
// Face-rooted march. Render frame is a CubedSphereFace subtree node
// at face_depth >= 1. UVR semantics throughout: cell addressing is
// (u_slot, v_slot, r_slot); hit normals are the face's (u_axis,
// v_axis, n_axis) in world frame; shading uses face-normalized
// (un, vn, rn) for depth tint.
//
// The per-step intersection math is flat-at-this-scale DDA: u/v
// boundaries are axis-aligned planes in render-frame-local [0, 3)³,
// r boundaries likewise. That's NOT a coordinate-system change —
// it's the observation that a UVR cell's curvature over its own
// extent is O(cell_size / body_radius) = O(3^-D), below f32
// precision at D >= 1. So the curved-UVR math degenerates to flat
// at face_depth >= 1; the flat form is the precision-safe way to
// express the same UVR stepping.
//
// `root_node_idx`: the face-subtree node acting as render root.
// `ray_origin / ray_dir`: in the render frame's [0, 3)³ coords.
// `root_face_bounds`: (u_min, v_min, r_min, size) of the render
// cell within the full face. Used only for shading (depth tint +
// hit-path reconstruction) — the DDA itself stays in render-frame-
// local coords and doesn't touch face-absolute math.
fn march_face_root(
    root_node_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    root_face_bounds: vec4<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let face = uniforms.root_face_meta.x;
    let u_axis = face_u_axis(face);
    let v_axis = face_v_axis(face);
    let n_axis = face_normal(face);

    // Box-entry into the render frame's [0, 3)³ cell.
    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );
    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    // Axis convention: the face subtree's child slot (us, vs, rs)
    // maps to render-frame-local cell indices (i_x = us, i_y = vs,
    // i_z = rs). UVR is directly the render-frame axes; no
    // additional rotation is needed at this scale.
    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;

    let pixel_density = uniforms.screen_height
        / (2.0 * tan(camera.fov * 0.5));

    // Per-ray LOD cap in face subtree levels. Cell size at walker
    // depth d is `root_face_bounds.w * 3^(1-d)` in face-normalized
    // units; multiplied by `shell = (outer_r - inner_r) * 3` to
    // get render-frame-local cell extent. Same formula the body
    // march uses — consistent LOD across root kinds.
    let inner_r = uniforms.root_radii.x;
    let outer_r = uniforms.root_radii.y;
    let shell = (outer_r - inner_r) * 3.0;
    let window_scale = shell * root_face_bounds.w;

    // Single-step DDA through the 27-ary face subtree. One function
    // body; the walker descends into each 27-slot cell by recursing
    // (via explicit stack) when the slot contains a Node child and
    // the cell's projected pixel size is above the LOD threshold.
    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;
    var cur_cell_size: f32 = 1.0;
    var cur_node_origin: vec3<f32> = vec3<f32>(0.0);
    var cur_side_dist: vec3<f32>;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = root_node_idx;

    let root_header_off = node_offsets[root_node_idx];
    var cur_occupancy: u32 = tree[root_header_off];
    var cur_first_child: u32 = tree[root_header_off + 1u];

    let root_cell = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    s_cell[0] = pack_cell(root_cell);
    let cell_f = vec3<f32>(root_cell);
    cur_side_dist = vec3<f32>(
        select((cell_f.x - entry_pos.x) * inv_dir.x,
               (cell_f.x + 1.0 - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
        select((cell_f.y - entry_pos.y) * inv_dir.y,
               (cell_f.y + 1.0 - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
        select((cell_f.z - entry_pos.z) * inv_dir.z,
               (cell_f.z + 1.0 - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
    );

    let ray_metric = max(length(ray_dir), 1e-6);
    var iterations = 0u;
    let max_iterations = 2048u;

    loop {
        if iterations >= max_iterations { break; }
        iterations += 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;
            cur_cell_size = cur_cell_size * 3.0;
            let parent_cell = unpack_cell(s_cell[depth]);
            cur_node_origin = cur_node_origin - vec3<f32>(parent_cell) * cur_cell_size;
            let lc_pop = vec3<f32>(parent_cell);
            cur_side_dist = vec3<f32>(
                select((cur_node_origin.x + lc_pop.x * cur_cell_size - entry_pos.x) * inv_dir.x,
                       (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
                select((cur_node_origin.y + lc_pop.y * cur_cell_size - entry_pos.y) * inv_dir.y,
                       (cur_node_origin.y + (lc_pop.y + 1.0) * cur_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
                select((cur_node_origin.z + lc_pop.z * cur_cell_size - entry_pos.z) * inv_dir.z,
                       (cur_node_origin.z + (lc_pop.z + 1.0) * cur_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
            );
            let parent_header_off = node_offsets[s_node_idx[depth]];
            cur_occupancy = tree[parent_header_off];
            cur_first_child = tree[parent_header_off + 1u];
            let m_oob = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(parent_cell + vec3<i32>(m_oob) * step);
            cur_side_dist += m_oob * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_oob;
            continue;
        }

        // UVR slot index from the cell's (x, y, z) position —
        // semantically (us, vs, rs) for a face-subtree node.
        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let slot_bit = 1u << slot;
        if ((cur_occupancy & slot_bit) == 0u) {
            let m_empty = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_empty) * step);
            cur_side_dist += m_empty * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_empty;
            continue;
        }

        let rank = countOneBits(cur_occupancy & (slot_bit - 1u));
        let child_base = cur_first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;

        if tag == 1u {
            // Block hit. Compute face-space hit normal from the
            // axis-aligned normal we tracked across the DDA step.
            // normal is currently (-step * min_axis_mask); convert
            // to UVR-face space: x→u_axis, y→v_axis, z→n_axis.
            var hit_normal: vec3<f32>;
            if abs(normal.x) > 0.5 { hit_normal = u_axis * sign(normal.x); }
            else if abs(normal.y) > 0.5 { hit_normal = v_axis * sign(normal.y); }
            else { hit_normal = n_axis * sign(normal.z); }

            let cell_min_h = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_min_h + vec3<f32>(cur_cell_size));
            let t_hit = max(cell_box_h.t_enter, 0.0);
            let hit_pos = ray_origin + ray_dir * t_hit;

            // Face-normalized (un, vn, rn) at hit — for shading.
            // hit_pos is in render-frame-local [0, 3)³. Map back to
            // face-normalized via the render cell's bounds:
            //   un = u_min + (hit_pos.x / 3) * size
            let un = clamp(root_face_bounds.x + (hit_pos.x / 3.0) * root_face_bounds.w, 0.0, 1.0);
            let vn = clamp(root_face_bounds.y + (hit_pos.y / 3.0) * root_face_bounds.w, 0.0, 1.0);
            let rn = clamp(root_face_bounds.z + (hit_pos.z / 3.0) * root_face_bounds.w, 0.0, 1.0);

            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            // Single-level bevel on the walker's terminal cell.
            let cell_frac = clamp(
                (hit_pos - cell_min_h) / cur_cell_size,
                vec3<f32>(0.0), vec3<f32>(1.0),
            );
            let base_px = cur_cell_size * window_scale / max(t_hit, 1e-6) * pixel_density / 3.0;
            let bevel = bevel_level(cell_frac.x, cell_frac.y, 0.0, 0.0, 1.0, base_px);
            let depth_tint = sphere_depth_tint(rn);

            result.hit = true;
            result.t = t_hit;
            result.normal = hit_normal;
            result.color = palette[(packed >> 8u) & 0xFFFFu].rgb
                         * (ambient + diffuse * 0.78)
                         * axis_tint * bevel * depth_tint;
            result.cell_min = cell_min_h;
            result.cell_size = cur_cell_size;
            _ = un; _ = vn; // silence warnings if unused later
            return result;
        }

        if tag != 2u {
            // EntityRef or unexpected tag — skip.
            let m_skip = min_axis_mask(cur_side_dist);
            s_cell[depth] = pack_cell(cell + vec3<i32>(m_skip) * step);
            cur_side_dist += m_skip * delta_dist * cur_cell_size;
            normal = -vec3<f32>(step) * m_skip;
            continue;
        }

        // tag == 2: Node. Descend (with LOD + stack-depth guard).
        let child_idx = tree[child_base + 1u];
        let at_max = depth + 1u >= MAX_STACK_DEPTH;
        let child_cell_size = cur_cell_size / 3.0;
        let min_side = min(cur_side_dist.x, min(cur_side_dist.y, cur_side_dist.z));
        let ray_dist = max(min_side * ray_metric, 0.001);
        let cell_world = child_cell_size * window_scale;
        let lod_pixels = cell_world / ray_dist * pixel_density;
        let at_lod = lod_pixels < LOD_PIXEL_THRESHOLD;

        if at_max || at_lod {
            let bt = (packed >> 8u) & 0xFFFFu;
            if bt == 0xFFFEu || bt == 0xFFFDu {
                let m_lodt = min_axis_mask(cur_side_dist);
                s_cell[depth] = pack_cell(cell + vec3<i32>(m_lodt) * step);
                cur_side_dist += m_lodt * delta_dist * cur_cell_size;
                normal = -vec3<f32>(step) * m_lodt;
                continue;
            }
            var hit_normal: vec3<f32>;
            if abs(normal.x) > 0.5 { hit_normal = u_axis * sign(normal.x); }
            else if abs(normal.y) > 0.5 { hit_normal = v_axis * sign(normal.y); }
            else { hit_normal = n_axis * sign(normal.z); }
            let cell_min_l = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
            let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_min_l + vec3<f32>(cur_cell_size));
            let t_hit = max(cell_box_l.t_enter, 0.0);
            let hit_pos = ray_origin + ray_dir * t_hit;
            let rn = clamp(root_face_bounds.z + (hit_pos.z / 3.0) * root_face_bounds.w, 0.0, 1.0);
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let cell_frac = clamp(
                (hit_pos - cell_min_l) / cur_cell_size,
                vec3<f32>(0.0), vec3<f32>(1.0),
            );
            let base_px = cur_cell_size * window_scale / max(t_hit, 1e-6) * pixel_density / 3.0;
            let bevel = bevel_level(cell_frac.x, cell_frac.y, 0.0, 0.0, 1.0, base_px);
            let depth_tint = sphere_depth_tint(rn);
            result.hit = true;
            result.t = t_hit;
            result.normal = hit_normal;
            result.color = palette[bt].rgb
                         * (ambient + diffuse * 0.78)
                         * axis_tint * bevel * depth_tint;
            result.cell_min = cell_min_l;
            result.cell_size = cur_cell_size;
            return result;
        }

        // Descend into child node.
        let child_origin = cur_node_origin + vec3<f32>(cell) * cur_cell_size;
        let ct_start = max(root_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
        let child_entry = ray_origin + ray_dir * ct_start;
        let local_entry = (child_entry - child_origin) / child_cell_size;
        depth += 1u;
        s_node_idx[depth] = child_idx;
        cur_node_origin = child_origin;
        cur_cell_size = child_cell_size;
        let child_header_off = node_offsets[child_idx];
        cur_occupancy = tree[child_header_off];
        cur_first_child = tree[child_header_off + 1u];
        let new_cell = vec3<i32>(
            clamp(i32(floor(local_entry.x)), 0, 2),
            clamp(i32(floor(local_entry.y)), 0, 2),
            clamp(i32(floor(local_entry.z)), 0, 2),
        );
        s_cell[depth] = pack_cell(new_cell);
        let lc = vec3<f32>(new_cell);
        cur_side_dist = vec3<f32>(
            select((child_origin.x + lc.x * child_cell_size - entry_pos.x) * inv_dir.x,
                   (child_origin.x + (lc.x + 1.0) * child_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
            select((child_origin.y + lc.y * child_cell_size - entry_pos.y) * inv_dir.y,
                   (child_origin.y + (lc.y + 1.0) * child_cell_size - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
            select((child_origin.z + lc.z * child_cell_size - entry_pos.z) * inv_dir.z,
                   (child_origin.z + (lc.z + 1.0) * child_cell_size - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
        );
    }

    return result;
}

fn march_sphere_body(
    body_node_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let cs_outer = outer_r_local * body_size;
    let cs_inner = inner_r_local * body_size;
    let shell = cs_outer - cs_inner;
    if shell <= 0.0 { return result; }

    // Ray-outer-sphere entry. Standard quadratic; `oc` is bounded by
    // `body_size`, so none of the intermediate values overflow.
    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - cs_outer * cs_outer;
    let disc = b * b - c_outer;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let eps_init = max(shell * 1e-5, 1e-7);
    var t = t_enter + eps_init;
    var steps: u32 = 0u;
    var last_face_axis: u32 = 6u;
    let pixel_density = uniforms.screen_height
        / (2.0 * tan(camera.fov * 0.5));

    loop {
        if t >= t_exit || steps > 4096u { break; }
        steps = steps + 1u;
        if ENABLE_STATS { ray_steps = ray_steps + 1u; }

        let local = oc + ray_dir * t;
        let r = length(local);
        if r >= cs_outer || r < cs_inner { break; }

        // Pick dominant face from the radial unit direction.
        let n = local / r;
        let face = pick_face(n);
        let n_axis = face_normal(face);
        let u_axis = face_u_axis(face);
        let v_axis = face_v_axis(face);

        // Cube-UV coordinates on the face. At grazing angles the
        // projection stretches to infinity; the `un`/`vn` clamp to
        // [0, 0.9999999] below absorbs the out-of-range values so
        // the walker still finds the nearest cell.
        let axis_dot = dot(n, n_axis);
        let cube_u = dot(n, u_axis) / axis_dot;
        let cube_v = dot(n, v_axis) / axis_dot;
        let u_ea = cube_to_ea(cube_u);
        let v_ea = cube_to_ea(cube_v);

        let un = clamp((u_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let vn = clamp((v_ea + 1.0) * 0.5, 0.0, 0.9999999);
        let rn = clamp((r - cs_inner) / shell, 0.0, 0.9999999);

        // Walk the face subtree with a per-ray LOD cap.
        let walk_depth = face_lod_cap(t, shell);
        let walk = walk_face_subtree(body_node_idx, face, un, vn, rn, walk_depth);
        let block_id = walk.block;

        if block_id != 0u {
            var hit_normal: vec3<f32>;
            switch last_face_axis {
                case 0u: { hit_normal = -u_axis; }
                case 1u: { hit_normal =  u_axis; }
                case 2u: { hit_normal = -v_axis; }
                case 3u: { hit_normal =  v_axis; }
                case 4u: { hit_normal = -n; }
                case 5u: { hit_normal =  n; }
                default: { hit_normal =  n; }
            }
            let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
            let diffuse = max(dot(hit_normal, sun_dir), 0.0);
            let axis_tint = abs(hit_normal.y) * 1.0
                          + (abs(hit_normal.x) + abs(hit_normal.z)) * 0.82;
            let ambient = 0.22;
            let bevel = sphere_bevel_stack(
                un, vn,
                walk.u_lo, walk.v_lo, walk.size,
                shell, t, pixel_density,
            );
            let depth_tint = sphere_depth_tint(rn);
            result.hit = true;
            result.t = t;
            result.normal = hit_normal;
            result.color = palette[block_id].rgb
                         * (ambient + diffuse * 0.78)
                         * axis_tint * bevel * depth_tint;
            return result;
        }

        // Empty cell: advance to next cell boundary. Candidates are
        // the four u/v radial planes and two r spherical shells that
        // bound the current walker cell. Every plane normal / sphere
        // center is local to this body cell (passed in by caller);
        // nothing references a global body-absolute constant.
        let u_lo_ea = walk.u_lo * 2.0 - 1.0;
        let u_hi_ea = (walk.u_lo + walk.size) * 2.0 - 1.0;
        let n_u_lo = u_axis - ea_to_cube(u_lo_ea) * n_axis;
        let n_u_hi = u_axis - ea_to_cube(u_hi_ea) * n_axis;

        let v_lo_ea = walk.v_lo * 2.0 - 1.0;
        let v_hi_ea = (walk.v_lo + walk.size) * 2.0 - 1.0;
        let n_v_lo = v_axis - ea_to_cube(v_lo_ea) * n_axis;
        let n_v_hi = v_axis - ea_to_cube(v_hi_ea) * n_axis;

        let r_lo = cs_inner + walk.r_lo * shell;
        let r_hi = cs_inner + (walk.r_lo + walk.size) * shell;

        var t_next = t_exit + 1.0;
        var winning_axis: u32 = 6u;
        let zero3 = vec3<f32>(0.0);
        let c_u_lo = ray_plane_t(oc, ray_dir, zero3, n_u_lo);
        if c_u_lo > t && c_u_lo < t_next { t_next = c_u_lo; winning_axis = 0u; }
        let c_u_hi = ray_plane_t(oc, ray_dir, zero3, n_u_hi);
        if c_u_hi > t && c_u_hi < t_next { t_next = c_u_hi; winning_axis = 1u; }
        let c_v_lo = ray_plane_t(oc, ray_dir, zero3, n_v_lo);
        if c_v_lo > t && c_v_lo < t_next { t_next = c_v_lo; winning_axis = 2u; }
        let c_v_hi = ray_plane_t(oc, ray_dir, zero3, n_v_hi);
        if c_v_hi > t && c_v_hi < t_next { t_next = c_v_hi; winning_axis = 3u; }
        let c_r_lo = ray_sphere_after(oc, ray_dir, zero3, r_lo, t);
        if c_r_lo > t && c_r_lo < t_next { t_next = c_r_lo; winning_axis = 4u; }
        let c_r_hi = ray_sphere_after(oc, ray_dir, zero3, r_hi, t);
        if c_r_hi > t && c_r_hi < t_next { t_next = c_r_hi; winning_axis = 5u; }

        if t_next >= t_exit { break; }
        last_face_axis = winning_axis;
        // SDF-min-cell reach floor: advance by at least the current
        // cell's radial extent to guarantee we leave the walker cell,
        // preventing stepper stall at deep zoom.
        let t_ulp = max(abs(t) * 1.2e-7, 1e-30);
        let cell_eps = max(shell * walk.size * 1e-3, t_ulp * 4.0);
        t = t_next + cell_eps;
    }

    return result;
}
