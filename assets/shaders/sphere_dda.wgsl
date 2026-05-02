// Sphere DDA — body-rooted UV-sphere render of a `WrappedPlane`
// node. Operates in the WP's local `[0, 3)³` frame with the
// implied sphere centered at `(1.5, 1.5, 1.5)` of radius
// `body_size / (2π)`. Composed of:
//
//   * `SlabSample` + `sample_slab_cell`: walk the slab tree from
//     its root to (cell_x, cell_y, cell_z) at slab_depth, returning
//     the cell's block_type and (for non-uniform anchors) its
//     subtree's child node index.
//   * `make_sphere_hit`: closest-face UV bevel + HitResult assembly,
//     shared between the slab-cell DDA and the anchor sub-cell DDA.
//   * `sphere_descend_anchor`: stack-based DDA over a non-uniform
//     anchor's 27-children grid, recursively descending until each
//     sub-cell terminates as a Block / uniform-flatten / empty-rep.
//   * `sphere_uv_in_cell`: top-level sphere DDA at slab cell
//     granularity. Dispatches `sphere_descend_anchor` for any
//     non-uniform slab cell.
//
// Step 7 of the sphere sub-frame architecture splits this body-
// rooted DDA from the sub-frame-rooted DDA (`sphere_subframe.wgsl`)
// so each can evolve independently. Both share the same primitive
// helpers in `ray_prim.wgsl`.

struct SlabSample {
    block_type: u32,
    tag: u32,
    child_idx: u32,
};

// Phase 3 REVISED Step A.1 — sample the slab tree at (cx, cy, cz).
//
// Walks `slab_depth` levels of 27-children Cartesian descent. At
// each level: integer-divide the cell coords by `3^(remaining_levels)`
// to get the slot index, look up the slab tree's child entry there.
// At tag=1 (uniform-flatten Block leaf) → return that material.
// At tag=2 (non-uniform Node) at the LAST level → return its
// representative_block AND child_idx so the caller can decide whether
// to LOD-splat or descend into the subtree.
// On tag=2 mid-walk → descend one level. tag=0 / unknown → empty.
fn sample_slab_cell(
    slab_root_idx: u32,
    slab_depth: u32,
    cx: i32, cy: i32, cz: i32,
) -> SlabSample {
    var out: SlabSample;
    out.block_type = 0xFFFEu;
    out.tag = 0u;
    out.child_idx = 0u;
    var idx = slab_root_idx;
    var cells_per_slot: i32 = 1;
    for (var k: u32 = 1u; k < slab_depth; k = k + 1u) {
        cells_per_slot = cells_per_slot * 3;
    }
    for (var level: u32 = 0u; level < slab_depth; level = level + 1u) {
        let sx = (cx / cells_per_slot) % 3;
        let sy = (cy / cells_per_slot) % 3;
        let sz = (cz / cells_per_slot) % 3;
        let slot = u32(sx + sy * 3 + sz * 9);

        let header_off = node_offsets[idx];
        let occ = tree[header_off];
        let bit = 1u << slot;
        if (occ & bit) == 0u { return out; }
        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            out.block_type = block_type;
            out.tag = 1u;
            return out;
        }
        if level == slab_depth - 1u {
            out.block_type = block_type;
            out.tag = tag;
            if tag == 2u {
                out.child_idx = tree[child_base + 1u];
            }
            return out;
        }
        if tag == 2u {
            idx = tree[child_base + 1u];
        } else {
            return out;
        }
        cells_per_slot = cells_per_slot / 3;
    }
    return out;
}

// Closest-face UV bevel for a (lon, lat, r) cell. Same recipe as the
// existing slab-cell hit code, extracted as a helper so the slab DDA
// and the sub-cell anchor DDA share one source of truth.
//
// `r, lat_p, lon_p` are the ray's spherical coords at hit. The cell's
// six faces are at the supplied lo/hi bounds; closest-face is chosen
// by world-space arc length (lon arcs scale by r·cos(lat), lat arcs
// by r, r-shells in radial units). The other two axes' in-cell
// fractional positions form the 2D UV used to bevel the face edge.
fn make_sphere_hit(
    pos: vec3<f32>, n_step: vec3<f32>, t_param: f32, inv_norm: f32,
    block_type: u32,
    r: f32, lat_p: f32, lon_p: f32,
    lon_lo_c: f32, lon_hi_c: f32,
    lat_lo_c: f32, lat_hi_c: f32,
    r_lo_c: f32, r_hi_c: f32,
    lon_step_c: f32, lat_step_c: f32, r_step_c: f32,
) -> HitResult {
    var result: HitResult;
    let cos_lat = max(cos(lat_p), 1e-3);
    let arc_lon_lo = r * cos_lat * abs(lon_p - lon_lo_c);
    let arc_lon_hi = r * cos_lat * abs(lon_p - lon_hi_c);
    let arc_lat_lo = r * abs(lat_p - lat_lo_c);
    let arc_lat_hi = r * abs(lat_p - lat_hi_c);
    let arc_r_lo  = abs(r - r_lo_c);
    let arc_r_hi  = abs(r - r_hi_c);
    var best = arc_lon_lo;
    var axis: u32 = 0u;
    if arc_lon_hi < best { best = arc_lon_hi; axis = 0u; }
    if arc_lat_lo < best { best = arc_lat_lo; axis = 1u; }
    if arc_lat_hi < best { best = arc_lat_hi; axis = 1u; }
    if arc_r_lo  < best { best = arc_r_lo;  axis = 2u; }
    if arc_r_hi  < best { best = arc_r_hi;  axis = 2u; }

    let lon_in_cell = clamp((lon_p - lon_lo_c) / lon_step_c, 0.0, 1.0);
    let lat_in_cell = clamp((lat_p - lat_lo_c) / lat_step_c, 0.0, 1.0);
    let r_in_cell   = clamp((r     - r_lo_c)   / r_step_c,   0.0, 1.0);
    var u_in_face: f32;
    var v_in_face: f32;
    if axis == 0u {
        u_in_face = lat_in_cell;
        v_in_face = r_in_cell;
    } else if axis == 1u {
        u_in_face = lon_in_cell;
        v_in_face = r_in_cell;
    } else {
        u_in_face = lon_in_cell;
        v_in_face = lat_in_cell;
    }
    let face_edge = min(
        min(u_in_face, 1.0 - u_in_face),
        min(v_in_face, 1.0 - v_in_face),
    );
    let shape = smoothstep(0.02, 0.14, face_edge);
    let bevel_strength = 0.7 + 0.3 * shape;

    result.hit = true;
    result.t = t_param * inv_norm;
    result.color = palette[block_type].rgb * bevel_strength;
    result.normal = n_step;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    // Neutralize main.wgsl::cube_face_bevel — see slab DDA hit comment.
    result.cell_min = pos - vec3<f32>(0.5);
    result.cell_size = 1.0;
    return result;
}


// Stack-based DDA inside an anchor block at sub-cell granularity.
// Mirrors `march_cartesian` but in (lon, lat, r) space using the
// sphere-cell boundary primitives (ray_meridian_t, ray_parallel_t,
// ray_sphere_after).
//
// Recipe — tree-structure descent (no camera-distance LOD):
// * 27-children descent — each level divides (lon, lat, r) cell sizes
//   by 3.
// * tag=1 → hit the leaf Block at this sub-cell.
// * tag=2 + non-empty representative → push child frame, continue
//   DDA at finer cells. (Pack format auto-flattens uniform Cartesian
//   subtrees to tag=1, so descent only enters genuinely non-uniform
//   subtrees — i.e. the path the user actually edited.)
// * tag=2 + empty representative → skip whole cell.
// * empty slot / unknown → advance ray to the cell's nearest face.
// * stack at MAX_STACK_DEPTH → splat representative (hardware ceiling).
//
// On exit (ray leaves the slab cell or stack underflows), returns
// hit=false so the caller's main slab DDA can continue past this
// slab cell.
fn sphere_descend_anchor(
    anchor_idx: u32,
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    oc: vec3<f32>, cs_center: vec3<f32>, inv_norm: f32,
    t_in: f32, t_exit: f32,
    slab_lon_lo: f32, slab_lon_step: f32,
    slab_lat_lo: f32, slab_lat_step: f32,
    slab_r_lo:   f32, slab_r_step:   f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;

    // Frame 0 = anchor block. Each axis is 1/3 of the slab cell.
    var cur_lon_org = slab_lon_lo;
    var cur_lat_org = slab_lat_lo;
    var cur_r_org   = slab_r_lo;
    var cur_lon_step = slab_lon_step / 3.0;
    var cur_lat_step = slab_lat_step / 3.0;
    var cur_r_step   = slab_r_step   / 3.0;

    var t = t_in;

    // Initial sub-cell from the ray's t.
    {
        let pos0 = ray_origin + ray_dir * t;
        let off0 = pos0 - cs_center;
        let r0 = max(length(off0), 1e-9);
        let n0 = off0 / r0;
        let lat0 = asin(clamp(n0.y, -1.0, 1.0));
        let lon0 = atan2(n0.z, n0.x);
        let cx0 = clamp(i32(floor((lon0 - cur_lon_org) / cur_lon_step)), 0, 2);
        let cy0 = clamp(i32(floor((r0   - cur_r_org)   / cur_r_step)),   0, 2);
        let cz0 = clamp(i32(floor((lat0 - cur_lat_org) / cur_lat_step)), 0, 2);
        s_cell[0] = pack_cell(vec3<i32>(cx0, cy0, cz0));
    }

    var iters: u32 = 0u;
    loop {
        if iters > 256u { break; }
        iters = iters + 1u;
        if t > t_exit { break; }

        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            // OOB: pop one frame, or exit if at the anchor's outer face.
            if depth == 0u { break; }
            depth = depth - 1u;
            cur_lon_step = cur_lon_step * 3.0;
            cur_lat_step = cur_lat_step * 3.0;
            cur_r_step   = cur_r_step   * 3.0;
            let popped = unpack_cell(s_cell[depth]);
            cur_lon_org = cur_lon_org - f32(popped.x) * cur_lon_step;
            cur_r_org   = cur_r_org   - f32(popped.y) * cur_r_step;
            cur_lat_org = cur_lat_org - f32(popped.z) * cur_lat_step;
            // After the pop, recompute which cell we're in at the
            // parent frame using the ray's current t. This may match
            // `popped` (we exited an axis only), or its neighbour
            // (the axis stepped over a parent boundary already).
            let pos_p = ray_origin + ray_dir * t;
            let off_p = pos_p - cs_center;
            let r_p = max(length(off_p), 1e-9);
            let n_p = off_p / r_p;
            let lat_p = asin(clamp(n_p.y, -1.0, 1.0));
            let lon_p = atan2(n_p.z, n_p.x);
            let cx_p = i32(floor((lon_p - cur_lon_org) / cur_lon_step));
            let cy_p = i32(floor((r_p   - cur_r_org)   / cur_r_step));
            let cz_p = i32(floor((lat_p - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_p, cy_p, cz_p));
            continue;
        }

        // Cell bounds for boundary crossings + occupancy lookup.
        let lon_lo = cur_lon_org + f32(cell.x) * cur_lon_step;
        let lon_hi = lon_lo + cur_lon_step;
        let r_lo   = cur_r_org   + f32(cell.y) * cur_r_step;
        let r_hi   = r_lo + cur_r_step;
        let lat_lo = cur_lat_org + f32(cell.z) * cur_lat_step;
        let lat_hi = lat_lo + cur_lat_step;

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;

        // t_next: smallest cell-face crossing > t. Used by every
        // "advance past this cell" branch below.
        var t_next = t_exit + 1.0;
        let t_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
        if t_lon_lo > 0.0 && t_lon_lo < t_next { t_next = t_lon_lo; }
        let t_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
        if t_lon_hi > 0.0 && t_lon_hi < t_next { t_next = t_lon_hi; }
        let t_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
        if t_lat_lo > 0.0 && t_lat_lo < t_next { t_next = t_lat_lo; }
        let t_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
        if t_lat_hi > 0.0 && t_lat_hi < t_next { t_next = t_lat_hi; }
        let t_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
        if t_r_lo > 0.0 && t_r_lo < t_next { t_next = t_r_lo; }
        let t_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
        if t_r_hi > 0.0 && t_r_hi < t_next { t_next = t_r_hi; }

        if (occ & bit) == 0u {
            // Empty slot. Advance to the next cell within this frame.
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_a = ray_origin + ray_dir * t;
            let off_a = pos_a - cs_center;
            let r_a = max(length(off_a), 1e-9);
            let n_a = off_a / r_a;
            let lat_a = asin(clamp(n_a.y, -1.0, 1.0));
            let lon_a = atan2(n_a.z, n_a.x);
            let cx_a = i32(floor((lon_a - cur_lon_org) / cur_lon_step));
            let cy_a = i32(floor((r_a   - cur_r_org)   / cur_r_step));
            let cz_a = i32(floor((lat_a - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_a, cy_a, cz_a));
            continue;
        }

        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            // Block leaf. Hit at this sub-cell.
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            let lon_h = atan2(n_h.z, n_h.x);
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h, lon_h,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        if tag != 2u {
            // EntityRef etc. — treat as empty for sphere subtree DDA
            // (no entities-inside-anchor for now). Advance.
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_e = ray_origin + ray_dir * t;
            let off_e = pos_e - cs_center;
            let r_e = max(length(off_e), 1e-9);
            let n_e = off_e / r_e;
            let lat_e = asin(clamp(n_e.y, -1.0, 1.0));
            let lon_e = atan2(n_e.z, n_e.x);
            let cx_e = i32(floor((lon_e - cur_lon_org) / cur_lon_step));
            let cy_e = i32(floor((r_e   - cur_r_org)   / cur_r_step));
            let cz_e = i32(floor((lat_e - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_e, cy_e, cz_e));
            continue;
        }

        // tag == 2u: non-uniform Node. LOD-gate descent.
        if block_type == 0xFFFEu {
            // Subtree empty per `representative_block` — skip whole cell.
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_r = ray_origin + ray_dir * t;
            let off_r = pos_r - cs_center;
            let r_r = max(length(off_r), 1e-9);
            let n_r = off_r / r_r;
            let lat_r = asin(clamp(n_r.y, -1.0, 1.0));
            let lon_r = atan2(n_r.z, n_r.x);
            let cx_r = i32(floor((lon_r - cur_lon_org) / cur_lon_step));
            let cy_r = i32(floor((r_r   - cur_r_org)   / cur_r_step));
            let cz_r = i32(floor((lat_r - cur_lat_org) / cur_lat_step));
            s_cell[depth] = pack_cell(vec3<i32>(cx_r, cy_r, cz_r));
            continue;
        }

        // Stack ceiling = hardware limit. No camera-distance LOD here:
        // sphere mode locks the render frame at WrappedPlane, so the
        // Cartesian "lod_pixels < threshold" check would collapse deep
        // edits to the representative_block — exactly the bug we're
        // fixing. Tree structure (uniform-flatten = tag=1, terminate)
        // alone drives descent.
        let at_max = depth + 1u >= MAX_STACK_DEPTH;
        if at_max {
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            let lon_h = atan2(n_h.z, n_h.x);
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h, lon_h,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        // Push child frame.
        let child_idx = tree[child_base + 1u];
        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_lon_org = lon_lo;
        cur_lat_org = lat_lo;
        cur_r_org   = r_lo;
        cur_lon_step = cur_lon_step / 3.0;
        cur_lat_step = cur_lat_step / 3.0;
        cur_r_step   = cur_r_step   / 3.0;

        // Pick the entry sub-cell from the ray's current t.
        let pos_c = ray_origin + ray_dir * t;
        let off_c = pos_c - cs_center;
        let r_c = max(length(off_c), 1e-9);
        let n_c = off_c / r_c;
        let lat_c = asin(clamp(n_c.y, -1.0, 1.0));
        let lon_c = atan2(n_c.z, n_c.x);
        let cx_c = clamp(i32(floor((lon_c - cur_lon_org) / cur_lon_step)), 0, 2);
        let cy_c = clamp(i32(floor((r_c   - cur_r_org)   / cur_r_step)),   0, 2);
        let cz_c = clamp(i32(floor((lat_c - cur_lat_org) / cur_lat_step)), 0, 2);
        s_cell[depth] = pack_cell(vec3<i32>(cx_c, cy_c, cz_c));
    }

    return result;
}

// Step 6 (sphere sub-frame architecture): sphere DDA scoped to a
// sub-frame inside the WrappedPlane. The ray is already in
// sub-frame local rotated+translated coords (= origin near 0,
// basis aligned to the sub-frame's lon-tangent / lat-tangent /
// radial axes). Sphere center in this frame is at (0, 0, -r_c)
// where `r_c = uniforms.subframe_r.z`. Sphere radius = `body_size
// / (2π)`, body_size = 3 (= the WP's local frame size).
//
// Currently a STUB: returns a fixed magenta hit so we can verify
// the dispatch path wires correctly when the active frame is a
// SphereSubFrame. Step 7 will replace this with the full DDA in
// sub-frame local coords using the sphere boundary primitives
fn sphere_uv_in_cell(
    body_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    ray_origin: vec3<f32>,
    ray_dir_in: vec3<f32>,
    lat_max: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    // Quadratic discriminant assumes unit ray_dir; the marcher
    // passes camera.forward + right·ndc + up·ndc which isn't unit.
    // Renormalise for the intersect — `t` returned to the caller is
    // in the same parameterisation (scaled by 1 / |ray_dir_in|).
    let ray_dir = normalize(ray_dir_in);

    // Sphere center: middle of the body cell.
    // Radius: body_size / (2π) — the slab's circumference is
    // exactly body_size (since dims_x fills X), so R = C / (2π).
    let cs_center = body_origin + vec3<f32>(body_size * 0.5);
    let r_sphere = body_size / (2.0 * 3.14159265);

    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - r_sphere * r_sphere;
    let disc = b * b - c;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter = max(-b - sq, 0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return result; }

    let hit_pos = ray_origin + ray_dir * t_enter;
    let n = (hit_pos - cs_center) / r_sphere;

    // A.3 — Full cell-by-cell DDA in spherical (lon, lat, r) cell
    // coords at slab granularity. At each step, sample the cell; if
    // empty, compute t to ALL 6 cell-boundary surfaces (2 meridians,
    // 2 parallels, 2 spheres) and step to the smallest crossing.
    //
    // Slab Y maps to radial: cy = dims_y - 1 at r_outer, cy = 0 at
    // r_inner. Slab X (longitude) wraps the full 2π; cells span
    // lon ∈ [-π, π]. Slab Z (latitude) is bounded to ±lat_max
    // (poles banned outside this band).
    //
    // On a non-empty slab cell we mirror march_cartesian's recipe:
    //  * tag=1 (uniform-flatten Block) → render Block at slab scale.
    //  * tag=2 (non-uniform Node) → LOD-gate. If sub-cells project
    //    above LOD_PIXEL_THRESHOLD, dispatch sphere_descend_anchor to
    //    walk the anchor's subtree at finer (lon, lat, r) granularity.
    //    Otherwise splat the representative_block at slab scale.
    let dims_x = i32(uniforms.slab_dims.x);
    let dims_y = i32(uniforms.slab_dims.y);
    let dims_z = i32(uniforms.slab_dims.z);
    let slab_depth = uniforms.slab_dims.w;
    let pi = 3.14159265;
    let shell_thickness = r_sphere * 0.25;
    let r_outer = r_sphere;
    let r_inner = r_sphere - shell_thickness;
    let inv_norm = 1.0 / max(length(ray_dir_in), 1e-6);
    let lon_step = 2.0 * pi / f32(dims_x);
    let lat_step = 2.0 * lat_max / f32(dims_z);
    let r_step = shell_thickness / f32(dims_y);

    var t = max(t_enter, 0.0) + 1e-5;
    var iters: u32 = 0u;
    loop {
        if t > t_exit { break; }
        if iters > 256u { break; }
        iters = iters + 1u;

        // Current cell (cell_x, cell_y, cell_z) at the ray's t.
        let pos = ray_origin + ray_dir * t;
        let off = pos - cs_center;
        let r = length(off);
        if r < r_inner || r > r_outer + 1e-3 { break; }
        let n_step = off / r;
        let lat_p = asin(clamp(n_step.y, -1.0, 1.0));
        if abs(lat_p) > lat_max { break; }
        let lon_p = atan2(n_step.z, n_step.x);

        let u_p = (lon_p + pi) / (2.0 * pi);
        let v_p = (lat_p + lat_max) / (2.0 * lat_max);
        let r_frac = (r - r_inner) / shell_thickness;
        let cell_x = clamp(i32(floor(u_p * f32(dims_x))), 0, dims_x - 1);
        let cell_z = clamp(i32(floor(v_p * f32(dims_z))), 0, dims_z - 1);
        let cell_y = clamp(i32(floor(r_frac * f32(dims_y))), 0, dims_y - 1);

        // Slab cell bounds in (lon, lat, r).
        let lon_lo = -pi + f32(cell_x) * lon_step;
        let lon_hi = lon_lo + lon_step;
        let lat_lo = -lat_max + f32(cell_z) * lat_step;
        let lat_hi = lat_lo + lat_step;
        let r_lo = r_inner + f32(cell_y) * r_step;
        let r_hi = r_lo + r_step;

        let sample = sample_slab_cell(body_idx, slab_depth, cell_x, cell_y, cell_z);
        if sample.block_type != 0xFFFEu {
            // tag=2 (non-uniform anchor) ⇒ recursive descent into the
            // anchor's subtree. NO camera-distance LOD gate: in sphere
            // mode the render frame is locked at WrappedPlane (the
            // camera doesn't deepen its frame as anchor_depth grows
            // the way Cartesian does), so a "cell_size/ray_dist <
            // threshold" check would collapse deep edits to the
            // representative — exactly the bug we're fixing. Descend
            // until tag=1 (uniform-flatten Block leaf) or stack max.
            // Tree structure ALONE drives descent — same principle as
            // Cartesian (the LOD gate there only works because frame
            // shifting keeps cells at "normal voxel size" relative to
            // the camera).
            if sample.tag == 2u {
                let sub = sphere_descend_anchor(
                    sample.child_idx,
                    ray_origin, ray_dir,
                    oc, cs_center, inv_norm,
                    t, t_exit,
                    lon_lo, lon_step, lat_lo, lat_step, r_lo, r_step,
                );
                if sub.hit { return sub; }
                // Anchor descent exited without a hit — every sub-cell
                // along the ray's chord was empty. Advance past the
                // slab cell.
                var t_n = t_exit + 1.0;
                let tn_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
                if tn_lon_lo > 0.0 && tn_lon_lo < t_n { t_n = tn_lon_lo; }
                let tn_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
                if tn_lon_hi > 0.0 && tn_lon_hi < t_n { t_n = tn_lon_hi; }
                let tn_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
                if tn_lat_lo > 0.0 && tn_lat_lo < t_n { t_n = tn_lat_lo; }
                let tn_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
                if tn_lat_hi > 0.0 && tn_lat_hi < t_n { t_n = tn_lat_hi; }
                let tn_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
                if tn_r_lo > 0.0 && tn_r_lo < t_n { t_n = tn_r_lo; }
                let tn_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
                if tn_r_hi > 0.0 && tn_r_hi < t_n { t_n = tn_r_hi; }
                if t_n >= t_exit { break; }
                t = t_n + max(r_step * 1e-4, 1e-6);
                continue;
            }
            // tag=1 (uniform-flatten): the entire anchor subtree is
            // one Block — render at slab cell scale, no descent
            // needed (uniform Cartesian subtrees pack-flatten so the
            // shader sees one Block at any zoom).
            return make_sphere_hit(
                pos, n_step, t, inv_norm, sample.block_type,
                r, lat_p, lon_p,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
                lon_step, lat_step, r_step,
            );
        }

        // Empty cell — advance to the nearest cell-face crossing.
        var t_next = t_exit + 1.0;
        let t_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
        if t_lon_lo > 0.0 && t_lon_lo < t_next { t_next = t_lon_lo; }
        let t_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
        if t_lon_hi > 0.0 && t_lon_hi < t_next { t_next = t_lon_hi; }
        let t_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
        if t_lat_lo > 0.0 && t_lat_lo < t_next { t_next = t_lat_lo; }
        let t_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
        if t_lat_hi > 0.0 && t_lat_hi < t_next { t_next = t_lat_hi; }
        let t_r_lo = ray_sphere_after(ray_origin, ray_dir, cs_center, r_lo, t);
        if t_r_lo > 0.0 && t_r_lo < t_next { t_next = t_r_lo; }
        let t_r_hi = ray_sphere_after(ray_origin, ray_dir, cs_center, r_hi, t);
        if t_r_hi > 0.0 && t_r_hi < t_next { t_next = t_r_hi; }

        if t_next >= t_exit { break; }
        t = t_next + max(r_step * 1e-4, 1e-6);
    }

    return result;
}

// Top-level march. Dispatches the current frame's Cartesian DDA,
// then on miss pops to the next ancestor in the ribbon and
// continues. When ribbon is exhausted, returns sky (hit=false).
//
// Each pop transforms the ray into the parent's frame coords:
// `parent_pos = slot_xyz + frame_pos / 3`, `parent_dir = frame_dir / 3`.
// The parent's frame cell still spans `[0, 3)³` in its own
// coords, so the inner DDA is unchanged — only the ray is
