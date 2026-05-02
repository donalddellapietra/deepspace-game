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

    var s_node_idx: array<u32, SPHERE_DESCENT_DEPTH>;
    var s_cell: array<u32, SPHERE_DESCENT_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;

    // ── State tracked through descent (Approach A) ───────────────────
    //
    // `cur_*_center` is the absolute (lon, lat, r) coord of the CURRENT
    // CELL's center. Maintained per-cell via integer-multiple updates:
    //   on push:    center += (new_cell_idx - 1) * new_step
    //   on pop:     center -= (popped_idx - 1) * old_step ; then *=3
    //   on advance: center += step_dir * cur_step (one of 6 faces)
    // Every update is an exact-step shift — never an absolute add of a
    // tiny fraction to a large base, so it doesn't accumulate ULP drift
    // the way `cur_*_org += cell.x * step` did.
    //
    // Cell bounds are derived as `center ± half_step`: half_step is
    // exact (step is divided by 3 each push, bit-exact in f32) and the
    // ± is symmetric, so the meridian/parallel/sphere primitives see
    // face positions without the cumulative drift the old `cur_*_org`
    // pattern produced.
    //
    // `frame_in_frac` is the ray's [0,1]³ position inside the CURRENT
    // FRAME's 3×3×3 grid. On push it's `parent_frac * 3 - cell_idx`
    // (bit-exact); on pop `(popped + child_frac) / 3`. At in-frame
    // advance we use FACE-CROSSING DETECTION (which `t_xxx == t_next`)
    // to step the cell index by ±1 along that axis — pure integer
    // math, immune to f32 cell-width-vs-coord-magnitude issues that
    // plagued `floor((lon_p - cur_lon_org) / step)`. The ray's
    // crossed-axis position at the new t is exactly the face's
    // boundary, so the crossed component of `frame_in_frac` snaps
    // to a precise integer-aligned fraction; non-crossed components
    // are left at their pre-advance value (the ray's other axes
    // didn't move enough within one cell to matter for bevel).
    // ─────────────────────────────────────────────────────────────────

    var cur_lon_step = slab_lon_step / 3.0;
    var cur_lat_step = slab_lat_step / 3.0;
    var cur_r_step   = slab_r_step   / 3.0;

    var t = t_in;

    // Initial cell + frame_in_frac at depth 0. Slab cell is large
    // (~0.077 rad in lon) so absolute math here has plenty of f32
    // precision (~5 decimal digits in the [0, 1] result).
    let pos0 = ray_origin + ray_dir * t;
    let off0 = pos0 - cs_center;
    let r0 = max(length(off0), 1e-9);
    let n0 = off0 / r0;
    let lat0 = asin(clamp(n0.y, -1.0, 1.0));
    let lon0 = atan2(n0.z, n0.x);
    var frame_in_frac = vec3<f32>(
        (lon0 - slab_lon_lo) / slab_lon_step,
        (r0   - slab_r_lo)   / slab_r_step,
        (lat0 - slab_lat_lo) / slab_lat_step,
    );
    let cell0 = clamp(vec3<i32>(
        i32(floor(frame_in_frac.x * 3.0)),
        i32(floor(frame_in_frac.y * 3.0)),
        i32(floor(frame_in_frac.z * 3.0)),
    ), vec3<i32>(0), vec3<i32>(2));
    s_cell[0] = pack_cell(cell0);

    // Cell center: absolute, but only updated via exact integer-step
    // shifts after this initial computation. No drift accumulation.
    var cur_lon_center = slab_lon_lo + (f32(cell0.x) + 0.5) * cur_lon_step;
    var cur_r_center   = slab_r_lo   + (f32(cell0.y) + 0.5) * cur_r_step;
    var cur_lat_center = slab_lat_lo + (f32(cell0.z) + 0.5) * cur_lat_step;

    var iters: u32 = 0u;
    loop {
        if iters > 256u { break; }
        iters = iters + 1u;
        if t > t_exit { break; }

        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            // OOB at depth d: ray exited depth-d's frame. The exit axis
            // tells us which face of the parent's cell (depth d-1) we
            // crossed; we pop and advance the parent's cell by ±1 on
            // that axis.
            if depth == 0u { break; }
            let oob_x = select(0, select(1, -1, cell.x < 0), cell.x < 0 || cell.x > 2);
            let oob_y = select(0, select(1, -1, cell.y < 0), cell.y < 0 || cell.y > 2);
            let oob_z = select(0, select(1, -1, cell.z < 0), cell.z < 0 || cell.z > 2);
            let old_step_lon = cur_lon_step;
            let old_step_lat = cur_lat_step;
            let old_step_r   = cur_r_step;
            depth = depth - 1u;
            cur_lon_step = old_step_lon * 3.0;
            cur_lat_step = old_step_lat * 3.0;
            cur_r_step   = old_step_r   * 3.0;
            let popped = unpack_cell(s_cell[depth]);
            // popped is the cell at parent depth d that contained the
            // (d+1) frame we just exited.
            //
            // Pre-pop, cur_*_center is the (d+1) cell's center, which
            // = (d+1)_frame_center + (cell - 1) * old_step. (`cell`
            // can be OOB, so its center is conceptually outside the
            // frame.) (d+1)_frame_center IS the popped cell's center at
            // depth d, so:
            //   parent_neighbor_center = (d+1)_frame_center + oob * d_step
            //                          = cur_*_center - (cell - 1) * old_step
            //                            + oob * d_step
            //
            // Note: uses `cell` (the (d+1) cell index, possibly OOB),
            // NOT `popped` (the parent cell index).
            cur_lon_center = cur_lon_center
                - (f32(cell.x) - 1.0) * old_step_lon
                + f32(oob_x) * cur_lon_step;
            cur_r_center = cur_r_center
                - (f32(cell.y) - 1.0) * old_step_r
                + f32(oob_y) * cur_r_step;
            cur_lat_center = cur_lat_center
                - (f32(cell.z) - 1.0) * old_step_lat
                + f32(oob_z) * cur_lat_step;
            // Advance parent's cell index by the OOB direction.
            let new_cell_p = vec3<i32>(
                popped.x + oob_x,
                popped.y + oob_y,
                popped.z + oob_z,
            );
            s_cell[depth] = pack_cell(new_cell_p);
            // frame_in_frac on pop: `(popped + child_frac) / 3`. Then
            // on the advanced axis(es), snap to precise face fraction.
            frame_in_frac = (vec3<f32>(popped) + frame_in_frac) / 3.0;
            if oob_x < 0 { frame_in_frac.x = f32(new_cell_p.x + 1) / 3.0; }
            if oob_x > 0 { frame_in_frac.x = f32(new_cell_p.x) / 3.0; }
            if oob_y < 0 { frame_in_frac.y = f32(new_cell_p.y + 1) / 3.0; }
            if oob_y > 0 { frame_in_frac.y = f32(new_cell_p.y) / 3.0; }
            if oob_z < 0 { frame_in_frac.z = f32(new_cell_p.z + 1) / 3.0; }
            if oob_z > 0 { frame_in_frac.z = f32(new_cell_p.z) / 3.0; }
            continue;
        }

        let cell_f = vec3<f32>(cell);
        // Precision-stable in-cell fraction. Used for bevel.
        let in_cell = clamp(frame_in_frac * 3.0 - cell_f, vec3<f32>(0.0), vec3<f32>(1.0));

        // Cell bounds via center ± half_step. No accumulating drift.
        let half_lon = cur_lon_step * 0.5;
        let half_lat = cur_lat_step * 0.5;
        let half_r   = cur_r_step * 0.5;
        let lon_lo = cur_lon_center - half_lon;
        let lon_hi = cur_lon_center + half_lon;
        let r_lo   = cur_r_center   - half_r;
        let r_hi   = cur_r_center   + half_r;
        let lat_lo = cur_lat_center - half_lat;
        let lat_hi = cur_lat_center + half_lat;

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;

        // t to all 6 cell faces. t_next = smallest > current t.
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

        // Helper: advance cell + center + frame_in_frac via face-
        // crossing detection. Used by every "skip this cell" branch.
        // Inlined as a closure-like block since WGSL has no closures.
        // (Each branch repeats the same 30 lines with a `continue`.)

        if (occ & bit) == 0u {
            // Empty slot. Step cell index ±1 on the crossed axis
            // (precision-stable integer math).
            var step_x: i32 = 0;
            var step_y: i32 = 0;
            var step_z: i32 = 0;
            if t_next == t_lon_lo { step_x = -1; }
            else if t_next == t_lon_hi { step_x = 1; }
            else if t_next == t_r_lo { step_y = -1; }
            else if t_next == t_r_hi { step_y = 1; }
            else if t_next == t_lat_lo { step_z = -1; }
            else if t_next == t_lat_hi { step_z = 1; }
            let new_cell = vec3<i32>(cell.x + step_x, cell.y + step_y, cell.z + step_z);
            s_cell[depth] = pack_cell(new_cell);
            cur_lon_center = cur_lon_center + f32(step_x) * cur_lon_step;
            cur_r_center   = cur_r_center   + f32(step_y) * cur_r_step;
            cur_lat_center = cur_lat_center + f32(step_z) * cur_lat_step;
            // For non-crossed axes the ray's position changes continuously
            // over delta_t; update frame_in_frac differentially via the
            // analytic d{lon,r,lat}/dt rates at the current ray point.
            // This is precise (rates are O(1), delta_t is small but
            // well-conditioned). Crossed axis snaps below.
            let delta_t = t_next - t;
            let pos_now = ray_origin + ray_dir * t;
            let q = pos_now - cs_center;
            let r_now = max(length(q), 1e-9);
            let r2_xz = max(q.x * q.x + q.z * q.z, 1e-12);
            let rate_lon = (q.x * ray_dir.z - q.z * ray_dir.x) / r2_xz;
            let rate_r = dot(q, ray_dir) / r_now;
            let arg_lat = clamp(q.y / r_now, -0.99999, 0.99999);
            let darg_dt = ray_dir.y / r_now - q.y * dot(q, ray_dir) / (r_now * r_now * r_now);
            let rate_lat = darg_dt / sqrt(max(1.0 - arg_lat * arg_lat, 1e-12));
            if step_x == 0 { frame_in_frac.x = frame_in_frac.x + rate_lon * delta_t / (3.0 * cur_lon_step); }
            if step_y == 0 { frame_in_frac.y = frame_in_frac.y + rate_r   * delta_t / (3.0 * cur_r_step); }
            if step_z == 0 { frame_in_frac.z = frame_in_frac.z + rate_lat * delta_t / (3.0 * cur_lat_step); }
            // Snap crossed axis frame_in_frac to the new cell's near face.
            if step_x < 0 { frame_in_frac.x = f32(new_cell.x + 1) / 3.0; }
            if step_x > 0 { frame_in_frac.x = f32(new_cell.x)     / 3.0; }
            if step_y < 0 { frame_in_frac.y = f32(new_cell.y + 1) / 3.0; }
            if step_y > 0 { frame_in_frac.y = f32(new_cell.y)     / 3.0; }
            if step_z < 0 { frame_in_frac.z = f32(new_cell.z + 1) / 3.0; }
            if step_z > 0 { frame_in_frac.z = f32(new_cell.z)     / 3.0; }
            t = t_next + max(cur_r_step * 1e-2, abs(t_next) * 2e-7);
            if t > t_exit { break; }
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
            //
            // Axis-convention swizzle: descent's frame_in_frac is in
            // (lon, r, lat) order to match the slab-tree slot layout
            // (cell.y = r, cell.z = lat). make_sphere_hit takes
            // (lon, lat, r) — swap y↔z.
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h,
                vec3<f32>(in_cell.x, in_cell.z, in_cell.y),
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        if tag != 2u {
            // EntityRef etc. — treat as empty for sphere subtree DDA.
            // Same advance shape as the empty-slot branch.
            var step_x: i32 = 0;
            var step_y: i32 = 0;
            var step_z: i32 = 0;
            if t_next == t_lon_lo { step_x = -1; }
            else if t_next == t_lon_hi { step_x = 1; }
            else if t_next == t_r_lo { step_y = -1; }
            else if t_next == t_r_hi { step_y = 1; }
            else if t_next == t_lat_lo { step_z = -1; }
            else if t_next == t_lat_hi { step_z = 1; }
            let new_cell = vec3<i32>(cell.x + step_x, cell.y + step_y, cell.z + step_z);
            s_cell[depth] = pack_cell(new_cell);
            cur_lon_center = cur_lon_center + f32(step_x) * cur_lon_step;
            cur_r_center   = cur_r_center   + f32(step_y) * cur_r_step;
            cur_lat_center = cur_lat_center + f32(step_z) * cur_lat_step;
            let delta_t = t_next - t;
            let pos_now = ray_origin + ray_dir * t;
            let q = pos_now - cs_center;
            let r_now = max(length(q), 1e-9);
            let r2_xz = max(q.x * q.x + q.z * q.z, 1e-12);
            let rate_lon = (q.x * ray_dir.z - q.z * ray_dir.x) / r2_xz;
            let rate_r = dot(q, ray_dir) / r_now;
            let arg_lat = clamp(q.y / r_now, -0.99999, 0.99999);
            let darg_dt = ray_dir.y / r_now - q.y * dot(q, ray_dir) / (r_now * r_now * r_now);
            let rate_lat = darg_dt / sqrt(max(1.0 - arg_lat * arg_lat, 1e-12));
            if step_x == 0 { frame_in_frac.x = frame_in_frac.x + rate_lon * delta_t / (3.0 * cur_lon_step); }
            if step_y == 0 { frame_in_frac.y = frame_in_frac.y + rate_r   * delta_t / (3.0 * cur_r_step); }
            if step_z == 0 { frame_in_frac.z = frame_in_frac.z + rate_lat * delta_t / (3.0 * cur_lat_step); }
            if step_x < 0 { frame_in_frac.x = f32(new_cell.x + 1) / 3.0; }
            if step_x > 0 { frame_in_frac.x = f32(new_cell.x)     / 3.0; }
            if step_y < 0 { frame_in_frac.y = f32(new_cell.y + 1) / 3.0; }
            if step_y > 0 { frame_in_frac.y = f32(new_cell.y)     / 3.0; }
            if step_z < 0 { frame_in_frac.z = f32(new_cell.z + 1) / 3.0; }
            if step_z > 0 { frame_in_frac.z = f32(new_cell.z)     / 3.0; }
            t = t_next + max(cur_r_step * 1e-2, abs(t_next) * 2e-7);
            if t > t_exit { break; }
            continue;
        }

        // tag == 2u: non-uniform Node.
        if block_type == 0xFFFEu {
            // Subtree empty per `representative_block` — skip whole
            // cell. Same advance shape.
            var step_x: i32 = 0;
            var step_y: i32 = 0;
            var step_z: i32 = 0;
            if t_next == t_lon_lo { step_x = -1; }
            else if t_next == t_lon_hi { step_x = 1; }
            else if t_next == t_r_lo { step_y = -1; }
            else if t_next == t_r_hi { step_y = 1; }
            else if t_next == t_lat_lo { step_z = -1; }
            else if t_next == t_lat_hi { step_z = 1; }
            let new_cell = vec3<i32>(cell.x + step_x, cell.y + step_y, cell.z + step_z);
            s_cell[depth] = pack_cell(new_cell);
            cur_lon_center = cur_lon_center + f32(step_x) * cur_lon_step;
            cur_r_center   = cur_r_center   + f32(step_y) * cur_r_step;
            cur_lat_center = cur_lat_center + f32(step_z) * cur_lat_step;
            let delta_t = t_next - t;
            let pos_now = ray_origin + ray_dir * t;
            let q = pos_now - cs_center;
            let r_now = max(length(q), 1e-9);
            let r2_xz = max(q.x * q.x + q.z * q.z, 1e-12);
            let rate_lon = (q.x * ray_dir.z - q.z * ray_dir.x) / r2_xz;
            let rate_r = dot(q, ray_dir) / r_now;
            let arg_lat = clamp(q.y / r_now, -0.99999, 0.99999);
            let darg_dt = ray_dir.y / r_now - q.y * dot(q, ray_dir) / (r_now * r_now * r_now);
            let rate_lat = darg_dt / sqrt(max(1.0 - arg_lat * arg_lat, 1e-12));
            if step_x == 0 { frame_in_frac.x = frame_in_frac.x + rate_lon * delta_t / (3.0 * cur_lon_step); }
            if step_y == 0 { frame_in_frac.y = frame_in_frac.y + rate_r   * delta_t / (3.0 * cur_r_step); }
            if step_z == 0 { frame_in_frac.z = frame_in_frac.z + rate_lat * delta_t / (3.0 * cur_lat_step); }
            if step_x < 0 { frame_in_frac.x = f32(new_cell.x + 1) / 3.0; }
            if step_x > 0 { frame_in_frac.x = f32(new_cell.x)     / 3.0; }
            if step_y < 0 { frame_in_frac.y = f32(new_cell.y + 1) / 3.0; }
            if step_y > 0 { frame_in_frac.y = f32(new_cell.y)     / 3.0; }
            if step_z < 0 { frame_in_frac.z = f32(new_cell.z + 1) / 3.0; }
            if step_z > 0 { frame_in_frac.z = f32(new_cell.z)     / 3.0; }
            t = t_next + max(cur_r_step * 1e-2, abs(t_next) * 2e-7);
            if t > t_exit { break; }
            continue;
        }

        // Stack ceiling — same as Cartesian's MAX_STACK_DEPTH gate.
        let at_max = depth + 1u >= SPHERE_DESCENT_DEPTH;
        if at_max {
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - cs_center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            return make_sphere_hit(
                pos_h, n_h, t, inv_norm, block_type,
                r_h, lat_h,
                vec3<f32>(in_cell.x, in_cell.z, in_cell.y),
                cur_lon_step, cur_lat_step, cur_r_step,
            );
        }

        // Push child frame.
        let child_idx = tree[child_base + 1u];
        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_lon_step = cur_lon_step / 3.0;
        cur_lat_step = cur_lat_step / 3.0;
        cur_r_step   = cur_r_step   / 3.0;

        // Precision-stable: child's in-frame frac = parent's in_cell.
        frame_in_frac = in_cell;

        // Pick entry sub-cell at depth d+1 from the (now-precise) frac.
        let cell_c = clamp(vec3<i32>(
            i32(floor(frame_in_frac.x * 3.0)),
            i32(floor(frame_in_frac.y * 3.0)),
            i32(floor(frame_in_frac.z * 3.0)),
        ), vec3<i32>(0), vec3<i32>(2));
        s_cell[depth] = pack_cell(cell_c);

        // New cell center: parent (the cell we just pushed into) was
        // at cur_*_center. Shift to new sub-cell:
        //   new_center = parent_center + (cell_c - 1) * new_step
        cur_lon_center = cur_lon_center + (f32(cell_c.x) - 1.0) * cur_lon_step;
        cur_r_center   = cur_r_center   + (f32(cell_c.y) - 1.0) * cur_r_step;
        cur_lat_center = cur_lat_center + (f32(cell_c.z) - 1.0) * cur_lat_step;
    }

    return result;
}
