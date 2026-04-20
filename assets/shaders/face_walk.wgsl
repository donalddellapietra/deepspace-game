#include "bindings.wgsl"
#include "tree.wgsl"
#include "face_math.wgsl"

// Walk a face subtree from a body node. Given a point (un, vn, rn)
// in face-normalized [0,1]³ coords, descends the body's 27-ary tree
// through the face-center slot, then through child subtrees keyed by
// (u_slot, v_slot, r_slot). Returns the terminal block + the cell's
// [0,1]³ bounds.
//
// Cell bounds are accumulated via Kahan compensation per axis so
// cumulative error stays at ~1 ULP regardless of depth. A naive
// accumulator ("lo += step_size * slot") would drift ~depth ULPs,
// visible as grid-line misalignment past depth ~14. Kahan keeps the
// walker precision-correct up to MAX_FACE_DEPTH.

struct FaceWalkResult {
    block: u32,
    depth: u32,
    u_lo: f32,
    v_lo: f32,
    r_lo: f32,
    size: f32, // 3^-depth; same on all axes
}

fn walk_face_subtree(
    body_node_idx: u32, face: u32,
    un_in: f32, vn_in: f32, rn_in: f32,
    depth_limit: u32,
) -> FaceWalkResult {
    var result: FaceWalkResult;
    result.block = 0u;
    result.depth = 1u;
    result.u_lo = 0.0;
    result.v_lo = 0.0;
    result.r_lo = 0.0;
    result.size = 1.0;

    let fs = face_slot(face);
    let body_h = node_offsets[body_node_idx];
    let body_occ = tree[body_h];
    let body_first = tree[body_h + 1u];
    let face_bit = 1u << fs;
    if (body_occ & face_bit) == 0u {
        return result;
    }
    let body_rank = countOneBits(body_occ & (face_bit - 1u));
    let face_base = body_first + body_rank * 2u;
    let face_packed = tree[face_base];
    let face_tag = face_packed & 0xFFu;
    if face_tag == 1u {
        result.block = (face_packed >> 8u) & 0xFFFFu;
        return result;
    }

    var node = tree[face_base + 1u];
    var un = clamp(un_in, 0.0, 0.9999999);
    var vn = clamp(vn_in, 0.0, 0.9999999);
    var rn = clamp(rn_in, 0.0, 0.9999999);

    // Kahan-compensated accumulators per axis.
    var u_sum: f32 = 0.0; var u_comp: f32 = 0.0;
    var v_sum: f32 = 0.0; var v_comp: f32 = 0.0;
    var r_sum: f32 = 0.0; var r_comp: f32 = 0.0;
    var size: f32 = 1.0;

    let limit = min(depth_limit, MAX_FACE_DEPTH);
    if limit <= 1u {
        let bt = (face_packed >> 8u) & 0xFFFFu;
        result.block = select(0u, bt, bt != 0xFFFEu);
        return result;
    }

    for (var d: u32 = 2u; d <= limit; d = d + 1u) {
        let us = min(u32(un * 3.0), 2u);
        let vs = min(u32(vn * 3.0), 2u);
        let rs = min(u32(rn * 3.0), 2u);
        let slot = rs * 9u + vs * 3u + us;

        let h = node_offsets[node];
        let occ = tree[h];
        let first = tree[h + 1u];
        let bit = 1u << slot;
        let occupied = (occ & bit) != 0u;
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first + rank * 2u;
        let packed = select(0u, tree[child_base], occupied);
        let tag = packed & 0xFFu;

        // Update bounds: this step's child in its parent contributes
        // (size/3) * slot to the lo-bound, and shrinks size by 3.
        let step_size = size * (1.0 / 3.0);
        let u_add = step_size * f32(us);
        let v_add = step_size * f32(vs);
        let r_add = step_size * f32(rs);

        let yu = u_add - u_comp;
        let tu = u_sum + yu;
        u_comp = (tu - u_sum) - yu;
        u_sum = tu;

        let yv = v_add - v_comp;
        let tv = v_sum + yv;
        v_comp = (tv - v_sum) - yv;
        v_sum = tv;

        let yr = r_add - r_comp;
        let tr = r_sum + yr;
        r_comp = (tr - r_sum) - yr;
        r_sum = tr;

        size = step_size;

        if tag == 0u || tag == 1u {
            result.block = select(0u, (packed >> 8u) & 0xFFFFu, tag == 1u);
            result.depth = d;
            result.u_lo = u_sum + u_comp;
            result.v_lo = v_sum + v_comp;
            result.r_lo = r_sum + r_comp;
            result.size = size;
            return result;
        }
        if d >= limit {
            let bt = (packed >> 8u) & 0xFFFFu;
            result.block = select(0u, bt, bt != 0xFFFEu);
            result.depth = d;
            result.u_lo = u_sum + u_comp;
            result.v_lo = v_sum + v_comp;
            result.r_lo = r_sum + r_comp;
            result.size = size;
            return result;
        }
        node = tree[child_base + 1u];
        un = un * 3.0 - f32(us);
        vn = vn * 3.0 - f32(vs);
        rn = rn * 3.0 - f32(rs);
    }

    result.depth = limit;
    result.u_lo = u_sum + u_comp;
    result.v_lo = v_sum + v_comp;
    result.r_lo = r_sum + r_comp;
    result.size = size;
    return result;
}
