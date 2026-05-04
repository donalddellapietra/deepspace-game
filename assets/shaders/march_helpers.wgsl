
// Cell-packing helpers. Cell coords at each depth range -1..=3
// (legal 0..=2 plus ±1 over-step to trigger pop). Pack +1-shifted
// into 3 bits per axis = 9 bits per u32. Shrinks the per-thread
// `s_cell` stack from 96 B (vec3<i32>×8) to 32 B (u32×8), targeting
// the Fragment Occupancy register cliff measured at 9.7% on
// Jerusalem nucleus (rule: <25% = register pressure).
fn pack_cell(c: vec3<i32>) -> u32 {
    let ux = u32(c.x + 1) & 7u;
    let uy = u32(c.y + 1) & 7u;
    let uz = u32(c.z + 1) & 7u;
    return ux | (uy << 3u) | (uz << 6u);
}

fn unpack_cell(p: u32) -> vec3<i32> {
    return vec3<i32>(
        i32(p & 7u) - 1,
        i32((p >> 3u) & 7u) - 1,
        i32((p >> 6u) & 7u) - 1,
    );
}

// Conservative 27-bit "path mask" — the tensor product of per-axis
// 3-bit masks of cells reachable from `entry_cell` moving in `step`
// direction. Over-approximates the actual ray path (any axis-wise
// reachable cell triple, not only the specific 3D path the ray
// traces). Safe for occupancy-intersection culling: if the full
// superset misses all occupied slots, the actual path certainly
// does. Used for instrumentation only right now — does not affect
// traversal.
fn path_mask_conservative(entry_cell: vec3<i32>, step: vec3<i32>) -> u32 {
    let ec = vec3<u32>(
        u32(clamp(entry_cell.x, 0, 2)),
        u32(clamp(entry_cell.y, 0, 2)),
        u32(clamp(entry_cell.z, 0, 2)),
    );
    // Per-axis 3-bit mask. step > 0: bits [ec..2]; step < 0: bits
    // [0..ec]. step is always ±1 in march_cartesian (non-zero).
    let mx: u32 = select((1u << (ec.x + 1u)) - 1u, (7u << ec.x) & 7u, step.x > 0);
    let my: u32 = select((1u << (ec.y + 1u)) - 1u, (7u << ec.y) & 7u, step.y > 0);
    let mz: u32 = select((1u << (ec.z + 1u)) - 1u, (7u << ec.z) & 7u, step.z > 0);
    // Smear each 3-bit axis mask into its 27-bit "axis active"
    // pattern. x repeats stride-3 (bits 0,3,6,...); y expands to a
    // 9-bit xy-plane then repeats stride-9; z gates whole 9-bit
    // planes. Closed-form — no loops, no lookups.
    let x_active: u32 = mx * 0x01249249u;
    let y_9: u32 = ((my & 1u) * 0x007u)
                 | (((my >> 1u) & 1u) * 0x038u)
                 | (((my >> 2u) & 1u) * 0x1C0u);
    let y_active: u32 = y_9 * 0x00040201u;
    let z_active: u32 = ((mz & 1u) * 0x000001FFu)
                     | (((mz >> 1u) & 1u) * 0x0003FE00u)
                     | (((mz >> 2u) & 1u) * 0x07FC0000u);
    return x_active & y_active & z_active;
}

// Entity subtree walker. Cartesian DDA walking a standalone voxel
// subtree — no sphere/face/ribbon dispatch, no AABB side-buffer,
// no beam-prepass coupling. Called from `march_cartesian`'s tag==3
// branch after the ray has been transformed into the entity's
// `[0, 3)³` local frame. WGSL's no-recursion rule forces this to
// be a separate function rather than a re-entrant call to
// `march_cartesian`.
//
// On hit, returned `HitResult.t / cell_min / cell_size` are in
// entity-local units; the caller scales back to world via the
// entity's bbox size.

/// TangentBlock boundary transform — symmetric counterpart to
/// `TbBoundary` in `world/gpu/types.rs`. The four helpers each read
/// `(rot_col0.xyz, rot_col1.xyz, rot_col2.xyz, rot_col0.w)` from
/// the child's `node_kinds[]` entry and apply the unified rule.
///
/// Descent (parent → TB-storage frame, pivot 1.5 in `[0, 3)³`):
///     `p' = R^T · (p − pivot) / tb_scale + pivot`
///     `d' = R^T · d / tb_scale`
/// Pop (TB-storage → parent), the inverse:
///     `p' = R · (p − pivot) · tb_scale + pivot`
///     `d' = R · d · tb_scale`
///
/// `R` is column-major: `rc{0,1,2}` are columns; `R^T·v` is built
/// via `dot(rcN, v)` (rows of `R^T`), and `R·v` via the column
/// linear combination `rc0·v.x + rc1·v.y + rc2·v.z`.
fn tb_enter_point(child_idx: u32, p: vec3<f32>, pivot: f32) -> vec3<f32> {
    let rc0 = node_kinds[child_idx].rot_col0.xyz;
    let rc1 = node_kinds[child_idx].rot_col1.xyz;
    let rc2 = node_kinds[child_idx].rot_col2.xyz;
    let s = node_kinds[child_idx].rot_col0.w;
    let c = p - vec3<f32>(pivot);
    return vec3<f32>(dot(rc0, c), dot(rc1, c), dot(rc2, c)) / s + vec3<f32>(pivot);
}

fn tb_enter_dir(child_idx: u32, d: vec3<f32>) -> vec3<f32> {
    let rc0 = node_kinds[child_idx].rot_col0.xyz;
    let rc1 = node_kinds[child_idx].rot_col1.xyz;
    let rc2 = node_kinds[child_idx].rot_col2.xyz;
    let s = node_kinds[child_idx].rot_col0.w;
    return vec3<f32>(dot(rc0, d), dot(rc1, d), dot(rc2, d)) / s;
}

fn tb_exit_point(child_idx: u32, p: vec3<f32>, pivot: f32) -> vec3<f32> {
    let rc0 = node_kinds[child_idx].rot_col0.xyz;
    let rc1 = node_kinds[child_idx].rot_col1.xyz;
    let rc2 = node_kinds[child_idx].rot_col2.xyz;
    let s = node_kinds[child_idx].rot_col0.w;
    let c = (p - vec3<f32>(pivot)) * s;
    return rc0 * c.x + rc1 * c.y + rc2 * c.z + vec3<f32>(pivot);
}

fn tb_exit_dir(child_idx: u32, d: vec3<f32>) -> vec3<f32> {
    let rc0 = node_kinds[child_idx].rot_col0.xyz;
    let rc1 = node_kinds[child_idx].rot_col1.xyz;
    let rc2 = node_kinds[child_idx].rot_col2.xyz;
    let s = node_kinds[child_idx].rot_col0.w;
    let v = d * s;
    return rc0 * v.x + rc1 * v.y + rc2 * v.z;
}

/// Map an axis-aligned face normal (single ±1 component) to a face id
/// matching the cube-face convention used by the CPU debug printer:
/// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z, 7=unknown / non-axis.
fn normal_to_face(n: vec3<f32>) -> u32 {
    if n.x >  0.5 { return 0u; }
    if n.x < -0.5 { return 1u; }
    if n.y >  0.5 { return 2u; }
    if n.y < -0.5 { return 3u; }
    if n.z >  0.5 { return 4u; }
    if n.z < -0.5 { return 5u; }
    return 7u;
}

/// Walker-probe writer. Called from `march_cartesian`'s hit / miss
/// return points. Gated on `uniforms.probe_pixel.z != 0u` AND the
/// current pixel matching `uniforms.probe_pixel.xy`. Non-atomic
/// stores are safe because at most one fragment invocation in the
/// grid passes both gates.
fn write_walker_probe(
    hit_flag: u32,
    steps: u32,
    final_depth: u32,
    cell: vec3<i32>,
    cur_node_origin: vec3<f32>,
    cur_cell_size: f32,
    hit_t: f32,
    normal: vec3<f32>,
    content_flag: u32,
    curvature_offset: f32,
) {
    if uniforms.probe_pixel.z == 0u { return; }
    if current_pixel.x != uniforms.probe_pixel.x { return; }
    if current_pixel.y != uniforms.probe_pixel.y { return; }
    walker_probe.hit_flag = hit_flag;
    walker_probe.ray_steps = steps;
    walker_probe.final_depth = final_depth;
    let cx = u32(clamp(cell.x + 1, 0, 7)) & 7u;
    let cy = u32(clamp(cell.y + 1, 0, 7)) & 7u;
    let cz = u32(clamp(cell.z + 1, 0, 7)) & 7u;
    walker_probe.terminal_cell = cx | (cy << 2u) | (cz << 4u);
    walker_probe.cur_node_origin_x_bits = bitcast<u32>(cur_node_origin.x);
    walker_probe.cur_node_origin_y_bits = bitcast<u32>(cur_node_origin.y);
    walker_probe.cur_node_origin_z_bits = bitcast<u32>(cur_node_origin.z);
    walker_probe.cur_cell_size_bits = bitcast<u32>(cur_cell_size);
    walker_probe.hit_t_bits = bitcast<u32>(hit_t);
    walker_probe.hit_face = normal_to_face(normal);
    walker_probe.content_flag = content_flag;
    walker_probe.curvature_offset_bits = bitcast<u32>(curvature_offset);
}
