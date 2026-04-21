// Ray-primitive intersections + small pure-math helpers.

struct BoxHit { t_enter: f32, t_exit: f32, }

fn ray_plane_t(origin: vec3<f32>, dir: vec3<f32>,
               through: vec3<f32>, plane_n: vec3<f32>) -> f32 {
    let denom = dot(dir, plane_n);
    if abs(denom) < 1e-12 { return -1.0; }
    return -dot(origin - through, plane_n) / denom;
}

// Numerical-Recipes stable ray-sphere intersection.
fn ray_sphere_after(origin: vec3<f32>, dir: vec3<f32>,
                    center: vec3<f32>, radius: f32, after: f32) -> f32 {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let s = select(-1.0, 1.0, b >= 0.0);
    let q = -b - s * sq;
    if abs(q) < 1e-30 { return -1.0; }
    let t0 = q;
    let t1 = c / q;
    let t_lo = min(t0, t1);
    let t_hi = max(t0, t1);
    if t_lo > after { return t_lo; }
    if t_hi > after { return t_hi; }
    return -1.0;
}

fn ray_box(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> BoxHit {
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_lo = min(t1, t2);
    let t_hi = max(t1, t2);
    return BoxHit(
        max(max(t_lo.x, t_lo.y), t_lo.z),
        min(min(t_hi.x, t_hi.y), t_hi.z),
    );
}

fn pow3_u(exp: u32) -> f32 {
    var scale = 1.0;
    for (var i: u32 = 0u; i < exp; i = i + 1u) {
        scale = scale * 3.0;
    }
    return scale;
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

// Cube-face UV lookup: picks the 2D face plane matching the
// dominant component of `normal` and returns the other two
// components of `local` as (u, v). Used by the bevel shader and any
// other per-face cell-shading helper.
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

// Soft-edge bevel for cube-face cells. Returns 0.0 at the face
// edges and 1.0 in the face interior, smoothstepped over the
// outer 2–14 % of the face in each axis.
fn cube_face_bevel(local: vec3<f32>, normal: vec3<f32>) -> f32 {
    let uv = face_uv_for_normal(local, normal);
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.02, 0.14, edge);
}

// Bergamo forward cube→sphere map F: [-1, 1]³ → closed unit ball.
// `A_a(b, c) = 1 − b²/2 − c²/2 + b²c²/3` with cyclic permutation
// `(a, b, c) ∈ {(x,y,z), (y,z,x), (z,x,y)}`.
fn bergamo_map(p: vec3<f32>) -> vec3<f32> {
    let ax = max(0.0, 1.0 - 0.5 * p.y * p.y - 0.5 * p.z * p.z + p.y * p.y * p.z * p.z / 3.0);
    let ay = max(0.0, 1.0 - 0.5 * p.z * p.z - 0.5 * p.x * p.x + p.z * p.z * p.x * p.x / 3.0);
    let az = max(0.0, 1.0 - 0.5 * p.x * p.x - 0.5 * p.y * p.y + p.x * p.x * p.y * p.y / 3.0);
    return vec3<f32>(p.x * sqrt(ax), p.y * sqrt(ay), p.z * sqrt(az));
}

// Jacobian of F at `p`. Returned in WGSL's native column-major form:
// column k = `∂F/∂p_k`. `mat3x3<f32> * vec3<f32>` then gives J·v
// (the correct application for direction transforms like lighting).
//
// Diagonal: `∂s_a/∂p_a = √A_a`. Off-diagonal cyclic pattern:
// `∂s_a/∂p_b = p_a · p_b · (2p_c² − 3) / (6 · √A_a)` where
// `(a, b, c)` cycles through `(x,y,z), (y,z,x), (z,x,y)`.
fn bergamo_jacobian(p: vec3<f32>) -> mat3x3<f32> {
    let x = p.x;
    let y = p.y;
    let z = p.z;
    let ax = max(1e-12, 1.0 - 0.5 * y * y - 0.5 * z * z + y * y * z * z / 3.0);
    let ay = max(1e-12, 1.0 - 0.5 * z * z - 0.5 * x * x + z * z * x * x / 3.0);
    let az = max(1e-12, 1.0 - 0.5 * x * x - 0.5 * y * y + x * x * y * y / 3.0);
    let sax = sqrt(ax);
    let say = sqrt(ay);
    let saz = sqrt(az);
    // Rows of the row-major Jacobian.
    let r0 = vec3<f32>(sax,
                       x * y * (2.0 * z * z - 3.0) / (6.0 * sax),
                       x * z * (2.0 * y * y - 3.0) / (6.0 * sax));
    let r1 = vec3<f32>(y * x * (2.0 * z * z - 3.0) / (6.0 * say),
                       say,
                       y * z * (2.0 * x * x - 3.0) / (6.0 * say));
    let r2 = vec3<f32>(z * x * (2.0 * y * y - 3.0) / (6.0 * saz),
                       z * y * (2.0 * x * x - 3.0) / (6.0 * saz),
                       saz);
    // Column-major constructor: column k gets (r0[k], r1[k], r2[k]).
    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x),
        vec3<f32>(r0.y, r1.y, r2.y),
        vec3<f32>(r0.z, r1.z, r2.z),
    );
}

// Inverse of a column-major 3×3 matrix via adjugate ÷ determinant.
// Returned column-major so `M⁻¹ · v` is the expected inverse op.
fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0][0]; let d = m[0][1]; let g = m[0][2];
    let b = m[1][0]; let e = m[1][1]; let h = m[1][2];
    let c = m[2][0]; let f = m[2][1]; let i = m[2][2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;
    // Row-major M⁻¹ rows:
    let r0 = vec3<f32>( (e * i - f * h), -(b * i - c * h),  (b * f - c * e)) * inv_det;
    let r1 = vec3<f32>(-(d * i - f * g),  (a * i - c * g), -(a * f - c * d)) * inv_det;
    let r2 = vec3<f32>( (d * h - e * g), -(a * h - b * g),  (a * e - b * d)) * inv_det;
    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x),
        vec3<f32>(r0.y, r1.y, r2.y),
        vec3<f32>(r0.z, r1.z, r2.z),
    );
}

// Inverse-transpose of a column-major 3×3. Column k of M⁻ᵀ = row k
// of M⁻¹. Used to transform body-frame surface normals into
// world-frame lighting normals.
fn mat3_inverse_transpose(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0][0]; let d = m[0][1]; let g = m[0][2];
    let b = m[1][0]; let e = m[1][1]; let h = m[1][2];
    let c = m[2][0]; let f = m[2][1]; let i = m[2][2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;
    let r0 = vec3<f32>( (e * i - f * h), -(b * i - c * h),  (b * f - c * e)) * inv_det;
    let r1 = vec3<f32>(-(d * i - f * g),  (a * i - c * g), -(a * f - c * d)) * inv_det;
    let r2 = vec3<f32>( (d * h - e * g), -(a * h - b * g),  (a * e - b * d)) * inv_det;
    // Column k of (M⁻¹)ᵀ == row k of M⁻¹.
    return mat3x3<f32>(r0, r1, r2);
}

// Solve F(body) = q for body, via Newton iteration on the residual
// `F(body) − q`. `q` should lie on or near the unit sphere; the
// returned `body` lies on or near the cube surface.
//
// Initial guess: project `q` onto the cube surface by scaling so
// that `max(|body|) = 1`. At face centers and corners this is
// already the exact answer; at intermediate points 2-3 iterations
// drive the residual under `1e-5`.
fn bergamo_inverse(q: vec3<f32>) -> vec3<f32> {
    let abs_q = abs(q);
    let max_comp = max(max(abs_q.x, abs_q.y), abs_q.z);
    var body = q * (1.0 / max(max_comp, 1e-6));
    for (var i = 0u; i < 6u; i = i + 1u) {
        let fp = bergamo_map(body);
        let residual = fp - q;
        if dot(residual, residual) < 1e-10 {
            break;
        }
        let j = bergamo_jacobian(body);
        let j_inv = mat3_inverse(j);
        body = body - j_inv * residual;
    }
    return body;
}

// Walk the packed tree from `root_bfs` down `depth` levels to decide
// whether the body-voxel at integer grid `(gx, gy, gz)` (each in
// `[0, 3^depth)`) is solid stone or an empty carved-out cell.
//
// At each level, the grid's current slot digit (base-3, most-
// significant first) is decoded, combined into a 27-slot index, and
// looked up in the node's occupancy mask. Hits can terminate early
// at any tag=1 (flattened Block) cell — the uniform-stone subtree is
// flat at pack time, so that's the common case. tag=2 descends
// further. tag=0 (empty slot) = the voxel has been carved, return
// false so the caller renders a hole.
//
// Returns `true` if the voxel is solid. Returns `false` on empty or
// any unexpected tag / out-of-bounds BFS.
fn sphere_body_voxel_solid(
    root_bfs: u32,
    grid: vec3<i32>,
    depth: u32,
) -> bool {
    var cur_bfs = root_bfs;
    // Starting shift: 3^(depth-1). For depth=3, shift=9 picks the
    // most-significant base-3 digit.
    var shift: i32 = 1;
    for (var d: u32 = 1u; d < depth; d = d + 1u) {
        shift = shift * 3;
    }

    for (var d: u32 = 0u; d < depth; d = d + 1u) {
        let sx = (grid.x / shift) % 3;
        let sy = (grid.y / shift) % 3;
        let sz = (grid.z / shift) % 3;
        let slot = u32(sx + sy * 3 + sz * 9);

        let header_off = node_offsets[cur_bfs];
        let occ = tree[header_off];
        let first_child = tree[header_off + 1u];
        let slot_bit = 1u << slot;
        if (occ & slot_bit) == 0u {
            return false;
        }
        let rank = countOneBits(occ & (slot_bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        if tag == 1u {
            // Flattened leaf. Empty sentinel (0xFFFE) = hole; anything
            // else = real block.
            let bt = (packed >> 8u) & 0xFFFFu;
            return bt != 0xFFFEu;
        }
        if tag != 2u {
            // Unknown tag (tag=0 never appears; tag=3 = entity cell
            // shouldn't live inside a SphereBody).
            return false;
        }
        cur_bfs = tree[child_base + 1u];
        shift = shift / 3;
    }

    // Walked the full depth and still on tag=2. The deeper subtree is
    // non-empty (we'd have bailed on tag=1 with the empty sentinel),
    // so the voxel is solid.
    return true;
}

// Analytic ray-vs-sphere-body hit. Runs ray-sphere against the
// inscribed sphere at `(sphere_center, sphere_radius)` (render-frame
// local coords); on hit, inverts Bergamo's F to find the cube-surface
// point in body space, then returns a HitResult with the appropriate
// world-space radial normal and a body-space-grid bevel baked into
// the color. Returns `hit = false` if the ray misses the sphere or
// enters it from behind the camera.
//
// `n_per_axis` sets the body-space voxel grid resolution per face.
// `rep_block` selects the palette entry for the material.
//
// This is the SINGLE sphere-body rendering path — it's called from
// `march_cartesian`'s tag=2 branch when a SphereBody cell is
// encountered (external view) AND from `march` entry when the
// camera's render frame is already inside a SphereBody subtree
// (surface view). No matter how the ray arrives at the planet, the
// visual is identical — no transition discontinuity.
fn analytic_sphere_hit(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
    rep_block: u32,
    n_per_axis: f32,
    body_root_bfs: u32,
    body_depth: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.is_sphere = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    let dir_len = length(ray_dir);
    if dir_len < 1e-8 { return result; }
    let unit_dir = ray_dir * (1.0 / dir_len);
    let t_unit = ray_sphere_after(
        ray_origin, unit_dir, sphere_center, sphere_radius, 0.0,
    );
    if t_unit <= 0.0 { return result; }

    let t_sphere = t_unit / dir_len;
    let world_hit = ray_origin + ray_dir * t_sphere;
    let q = (world_hit - sphere_center) * (1.0 / sphere_radius);
    let body_hit = bergamo_inverse(q);

    let abs_body = abs(body_hit);
    var face_axis: u32 = 0u;
    var face_mag: f32 = abs_body.x;
    if abs_body.y > face_mag { face_axis = 1u; face_mag = abs_body.y; }
    if abs_body.z > face_mag { face_axis = 2u; face_mag = abs_body.z; }
    var n_body = vec3<f32>(0.0);
    var face_uv = vec2<f32>(0.0);
    if face_axis == 0u {
        n_body = vec3<f32>(select(-1.0, 1.0, body_hit.x > 0.0), 0.0, 0.0);
        face_uv = body_hit.yz;
    } else if face_axis == 1u {
        n_body = vec3<f32>(0.0, select(-1.0, 1.0, body_hit.y > 0.0), 0.0);
        face_uv = body_hit.xz;
    } else {
        n_body = vec3<f32>(0.0, 0.0, select(-1.0, 1.0, body_hit.z > 0.0));
        face_uv = body_hit.xy;
    }

    let j = bergamo_jacobian(body_hit);
    let j_inv_t = mat3_inverse_transpose(j);
    let world_n = normalize(j_inv_t * n_body);

    let uv_grid = (face_uv + vec2<f32>(1.0)) * (n_per_axis * 0.5);
    let local_uv = fract(uv_grid);
    let edge = min(min(local_uv.x, 1.0 - local_uv.x),
                   min(local_uv.y, 1.0 - local_uv.y));
    let bevel = smoothstep(0.02, 0.14, edge);

    // Resolve the body-grid integer index from `body_hit`. N voxels
    // per axis in body space means each axis maps `[-1, 1]` → `[0, N)`.
    // On the selected face the face axis is pinned to the ±1-most
    // voxel (N-1 or 0).
    let n_int = i32(n_per_axis);
    let grid_f = (body_hit + vec3<f32>(1.0)) * (n_per_axis * 0.5);
    var grid = vec3<i32>(
        clamp(i32(floor(grid_f.x)), 0, n_int - 1),
        clamp(i32(floor(grid_f.y)), 0, n_int - 1),
        clamp(i32(floor(grid_f.z)), 0, n_int - 1),
    );

    // Walk the body tree. If the voxel is solid we render the patch;
    // if the voxel has been carved (empty), return a miss so the
    // pixel shows the sky (for thin holes; deep carving needs a
    // curved-ray DDA that's still TODO).
    let solid = sphere_body_voxel_solid(body_root_bfs, grid, body_depth);
    if !solid {
        result.hit = false;
        return result;
    }

    result.hit = true;
    result.is_sphere = true;
    result.t = t_sphere;
    result.color = palette[rep_block].rgb * (0.7 + 0.3 * bevel);
    result.normal = world_n;
    result.cell_min = sphere_center;
    result.cell_size = sphere_radius;
    return result;
}

// Branchless argmin mask for the DDA min-side_dist selection.
// Returns a (0/1) vec3 where exactly one component is 1: the axis whose
// `side_dist` is smallest. Tie-break priority matches the original
// if/else if/else chain: x > y > z (z wins all-equal ties).
//
// The three `pick_*` bools are pairwise independent — compiler can
// issue the 4 compares in parallel, shortening the loop-carried
// dependency chain that a branching `if/else if` version forces.
fn min_axis_mask(sd: vec3<f32>) -> vec3<f32> {
    let pick_x = sd.x < sd.y && sd.x < sd.z;
    let pick_y = sd.x >= sd.y && sd.y < sd.z;
    let pick_z = !pick_x && !pick_y;
    return vec3<f32>(
        select(0.0, 1.0, pick_x),
        select(0.0, 1.0, pick_y),
        select(0.0, 1.0, pick_z),
    );
}
