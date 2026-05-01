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

// Cartesian-cell bevel: pick the two axes orthogonal to the hit
// normal, then darken cells near their cell-local boundary so each
// voxel reads as a discrete cube under flat lighting.
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

// Phase 3 REVISED A.3 — meridian (constant-longitude) plane through
// the planet's Y axis at longitude `lon_b`. The plane contains the
// Y axis and the direction (cos(lon_b), 0, sin(lon_b)); its normal
// is (-sin(lon_b), 0, cos(lon_b)). Plane passes through the sphere
// center.
//
// Returns the t > `after` for the ray's first crossing of this
// plane, or -1 if no valid hit. Caller must verify the hit point's
// longitude actually equals lon_b (vs lon_b + π on the opposite
// half-plane); for the cell DDA this is naturally enforced because
// successive boundary crossings keep the ray inside one cell at a
// time.
fn ray_meridian_t(oc: vec3<f32>, dir: vec3<f32>, lon_b: f32, after: f32) -> f32 {
    let n = vec3<f32>(-sin(lon_b), 0.0, cos(lon_b));
    let denom = dot(dir, n);
    if abs(denom) < 1e-12 { return -1.0; }
    let t = -dot(oc, n) / denom;
    if t > after { return t; }
    return -1.0;
}

// Phase 3 REVISED A.3 — parallel (constant-latitude) cone with apex
// at the sphere center, axis along +Y. Cone equation:
//   (x² + z²) · tan²(lat_b) − y² = 0
// Has two halves (upper and lower); we pick the half on the same
// side as `lat_b` (positive lat_b → y > 0, negative → y < 0).
//
// For lat_b ≈ 0 the cone degenerates to the y = 0 plane (equator);
// we handle that as a special case.
//
// Returns the t > `after` for the ray's first crossing of the
// half-cone, or -1 if no valid hit.
fn ray_parallel_t(oc: vec3<f32>, dir: vec3<f32>, lat_b: f32, after: f32) -> f32 {
    if abs(lat_b) < 1e-9 {
        if abs(dir.y) < 1e-12 { return -1.0; }
        let t = -oc.y / dir.y;
        if t > after { return t; }
        return -1.0;
    }
    let tan_l = tan(lat_b);
    let tan2 = tan_l * tan_l;
    let aa = (dir.x * dir.x + dir.z * dir.z) * tan2 - dir.y * dir.y;
    let bb = 2.0 * ((oc.x * dir.x + oc.z * dir.z) * tan2 - oc.y * dir.y);
    let cc = (oc.x * oc.x + oc.z * oc.z) * tan2 - oc.y * oc.y;
    if abs(aa) < 1e-12 {
        if abs(bb) < 1e-12 { return -1.0; }
        let t = -cc / bb;
        if t > after { return t; }
        return -1.0;
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let t0 = (-bb - sq) / (2.0 * aa);
    let t1 = (-bb + sq) / (2.0 * aa);
    let t_lo = min(t0, t1);
    let t_hi = max(t0, t1);
    let want_pos = lat_b > 0.0;
    if t_lo > after {
        let y_at = oc.y + t_lo * dir.y;
        if (y_at > 0.0) == want_pos { return t_lo; }
    }
    if t_hi > after {
        let y_at = oc.y + t_hi * dir.y;
        if (y_at > 0.0) == want_pos { return t_hi; }
    }
    return -1.0;
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
