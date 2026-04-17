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
