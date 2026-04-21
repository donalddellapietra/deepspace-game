// Cube-to-sphere remap (Math Proofs / Nowell formula). GPU mirror of
// src/world/sphere_remap.rs — intentionally kept close to the Rust
// source so the port is a line-by-line check. No external deps.
//
// Forward:  F  : (-1, 1)^3 → open unit ball.
// Inverse:  F^-1 via Newton iteration on F(c) = w.
// Jacobian: analytic closed form.
// σ_min(J): smallest singular value, via closed-form eigendecomp of J^T J.

// -- forward --

fn sremap_forward(c: vec3<f32>) -> vec3<f32> {
    let c2 = c * c;
    let sx = sqrt(max(0.0, 1.0 - 0.5 * c2.y - 0.5 * c2.z + c2.y * c2.z / 3.0));
    let sy = sqrt(max(0.0, 1.0 - 0.5 * c2.z - 0.5 * c2.x + c2.z * c2.x / 3.0));
    let sz = sqrt(max(0.0, 1.0 - 0.5 * c2.x - 0.5 * c2.y + c2.x * c2.y / 3.0));
    return vec3<f32>(c.x * sx, c.y * sy, c.z * sz);
}

// -- jacobian --

fn sremap_jacobian(c: vec3<f32>) -> mat3x3<f32> {
    let x = c.x; let y = c.y; let z = c.z;
    let x2 = x * x; let y2 = y * y; let z2 = z * z;
    let sx = sqrt(max(1e-20, 1.0 - 0.5 * y2 - 0.5 * z2 + y2 * z2 / 3.0));
    let sy = sqrt(max(1e-20, 1.0 - 0.5 * z2 - 0.5 * x2 + z2 * x2 / 3.0));
    let sz = sqrt(max(1e-20, 1.0 - 0.5 * x2 - 0.5 * y2 + x2 * y2 / 3.0));
    let fxy = x * y * (2.0 * z2 / 3.0 - 1.0) / (2.0 * sx);
    let fxz = x * z * (2.0 * y2 / 3.0 - 1.0) / (2.0 * sx);
    let fyx = y * x * (2.0 * z2 / 3.0 - 1.0) / (2.0 * sy);
    let fyz = y * z * (2.0 * x2 / 3.0 - 1.0) / (2.0 * sy);
    let fzx = z * x * (2.0 * y2 / 3.0 - 1.0) / (2.0 * sz);
    let fzy = z * y * (2.0 * x2 / 3.0 - 1.0) / (2.0 * sz);
    // WGSL mat3x3 is column-major: M[c][r] gives row-r element of col-c.
    // Rust row-major j[row][col] = ∂F_row/∂c_col. In WGSL cols:
    //   col 0 (∂F/∂x) = (sx, fyx, fzx)
    //   col 1 (∂F/∂y) = (fxy, sy, fzy)
    //   col 2 (∂F/∂z) = (fxz, fyz, sz)
    return mat3x3<f32>(
        vec3<f32>(sx,  fyx, fzx),
        vec3<f32>(fxy, sy,  fzy),
        vec3<f32>(fxz, fyz, sz),
    );
}

// -- solve 3x3 --

fn sremap_solve3(m: mat3x3<f32>, b: vec3<f32>) -> vec3<f32> {
    // cofactor-matrix transpose · b / det
    let m00 = m[0][0]; let m01 = m[1][0]; let m02 = m[2][0];
    let m10 = m[0][1]; let m11 = m[1][1]; let m12 = m[2][1];
    let m20 = m[0][2]; let m21 = m[1][2]; let m22 = m[2][2];
    let c00 = m11 * m22 - m12 * m21;
    let c01 = m12 * m20 - m10 * m22;
    let c02 = m10 * m21 - m11 * m20;
    let c10 = m02 * m21 - m01 * m22;
    let c11 = m00 * m22 - m02 * m20;
    let c12 = m01 * m20 - m00 * m21;
    let c20 = m01 * m12 - m02 * m11;
    let c21 = m02 * m10 - m00 * m12;
    let c22 = m00 * m11 - m01 * m10;
    let det = m00 * c00 + m01 * c01 + m02 * c02;
    // Callers must guard against near-singular; shader path returns zero
    // delta on ill-conditioned, which Newton then retries.
    let inv_det = select(0.0, 1.0 / det, abs(det) > 1e-18);
    return vec3<f32>(
        inv_det * (c00 * b.x + c10 * b.y + c20 * b.z),
        inv_det * (c01 * b.x + c11 * b.y + c21 * b.z),
        inv_det * (c02 * b.x + c12 * b.y + c22 * b.z),
    );
}

// -- Newton inverse --

fn sremap_inverse(w: vec3<f32>, start: vec3<f32>, iters: u32) -> vec3<f32> {
    var c = start;
    for (var i: u32 = 0u; i < iters; i = i + 1u) {
        let f = sremap_forward(c);
        let r = f - w;
        if (dot(r, r) < 1e-14) { break; }
        let j = sremap_jacobian(c);
        let delta = sremap_solve3(j, -r);
        c = c + delta;
    }
    return c;
}

// -- smallest singular value (σ_min) --

// Smallest eigenvalue of a 3x3 SYMMETRIC matrix via the closed-form
// trig solution (Smith 1961). `a` is assumed symmetric.
fn symm3_eig_min(a00: f32, a11: f32, a22: f32, a01: f32, a02: f32, a12: f32) -> f32 {
    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    if (p1 < 1e-20) {
        return min(a00, min(a11, a22));
    }
    let q = (a00 + a11 + a22) / 3.0;
    let d00 = a00 - q;
    let d11 = a11 - q;
    let d22 = a22 - q;
    let p2 = d00 * d00 + d11 * d11 + d22 * d22 + 2.0 * p1;
    let p = sqrt(p2 / 6.0);
    let b00 = d00 / p;
    let b11 = d11 / p;
    let b22 = d22 / p;
    let b01 = a01 / p;
    let b02 = a02 / p;
    let b12 = a12 / p;
    // det(B) for symmetric B.
    let detb = b00 * (b11 * b22 - b12 * b12)
             - b01 * (b01 * b22 - b12 * b02)
             + b02 * (b01 * b12 - b11 * b02);
    let r = clamp(detb * 0.5, -1.0, 1.0);
    let phi = acos(r) / 3.0;
    // Smallest eigenvalue is at phi + 2π/3.
    return q + 2.0 * p * cos(phi + 2.0 * 3.14159265358979 / 3.0);
}

fn sremap_sigma_min(c: vec3<f32>) -> f32 {
    let j = sremap_jacobian(c);
    // a = J^T · J — symmetric PSD.
    // Compute the 6 unique entries.
    let j00 = j[0][0]; let j01 = j[1][0]; let j02 = j[2][0];
    let j10 = j[0][1]; let j11 = j[1][1]; let j12 = j[2][1];
    let j20 = j[0][2]; let j21 = j[1][2]; let j22 = j[2][2];
    let a00 = j00 * j00 + j10 * j10 + j20 * j20;
    let a11 = j01 * j01 + j11 * j11 + j21 * j21;
    let a22 = j02 * j02 + j12 * j12 + j22 * j22;
    let a01 = j00 * j01 + j10 * j11 + j20 * j21;
    let a02 = j00 * j02 + j10 * j12 + j20 * j22;
    let a12 = j01 * j02 + j11 * j12 + j21 * j22;
    let lam = symm3_eig_min(a00, a11, a22, a01, a02, a12);
    return sqrt(max(0.0, lam));
}

// -- utility --

// L∞ distance from c to the nearest face of the AABB
// [cell_min, cell_min + cell_size]. Non-negative when inside.
fn sremap_safe_cube_l_inf(c: vec3<f32>, cell_min: vec3<f32>, cell_size: f32) -> f32 {
    let lo = c - cell_min;
    let hi = cell_min + vec3<f32>(cell_size) - c;
    let dx = min(lo.x, hi.x);
    let dy = min(lo.y, hi.y);
    let dz = min(lo.z, hi.z);
    return max(min(dx, min(dy, dz)), 0.0);
}

// Analytic ray vs. unit ball. `dir` must be normalized. Returns
// (t_enter, t_exit, hit) packed as vec3<f32>: .z > 0 means hit.
fn sremap_ray_unit_ball(origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let od = dot(origin, dir);
    let oo = dot(origin, origin);
    let disc = od * od - (oo - 1.0);
    if (disc < 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let s = sqrt(disc);
    return vec3<f32>(-od - s, -od + s, 1.0);
}
