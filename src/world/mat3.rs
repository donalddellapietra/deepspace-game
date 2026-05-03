//! 3×3 matrix helpers (column-major: `m[col][row]`).
//!
//! Used by code paths that compose TangentBlock rotations along
//! anchor / frame paths — a single source of truth for the matmul
//! and matrix-vector operations that previously had three identical
//! private copies in `world_pos`, `app::mod`, and `app::event_loop`.

pub type Mat3 = [[f32; 3]; 3];
pub type Vec3 = [f32; 3];

/// `(a · b)[r, c] = sum_k a[r, k] · b[k, c]` for column-major
/// matrices stored as `m[col][row]`.
#[inline]
pub fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for c in 0..3 {
        for r in 0..3 {
            let mut s = 0.0f32;
            for k in 0..3 {
                s += a[k][r] * b[c][k];
            }
            out[c][r] = s;
        }
    }
    out
}

/// `(m · v).i = sum_j m[j][i] · v.j` — apply a column-major 3×3 to
/// a 3-vector.
#[inline]
pub fn mul_vec3(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// `(m^T · v).i = sum_j m[i][j] · v.j` — apply the transpose of a
/// column-major 3×3 to a 3-vector. Equivalent to `mul_vec3(&m^T, v)`
/// without materializing the transpose.
#[inline]
pub fn transpose_mul_vec3(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
