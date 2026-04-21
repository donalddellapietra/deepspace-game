//! Cubed-sphere geometry + worldgen. One file, pure functions.
//!
//! A spherical body lives inside the Cartesian voxel tree as a
//! `NodeKind::CubedSphereBody` node. Its 27 children are laid out in
//! XYZ like any Cartesian node, but six specific slots (the face-
//! centers) hold face subtrees, one slot (the body center) holds a
//! uniform-stone core, and the 20 edge/corner slots are empty.
//!
//! Inside a face subtree, a node's 27 children are interpreted in
//! `(u, v, r)` axes on that face. The face root carries
//! `NodeKind::CubedSphereFace { face }`; deeper nodes stay Cartesian
//! (slot index semantics follow the face root's convention
//! contagiously along the descent path).
//!
//! Geometry uses the equal-angle cubed-sphere projection:
//! `dir = normalize(n + tan(u·π/4)·u_axis + tan(v·π/4)·v_axis)`.
//! That spreads solid angle evenly across a face, and gives the
//! UVR shell a smooth curved surface with no seams between faces.

use super::sdf::{self, Planet, Vec3};
use super::tree::{empty_children, slot_index, uniform_children, Child, NodeId, NodeKind, NodeLibrary};

// ─────────────────────────────────────────────────────── face enum

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ];

    pub fn from_index(i: u8) -> Face {
        match i {
            0 => Face::PosX, 1 => Face::NegX,
            2 => Face::PosY, 3 => Face::NegY,
            4 => Face::PosZ, 5 => Face::NegZ,
            _ => panic!("invalid face index {i}"),
        }
    }

    /// Inverse of `FACE_SLOTS[face as usize]`. Returns `None` if the
    /// slot is not one of the six face slots (e.g., the core slot).
    pub fn from_body_slot(slot: u8) -> Option<Face> {
        let s = slot as usize;
        for (i, &fs) in FACE_SLOTS.iter().enumerate() {
            if s == fs {
                return Some(Face::from_index(i as u8));
            }
        }
        None
    }

    pub fn normal(self) -> Vec3 {
        match self {
            Face::PosX => [ 1.0,  0.0,  0.0],
            Face::NegX => [-1.0,  0.0,  0.0],
            Face::PosY => [ 0.0,  1.0,  0.0],
            Face::NegY => [ 0.0, -1.0,  0.0],
            Face::PosZ => [ 0.0,  0.0,  1.0],
            Face::NegZ => [ 0.0,  0.0, -1.0],
        }
    }

    pub fn tangents(self) -> (Vec3, Vec3) {
        match self {
            Face::PosX => ([ 0.0,  0.0, -1.0], [ 0.0,  1.0,  0.0]),
            Face::NegX => ([ 0.0,  0.0,  1.0], [ 0.0,  1.0,  0.0]),
            Face::PosY => ([ 1.0,  0.0,  0.0], [ 0.0,  0.0, -1.0]),
            Face::NegY => ([ 1.0,  0.0,  0.0], [ 0.0,  0.0,  1.0]),
            Face::PosZ => ([ 1.0,  0.0,  0.0], [ 0.0,  1.0,  0.0]),
            Face::NegZ => ([-1.0,  0.0,  0.0], [ 0.0,  1.0,  0.0]),
        }
    }
}

/// Body slot holding each face's subtree. Indexed by `Face as usize`.
pub const FACE_SLOTS: [usize; 6] = [
    slot_index(2, 1, 1), // PosX
    slot_index(0, 1, 1), // NegX
    slot_index(1, 2, 1), // PosY
    slot_index(1, 0, 1), // NegY
    slot_index(1, 1, 2), // PosZ
    slot_index(1, 1, 0), // NegZ
];

/// Body slot holding the interior (uniform-stone core).
pub const CORE_SLOT: usize = slot_index(1, 1, 1);

// ─────────────────────────────────────────────── coord conversions

/// Pick the cube face whose outward normal aligns with `n`. `n` need
/// not be unit length; only direction matters.
#[inline]
pub fn pick_face(n: Vec3) -> Face {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    if ax >= ay && ax >= az {
        if n[0] >= 0.0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if n[1] >= 0.0 { Face::PosY } else { Face::NegY }
    } else {
        if n[2] >= 0.0 { Face::PosZ } else { Face::NegZ }
    }
}

/// Equal-angle warp: cube-plane coord ↔ EA coord. `ea_to_cube(c) =
/// tan(c·π/4)`; its inverse `cube_to_ea(c) = atan(c)·4/π`. Both map
/// `[-1, 1]` to `[-1, 1]` symmetrically. Spreads solid angle evenly
/// across a face so UVR cells look the same size from the center.
#[inline]
pub fn ea_to_cube(c: f32) -> f32 {
    (c * std::f32::consts::FRAC_PI_4).tan()
}
#[inline]
pub fn cube_to_ea(c: f32) -> f32 {
    c.atan() * (4.0 / std::f32::consts::PI)
}

/// `(face, u ∈ [-1,1], v ∈ [-1,1])` → unit direction from sphere center.
pub fn face_uv_to_dir(face: Face, u: f32, v: f32) -> Vec3 {
    let cu = ea_to_cube(u);
    let cv = ea_to_cube(v);
    let n = face.normal();
    let (ua, va) = face.tangents();
    sdf::normalize([
        n[0] + cu * ua[0] + cv * va[0],
        n[1] + cu * ua[1] + cv * va[1],
        n[2] + cu * ua[2] + cv * va[2],
    ])
}

/// Face-space coordinates relative to a sphere at `center`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FacePoint {
    pub face: Face,
    /// Normalized face-u ∈ [0, 1).
    pub un: f32,
    /// Normalized face-v ∈ [0, 1).
    pub vn: f32,
    /// Normalized radial ∈ [0, 1): 0 at inner shell, 1 at outer shell.
    pub rn: f32,
}

/// World-point inside body → cubed-sphere `(face, u, v, r)`. Returns
/// `None` if the point is at the sphere's exact center or radii are
/// degenerate.
///
/// `inner_r_local` / `outer_r_local` are in the body cell's local
/// `[0, 1)` frame; `body_size` is the body cell's size in the caller's
/// frame. `point_body` is in the same frame as `body_size`, measured
/// from the body cell's origin (so center is at `body_size * 0.5`).
pub fn body_point_to_face_space(
    point_body: Vec3,
    inner_r_local: f32,
    outer_r_local: f32,
    body_size: f32,
) -> Option<FacePoint> {
    let center = [body_size * 0.5; 3];
    let offset = sdf::sub(point_body, center);
    let r = sdf::length(offset);
    if r <= 1e-12 { return None; }
    let n = sdf::scale(offset, 1.0 / r);
    let face = pick_face(n);
    let n_axis = face.normal();
    let (u_axis, v_axis) = face.tangents();
    let axis_dot = sdf::dot(n, n_axis);
    if axis_dot.abs() <= 1e-12 { return None; }
    let cube_u = sdf::dot(n, u_axis) / axis_dot;
    let cube_v = sdf::dot(n, v_axis) / axis_dot;
    let inner = inner_r_local * body_size;
    let outer = outer_r_local * body_size;
    let shell = outer - inner;
    if shell <= 0.0 { return None; }
    Some(FacePoint {
        face,
        un: ((cube_to_ea(cube_u) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        vn: ((cube_to_ea(cube_v) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        rn: ((r - inner) / shell).clamp(0.0, 0.9999999),
    })
}

/// Inverse of `body_point_to_face_space`: cubed-sphere coords → body-
/// local XYZ.
pub fn face_space_to_body_point(
    face: Face,
    un: f32, vn: f32, rn: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    body_size: f32,
) -> Vec3 {
    let center = [body_size * 0.5; 3];
    let radius = (inner_r_local + rn * (outer_r_local - inner_r_local)) * body_size;
    let dir = face_uv_to_dir(face, un * 2.0 - 1.0, vn * 2.0 - 1.0);
    sdf::add(center, sdf::scale(dir, radius))
}

// ─────────────────────────────── linearized face-frame Jacobian

/// A small 3×3 matrix used for the face-subtree frame's local ↔ body
/// transform. Columns are body-XYZ basis vectors of the local frame.
pub type Mat3 = [[f32; 3]; 3];

/// Body-XYZ position of local `(u_l, v_l, r_l)` under the linearized
/// face map: `body_pos ≈ c_body + J · (u_l, v_l, r_l)`.
#[inline]
pub fn mat3_mul_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Analytic inverse of a 3×3 matrix, stored column-major as `m[c][r]`.
///
/// Uses f64 arithmetic internally. The face-frame Jacobian has
/// columns of magnitude O(frame_size · body_scale), so at deep
/// face-subtree levels the determinant shrinks as O(frame_size³).
/// f32 intermediate products collapse below subnormal range; f64
/// preserves precision, and the final division naturally rescales
/// the inverse entries to O(1/frame_size) which f32 represents
/// cleanly (magnitudes up to ~3e38).
pub fn mat3_inv(m: &Mat3) -> Mat3 {
    let a = m[0][0] as f64; let b = m[1][0] as f64; let c = m[2][0] as f64;
    let d = m[0][1] as f64; let e = m[1][1] as f64; let f = m[2][1] as f64;
    let g = m[0][2] as f64; let h = m[1][2] as f64; let i = m[2][2] as f64;
    let c00 =   e * i - f * h;
    let c01 = -(d * i - f * g);
    let c02 =   d * h - e * g;
    let c10 = -(b * i - c * h);
    let c11 =   a * i - c * g;
    let c12 = -(a * h - b * g);
    let c20 =   b * f - c * e;
    let c21 = -(a * f - c * d);
    let c22 =   a * e - b * d;
    let det = a * c00 + b * c01 + c * c02;
    debug_assert!(det.abs() > 1e-70, "mat3_inv: near-singular (det={det})");
    let inv_det = 1.0 / det;
    // A^{-1}[r][c] = C[c][r] / det, and storage M[c][r] = A[r][c],
    // so M_inv[c][r] = C[c][r] / det. Column c of the stored inverse
    // is therefore [C[c][0], C[c][1], C[c][2]] / det.
    [
        [(c00 * inv_det) as f32, (c01 * inv_det) as f32, (c02 * inv_det) as f32],
        [(c10 * inv_det) as f32, (c11 * inv_det) as f32, (c12 * inv_det) as f32],
        [(c20 * inv_det) as f32, (c21 * inv_det) as f32, (c22 * inv_det) as f32],
    ]
}

/// Linearized face-frame transform at the corner of a face-subtree
/// cell. The cell covers face-normalized range
/// `[un_corner, un_corner+frame_size] × [vn_corner, vn_corner+frame_size]
///  × [rn_corner, rn_corner+frame_size]`, which in local `[0, 3)³`
/// coords is mapped by
///
///     body_pos ≈ c_body + J · (u_l, v_l, r_l)
///
/// where `J` columns are `∂body_pos / ∂(u_l, v_l, r_l)` evaluated at
/// the frame corner. The linearization error is `O(frame_size²)` — at
/// face-subtree depth ≥ 3 this is below 0.14 % of a cell width and
/// drops geometrically with depth.
pub fn face_frame_jacobian(
    face: Face,
    un_corner: f32, vn_corner: f32, rn_corner: f32,
    frame_size: f32,
    inner_r: f32, outer_r: f32,
    body_size: f32,
) -> (Vec3, Mat3) {
    let center = [body_size * 0.5; 3];
    let n_axis = face.normal();
    let (u_axis, v_axis) = face.tangents();

    let e_u = un_corner * 2.0 - 1.0;
    let e_v = vn_corner * 2.0 - 1.0;
    let cu = ea_to_cube(e_u);
    let cv = ea_to_cube(e_v);
    // d(ea_to_cube)/d(un) = sec²(e · π/4) · π/2
    let cos_u = (e_u * std::f32::consts::FRAC_PI_4).cos();
    let cos_v = (e_v * std::f32::consts::FRAC_PI_4).cos();
    let alpha_u = std::f32::consts::FRAC_PI_2 / (cos_u * cos_u);
    let alpha_v = std::f32::consts::FRAC_PI_2 / (cos_v * cos_v);

    let raw = [
        n_axis[0] + cu * u_axis[0] + cv * v_axis[0],
        n_axis[1] + cu * u_axis[1] + cv * v_axis[1],
        n_axis[2] + cu * u_axis[2] + cv * v_axis[2],
    ];
    let nm2 = sdf::dot(raw, raw);
    let nm = nm2.sqrt();
    let inv_nm = 1.0 / nm;
    let dir = sdf::scale(raw, inv_nm);

    let r_body = (inner_r + rn_corner * (outer_r - inner_r)) * body_size;
    let dr_dbody = (outer_r - inner_r) * body_size;

    let c_body = sdf::add(center, sdf::scale(dir, r_body));

    // ∂dir/∂un = (α_u / nm) · (u_axis − (cu/nm) · dir)
    // ∂body/∂un = r_body · ∂dir/∂un (R doesn't depend on un)
    // Scaling to local unit: per-local-u = frame_size/3 of face-normalized u.
    let s = frame_size / 3.0;
    let k_u = s * r_body * alpha_u * inv_nm;
    let k_v = s * r_body * alpha_v * inv_nm;
    let cu_nm = cu * inv_nm;
    let cv_nm = cv * inv_nm;
    let col_u = [
        k_u * (u_axis[0] - cu_nm * dir[0]),
        k_u * (u_axis[1] - cu_nm * dir[1]),
        k_u * (u_axis[2] - cu_nm * dir[2]),
    ];
    let col_v = [
        k_v * (v_axis[0] - cv_nm * dir[0]),
        k_v * (v_axis[1] - cv_nm * dir[1]),
        k_v * (v_axis[2] - cv_nm * dir[2]),
    ];
    let col_r = [s * dr_dbody * dir[0], s * dr_dbody * dir[1], s * dr_dbody * dir[2]];

    (c_body, [col_u, col_v, col_r])
}

/// Ray–outer-sphere entry time, in body-frame units. `None` if miss.
pub fn ray_outer_sphere_hit(
    ray_origin_body: Vec3,
    ray_dir: Vec3,
    outer_r_local: f32,
    body_size: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let outer = outer_r_local * body_size;
    let oc = sdf::sub(ray_origin_body, center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - outer * outer;
    let disc = b * b - c;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    let t = if t_enter > 0.0 { t_enter } else { t_exit };
    if t > 0.0 { Some(t) } else { None }
}

/// Scan a hit path for the first `CubedSphereBody` ancestor. Returns
/// `(path_index, inner_r, outer_r)` where `path_index` is the entry
/// whose child is the body node (so `path[index+1]` is the face slot
/// if the hit continues into a face subtree).
pub fn find_body_ancestor_in_path(
    library: &NodeLibrary,
    hit_path: &[(NodeId, usize)],
) -> Option<(usize, f32, f32)> {
    for (index, &(node_id, slot)) in hit_path.iter().enumerate() {
        let Some(node) = library.get(node_id) else { continue };
        let Child::Node(child_id) = node.children[slot] else { continue };
        let Some(child) = library.get(child_id) else { continue };
        if let NodeKind::CubedSphereBody { inner_r, outer_r } = child.kind {
            return Some((index, inner_r, outer_r));
        }
    }
    None
}

// ────────────────────────────────────────────────── worldgen

/// The demo planet. Inner/outer radii in the body cell's local
/// `[0, 1)` frame — `0 < inner_r < outer_r ≤ 0.5` so the sphere fits
/// cleanly inside one Cartesian cell. A smooth surface (no noise),
/// stone core, grass surface.
#[derive(Clone, Debug)]
pub struct PlanetSetup {
    pub inner_r: f32,
    pub outer_r: f32,
    /// Face subtree depth. Internal SDF sampling caps at
    /// `SDF_DETAIL_LEVELS` past which uniform-stone/uniform-empty
    /// fillers extend the subtree via dedup.
    pub depth: u32,
    pub sdf: Planet,
}

pub fn demo_planet() -> PlanetSetup {
    let center: Vec3 = [0.5, 0.5, 0.5];
    let inner_r = 0.12_f32;
    let outer_r = 0.45_f32;
    PlanetSetup {
        inner_r,
        outer_r,
        depth: 28,
        sdf: Planet {
            center,
            radius: 0.30,
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 0,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: crate::world::palette::block::GRASS,
            core_block: crate::world::palette::block::STONE,
        },
    }
}

/// Max levels of SDF recursion into a face subtree. Below this, each
/// cell commits to solid-or-empty from its center sample and extends
/// via uniform dedup. Limits worldgen cost without visibly changing
/// a smooth sphere.
const SDF_DETAIL_LEVELS: u32 = 4;

/// Build a spherical body node and return its `NodeId`. Caller is
/// responsible for placing it inside a parent (e.g., world root's
/// center slot) and bumping its refcount.
pub fn insert_spherical_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(0.0 < inner_r && inner_r < outer_r && outer_r <= 0.5);

    // Build each face subtree, tagging only the root with
    // CubedSphereFace — internal nodes stay Cartesian for maximal
    // dedup (slot-index UVR convention is established at the root).
    let mut body_children = empty_children();
    for &face in &Face::ALL {
        let child = build_face_subtree(
            lib, face, inner_r, outer_r,
            -1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
            depth, depth.min(SDF_DETAIL_LEVELS), sdf,
        );
        let face_root = match child {
            Child::Node(id) => {
                let children = lib.get(id).expect("face root just inserted").children;
                lib.insert_with_kind(children, NodeKind::CubedSphereFace { face })
            }
            Child::Empty => lib.insert_with_kind(empty_children(), NodeKind::CubedSphereFace { face }),
            Child::Block(b) => {
                lib.insert_with_kind(uniform_children(Child::Block(b)), NodeKind::CubedSphereFace { face })
            }
            Child::EntityRef(_) => unreachable!("worldgen never emits entity refs"),
        };
        body_children[FACE_SLOTS[face as usize]] = Child::Node(face_root);
    }
    body_children[CORE_SLOT] = lib.build_uniform_subtree(sdf.core_block, depth);

    lib.insert_with_kind(body_children, NodeKind::CubedSphereBody { inner_r, outer_r })
}

/// Recursive build of one face subtree. Returns a `Child` so the
/// caller can collapse uniform subtrees. Emit cells by sampling the
/// SDF at the cell center under the equal-angle UVR-to-world map.
#[allow(clippy::too_many_arguments)]
fn build_face_subtree(
    lib: &mut NodeLibrary,
    face: Face,
    inner_r: f32,
    outer_r: f32,
    u_lo: f32, u_hi: f32,
    v_lo: f32, v_hi: f32,
    rn_lo: f32, rn_hi: f32,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
) -> Child {
    let body_size = 1.0f32; // sampled in body-local [0, 1)³
    let u_c = 0.5 * (u_lo + u_hi);
    let v_c = 0.5 * (v_lo + v_hi);
    let rn_c = 0.5 * (rn_lo + rn_hi);
    let p_center = face_space_to_body_point(
        face,
        (u_c + 1.0) * 0.5, (v_c + 1.0) * 0.5, rn_c,
        inner_r, outer_r, body_size,
    );
    let d_center = sdf.distance(p_center);
    let radial_half = 0.5 * (rn_hi - rn_lo) * (outer_r - inner_r);
    let lateral_half = 0.5 * (u_hi - u_lo).max(v_hi - v_lo) * outer_r;
    let cell_rad = (lateral_half * lateral_half + radial_half * radial_half).sqrt();

    if d_center > cell_rad {
        return if depth == 0 { Child::Empty } else { Child::Node(uniform_empty_chain(lib, depth)) };
    }
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        return if depth == 0 { Child::Block(b) } else { lib.build_uniform_subtree(b, depth) };
    }
    if depth == 0 {
        return if d_center < 0.0 { Child::Block(sdf.block_at(p_center)) } else { Child::Empty };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            lib.build_uniform_subtree(sdf.block_at(p_center), depth)
        } else {
            Child::Node(uniform_empty_chain(lib, depth))
        };
    }

    let mut children = empty_children();
    let du = (u_hi - u_lo) / 3.0;
    let dv = (v_hi - v_lo) / 3.0;
    let drn = (rn_hi - rn_lo) / 3.0;
    for rs in 0..3 {
        for vs in 0..3 {
            for us in 0..3 {
                children[slot_index(us, vs, rs)] = build_face_subtree(
                    lib, face, inner_r, outer_r,
                    u_lo + du * us as f32, u_lo + du * (us + 1) as f32,
                    v_lo + dv * vs as f32, v_lo + dv * (vs + 1) as f32,
                    rn_lo + drn * rs as f32, rn_lo + drn * (rs + 1) as f32,
                    depth - 1, sdf_budget - 1, sdf,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

fn uniform_empty_chain(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
}

/// Install a body into the world tree at `host_slots`, returning the
/// new world root and the body's path from the new root.
pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &PlanetSetup,
) -> (NodeId, crate::world::anchor::Path) {
    let body_id = insert_spherical_body(lib, setup.inner_r, setup.outer_r, setup.depth, &setup.sdf);
    let host_slot = slot_index(1, 1, 1) as u8;
    let root_node = lib.get(world_root).expect("world root exists");
    let mut children = root_node.children;
    children[host_slot as usize] = Child::Node(body_id);
    let new_root = lib.insert(children);
    let mut body_path = crate::world::anchor::Path::root();
    body_path.push(host_slot);
    (new_root, body_path)
}

// ─────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;

    #[test]
    fn face_center_matches_normal() {
        for &f in &Face::ALL {
            let dir = face_uv_to_dir(f, 0.0, 0.0);
            let n = f.normal();
            for i in 0..3 { assert!((dir[i] - n[i]).abs() < 1e-5); }
        }
    }

    #[test]
    fn ea_cube_round_trip() {
        for x in [-0.9_f32, -0.3, 0.0, 0.5, 0.99] {
            assert!((cube_to_ea(ea_to_cube(x)) - x).abs() < 1e-5);
        }
    }

    #[test]
    fn body_face_space_round_trip() {
        for &face in &Face::ALL {
            for &(u, v, r) in &[(0.1_f32, 0.1, 0.1), (0.5, 0.5, 0.5), (0.9, 0.9, 0.9)] {
                let body = face_space_to_body_point(face, u, v, r, 0.12, 0.45, 1.0);
                let back = body_point_to_face_space(body, 0.12, 0.45, 1.0).unwrap();
                assert_eq!(back.face, face);
                assert!((back.un - u).abs() < 1e-4, "un {u} → {}", back.un);
                assert!((back.vn - v).abs() < 1e-4);
                assert!((back.rn - r).abs() < 1e-4);
            }
        }
    }

    fn numeric_jacobian(
        face: Face, un: f32, vn: f32, rn: f32, frame_size: f32,
        inner_r: f32, outer_r: f32, body_size: f32,
    ) -> (Vec3, Mat3) {
        // Central-difference numerical Jacobian, for cross-checking the
        // analytical result. Eval at frame corner (local = [0,0,0]).
        // 1e-2 local-coord step keeps the f32 body_pos delta well above
        // f32 eps (smaller eps loses precision in the differenced pair).
        let eps = 1e-2_f32;
        let s = frame_size / 3.0;
        let at = |du: f32, dv: f32, dr: f32| {
            face_space_to_body_point(
                face,
                un + du * s, vn + dv * s, rn + dr * s,
                inner_r, outer_r, body_size,
            )
        };
        let c_body = at(0.0, 0.0, 0.0);
        let col_u = sdf::scale(sdf::sub(at(eps, 0.0, 0.0), at(-eps, 0.0, 0.0)), 0.5 / eps);
        let col_v = sdf::scale(sdf::sub(at(0.0, eps, 0.0), at(0.0, -eps, 0.0)), 0.5 / eps);
        let col_r = sdf::scale(sdf::sub(at(0.0, 0.0, eps), at(0.0, 0.0, -eps)), 0.5 / eps);
        (c_body, [col_u, col_v, col_r])
    }

    #[test]
    fn face_frame_jacobian_matches_numeric() {
        // A few representative points: face center, off-center,
        // deeper sub-cell. Central-difference + f32 loses relative
        // precision as `frame_size` shrinks — past ~1/729 the
        // differenced body-XYZ pair drops below f32 eps and cannot
        // serve as a cross-check. The analytical form remains valid
        // at arbitrary depth (it's a closed-form derivative).
        let cases: &[(Face, f32, f32, f32, f32)] = &[
            (Face::PosX, 0.5, 0.5, 0.5, 1.0_f32 / 3.0),
            (Face::PosX, 0.1, 0.9, 0.3, 1.0_f32 / 27.0),
            (Face::PosY, 0.75, 0.25, 0.5, 1.0_f32 / 81.0),
            (Face::NegZ, 0.2, 0.8, 0.8, 1.0_f32 / 243.0),
        ];
        for &(face, un, vn, rn, size) in cases {
            let (c_body, j) = face_frame_jacobian(face, un, vn, rn, size, 0.12, 0.45, 1.0);
            let (c_num, j_num) = numeric_jacobian(face, un, vn, rn, size, 0.12, 0.45, 1.0);
            for i in 0..3 {
                assert!(
                    (c_body[i] - c_num[i]).abs() < 1e-5,
                    "c_body mismatch face={face:?} un={un} vn={vn} rn={rn}"
                );
            }
            for col in 0..3 {
                for row in 0..3 {
                    let analytic = j[col][row];
                    let numeric = j_num[col][row];
                    let scale = analytic.abs().max(numeric.abs()).max(1e-6);
                    let rel = (analytic - numeric).abs() / scale;
                    // Central-difference + f32 tops out around 3 % at
                    // tiny step sizes; tolerate up to 5 %.
                    assert!(
                        rel < 5e-2,
                        "J[{col}][{row}] mismatch face={face:?} un={un} vn={vn} rn={rn} size={size}: \
                         analytic={analytic}  numeric={numeric}  rel={rel}"
                    );
                }
            }
        }
    }

    #[test]
    fn mat3_inv_round_trip() {
        // Invert a face-frame Jacobian and verify J · J_inv = I.
        let (_, j) = face_frame_jacobian(Face::PosX, 0.5, 0.5, 0.5, 1.0 / 27.0, 0.12, 0.45, 1.0);
        let j_inv = mat3_inv(&j);
        // Check J · J_inv applied to basis vectors yields identity.
        for i in 0..3 {
            let mut basis = [0.0_f32; 3];
            basis[i] = 1.0;
            let temp = mat3_mul_vec(&j_inv, basis);
            let back = mat3_mul_vec(&j, temp);
            for k in 0..3 {
                let expected = if i == k { 1.0 } else { 0.0 };
                assert!(
                    (back[k] - expected).abs() < 1e-4,
                    "J·J_inv[{i}][{k}] = {} ≠ {}", back[k], expected
                );
            }
        }
    }

    #[test]
    fn insert_body_creates_structured_children() {
        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5; 3], radius: 0.30,
            noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
        };
        let body = insert_spherical_body(&mut lib, 0.12, 0.45, 6, &sdf);
        let body_node = lib.get(body).unwrap();
        assert!(matches!(body_node.kind, NodeKind::CubedSphereBody { .. }));
        for &face in &Face::ALL {
            let slot = FACE_SLOTS[face as usize];
            match body_node.children[slot] {
                Child::Node(id) => {
                    let n = lib.get(id).unwrap();
                    assert!(matches!(n.kind, NodeKind::CubedSphereFace { face: f } if f == face));
                }
                _ => panic!("face slot {slot} not a Node"),
            }
        }
        match body_node.children[CORE_SLOT] {
            Child::Node(id) => {
                assert_eq!(lib.get(id).unwrap().uniform_type, block::STONE);
            }
            Child::Block(b) => assert_eq!(b, block::STONE),
            _ => panic!("core slot empty"),
        }
        for s in 0..27 {
            if s == CORE_SLOT || FACE_SLOTS.contains(&s) { continue; }
            assert!(matches!(body_node.children[s], Child::Empty));
        }
    }
}
