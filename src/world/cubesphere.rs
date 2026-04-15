//! Cubed-sphere coordinates: the math that lets a planet be built
//! from voxels that bulge outward at large scales but feel like flat
//! cubes at the surface.
//!
//! A "planet" is six cube faces wrapped around a sphere. Each face
//! carries its own flat grid of cells parameterized by (u, v) ∈
//! [-1, 1]², plus a radial axis r = distance from planet center. A
//! "block" is a cell in (face, u, v, r) space; its 8 world-space
//! corners live on two concentric spheres at radii r and r + Δr,
//! which gives the block its signature bulged-square shape.
//!
//! The six faces index as:
//!   0 = +X   1 = -X   2 = +Y   3 = -Y   4 = +Z   5 = -Z
//! For each face we pick two orthogonal tangent axes (u, v) so that
//! adjacent cells on the same face share exact edges, and the three
//! faces meeting at every cube corner align there as well.
//!
//! There are no poles — one of the main reasons to use a cubed-sphere
//! over a lat/lon parameterization. Cells near cube-face seams are
//! only mildly stretched (at most ~1.5× area ratio between the
//! center of a face and its corners), which is invisible in
//! gameplay.
//!
//! This module is pure math. It touches nothing else in the engine;
//! later passes (tree integration, renderer, collision) will build
//! on it.

use super::sdf::{self, Planet, Vec3};
use super::tree::{
    empty_children, slot_index, uniform_children, Child, NodeId, NodeKind,
    NodeLibrary, UNIFORM_EMPTY,
};

/// One of the six cube faces.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::PosX, Face::NegX,
        Face::PosY, Face::NegY,
        Face::PosZ, Face::NegZ,
    ];

    pub fn from_index(i: u8) -> Face {
        match i {
            0 => Face::PosX, 1 => Face::NegX,
            2 => Face::PosY, 3 => Face::NegY,
            4 => Face::PosZ, 5 => Face::NegZ,
            _ => panic!("invalid face index {i}"),
        }
    }

    /// Unit vector pointing "out" of this face's center.
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

    /// Tangent basis (u_axis, v_axis) for this face. Chosen so that
    /// (u_axis × v_axis) = normal — right-handed, consistent winding.
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

/// A point in cubed-sphere coordinates, relative to some planet
/// center.  `face` selects one of 6 cube faces; `u, v ∈ [-1, 1]` is
/// the 2D position on that face; `r` is the radial distance from
/// planet center in world units.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CubeSphereCoord {
    pub face: Face,
    pub u: f32,
    pub v: f32,
    pub r: f32,
}

/// Convert (face, u, v) → unit direction pointing outward from the
/// planet center. `u, v ∈ [-1, 1]` live in **equal-angle** space:
/// each uniform step in `(u, v)` covers the same angular slice of
/// the sphere as seen from its center, which makes cells project
/// to nearly-uniform area (~6% variation peak-to-peak vs. ~50% for
/// the basic gnomonic projection).
///
/// The warp is one `tan` each on `u` and `v`: a uniform cell grid
/// gets pre-compressed toward the face's middle, exactly canceling
/// the expansion `normalize` would otherwise apply near the corners.
pub fn face_uv_to_dir(face: Face, u: f32, v: f32) -> Vec3 {
    let cube_u = (u * std::f32::consts::FRAC_PI_4).tan();
    let cube_v = (v * std::f32::consts::FRAC_PI_4).tan();
    let n = face.normal();
    let (ua, va) = face.tangents();
    let cube_pt = [
        n[0] + cube_u * ua[0] + cube_v * va[0],
        n[1] + cube_u * ua[1] + cube_v * va[1],
        n[2] + cube_u * ua[2] + cube_v * va[2],
    ];
    sdf::normalize(cube_pt)
}

/// Inverse of the equal-angle warp: raw cube-face coord → equal-angle.
/// The shader + raymarchers compute `cube_u = ratio` (e.g. `-n.z/ax`),
/// then call this to get the `u ∈ [-1, 1]` used for cell indexing.
#[inline]
pub fn cube_to_ea(c: f32) -> f32 {
    c.atan() * (4.0 / std::f32::consts::PI)
}

/// Given a planet center and a `CubeSphereCoord`, return the
/// world-space position of that point.
pub fn coord_to_world(center: Vec3, c: CubeSphereCoord) -> Vec3 {
    let dir = face_uv_to_dir(c.face, c.u, c.v);
    sdf::add(center, sdf::scale(dir, c.r))
}

/// Convert a world-space position to cubed-sphere coordinates
/// relative to a planet center. Picks the face whose axis is most
/// aligned with (pos - center); ties break toward the face with
/// lower index. Returns `None` if `pos == center` (undefined
/// direction).
pub fn world_to_coord(center: Vec3, pos: Vec3) -> Option<CubeSphereCoord> {
    let d = sdf::sub(pos, center);
    let r = sdf::length(d);
    if r < 1e-12 { return None; }
    let dir = sdf::scale(d, 1.0 / r);

    // Pick the dominant axis; the raw ratios give the CUBE-space
    // (u, v) on that face. Then apply the equal-angle inverse so
    // `u, v` are the same coords cells are indexed by.
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();
    let (face, cube_u, cube_v) = if ax >= ay && ax >= az {
        if dir[0] > 0.0 { (Face::PosX, -dir[2] / ax,  dir[1] / ax) }
        else            { (Face::NegX,  dir[2] / ax,  dir[1] / ax) }
    } else if ay >= az {
        if dir[1] > 0.0 { (Face::PosY,  dir[0] / ay, -dir[2] / ay) }
        else            { (Face::NegY,  dir[0] / ay,  dir[2] / ay) }
    } else {
        if dir[2] > 0.0 { (Face::PosZ,  dir[0] / az,  dir[1] / az) }
        else            { (Face::NegZ, -dir[0] / az,  dir[1] / az) }
    };

    Some(CubeSphereCoord {
        face,
        u: cube_to_ea(cube_u),
        v: cube_to_ea(cube_v),
        r,
    })
}

/// The eight world-space corners of a block spanning
/// `[u, u+du] × [v, v+dv] × [r, r+dr]` on `face`, around `center`.
/// Returned order: `[u0,v0,r0], [u1,v0,r0], [u0,v1,r0], [u1,v1,r0],
///                  [u0,v0,r1], [u1,v0,r1], [u0,v1,r1], [u1,v1,r1]`.
/// These corners are NOT coplanar: the top and bottom faces are
/// spherical patches (bulged), and the four side faces are
/// frustum-like walls between them. That bulge is the whole point.
pub fn block_corners(
    center: Vec3,
    face: Face,
    u: f32, v: f32, r: f32,
    du: f32, dv: f32, dr: f32,
) -> [Vec3; 8] {
    let mut out = [[0.0; 3]; 8];
    let coords = [
        (u,      v,      r),
        (u + du, v,      r),
        (u,      v + dv, r),
        (u + du, v + dv, r),
        (u,      v,      r + dr),
        (u + du, v,      r + dr),
        (u,      v + dv, r + dr),
        (u + du, v + dv, r + dr),
    ];
    for (i, &(cu, cv, cr)) in coords.iter().enumerate() {
        out[i] = coord_to_world(center, CubeSphereCoord { face, u: cu, v: cv, r: cr });
    }
    out
}

// ────────────────────────────────────────────────── planet data

/// A planet made of 6 face-subtrees in the content-addressed
/// `NodeLibrary`. Each face subtree's 3 recursive axes are
/// `(u_slot, v_slot, r_slot)` in the planet's equal-angle cubed-sphere
/// frame — NOT (x, y, z). A subtree leaf is a block / empty terminal
/// exactly like any other subtree, so the existing content-addressed
/// dedup, LOD cascade, and editing primitives all apply.
///
/// Recursion semantics: each node splits its local (u, v, r) box
/// into 3×3×3 children. The slot at `(us, vs, rs)` covers
/// `u ∈ [u_lo + us·du/3, u_lo + (us+1)·du/3]` and analogously for
/// v and r, where `(u_lo, u_hi, v_lo, v_hi, r_lo, r_hi)` is the
/// current node's box. At the top level each face's box is
/// `[-1, 1] × [-1, 1] × [inner_r, outer_r]`.
///
/// A leaf cell is a "bulged voxel": its world-space corners come
/// from `block_corners` at the cell's (u, v, r) extents. Zoom in
/// and you descend into the subtree, revealing 27 smaller bulged
/// voxels per parent — the same "blocks inside blocks" mechanic
/// the rest of the engine uses, just interpreted in spherical
/// coordinates.
#[derive(Clone, Debug)]
pub struct SphericalPlanet {
    pub center: Vec3,
    /// Cells span `r ∈ [inner_r, outer_r]`. A solid column typical of
    /// a rocky planet has its surface near the midpoint and empty
    /// cells above, solid cells below.
    pub inner_r: f32,
    pub outer_r: f32,
    /// Levels of recursion under each face root. Zoom-in reveals up
    /// to `depth` cascades of 27 sub-cells before bottoming out.
    pub depth: u32,
    /// `NodeKind::CubedSphereBody` node whose 27 children include
    /// the 6 face subtrees at face-center slots. This is the one
    /// place the engine stores the planet's identity — face roots
    /// are read from its children via [`face_root_of`], and the
    /// shell radii come from its `NodeKind` payload.
    pub body_node: NodeId,
}

/// Return the face subtree root for `face` under `body_node` —
/// `body_node`'s child at the face-center slot for that face.
/// Returns `None` if `body_node` isn't in the library (e.g., it was
/// evicted after an edit) or its face-center slot isn't a
/// `Child::Node` (e.g., a broken body with a solid filler).
pub fn face_root_of(lib: &NodeLibrary, body_node: NodeId, face: Face) -> Option<NodeId> {
    let node = lib.get(body_node)?;
    match node.children[body_face_center_slot(face)] {
        Child::Node(id) => Some(id),
        _ => None,
    }
}

/// Max levels the SDF recursion is allowed to descend into a face
/// subtree. Beyond this, each cell commits to solid-or-empty based
/// on the center sample and wraps the remaining tree depth in a
/// dedup'd uniform filler. This is what makes `depth = 20` feasible:
/// we never do more than `27^SDF_DETAIL_LEVELS` SDF sample chains per
/// face; the filler chain below is O(depth) unique nodes shared
/// across the whole planet.
const SDF_DETAIL_LEVELS: u32 = 4;

/// Resolution of a body node's world-space footprint: center, radii,
/// and the face-subtree recursion depth, derived from its anchor
/// path and `NodeKind::CubedSphereBody` payload. The anchor is the
/// source of truth; this is a projection for downstream consumers
/// (gravity math, shader uniform population) that need world units.
pub struct ResolvedBody {
    pub center: [f32; 3],
    pub inner_r_world: f32,
    pub outer_r_world: f32,
    /// Levels of recursion under each face root.
    pub face_subtree_depth: u32,
}

/// Walk `world_root` down `anchor` to the body node and compute its
/// world-space footprint. Returns `None` if the walk hits a non-Node
/// child or the resolved node doesn't have `CubedSphereBody` kind.
pub fn resolve_body_from_anchor(
    lib: &NodeLibrary,
    world_root: NodeId,
    anchor: &super::coords::Path,
) -> Option<ResolvedBody> {
    let mut id = world_root;
    for &slot in anchor.slots() {
        let node = lib.get(id)?;
        match node.children[slot as usize] {
            Child::Node(child) => id = child,
            _ => return None,
        }
    }
    let body = lib.get(id)?;
    let (inner_r, outer_r) = match body.kind {
        NodeKind::CubedSphereBody { inner_r, outer_r } => (inner_r, outer_r),
        _ => return None,
    };
    // Cell size at `anchor.depth()` in world units.
    let cell_size = super::coords::ROOT_EXTENT / 3f32.powi(anchor.depth() as i32);
    let center = super::coords::world_pos_to_f32(&super::coords::WorldPos {
        anchor: *anchor,
        offset: [0.5, 0.5, 0.5],
    });
    // Face-subtree depth: count Node descents from any face root
    // (they're all the same depth by construction).
    let face_root_id = match body.children[body_face_center_slot(Face::PosX)] {
        Child::Node(id) => id,
        _ => return None,
    };
    let face_subtree_depth = measure_node_depth(lib, face_root_id);
    Some(ResolvedBody {
        center,
        inner_r_world: inner_r * cell_size,
        outer_r_world: outer_r * cell_size,
        face_subtree_depth,
    })
}

fn measure_node_depth(lib: &NodeLibrary, start: NodeId) -> u32 {
    // Walk the leftmost Node child chain to count depth.
    let mut d = 0u32;
    let mut id = start;
    loop {
        let Some(n) = lib.get(id) else { break; };
        d += 1;
        match n.children[0] {
            Child::Node(child) => id = child,
            _ => break,
        }
    }
    d
}

/// Build a 6-face `SphericalPlanet` in `lib` by recursively sampling
/// `sdf` along each face's `(u, v, r)` subtree. SDF-driven recursion
/// is bounded by `SDF_DETAIL_LEVELS`; below it, each straddling cell
/// becomes a uniform filler of the remaining depth. Total unique
/// nodes: `O(surface_cells_at_SDF_DETAIL_LEVELS · 27 + depth)`.
pub fn generate_spherical_planet(
    lib: &mut NodeLibrary,
    center: Vec3,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> SphericalPlanet {
    let mut face_roots = [0u64; 6];
    let sdf_budget = depth.min(SDF_DETAIL_LEVELS);
    for &face in &Face::ALL {
        let child = build_face_subtree(
            lib, face, center,
            -1.0, 1.0, -1.0, 1.0, inner_r, outer_r,
            depth, sdf_budget, sdf,
        );
        let plain_id: NodeId = match child {
            Child::Node(id) => id,
            Child::Empty => build_uniform_empty(lib, depth.max(1)),
            Child::Block(b) => {
                match lib.build_uniform_subtree(b, depth.max(1)) {
                    Child::Node(id) => id,
                    _ => lib.insert(uniform_children(Child::Block(b))),
                }
            }
        };
        // Retag the face root with `NodeKind::CubedSphereFace` so
        // callers that dispatch on `NodeKind` know they're entering
        // `(u, v, r)` axis semantics. The underlying children are
        // identical — dedup still shares deeper subtrees across
        // faces, the top-level node is unique per face.
        let kids = lib.get(plain_id).map(|n| n.children).expect("face root must exist");
        let face_root = lib.insert_with_kind(
            kids,
            NodeKind::CubedSphereFace { face: face as u8 },
        );
        face_roots[face as usize] = face_root;
        lib.ref_inc(face_root);
    }
    // Assemble the body node: 6 face-center slots hold the face
    // subtrees, 20 corner/edge slots are Empty, the center slot is
    // the interior filler (uniform `core_block` of depth `depth`).
    let mut body_children = empty_children();
    for &face in &Face::ALL {
        let slot = body_face_center_slot(face);
        body_children[slot] = Child::Node(face_roots[face as usize]);
    }
    body_children[slot_index(1, 1, 1)] = lib.build_uniform_subtree(sdf.core_block, depth.max(1));
    let body_node = lib.insert_with_kind(
        body_children,
        NodeKind::CubedSphereBody { inner_r, outer_r },
    );
    lib.ref_inc(body_node);

    let _ = face_roots; // face_roots are now reachable via `body_node.children`.
    SphericalPlanet { center, inner_r, outer_r, depth, body_node }
}

/// Slot index in a `CubedSphereBody` node's 3×3×3 children that holds
/// the subtree for `face`. Face-center slots: `+X→(2,1,1)`,
/// `-X→(0,1,1)`, `+Y→(1,2,1)`, `-Y→(1,0,1)`, `+Z→(1,1,2)`,
/// `-Z→(1,1,0)`.
pub fn body_face_center_slot(face: Face) -> usize {
    match face {
        Face::PosX => slot_index(2, 1, 1),
        Face::NegX => slot_index(0, 1, 1),
        Face::PosY => slot_index(1, 2, 1),
        Face::NegY => slot_index(1, 0, 1),
        Face::PosZ => slot_index(1, 1, 2),
        Face::NegZ => slot_index(1, 1, 0),
    }
}

/// Inverse of `body_face_center_slot`: decode a body slot index
/// back to its `Face`. Returns `None` for the center slot (interior
/// filler) and the 20 corner / edge slots.
pub fn face_from_body_center_slot(slot: usize) -> Option<Face> {
    for &face in &Face::ALL {
        if body_face_center_slot(face) == slot {
            return Some(face);
        }
    }
    None
}

/// Recursive builder for one cubed-sphere face. Returns a `Child` so
/// the caller can collapse uniform subtrees up the call chain.
///
/// `depth` = remaining tree depth (total levels still to build below
/// this cell). `sdf_budget` = how many more SDF-driven recursions
/// are allowed. When the budget hits zero but more depth remains,
/// the cell commits to solid-or-empty by the center sample and
/// wraps the remaining depth in a dedup'd uniform filler. This
/// keeps recursive work bounded by `27^SDF_DETAIL_LEVELS` regardless
/// of how deep the tree goes.
fn build_face_subtree(
    lib: &mut NodeLibrary,
    face: Face,
    center: Vec3,
    u_lo: f32, u_hi: f32,
    v_lo: f32, v_hi: f32,
    r_lo: f32, r_hi: f32,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
) -> Child {
    let u_c = 0.5 * (u_lo + u_hi);
    let v_c = 0.5 * (v_lo + v_hi);
    let r_c = 0.5 * (r_lo + r_hi);
    let p_center = coord_to_world(center, CubeSphereCoord { face, u: u_c, v: v_c, r: r_c });
    let d_center = sdf.distance(p_center);

    let du = u_hi - u_lo;
    let dv = v_hi - v_lo;
    let dr = r_hi - r_lo;
    let lateral_half = r_hi * 0.5 * du.max(dv);
    let radial_half = 0.5 * dr;
    let cell_rad = (lateral_half * lateral_half + radial_half * radial_half).sqrt();

    // Fully outside the SDF surface → uniform empty subtree.
    if d_center > cell_rad {
        if depth == 0 { return Child::Empty; }
        return Child::Node(build_uniform_empty(lib, depth));
    }
    // Fully inside → uniform block subtree with the center's block type.
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        if depth == 0 { return Child::Block(b); }
        return lib.build_uniform_subtree(b, depth);
    }

    // Straddling the surface. If we still have tree depth but no
    // more SDF budget, commit to the center sample and fill the
    // rest of the subtree with a uniform filler. This is the
    // critical cap: without it, `depth = 20` would recurse into
    // ~3^40 surface-band calls before bottoming out.
    if depth == 0 {
        return if d_center < 0.0 {
            Child::Block(sdf.block_at(p_center))
        } else {
            Child::Empty
        };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            lib.build_uniform_subtree(sdf.block_at(p_center), depth)
        } else {
            Child::Node(build_uniform_empty(lib, depth))
        };
    }

    // Subdivide into 3×3×3, with both depth and sdf_budget decremented.
    let mut children = empty_children();
    for rs in 0..3 {
        for vs in 0..3 {
            for us in 0..3 {
                let us_lo = u_lo + du * (us as f32) / 3.0;
                let us_hi = u_lo + du * (us as f32 + 1.0) / 3.0;
                let vs_lo = v_lo + dv * (vs as f32) / 3.0;
                let vs_hi = v_lo + dv * (vs as f32 + 1.0) / 3.0;
                let rs_lo = r_lo + dr * (rs as f32) / 3.0;
                let rs_hi = r_lo + dr * (rs as f32 + 1.0) / 3.0;
                children[slot_index(us, vs, rs)] = build_face_subtree(
                    lib, face, center,
                    us_lo, us_hi, vs_lo, vs_hi, rs_lo, rs_hi,
                    depth - 1, sdf_budget - 1, sdf,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

/// Build (and cache) a uniform-empty subtree of the given depth.
fn build_uniform_empty(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
}

/// True iff the tree rooted at `c` is semantically equivalent to a
/// single `target` terminal — `c` is literally `target`, OR `c` is a
/// Node whose entire subtree is uniform `target`. Used so edits that
/// would replace a uniform subtree with an equivalent terminal are
/// detected as no-ops and don't churn the NodeLibrary.
fn child_equivalent_to(lib: &NodeLibrary, c: Child, target: Child) -> bool {
    if c == target { return true; }
    let Child::Node(nid) = c else { return false; };
    let Some(node) = lib.get(nid) else { return false; };
    match target {
        Child::Empty => node.uniform_type == UNIFORM_EMPTY,
        Child::Block(b) => node.uniform_type == b,
        Child::Node(_) => false,
    }
}

/// Rebuild a face subtree with the cell at the end of `slots` replaced
/// by `new_child`. Descends through Node children, expanding uniform
/// terminals on the path so the edit lands at the requested depth.
/// Content-addressed inserts mean no-op edits collapse back to the
/// original NodeId at the top.
fn rebuild_with_edit(
    lib: &mut NodeLibrary,
    current_id: NodeId,
    slots: &[usize],
    level: usize,
    new_child: Child,
) -> NodeId {
    let Some(node) = lib.get(current_id) else { return current_id; };
    let kind = node.kind;
    let target_slot = slots[level];
    let mut new_children = node.children;

    if level + 1 == slots.len() {
        if child_equivalent_to(lib, node.children[target_slot], new_child) {
            return current_id;
        }
        new_children[target_slot] = new_child;
    } else {
        let child_next_id = match node.children[target_slot] {
            Child::Node(nid) => {
                if child_equivalent_to(lib, Child::Node(nid), new_child) {
                    return current_id;
                }
                rebuild_with_edit(lib, nid, slots, level + 1, new_child)
            }
            other => {
                if other == new_child {
                    return current_id;
                }
                let expanded = lib.insert(uniform_children(other));
                rebuild_with_edit(lib, expanded, slots, level + 1, new_child)
            }
        };
        new_children[target_slot] = Child::Node(child_next_id);
    }

    // Preserve the rebuilt node's kind so a body / face that's edited
    // doesn't get demoted to Cartesian on the way back up.
    lib.insert_with_kind(new_children, kind)
}

impl SphericalPlanet {
    /// Project `world` into the planet's (face, u_n, v_n, r_n)
    /// normalized cubed-sphere frame. `u_n` / `v_n` / `r_n` all live
    /// in `[0, 1)` — same coordinates the shader uses to descend a
    /// face subtree. Returns `None` if `world` is outside the shell
    /// or degenerate at the center.
    pub fn world_to_normalized(
        &self,
        world: Vec3,
    ) -> Option<(Face, f32, f32, f32)> {
        let local = sdf::sub(world, self.center);
        let r = sdf::length(local);
        if r < self.inner_r || r >= self.outer_r { return None; }
        let coord = world_to_coord(self.center, world)?;
        let u_n = ((coord.u + 1.0) * 0.5).clamp(0.0, 0.9999999);
        let v_n = ((coord.v + 1.0) * 0.5).clamp(0.0, 0.9999999);
        let r_n = ((r - self.inner_r) / (self.outer_r - self.inner_r))
            .clamp(0.0, 0.9999999);
        Some((coord.face, u_n, v_n, r_n))
    }

    /// Walk the face subtree for `(face, u_n, v_n, r_n)` down to
    /// `max_depth` levels, stopping early at the first non-Node
    /// terminal. Returns the palette index (0 = empty) and the
    /// depth at which the walk stopped. Useful for collision /
    /// highlight queries that want to limit resolution.
    pub fn sample_subtree(
        &self,
        lib: &NodeLibrary,
        face: Face,
        u_n: f32,
        v_n: f32,
        r_n: f32,
        max_depth: u32,
    ) -> (u8, u32) {
        let Some(mut node_id) = face_root_of(lib, self.body_node, face) else {
            return (0, 0);
        };
        let mut un = u_n.clamp(0.0, 0.9999999);
        let mut vn = v_n.clamp(0.0, 0.9999999);
        let mut rn = r_n.clamp(0.0, 0.9999999);
        let limit = max_depth.min(self.depth);
        for d in 0..limit {
            let Some(node) = lib.get(node_id) else { return (0, d); };
            let us = ((un * 3.0) as usize).min(2);
            let vs = ((vn * 3.0) as usize).min(2);
            let rs = ((rn * 3.0) as usize).min(2);
            let slot = slot_index(us, vs, rs);
            match node.children[slot] {
                Child::Empty => return (0, d + 1),
                Child::Block(b) => return (b, d + 1),
                Child::Node(nid) => {
                    node_id = nid;
                    un = un * 3.0 - us as f32;
                    vn = vn * 3.0 - vs as f32;
                    rn = rn * 3.0 - rs as f32;
                }
            }
        }
        // Bottomed out at limit without a terminal. Use the subtree's
        // representative block so "any solid content inside this
        // chunk" counts as a hit at the requested depth.
        let repr = lib.get(node_id).map(|n| n.representative_block).unwrap_or(255);
        if repr < 255 { (repr, limit) } else { (0, limit) }
    }

    /// Replace the `(iu, iv, ir)` cell at `depth` on `face` with
    /// `new_child` (typically `Empty` for break, `Block(b)` for
    /// place). The target is identified by integer coordinates in a
    /// `3^depth` grid — the same coordinates `raycast_highlight`
    /// returns — so a highlight at depth 2 becomes a depth-2 edit.
    ///
    /// If the path to the target passes through a uniform terminal
    /// shallower than `depth` (a pack-time flattened chunk), the
    /// terminal is re-expanded into a Node of 27 identical children
    /// and the edit proceeds into the slot. Content-addressed dedup
    /// keeps that expansion cheap. Returns `true` if the face root
    /// changed.
    pub fn set_cell_at_depth(
        &mut self,
        world: &mut super::state::WorldState,
        body_anchor: &super::coords::Path,
        face: Face,
        iu: u32, iv: u32, ir: u32,
        depth: u32,
        new_child: Child,
    ) -> bool {
        let d = depth.clamp(1, self.depth);
        let cells = 3u32.pow(d);
        if iu >= cells || iv >= cells || ir >= cells { return false; }

        // Per-level slot indices from face root toward the target.
        // Sized for `MAX_DEPTH` from `tree.rs` (63) with headroom;
        // depth=20 planets were hitting a 16-slot cap here.
        let mut slots: [usize; 64] = [0; 64];
        let levels = d as usize;
        for level in 0..levels {
            let shift = (d - 1 - level as u32) as u32;
            let div = 3u32.pow(shift);
            let us = ((iu / div) % 3) as usize;
            let vs = ((iv / div) % 3) as usize;
            let rs = ((ir / div) % 3) as usize;
            slots[level] = slot_index(us, vs, rs);
        }

        let face_slot = body_face_center_slot(face);
        let root = match face_root_of(&world.library, self.body_node, face) {
            Some(id) => id,
            None => return false,
        };
        let new_face_root =
            rebuild_with_edit(&mut world.library, root, &slots[..levels], 0, new_child);
        if new_face_root == root { return false; }

        // Compose the chain of slot rewrites from face root up through
        // the body to the world root, then swap in the new root.
        // Rust's `insert_with_kind` dedups along the way — identical
        // post-edit states collapse to existing nodes.
        let body_kind = world.library.get(self.body_node)
            .map(|b| b.kind)
            .expect("body node must exist");
        let mut body_children = world.library.get(self.body_node)
            .expect("body node must exist")
            .children;
        body_children[face_slot] = Child::Node(new_face_root);
        let new_body = world.library.insert_with_kind(body_children, body_kind);

        // Walk back up from the body's parent to the world root,
        // rebuilding each Cartesian ancestor with the new child.
        let mut current_new = new_body;
        let mut current_anchor_level = body_anchor.depth() as usize;
        while current_anchor_level > 0 {
            let anchor_slot = body_anchor.slots()[current_anchor_level - 1] as usize;
            // Resolve the parent's NodeId by walking from the world
            // root down the path.
            let mut parent_id = world.root;
            for i in 0..(current_anchor_level - 1) {
                let slot = body_anchor.slots()[i] as usize;
                match world.library.get(parent_id).unwrap().children[slot] {
                    Child::Node(child) => parent_id = child,
                    _ => return false,
                }
            }
            let parent = world.library.get(parent_id).expect("parent must exist");
            let mut new_children = parent.children;
            new_children[anchor_slot] = Child::Node(current_new);
            current_new = world.library.insert_with_kind(new_children, parent.kind);
            current_anchor_level -= 1;
        }

        // Swap in the new world root; old subtrees drop via ref counting.
        if current_new != world.root {
            world.swap_root(current_new);
            self.body_node = new_body;
        }
        true
    }

    /// Cursor raycast. Walks the ray through the shell, sampling the
    /// subtree at every step, and returns the first position whose
    /// `(face, u_n, v_n, r_n)` cell contains solid content at
    /// `highlight_depth`. Integer cell indices live in a
    /// `3^highlight_depth` grid per axis — so `highlight_depth = 1`
    /// gives a 3×3×3 grid per face (one of 54 global "chunks" per
    /// planet), and `highlight_depth = subtree.depth` gives the
    /// finest cell.
    ///
    /// The returned `prev` is the last empty cell the ray was in
    /// immediately before the solid hit, at the same depth. This is
    /// the Minecraft-style "place against the face you're looking
    /// at" target. `prev = None` means the ray spawned already
    /// inside solid, in which case placement isn't meaningful.
    pub fn raycast(
        &self,
        lib: &NodeLibrary,
        origin: Vec3,
        dir: Vec3,
        highlight_depth: u32,
    ) -> Option<CsRayHit> {
        let oc = sdf::sub(origin, self.center);
        let b = sdf::dot(oc, dir);
        let c = sdf::dot(oc, oc) - self.outer_r * self.outer_r;
        let disc = b * b - c;
        if disc <= 0.0 { return None; }
        let sq = disc.sqrt();
        let t_enter = (-b - sq).max(0.0);
        let t_exit = -b + sq;
        if t_exit <= 0.0 { return None; }

        let shell = self.outer_r - self.inner_r;
        let d_eff = highlight_depth.clamp(1, self.depth);
        // Cap step-size resolution at 3^8 = 6561 per axis. Deeper
        // subtrees (depth 20) don't need a 3^20-granularity CPU
        // raycast; the cursor can't visibly distinguish individual
        // sub-picometer cells. The subtree walker still samples at
        // the requested `d_eff` — only the ray's stride is capped,
        // and a feature smaller than the step that the walker does
        // see will still register when its cell is sampled.
        let step_cells = 3u32.pow(d_eff.min(8)) as f32;
        let step = (shell / step_cells) * 0.5;
        if step <= 0.0 { return None; }
        let cells_at_depth = 3u32.pow(d_eff) as f32;
        let cells_max = cells_at_depth as u32 - 1;
        let mut t = t_enter + step * 0.25;
        let mut prev: Option<(Face, u32, u32, u32)> = None;
        let max_steps = 20_000usize;
        for _ in 0..max_steps {
            if t >= t_exit { break; }
            let p = sdf::add(origin, sdf::scale(dir, t));
            if let Some((face, un, vn, rn)) = self.world_to_normalized(p) {
                let (block, _) = self.sample_subtree(lib, face, un, vn, rn, d_eff);
                let iu = ((un * cells_at_depth).floor() as u32).min(cells_max);
                let iv = ((vn * cells_at_depth).floor() as u32).min(cells_max);
                let ir = ((rn * cells_at_depth).floor() as u32).min(cells_max);
                if block != 0 {
                    return Some(CsRayHit {
                        t, face, iu, iv, ir, depth: d_eff, prev,
                    });
                }
                // Update prev only when we actually move to a
                // different cell — otherwise a sub-step oversample
                // would keep overwriting prev with the same coords.
                let new_prev = Some((face, iu, iv, ir));
                if new_prev != prev {
                    prev = new_prev;
                }
            }
            t += step;
        }
        None
    }
}

/// Output of a cursor raycast against the planet.
#[derive(Copy, Clone, Debug)]
pub struct CsRayHit {
    /// Ray t at the first solid-cell sample.
    pub t: f32,
    pub face: Face,
    pub iu: u32,
    pub iv: u32,
    pub ir: u32,
    /// The depth the raycast was performed at. `iu/iv/ir` live in a
    /// `3^depth` grid.
    pub depth: u32,
    /// The cell adjacent to the hit on the ray-entry side, or `None`
    /// if the ray began already inside solid. Used as the placement
    /// target for right-click.
    pub prev: Option<(Face, u32, u32, u32)>,
}

/// The twelve edges of a cubed-sphere block as pairs of corner
/// indices into `block_corners`'s output. Useful for drawing the
/// bulged wireframe outline that Minecraft's flat-cube outline
/// becomes on a planet.
pub const BLOCK_EDGES: [(usize, usize); 12] = [
    // Bottom square (r = r0).
    (0, 1), (1, 3), (3, 2), (2, 0),
    // Top square (r = r0 + dr).
    (4, 5), (5, 7), (7, 6), (6, 4),
    // Verticals connecting bottom to top.
    (0, 4), (1, 5), (2, 6), (3, 7),
];

// ──────────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }
    fn approx_v(a: Vec3, b: Vec3) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    #[test]
    fn face_center_matches_normal() {
        for &f in &Face::ALL {
            let dir = face_uv_to_dir(f, 0.0, 0.0);
            assert!(approx_v(dir, f.normal()),
                "face {:?} center should equal its normal", f);
        }
    }

    #[test]
    fn face_corners_are_unit_vectors() {
        // Every (u, v) in [-1, 1]² should project to a unit vector.
        for &f in &Face::ALL {
            for &(u, v) in &[(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
                              (0.5, -0.3), (0.0, 0.0)] {
                let d = face_uv_to_dir(f, u, v);
                assert!(approx(sdf::length(d), 1.0),
                    "face {:?} at ({}, {}) not unit: {:?}", f, u, v, d);
            }
        }
    }

    #[test]
    fn tangents_are_orthonormal_to_normal() {
        for &f in &Face::ALL {
            let n = f.normal();
            let (ua, va) = f.tangents();
            assert!(approx(sdf::dot(ua, n), 0.0),
                "face {:?} u_axis · normal != 0", f);
            assert!(approx(sdf::dot(va, n), 0.0),
                "face {:?} v_axis · normal != 0", f);
            assert!(approx(sdf::dot(ua, va), 0.0),
                "face {:?} u_axis · v_axis != 0", f);
            assert!(approx(sdf::length(ua), 1.0));
            assert!(approx(sdf::length(va), 1.0));
        }
    }

    #[test]
    fn tangent_basis_is_right_handed() {
        // u × v should equal the outward normal (right-handed).
        for &f in &Face::ALL {
            let n = f.normal();
            let (ua, va) = f.tangents();
            let cross = [
                ua[1] * va[2] - ua[2] * va[1],
                ua[2] * va[0] - ua[0] * va[2],
                ua[0] * va[1] - ua[1] * va[0],
            ];
            assert!(approx_v(cross, n),
                "face {:?}: u × v = {:?}, expected {:?}", f, cross, n);
        }
    }

    #[test]
    fn world_to_coord_inverts_coord_to_world() {
        let center = [1.5, 1.5, 1.5];
        // Try a bunch of face/uv/r combinations.
        let cases = [
            (Face::PosX, 0.0, 0.0, 0.5),
            (Face::NegY, 0.3, -0.7, 1.2),
            (Face::PosZ, -0.9, 0.9, 0.01),
            (Face::NegX, 0.6, 0.4, 2.0),
            (Face::PosY, -0.2, -0.2, 0.8),
            (Face::NegZ, 0.0, 0.0, 1.0),
        ];
        for &(face, u, v, r) in &cases {
            let world = coord_to_world(center, CubeSphereCoord { face, u, v, r });
            let back = world_to_coord(center, world).unwrap();
            assert_eq!(back.face, face, "face mismatch for {:?}", (face, u, v, r));
            assert!(approx(back.u, u), "u mismatch: {} vs {}", back.u, u);
            assert!(approx(back.v, v), "v mismatch: {} vs {}", back.v, v);
            assert!(approx(back.r, r), "r mismatch: {} vs {}", back.r, r);
        }
    }

    #[test]
    fn adjacent_cells_share_edges() {
        // Two neighboring cells on the same face, sharing the
        // edge u = u_shared: their matching corners must be
        // bit-identical in world space (same call → same floats).
        let center = [0.0, 0.0, 0.0];
        let (face, v0, v1, r0, r1) = (Face::PosX, -0.2, -0.1, 1.0, 1.01);
        let shared_u = 0.5;

        let left_right_u1 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v0, r: r0 });
        let right_left_u0 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v0, r: r0 });
        assert_eq!(left_right_u1, right_left_u0,
            "cells meeting at u={shared_u} on the same face must share corners exactly");

        // And at the outer radius.
        let lr1 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v1, r: r1 });
        let rl1 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v1, r: r1 });
        assert_eq!(lr1, rl1);
    }

    #[test]
    fn faces_meet_at_cube_edges() {
        // The seam between +X face and +Y face lives on the line
        // x = y = 1 in cube-space. On +X this is v = 1 (since
        // v_axis = +Y); on +Y this is u = 1 (since u_axis = +X).
        // Both parameterize the same seam line with a free z
        // coordinate — varied here by u on +X and v on +Y.
        for &t in &[-0.8, -0.3, 0.0, 0.3, 0.8] {
            let on_posx = face_uv_to_dir(Face::PosX, t, 1.0);
            let on_posy = face_uv_to_dir(Face::PosY, 1.0, t);
            assert!(approx_v(on_posx, on_posy),
                "seam +X/+Y at t={}: {:?} vs {:?}", t, on_posx, on_posy);
        }
    }

    #[test]
    fn block_corners_have_radial_bulge() {
        // Inner-face corners are at radius r, outer at r + dr.
        let center = [0.0, 0.0, 0.0];
        let corners = block_corners(center, Face::PosX, -0.1, -0.1, 1.0, 0.2, 0.2, 0.05);
        for i in 0..4 {
            let len = sdf::length(corners[i]);
            assert!(approx(len, 1.0), "inner corner {} not at r=1: len={}", i, len);
        }
        for i in 4..8 {
            let len = sdf::length(corners[i]);
            assert!(approx(len, 1.05), "outer corner {} not at r=1.05: len={}", i, len);
        }
    }

    #[test]
    fn block_edges_reference_valid_corners() {
        for &(a, b) in &BLOCK_EDGES {
            assert!(a < 8 && b < 8);
            assert_ne!(a, b);
        }
        assert_eq!(BLOCK_EDGES.len(), 12);
    }

    #[test]
    fn world_to_coord_picks_correct_face() {
        let c = [0.0, 0.0, 0.0];
        // Strong +X direction.
        assert_eq!(world_to_coord(c, [1.0, 0.1, 0.1]).unwrap().face, Face::PosX);
        // Strong -Y direction.
        assert_eq!(world_to_coord(c, [0.1, -1.0, 0.1]).unwrap().face, Face::NegY);
        // Strong +Z direction.
        assert_eq!(world_to_coord(c, [0.2, 0.2, 1.0]).unwrap().face, Face::PosZ);
    }

    fn test_sdf(radius: f32, noise: f32) -> Planet {
        Planet {
            center: [0.0, 0.0, 0.0],
            radius,
            noise_scale: noise,
            noise_freq: 5.0,
            noise_seed: 1,
            gravity: 1.0,
            influence_radius: radius * 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        }
    }

    /// Wrap a freshly generated planet in a minimal `WorldState` whose
    /// root IS the body node. With `body_anchor = Path::root()`,
    /// `set_cell_at_depth`'s ancestor walk is a no-op and the test
    /// exercises the face-root + body-node rebuild path cleanly.
    fn world_with_body_as_root(
        mut lib: NodeLibrary,
        body_node: NodeId,
    ) -> (super::super::state::WorldState, super::super::coords::Path) {
        lib.ref_inc(body_node);
        (
            super::super::state::WorldState { root: body_node, library: lib },
            super::super::coords::Path::root(),
        )
    }

    /// Walk into a face's subtree at the given slot path and return
    /// the resolved child (Empty / Block / Node).
    fn walk_subtree(
        lib: &NodeLibrary,
        root: NodeId,
        path: &[(usize, usize, usize)],
    ) -> Child {
        let mut node = lib.get(root).unwrap();
        let mut current_child = Child::Node(root);
        for &(us, vs, rs) in path {
            let slot = slot_index(us, vs, rs);
            current_child = node.children[slot];
            match current_child {
                Child::Node(id) => node = lib.get(id).unwrap(),
                _ => return current_child,
            }
        }
        current_child
    }

    #[test]
    fn body_node_carries_cubed_sphere_body_kind() {
        let mut lib = NodeLibrary::default();
        let planet = generate_spherical_planet(
            &mut lib,
            [1.5, 2.3, 1.5],
            0.12, 0.52, 6,
            &Planet {
                center: [1.5, 2.3, 1.5],
                radius: 0.32,
                noise_scale: 0.0,
                noise_freq: 1.0,
                noise_seed: 0,
                gravity: 0.0,
                influence_radius: 1.0,
                surface_block: crate::world::palette::block::GRASS,
                core_block: crate::world::palette::block::STONE,
            },
        );
        let body = lib.get(planet.body_node).expect("body node must exist");
        match body.kind {
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                assert_eq!(inner_r, 0.12);
                assert_eq!(outer_r, 0.52);
            }
            _ => panic!("body node must have CubedSphereBody kind, got {:?}", body.kind),
        }
        // Face-center slots must each resolve to a Node in the library.
        for &face in &Face::ALL {
            let slot = body_face_center_slot(face);
            match body.children[slot] {
                Child::Node(id) => assert!(lib.get(id).is_some()),
                other => panic!("expected Node at face-center slot, got {:?}", other),
            }
        }
        // `face_root_of` agrees with the raw slot lookup.
        for &face in &Face::ALL {
            let id = face_root_of(&lib, planet.body_node, face).unwrap();
            assert!(lib.get(id).is_some());
        }
    }

    #[test]
    fn spherical_planet_has_six_faces() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.05);
        let planet = generate_spherical_planet(&mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 3, &sdf);
        for &face in &Face::ALL {
            let id = face_root_of(&lib, planet.body_node, face)
                .expect("face root must resolve");
            assert!(lib.get(id).is_some(), "every face root must exist in library");
        }
    }

    #[test]
    fn spherical_planet_outer_is_empty_inner_is_solid() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        // Inner 0.5, outer 1.5 → SDF surface at r=1 is the midpoint.
        // Bottom (rs=0) layer is deep inside solid; top (rs=2) is deep air.
        let planet = generate_spherical_planet(&mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 2, &sdf);
        // Walk one level into the +X face, middle u/v, bottom rs: solid.
        let inner = walk_subtree(&lib, face_root_of(&lib, planet.body_node, Face::PosX).unwrap(), &[(1, 1, 0)]);
        match inner {
            Child::Block(_) => {}
            Child::Node(id) => {
                // Descend one more step; deepest layer should contain Block terminals.
                let node = lib.get(id).unwrap();
                assert!(node.children.iter().any(|c| matches!(c, Child::Block(_))),
                    "inner subtree should contain solid blocks");
            }
            Child::Empty => panic!("inner slot at r_lo should be solid"),
        }
        // Top slot: empty.
        let outer = walk_subtree(&lib, face_root_of(&lib, planet.body_node, Face::PosX).unwrap(), &[(1, 1, 2)]);
        assert!(matches!(outer, Child::Empty | Child::Node(_)),
            "outer slot should resolve to empty or an empty subtree");
    }

    #[test]
    fn set_cell_at_depth_clears_solid_region() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        let mut planet = generate_spherical_planet(
            &mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 3, &sdf,
        );
        let (mut world, body_anchor) = world_with_body_as_root(lib, planet.body_node);
        let (before, _) = planet.sample_subtree(&world.library, Face::PosX, 0.5, 0.5, 0.5, 1);
        assert!(before != 0, "middle chunk should be solid before break");
        let changed = planet.set_cell_at_depth(
            &mut world, &body_anchor, Face::PosX, 1, 1, 1, 1, Child::Empty,
        );
        assert!(changed, "edit should change the face root");
        let (after, _) = planet.sample_subtree(&world.library, Face::PosX, 0.5, 0.5, 0.5, 1);
        assert_eq!(after, 0, "broken chunk must read as empty");
    }

    #[test]
    fn raycast_reports_prev_empty_cell_for_placement() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        let planet = generate_spherical_planet(
            &mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 3, &sdf,
        );
        // Fire a ray from far +X back toward origin. It enters the
        // shell above the SDF surface (empty), then drops into the
        // solid interior. The last empty cell before the hit is the
        // placement target.
        let hit = planet.raycast(&lib, [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(hit.face, Face::PosX);
        assert!(hit.prev.is_some(), "should have an adjacent empty cell");
        let (prev_face, prev_iu, prev_iv, prev_ir) = hit.prev.unwrap();
        assert_eq!(prev_face, Face::PosX);
        // The prev cell sits one radial step outward from the hit.
        assert!(prev_ir > hit.ir, "placement cell should be radially outside the hit");
        assert_eq!((prev_iu, prev_iv), (hit.iu, hit.iv),
            "placement cell shares u/v columns with the hit");
    }

    #[test]
    fn set_cell_at_depth_places_block_on_adjacent_empty() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        let mut planet = generate_spherical_planet(
            &mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 3, &sdf,
        );
        let hit = planet.raycast(&lib, [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 2).unwrap();
        let (face, iu, iv, ir) = hit.prev.unwrap();
        let (mut world, body_anchor) = world_with_body_as_root(lib, planet.body_node);
        let changed = planet.set_cell_at_depth(
            &mut world, &body_anchor, face, iu, iv, ir, hit.depth,
            Child::Block(block::BRICK),
        );
        assert!(changed);
        // Confirm the new block stuck.
        let cells = 3.0f32.powi(hit.depth as i32);
        let un = (iu as f32 + 0.5) / cells;
        let vn = (iv as f32 + 0.5) / cells;
        let rn = (ir as f32 + 0.5) / cells;
        let (b, _) = planet.sample_subtree(&world.library, face, un, vn, rn, hit.depth);
        assert_eq!(b, block::BRICK);
    }

    #[test]
    fn set_cell_at_depth_handles_max_planet_depth() {
        // Regression: the slots-path array inside `set_cell_at_depth`
        // must hold as many entries as the deepest planet's depth.
        // At depth 20 the array needs ≥20 slots — 16 overflows.
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(0.34, 0.04);
        let mut planet = generate_spherical_planet(
            &mut lib, [0.0, 0.0, 0.0], 0.12, 0.52, 20, &sdf,
        );
        let (mut world, body_anchor) = world_with_body_as_root(lib, planet.body_node);
        let iu_last = 3u32.pow(20) - 1;
        planet.set_cell_at_depth(
            &mut world, &body_anchor, Face::PosX, iu_last, iu_last, iu_last, 20,
            Child::Empty,
        );
    }

    #[test]
    fn set_cell_at_depth_noop_for_already_empty_cell() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        let mut planet = generate_spherical_planet(
            &mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 3, &sdf,
        );
        let (mut world, body_anchor) = world_with_body_as_root(lib, planet.body_node);
        let changed = planet.set_cell_at_depth(
            &mut world, &body_anchor, Face::PosX, 1, 1, 2, 1, Child::Empty,
        );
        assert!(!changed, "breaking an already-empty cell should no-op");
    }

    #[test]
    fn depth_20_planet_generates_bounded_node_count() {
        // Regression: without the SDF_DETAIL_LEVELS budget cap, a
        // depth-20 planet's surface band would recurse ~3^40 times.
        // With the cap in place, SDF-driven recursion stops at
        // ~27^SDF_DETAIL_LEVELS cells per face + O(depth) filler
        // nodes total. Should finish in well under a second and use
        // a few thousand unique nodes.
        let start = std::time::Instant::now();
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(0.34, 0.04);
        let _ = generate_spherical_planet(
            &mut lib, [0.0, 0.0, 0.0], 0.12, 0.52, 20, &sdf,
        );
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs_f64() < 5.0,
            "depth-20 planet generation took {:?} (regression?)", elapsed);
        assert!(lib.len() < 200_000,
            "depth-20 planet produced {} unique nodes (regression?)", lib.len());
    }

    #[test]
    fn spherical_planet_dedup_keeps_node_count_modest() {
        // A fully pristine sphere with a simple SDF should produce a
        // small number of unique nodes — most subtrees collapse into
        // a single uniform-empty / uniform-solid cache entry.
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        let _ = generate_spherical_planet(&mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 4, &sdf);
        // Depth 4 means up to 27^4 ≈ 530k potential unique cells per
        // face; dedup should bring us well under that.
        assert!(lib.len() < 20_000,
            "dedup'd spherical planet should have < 20k unique nodes, got {}",
            lib.len());
    }

    #[test]
    fn face_uv_stays_in_unit_square_for_on_face_points() {
        // Any direction on or near a face (tilted less than 45°
        // from the normal) should yield |u|, |v| ≤ 1.
        for &f in &Face::ALL {
            for &tilt in &[0.1, 0.3, 0.7] {
                // Tilt by `tilt` along each tangent.
                let (ua, va) = f.tangents();
                let dir = sdf::normalize(sdf::add(
                    f.normal(),
                    sdf::add(sdf::scale(ua, tilt), sdf::scale(va, tilt)),
                ));
                let c = world_to_coord([0.0, 0.0, 0.0], dir).unwrap();
                assert_eq!(c.face, f, "tilt {} on face {:?} picked {:?}", tilt, f, c.face);
                assert!(c.u.abs() <= 1.0 + 1e-5, "u = {} out of range", c.u);
                assert!(c.v.abs() <= 1.0 + 1e-5, "v = {} out of range", c.v);
            }
        }
    }
}
