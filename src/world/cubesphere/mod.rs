//! Cubed-sphere module: geometry, ribbon-pop frame state, worldgen.
//!
//! A spherical body lives inside the Cartesian voxel tree as a
//! [`NodeKind::CubedSphereBody`] node. Its 27 children are laid out
//! in XYZ like any Cartesian node, but six specific slots (the
//! face-centers) hold face subtrees, one slot (the body center)
//! holds a uniform-stone core, and the 20 edge/corner slots are
//! empty.
//!
//! Inside a face subtree, a node's 27 children are interpreted in
//! `(u, v, r)` slot order — `slot_index(us, vs, rs)`. The face root
//! carries [`NodeKind::CubedSphereFace`]; deeper nodes stay
//! `Cartesian` (the UVR convention is contagious along the descent
//! path).
//!
//! Submodules:
//!   * [`geometry`] — `Face`, slot layout, equal-angle warp, body
//!     ↔ face-space conversions, ray-sphere entry.
//!   * [`frame`]    — [`FaceFrame`], ribbon-popped per-cell plane
//!     normals for precision-stable ray marching at arbitrary
//!     face-subtree depth.
//!   * [`worldgen`] — building a planet body from a `Planet` SDF
//!     and installing it into the world tree.
//!
//! [`NodeKind::CubedSphereBody`]: crate::world::tree::NodeKind::CubedSphereBody
//! [`NodeKind::CubedSphereFace`]: crate::world::tree::NodeKind::CubedSphereFace

pub mod frame;
pub mod geometry;
pub mod worldgen;

pub use frame::FaceFrame;
pub use geometry::{
    body_point_to_face_space, cube_to_ea, ea_to_cube, face_space_to_body_point,
    face_uv_to_dir, find_body_ancestor_in_path, pick_face, ray_outer_sphere_hit,
    Face, FacePoint, CORE_SLOT, FACE_SLOTS,
};
pub use worldgen::{demo_planet, insert_spherical_body, install_at_root_center, PlanetSetup};
