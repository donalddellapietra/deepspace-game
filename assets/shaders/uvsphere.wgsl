// UV-sphere ray marcher.
//
// The body's tree is a recursive 27-children layout with children
// indexed by `pt + tt*3 + rt*9` along `(φ, θ, r)`. Two march paths
// share most primitives:
//
// - `march_uv_sphere` (`march_root.wgsl`) — body-root frame, per-cell
//   basis recomputed from world `(φ, θ, r)` each iteration. Used at
//   shallow zoom where the body fits in the rendered frame.
//
// - `march_uv_subcell` (`march_subcell.wgsl`) — frame is a UV cell
//   nested at body-tree depth K ≥ 1. Frame-centre basis is constant;
//   ray state is the world parameter `t` and frame-level cell-local
//   fractions `un_*_frame = cam_un + d_un_frame · t`. No
//   `(phi_w − phi_min) / frame_dphi` evaluation per iteration —
//   that's the cancellation cliff that produced the bevel swarm.
//
// Top-level dispatch lives in `march.wgsl`'s `march()`.

#include "bindings.wgsl"
#include "uvsphere/types.wgsl"
#include "uvsphere/basis.wgsl"
#include "uvsphere/cell.wgsl"
#include "uvsphere/march_root.wgsl"
#include "uvsphere/march_subcell.wgsl"
