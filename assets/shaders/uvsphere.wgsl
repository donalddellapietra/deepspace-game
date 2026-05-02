// UV-sphere ray marcher.
//
// The body's tree is a recursive 27-children layout with children
// indexed by `pt + tt*3 + rt*9` along `(φ, θ, r)`. Single render
// path: `march_uv_sphere` walks from the body root using
// delta-tracked descent (precision-stable at any depth) and steps
// the ray via absolute-bound ray-vs-{φ-plane, θ-cone, r-sphere}
// intersections. Mirrors the CPU walker in
// `src/world/raycast/uvsphere.rs`.
//
// Top-level dispatch lives in `march.wgsl`'s `march()`.

#include "bindings.wgsl"
#include "uvsphere/types.wgsl"
#include "uvsphere/cell.wgsl"
#include "uvsphere/proto_block.wgsl"
#include "uvsphere/march_root.wgsl"
