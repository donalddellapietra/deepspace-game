# Spherical Wrapped Plane Sketch

Goal: make the `WrappedPlane` storage layout the ground truth, while
rendering and interacting with it as a clean UV sphere.

The important constraint from the current tangent-plane branch is:
camera state must not transition when the renderer enters or leaves a
local frame. The canonical camera remains `WorldPos + yaw/pitch`.
Every render/raycast frame is a pure projection of that state.

## Ground Truth

The tree stays parameter-space first:

```text
NodeKind::WrappedPlane { dims, slab_depth }
  x axis: longitude, wraps modulo dims.x
  y axis: radial/material layer
  z axis: latitude, bounded by lat_max
```

Slot paths under the `WrappedPlane` are interpreted as recursive
subdivision of `(lon, r, lat)`:

```text
slot.x -> lon third
slot.y -> r third
slot.z -> lat third
```

That means edits, movement wrap, persistence, and path identity are
all still `WrappedPlane` facts. The sphere is an interpretation of
that parameter space for rendering and picking, not a separate world.

## What Not To Do

Do not implement this as "a `TangentBlock` rotation on every cell" and
call it done.

That solves orientation only. A UV sphere also needs:

- spherical placement of each parameter cell;
- radial shell traversal;
- longitude/latitude boundary crossing;
- latitude-dependent metric scale;
- matching CPU raycast path selection;
- a local frame when the camera is deep inside one parameter cell.

Existing `TbBoundary` works for a child that still occupies a
Cartesian parent slot. A sphere cell is not selected by entering a
Cartesian slot. It is selected by intersecting `(lon, r, lat)` cell
boundaries. The right abstraction is TB-inspired, not TB-identical.

## Core Rule

The renderer may switch frames. The camera must not.

Canonical state:

```text
camera.position: WorldPos
camera.yaw/pitch or basis
```

Per-frame derived state:

```text
CartesianFrameCamera
TangentBlockFrameCamera
SphereBodyFrameCamera
SphereSubFrameCamera
```

All of those are pure projections from the canonical camera. No
render-frame selection code may rewrite `camera.position`, snap
anchor paths, or reconstruct the camera from local coordinates.

TAA history may reset when the render-frame signature changes. That
is not a camera transition.

## Frame Levels

There are two useful sphere frames.

### 1. Body Frame

Rooted at the `WrappedPlane` node. Its local coordinates are the
existing `[0, 3)^3` WP frame. The sphere center is `(1.5, 1.5, 1.5)`.
The sphere radius is derived from the body size, currently:

```text
r_sphere = 3.0 / (2π)
shell_thickness = r_sphere * 0.25
```

The body-frame marcher is good when the camera is shallow or outside
the body. It ray-intersects radial shells, maps surface points to
`(lon, lat)`, samples the slab cell, then descends inside that
parameter cell.

This is close to the old `sphere_dda.wgsl` idea, but it must be wired
into the current frame-local camera/ribbon system.

### 2. Sphere Subframe

Rooted at a path below the `WrappedPlane`. The path defines a bounded
range:

```text
lon_lo..lon_hi
r_lo..r_hi
lat_lo..lat_hi
```

The subframe origin is the center of that range on the sphere:

```text
center = sphere_center + r_c * radial(lon_c, lat_c)
```

The basis is:

```text
x = longitude tangent / east
y = latitude tangent / north
z = radial outward
```

The camera projection is:

```text
cam_wp = camera.position.in_frame(wp_path)
cam_sub.origin = basis^T * (cam_wp - center)
cam_sub.forward = basis^T * camera.forward
cam_sub.right = basis^T * camera.right
cam_sub.up = basis^T * camera.up
```

This is where deep precision comes from. The numbers are bounded by
distance to the local parameter cell, not by the whole body.

## Render-Frame Selection

`compute_render_frame` currently stops at `WrappedPlane`. For sphere
mode it should produce one of:

```rust
enum ActiveFrameKind {
    Cartesian,
    WrappedPlane { dims: [u32; 3], slab_depth: u8 },
    SphereBody { dims: [u32; 3], slab_depth: u8, lat_max: f32 },
    SphereSubFrame {
        wp_path: Path,
        render_path: Path,
        range: SphereRange,
    },
}
```

The initial version can skip the enum expansion and use existing
`WrappedPlane` as "sphere body" mode. But the design should leave room
for `SphereSubFrame`, because that is the fix for deep precision.

Frame choice should be based on the camera anchor path:

1. If the active path has not entered a `WrappedPlane`, render as
   Cartesian.
2. If it has entered a `WrappedPlane` but is near the body scale,
   render as `SphereBody`.
3. If the camera anchor is sufficiently deep below the `WrappedPlane`,
   pick a sphere subframe along the camera path and render as
   `SphereSubFrame`.

The selected frame is disposable. Recompute it every frame.

## Sphere Range Helper

Add a pure helper:

```rust
pub struct SphereRange {
    pub lon_lo: f32,
    pub lon_hi: f32,
    pub lat_lo: f32,
    pub lat_hi: f32,
    pub r_lo: f32,
    pub r_hi: f32,
    pub wp_path_depth: u8,
    pub dims: [u32; 3],
    pub slab_depth: u8,
}

pub fn sphere_range_for_path(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
    lat_max: f32,
) -> Option<SphereRange>;
```

It walks to the first `WrappedPlane` node, then interprets the
remaining path slots as `(lon, r, lat)` thirds.

This helper is the shared contract for:

- render-frame selection;
- GPU uniform generation;
- CPU raycast;
- tests that prove path-to-range stability.

## GPU Shape

Start with a body-frame sphere shader path:

```wgsl
fn march_sphere_body(
    wp_node_idx: u32,
    ray_origin_wp: vec3<f32>,
    ray_dir_wp: vec3<f32>,
    dims: vec3<u32>,
    slab_depth: u32,
    lat_max: f32,
) -> HitResult
```

It should:

1. Intersect the outer sphere / shell.
2. For each candidate radial layer, compute `(lon, lat, r)`.
3. Convert `(lon, r, lat)` to a slab cell index.
4. Walk the existing tree storage to that slab cell.
5. Descend deeper by splitting `(lon, r, lat)` thirds, not by
   Cartesian DDA.

Then add:

```wgsl
fn march_sphere_subframe(
    sub_node_idx: u32,
    ray_origin_sub: vec3<f32>,
    ray_dir_sub: vec3<f32>,
    node_range: SphereRangeUniform,
    frame_basis_metadata: SphereSubFrameUniform,
) -> HitResult
```

This does the same partition walk, but all ray math runs in the
subframe basis. It converts subframe-local points back to body-sphere
`(lon, lat, r)` only for boundary tests and cell selection.

## CPU Raycast

CPU picking must mirror the GPU partition exactly.

Add a sphere raycast path for `ActiveFrameKind::SphereBody` and
`SphereSubFrame`. It returns normal `HitInfo` with full world-tree
paths:

```rust
pub fn cpu_raycast_sphere_body(...) -> Option<HitInfo>;
pub fn cpu_raycast_sphere_subframe(...) -> Option<HitInfo>;
```

The returned path is still:

```text
world root -> ... -> WrappedPlane -> lon/r/lat slots -> deeper slots
```

Do not return synthetic sphere cells. Edits apply to the same
`WrappedPlane` tree paths the storage already owns.

## Movement And Wrap

Movement continues to use `WorldPos` and the existing
`WrappedPlane` X-wrap behavior. The longitude axis is still the path
axis that wraps.

Sphere rendering does not change movement semantics. It only changes
how those path cells are projected into pixels and ray hits.

## Tests

The first implementation should land with tests before visuals:

1. `sphere_range_for_path` returns full body range at the
   `WrappedPlane` root.
2. One slot below WP maps `slot.x` to lon third, `slot.y` to radial
   third, `slot.z` to lat third.
3. Body-frame CPU raycast aimed at the equator picks the expected
   `(lon, r, lat)` slab path.
4. Body-frame CPU raycast and shader debug probe agree on a fixed
   pixel.
5. Subframe camera projection is pure: projecting the same canonical
   camera into adjacent frames changes local coordinates but does not
   mutate `camera.position`.
6. Deep subframe raycast still resolves a subcell path below the slab
   without root-frame coordinate collapse.

## Implementation Order

1. Add `sphere_range_for_path` and unit tests.
2. Add a CPU body-frame sphere raycast that mirrors the old UV sphere
   partition but uses current `Path`, `WorldPos`, and `WrappedPlane`
   types.
3. Wire `frame_aware_raycast` to optionally use the sphere raycast for
   `WrappedPlane` frames.
4. Split the current `march_wrapped_planet` path behind a clear
   render mode: tangent-cube sphere vs UV sphere.
5. Add body-frame UV sphere shader path and harness screenshot test.
6. Add sphere subframe camera projection helper.
7. Extend active-frame selection to choose sphere subframes under
   deep anchors.
8. Add subframe shader path and CPU raycast mirror.

## Non-Goals For This Branch

- No camera blending when entering/exiting sphere subframes.
- No mutation of canonical camera state during render-frame changes.
- No attempt to make per-cell `TangentBlock` rotation alone represent
  the sphere.
- No new persistence model separate from `WrappedPlane`.
