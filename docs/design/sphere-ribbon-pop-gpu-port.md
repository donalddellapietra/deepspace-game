# Sphere ribbon-pop — GPU port

Stage 4 of the sphere refactor. Mirrors `cs_raycast_local` on the
GPU so the shader terminates at the same cell the CPU raycast picks,
at any face-subtree depth.

**Mandatory constraint:** lands as one commit. Partial shader state
produces garbage pixels indistinguishable from real bugs
(`feedback_no_intermediate_visual_states`).

## What the shader needs to do

For a render frame of kind `SphereSub`, the ray-march algorithm is:

1. Transform the ray from the sub-frame's camera-space basis into
   the sub-frame's **local** `[0, 3)³` frame.
2. Step cell-by-cell through the local frame using axis-aligned
   boundaries at integer local K. At each step, walk the face
   subtree from the sub-frame's root node down to the LOD walker
   limit and shade the returned block (or advance past the empty
   cell).
3. Exit when the ray leaves `[0, 3)³` — same semantics as `Body` /
   `Cartesian` termination.

This is the pixel-level mirror of `src/world/raycast/sphere_sub.rs`.
Logic parity is what gives us three-way agreement (CPU raycast +
GPU shader + highlight AABB all converge on the same cell).

## Uniform layout

Extends `GpuUniforms` in `src/renderer/mod.rs` and `Uniforms` in
`assets/shaders/bindings.wgsl`:

    /// New root kind.
    const ROOT_KIND_SPHERE_SUB: u32 = 3u;

    /// Sub-frame metadata (used iff root_kind == 3).
    sub_c_body: vec4<f32>,           // xyz = body-XYZ of local (0,0,0); w unused
    sub_j_col0: vec4<f32>,           // xyz = J column 0 (∂body/∂u_l); w unused
    sub_j_col1: vec4<f32>,           // xyz = J column 1 (∂body/∂v_l)
    sub_j_col2: vec4<f32>,           // xyz = J column 2 (∂body/∂r_l)
    sub_j_inv_col0: vec4<f32>,       // xyz = J_inv column 0; w unused
    sub_j_inv_col1: vec4<f32>,
    sub_j_inv_col2: vec4<f32>,
    sub_face_corner: vec4<f32>,      // x=un_corner, y=vn_corner, z=rn_corner, w=frame_size
    sub_meta: vec4<u32>,             // x=face (0..5), y=inner_r_bits, z=outer_r_bits, w=unused

All vec4-aligned; std140/uniform buffer layout requires 16 B stride
for vec3 anyway. Face index as u32; radii as f32 bit patterns stored
in u32 fields (or as a separate vec4<f32>). Clean option — keep the
existing `root_radii` for `inner_r`/`outer_r`; reuse.

Total additional uniform bytes: 8 × 16 = 128 B. Current uniform
buffer is ~128 B; doubles. Well within wgpu's 64 KB uniform limit.

## Shader file layout

One new WGSL function in `assets/shaders/sphere.wgsl`:

    fn sphere_in_sub_frame(
        sub_frame_node_idx: u32,
        ray_origin_local: vec3<f32>,   // in [0, 3)³ local
        ray_dir_local: vec3<f32>,      // local basis (J_inv applied upstream)
        walker_limit: u32,
    ) -> HitResult { ... }

And supporting:

    fn walk_sub_frame(
        node_idx: u32,
        u_l: f32, v_l: f32, r_l: f32,
        max_depth: u32,
    ) -> FaceWalkResult { ... }   // reuses the tag/rank/occupancy walker

Both are ports of the CPU versions line-for-line. The walker can
reuse `face_walk_result` struct — the sub-frame is structurally a
face subtree, just rooted deeper.

## Main shader dispatch

`assets/shaders/main.wgsl`'s top-level `march` gets a third branch:

    switch uniforms.root_kind {
        case ROOT_KIND_CARTESIAN: { march_cartesian(...); }
        case ROOT_KIND_BODY:      { sphere_in_cell(..., window_active = 0u); }
        case ROOT_KIND_SPHERE_SUB: {
            sphere_in_sub_frame(
                root_index,
                camera.position,   // already in sub-frame local
                ray_dir,           // already in sub-frame local
                uniforms.max_depth,
            );
        }
        default: { ... }
    }

## Camera transform (CPU side)

`src/app/mod.rs::gpu_camera_for_frame` grows a `SphereSub` arm:

    ActiveFrameKind::SphereSub(sub) => {
        let pos_local = self.camera.position.in_frame(&frame.render_path);
        let (fwd_world, right_world, up_world) = self.camera.basis();
        // Basis in sub-frame local via J_inv — directions only.
        let fwd_local   = mat3_mul_vec(&sub.j_inv, fwd_world);
        let right_local = mat3_mul_vec(&sub.j_inv, right_world);
        let up_local    = mat3_mul_vec(&sub.j_inv, up_world);
        // Upload. Shader sees pos + basis in sub-frame local coords
        // directly — no further transform needed mid-shader.
    }

Magnitude of the local basis vectors scales as O(3^depth) (large,
but within f32 range). DDA operates on t ratios; basis magnitude
doesn't affect ordering.

**Design choice**: CPU-side J_inv (one-time per frame) instead of
shader-side J_inv (per-pixel). Saves ~45 MUL/ADD per pixel.

## Renderer API

`src/renderer/mod.rs` adds:

    pub const ROOT_KIND_SPHERE_SUB: u32 = 3;

    pub fn set_root_kind_sphere_sub(
        &mut self,
        inner_r: f32, outer_r: f32, face: u32,
        corner_and_size: [f32; 4],  // un, vn, rn, frame_size
        c_body: [f32; 3],
        j: [[f32; 3]; 3],
        j_inv: [[f32; 3]; 3],
    ) {
        self.root_kind = ROOT_KIND_SPHERE_SUB;
        self.root_radii = [inner_r, outer_r, 0.0, 0.0];
        // Pack into the new uniform slots.
    }

## Dispatch from `edit_actions/upload.rs`

Remove the current body-fallback:

    match self.active_frame.kind {
        ActiveFrameKind::SphereSub(sub) => {
            renderer.set_root_kind_sphere_sub(
                sub.inner_r, sub.outer_r, sub.face as u32,
                [sub.un_corner, sub.vn_corner, sub.rn_corner, sub.frame_size],
                sub.c_body, sub.j, sub.j_inv,
            );
        }
        ActiveFrameKind::Body { inner_r, outer_r } => {
            renderer.set_root_kind_body(inner_r, outer_r);
        }
        ActiveFrameKind::Cartesian => {
            renderer.set_root_kind_cartesian();
        }
    }

## Tests

All land in the same commit as the shader. Without them, the shader
is unverified.

### Unit tests (CPU-side, via harness)

* `gpu_camera_for_sphere_sub_basis_matches_j_inv`: given a
  `SphereSubFrame`, verify `fwd_local = J_inv · fwd_world` (close to
  1e-4 relative tol). Isolates the CPU transform from the shader.

### Visual regression (render-harness)

All driven by `--render-harness` headless render + PNG diff.

1. **`sphere_sub_body_transition_silhouette.rs`**: same camera
   position, two renders — once at anchor depth 2 (forces `Body`
   kind), once at anchor depth 3 (forces `SphereSub`). Extract the
   sphere's silhouette from each (alpha or
   edge-detection). Pixel-wise row-count difference along the
   silhouette must be < 1 row across ≥ 95 % of the silhouette
   height. This is the seamlessness proof.

2. **`sphere_sub_depth_35_renders.rs`**: anchor at face-subtree
   depth 35. Render must produce a non-sky pixel at the image
   center (crosshair). Proves the shader descends past the body-
   march precision wall without artifacts.

3. **`sphere_sub_zoom_continuity.rs`**: capture screenshots at
   anchor depths {2, 3, 4, 5, 10, 20, 30}. Pairwise-diff adjacent
   depths; SSIM ≥ 0.98 for the sphere region. Catches any
   discontinuity when the frame rebases between levels.

### CPU/GPU agreement

4. **`cpu_gpu_sub_frame_hit_cell_matches.rs`**: same camera, same
   ray direction. CPU `cpu_raycast_in_sub_frame` gets the hit path.
   GPU reports pixel color → map back to cell via the highlight
   AABB path. Both must identify the same cell (path prefix match).
   At several sub-frame depths (3, 10, 25).

### Math sanity

5. **`sub_j_inv_basis_round_trip.rs`**: CPU-side check that for the
   standard basis vectors `{e_x, e_y, e_z}` in body, the
   round-trip `J · (J_inv · e) ≈ e` to 1e-4. Already covered in
   `sphere_sub.rs` tests but worth restating for the uniform-pack
   path.

## Pack (no changes expected)

The CPU pack produces the same tree structure for `CubedSphereFace`
and Cartesian face-subtree nodes. The sub-frame's node_idx is just
an entry in the existing `node_offsets_buffer`. No pack-side
changes needed — the shader reuses the existing face-subtree walker
machinery.

## Gotchas & risks

1. **f32 precision in uniform upload of J_inv**. `|J_inv| = O(3^depth)`;
   at depth 30 that's 2e14. Representable in f32 (max 3e38). But
   `write_buffer` stores as f32 — no f64 precision retained even
   though CPU computed J_inv in f64. Mitigation: the shader uses the
   final f32 values; CPU's f64 computation matters for the cofactor
   arithmetic, but the stored result is f32 throughout.

2. **Camera basis magnitude after `J_inv`**. `|J_inv · forward| = O(3^depth)`.
   Fine in f32, but `pos + basis * ndc` in the shader advances local
   coords by `O(ndc × 3^depth)`. At ndc ~1 and depth 30, `ro + t_max × rd`
   could be 1e14. If t_max is clamped small (local box is [0, 3)³,
   t_max = O(1 / |rd_local|) ≈ O(3^-depth)), so `t_max · rd = O(1)`.
   Math stays bounded. Same nudge-discipline as the CPU DDA
   (`t_nudge = t_span * 1e-5`, never clamped to absolute floor).

3. **Walker limit at deep sub-frames**. The shader's walker cost is
   `O(walker_limit)` per DDA step. Typical walker_limit ≈ render_margin
   (≈ 3). At depth 35 + walker 3 = total depth 38 — well within
   f32 precision inside the local frame. No change needed.

4. **Shader uniform buffer alignment**. Each `vec3` field in a
   uniform block pads to 16 B. The plan uses `vec4` throughout to
   make the alignment explicit; Rust-side `#[repr(C)]` must match.
   Add a `#[test] fn uniform_layout_matches_wgsl` that compares the
   Rust struct size to a WGSL-generated reference (hand-coded byte
   count). 128 B new, total 256 B — comfortably aligned.

5. **Bind group layout changes**. Uniform buffer grows but doesn't
   need a new binding slot — same `uniforms` binding. Updated
   buffer size in the bind group descriptor, nothing else.

6. **Fallback for pre-transition depths**. At face-subtree depth
   < 3 the frame is still `Body`; the body march handles it. No
   shader change needed for that path.

7. **Pop semantics**. Sub-frame does NOT pop back to a parent frame
   on ray-exit (cs_raycast_local returns None; the shader returns
   sky). This is a deliberate scope cut — the alternative would be
   to chain to the parent face-subtree cell, which requires a much
   more elaborate GPU-side pop stack. Acceptable because the render
   frame is chosen such that the camera lives inside it, and near-
   boundary rays hit empty/sky past the frame.

## Execution order inside the commit

1. `bindings.wgsl` + renderer GpuUniforms struct — add the new
   fields. Keep them zero-initialized; shader doesn't reference
   them yet.
2. `set_root_kind_sphere_sub` on `Renderer` — packs fields.
3. `upload.rs` — swap the SphereSub body-fallback for the new
   method. Body-rendered visual behavior unchanged because the
   shader branch doesn't exist yet → `ROOT_KIND_SPHERE_SUB` hits
   the default arm (fall through to body march).
4. `sphere.wgsl` — add `sphere_in_sub_frame` + `walk_sub_frame`.
5. `main.wgsl` — add the `ROOT_KIND_SPHERE_SUB` branch in `march`.
6. `app/mod.rs::gpu_camera_for_frame` — J_inv basis for SphereSub.
7. Visual regression tests — run against a fresh build; iterate
   until green.

Order 1–3 keep the build green with the pipeline shader unchanged;
4–5 flip the behavior. 6–7 gate the actual visual change.

## Why not split?

Same reason stage 1–3 couldn't be split: per
`feedback_no_intermediate_visual_states.md`, a half-done renderer
renders garbage that looks like a real bug, wasting debugging
cycles. The shader + camera + uniforms + test set are a cohesive
unit that only means anything when all four are live together.

## Success criteria

- All four visual regression tests green.
- 144+ lib tests still green.
- `/loop` or manual inspection at anchor depths {2, 3, 5, 10, 25, 35}
  shows no visible seam at any depth transition.
- CPU `cpu_raycast_in_sub_frame` and GPU hit pixel point at the same
  cell for the same ray (path prefix match) at all tested depths.
