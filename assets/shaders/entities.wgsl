// Entity ray-march pass.
//
// Runs AFTER `march(...)` in the fragment shader; the caller
// composes the two hits by picking min(t). The world ray-march
// (ribbon pops, sphere/face dispatch, anchor-coord shimming) is
// entirely in `march.wgsl`/`sphere.wgsl` and does not know
// entities exist.
//
// Algorithm (v1):
//   for each entity e in 0..uniforms.entity_count:
//     ray-box test ray vs e.bbox_min / e.bbox_max (in frame coords)
//     if hit closer than current best:
//       transform ray into entity's local [0, 3)³ space
//       call march_cartesian(e.subtree_bfs, local_o, local_d, ...)
//       if subtree hit: promote to best, convert cell coords back
//                       to frame coords for consistent shading
//
// Complexity: O(entity_count) per ray. Fine for ~1k entities;
// hash-grid binning replaces the linear scan when we push to 100k+.
//
// The ray transform is a uniform scale (size.x/3 per axis). Because
// we scale both origin and direction by the same factor, the local
// ray's `t` parameter is IDENTICAL to the world ray's `t` — no
// conversion needed when comparing t values or reading hit.t back.
// Only cell_min/cell_size need rescaling back to world units so the
// fragment shader's bevel/grid math stays consistent.

fn march_entities(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var best: HitResult;
    best.hit = false;
    best.t = 1e20;
    best.frame_level = 0u;
    best.frame_scale = 1.0;
    best.cell_min = vec3<f32>(0.0);
    best.cell_size = 1.0;

    if uniforms.entity_count == 0u { return best; }

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );

    for (var i: u32 = 0u; i < uniforms.entity_count; i = i + 1u) {
        let e = entities[i];
        let bb = ray_box(ray_origin, inv_dir, e.bbox_min, e.bbox_max);
        if bb.t_enter >= bb.t_exit || bb.t_exit < 0.0 { continue; }
        if bb.t_enter >= best.t { continue; }

        // Transform ray into entity's local [0, 3)³ space. The
        // entity's subtree (at subtree_bfs) is a regular Cartesian
        // node whose children fill [0, 3)³ in their own coords —
        // identical to the world tree's root.
        let size = e.bbox_max - e.bbox_min;
        let scale3 = vec3<f32>(3.0) / size;
        let local_origin = (ray_origin - e.bbox_min) * scale3;
        let local_dir = ray_dir * scale3;

        let local_hit = march_cartesian(
            e.subtree_bfs, local_origin, local_dir,
            MAX_STACK_DEPTH, 0xFFFFFFFFu,
        );
        if !local_hit.hit { continue; }
        if local_hit.t >= best.t { continue; }

        // t is invariant under the uniform scale (see header).
        // Convert cell_min/cell_size back to frame coords so the
        // fragment shader's cube-face bevel stays coherent.
        let size_over_3 = size * (1.0 / 3.0);
        best = local_hit;
        best.cell_min = e.bbox_min + local_hit.cell_min * size_over_3;
        best.cell_size = local_hit.cell_size * size_over_3.x;
    }

    return best;
}
