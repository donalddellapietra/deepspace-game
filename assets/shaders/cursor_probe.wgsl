// Cursor-probe compute shader. Casts a single ray from the screen
// centre using the same `march()` pipeline the fragment shader uses,
// writes the hit result to a storage buffer the CPU maps back, and
// exits. Removes the need for a parallel CPU raycast — the CPU
// consumes GPU-produced paths for highlight visualization, edit
// targeting, and test-harness probes.
//
// Dispatch: `@workgroup_size(1, 1, 1)` with a single invocation per
// frame. Sub-millisecond cost (one ray's worth of tree descent).

#include "bindings.wgsl"
#include "tree.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"
#include "march.wgsl"

/// CPU-mapped cursor-hit output. Layout mirrors
/// `src/renderer/cursor_probe.rs::CursorProbeResult`. The shader
/// writes a single packed path (not the walker-relative hit_path in
/// `HitResult`) so the CPU reads a single full-world-root path
/// without needing to redo the render_path prefix concatenation.
struct CursorProbeOut {
    hit: u32,
    depth: u32,
    t: f32,
    face: u32,
    path: array<vec4<u32>, 4>,
}

@group(1) @binding(0) var<storage, read_write> cursor_out: CursorProbeOut;

@compute @workgroup_size(1, 1, 1)
fn cs_cursor_probe() {
    // Screen-center ray == `camera.forward`. Crosshair is always at
    // NDC (0, 0) which reduces to `camera.forward` in the fragment
    // shader's `camera.forward + camera.right * 0 + camera.up * 0`
    // formula.
    let ray_dir = normalize(camera.forward);
    let result = march(camera.pos, ray_dir);

    if !result.hit {
        cursor_out.hit = 0u;
        cursor_out.depth = 0u;
        cursor_out.t = 1e20;
        cursor_out.face = 0u;
        cursor_out.path = array<vec4<u32>, 4>(
            vec4<u32>(0u), vec4<u32>(0u), vec4<u32>(0u), vec4<u32>(0u),
        );
        return;
    }

    // Reconstruct full world-root-relative slot path: the hit lives
    // at `render_path[0..render_depth − frame_level]` + the walker's
    // local `hit_path`.
    let r_depth = uniforms.render_path_depth;
    let pop_level = result.frame_level;
    let frame_prefix_len = select(0u, r_depth - pop_level, r_depth >= pop_level);
    let full_depth = frame_prefix_len + result.hit_path_depth;

    var packed: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
        vec4<u32>(0u), vec4<u32>(0u), vec4<u32>(0u), vec4<u32>(0u),
    );
    for (var i: u32 = 0u; i < full_depth; i = i + 1u) {
        var slot: u32;
        if i < frame_prefix_len {
            slot = unpack_slot_from_path(uniforms.render_path, i);
        } else {
            slot = unpack_slot_from_path(result.hit_path, i - frame_prefix_len);
        }
        pack_slot_into_path(&packed, i, slot);
    }

    cursor_out.hit = 1u;
    cursor_out.depth = full_depth;
    cursor_out.t = result.t;
    // Pack the hit normal into a face id matching the Cartesian DDA
    // convention: 0/1 = ±X, 2/3 = ±Y, 4/5 = ±Z. Positive axis = 1,3,5.
    let n = result.normal;
    var face_id: u32 = 0u;
    let ax = abs(n.x);
    let ay = abs(n.y);
    let az = abs(n.z);
    if ax >= ay && ax >= az {
        face_id = select(0u, 1u, n.x > 0.0);
    } else if ay >= az {
        face_id = select(2u, 3u, n.y > 0.0);
    } else {
        face_id = select(4u, 5u, n.z > 0.0);
    }
    cursor_out.face = face_id;
    cursor_out.path = packed;
}
