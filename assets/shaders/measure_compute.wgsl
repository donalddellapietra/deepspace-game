// DIAGNOSTIC-ONLY compute entry point. Not used at runtime. Exists
// purely to let us build a compute pipeline from `unified_dda` so
// we can query `max_total_threads_per_threadgroup` (register-pressure
// proxy) via the Metal API. Apple doesn't expose this stat for
// fragment pipelines.

#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "march.wgsl"

@compute @workgroup_size(8, 8, 1)
fn cs_measure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let origin = vec3<f32>(f32(gid.x) * 0.01, f32(gid.y) * 0.01, 0.1);
    let dir = normalize(vec3<f32>(1.0, 0.1, 0.1));
    let r = unified_dda(0u, origin, dir, 0xFFFFFFFFu, MAX_STACK_DEPTH);
    // Use the result so the compiler can't eliminate the call.
    atomicStore(&shader_stats.ray_count, u32(r.t * 100.0));
}
