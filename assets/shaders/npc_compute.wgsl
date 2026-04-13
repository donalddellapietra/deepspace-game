// GPU compute shader for NPC simulation.
//
// Each thread updates one NPC: AI decisions, velocity, heightmap
// collision, animation time advance. No CPU involvement per NPC.

struct NpcState {
    position: vec3<f32>,
    heading: f32,
    velocity: vec3<f32>,
    ai_timer: f32,
    anim_time: f32,
    speed: f32,
    seed: u32,
    flags: u32,  // bit 0 = alive
};

struct Uniforms {
    delta_time: f32,
    frame: u32,
    npc_count: u32,
    gravity: f32,
    world_min_xz: vec2<f32>,
    world_size_xz: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> npcs: array<NpcState>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var heightmap: texture_2d<f32>;
@group(0) @binding(3) var heightmap_sampler: sampler;

// PCG hash for GPU-side RNG.
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: u32) -> f32 {
    return f32(pcg_hash(seed) % 10000u) / 10000.0;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.npc_count) { return; }

    var npc = npcs[idx];
    if ((npc.flags & 1u) == 0u) { return; }

    let dt = u.delta_time;

    // ---- AI: countdown timer, pick new heading when expired ----
    npc.ai_timer -= dt;
    if (npc.ai_timer <= 0.0) {
        let rng = npc.seed ^ (u.frame * 1000u + idx);
        npc.heading = rand_f32(rng) * 6.2831853;
        npc.ai_timer = 2.0 + rand_f32(rng + 1u) * 3.0;
        npc.seed = pcg_hash(rng);
    }

    // ---- Physics: velocity from heading, gravity, heightmap clamp ----
    npc.velocity.x = -sin(npc.heading) * npc.speed;
    npc.velocity.z = -cos(npc.heading) * npc.speed;
    npc.velocity.y -= u.gravity * dt;

    npc.position += npc.velocity * dt;

    // Sample heightmap for ground Y.
    let uv = (npc.position.xz - u.world_min_xz) / u.world_size_xz;
    let clamped_uv = clamp(uv, vec2(0.0), vec2(1.0));
    let ground_y = textureSampleLevel(heightmap, heightmap_sampler, clamped_uv, 0.0).r;

    if (npc.position.y < ground_y) {
        npc.position.y = ground_y;
        npc.velocity.y = 0.0;
    }

    // ---- Animation: advance time ----
    npc.anim_time += dt;

    npcs[idx] = npc;
}
