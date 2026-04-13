// Compute shader: build NPC instance data from NPC state + animation.
//
// One thread per (NPC × part). Reads NPC position/heading/anim_time
// from the state buffer, interpolates animation keyframes, and writes
// the resulting transform + color to the instance buffer.
//
// This replaces the CPU-side collect_overlays + reconcile_instanced.

struct NpcState {
    position: vec3<f32>,
    heading: f32,
    velocity: vec3<f32>,
    ai_timer: f32,
    anim_time: f32,
    speed: f32,
    seed: u32,
    flags: u32,
};

struct PartInfo {
    rest_offset: vec3<f32>,
    _pad0: f32,
    pivot: vec3<f32>,
    _pad1: f32,
};

struct Keyframe {
    offset: vec3<f32>,
    _pad: f32,
    rotation: vec4<f32>,  // quaternion xyzw
};

struct InstanceData {
    transform: mat4x4<f32>,
    color: vec4<f32>,
};

struct BuildUniforms {
    npc_count: u32,
    num_parts: u32,
    num_keyframes: u32,
    _pad: u32,
    frame_duration: f32,
    total_duration: f32,
    scale: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> npcs: array<NpcState>;
@group(0) @binding(1) var<uniform> u: BuildUniforms;
@group(0) @binding(2) var<storage, read> parts: array<PartInfo>;
@group(0) @binding(3) var<storage, read> keyframes: array<Keyframe>;  // [keyframe_idx * num_parts + part_idx]
@group(0) @binding(4) var<storage, read_write> instances: array<InstanceData>;
@group(0) @binding(5) var<storage, read> colors: array<vec4<f32>>;  // per-part color

// Quaternion to rotation matrix.
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

// Slerp between two quaternions.
fn quat_slerp(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    var b_adj = b;
    var cos_theta = dot(a, b);
    if (cos_theta < 0.0) {
        b_adj = -b;
        cos_theta = -cos_theta;
    }
    if (cos_theta > 0.9995) {
        return normalize(mix(a, b_adj, t));
    }
    let theta = acos(cos_theta);
    let sin_theta = sin(theta);
    let wa = sin((1.0 - t) * theta) / sin_theta;
    let wb = sin(t * theta) / sin_theta;
    return a * wa + b_adj * wb;
}

@compute @workgroup_size(64)
fn build_instances(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.npc_count * u.num_parts;
    if (idx >= total) { return; }

    let npc_idx = idx / u.num_parts;
    let part_idx = idx % u.num_parts;

    let npc = npcs[npc_idx];
    if ((npc.flags & 1u) == 0u) { return; }

    let part = parts[part_idx];

    // Interpolate animation keyframes.
    var anim_offset = vec3<f32>(0.0);
    var anim_rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    if (u.num_keyframes > 0u && u.total_duration > 0.0) {
        let t = npc.anim_time % u.total_duration;
        let frame_f = t / u.frame_duration;
        let frame_a = u32(frame_f) % u.num_keyframes;
        let frame_b = (frame_a + 1u) % u.num_keyframes;
        let blend = fract(frame_f);

        let kf_a = keyframes[frame_a * u.num_parts + part_idx];
        let kf_b = keyframes[frame_b * u.num_parts + part_idx];

        anim_offset = mix(kf_a.offset, kf_b.offset, blend);
        anim_rot = quat_slerp(kf_a.rotation, kf_b.rotation, blend);
    }

    // Build the world transform for this part.
    let heading_cos = cos(npc.heading + 3.14159265);
    let heading_sin = sin(npc.heading + 3.14159265);
    let heading_rot = mat3x3<f32>(
        vec3<f32>(heading_cos, 0.0, heading_sin),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(-heading_sin, 0.0, heading_cos),
    );

    let part_offset = part.rest_offset + anim_offset;
    let part_rot = quat_to_mat3(anim_rot);

    // Combined: npc_pos + heading_rot * (scale * part_offset), then part_rot, then -pivot
    let world_offset = heading_rot * (part_offset * u.scale);
    let world_pos = npc.position + world_offset;
    let combined_rot = heading_rot * part_rot;

    // Build 4x4 matrix: rotation * scale, then translate by (world_pos + combined_rot * (-pivot * scale))
    let pivot_offset = combined_rot * (-part.pivot * u.scale);
    let final_pos = world_pos + pivot_offset;

    let s = u.scale;
    let r = combined_rot;
    let transform = mat4x4<f32>(
        vec4<f32>(r[0] * s, 0.0),
        vec4<f32>(r[1] * s, 0.0),
        vec4<f32>(r[2] * s, 0.0),
        vec4<f32>(final_pos, 1.0),
    );

    let color = colors[part_idx];

    instances[idx] = InstanceData(transform, color);
}
