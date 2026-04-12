#!/usr/bin/env python3
"""Extract skeleton hierarchy and animations from a glTF/GLB file.

Outputs a JSON file describing:
  - Bone hierarchy with rest-pose transforms
  - Bounding box hints for splitting a voxelized model into body parts
  - Animation keyframes as per-bone (translation, rotation) samples

Usage:
    python3 tools/extract_skeleton.py assets/characters/Fox.glb -o fox_skeleton.json
    python3 tools/extract_skeleton.py assets/characters/Soldier.glb -o soldier_skeleton.json

The output JSON is consumed by the game's NPC blueprint loader to
drive voxel body-part animation.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from pygltflib import GLTF2
except ImportError:
    print("Error: pygltflib is required. Install with: pip3 install pygltflib")
    sys.exit(1)


# ---------------------------------------------------------------- math

def mat4_from_gltf_node(node):
    """Build a 4x4 matrix from a glTF node's TRS or matrix."""
    if node.matrix is not None:
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T  # glTF is column-major
    mat = np.eye(4, dtype=np.float64)
    if node.scale is not None:
        s = node.scale
        mat = np.diag([s[0], s[1], s[2], 1.0]) @ mat
    if node.rotation is not None:
        q = node.rotation  # [x, y, z, w]
        mat = quat_to_mat4(q) @ mat
    if node.translation is not None:
        t = node.translation
        tr = np.eye(4, dtype=np.float64)
        tr[0, 3] = t[0]
        tr[1, 3] = t[1]
        tr[2, 3] = t[2]
        mat = tr @ mat
    return mat


def quat_to_mat4(q):
    """Quaternion [x,y,z,w] to 4x4 rotation matrix."""
    x, y, z, w = q
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y - z * w)
    m[0, 2] = 2 * (x * z + y * w)
    m[1, 0] = 2 * (x * y + z * w)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z - x * w)
    m[2, 0] = 2 * (x * z - y * w)
    m[2, 1] = 2 * (y * z + x * w)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat4_to_trs(mat):
    """Extract translation and rotation (as quaternion) from a 4x4 matrix."""
    t = mat[:3, 3].tolist()
    # Extract rotation (ignore scale for animation purposes)
    r = mat4_to_quat(mat[:3, :3])
    return t, r


def mat4_to_quat(m):
    """3x3 rotation matrix to quaternion [x,y,z,w]."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


# --------------------------------------------------------- accessor reading

def read_accessor(gltf, accessor_index):
    """Read a glTF accessor into a numpy array."""
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]

    # Get the binary blob
    buffer = gltf.buffers[buffer_view.buffer]
    if hasattr(gltf, '_glb_data') and gltf._glb_data is not None:
        blob = gltf._glb_data
    else:
        blob = gltf.binary_blob()

    if blob is None:
        raise ValueError(f"No binary data found for accessor {accessor_index}")

    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

    # Component type → numpy dtype
    COMPONENT_TYPES = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    dtype = COMPONENT_TYPES[accessor.componentType]

    # Type → number of components
    TYPE_COUNTS = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16,
    }
    n_components = TYPE_COUNTS[accessor.type]

    byte_stride = buffer_view.byteStride
    if byte_stride is not None and byte_stride > 0:
        # Strided access
        component_size = np.dtype(dtype).itemsize * n_components
        result = np.zeros((accessor.count, n_components), dtype=dtype)
        for i in range(accessor.count):
            start = byte_offset + i * byte_stride
            end = start + component_size
            result[i] = np.frombuffer(blob[start:end], dtype=dtype, count=n_components)
        return result
    else:
        count = accessor.count * n_components
        data = np.frombuffer(blob, dtype=dtype, count=count, offset=byte_offset)
        if n_components > 1:
            data = data.reshape(accessor.count, n_components)
        return data


# ------------------------------------------------------------ skeleton

def extract_skeleton(gltf):
    """Extract bone hierarchy and rest-pose transforms."""
    if not gltf.skins:
        print("Warning: No skeleton (skin) found in this file.")
        return None

    skin = gltf.skins[0]
    joint_indices = skin.joints

    # Read inverse bind matrices if available
    ibms = None
    if skin.inverseBindMatrices is not None:
        ibm_data = read_accessor(gltf, skin.inverseBindMatrices)
        ibms = ibm_data.reshape(-1, 4, 4)

    # Build node name lookup and parent map
    node_names = {}
    node_children = {}
    for i, node in enumerate(gltf.nodes):
        node_names[i] = node.name or f"node_{i}"
        node_children[i] = list(node.children) if node.children else []

    # Build parent map
    parent_map = {}
    for i, children in node_children.items():
        for c in children:
            parent_map[c] = i

    # Compute world-space transforms for all nodes
    world_transforms = {}

    def compute_world_transform(node_idx):
        if node_idx in world_transforms:
            return world_transforms[node_idx]
        local = mat4_from_gltf_node(gltf.nodes[node_idx])
        if node_idx in parent_map:
            parent_world = compute_world_transform(parent_map[node_idx])
            world = parent_world @ local
        else:
            world = local
        world_transforms[node_idx] = world
        return world

    for idx in joint_indices:
        compute_world_transform(idx)

    # Extract bone data
    bones = {}
    for i, joint_idx in enumerate(joint_indices):
        name = node_names[joint_idx]
        node = gltf.nodes[joint_idx]

        # Local transform
        local_mat = mat4_from_gltf_node(node)
        local_t, local_r = mat4_to_trs(local_mat)

        # World-space position (rest pose)
        world_mat = world_transforms[joint_idx]
        world_pos = world_mat[:3, 3].tolist()

        # Parent bone name
        parent_name = None
        if joint_idx in parent_map and parent_map[joint_idx] in joint_indices:
            parent_name = node_names[parent_map[joint_idx]]

        bones[name] = {
            "index": i,
            "parent": parent_name,
            "local_translation": [round(v, 6) for v in local_t],
            "local_rotation": [round(v, 6) for v in local_r],
            "world_position": [round(v, 6) for v in world_pos],
        }

    return {
        "bone_count": len(bones),
        "bones": bones,
    }


# ---------------------------------------------------------- animations

def extract_animations(gltf):
    """Extract all animation clips as sampled keyframes."""
    if not gltf.animations:
        print("Warning: No animations found in this file.")
        return {}

    # Map node index to name
    node_names = {}
    for i, node in enumerate(gltf.nodes):
        node_names[i] = node.name or f"node_{i}"

    # Which nodes are joints?
    joint_set = set()
    if gltf.skins:
        for skin in gltf.skins:
            joint_set.update(skin.joints)

    animations = {}
    for anim in gltf.animations:
        anim_name = anim.name or "unnamed"
        channels_data = []

        for channel in anim.channels:
            sampler = anim.samplers[channel.sampler]
            target_node = channel.target.node
            target_path = channel.target.path  # translation, rotation, scale

            # Only care about joint bones
            if target_node not in joint_set:
                continue

            bone_name = node_names[target_node]

            # Read timestamps
            times = read_accessor(gltf, sampler.input).flatten()

            # Read values
            values = read_accessor(gltf, sampler.output)

            channels_data.append({
                "bone": bone_name,
                "path": target_path,
                "interpolation": sampler.interpolation or "LINEAR",
                "times": times,
                "values": values,
            })

        if not channels_data:
            continue

        # Find the time range and sample at regular intervals
        all_times = np.concatenate([c["times"] for c in channels_data])
        t_min = float(all_times.min())
        t_max = float(all_times.max())
        duration = t_max - t_min

        # Sample at ~30fps, or use actual keyframe times if sparse
        n_samples = max(2, int(np.ceil(duration * 30)))
        sample_times = np.linspace(t_min, t_max, n_samples)

        # Group channels by bone
        bone_channels = {}
        for ch in channels_data:
            bone = ch["bone"]
            if bone not in bone_channels:
                bone_channels[bone] = {}
            bone_channels[bone][ch["path"]] = ch

        # Sample each bone at each time
        keyframes = []
        for t in sample_times:
            frame = {"time": round(float(t), 4)}
            bone_transforms = {}

            for bone_name, paths in bone_channels.items():
                bt = {}
                for path, ch in paths.items():
                    if path == "scale":
                        continue  # skip scale — voxel parts are rigid
                    value = interpolate_channel(ch, float(t))
                    bt[path] = [round(float(v), 3) for v in value]
                bone_transforms[bone_name] = bt

            frame["bones"] = bone_transforms
            keyframes.append(frame)

        animations[anim_name] = {
            "duration": round(duration, 4),
            "keyframe_count": len(keyframes),
            "bones_animated": sorted(bone_channels.keys()),
            "keyframes": keyframes,
        }

    return animations


def interpolate_channel(channel, t):
    """Linearly interpolate a channel at time t."""
    times = channel["times"]
    values = channel["values"]

    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]

    # Find bracketing keyframes
    idx = np.searchsorted(times, t, side='right') - 1
    idx = max(0, min(idx, len(times) - 2))

    t0, t1 = times[idx], times[idx + 1]
    dt = t1 - t0
    if dt < 1e-10:
        return values[idx]

    alpha = (t - t0) / dt

    v0 = values[idx]
    v1 = values[idx + 1]

    if channel["path"] == "rotation":
        # Slerp for quaternions
        return slerp(v0, v1, alpha)
    else:
        # Lerp for translation/scale
        return v0 * (1 - alpha) + v1 * alpha


def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions [x,y,z,w]."""
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)

    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


# --------------------------------------------------------- mesh bounds

def extract_mesh_bounds(gltf):
    """Get the bounding box of the first mesh for scale reference."""
    if not gltf.meshes:
        return None

    all_min = np.array([np.inf, np.inf, np.inf])
    all_max = np.array([-np.inf, -np.inf, -np.inf])

    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            if prim.attributes.POSITION is not None:
                accessor = gltf.accessors[prim.attributes.POSITION]
                if accessor.min is not None:
                    all_min = np.minimum(all_min, accessor.min[:3])
                if accessor.max is not None:
                    all_max = np.maximum(all_max, accessor.max[:3])

    if np.any(np.isinf(all_min)):
        return None

    return {
        "min": [round(float(v), 4) for v in all_min],
        "max": [round(float(v), 4) for v in all_max],
        "size": [round(float(v), 4) for v in (all_max - all_min)],
    }


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Extract skeleton and animations from a glTF/GLB file."
    )
    parser.add_argument("input", help="Path to .glb or .gltf file")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    parser.add_argument("--pretty", action="store_true", default=True,
                        help="Pretty-print JSON output (default)")
    parser.add_argument("--compact", action="store_true",
                        help="Compact JSON output")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path}...", file=sys.stderr)
    gltf = GLTF2().load(str(input_path))

    result = {
        "source": input_path.name,
    }

    # Mesh bounds
    bounds = extract_mesh_bounds(gltf)
    if bounds:
        result["mesh_bounds"] = bounds
        print(f"  Mesh bounds: {bounds['size']}", file=sys.stderr)

    # Skeleton
    skeleton = extract_skeleton(gltf)
    if skeleton:
        result["skeleton"] = skeleton
        print(f"  Bones: {skeleton['bone_count']}", file=sys.stderr)
        # Print bone tree
        bones = skeleton["bones"]
        roots = [name for name, b in bones.items() if b["parent"] is None]
        def print_tree(name, indent=0):
            b = bones[name]
            pos = b["world_position"]
            print(f"  {'  ' * indent}{name} @ ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                  file=sys.stderr)
            children = [n for n, bb in bones.items() if bb["parent"] == name]
            for c in children:
                print_tree(c, indent + 1)
        for r in roots:
            print_tree(r)

    # Animations
    animations = extract_animations(gltf)
    if animations:
        result["animations"] = animations
        for name, anim in animations.items():
            print(f"  Animation '{name}': {anim['duration']:.2f}s, "
                  f"{anim['keyframe_count']} keyframes, "
                  f"{len(anim['bones_animated'])} bones",
                  file=sys.stderr)

    # Output
    indent = None if args.compact else 2
    json_str = json.dumps(result, indent=indent)

    if args.output:
        Path(args.output).write_text(json_str)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
