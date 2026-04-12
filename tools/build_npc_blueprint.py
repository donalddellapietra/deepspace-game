#!/usr/bin/env python3
"""Build an NPC blueprint from a skeleton JSON + .vox file.

The full offline pipeline:
  1. Load skeleton JSON (from extract_skeleton.py)
  2. Load .vox file (the voxelized character model)
  3. Auto-detect skeleton type and map bones → body parts
  4. Map bone world positions to voxel coordinates
  5. Split the voxel grid into per-part subgrids
  6. Convert per-bone animation keyframes → per-part keyframes
  7. Output a blueprint JSON the game loads at runtime

Usage:
    # Using a GLB that was voxelized with filetovox.sh:
    python3 tools/build_npc_blueprint.py \\
        --skeleton fox_skeleton.json \\
        --vox assets/vox/chr_fox.vox \\
        -o assets/npcs/fox.blueprint.json

    # With a custom part mapping override:
    python3 tools/build_npc_blueprint.py \\
        --skeleton soldier_skeleton.json \\
        --vox soldier.vox \\
        --mapping custom_parts.json \\
        -o assets/npcs/soldier.blueprint.json
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import pyvox.parser
except ImportError:
    print("Error: py-vox-io is required. Install with: pip3 install py-vox-io")
    sys.exit(1)


# ============================================================ part mappings
#
# Each mapping defines body parts as groups of bones. The "pivot_bone"
# determines the part's rotation center. The "anim_bone" is the bone
# whose animation keyframes drive the part (usually the root of the
# limb chain).
#
# Bone names are matched with fuzzy substring matching — "LeftArm"
# matches "mixamorig:LeftArm" and "b_LeftUpperArm_09".

HUMANOID_PARTS = {
    "head": {
        "bones": ["Head"],
        "pivot_bone": "Head",
        "anim_bone": "Head",
    },
    "torso": {
        "bones": ["Spine", "Spine1", "Spine2", "Neck", "Hips",
                   "Shoulder", "LeftShoulder", "RightShoulder"],
        "pivot_bone": "Hips",
        "anim_bone": "Hips",
    },
    "arm_l": {
        "bones": ["LeftArm", "LeftForeArm", "LeftHand",
                   "LeftHandThumb", "LeftHandIndex", "LeftHandMiddle",
                   "LeftHandRing", "LeftHandPinky"],
        "pivot_bone": "LeftArm",
        "anim_bone": "LeftArm",
    },
    "arm_r": {
        "bones": ["RightArm", "RightForeArm", "RightHand",
                   "RightHandThumb", "RightHandIndex", "RightHandMiddle",
                   "RightHandRing", "RightHandPinky"],
        "pivot_bone": "RightArm",
        "anim_bone": "RightArm",
    },
    "leg_l": {
        "bones": ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe"],
        "pivot_bone": "LeftUpLeg",
        "anim_bone": "LeftUpLeg",
    },
    "leg_r": {
        "bones": ["RightUpLeg", "RightLeg", "RightFoot", "RightToe"],
        "pivot_bone": "RightUpLeg",
        "anim_bone": "RightUpLeg",
    },
}

QUADRUPED_PARTS = {
    "head": {
        "bones": ["Head", "Neck"],
        "pivot_bone": "Neck",
        "anim_bone": "Head",
    },
    "torso": {
        "bones": ["Spine", "Hip", "Root"],
        "pivot_bone": "Hip",
        "anim_bone": "Hip",
    },
    "leg_fl": {
        "bones": ["LeftUpperArm", "LeftForeArm", "LeftHand"],
        "pivot_bone": "LeftUpperArm",
        "anim_bone": "LeftUpperArm",
    },
    "leg_fr": {
        "bones": ["RightUpperArm", "RightForeArm", "RightHand"],
        "pivot_bone": "RightUpperArm",
        "anim_bone": "RightUpperArm",
    },
    "leg_bl": {
        "bones": ["LeftLeg01", "LeftLeg02", "LeftFoot"],
        "pivot_bone": "LeftLeg01",
        "anim_bone": "LeftLeg01",
    },
    "leg_br": {
        "bones": ["RightLeg01", "RightLeg02", "RightFoot"],
        "pivot_bone": "RightLeg01",
        "anim_bone": "RightLeg01",
    },
    "tail": {
        "bones": ["Tail"],
        "pivot_bone": "Tail01",
        "anim_bone": "Tail01",
    },
}


# ============================================================ bone matching

def fuzzy_match_bone(pattern, bone_names):
    """Find bone names that contain the pattern (case-insensitive).
    Strips common prefixes like 'mixamorig:', 'b_', etc."""
    results = []
    pattern_lower = pattern.lower()
    for name in bone_names:
        # Strip common prefixes for matching
        stripped = name
        for prefix in ["mixamorig:", "b_", "Bip01_"]:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        # Also strip trailing numbers like _01, _015
        stripped_base = re.sub(r'_?\d+$', '', stripped)
        if (pattern_lower in stripped.lower() or
            pattern_lower in stripped_base.lower() or
            pattern_lower == stripped_base.lower()):
            results.append(name)
    return results


def detect_skeleton_type(bone_names):
    """Auto-detect if this is a humanoid or quadruped skeleton."""
    names_lower = " ".join(bone_names).lower()
    # Quadruped: has "tail" bones and leg-named front limbs
    if "tail" in names_lower and ("leg01" in names_lower or "leg02" in names_lower):
        return "quadruped"
    # Humanoid: has UpLeg/Thigh bones
    if "upleg" in names_lower or "thigh" in names_lower:
        return "humanoid"
    # Default to humanoid
    return "humanoid"


def build_bone_to_part_map(skeleton, part_mapping):
    """Map each bone to its body part using fuzzy matching."""
    bone_names = list(skeleton["bones"].keys())
    bone_to_part = {}
    part_bones = {}  # part_name → [bone_names]

    for part_name, part_def in part_mapping.items():
        matched = []
        for pattern in part_def["bones"]:
            matched.extend(fuzzy_match_bone(pattern, bone_names))
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for b in matched:
            if b not in seen:
                seen.add(b)
                unique.append(b)
        part_bones[part_name] = unique
        for b in unique:
            if b not in bone_to_part:  # first match wins
                bone_to_part[b] = part_name

    return bone_to_part, part_bones


def resolve_bone(pattern, bone_names):
    """Find a single bone matching a pattern."""
    matches = fuzzy_match_bone(pattern, bone_names)
    return matches[0] if matches else None


# ============================================================ vox loading

def load_vox(path):
    """Load a .vox file and return (size, voxel_dict, palette)."""
    vox = pyvox.parser.VoxParser(str(path)).parse()
    model = vox.models[0]
    sx, sy, sz = model.size.x, model.size.y, model.size.z

    # Build dict of (x, y, z) → color_index
    # .vox uses Z-up; we convert to Y-up here:
    #   vox_x → our_x, vox_z → our_y (up), vox_y → our_z (depth)
    voxels = {}
    for v in model.voxels:
        our_x = v.x
        our_y = v.z  # vox Z → our Y (up)
        our_z = v.y  # vox Y → our Z (depth)
        voxels[(our_x, our_y, our_z)] = v.c  # color index

    # Our dimensions after axis swap
    out_sx = sx      # vox X → our X
    out_sy = sz      # vox Z → our Y
    out_sz = sy      # vox Y → our Z

    # Palette (256 RGBA entries)
    palette = []
    for c in vox.palette:
        palette.append((c.r, c.g, c.b, c.a))

    return (out_sx, out_sy, out_sz), voxels, palette


# ========================================== coordinate mapping

def compute_bone_to_voxel_transform(skeleton, vox_size):
    """Compute the affine transform from GLB bone world-space to voxel coords.

    Uses the mesh bounding box from the skeleton JSON to map
    model-space positions into the voxel grid.
    """
    bounds = skeleton.get("mesh_bounds")
    if not bounds:
        print("Warning: No mesh_bounds in skeleton — using identity mapping",
              file=sys.stderr)
        return np.eye(4)

    mesh_min = np.array(bounds["min"])
    mesh_max = np.array(bounds["max"])
    mesh_size = mesh_max - mesh_min

    # The voxelizer maps the mesh bounding box to the voxel grid.
    # GLB uses Y-up; .vox uses Z-up. After our axis swap in load_vox,
    # voxel coords are already Y-up. But we need to check how
    # FileToVox maps the axes.
    #
    # FileToVox preserves axes: GLB X→vox X, GLB Y→vox Z, GLB Z→vox Y
    # After our load_vox swap: vox X→our X, vox Z→our Y, vox Y→our Z
    # Net: GLB X→our X, GLB Y→our Y, GLB Z→our Z (identity axes!)
    #
    # So we just need to scale from mesh bounds to voxel bounds.
    vox_sx, vox_sy, vox_sz = vox_size

    # Scale factor: mesh units → voxel units
    # We use the largest axis to preserve aspect ratio (FileToVox does this)
    max_mesh_dim = max(mesh_size)
    max_vox_dim = max(vox_sx, vox_sy, vox_sz)
    if max_mesh_dim < 1e-6:
        scale = 1.0
    else:
        scale = max_vox_dim / max_mesh_dim

    # Offset: mesh_min maps to voxel 0 (approximately — FileToVox may center)
    # We center the mapping: mesh center → voxel center
    mesh_center = (mesh_min + mesh_max) / 2
    vox_center = np.array([vox_sx / 2, vox_sy / 2, vox_sz / 2])

    return mesh_center, vox_center, scale


def bone_pos_to_voxel(bone_world_pos, mesh_center, vox_center, scale):
    """Convert a bone's world position to voxel coordinates."""
    pos = np.array(bone_world_pos)
    voxel_pos = (pos - mesh_center) * scale + vox_center
    return voxel_pos


# ============================================================ voxel splitting

def split_voxels_by_part(voxels, vox_size, skeleton, bone_to_part, part_bones,
                         mesh_center, vox_center, scale):
    """Assign each voxel to the nearest bone's body part."""
    bone_names = list(skeleton["bones"].keys())

    # Precompute bone positions in voxel space
    bone_voxel_pos = {}
    for name, bone in skeleton["bones"].items():
        pos = bone_pos_to_voxel(bone["world_position"],
                                mesh_center, vox_center, scale)
        bone_voxel_pos[name] = pos

    # For each voxel, find nearest bone and assign to its part
    part_voxels = {}  # part_name → dict of (x,y,z) → color_index
    unassigned = {}

    for (x, y, z), color in voxels.items():
        vpos = np.array([x, y, z], dtype=np.float64)

        best_dist = float('inf')
        best_bone = None
        for bname, bpos in bone_voxel_pos.items():
            dist = np.linalg.norm(vpos - bpos)
            if dist < best_dist:
                best_dist = dist
                best_bone = bname

        part = bone_to_part.get(best_bone)
        if part is None:
            # Bone not mapped to any part — assign to nearest mapped bone
            best_dist2 = float('inf')
            for bname in bone_to_part:
                bpos = bone_voxel_pos.get(bname)
                if bpos is not None:
                    dist = np.linalg.norm(vpos - bpos)
                    if dist < best_dist2:
                        best_dist2 = dist
                        part = bone_to_part[bname]
            if part is None:
                part = "torso"  # fallback

        if part not in part_voxels:
            part_voxels[part] = {}
        part_voxels[part][(x, y, z)] = color

    return part_voxels


def extract_part_grid(part_voxels_dict):
    """Convert a sparse dict of (x,y,z)→color into a dense grid + bounds."""
    if not part_voxels_dict:
        return None

    coords = list(part_voxels_dict.keys())
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    sx = max_x - min_x + 1
    sy = max_y - min_y + 1
    sz = max_z - min_z + 1

    # Dense grid (0 = empty)
    grid = [0] * (sx * sy * sz)
    for (x, y, z), color in part_voxels_dict.items():
        lx = x - min_x
        ly = y - min_y
        lz = z - min_z
        # Row-major: (z * sy + y) * sx + x
        grid[(lz * sy + ly) * sx + lx] = color

    return {
        "min": [min_x, min_y, min_z],
        "size": [sx, sy, sz],
        "voxels": grid,
    }


# ============================================= animation conversion

def convert_animations(skeleton_data, part_mapping, bone_names,
                       mesh_center, vox_center, scale):
    """Convert per-bone keyframes to per-part keyframes.

    For each part, we use the anim_bone's rotation keyframes.
    Translation is relative to the rest pose.
    """
    animations_in = skeleton_data.get("animations", {})
    bones = skeleton_data["skeleton"]["bones"]
    animations_out = {}

    for anim_name, anim_data in animations_in.items():
        # Find the actual bone name for each part's anim_bone
        part_anim_bones = {}
        for part_name, part_def in part_mapping.items():
            resolved = resolve_bone(part_def["anim_bone"], bone_names)
            if resolved:
                part_anim_bones[part_name] = resolved

        keyframes = []
        for kf in anim_data["keyframes"]:
            frame = {"time": kf["time"], "parts": {}}
            for part_name, anim_bone in part_anim_bones.items():
                bone_kf = kf["bones"].get(anim_bone, {})
                part_transform = {}

                if "rotation" in bone_kf:
                    part_transform["rotation"] = bone_kf["rotation"]
                if "translation" in bone_kf:
                    # Convert translation delta to voxel space
                    rest_t = bones[anim_bone]["local_translation"]
                    anim_t = bone_kf["translation"]
                    delta = [
                        (anim_t[0] - rest_t[0]) * scale,
                        (anim_t[1] - rest_t[1]) * scale,
                        (anim_t[2] - rest_t[2]) * scale,
                    ]
                    part_transform["translation"] = [
                        round(d, 3) for d in delta
                    ]

                if part_transform:
                    frame["parts"][part_name] = part_transform

            keyframes.append(frame)

        animations_out[anim_name] = {
            "duration": anim_data["duration"],
            "looping": anim_name.lower() in ("walk", "run", "idle"),
            "keyframes": keyframes,
        }

    return animations_out


# ============================================================ palette

def convert_palette(vox_palette, used_indices):
    """Convert .vox palette entries for used color indices to RGBA arrays."""
    palette_out = {}
    for idx in sorted(used_indices):
        if idx == 0:
            continue  # empty
        # .vox palette is 1-indexed in voxel data (color index 1 = palette[0])
        pal_idx = (idx - 1) % len(vox_palette)
        r, g, b, a = vox_palette[pal_idx]
        palette_out[str(idx)] = [r, g, b, a]
    return palette_out


# ============================================================ main

def main():
    parser = argparse.ArgumentParser(
        description="Build an NPC blueprint from skeleton JSON + .vox file."
    )
    parser.add_argument("--skeleton", required=True,
                        help="Skeleton JSON from extract_skeleton.py")
    parser.add_argument("--vox", required=True,
                        help=".vox file (voxelized character)")
    parser.add_argument("--mapping", help="Custom part mapping JSON (optional)")
    parser.add_argument("--type", choices=["humanoid", "quadruped", "auto"],
                        default="auto", help="Skeleton type (default: auto)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output blueprint JSON path")
    args = parser.parse_args()

    # Load skeleton
    print(f"Loading skeleton: {args.skeleton}", file=sys.stderr)
    with open(args.skeleton) as f:
        skeleton_data = json.load(f)

    if "skeleton" not in skeleton_data:
        print("Error: No skeleton data in JSON", file=sys.stderr)
        sys.exit(1)

    skeleton = skeleton_data["skeleton"]
    bone_names = list(skeleton["bones"].keys())
    print(f"  {len(bone_names)} bones", file=sys.stderr)

    # Load .vox
    print(f"Loading vox: {args.vox}", file=sys.stderr)
    vox_size, voxels, vox_palette = load_vox(args.vox)
    print(f"  Size: {vox_size[0]}x{vox_size[1]}x{vox_size[2]}, "
          f"{len(voxels)} voxels", file=sys.stderr)

    # Select part mapping
    if args.mapping:
        with open(args.mapping) as f:
            part_mapping = json.load(f)
        skel_type = "custom"
    else:
        if args.type == "auto":
            skel_type = detect_skeleton_type(bone_names)
        else:
            skel_type = args.type
        part_mapping = QUADRUPED_PARTS if skel_type == "quadruped" else HUMANOID_PARTS

    print(f"  Skeleton type: {skel_type}", file=sys.stderr)

    # Map bones → parts
    bone_to_part, part_bones = build_bone_to_part_map(skeleton, part_mapping)
    for part_name, bones in part_bones.items():
        print(f"  {part_name}: {len(bones)} bones → {bones}", file=sys.stderr)

    # Unmatched bones
    unmatched = [b for b in bone_names if b not in bone_to_part]
    if unmatched:
        print(f"  Unmatched bones (assigned by proximity): {unmatched}",
              file=sys.stderr)

    # Coordinate mapping
    mesh_center, vox_center, scale = compute_bone_to_voxel_transform(
        skeleton_data, vox_size)
    print(f"  Scale: {scale:.4f}, mesh_center: {mesh_center}, "
          f"vox_center: {vox_center}", file=sys.stderr)

    # Show bone positions in voxel space for debugging
    print("  Bone positions (voxel space):", file=sys.stderr)
    for bname, bone in skeleton["bones"].items():
        vpos = bone_pos_to_voxel(bone["world_position"],
                                 mesh_center, vox_center, scale)
        part = bone_to_part.get(bname, "???")
        print(f"    {bname} → ({vpos[0]:.1f}, {vpos[1]:.1f}, {vpos[2]:.1f})"
              f" [{part}]", file=sys.stderr)

    # Split voxels
    part_voxels = split_voxels_by_part(
        voxels, vox_size, skeleton, bone_to_part, part_bones,
        mesh_center, vox_center, scale)

    print("  Part voxel counts:", file=sys.stderr)
    for part_name, pvox in part_voxels.items():
        print(f"    {part_name}: {len(pvox)} voxels", file=sys.stderr)

    # Build part grids
    parts_out = {}
    all_used_colors = set()
    for part_name in part_mapping:
        pvox = part_voxels.get(part_name, {})
        if not pvox:
            print(f"  Warning: no voxels for part '{part_name}'", file=sys.stderr)
            continue

        grid_data = extract_part_grid(pvox)
        all_used_colors.update(c for c in grid_data["voxels"] if c != 0)

        # Pivot: resolved bone position in part-local coordinates
        pivot_pattern = part_mapping[part_name]["pivot_bone"]
        pivot_bone = resolve_bone(pivot_pattern, bone_names)
        if pivot_bone:
            pivot_world = bone_pos_to_voxel(
                skeleton["bones"][pivot_bone]["world_position"],
                mesh_center, vox_center, scale)
            pivot_local = [
                round(pivot_world[0] - grid_data["min"][0], 2),
                round(pivot_world[1] - grid_data["min"][1], 2),
                round(pivot_world[2] - grid_data["min"][2], 2),
            ]
        else:
            # Center of the part
            pivot_local = [s / 2 for s in grid_data["size"]]

        # Rest offset: part min corner relative to model origin (voxel 0,0,0)
        rest_offset = [float(v) for v in grid_data["min"]]

        # Encode voxels as base64 for compactness
        voxel_bytes = bytes(grid_data["voxels"])
        voxels_b64 = base64.b64encode(voxel_bytes).decode('ascii')

        parts_out[part_name] = {
            "size": grid_data["size"],
            "pivot": pivot_local,
            "rest_offset": rest_offset,
            "voxels_b64": voxels_b64,
            "voxel_count": sum(1 for v in grid_data["voxels"] if v != 0),
        }

    # Convert animations
    animations_out = convert_animations(
        skeleton_data, part_mapping, bone_names,
        mesh_center, vox_center, scale)

    for anim_name, anim in animations_out.items():
        print(f"  Animation '{anim_name}': {anim['duration']:.2f}s, "
              f"{len(anim['keyframes'])} frames, "
              f"looping={anim['looping']}", file=sys.stderr)

    # Palette
    palette_out = convert_palette(vox_palette, all_used_colors)

    # Build final blueprint
    blueprint = {
        "source": {
            "skeleton": Path(args.skeleton).name,
            "vox": Path(args.vox).name,
            "skeleton_type": skel_type,
        },
        "palette": palette_out,
        "parts": parts_out,
        "animations": animations_out,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(blueprint, f, indent=2)

    file_size = output_path.stat().st_size
    print(f"\nWritten to {output_path} ({file_size / 1024:.1f} KB)",
          file=sys.stderr)
    print(f"  {len(parts_out)} parts, {len(animations_out)} animations, "
          f"{len(palette_out)} palette colors", file=sys.stderr)


if __name__ == "__main__":
    main()
