#!/usr/bin/env python3
"""Build an NPC blueprint from a .vox file, with optional skeleton.

Two modes:

  1. **Skeleton mode** — .vox + skeleton JSON (from extract_skeleton.py):
     Maps bones to body parts, splits voxels by nearest bone, uses real
     animation keyframes.

  2. **Auto mode** (no skeleton) — .vox only:
     Splits the voxel model into body parts using spatial heuristics
     (height bands + lateral symmetry) and generates procedural walk/idle
     animations. Works with any standalone voxel character art.

Usage:
    # Auto mode (standalone .vox — no skeleton needed):
    python3 tools/build_npc_blueprint.py \\
        --vox assets/vox/chr_fox.vox \\
        -o assets/npcs/fox.blueprint.json

    # Skeleton mode (matched GLB→vox pair):
    python3 tools/build_npc_blueprint.py \\
        --skeleton fox_skeleton.json \\
        --vox assets/vox/fox_voxelized.vox \\
        -o assets/npcs/fox.blueprint.json
"""

import argparse
import base64
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import pyvox.parser
except ImportError:
    print("Error: py-vox-io is required. Install with: pip3 install py-vox-io")
    sys.exit(1)


# ============================================================ vox loading

def load_vox(path):
    """Load a .vox file → (size_xyz, sparse_dict, palette).

    Axis swap: .vox Z-up → our Y-up.
    Returns voxels as dict (x, y, z) → color_index.
    """
    vox = pyvox.parser.VoxParser(str(path)).parse()
    model = vox.models[0]
    sx, sy, sz = model.size.x, model.size.y, model.size.z

    voxels = {}
    for v in model.voxels:
        our_x = v.x
        our_y = v.z  # vox Z → our Y (up)
        our_z = v.y  # vox Y → our Z (depth)
        voxels[(our_x, our_y, our_z)] = v.c

    # Dimensions after axis swap
    out_sx = sx  # vox X
    out_sy = sz  # vox Z → our Y
    out_sz = sy  # vox Y → our Z

    palette = [(c.r, c.g, c.b, c.a) for c in vox.palette]
    return (out_sx, out_sy, out_sz), voxels, palette


# ============================================================ auto split
#
# Split a voxel model into body parts using spatial heuristics.
# Works for humanoid and quadruped characters without any skeleton data.

def compute_bounds(voxels):
    """Bounding box of occupied voxels."""
    coords = list(voxels.keys())
    if not coords:
        return (0, 0, 0), (0, 0, 0)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def auto_split_humanoid(voxels, vox_size):
    """Split voxels into humanoid parts by height bands + lateral symmetry.

    Height bands (fraction of occupied Y range):
      - head:  top 25%
      - torso: middle 35%  (between head and legs)
      - legs:  bottom 40%

    Lateral split (X axis) for arms and legs:
      - Arms: voxels in the torso band that extend past the torso core width
      - Legs: left/right halves of the leg band
    """
    (min_x, min_y, min_z), (max_x, max_y, max_z) = compute_bounds(voxels)
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    mid_x = (min_x + max_x) / 2

    # Height thresholds
    leg_top = min_y + height * 0.40
    head_bot = min_y + height * 0.75

    # Find the "core" X width of the torso (densest X range in torso band)
    torso_xs = [x for (x, y, z), _ in voxels.items() if leg_top <= y < head_bot]
    if torso_xs:
        # Core = central 60% of torso width
        torso_min_x = min(torso_xs)
        torso_max_x = max(torso_xs)
        torso_w = torso_max_x - torso_min_x + 1
        core_margin = torso_w * 0.2
        core_left = torso_min_x + core_margin
        core_right = torso_max_x - core_margin
    else:
        core_left = mid_x - width * 0.15
        core_right = mid_x + width * 0.15

    # Also compute head core width — the actual head is narrower than
    # the shoulders/arms that may be at the same height (T-pose).
    # Scan the very top of the model to find the head's X extent.
    top_10 = min_y + height * 0.90
    top_xs = [x for (x, y, z), _ in voxels.items() if y >= top_10]
    if top_xs:
        head_min_x = min(top_xs)
        head_max_x = max(top_xs)
        # Head core: the top-10% width plus some margin
        head_w = head_max_x - head_min_x + 1
        head_margin = head_w * 0.3
        head_left = head_min_x - head_margin
        head_right = head_max_x + head_margin
    else:
        head_left = core_left
        head_right = core_right

    parts = {
        "head": {},
        "torso": {},
        "arm_l": {},
        "arm_r": {},
        "leg_l": {},
        "leg_r": {},
    }

    for (x, y, z), c in voxels.items():
        if y >= head_bot:
            # Head band: only voxels within the head's X width are "head".
            # Voxels outside that are arms (T-pose) or shoulders (torso).
            if head_left <= x <= head_right:
                parts["head"][(x, y, z)] = c
            elif x < core_left:
                parts["arm_l"][(x, y, z)] = c
            elif x > core_right:
                parts["arm_r"][(x, y, z)] = c
            else:
                parts["torso"][(x, y, z)] = c
        elif y >= leg_top:
            # Torso band: split arms from core torso
            if x < core_left:
                parts["arm_l"][(x, y, z)] = c
            elif x > core_right:
                parts["arm_r"][(x, y, z)] = c
            else:
                parts["torso"][(x, y, z)] = c
        else:
            # Leg band: split left/right
            if x < mid_x:
                parts["leg_l"][(x, y, z)] = c
            else:
                parts["leg_r"][(x, y, z)] = c

    # If arms are empty (common for small models), merge them into torso
    for arm in ["arm_l", "arm_r"]:
        if len(parts[arm]) == 0:
            del parts[arm]

    return parts


def auto_split_quadruped(voxels, vox_size):
    """Split voxels into quadruped parts by spatial heuristics.

    Quadruped layout (Z axis = front-to-back):
      - head:   front 25% (high Z values after axis swap)
      - torso:  middle
      - tail:   back 15% (low Z) and above leg height
      - legs:   4 quadrants (left/right × front/back) below torso center
    """
    (min_x, min_y, min_z), (max_x, max_y, max_z) = compute_bounds(voxels)
    depth = max_z - min_z + 1
    height = max_y - min_y + 1
    mid_x = (min_x + max_x) / 2
    mid_z = (min_z + max_z) / 2

    # Determine front/back of the model. The "front" is wherever
    # the highest voxels cluster along Z — the head sticks up.
    top_25 = min_y + height * 0.75
    top_zs = [z for (x, y, z) in voxels if y >= top_25]
    if top_zs:
        avg_top_z = sum(top_zs) / len(top_zs)
        front_is_high_z = avg_top_z > mid_z
    else:
        front_is_high_z = True

    if front_is_high_z:
        head_z = min_z + depth * 0.65
        tail_z = min_z + depth * 0.25
    else:
        head_z = min_z + depth * 0.35  # inverted
        tail_z = min_z + depth * 0.75

    # Y threshold: legs are the bottom portion
    leg_top = min_y + height * 0.50

    # Head: the front-top region (above leg line AND in front)
    head_y = min_y + height * 0.55  # head must be in upper half

    parts = {
        "head": {},
        "torso": {},
        "leg_fl": {},
        "leg_fr": {},
        "leg_bl": {},
        "leg_br": {},
        "tail": {},
    }

    for (x, y, z), c in voxels.items():
        is_front = z >= head_z if front_is_high_z else z <= head_z
        is_back = z <= tail_z if front_is_high_z else z >= tail_z
        is_upper = y >= leg_top
        front_half = z >= mid_z if front_is_high_z else z <= mid_z

        if is_front and y >= head_y:
            parts["head"][(x, y, z)] = c
        elif is_back and is_upper:
            parts["tail"][(x, y, z)] = c
        elif not is_upper:
            # Legs: 4 quadrants
            if front_half:
                if x < mid_x:
                    parts["leg_fl"][(x, y, z)] = c
                else:
                    parts["leg_fr"][(x, y, z)] = c
            else:
                if x < mid_x:
                    parts["leg_bl"][(x, y, z)] = c
                else:
                    parts["leg_br"][(x, y, z)] = c
        else:
            parts["torso"][(x, y, z)] = c

    # Remove empty parts
    parts = {k: v for k, v in parts.items() if len(v) > 0}
    return parts


def detect_shape(voxels, vox_size):
    """Guess if this is a humanoid or quadruped from mass distribution.

    Humanoids have most mass in the top half vertically.
    Quadrupeds have mass spread evenly across the depth axis with
    legs below the center of mass.
    """
    (min_x, min_y, min_z), (max_x, max_y, max_z) = compute_bounds(voxels)
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    d = max_z - min_z + 1

    # Strong aspect ratio signal
    if h > d * 1.5:
        return "humanoid"
    if d > h * 1.2:
        return "quadruped"

    # Ambiguous aspect ratio — check mass distribution.
    # Humanoids: narrow in X relative to Y. Quadrupeds: wide in both X and Z.
    if w < h * 0.6:
        return "humanoid"

    # Default to quadruped for roughly cubic/squat shapes (animals)
    return "quadruped"


# ============================================= procedural animations

def make_procedural_animations(parts, shape):
    """Generate walk and idle animations for auto-split parts."""
    animations = {}

    if shape == "humanoid":
        animations["walk"] = make_humanoid_walk(parts)
        animations["idle"] = make_humanoid_idle(parts)
    else:
        animations["walk"] = make_quadruped_walk(parts)
        animations["idle"] = make_quadruped_idle(parts)

    return animations


def _swing_keyframes(part_names, swing_rad, n_frames=12, duration=0.8):
    """Generate a looping swing cycle: parts alternate sinusoidally."""
    keyframes = []
    for i in range(n_frames):
        t = i / n_frames
        time_s = t * duration
        angle = math.sin(t * 2 * math.pi) * swing_rad
        parts = {}
        for j, name in enumerate(part_names):
            # Alternate sign for opposing limbs
            sign = 1 if j % 2 == 0 else -1
            q = _quat_from_axis_angle([1, 0, 0], sign * angle)  # rotate around X
            parts[name] = {"rotation": q}
        keyframes.append({"time": round(time_s, 4), "parts": parts})
    return keyframes


def make_humanoid_walk(parts):
    limbs = []
    for name in ["leg_l", "leg_r", "arm_l", "arm_r"]:
        if name in parts:
            limbs.append(name)

    kf = _swing_keyframes(limbs, swing_rad=0.5, n_frames=16, duration=0.8)
    return {"duration": 0.8, "looping": True, "keyframes": kf}


def make_humanoid_idle(parts):
    # Subtle head bob
    keyframes = [
        {"time": 0.0, "parts": {}},
        {"time": 0.8, "parts": {
            "head": {"translation": [0, 0.15, 0]}
        } if "head" in parts else {}},
        {"time": 1.6, "parts": {}},
    ]
    return {"duration": 1.6, "looping": True, "keyframes": keyframes}


def make_quadruped_walk(parts):
    # Quadruped walk: diagonal gait (FL+BR together, FR+BL together)
    leg_pairs = []
    if "leg_fl" in parts:
        leg_pairs.append("leg_fl")
    if "leg_br" in parts:
        leg_pairs.append("leg_br")
    if "leg_fr" in parts:
        leg_pairs.append("leg_fr")
    if "leg_bl" in parts:
        leg_pairs.append("leg_bl")

    # Also animate tail
    tail_parts = [n for n in parts if "tail" in n]

    n_frames = 16
    duration = 0.7
    swing = 0.4
    keyframes = []
    for i in range(n_frames):
        t = i / n_frames
        time_s = t * duration
        angle = math.sin(t * 2 * math.pi) * swing
        frame_parts = {}

        # Legs: diagonal pairs swing together
        for j, name in enumerate(leg_pairs):
            sign = 1 if j < 2 else -1  # first pair vs second pair
            q = _quat_from_axis_angle([1, 0, 0], sign * angle)
            frame_parts[name] = {"rotation": q}

        # Tail wag
        for name in tail_parts:
            tail_angle = math.sin(t * 4 * math.pi) * 0.3  # faster wag
            q = _quat_from_axis_angle([0, 1, 0], tail_angle)
            frame_parts[name] = {"rotation": q}

        keyframes.append({"time": round(time_s, 4), "parts": frame_parts})

    return {"duration": duration, "looping": True, "keyframes": keyframes}


def make_quadruped_idle(parts):
    # Subtle body sway + tail wag
    n_frames = 12
    duration = 2.0
    keyframes = []
    tail_parts = [n for n in parts if "tail" in n]

    for i in range(n_frames):
        t = i / n_frames
        frame_parts = {}
        for name in tail_parts:
            angle = math.sin(t * 2 * math.pi) * 0.2
            q = _quat_from_axis_angle([0, 1, 0], angle)
            frame_parts[name] = {"rotation": q}
        keyframes.append({"time": round(t * duration, 4), "parts": frame_parts})

    return {"duration": duration, "looping": True, "keyframes": keyframes}


def _quat_from_axis_angle(axis, angle):
    """Axis-angle to quaternion [x, y, z, w]."""
    ax = np.array(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    return [round(ax[0] * s, 4), round(ax[1] * s, 4), round(ax[2] * s, 4), round(c, 4)]


# ============================================= grid extraction

def extract_part_grid(part_voxels_dict):
    """Convert sparse dict → dense grid + bounds."""
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

    grid = [0] * (sx * sy * sz)
    for (x, y, z), color in part_voxels_dict.items():
        lx = x - min_x
        ly = y - min_y
        lz = z - min_z
        grid[(lz * sy + ly) * sx + lx] = color

    return {
        "min": [min_x, min_y, min_z],
        "size": [sx, sy, sz],
        "voxels": grid,
    }


def compute_pivot(part_voxels, grid_data, shape, part_name):
    """Compute a reasonable pivot point for a part.

    Pivot is where the part rotates from:
    - Limbs: top-center (shoulder/hip joint)
    - Head: bottom-center (neck joint)
    - Torso: center
    - Tail: front-center (base of tail)
    """
    sx, sy, sz = grid_data["size"]
    cx, cy, cz = sx / 2, sy / 2, sz / 2

    if "head" in part_name:
        return [cx, 0, cz]  # rotate from neck (bottom)
    elif "arm" in part_name or "leg" in part_name:
        return [cx, sy, cz]  # rotate from shoulder/hip (top)
    elif "tail" in part_name:
        # Rotate from the end closest to the torso
        return [cx, cy, sz]  # back end (highest Z = closest to torso for quadruped)
    else:
        return [cx, cy, cz]  # center


# ============================================= skeleton mode helpers

def fuzzy_match_bone(pattern, bone_names):
    results = []
    pattern_lower = pattern.lower()
    for name in bone_names:
        stripped = name
        for prefix in ["mixamorig:", "b_", "Bip01_"]:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        stripped_base = re.sub(r'_?\d+$', '', stripped)
        if (pattern_lower in stripped.lower() or
            pattern_lower in stripped_base.lower() or
            pattern_lower == stripped_base.lower()):
            results.append(name)
    return results


def resolve_bone(pattern, bone_names):
    matches = fuzzy_match_bone(pattern, bone_names)
    return matches[0] if matches else None


# ============================================= palette

def convert_palette(vox_palette, used_indices):
    palette_out = {}
    for idx in sorted(used_indices):
        if idx == 0:
            continue
        pal_idx = (idx - 1) % len(vox_palette)
        r, g, b, a = vox_palette[pal_idx]
        palette_out[str(idx)] = [r, g, b, a]
    return palette_out


# ============================================================ main

def main():
    parser = argparse.ArgumentParser(
        description="Build NPC blueprint from .vox file (with optional skeleton)."
    )
    parser.add_argument("--vox", required=True, help=".vox character file")
    parser.add_argument("--skeleton", help="Skeleton JSON (optional — enables skeleton mode)")
    parser.add_argument("--type", choices=["humanoid", "quadruped", "auto"],
                        default="auto", help="Body type (default: auto-detect)")
    parser.add_argument("-o", "--output", required=True, help="Output blueprint JSON")
    args = parser.parse_args()

    # Load .vox
    print(f"Loading: {args.vox}", file=sys.stderr)
    vox_size, voxels, vox_palette = load_vox(args.vox)
    print(f"  Size: {vox_size[0]}x{vox_size[1]}x{vox_size[2]}, "
          f"{len(voxels)} voxels", file=sys.stderr)

    # Detect shape
    if args.type == "auto":
        shape = detect_shape(voxels, vox_size)
    else:
        shape = args.type
    print(f"  Shape: {shape}", file=sys.stderr)

    # Split into parts
    if args.skeleton:
        # Skeleton mode — not yet reimplemented in this version
        print("Skeleton mode: loading skeleton...", file=sys.stderr)
        with open(args.skeleton) as f:
            skel_data = json.load(f)
        # TODO: skeleton-driven splitting (previous implementation)
        # For now, fall through to auto mode with a warning
        print("  Warning: skeleton-driven splitting not yet integrated, using auto mode",
              file=sys.stderr)
        if shape == "quadruped":
            part_voxels = auto_split_quadruped(voxels, vox_size)
        else:
            part_voxels = auto_split_humanoid(voxels, vox_size)
        skeleton_animations = skel_data.get("animations", {})
    else:
        if shape == "quadruped":
            part_voxels = auto_split_quadruped(voxels, vox_size)
        else:
            part_voxels = auto_split_humanoid(voxels, vox_size)
        skeleton_animations = None

    # Report split
    total_assigned = sum(len(v) for v in part_voxels.values())
    print(f"  Split into {len(part_voxels)} parts ({total_assigned}/{len(voxels)} voxels assigned):",
          file=sys.stderr)
    for name, pvox in sorted(part_voxels.items()):
        print(f"    {name}: {len(pvox)} voxels", file=sys.stderr)

    # Compute the model's origin: center X/Z, bottom Y.
    # All rest_offsets will be relative to this point so the NPC's
    # WorldPosition corresponds to its feet-center.
    (all_min_x, all_min_y, all_min_z), (all_max_x, all_max_y, all_max_z) = compute_bounds(voxels)
    origin_x = (all_min_x + all_max_x + 1) / 2  # center X
    origin_y = all_min_y                          # feet at Y=0
    origin_z = (all_min_z + all_max_z + 1) / 2  # center Z
    print(f"  Model origin (feet-center): ({origin_x:.1f}, {origin_y:.1f}, {origin_z:.1f})",
          file=sys.stderr)

    # Build part grids
    parts_out = {}
    all_used_colors = set()
    for part_name, pvox in part_voxels.items():
        if not pvox:
            continue

        grid_data = extract_part_grid(pvox)
        all_used_colors.update(c for c in grid_data["voxels"] if c != 0)

        pivot = compute_pivot(pvox, grid_data, shape, part_name)

        # Rest offset: the pivot's position in NPC space.
        # The Rust renderer places the part entity at rest_offset,
        # then offsets the mesh by -pivot. So the mesh corner ends
        # up at (rest_offset - pivot) = (min_corner - origin), which
        # is correct only when rest_offset = min_corner - origin + pivot.
        rest_offset = [
            grid_data["min"][0] - origin_x + pivot[0],
            grid_data["min"][1] - origin_y + pivot[1],
            grid_data["min"][2] - origin_z + pivot[2],
        ]

        voxel_bytes = bytes(grid_data["voxels"])
        voxels_b64 = base64.b64encode(voxel_bytes).decode('ascii')

        parts_out[part_name] = {
            "size": grid_data["size"],
            "pivot": [round(v, 2) for v in pivot],
            "rest_offset": [round(v, 2) for v in rest_offset],
            "voxels_b64": voxels_b64,
            "voxel_count": sum(1 for v in grid_data["voxels"] if v != 0),
        }

    # Generate animations
    if skeleton_animations:
        # Use skeleton animation data — convert to per-part format
        # For now, generate procedural as fallback
        animations_out = make_procedural_animations(part_voxels, shape)
        print(f"  Using procedural animations (skeleton anim integration TODO)",
              file=sys.stderr)
    else:
        animations_out = make_procedural_animations(part_voxels, shape)

    for anim_name, anim in animations_out.items():
        print(f"  Animation '{anim_name}': {anim['duration']:.2f}s, "
              f"{len(anim['keyframes'])} frames, looping={anim['looping']}",
              file=sys.stderr)

    # Palette
    palette_out = convert_palette(vox_palette, all_used_colors)

    # Build output
    blueprint = {
        "source": {
            "vox": Path(args.vox).name,
            "shape": shape,
            "mode": "skeleton" if args.skeleton else "auto",
        },
        "palette": palette_out,
        "parts": parts_out,
        "animations": animations_out,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(blueprint, f, indent=2)

    file_size = output_path.stat().st_size
    print(f"\nWritten: {output_path} ({file_size / 1024:.1f} KB)", file=sys.stderr)
    print(f"  {len(parts_out)} parts, {len(animations_out)} animations, "
          f"{len(palette_out)} palette colors", file=sys.stderr)


if __name__ == "__main__":
    main()
