#!/usr/bin/env python3
"""Render NPC blueprint voxels as 3D images for visual inspection.

Usage:
    python3 tools/preview_blueprint.py assets/npcs/fox.blueprint.json -o /tmp/fox_preview.png
    python3 tools/preview_blueprint.py assets/npcs/fox.blueprint.json --animate -o /tmp/fox_walk.png

Renders:
  - Default: assembled rest pose from front + side views
  - --animate: walk animation frames in a grid
  - --explode: each body part rendered separately
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import to_rgba
except ImportError:
    print("Error: matplotlib required. Install: pip3 install matplotlib")
    sys.exit(1)


def load_blueprint(path):
    with open(path) as f:
        return json.load(f)


def decode_voxels(part):
    raw = base64.b64decode(part["voxels_b64"])
    sx, sy, sz = part["size"]
    voxels = {}
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                idx = (z * sy + y) * sx + x
                if idx < len(raw) and raw[idx] != 0:
                    voxels[(x, y, z)] = raw[idx]
    return voxels


def palette_to_colors(palette):
    """Convert palette dict to color_index → RGBA float tuple."""
    colors = {}
    for idx_str, rgba in palette.items():
        r, g, b, a = rgba
        colors[int(idx_str)] = (r / 255, g / 255, b / 255, a / 255)
    return colors


def collect_voxels_assembled(bp, offset_map=None):
    """Collect all voxels in world space (rest pose)."""
    colors = palette_to_colors(bp["palette"])
    all_voxels = []  # (x, y, z, color_rgba)

    for part_name, part in bp["parts"].items():
        voxels = decode_voxels(part)
        ox, oy, oz = part["rest_offset"]
        if offset_map and part_name in offset_map:
            dx, dy, dz = offset_map[part_name]
            ox += dx
            oy += dy
            oz += dz

        for (x, y, z), cidx in voxels.items():
            color = colors.get(cidx, (0.5, 0.5, 0.5, 1.0))
            all_voxels.append((x + ox, y + oy, z + oz, color))

    return all_voxels


def render_voxels(ax, voxels, title="", elev=20, azim=-60):
    """Scatter-plot voxels on a 3D axis."""
    if not voxels:
        ax.set_title(f"{title} (empty)")
        return

    xs = [v[0] for v in voxels]
    ys = [v[2] for v in voxels]  # z → depth axis in plot
    zs = [v[1] for v in voxels]  # y → up axis in plot
    cs = [v[3] for v in voxels]

    ax.scatter(xs, ys, zs, c=cs, s=12, marker='s', depthshade=True, edgecolors='none')

    # Equal aspect ratio
    max_range = max(
        max(xs) - min(xs) if xs else 1,
        max(ys) - min(ys) if ys else 1,
        max(zs) - min(zs) if zs else 1,
    ) / 2 + 1
    mid_x = (max(xs) + min(xs)) / 2
    mid_y = (max(ys) + min(ys)) / 2
    mid_z = (max(zs) + min(zs)) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y (up)')
    ax.grid(True, alpha=0.3)


def render_rest_pose(bp, output_path):
    """Render assembled rest pose from 3 angles."""
    voxels = collect_voxels_assembled(bp)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"NPC Blueprint: {bp.get('source', {}).get('vox', '?')} — "
                 f"{len(bp['parts'])} parts, {sum(p.get('voxel_count', 0) for p in bp['parts'].values())} voxels",
                 fontsize=12)

    angles = [
        ("Front", 10, -90),
        ("3/4 View", 20, -60),
        ("Side", 10, 0),
    ]
    for i, (name, elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        render_voxels(ax, voxels, title=name, elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def render_exploded(bp, output_path):
    """Render each body part separately."""
    colors = palette_to_colors(bp["palette"])
    parts = bp["parts"]
    n = len(parts)
    if n == 0:
        print("No parts to render")
        return

    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.suptitle("Body Parts (exploded view)", fontsize=12)

    for i, (part_name, part) in enumerate(sorted(parts.items())):
        voxels_dict = decode_voxels(part)
        voxels = []
        for (x, y, z), cidx in voxels_dict.items():
            color = colors.get(cidx, (0.5, 0.5, 0.5, 1.0))
            voxels.append((x, y, z, color))

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        render_voxels(ax, voxels,
                      title=f"{part_name} ({len(voxels_dict)} voxels)\n"
                            f"size={part['size']}, pivot={[round(v,1) for v in part['pivot']]}",
                      elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def render_animation_frames(bp, anim_name, output_path, n_frames=8):
    """Render frames of an animation in a grid."""
    if "animations" not in bp or anim_name not in bp["animations"]:
        available = list(bp.get("animations", {}).keys())
        print(f"Animation '{anim_name}' not found. Available: {available}")
        return

    anim = bp["animations"][anim_name]
    keyframes = anim["keyframes"]
    if not keyframes:
        print("No keyframes")
        return

    # Sample n_frames evenly across the animation
    total = len(keyframes)
    indices = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]

    fig = plt.figure(figsize=(4 * n_frames, 6))
    fig.suptitle(f"Animation: {anim_name} ({anim['duration']:.2f}s, {total} keyframes)", fontsize=12)

    for plot_i, kf_idx in enumerate(indices):
        kf = keyframes[kf_idx]

        # Build offset map from keyframe part transforms
        # (for now just show the rotation as-is in the title; we'd need
        # full transform application for proper animated rendering)
        offset_map = {}
        for part_name, pt in kf.get("parts", {}).items():
            t = pt.get("translation", [0, 0, 0])
            offset_map[part_name] = t

        voxels = collect_voxels_assembled(bp, offset_map)
        ax = fig.add_subplot(1, n_frames, plot_i + 1, projection='3d')
        render_voxels(ax, voxels,
                      title=f"t={kf['time']:.2f}s",
                      elev=15, azim=-75)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preview NPC blueprint voxels")
    parser.add_argument("blueprint", help="Blueprint JSON file")
    parser.add_argument("-o", "--output", default="/tmp/npc_preview.png",
                        help="Output image path")
    parser.add_argument("--explode", action="store_true",
                        help="Show each part separately")
    parser.add_argument("--animate", metavar="ANIM_NAME", nargs='?', const="Walk",
                        help="Show animation frames (default: Walk)")
    parser.add_argument("--frames", type=int, default=8,
                        help="Number of animation frames to show")
    args = parser.parse_args()

    bp = load_blueprint(args.blueprint)
    print(f"Blueprint: {len(bp['parts'])} parts, "
          f"{len(bp.get('animations', {}))} animations, "
          f"{len(bp.get('palette', {}))} colors")

    if args.explode:
        render_exploded(bp, args.output)
    elif args.animate:
        render_animation_frames(bp, args.animate, args.output, args.frames)
    else:
        render_rest_pose(bp, args.output)


if __name__ == "__main__":
    main()
