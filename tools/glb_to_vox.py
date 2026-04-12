#!/usr/bin/env python3
"""Voxelize a GLB/glTF mesh and export as .vox (MagicaVoxel format).

Usage:
    python3 tools/glb_to_vox.py assets/characters/Soldier.glb -o soldier.vox --resolution 64
    python3 tools/glb_to_vox.py assets/characters/Fox.glb -o fox.vox --resolution 80

Requires: pip3 install trimesh numpy
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    print("Error: trimesh required. Install: pip3 install trimesh")
    sys.exit(1)


def voxelize_glb(path, resolution):
    """Load a GLB and voxelize it. Returns (filled_coords, colors, shape)."""
    scene_or_mesh = trimesh.load(str(path))

    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = list(scene_or_mesh.dump())
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces",
          file=sys.stderr)
    print(f"  Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}", file=sys.stderr)
    print(f"  Extents: {mesh.extents}", file=sys.stderr)

    # Pitch: longest axis gets `resolution` voxels
    pitch = max(mesh.extents) / resolution
    voxelized = mesh.voxelized(pitch)
    print(f"  Voxel grid: {voxelized.shape}, {voxelized.filled_count} filled",
          file=sys.stderr)

    # Get filled voxel coordinates
    filled = voxelized.sparse_indices  # (N, 3) array of integer coords

    # Try to get vertex colors for each voxel
    colors = sample_colors(mesh, voxelized, filled)

    return filled, colors, voxelized.shape


def sample_colors(mesh, voxelized, filled_coords):
    """Sample mesh vertex colors at voxel centers. Falls back to grey."""
    has_colors = (hasattr(mesh.visual, 'vertex_colors') and
                  mesh.visual.vertex_colors is not None and
                  len(mesh.visual.vertex_colors) > 0)

    if not has_colors:
        # Try face colors
        has_colors = (hasattr(mesh.visual, 'face_colors') and
                      mesh.visual.face_colors is not None)

    if not has_colors:
        print("  No vertex/face colors found, using uniform grey", file=sys.stderr)
        return np.full((len(filled_coords), 4), [128, 128, 128, 255], dtype=np.uint8)

    # Get world-space centers of filled voxels
    centers = voxelized.indices_to_points(filled_coords)

    # Find nearest face for each voxel center
    closest_points, distances, face_indices = mesh.nearest.on_surface(centers)

    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vc = mesh.visual.vertex_colors
        # Average vertex colors of the nearest face
        face_verts = mesh.faces[face_indices]  # (N, 3) vertex indices
        colors = np.mean(vc[face_verts], axis=1).astype(np.uint8)
    else:
        colors = mesh.visual.face_colors[face_indices].astype(np.uint8)

    # Ensure RGBA
    if colors.shape[1] == 3:
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        colors = np.hstack([colors, alpha])

    return colors[:, :4]


def write_vox(path, filled, colors, shape):
    """Write a MagicaVoxel .vox file.

    .vox format: https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt
    """
    # Build palette: quantize colors to 256 entries
    unique_colors, color_indices = quantize_palette(colors)

    sx, sy, sz = int(shape[0]), int(shape[1]), int(shape[2])
    # .vox uses Z-up; our coords are Y-up
    # our (x, y, z) → vox (x, z, y)  [y→z because .vox Z is up]
    vox_sx, vox_sy, vox_sz = sx, sz, sy  # .vox dimensions

    # Clamp to 256 (MagicaVoxel limit)
    if max(vox_sx, vox_sy, vox_sz) > 256:
        print(f"  Warning: grid {vox_sx}x{vox_sy}x{vox_sz} exceeds 256, clamping",
              file=sys.stderr)
        vox_sx = min(vox_sx, 256)
        vox_sy = min(vox_sy, 256)
        vox_sz = min(vox_sz, 256)

    # Build XYZI chunk (voxel data)
    voxel_data = []
    for i, (x, y, z) in enumerate(filled):
        # Convert Y-up to Z-up: (x, y, z) → (x, z, y)
        vx, vy, vz = int(x), int(z), int(y)
        if vx >= vox_sx or vy >= vox_sy or vz >= vox_sz:
            continue
        ci = color_indices[i] + 1  # .vox color indices are 1-based
        voxel_data.append((vx, vy, vz, ci))

    # SIZE chunk
    size_content = struct.pack('<III', vox_sx, vox_sy, vox_sz)
    size_chunk = b'SIZE' + struct.pack('<II', len(size_content), 0) + size_content

    # XYZI chunk
    xyzi_content = struct.pack('<I', len(voxel_data))
    for vx, vy, vz, ci in voxel_data:
        xyzi_content += struct.pack('<BBBB', vx, vy, vz, ci)
    xyzi_chunk = b'XYZI' + struct.pack('<II', len(xyzi_content), 0) + xyzi_content

    # RGBA chunk (palette — 256 entries, each 4 bytes)
    rgba_content = b''
    for i in range(256):
        if i < len(unique_colors):
            r, g, b, a = unique_colors[i]
            rgba_content += struct.pack('<BBBB', r, g, b, a)
        else:
            rgba_content += struct.pack('<BBBB', 0, 0, 0, 0)
    rgba_chunk = b'RGBA' + struct.pack('<II', len(rgba_content), 0) + rgba_content

    # MAIN chunk
    children = size_chunk + xyzi_chunk + rgba_chunk
    main_chunk = b'MAIN' + struct.pack('<II', 0, len(children)) + children

    # File header
    with open(path, 'wb') as f:
        f.write(b'VOX ')
        f.write(struct.pack('<I', 150))  # version
        f.write(main_chunk)

    print(f"  Written: {path} ({len(voxel_data)} voxels, "
          f"{len(unique_colors)} palette colors)", file=sys.stderr)


def quantize_palette(colors, max_colors=255):
    """Quantize RGBA colors to at most max_colors entries."""
    # Simple approach: unique colors, then merge if > 255
    unique, inverse = np.unique(colors.reshape(-1, 4), axis=0, return_inverse=True)

    if len(unique) <= max_colors:
        return unique.tolist(), inverse.tolist()

    # Too many colors — do simple k-means-ish binning
    # Group into max_colors bins by sorting on luminance
    lum = unique[:, 0].astype(float) * 0.299 + unique[:, 1].astype(float) * 0.587 + unique[:, 2].astype(float) * 0.114
    order = np.argsort(lum)
    bin_size = len(unique) // max_colors + 1

    palette = []
    remap = np.zeros(len(unique), dtype=np.int32)
    for i in range(0, len(order), bin_size):
        group = order[i:i + bin_size]
        avg = np.mean(unique[group], axis=0).astype(np.uint8)
        pal_idx = len(palette)
        palette.append(avg.tolist())
        for g in group:
            remap[g] = pal_idx

    new_indices = remap[inverse]
    return palette, new_indices.tolist()


def main():
    parser = argparse.ArgumentParser(description="Voxelize GLB/glTF → .vox")
    parser.add_argument("input", help="GLB or glTF file")
    parser.add_argument("-o", "--output", required=True, help="Output .vox path")
    parser.add_argument("-r", "--resolution", type=int, default=64,
                        help="Voxels on longest axis (default: 64)")
    args = parser.parse_args()

    print(f"Loading: {args.input}", file=sys.stderr)
    filled, colors, shape = voxelize_glb(args.input, args.resolution)

    write_vox(args.output, filled, colors, shape)


if __name__ == "__main__":
    main()
