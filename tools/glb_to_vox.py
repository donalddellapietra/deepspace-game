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
    """Sample mesh colors at voxel centers.

    Fast path: precompute per-vertex colors from the texture (one UV
    lookup per vertex, not per voxel), then nearest-vertex lookup via
    scipy cKDTree for each voxel. O(V log V) tree build + O(N log V)
    queries instead of O(N * M) mesh-surface queries on faces.

    Priority:
      1. Texture: build per-vertex colors from baseColorTexture @ UVs.
      2. Vertex colors baked in the mesh.
      3. Face colors.
      4. Fallback uniform grey.
    """
    centers = voxelized.indices_to_points(filled_coords)

    # --- Step 1: resolve per-vertex RGBA colors ---
    vertex_colors = _resolve_vertex_colors(mesh)
    if vertex_colors is None:
        print("  No texture / vertex / face colors found, using uniform grey",
              file=sys.stderr)
        return np.full((len(filled_coords), 4), [128, 128, 128, 255],
                       dtype=np.uint8)

    # --- Step 2: nearest-vertex query for each voxel ---
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  scipy missing — falling back to slow mesh.nearest path",
              file=sys.stderr)
        return _slow_face_colors(mesh, vertex_colors, centers)

    tree = cKDTree(mesh.vertices)
    _dist, nearest_vidx = tree.query(centers, k=1, workers=-1)
    colors = vertex_colors[nearest_vidx].copy()
    # Force full opacity — a voxel that survived the fill pass IS present.
    # Texture alpha channels are commonly used for mask/cutout in GLBs,
    # which we don't want interpreted as per-voxel transparency.
    colors[:, 3] = 255
    return colors


def _resolve_vertex_colors(mesh):
    """Return Nx4 uint8 RGBA per vertex, or None."""
    n_verts = len(mesh.vertices)

    # Attempt 1: sample the texture at each vertex's UV.
    img = _get_texture_image(mesh)
    uvs = _get_uvs(mesh)
    if img is not None and uvs is not None:
        print(f"  Texture: {img.size[0]}x{img.size[1]} ({img.mode}), "
              f"UVs present — sampling per-vertex",
              file=sys.stderr)
        w, h = img.size
        px = np.clip((uvs[:, 0] * w).astype(np.int32), 0, w - 1)
        py = np.clip(((1.0 - uvs[:, 1]) * h).astype(np.int32), 0, h - 1)
        img_rgba = np.asarray(img.convert('RGBA'))   # (H, W, 4)
        colors = img_rgba[py, px, :].astype(np.uint8)
        if len(colors) == n_verts:
            return colors

    # Attempt 2: baked vertex colors.
    vc = getattr(mesh.visual, 'vertex_colors', None)
    if vc is not None and len(vc) == n_verts:
        colors = np.asarray(vc).astype(np.uint8)
        if colors.shape[1] == 3:
            alpha = np.full((n_verts, 1), 255, dtype=np.uint8)
            colors = np.hstack([colors, alpha])
        return colors[:, :4]

    # Attempt 3: face colors — expand to per-vertex by averaging adjacent faces.
    fc = getattr(mesh.visual, 'face_colors', None)
    if fc is not None and len(fc) == len(mesh.faces):
        # Accumulate each vertex's incident face colors.
        sums = np.zeros((n_verts, 4), dtype=np.float64)
        counts = np.zeros(n_verts, dtype=np.int64)
        for face_idx, face in enumerate(mesh.faces):
            c = fc[face_idx][:4]
            for v in face:
                sums[v] += c
                counts[v] += 1
        counts = np.maximum(counts, 1)
        return (sums / counts[:, None]).astype(np.uint8)

    return None


def _slow_face_colors(mesh, vertex_colors, centers):
    """Fallback path without scipy — slower but works."""
    _cp, _d, face_indices = mesh.nearest.on_surface(centers)
    face_verts = mesh.faces[face_indices]                  # (N, 3)
    colors = np.mean(vertex_colors[face_verts], axis=1).astype(np.uint8)
    return colors


def _get_texture_image(mesh):
    """Return the baseColor PIL image from a PBR material, or None."""
    visual = mesh.visual
    if not hasattr(visual, 'material') or visual.material is None:
        return None
    mat = visual.material
    for attr in ('baseColorTexture', 'image'):
        img = getattr(mat, attr, None)
        if img is not None:
            return img
    return None


def _get_uvs(mesh):
    """Return per-vertex UVs (Nx2 array) or None."""
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        return None
    uv = np.asarray(mesh.visual.uv)
    if uv.shape[0] != len(mesh.vertices):
        return None
    return uv


def write_output(path, filled, colors, shape):
    """Dispatch write based on file extension and grid size.

    - `.vxs` → custom sparse binary (no per-axis limit, arbitrary palette).
    - `.vox` → MagicaVoxel binary (256-per-axis limit, 255-color palette).
      When any dimension exceeds 256, we auto-switch to `.vxs` with a
      warning so content doesn't get silently truncated.
    """
    path = str(path)
    sx, sy, sz = int(shape[0]), int(shape[1]), int(shape[2])
    if path.endswith('.vxs') or max(sx, sy, sz) > 256:
        if path.endswith('.vox'):
            path = path[:-4] + '.vxs'
            print(f"  grid {sx}x{sy}x{sz} exceeds 256 — writing .vxs at {path}",
                  file=sys.stderr)
        write_vxs(path, filled, colors, shape)
    else:
        write_vox(path, filled, colors, shape)


def write_vxs(path, filled, colors, shape):
    """Custom sparse voxel format.

    Layout:
        magic:        b"DSVX"           (4 bytes)
        version:      u32 (=1)
        size_x, y, z: u32 × 3
        palette_n:    u32               (1..=4096, dedup'd RGBA)
        palette:      [u8; 4] × palette_n
        voxel_n:      u32
        voxels:       (u32 x, u32 y, u32 z, u32 palette_idx) × voxel_n

    Read by `src/import/vxs.rs`.
    """
    # Cap palette at 200 colors to fit comfortably in ColorRegistry
    # (which has a 256-entry limit shared with system/block colors).
    unique_colors, color_indices = quantize_palette(colors, max_colors=200)
    sx, sy, sz = int(shape[0]), int(shape[1]), int(shape[2])

    with open(path, 'wb') as f:
        f.write(b'DSVX')
        f.write(struct.pack('<I', 1))                       # version
        f.write(struct.pack('<III', sx, sy, sz))
        f.write(struct.pack('<I', len(unique_colors)))
        for r, g, b, a in unique_colors:
            f.write(struct.pack('<BBBB', r, g, b, a))
        f.write(struct.pack('<I', len(filled)))
        # Pack as contiguous ndarray for speed. trimesh is already Y-up,
        # matching our convention — no axis swap needed.
        idx = np.asarray(color_indices, dtype=np.uint32)    # 0-based
        buf = np.empty((len(filled), 4), dtype=np.uint32)
        buf[:, 0] = filled[:, 0]
        buf[:, 1] = filled[:, 1]
        buf[:, 2] = filled[:, 2]
        buf[:, 3] = idx
        f.write(buf.tobytes())

    print(f"  Written: {path} ({len(filled)} voxels, "
          f"{len(unique_colors)} palette colors, vxs format)", file=sys.stderr)


def write_vox(path, filled, colors, shape):
    """Write a MagicaVoxel .vox file.

    .vox format: https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt
    Limited to 256-per-axis and 255 palette colors. Use .vxs for larger.
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

    write_output(args.output, filled, colors, shape)


if __name__ == "__main__":
    main()
