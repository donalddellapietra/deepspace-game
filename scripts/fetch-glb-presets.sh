#!/usr/bin/env bash
# Fetch Sponza, Bistro, and San Miguel as packed .glb files into
# <worktree>/assets/scenes/ (gitignored).
#
# Usage:
#   scripts/fetch-glb-presets.sh                 # all three
#   scripts/fetch-glb-presets.sh sponza
#   scripts/fetch-glb-presets.sh bistro
#   scripts/fetch-glb-presets.sh san_miguel
#
# Override the output dir with GLB_DEST=/path/to/dir if needed (e.g. to feed
# external/voxel-raymarching/app/assets/models).
#
# Tools required: curl, git, unzip, npx (Node.js).
# npx will fetch @gltf-transform/cli and obj2gltf on demand.
#
# Disk: ~466 MB (San Miguel zips) + ~1.2 GB (Bistro clone) + ~200 MB (Sponza).
# Output GLBs end up multi-GB for San Miguel; expect long conversion times.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEST="${GLB_DEST:-$WORKTREE_ROOT/assets/scenes}"
WORK="${GLB_WORK:-$WORKTREE_ROOT/tmp/glb-fetch}"

mkdir -p "$DEST" "$WORK"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1" >&2; exit 1; }; }
need curl
need git
need unzip
need npx
# `magick` is only required for bistro (DDS -> PNG), checked inline.

log() { printf '\n[fetch-glb] %s\n' "$*"; }

# ----------------------------------------------------------------------------
# Sponza — Khronos glTF Sample Assets, packed to a single .glb
# ----------------------------------------------------------------------------
fetch_sponza() {
    local out="$DEST/sponza.glb"
    if [[ -f "$out" ]]; then log "sponza.glb exists, skipping"; return; fi

    local src="$WORK/sponza-khronos"
    if [[ ! -f "$src/Sponza.gltf" ]]; then
        log "cloning Khronos glTF-Sample-Assets (sparse, Sponza only)"
        rm -rf "$src" "$WORK/khronos-clone"
        git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/KhronosGroup/glTF-Sample-Assets.git "$WORK/khronos-clone"
        git -C "$WORK/khronos-clone" sparse-checkout set Models/Sponza
        mkdir -p "$src"
        cp -R "$WORK/khronos-clone/Models/Sponza/glTF/." "$src/"
    fi

    log "packing Sponza.gltf -> $out"
    npx --yes @gltf-transform/cli copy "$src/Sponza.gltf" "$out"
    log "done: $(du -h "$out" | awk '{print $1}')"
}

# ----------------------------------------------------------------------------
# Bistro — zeux/niagara_bistro (pre-converted from NVIDIA ORCA FBX)
# ----------------------------------------------------------------------------
fetch_bistro() {
    local out="$DEST/bistro.glb"
    if [[ -f "$out" ]]; then log "bistro.glb exists, skipping"; return; fi

    local src="$WORK/niagara_bistro"
    if [[ ! -f "$src/bistro.gltf" ]]; then
        log "cloning zeux/niagara_bistro (~1.2 GB)"
        rm -rf "$src"
        git clone --depth 1 https://github.com/zeux/niagara_bistro.git "$src"
    fi

    # The repo ships only DDS textures, but bistro.gltf references *.png. The
    # maintainer's etcpak.sh went PNG -> DDS and the PNG sources weren't kept.
    # Reverse the conversion (BC7 DDS -> PNG) so gltf-transform can pack.
    local dds_count png_count
    dds_count=$(find "$src" -name '*.dds' | wc -l | tr -d ' ')
    png_count=$(find "$src" -name '*.png' | wc -l | tr -d ' ')
    if [[ "$png_count" -lt "$dds_count" ]]; then
        need magick
        log "converting $dds_count DDS -> PNG (parallel, ~90s)"
        find "$src" -name '*.dds' -print0 \
            | xargs -0 -P 8 -I {} sh -c 'magick "$1" "${1%.dds}.png"' _ {}
    fi

    log "packing bistro.gltf -> $out"
    npx --yes @gltf-transform/cli copy "$src/bistro.gltf" "$out"
    log "done: $(du -h "$out" | awk '{print $1}')"
}

# ----------------------------------------------------------------------------
# San Miguel 2.0 — split-zip OBJ mirror, OBJ -> GLB via obj2gltf
# ----------------------------------------------------------------------------
fetch_san_miguel() {
    local out="$DEST/san_miguel.glb"
    if [[ -f "$out" ]]; then log "san_miguel.glb exists, skipping"; return; fi

    local src="$WORK/san-miguel"
    mkdir -p "$src"

    if ! ls "$src"/*.obj >/dev/null 2>&1; then
        if [[ ! -f "$src/San_Miguel.zip" ]]; then
            log "downloading 5 split zips (~466 MB total)"
            local base="https://raw.githubusercontent.com/jvm-graphics-labs/awesome-3d-meshes/master/McGuire/San%20Miguel%202.0"
            for n in 001 002 003 004 005; do
                local part="$src/San_Miguel.zip.$n"
                if [[ ! -f "$part" ]]; then
                    log "  -> San_Miguel.zip.$n"
                    curl -L --fail -o "$part" "$base/San_Miguel.zip.$n"
                fi
            done
            log "concatenating split zip parts"
            cat "$src"/San_Miguel.zip.0?? > "$src/San_Miguel.zip"
            rm "$src"/San_Miguel.zip.0??
        fi
        log "unzipping (~1.1 GB OBJ + textures)"
        unzip -q -o "$src/San_Miguel.zip" -d "$src"
    fi

    local obj
    obj="$(find "$src" -type f \( -iname 'san-miguel*.obj' -o -iname 'san_miguel*.obj' \) | head -1)"
    [[ -n "$obj" ]] || { echo "san-miguel*.obj not found under $src" >&2; exit 1; }

    log "converting OBJ -> GLB (this is slow — multi-GB output)"
    npx --yes obj2gltf -i "$obj" -o "$out" --binary
    log "done: $(du -h "$out" | awk '{print $1}')"
}

# ----------------------------------------------------------------------------
case "${1:-all}" in
    sponza)               fetch_sponza ;;
    bistro)               fetch_bistro ;;
    san_miguel|san-miguel) fetch_san_miguel ;;
    all)                  fetch_sponza; fetch_bistro; fetch_san_miguel ;;
    *) echo "usage: $0 [sponza|bistro|san_miguel|all]" >&2; exit 1 ;;
esac

log "all GLBs in $DEST"
ls -lh "$DEST"
