#!/usr/bin/env bash
set -euo pipefail

# Extract and resize block textures from a Minecraft-format texture pack.
#
# Usage: ./tools/extract_textures.sh <zip_path> [output_dir]
#
# The BLOCKS list below is the only part that changes per texture pack.
# Each line is: block_name|path_inside_textures_block_dir|animated
# Set animated=1 for animation strips (crops first frame).

ZIP_PATH="${1:?Usage: $0 <zip_path> [output_dir]}"
OUTPUT_DIR="${2:-assets/textures/blocks}"
TILE_SIZE=32

# ── Mapping table ────────────────────────────────────────────────
# Edit these paths when switching to a different texture pack.
# Format: name|relative_path|animated(0/1)
BLOCKS="
stone|stone.png|0
dirt|dirt.png|0
grass|grass_block_top.png|0
wood|oak_log.png|0
leaf|oak_leaves.png|0
sand|sand.png|0
water|water_still.png|1
brick|bricks.png|0
metal|iron_block.png|0
glass|glass.png|0
"

# ── Preflight ────────────────────────────────────────────────────

if [ ! -f "$ZIP_PATH" ]; then
    echo "Error: file not found: $ZIP_PATH" >&2
    exit 1
fi

if ! command -v sips &>/dev/null; then
    echo "Error: sips not found (macOS built-in)." >&2
    exit 1
fi

# ── Extract ──────────────────────────────────────────────────────

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Extracting textures from $(basename "$ZIP_PATH")..."
unzip -q -o "$ZIP_PATH" "assets/minecraft/textures/block/*" -d "$TMPDIR" 2>/dev/null || true

TEX_ROOT="$TMPDIR/assets/minecraft/textures/block"

if [ ! -d "$TEX_ROOT" ]; then
    echo "Error: no assets/minecraft/textures/block/ directory found in zip" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Process each block ───────────────────────────────────────────

ok=0
fail=0

while IFS='|' read -r block relpath animated; do
    # Skip empty lines
    [ -z "$block" ] && continue

    src="$TEX_ROOT/$relpath"

    if [ ! -f "$src" ]; then
        echo "  MISS  $block  ($relpath not found)"
        fail=$((fail + 1))
        continue
    fi

    dst="$OUTPUT_DIR/${block}.png"
    cp "$src" "$dst"

    if [ "$animated" = "1" ]; then
        # Animated strip: crop to first tile-sized frame.
        sips -c "$TILE_SIZE" "$TILE_SIZE" "$dst" --out "$dst" &>/dev/null
    else
        sips -z "$TILE_SIZE" "$TILE_SIZE" "$dst" --out "$dst" &>/dev/null
    fi

    echo "  OK    $block  ($relpath)"
    ok=$((ok + 1))
done <<< "$BLOCKS"

echo ""
echo "Done: $ok extracted, $fail missing → $OUTPUT_DIR/"
