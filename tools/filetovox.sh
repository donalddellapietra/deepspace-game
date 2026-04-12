#!/usr/bin/env bash
# Convert a 3D model to MagicaVoxel .vox format using FileToVox.
#
# Usage:
#   ./tools/filetovox.sh input.glb output.vox [--scale 128]
#
# Prerequisites:
#   Install FileToVox from https://github.com/Zarbuz/FileToVox
#   (C# / .NET — run via `dotnet FileToVox.dll` or the compiled binary)
#
# The --scale flag controls voxel grid resolution (default: 128).
# Higher values = more detail but larger .vox file.

set -euo pipefail

INPUT="${1:?Usage: filetovox.sh <input> <output.vox> [--scale N]}"
OUTPUT="${2:?Usage: filetovox.sh <input> <output.vox> [--scale N]}"
SCALE="${3:---scale}"
SCALE_VAL="${4:-128}"

if command -v FileToVox &>/dev/null; then
    FileToVox --i "$INPUT" --o "$OUTPUT" "$SCALE" "$SCALE_VAL"
elif command -v dotnet &>/dev/null && [ -f "$(dirname "$0")/FileToVox.dll" ]; then
    dotnet "$(dirname "$0")/FileToVox.dll" --i "$INPUT" --o "$OUTPUT" "$SCALE" "$SCALE_VAL"
else
    echo "Error: FileToVox not found."
    echo "Install from: https://github.com/Zarbuz/FileToVox"
    exit 1
fi

echo "Converted: $INPUT → $OUTPUT (scale=$SCALE_VAL)"
