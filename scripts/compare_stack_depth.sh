#!/usr/bin/env bash
#
# Sweep plain_layers × spawn_xyz × spawn_depth × preset combinations,
# rendering each at MAX_STACK_DEPTH=8 (baseline) and =7 (reduced) and
# comparing pixel-by-pixel.
#
# Purpose: validate whether the stack-depth reduction from 8 to 7 is
# lossy on any tested pose. "IDENTICAL" or "pixel_diff_fraction=0"
# means Nyquist-LOD already pruned descent above depth 7 at that pose,
# so slot 8 was dead code. Any non-zero diff means stack=7 is cutting
# off detail the renderer would otherwise emit.
#
# Outputs:
#   tmp/stack_compare/stack8/<case>.png
#   tmp/stack_compare/stack7/<case>.png
#   Summary table to stdout

set -euo pipefail

cd "$(dirname "$0")/.."

ALT_STACK="${1:-7}"  # stack depth to compare against baseline=8
# Include resolution in outdir so sweeping multiple sizes doesn't
# collide.
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
OUTDIR="tmp/stack_compare/${WIDTH}x${HEIGHT}"
mkdir -p "$OUTDIR/stack8" "$OUTDIR/stack${ALT_STACK}"

# label|preset|plain_layers|spawn_x|spawn_y|spawn_z|spawn_depth
CASES=(
    "jerusalem_nucleus_d3_l20|jerusalem-cross|20|1.5|1.5|1.5|3"
    "jerusalem_nucleus_d5_l20|jerusalem-cross|20|1.5|1.5|1.5|5"
    "jerusalem_nucleus_d7_l20|jerusalem-cross|20|1.5|1.5|1.5|7"
    "jerusalem_nucleus_d10_l20|jerusalem-cross|20|1.5|1.5|1.5|10"
    "jerusalem_nucleus_d15_l20|jerusalem-cross|20|1.5|1.5|1.5|15"
    "jerusalem_corner_d5_l20|jerusalem-cross|20|2.8|2.8|2.8|5"
    "jerusalem_corner_d7_l20|jerusalem-cross|20|2.8|2.8|2.8|7"
    "jerusalem_corner_d10_l20|jerusalem-cross|20|2.8|2.8|2.8|10"
    "jerusalem_edge_d7_l20|jerusalem-cross|20|0.5|1.5|1.5|7"
    "jerusalem_nucleus_d7_l8|jerusalem-cross|8|1.5|1.5|1.5|7"
    "jerusalem_nucleus_d7_l12|jerusalem-cross|12|1.5|1.5|1.5|7"
    "jerusalem_nucleus_d7_l30|jerusalem-cross|30|1.5|1.5|1.5|7"
    "menger_d5_l8|menger|8|1.5|1.5|1.5|5"
    "menger_d7_l12|menger|12|1.5|1.5|1.5|7"
    "cantor_d5_l8|cantor-dust|8|1.5|1.5|1.5|5"
    "sierpinski_tet_d5_l8|sierpinski-tet|8|1.5|1.5|1.5|5"
    "mausoleum_d5_l8|mausoleum|8|1.5|1.5|1.5|5"
)

# (WIDTH/HEIGHT already set above with env-var defaults.)

set_stack() {
    local depth="$1"
    local file=assets/shaders/bindings.wgsl
    # macOS `sed -i` requires an explicit backup suffix; we delete
    # the .bak afterwards.
    sed -i .bak -E "s/^const MAX_STACK_DEPTH: u32 = [0-9]+u;/const MAX_STACK_DEPTH: u32 = ${depth}u;/" "$file"
    rm -f "${file}.bak"
}

render_all() {
    local outdir="$1"
    local depth="$2"
    echo "--- Rendering at MAX_STACK_DEPTH=$depth to $outdir ---"
    for case in "${CASES[@]}"; do
        IFS='|' read -r label preset layers sx sy sz sd <<< "$case"
        timeout 15 ./target/debug/deepspace-game \
            --render-harness --disable-overlay --disable-highlight \
            "--${preset}-world" --plain-layers "$layers" \
            --spawn-xyz "$sx" "$sy" "$sz" --spawn-depth "$sd" \
            --harness-width "$WIDTH" --harness-height "$HEIGHT" \
            --exit-after-frames 5 --timeout-secs 10 \
            --screenshot "$outdir/$label.png" \
            --suppress-startup-logs > /dev/null 2>&1 || echo "  WARN: $label render failed"
    done
}

# Remember current stack depth so we restore it at the end even
# if the comparison finds regressions.
orig=$(grep -oE 'MAX_STACK_DEPTH: u32 = [0-9]+' assets/shaders/bindings.wgsl | awk '{print $NF}')
trap 'set_stack "$orig"; cargo build --bin deepspace-game --quiet 2>&1 | tail -2' EXIT

set_stack 8
cargo build --bin deepspace-game --quiet 2>&1 | tail -2
render_all "$OUTDIR/stack8" 8

set_stack "$ALT_STACK"
cargo build --bin deepspace-game --quiet 2>&1 | tail -2
render_all "$OUTDIR/stack${ALT_STACK}" "$ALT_STACK"

# Build the compare bin (once, with whatever stack setting — doesn't
# matter since it's not a shader).
cargo build --bin compare_pngs --quiet 2>&1 | tail -2

echo
printf "=== Pixel-diff summary: stack=8 vs stack=%s at %dx%d ===\n" "$ALT_STACK" "$WIDTH" "$HEIGHT"
printf "%-32s %s\n" "case" "result"
for case in "${CASES[@]}"; do
    IFS='|' read -r label _ <<< "$case"
    a="$OUTDIR/stack8/$label.png"
    b="$OUTDIR/stack${ALT_STACK}/$label.png"
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then
        printf "%-32s MISSING\n" "$label"
        continue
    fi
    out=$(./target/debug/compare_pngs "$a" "$b" 2>&1 || true)
    printf "%-32s %s\n" "$label" "$out"
done
