#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  scripts/harness-run.sh screenshot DEPTH [OUT.png]
  scripts/harness-run.sh break      DEPTH [OUT.png]
  scripts/harness-run.sh place      DEPTH [OUT.png]
  scripts/harness-run.sh view       DEPTH

Environment overrides:
  PLAIN_LAYERS   default 40
  HARNESS_WIDTH  default 960
  HARNESS_HEIGHT default 540
  TIMEOUT_SECS   default 3
  EXTRA_ARGS     extra CLI flags appended to cargo run

Examples:
  scripts/harness-run.sh screenshot 39
  scripts/harness-run.sh break 22 /tmp/depth22_break.png
  EXTRA_ARGS="--show-window" scripts/harness-run.sh view 34
EOF
  exit 1
fi

mode="$1"
depth="$2"
out="${3:-/tmp/deepspace-depth${depth}-${mode}.png}"

plain_layers="${PLAIN_LAYERS:-40}"
width="${HARNESS_WIDTH:-960}"
height="${HARNESS_HEIGHT:-540}"
timeout_secs="${TIMEOUT_SECS:-3}"

script=""
exit_after="2"
show_window=""

case "$mode" in
  screenshot)
    ;;
  break)
    script="wait:30,break,wait:8"
    exit_after="40"
    ;;
  place)
    script="wait:30,place,wait:8"
    exit_after="40"
    ;;
  view)
    show_window="--show-window"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 1
    ;;
esac

cmd=(
  cargo run --quiet --bin deepspace-game --
  --render-harness
  --disable-overlay
  --plain-world
  --plain-layers "$plain_layers"
  --spawn-depth "$depth"
  --harness-width "$width"
  --harness-height "$height"
  --timeout-secs "$timeout_secs"
)

if [[ -n "$show_window" ]]; then
  cmd+=("$show_window")
fi

if [[ "$mode" == "view" ]]; then
  cmd+=(--run-for-secs 3)
else
  cmd+=(--screenshot "$out" --exit-after-frames "$exit_after")
fi

if [[ -n "$script" ]]; then
  cmd+=(--script "$script")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"

if [[ "$mode" != "view" ]]; then
  echo "saved $out"
fi
