#!/usr/bin/env bash
set -euo pipefail

mode="${1:-screenshot}"
shift || true

if [[ $# -eq 0 ]]; then
  layers=(39 36 34 32 22 20 18 16)
else
  layers=("$@")
fi

for depth in "${layers[@]}"; do
  out="/tmp/deepspace-depth${depth}-${mode}.png"
  scripts/harness-run.sh "$mode" "$depth" "$out"
  if [[ -f "$out" ]]; then
    scripts/analyze-png.py "$out"
  fi
done
