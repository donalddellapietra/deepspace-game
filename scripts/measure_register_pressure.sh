#!/bin/bash
#
# Measure per-thread register pressure of march_cartesian by compiling
# a compute-shader proxy (assets/shaders/measure_compute.wgsl) and
# querying MTLComputePipelineState.maxTotalThreadsPerThreadgroup via
# a Swift helper.
#
# Lower max_threads = more registers per thread. Trivial empty kernel
# baseline = 1024. Our DDA baseline = ~512. +256 B of scalar state
# drops it to ~448. See docs/testing/perf-occupancy-stack-slim.md for
# the full methodology and tier mapping.
#
# Usage: scripts/measure_register_pressure.sh
#   (run from repo root)
#
# Prereqs: naga (cargo install naga-cli), xcrun metal, swift

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${TMPDIR:-/tmp}/deepspace-rp"
mkdir -p "$OUT"

cd "$REPO_ROOT"

python3 scripts/compose_shader.py measure_compute.wgsl > "$OUT/m.wgsl" 2>&1
naga --metal-version 2.4 "$OUT/m.wgsl" "$OUT/m.metal" 2>&1 | head -2 >&2
xcrun metal -c "$OUT/m.metal" -o "$OUT/m.air" -std=macos-metal2.4 -Wno-unused-variable 2>&1 | tail -1 >&2
xcrun metal "$OUT/m.air" -o "$OUT/m.metallib" -std=macos-metal2.4 2>&1 >&2
swift scripts/query_pipeline_stats.swift "$OUT/m.metallib" cs_measure 2>&1 | grep max_total
