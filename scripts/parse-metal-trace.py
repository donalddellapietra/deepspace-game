#!/usr/bin/env python3
"""Parse a Metal GPU Counters trace (.trace bundle) and summarize key counters.

Extracts the XML via `xctrace export`, then streams through the XML line
by line to build per-counter distributions. The XML is typically 1+ GB
for a 20-second capture, so we stream rather than load-all.

Usage:
    scripts/parse-metal-trace.py <path-to-trace>

Typical output shows mean / p50 / p90 / p99 / max for counters like:
  - Fragment Occupancy: % of peak parallelism. Low = register pressure.
  - ALU Utilization: % of peak compute. Low = memory-stalled or low occupancy.
  - Buffer Read Limiter: % saturated on buffer loads.
  - Buffer Write Limiter: % saturated on tile resolve / writes.
  - GPU Last Level Cache Utilization: % saturated on SLC.

Quick interpretation rules:
  - Fragment Occupancy <25% → register pressure. Shader holds too much
    per-thread state. Reduce stack depth, pack values tighter, consider f16.
  - ALU Utilization >70%, Occupancy high → compute-bound. Optimize math.
  - Buffer Read Limiter >50% → memory-bandwidth-bound. Use less data.
  - All three low → launch/dispatch overhead or serial stalls.

See `docs/testing/gpu-telemetry.md` for the full interpretation guide.
"""
from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict


def extract_xml(trace_path: str, out_path: str) -> None:
    """Run `xctrace export` to dump the counter-intervals table as XML."""
    result = subprocess.run(
        [
            "xcrun", "xctrace", "export",
            "--input", trace_path,
            "--xpath",
            '/trace-toc/run/data/table[@schema="metal-gpu-counter-intervals"]',
        ],
        stdout=open(out_path, "wb"),
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode())
        sys.exit(result.returncode)


def parse_xml(path: str) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Stream-parse the XML.

    Returns (numeric_samples, top_limiter_counts).

    Each row references a counter by id or ref; we resolve refs to the
    name. Most counters store a pre-formatted percent in `<percent>` or
    `<fixed-decimal>` elements (text "0.142594111" = 0.1426%, not a
    fraction). `Top Performance Limiter` is categorical — its
    `<formatted-label>` embeds the winning limiter name, e.g.
    "0.0148% (9.Texture Write Limiter)" — so we tally those instead.
    """
    NAME_ID = re.compile(r'<gpu-counter-name id="(\d+)" fmt="([^"]*)">')
    NAME_REF = re.compile(r'<gpu-counter-name ref="(\d+)"/>')
    PERCENT = re.compile(
        r'<percent [^/>]*fmt="([^"]+)"[^/>]*>([0-9.]+)</percent>'
    )
    DECIMAL = re.compile(
        r'<fixed-decimal [^/>]*fmt="([^"]+)"[^/>]*>([0-9.]+)</fixed-decimal>'
    )
    LABEL = re.compile(r'<formatted-label[^>]*fmt="([^"]+)"')
    LIM_PAREN = re.compile(r'\(([^)]+)\)')

    id_to_name: dict[str, str] = {}
    peak: dict[str, list[float]] = defaultdict(list)
    top_limiter: dict[str, int] = defaultdict(int)
    n_rows = 0
    n_nonzero = 0

    with open(path, "rb") as f:
        for line in f:
            if not line.startswith(b"<row>"):
                continue
            n_rows += 1
            s = line.decode("utf-8", errors="ignore")

            name: str | None = None
            m = NAME_ID.search(s)
            if m:
                cid, nval = m.group(1), m.group(2)
                id_to_name[cid] = nval
                name = nval
            else:
                m = NAME_REF.search(s)
                if m:
                    name = id_to_name.get(m.group(1))
            if name is None:
                continue

            if name == "Top Performance Limiter":
                lm = LABEL.search(s)
                if lm:
                    pm = LIM_PAREN.search(lm.group(1))
                    top_limiter[pm.group(1) if pm else lm.group(1)] += 1
                continue

            val: float | None = None
            m = PERCENT.search(s)
            if m:
                try:
                    val = float(m.group(2))
                except ValueError:
                    pass
            if val is None:
                m = DECIMAL.search(s)
                if m:
                    try:
                        val = float(m.group(2))
                    except ValueError:
                        pass
            if val is None or val <= 0.0:
                continue

            n_nonzero += 1
            peak[name].append(val)

    sys.stderr.write(f"rows={n_rows} non_zero={n_nonzero}\n")
    return peak, top_limiter


# Counters most useful for diagnosing ray-march shader perf.
FOCUS = [
    "Fragment Occupancy",
    "Compute Occupancy",
    "ALU Utilization",
    "ALU Limiter",
    "F32 Utilization",
    "Buffer Read Limiter",
    "Buffer Load Utilization",
    "Buffer Write Limiter",
    "Buffer Store Utilization",
    "GPU Last Level Cache Limiter",
    "GPU Last Level Cache Utilization",
    "GPU Read Bandwidth",
    "GPU Write Bandwidth",
    "MMU Limiter",
    "MMU Utilization",
    "MMU TLB Miss Rate",
    "Threadgroup/Imageblock Load Limiter",
    "Threadgroup/Imageblock Store Limiter",
]


def summarize(peak: dict[str, list[float]], top_limiter: dict[str, int]) -> None:
    print(f"{'counter':<42} {'samples':>8} {'mean':>8} {'p50':>8} "
          f"{'p90':>8} {'p99':>8} {'max':>8}")
    print("-" * 100)
    for name in FOCUS:
        samples = peak.get(name, [])
        if not samples:
            print(f"{name:<42}  (no non-zero samples)")
            continue
        ss = sorted(samples)
        n = len(ss)
        print(
            f"{name:<42} {n:>8} {statistics.mean(ss):>8.2f} "
            f"{ss[n // 2]:>8.2f} "
            f"{ss[min(int(n * 0.9), n - 1)]:>8.2f} "
            f"{ss[min(int(n * 0.99), n - 1)]:>8.2f} "
            f"{ss[-1]:>8.2f}"
        )
    print()
    if top_limiter:
        total = sum(top_limiter.values())
        print(f"Top Performance Limiter ({total} samples):")
        for lim, n in sorted(top_limiter.items(), key=lambda x: -x[1]):
            pct = 100.0 * n / max(total, 1)
            print(f"  {pct:6.2f}%  {n:>6}  {lim}")
    else:
        print("Top Performance Limiter: (no samples)")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    trace_path = sys.argv[1]
    if not os.path.exists(trace_path):
        sys.exit(f"trace not found: {trace_path}")

    # Use a temp file because the XML is typically 1+ GB.
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
        xml_path = tf.name
    try:
        sys.stderr.write(f"extracting XML to {xml_path}\n")
        extract_xml(trace_path, xml_path)
        size_mb = os.path.getsize(xml_path) / 1e6
        sys.stderr.write(f"xml size: {size_mb:.1f} MB\n")

        peak, top_limiter = parse_xml(xml_path)
        summarize(peak, top_limiter)
    finally:
        os.unlink(xml_path)


if __name__ == "__main__":
    main()
