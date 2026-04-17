#!/usr/bin/env python3
"""Compose WGSL with #include directives into a single file for naga."""
import sys
from pathlib import Path

SHADER_DIR = Path("assets/shaders")

def compose(entry: str, included: set | None = None) -> str:
    if included is None:
        included = set()
    if entry in included:
        return ""
    included.add(entry)
    src = (SHADER_DIR / entry).read_text()
    out_lines = []
    for line in src.splitlines():
        t = line.strip()
        if t.startswith("#include"):
            name = t.split('"')[1]
            out_lines.append(compose(name, included))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)

if __name__ == "__main__":
    print(compose(sys.argv[1] if len(sys.argv) > 1 else "main.wgsl"))
