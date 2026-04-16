#!/usr/bin/env python3
import math
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required for scripts/analyze-png.py") from exc


def luminance(pixel):
    r, g, b, _ = pixel
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def analyze(path: Path) -> None:
    img = Image.open(path).convert("RGBA")
    width, height = img.size
    px = img.load()
    total = 0.0
    total_sq = 0.0
    yellow = 0
    edge = 0
    gray = 0
    count = width * height
    for y in range(height):
        for x in range(width):
            p = px[x, y]
            lum = luminance(p)
            total += lum
            total_sq += lum * lum
            r, g, b, _ = p
            if r > 220 and g > 180 and b < 140:
                yellow += 1
            if abs(r - g) < 6 and abs(g - b) < 6:
                gray += 1
            if x + 1 < width:
                if abs(lum - luminance(px[x + 1, y])) > 8:
                    edge += 1
            if y + 1 < height:
                if abs(lum - luminance(px[x, y + 1])) > 8:
                    edge += 1
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    print(
        f"{path} size={width}x{height} mean={mean:.2f} var={variance:.2f} "
        f"yellow={yellow} edge={edge} gray_frac={gray / count:.3f}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: scripts/analyze-png.py IMAGE.png [...]")
    for arg in sys.argv[1:]:
        analyze(Path(arg))


if __name__ == "__main__":
    main()
