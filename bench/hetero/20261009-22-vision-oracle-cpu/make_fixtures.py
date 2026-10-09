#!/usr/bin/env python3
"""Create deterministic RGB image inputs for the CPU strata-vision oracle."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, __version__ as pillow_version

OUT = Path(r"E:\Strata-Hetero-data\vision-fixtures\20261009-22-vision-oracle-cpu")
MANIFEST = Path(__file__).with_name("fixture-manifest.json")


def pixels(name: str, x: int, y: int, width: int, height: int) -> tuple[int, int, int]:
    if name == "gray-96":
        return (128, 128, 128)
    if name == "quadrants-96":
        return ((255, 0, 0) if y < 48 and x < 48 else
                (0, 255, 0) if y < 48 else
                (0, 0, 255) if x < 48 else (255, 255, 0))
    if name == "checkerboard-96":
        return (255, 255, 255) if ((x // 12) + (y // 12)) % 2 else (0, 0, 0)
    return ((x * 255) // (width - 1), (y * 255) // (height - 1), (x + 3 * y) % 256)


def main() -> None:
    if OUT.exists() or MANIFEST.exists():
        raise SystemExit("refusing to overwrite fixture directory or manifest")
    OUT.mkdir(parents=True)
    specs = (("gray-96", 96, 96), ("quadrants-96", 96, 96),
             ("checkerboard-96", 96, 96), ("gradient-192x96", 192, 96))
    entries = []
    for name, width, height in specs:
        image = Image.new("RGB", (width, height))
        image.putdata([pixels(name, x, y, width, height)
                       for y in range(height) for x in range(width)])
        path = OUT / f"{name}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        data = path.read_bytes()
        entries.append({"name": name, "path": str(path), "width": width, "height": height,
                        "mode": "RGB", "format": "PNG", "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "pixel_rule": ("constant RGB(128,128,128)" if name == "gray-96" else
                                       "four 48x48 RGB quadrants: red, green, blue, yellow" if name == "quadrants-96" else
                                       "12x12 alternating black/white checkerboard" if name == "checkerboard-96" else
                                       "RGB=(floor(x*255/191),floor(y*255/95),(x+3*y) mod 256)" )})
    receipt = {"schema": "strata-vision-fixtures-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "generator": "make_fixtures.py deterministic Pillow RGB pixels; PNG optimize=False compress_level=9",
               "python": __import__("platform").python_version(), "pillow": pillow_version,
               "fixtures": entries}
    MANIFEST.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
