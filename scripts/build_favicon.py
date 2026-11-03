"""Generate a 16x16 favicon.ico without any image dependency.

Browsers request `/favicon.ico` even when an SVG icon is linked, and a 404 in the console makes real
errors harder to spot. The icon is a small gradient square matching the brand mark.
"""

from __future__ import annotations

import struct
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "apps" / "web" / "public" / "favicon.ico"
SIZE = 16


def pixel(x: int, y: int) -> tuple[int, int, int, int]:
    # diagonal gradient from accent blue (124,156,255) to emerald (78,201,165)
    t = (x + y) / (2 * (SIZE - 1))
    r = int(124 + (78 - 124) * t)
    g = int(156 + (201 - 156) * t)
    b = int(255 + (165 - 255) * t)
    # rounded corners: transparent outside a 2px radius
    cx = min(x, SIZE - 1 - x)
    cy = min(y, SIZE - 1 - y)
    if cx < 2 and cy < 2 and (2 - cx) ** 2 + (2 - cy) ** 2 > 4:
        return (0, 0, 0, 0)
    # subtle dark border
    if cx == 0 or cy == 0:
        return (11, 15, 24, 255)
    return (b, g, r, 255)


def main() -> int:
    pixels = bytearray()
    for y in range(SIZE - 1, -1, -1):  # ICO stores bottom-up
        for x in range(SIZE):
            b, g, r, a = pixel(x, y)
            pixels += bytes((b, g, r, a))

    mask_row = (SIZE + 31) // 32 * 4
    and_mask = bytes(mask_row * SIZE)
    image_size = 40 + len(pixels) + len(and_mask)
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack("<BBBBHHII", SIZE, SIZE, 0, 0, 1, 32, image_size, 6 + 16)
    bitmap_header = struct.pack(
        "<IiiHHIIiiII",
        40,
        SIZE,
        SIZE * 2,  # height counts colour + mask
        1,
        32,
        0,
        len(pixels),
        0,
        0,
        0,
        0,
    )
    OUT.write_bytes(header + entry + bitmap_header + bytes(pixels) + and_mask)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
