"""Builds Magenta.zip, the StadiumCar skin collection drives with.

Body only: the zip holds Diffuse.dds (the body paint) and Icon.dds (the menu
thumbnail) and no Details.dds, so wheels, driver and suspension keep the game's
own textures. One flat colour that appears nowhere on a track, so the car is
unmistakable in every frame.

Solid DXT1 with a full mip chain: every 4x4 block is the same eight bytes,
both endpoints the colour and all indices zero.

    python skins/make_magenta.py

The seeded player profile records this zip's MD5 along with its path, so a
rebuilt zip has to come out byte for byte the same (hence the fixed
timestamps), or the profile has to be re-saved from the game with the new one.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

MAGENTA = (255, 0, 255)
OUT = Path(__file__).with_name("Magenta.zip")


def _rgb565(r: int, g: int, b: int) -> int:
    return (r >> 3) << 11 | (g >> 2) << 5 | b >> 3


def solid_dxt1(size: int, rgb: tuple[int, int, int]) -> bytes:
    """A square DXT1 DDS of one colour, with every mip level down to 1x1."""
    colour = _rgb565(*rgb)
    block = struct.pack("<HHI", colour, colour, 0)
    levels = []
    side = size
    while True:
        blocks = max(1, side // 4)
        levels.append(block * blocks * blocks)
        if side == 1:
            break
        side //= 2

    header = struct.pack(
        "<4s7I44x8I5I",
        b"DDS ",
        124,  # header size
        0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000,  # caps|h|w|pf|mips|linear
        size,
        size,
        len(levels[0]),  # linear size of the top level
        0,  # depth
        len(levels),
        # DDS_PIXELFORMAT
        32,
        0x4,  # DDPF_FOURCC
        int.from_bytes(b"DXT1", "little"),
        0, 0, 0, 0, 0,
        0x8 | 0x1000 | 0x400000,  # complex|texture|mipmap
        0, 0, 0, 0,
    )
    return header + b"".join(levels)


def build() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, size in (("Diffuse.dds", 512), ("Icon.dds", 64)):
            # Fixed timestamp, so rebuilding gives the same bytes.
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, solid_dxt1(size, MAGENTA))
    return buffer.getvalue()


if __name__ == "__main__":
    OUT.write_bytes(build())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
