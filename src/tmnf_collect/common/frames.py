"""How a captured frame becomes bytes on disk.

The plugin sends raw BGRA. What that turns into is paid for three times: encode
time on the writer thread, bytes on disk, and decode time on every training
epoch. Measured on 200 real 320x240 frames straight out of the game (the
benchmark script is in git history, `experiments/frame_codec_bench.py` as of
bcc828e):

    codec              size   1 thread   8 threads  scaling   decode
    qoi + zstd-6      70.0KB    ~590/s      3669/s     ~6x      0.9ms
    webp lossless m0  66.9KB      295/s      1054/s    3.6x      1.3ms
    png-6             75.9KB      183/s       835/s    4.6x      1.0ms
    png-3             80.5KB      374/s      2072/s    5.5x      1.1ms
    qoi alone         85.6KB     1402/s      6742/s    4.8x      0.4ms
    jpeg q80 (lossy)  11.6KB     5712/s      4849/s    0.85x     0.3ms

The scaling column is why this is QOI packed with zstd rather than the smallest
option. Every game lane has one socket reader and one encoder thread. The
default coordinator isolates lanes in subprocesses; the fallback thread mode
shares one interpreter. An encoder that holds its process's GIL can still
starve that lane's socket reader, and a reader that stops draining backs the
plugin's writes up until one fails, disconnecting the instance for the run.
Both codec stages release the GIL, so the reader and encoder overlap cleanly.

The size cost of that choice is 4.6% against lossless WebP, bought with 3.5x
the encoding rate; against PNG there is no trade at all, since this is smaller
*and* faster at every compression level. Eight instances at 2x need 320 frames
a second while eight games are already using the CPU, and the margin is the
point: falling behind does not degrade gracefully, it drops an instance.

Lossless is what makes frames comparable across a re-collection. JPEG q80 costs
six times less space, but its artifacts vary with content, so the same corner
recorded twice differs in ways the model can see.
"""

from __future__ import annotations

import io

import numpy as np
import qoi
import zstandard
from PIL import Image

# What `collect` writes. Every run records the codec it was written with, so a
# corpus recorded before this still reads back.
LOSSLESS = "qoi-zstd"
_ZSTD_LEVEL = 6


def to_rgb(pixels: bytes, width: int, height: int) -> np.ndarray:
    """The plugin's BGRA buffer as an RGB array.

    Pillow's raw decoder does the channel swap in C and measured 0.148 ms
    against 0.210 ms for the equivalent numpy stride trick.
    """
    image = Image.frombytes("RGBA", (width, height), pixels, "raw", "BGRA")
    return np.asarray(image.convert("RGB"))


class Encoder:
    """Encodes one run's frames.

    Held per run rather than shared: a `ZstdCompressor` is not safe to use from
    several threads at once, and doing so segfaults the interpreter rather than
    raising. One writer thread owns one of these.
    """

    def __init__(self, codec: str = LOSSLESS) -> None:
        if codec != LOSSLESS:
            raise ValueError(f"unknown frame codec {codec!r}")
        self.codec = codec
        self._zstd = zstandard.ZstdCompressor(level=_ZSTD_LEVEL)

    def encode(self, pixels: bytes, width: int, height: int) -> bytes:
        return self._zstd.compress(bytes(qoi.encode(to_rgb(pixels, width, height))))


def decode(payload: bytes, *, codec: str = LOSSLESS) -> Image.Image:
    """One stored frame as an image, whichever codec wrote it."""
    if codec == LOSSLESS:
        raw = zstandard.ZstdDecompressor().decompress(payload)
        return Image.fromarray(qoi.decode(np.frombuffer(raw, dtype=np.uint8)))
    # Everything else this project has written is self-describing.
    return Image.open(io.BytesIO(payload))
