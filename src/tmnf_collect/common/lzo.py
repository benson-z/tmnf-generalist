"""LZO1X for Gbx bodies, through lzokay rather than python-lzo.

python-lzo's newest wheels are cp311, so depending on it pinned the whole
project to Python 3.11 and made every other platform build it against liblzo2.
lzokay ships abi3 wheels. Both read each other's streams, and every stripped
map body in a 3860-map check decoded identically through liblzo2. Its compressor
is not python-lzo's LZO1X-999, though: the same maps came out about 400 bytes
larger each. Still a standard LZO1X stream, just not the smallest one.

pygbx does `import lzo` itself, so `register()` puts this module in its place
before pygbx is imported. Only the raw form Gbx uses is supported: no
python-lzo header, and the decompressed size known up front.
"""

from __future__ import annotations

import sys

import lzokay


def decompress(data: bytes, header: bool = True, buflen: int | None = None) -> bytes:
    if header or buflen is None:
        raise ValueError("only headerless LZO with a known size is supported")
    return lzokay.decompress(data, buflen)


def compress(data: bytes, level: int = 9, header: bool = True) -> bytes:
    # lzokay has one compression level; `level` is accepted for python-lzo's
    # signature.
    if header:
        raise ValueError("only headerless LZO is supported")
    return lzokay.compress(data)


def register() -> None:
    """Make `import lzo` (as pygbx does) resolve to this module."""
    sys.modules.setdefault("lzo", sys.modules[__name__])
