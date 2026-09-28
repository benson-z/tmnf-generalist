"""Removing a map's MediaTracker clips before it is ever loaded.

The intro flythrough is the single largest cost per map: measured over five
maps it averaged 20 seconds against 41 seconds of actual driving, and on some
maps it does not end at all without a keypress, so those maps cannot be
recorded unattended. Nothing in the game skips it from outside -- TMInterface's
`press` injects into the race input system, which does not exist yet during an
intro, and only a real keystroke to the foreground window works, which does not
survive parallel instances.

So take the intro out of the map instead. `CGameCtnChallenge` keeps its three
MediaTracker references -- the intro clip, the in-game clip group and the
end-race clip group -- inline in chunk 0x03043021, and each is a node written
with its own explicit index. Replacing the whole span with three null
references removes them: the node indices they used become gaps, which is
harmless because nothing outside the clips refers to them, and every later node
keeps the index it was written with.

Where the span ends is the one thing that has to be right, and body chunks are
written in ascending id order, so it ends at the next id above 0x03043021.

The map's UID lives in the uncompressed header and is untouched, so a stripped
map is still the map its replay was driven on, and the blocks are in earlier
chunks, so the track itself is byte for byte what it was. Checked on 465 maps:
all stripped, all kept their UID.

Losing the in-game clip group is a bonus rather than a cost here: those are what
hijack the camera mid-race on custom maps, which would otherwise put frames in
the training set that are not the chase camera at all.
"""

from __future__ import annotations

import struct
from pathlib import Path

from . import lzo

CLIPS_CHUNK = 0x03043021
CLASS_MIN, CLASS_MAX = 0x03043000, 0x030430FF
NULL_REF = b"\xff\xff\xff\xff"


class MediaTrackerError(RuntimeError):
    pass


def _split(data: bytes) -> tuple[int, bytes]:
    """Header length and the decompressed body."""
    if data[:3] != b"GBX":
        raise MediaTrackerError("not a Gbx file")
    offset = 3
    version, = struct.unpack_from("<H", data, offset)
    offset += 2
    _format, _ref_comp, body_comp = data[offset:offset + 3].decode("ascii")
    offset += 3
    if version >= 4:
        offset += 1
    offset += 4  # class id
    if version >= 6:
        user_size, = struct.unpack_from("<I", data, offset)
        offset += 4 + user_size
    offset += 4  # node count
    external, = struct.unpack_from("<I", data, offset)
    offset += 4
    if external:
        # The reference table is variable length and only appears on maps that
        # depend on external files. Rare, and not worth parsing to save 20s.
        raise MediaTrackerError(f"{external} external references")
    if body_comp != "C":
        raise MediaTrackerError("body is not compressed")

    head_end = offset
    raw_size, comp_size = struct.unpack_from("<II", data, offset)
    offset += 8
    body = lzo.decompress(data[offset:offset + comp_size], False, raw_size)
    return head_end, body


def strip_clips(data: bytes) -> tuple[bytes, int]:
    """Return the map with its MediaTracker clips removed, and bytes removed."""
    head_end, body = _split(data)
    at = body.find(struct.pack("<I", CLIPS_CHUNK))
    if at < 0:
        return data, 0  # no clips chunk: nothing to do, and nothing wrong
    start = at + 4

    end = None
    for offset in range(start, len(body) - 4):
        chunk, = struct.unpack_from("<I", body, offset)
        if CLIPS_CHUNK < chunk <= CLASS_MAX and chunk >= CLASS_MIN:
            end = offset
            break
    if end is None:
        raise MediaTrackerError("no chunk follows the clips")

    new_body = body[:start] + NULL_REF * 3 + body[end:]
    packed = lzo.compress(new_body, 9, False)
    rebuilt = (
        data[:head_end]
        + struct.pack("<II", len(new_body), len(packed))
        + packed
    )
    return rebuilt, end - start


def strip_file(source: Path, target: Path) -> int:
    """Write ``source`` to ``target`` without its clips. Returns bytes removed."""
    try:
        rebuilt, removed = strip_clips(source.read_bytes())
    except MediaTrackerError:
        raise
    except Exception as exc:
        # Truncated or otherwise malformed: the caller falls back to the map
        # as it is, which is only possible if this is the error it gets.
        raise MediaTrackerError(f"{source.name}: {exc}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and swapped in: a map fetched by --fetch-maps is stripped
    # in place, and a half-written one would be the only copy.
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(rebuilt)
    temporary.replace(target)
    return removed
