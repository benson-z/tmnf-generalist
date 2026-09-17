"""How the candidate codecs behave under the collector's actual concurrency.

Single-frame timings mislead here. Collection runs every game instance as a
thread in one process -- a socket reader and an encoder thread each -- so what
decides a codec is whether it releases the GIL while it works. One that does not
turns eight encoder threads back into one and starves the readers, and a reader
that stops draining backs its plugin's writes up until one fails, which
disconnects that instance for the rest of the run.

Reported speedup is against the same codec on one thread. Anything near 1.0 is
holding the GIL.

Run `frame_codec_bench.py` first; this reads the frames it captured.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def _pillow_encoder(fmt: str, **options):
    def encode(frame: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, fmt, **options)
        return buffer.getvalue()

    return encode


def finalists() -> dict:
    entries = {
        "jpeg-q80": _pillow_encoder("JPEG", quality=80, subsampling=2),
        "png-3": _pillow_encoder("PNG", compress_level=3),
        "png-6": _pillow_encoder("PNG", compress_level=6),
        "png-9": _pillow_encoder("PNG", compress_level=9),
        "webp-m0": _pillow_encoder("WEBP", lossless=True, method=0, quality=0),
        "webp-m1": _pillow_encoder("WEBP", lossless=True, method=1, quality=0),
        "webp-m2": _pillow_encoder("WEBP", lossless=True, method=2, quality=0),
    }
    try:
        import qoi

        entries["qoi"] = lambda frame: bytes(qoi.encode(frame))
    except ImportError:
        return entries

    try:
        import zstandard
    except ImportError:
        return entries

    # A ZstdCompressor is not safe to share between threads; sharing one
    # segfaults the interpreter rather than raising. One per thread.
    local = threading.local()

    def compressor(level: int):
        found = getattr(local, f"c{level}", None)
        if found is None:
            found = zstandard.ZstdCompressor(level=level)
            setattr(local, f"c{level}", found)
        return found

    entries["zstd-3"] = lambda frame: compressor(3).compress(frame.tobytes())
    for level in (1, 3, 6, 9):
        entries[f"qoi+zstd-{level}"] = (
            lambda frame, lvl=level: compressor(lvl).compress(bytes(qoi.encode(frame)))
        )
    return entries


def throughput(encode, frames: np.ndarray, threads: int, rounds: int) -> float:
    """Frames encoded per second with `threads` workers all encoding."""
    work = [frames[i % len(frames)] for i in range(rounds)]
    started = time.perf_counter()
    if threads == 1:
        for frame in work:
            encode(frame)
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(encode, work, chunksize=8))
    return len(work) / (time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames", type=Path, default=Path("out/codec-bench/frames.npy")
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument(
        "--out", type=Path, default=Path("out/codec-bench/threads.json")
    )
    parser.add_argument("--only", default=None, help="comma-separated codec names")
    args = parser.parse_args()

    frames = np.load(args.frames)
    chosen = set(args.only.split(",")) if args.only else None
    rows = []
    for name, encode in finalists().items():
        if chosen is not None and name not in chosen:
            continue
        size = sum(len(encode(frame)) for frame in frames) / len(frames)
        one = throughput(encode, frames, 1, args.rounds)
        many = throughput(encode, frames, args.threads, args.rounds)
        rows.append({
            "name": name,
            "bytes": size,
            "frames_per_second_1": one,
            "frames_per_second_n": many,
            "speedup": many / one,
        })

    rows.sort(key=lambda row: -row["frames_per_second_n"])
    header = (
        f"{'codec':14} {'size':>9} {'GB/hour':>8} {'1 thread':>10} "
        f"{f'{args.threads} threads':>11} {'scaling':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        # One hour of gameplay is 72,000 frames at 20 Hz.
        print(
            f"{row['name']:14} {row['bytes'] / 1024:7.1f}KB "
            f"{row['bytes'] * 72000 / 1e9:7.2f} "
            f"{row['frames_per_second_1']:9.0f}/s "
            f"{row['frames_per_second_n']:10.0f}/s "
            f"{row['speedup']:7.2f}x"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
