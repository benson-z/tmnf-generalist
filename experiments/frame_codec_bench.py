"""Pick the on-disk frame codec by measuring it on real captured frames.

The collector's cost per frame is paid three times: once encoding it on the
writer thread while the socket has to stay drained, once in bytes on disk, and
once decoding it every epoch at training time. A codec is only worth choosing
with all three measured, on frames that actually came out of the game -- game
frames are flat-shaded and heavily repeated between samples, which is nothing
like the photographic content codec defaults are tuned for.

Stage one captures a real run's frames at the requested size and keeps them raw.
Stage two runs every candidate over them. Re-running with --frames skips the
game entirely, which is what makes iterating on codecs cheap.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tmnf_collect import install, staging
from tmnf_collect.paths import detect
from tmnf_collect.session import Session
from tmnf_collect.smoke import SMOKE_SCRIPT


def capture(path: Path, *, width: int, height: int, samples: int) -> np.ndarray:
    """Drive the bootstrap map and keep every frame as raw RGB."""
    layout = detect()
    install.install(layout)
    _, track = staging.stage_bootstrap(layout)
    script = staging.write_script("tmnf_codec_bench.txt", SMOKE_SCRIPT, layout)

    frames: list[np.ndarray] = []
    with Session(port=8499, instance_id=19, width=width, height=height) as session:
        session.prepare()

        def receive(sample) -> None:
            pixels = np.frombuffer(sample.pixels, dtype=np.uint8)
            frames.append(
                pixels.reshape(sample.height, sample.width, 4)[:, :, 2::-1].copy()
            )

        session.record_map_run(
            track, script, max_samples=samples, timeout=180,
            on_sample=receive, on_reset=frames.clear,
        )

    stack = np.stack(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stack)
    return stack


# --------------------------------------------------------------- candidates


def _pillow(fmt: str, **options):
    def encode(frame: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, fmt, **options)
        return buffer.getvalue()

    def decode(payload: bytes) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))

    return encode, decode


def candidates(shape: tuple[int, int, int]) -> dict:
    height, width, _ = shape
    entries: dict[str, tuple] = {
        "jpeg-q80 (lossy, current)": _pillow("JPEG", quality=80, subsampling=2),
        "png-1": _pillow("PNG", compress_level=1),
        "png-3": _pillow("PNG", compress_level=3),
        "png-6 (default)": _pillow("PNG", compress_level=6),
        "webp-lossless-m0": _pillow("WEBP", lossless=True, method=0, quality=0),
    }

    try:
        import qoi  # type: ignore

        entries["qoi"] = (
            lambda frame: bytes(qoi.encode(frame)),
            lambda payload: qoi.decode(np.frombuffer(payload, dtype=np.uint8)),
        )
    except ImportError:
        pass

    try:
        import lz4.frame  # type: ignore

        entries["lz4-raw"] = (
            lambda frame: lz4.frame.compress(frame.tobytes()),
            lambda payload: np.frombuffer(
                lz4.frame.decompress(payload), dtype=np.uint8
            ).reshape(height, width, 3),
        )
    except ImportError:
        pass

    try:
        import zstandard  # type: ignore

        for level in (1, 3, 6, 10):
            compressor = zstandard.ZstdCompressor(level=level)
            decompressor = zstandard.ZstdDecompressor()
            entries[f"zstd-{level}-raw"] = (
                lambda frame, c=compressor: c.compress(frame.tobytes()),
                lambda payload, d=decompressor: np.frombuffer(
                    d.decompress(payload), dtype=np.uint8
                ).reshape(height, width, 3),
            )
    except ImportError:
        pass

    return entries


def _delta_sizes(frames: np.ndarray, level: int, keyframe: int) -> dict | None:
    """Frames differ little at 20 Hz; measure coding each against the last."""
    try:
        import zstandard
    except ImportError:
        return None

    compressor = zstandard.ZstdCompressor(level=level)
    decompressor = zstandard.ZstdDecompressor()
    payloads: list[bytes] = []
    started = time.perf_counter()
    previous = None
    for index, frame in enumerate(frames):
        if index % keyframe == 0 or previous is None:
            payloads.append(compressor.compress(frame.tobytes()))
        else:
            payloads.append(compressor.compress((frame - previous).tobytes()))
        previous = frame
    encode = (time.perf_counter() - started) / len(frames)

    started = time.perf_counter()
    previous = None
    for index, payload in enumerate(payloads):
        flat = np.frombuffer(decompressor.decompress(payload), dtype=np.uint8)
        block = flat.reshape(frames.shape[1:])
        current = block if index % keyframe == 0 else (previous + block)
        previous = current
    decode = (time.perf_counter() - started) / len(frames)

    return {
        "name": f"zstd-{level}-delta-k{keyframe}",
        "encode_ms": encode * 1000,
        "decode_ms": decode * 1000,
        "bytes": sum(len(p) for p in payloads) / len(payloads),
        "lossless": True,
    }


def _stream_reference(frames: np.ndarray) -> dict | None:
    """Best case: one zstd stream over the whole run, no per-frame seeking.

    Not usable on its own -- training needs to reach a frame without decoding
    every frame before it -- but it bounds what any blocked scheme can reach.
    """
    try:
        import zstandard
    except ImportError:
        return None

    buffer = io.BytesIO()
    started = time.perf_counter()
    with zstandard.ZstdCompressor(level=3).stream_writer(
        buffer, closefd=False
    ) as writer:
        for frame in frames:
            writer.write(frame.tobytes())
    encode = (time.perf_counter() - started) / len(frames)
    size = buffer.tell()
    return {
        "name": "zstd-3-whole-run (no seek)",
        "encode_ms": encode * 1000,
        "decode_ms": float("nan"),
        "bytes": size / len(frames),
        "lossless": True,
    }


def measure(frames: np.ndarray, name: str, encode, decode) -> dict:
    started = time.perf_counter()
    payloads = [encode(frame) for frame in frames]
    encode_seconds = time.perf_counter() - started

    started = time.perf_counter()
    restored = [decode(payload) for payload in payloads]
    decode_seconds = time.perf_counter() - started

    lossless = all(
        np.array_equal(a, b.reshape(a.shape)) for a, b in zip(frames, restored)
    )
    return {
        "name": name,
        "encode_ms": encode_seconds / len(frames) * 1000,
        "decode_ms": decode_seconds / len(frames) * 1000,
        "bytes": sum(len(p) for p in payloads) / len(frames),
        "lossless": lossless,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=Path("out/codec-bench/frames.npy"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--recapture", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out/codec-bench/codecs.json"))
    args = parser.parse_args()

    if args.recapture or not args.frames.is_file():
        frames = capture(
            args.frames, width=args.width, height=args.height, samples=args.samples
        )
    else:
        frames = np.load(args.frames)
    print(f"{len(frames)} frames of {frames.shape[2]}x{frames.shape[1]} RGB")
    raw = frames.shape[1] * frames.shape[2] * 3

    rows = [
        measure(frames, name, encode, decode)
        for name, (encode, decode) in candidates(frames.shape[1:]).items()
    ]
    for level in (1, 3, 6):
        for keyframe in (20, 100000):
            entry = _delta_sizes(frames, level, keyframe)
            if entry is not None:
                rows.append(entry)

    entry = _stream_reference(frames)
    if entry is not None:
        rows.append(entry)

    rows.sort(key=lambda row: row["bytes"])
    header = (
        f"{'codec':28} {'encode':>9} {'decode':>9} {'size':>10} "
        f"{'vs raw':>8} {'MB/hour':>9}  lossless"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        # One hour of gameplay is 72,000 frames at 20 Hz.
        print(
            f"{row['name']:28} {row['encode_ms']:7.2f}ms {row['decode_ms']:7.2f}ms "
            f"{row['bytes'] / 1024:8.1f}KB {raw / row['bytes']:7.1f}x "
            f"{row['bytes'] * 72000 / 1e6:8.0f} "
            f"  {'yes' if row['lossless'] else 'NO'}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
