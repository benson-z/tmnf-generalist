"""Frames: decoding the H.265 source, and the one preprocessing function.

``to_input`` is the single definition of what a frame looks like to the model
(horizon crop, then exact integer box downsampling, uint8). Training (video or
memmap source), the cache builder and the eval harness all call it, so the
320x240 and 160x120 inputs are derived from the same source the same way
everywhere.

Decoding: one ffmpeg process per run, whole run at once, streamed out as raw
RGB. With ``decoder: auto`` it uses NVDEC (``-hwaccel cuda``) if ffmpeg can,
else the CPU. The source files already meet the storage spec (H.265, one file
per run, closed GOP every 40 frames, no B-frames), so they are read as is.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import numpy as np

SRC_H, SRC_W = 240, 320


def to_input(frames: np.ndarray, downsample: int, crop_rows: list[int] | None) -> np.ndarray:
    """(..., 240, 320, 3) uint8 -> (..., H, W, 3) uint8 model input."""
    if crop_rows:
        frames = frames[..., crop_rows[0]:crop_rows[1], :, :]
    f = downsample
    if f == 1:
        return np.ascontiguousarray(frames)
    if f == 2:
        # Same integers as the general path, ~4x faster: four strided views
        # summed in 16 bit instead of a reshape-and-reduce.
        s = frames[..., 0::2, 0::2, :].astype(np.uint16)
        s += frames[..., 1::2, 0::2, :]
        s += frames[..., 0::2, 1::2, :]
        s += frames[..., 1::2, 1::2, :]
        s += 2
        s >>= 2
        return s.astype(np.uint8)
    *lead, h, w, c = frames.shape
    x = frames.reshape(*lead, h // f, f, w // f, f, c).astype(np.uint16)
    s = x.sum(axis=(-4, -2))
    # Exact box average with round-half-up, in integers.
    return ((s + (f * f) // 2) // (f * f)).astype(np.uint8)


@functools.lru_cache(maxsize=1)
def nvdec_available() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-f", "lavfi", "-i",
             "testsrc=size=320x240:rate=20:duration=0.2", "-c:v", "libx265", "-f", "matroska", "-"],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            return False
        r2 = subprocess.run(
            ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", "-", "-f", "null", "-"],
            input=r.stdout, capture_output=True, timeout=30,
        )
        return r2.returncode == 0 and b"rror" not in r2.stderr
    except (OSError, subprocess.SubprocessError):
        return False


def decode_run_input(path: Path, downsample: int, crop_rows: list[int] | None, decoder: str = "auto",
                     expected: int | None = None, chunk: int = 64) -> np.ndarray:
    """All frames of one run already at the training resolution.

    Streams ffmpeg's output and applies ``to_input`` per chunk of frames, so a
    full-resolution copy of the run (and ``to_input``'s 16-bit intermediate,
    together ~0.6 GB per run) never exists at once; that is what ran twelve
    loader workers out of RAM. Output is identical to
    ``to_input(decode_run(...))``.
    """
    hw = decoder == "nvdec" or (decoder == "auto" and nvdec_available())
    cmd = ["ffmpeg", "-v", "error", *(["-hwaccel", "cuda"] if hw else []), "-threads", "2",
           "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = SRC_H * SRC_W * 3
    parts = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes * chunk)
            if not buf:
                break
            n = len(buf) // frame_bytes
            if n * frame_bytes != len(buf):
                raise RuntimeError(f"{path}: partial frame in decoder output")
            parts.append(to_input(np.frombuffer(buf, np.uint8).reshape(n, SRC_H, SRC_W, 3), downsample, crop_rows))
    finally:
        proc.stdout.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{path}: ffmpeg exited with {code}")
    frames = np.concatenate(parts) if parts else np.zeros((0, 1, 1, 3), np.uint8)
    if expected is not None and len(frames) != expected:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames, expected {expected}")
    return frames


def decode_run(path: Path, decoder: str = "auto", expected: int | None = None) -> np.ndarray:
    """All frames of one run as (N, 240, 320, 3) uint8 RGB."""
    hw = decoder == "nvdec" or (decoder == "auto" and nvdec_available())
    cmd = ["ffmpeg", "-v", "error", *(["-hwaccel", "cuda"] if hw else []), "-threads", "2",
           "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(out, dtype=np.uint8)
    n = frames.size // (SRC_H * SRC_W * 3)
    if frames.size != n * SRC_H * SRC_W * 3 or (expected is not None and n != expected):
        raise RuntimeError(f"{path}: decoded {n} frames, expected {expected}")
    return frames.reshape(n, SRC_H, SRC_W, 3)
