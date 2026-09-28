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

`hevc` gives that up for size: the whole run goes through one ffmpeg process as
H.265 at CRF 18 into `frames.mkv`, one video frame per row, in row order.
Measured in the container on a real 692-frame run, eight encoders at once and
no games running:

    x265 crf 18         8 encoders   size     RGB PSNR
    preset medium        268/s       7.1KB    33.9 dB
    preset veryfast      404/s       7.0KB    33.6 dB
    veryfast, 4:4:4      352/s       7.1KB    35.5 dB

Eight lanes at 2x need 320 frames a second, which rules out `medium`. The
encoder is its own process, so it never holds the collector's GIL; if it falls
behind, raw frames wait in the writer's unbounded queue rather than stalling
the socket reader. 4:2:0 because it is what every hardware and library decoder
reads; most of the PSNR gap to 4:4:4 is chroma at 160x120.

`hevc-vaapi` is the same stream from the GPU's video engine, for Linux with
Mesa's VAAPI driver. It has no CRF, so it runs at the constant QP that matches
CRF 18. Measured on the Cezanne box by re-encoding the 16680 lossless frames
of one 20-replay collection, and by collecting those 20 replays live with each
codec at 8 lanes and 2x:

    codec          CPU/frame  size    RGB PSNR  collection  CPU idle
    qoi-zstd           -      88KB    lossless    138.8 s     34%
    hevc (x265)     14.4ms    7.6KB   33.65 dB    152.8 s     25%
    hevc-vaapi QP24  0.95ms   7.2KB   33.49 dB    137.0 s     35%

x265 takes the CPU the games need there; the GPU encoder costs nothing
measurable, though the same GPU renders the games.

A closed GOP every two seconds (40 frames at 20 Hz), with scene-cut detection
off so keyframes sit at fixed multiples, makes any frame reachable by decoding
at most 39 before it. x265's default open GOPs would leave those keyframes
depending on the frames before them. Measured cost: 7.26KB a frame against
7.16KB with x265's own keyframe placement, which put three in the whole run.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import qoi
import zstandard
from PIL import Image

# What `collect` writes. Every run records the codec it was written with, so a
# corpus recorded before this still reads back.
LOSSLESS = "qoi-zstd"
HEVC = "hevc"
HEVC_VAAPI = "hevc-vaapi"
# Both write frames.mkv and read back the same way.
VIDEO_CODECS = (HEVC, HEVC_VAAPI)
CODECS = (LOSSLESS, *VIDEO_CODECS)
_ZSTD_LEVEL = 6
_HEVC_CRF = 18
_HEVC_PRESET = "veryfast"
# VAAPI has no CRF; constant QP 24 is what matches CRF 18's size and quality.
_VAAPI_QP = 24
_VAAPI_DEVICE = "/dev/dri/renderD128"
_KEYFRAME_SECONDS = 2


def keyframe_interval(fps: float) -> int:
    """Frames between `hevc` keyframes; frame i's keyframe is i // this * this."""
    return max(1, round(fps * _KEYFRAME_SECONDS))


def find_ffmpeg(tool: str = "ffmpeg") -> str:
    found = shutil.which(tool)
    if not found:
        raise RuntimeError(
            f"{tool} not found on PATH; install it or add it to PATH"
        )
    return found


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


class VideoEncoder:
    """Streams one run's frames through ffmpeg into a single H.265 file.

    ffmpeg starts on the first frame, since that is when the size is known.
    Not thread safe; one writer thread owns one of these, like `Encoder`.
    """

    def __init__(self, path: Path, *, fps: float, codec: str = HEVC) -> None:
        if codec not in VIDEO_CODECS:
            raise ValueError(f"not a video codec: {codec!r}")
        self.path = path
        self.fps = fps
        self.codec = codec
        self._process: subprocess.Popen | None = None
        self._log = None

    def add(self, pixels: bytes, width: int, height: int) -> None:
        if self._process is None:
            self._start(width, height)
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(pixels)

    def _start(self, width: int, height: int) -> None:
        # Errors only, but kept off the terminal: the collector's live display
        # owns it, and a pipe nobody reads until exit could fill and deadlock.
        self._log = tempfile.TemporaryFile()
        keyint = keyframe_interval(self.fps)
        if self.codec == HEVC_VAAPI:
            device = ["-vaapi_device", _VAAPI_DEVICE]
            encode = [
                "-vf", "format=nv12,hwupload",
                "-c:v", "hevc_vaapi",
                "-qp", str(_VAAPI_QP),
                "-g", str(keyint),
                "-bf", "0",
            ]
        else:
            device = []
            encode = [
                "-c:v", "libx265",
                "-preset", _HEVC_PRESET,
                "-crf", str(_HEVC_CRF),
                "-x265-params",
                "log-level=error:scenecut=0:open-gop=0"
                f":keyint={keyint}:min-keyint={keyint}",
                "-pix_fmt", "yuv420p",
            ]
        self._process = subprocess.Popen(
            [
                find_ffmpeg(),
                "-y",
                "-loglevel", "error",
                *device,
                "-f", "rawvideo",
                # The plugin's buffer as is; ffmpeg does the conversion.
                "-pixel_format", "bgra",
                "-video_size", f"{width}x{height}",
                "-framerate", f"{self.fps:g}",
                "-i", "-",
                "-an",
                *encode,
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )

    def _finish(self) -> None:
        process, log = self._process, self._log
        self._process = self._log = None
        if process is None:
            return
        try:
            assert process.stdin is not None
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            code = process.wait()
            if code != 0:
                assert log is not None
                log.seek(0)
                message = log.read().decode(errors="replace").strip()
                raise RuntimeError(f"ffmpeg exited with {code}: {message}")
        finally:
            if log is not None:
                log.close()

    def discard(self) -> None:
        """Throw away everything encoded so far; the next frame starts over."""
        try:
            self._finish()
        finally:
            self.path.unlink(missing_ok=True)

    def close(self) -> None:
        self._finish()


def count_video_frames(path: Path) -> int:
    """Frames in an encoded run, counted from the container without decoding."""
    out = subprocess.run(
        [
            find_ffmpeg("ffprobe"),
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def read_video(
    path: Path,
    width: int,
    height: int,
    *,
    start: int = 0,
    fps: float = 20.0,
) -> Iterator[Image.Image]:
    """The frames of an encoded run in row order, from row ``start`` on.

    Seeking lands on the keyframe before ``start`` and decodes forward, then
    drops everything timed before it. Half a frame early, so the rounding of
    container timestamps to milliseconds cannot drop ``start`` itself.
    """
    seek = ["-ss", f"{(start - 0.5) / fps:.6f}"] if start > 0 else []
    process = subprocess.Popen(
        [
            find_ffmpeg(),
            "-loglevel", "error",
            *seek,
            "-i", str(path),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    size = width * height * 3
    try:
        while chunk := process.stdout.read(size):
            if len(chunk) != size:
                raise RuntimeError(f"{path} ends in a partial frame")
            yield Image.frombytes("RGB", (width, height), chunk)
    finally:
        # Killed first: a reader that stops early would otherwise leave ffmpeg
        # writing into a closed pipe and complaining about it.
        process.kill()
        process.wait()
        process.stdout.close()
