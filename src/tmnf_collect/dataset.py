"""Writing collected samples to disk.

Encoding happens on a worker thread. The socket read loop must never block: if
the controller stops reading, the plugin's writes back up and the game stalls,
which would distort the very timing we are trying to record.

Nothing the reader hands over may block it, frames included. A bounded queue
looks like sensible back pressure and is not: when the encoder falls behind --
which is what running several instances causes -- the queue fills, the put
blocks the reader, the plugin's writes back up and fail, and a single failed
write disconnects that instance for the rest of the run. Measured at nine
instances, that cost five of thirty-six runs. So both queues are unbounded and
the backlog is bounded by the encoder keeping up on average, which it does:
encoding is 0.4ms against a 50ms sample period.

Frames go into one file per run rather than one file each. Appending to an open
file measured 0.067ms a frame against 0.647ms for a new small file, and a
corpus of a few hundred runs is otherwise a couple of hundred thousand tiny
files to read back at training time. `samples.jsonl` carries the offset and
length of each frame in `frames.bin`.
"""

from __future__ import annotations

import io
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .protocol import Sample, Tick


@dataclass
class RunPaths:
    root: Path
    frames: Path
    samples: Path
    ticks: Path
    meta: Path

    @classmethod
    def under(cls, root: Path) -> RunPaths:
        return cls(
            root=root,
            frames=root / "frames.bin",
            samples=root / "samples.jsonl",
            ticks=root / "inputs.jsonl",
            meta=root / "meta.json",
        )


def tick_record(tick: Tick) -> dict:
    """One row of ``inputs.jsonl``: which keys were held for one 10ms step.

    Keys only. The resolved analog values and speed are in ``samples.jsonl``
    twenty times a second, and sending them again at a hundred cost four extra
    socket writes per tick -- which, since one failed write disconnects the
    plugin permanently, is what a whole run is risked on.
    """
    return {
        "race_time": tick.race_time,
        "up": tick.up,
        "down": tick.down,
        "left": tick.left,
        "right": tick.right,
    }


def sample_record(
    sample: Sample, index: int, frame_offset: int, frame_bytes: int
) -> dict:
    """One row of ``samples.jsonl``."""
    return {
        "i": index,
        "race_time": sample.race_time,
        # Where this frame lives in frames.bin.
        "frame_offset": frame_offset,
        "frame_bytes": frame_bytes,
        # The tick the game had reached when the frame was drawn; equal to
        # race_time means image and telemetry are the same instant.
        "render_race_time": sample.render_race_time,
        "speed_kmh": sample.display_speed,
        "up": sample.up,
        "down": sample.down,
        "left": sample.left,
        "right": sample.right,
        "gas": sample.gas,
        "steer": sample.steer,
        "gas_f": round(sample.gas_f, 5),
        "brake_f": round(sample.brake_f, 5),
        "steer_f": round(sample.steer_f, 5),
        "position": [round(v, 4) for v in sample.position],
        "velocity": [round(v, 4) for v in sample.velocity],
        "yaw_pitch_roll": [round(v, 5) for v in sample.yaw_pitch_roll],
        "local_speed": [round(v, 4) for v in sample.local_speed],
        "camera_position": [round(v, 4) for v in sample.camera_position],
        "camera_yaw_pitch_roll": [
            round(v, 5) for v in sample.camera_yaw_pitch_roll
        ],
        "camera_fov": round(sample.camera_fov, 5),
        "checkpoints": sample.checkpoints,
        "finished": sample.finished,
        "sliding": sample.sliding,
        "gearbox": sample.gearbox,
        "dropped_so_far": sample.dropped,
    }


class RunWriter:
    """Writes one run's frames and records; use as a context manager."""

    def __init__(
        self, root: Path, *, image_format: str = "jpeg", quality: int = 80
    ) -> None:
        self.paths = RunPaths.under(root)
        self.image_format = image_format
        self.quality = quality
        self.written = 0
        self.ticks = 0
        self.resets = 0
        self.frame_bytes = 0

        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._records = self.paths.samples.open("w", encoding="utf-8")
        self._ticks = self.paths.ticks.open("w", encoding="utf-8")
        self._frames = self.paths.frames.open("wb")

        # Unbounded on purpose; see the module docstring.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

        self._tick_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._tick_worker = threading.Thread(
            target=self._run_ticks, daemon=True
        )
        self._tick_worker.start()

    def _encode(self, sample: Sample) -> bytes:
        image = Image.frombytes(
            "RGBA",
            (sample.width, sample.height),
            sample.pixels,
            "raw",
            "BGRA",
        ).convert("RGB")
        buffer = io.BytesIO()
        if self.image_format == "jpeg":
            # 4:2:0 rather than 4:4:4: measured 0.50ms against 1.16ms per
            # frame and less than half the size, for chroma detail that does
            # not survive downscaling to a training resolution anyway.
            image.save(buffer, "JPEG", quality=self.quality, subsampling=2)
        else:
            image.save(buffer, self.image_format.upper())
        return buffer.getvalue()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if item[0] == "reset":
                    self._discard_everything()
                    continue
                _, index, sample = item
                payload = self._encode(sample)
                offset = self._frames.tell()
                self._frames.write(payload)
                self._records.write(
                    json.dumps(
                        sample_record(sample, index, offset, len(payload))
                    )
                    + "\n"
                )
            except BaseException as exc:  # surfaced on close()
                self._error = self._error or exc

    def _run_ticks(self) -> None:
        while True:
            item = self._tick_queue.get()
            if item is None:
                return
            try:
                if item[0] == "reset":
                    self._ticks.seek(0)
                    self._ticks.truncate()
                    continue
                self._ticks.write(json.dumps(tick_record(item[1])) + "\n")
            except BaseException as exc:  # surfaced on close()
                self._error = self._error or exc

    def _discard_everything(self) -> None:
        self._records.seek(0)
        self._records.truncate()
        self._frames.seek(0)
        self._frames.truncate()

    def add(self, sample: Sample) -> None:
        self._queue.put(("add", self.written, sample))
        self.written += 1

    def add_tick(self, tick: Tick) -> None:
        self._tick_queue.put(("tick", tick))
        self.ticks += 1

    def reset(self) -> None:
        """Throw away everything written so far and start the run again.

        The game sometimes runs a map once without applying the loaded inputs
        and then restarts itself to play them, so the frames before that
        restart are of a car sitting at the start line.
        """
        self._queue.put(("reset",))
        self._tick_queue.put(("reset",))
        self.written = 0
        self.ticks = 0
        self.resets += 1

    def write_meta(self, meta: dict) -> None:
        self.paths.meta.write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        self._queue.put(None)
        self._tick_queue.put(None)
        self._worker.join()
        self._tick_worker.join()
        self.frame_bytes = self._frames.tell()
        self._records.close()
        self._ticks.close()
        self._frames.close()
        if self._error is not None:
            raise self._error

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
