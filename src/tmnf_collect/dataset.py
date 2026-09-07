"""Writing collected samples to disk.

Encoding happens on a worker thread. The socket read loop must never block: if
the controller stops reading, the plugin's writes back up and the game stalls,
which would distort the very timing we are trying to record.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .protocol import Sample


@dataclass
class RunPaths:
    root: Path
    frames: Path
    samples: Path
    meta: Path

    @classmethod
    def under(cls, root: Path) -> RunPaths:
        return cls(
            root=root,
            frames=root / "frames",
            samples=root / "samples.jsonl",
            meta=root / "meta.json",
        )


def sample_record(sample: Sample, index: int, frame_name: str) -> dict:
    """One row of ``samples.jsonl``."""
    return {
        "i": index,
        "race_time": sample.race_time,
        # The tick the game had reached when the frame was drawn; equal to
        # race_time means image and telemetry are the same instant.
        "render_race_time": sample.render_race_time,
        "frame": frame_name,
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
        self, root: Path, *, image_format: str = "jpeg", quality: int = 90
    ) -> None:
        self.paths = RunPaths.under(root)
        self.image_format = image_format
        self.quality = quality
        self.suffix = "jpg" if image_format == "jpeg" else image_format
        self.written = 0
        self.resets = 0

        self.paths.frames.mkdir(parents=True, exist_ok=True)
        self._records = self.paths.samples.open("w", encoding="utf-8")
        # ("add", index, sample) | ("reset",) | None to stop. Reset goes
        # through the queue rather than acting directly so it stays ordered
        # against the frames already handed over.
        self._queue: queue.Queue[tuple | None] = queue.Queue(maxsize=256)
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

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
                name = f"{index:06d}.{self.suffix}"
                image = Image.frombytes(
                    "RGBA",
                    (sample.width, sample.height),
                    sample.pixels,
                    "raw",
                    "BGRA",
                ).convert("RGB")
                if self.image_format == "jpeg":
                    image.save(
                        self.paths.frames / name,
                        "JPEG",
                        quality=self.quality,
                        subsampling=0,
                    )
                else:
                    image.save(self.paths.frames / name)
                self._records.write(
                    json.dumps(sample_record(sample, index, f"frames/{name}"))
                    + "\n"
                )
            except BaseException as exc:  # surfaced on close()
                self._error = self._error or exc

    def _discard_everything(self) -> None:
        self._records.seek(0)
        self._records.truncate()
        for frame in self.paths.frames.iterdir():
            frame.unlink()

    def add(self, sample: Sample) -> None:
        self._queue.put(("add", self.written, sample))
        self.written += 1

    def reset(self) -> None:
        """Throw away everything written so far and start the run again.

        The game sometimes runs a map once without applying the loaded inputs
        and then restarts itself to play them, so the frames before that
        restart are of a car sitting at the start line.
        """
        self._queue.put(("reset",))
        self.written = 0
        self.resets += 1

    def write_meta(self, meta: dict) -> None:
        self.paths.meta.write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join()
        self._records.close()
        if self._error is not None:
            raise self._error

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
