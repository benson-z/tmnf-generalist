"""Does Graphics::ForceGameRender() disturb the camera?

The sampler drives rendering itself: at each 20 Hz tick it calls
``Graphics::ForceGameRender()`` so it can capture inside ``Render()``.  If that
extra, out-of-band frame advanced the camera differently from a normal frame,
every recorded image would be shot from a slightly wrong viewpoint.

This runs the same deterministic input script twice on one instance -- once
with forced rendering, once letting the game's own frames drive the capture --
and compares the images that came out at identical race times.  Physics is
deterministic, so the car is in exactly the same place in both runs; any
structural difference in the images is the camera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import install, staging
from .paths import detect
from .protocol import Sample
from .session import Session
from .smoke import SMOKE_SCRIPT


def _to_array(sample: Sample) -> np.ndarray:
    pixels = np.frombuffer(sample.pixels, dtype=np.uint8)
    return pixels.reshape(sample.height, sample.width, 4)[:, :, :3].astype(
        np.int16
    )


def _save_pair(
    forced: Sample, natural: Sample, path: Path
) -> None:
    from PIL import Image

    def rgb(sample: Sample) -> Image.Image:
        return Image.frombytes(
            "RGBA", (sample.width, sample.height), sample.pixels, "raw", "BGRA"
        ).convert("RGB")

    left, right = rgb(forced), rgb(natural)
    canvas = Image.new("RGB", (left.width * 2, left.height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(path)


def run(
    *,
    out_dir: Path,
    port: int = 8477,
    width: int = 320,
    height: int = 240,
    samples: int = 100,
    speed_up: float = 5.0,
) -> dict:
    layout = detect()
    install.install(layout)
    _, track = staging.stage_bootstrap(layout)
    script = staging.write_script(
        "tmnf_collect_smoke.txt", SMOKE_SCRIPT, layout
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    session = Session(
        port=port, layout=layout, width=width, height=height, force_render=True
    )
    session.start()
    try:
        session.prepare(speed=1.0)

        forced = session.record_map_run(
            track, script, max_samples=samples, timeout=180.0
        )

        session.controller.configure(force_render=False)  # type: ignore[union-attr]
        natural = session.record_map_run(
            track, script, max_samples=samples, timeout=180.0
        )

        # The one that actually matters: forced rendering must give the same
        # picture no matter how fast the game is running, or speeding up
        # collection would quietly change the training data.
        session.controller.configure(force_render=True)  # type: ignore[union-attr]
        session.prepare(speed=speed_up)
        fast = session.record_map_run(
            track, script, max_samples=samples, timeout=180.0
        )
        session.prepare(speed=1.0)
    finally:
        session.close()

    baseline = {s.race_time: s for s in forced.samples}
    comparisons = {
        "forced_vs_natural": {s.race_time: s for s in natural.samples},
        f"forced_vs_forced_at_{speed_up:g}x": {
            s.race_time: s for s in fast.samples
        },
    }

    summary: dict = {
        "samples": {
            "forced_1x": len(forced.samples),
            "natural_1x": len(natural.samples),
            f"forced_{speed_up:g}x": len(fast.samples),
        },
        "out_dir": str(out_dir),
    }

    for label, other in comparisons.items():
        common = sorted(set(baseline) & set(other))
        if not common:
            summary[label] = {"error": "no race times in common"}
            continue

        image_diffs: list[float] = []
        position_diffs: list[float] = []
        camera_diffs: list[float] = []
        camera_angle_diffs: list[float] = []
        worst = (0.0, common[0])
        for race_time in common:
            a, b = baseline[race_time], other[race_time]
            diff = float(np.abs(_to_array(a) - _to_array(b)).mean())
            image_diffs.append(diff)
            position_diffs.append(
                max(abs(x - y) for x, y in zip(a.position, b.position))
            )
            camera_diffs.append(
                float(
                    np.linalg.norm(
                        np.array(a.camera_position) - np.array(b.camera_position)
                    )
                )
            )
            camera_angle_diffs.append(
                max(
                    abs(x - y)
                    for x, y in zip(
                        a.camera_yaw_pitch_roll, b.camera_yaw_pitch_roll
                    )
                )
            )
            if diff > worst[0]:
                worst = (diff, race_time)

        _save_pair(
            baseline[worst[1]], other[worst[1]],
            out_dir / f"{label}_worst_{worst[1]:06d}.png",
        )
        mid = common[len(common) // 2]
        _save_pair(
            baseline[mid], other[mid], out_dir / f"{label}_mid_{mid:06d}.png"
        )

        summary[label] = {
            "compared_race_times": len(common),
            # If physics did not reproduce, the image comparison means nothing.
            "max_car_position_delta_m": max(position_diffs),
            "mean_camera_position_delta_m": sum(camera_diffs) / len(camera_diffs),
            "max_camera_position_delta_m": max(camera_diffs),
            "max_camera_angle_delta_rad": max(camera_angle_diffs),
            "mean_abs_pixel_diff": sum(image_diffs) / len(image_diffs),
            "max_abs_pixel_diff": worst[0],
            "worst_race_time": worst[1],
        }

    return summary
