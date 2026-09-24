"""End-to-end check of the game <-> controller bridge.

Launches one instance, gets it into a normal race on a stock campaign map,
drives it with a fixed input script and records 20 Hz samples to disk.  Proves
the sampling path before the replay queue is wired up.
"""

from __future__ import annotations

from pathlib import Path

from ..common.paths import detect
from . import install, staging
from .protocol import Sample
from .session import Session

# A fixed script, so the same run is reproducible across invocations.
SMOKE_SCRIPT = """0-8000 press up
900-1600 steer 45000
2200-2900 steer -45000
"""


def _save_frame(sample: Sample, path: Path) -> None:
    from PIL import Image

    image = Image.frombytes(
        "RGBA", (sample.width, sample.height), sample.pixels, "raw", "BGRA"
    )
    image.convert("RGB").save(path)


def run(
    *,
    out_dir: Path,
    port: int = 8477,
    width: int = 320,
    height: int = 240,
    period_ms: int = 50,
    max_samples: int = 120,
    keep_open: bool = False,
) -> dict:
    """Collect a short run and return a summary. Frames land in ``out_dir``."""
    layout = detect()
    install.install(layout)
    _, bootstrap_track = staging.stage_bootstrap(layout)
    script = staging.write_script(
        "tmnf_collect_smoke.txt", SMOKE_SCRIPT, layout
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    session = Session(
        port=port,
        layout=layout,
        width=width,
        height=height,
        period_ms=period_ms,
        # --keep-open is for inspecting the live game after the probe.
        offscreen=not keep_open,
    )
    session.start()
    try:
        session.prepare()
        result = session.record_map_run(
            bootstrap_track, script, max_samples=max_samples, timeout=180.0
        )
    finally:
        if keep_open:
            if session.controller is not None:
                session.controller.close()
        else:
            session.close()

    samples = result.samples
    for sample in samples[:: max(1, len(samples) // 8)][:8]:
        _save_frame(sample, out_dir / f"frame_{sample.race_time:06d}.png")

    times = [s.race_time for s in samples]
    gaps = sorted({b - a for a, b in zip(times, times[1:])})

    # How far the game had simulated past a sample's tick by the time it drew
    # the frame for it. Zero means the image and the telemetry are the same
    # instant; large values mean the game is outrunning its own rendering.
    lags = [s.render_race_time - s.race_time for s in samples]

    return {
        "samples": len(samples),
        "finished": result.finished,
        "finish_time": result.finish_time,
        "race_time_range": (times[0], times[-1]) if times else None,
        "race_time_gaps": gaps,
        "dropped_sample_points": samples[-1].dropped if samples else 0,
        "frame_lag_ms": {
            "max": max(lags) if lags else None,
            "mean": (sum(lags) / len(lags)) if lags else None,
            "exact": sum(1 for lag in lags if lag == 0),
        },
        "frame_size": (samples[0].width, samples[0].height) if samples else None,
        "frame_bytes": len(samples[0].pixels) if samples else 0,
        "speeds": [s.display_speed for s in samples[:20]],
        "out_dir": str(out_dir),
    }
