"""Does a speed/instance setting still reproduce a replay exactly?

Every instance records the same replay at each speed in turn, all instances
starting a round together so they contend for the machine the way a real
collection does. A setting passes when every run finishes on the replay's own
time; against a reference run it must also match its input stream and car
positions exactly.

Failures at the edge are scheduling-sensitive: a setting that passes one round
can miss by a single 10 ms physics step in the next, so a setting is only
trusted after several rounds. This is how the 2x-with-eight-instances limit in
the README was established.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..collect import install
from ..collect import runner
from ..collect.session import Session
from ..common.paths import detect


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class _TimedSession(Session):
    """A session that also notes when the first and last sample arrived."""

    capture_window: tuple[float, float] | None = None

    def record_map_run(self, *args, **kwargs):
        arrivals: list[float] = []
        callback = kwargs["on_sample"]
        reset = kwargs.get("on_reset")

        def receive(sample):
            arrivals.append(time.monotonic())
            callback(sample)

        def restart():
            arrivals.clear()
            if reset is not None:
                reset()

        kwargs.update(on_sample=receive, on_reset=restart)
        result = super().record_map_run(*args, **kwargs)
        self.capture_window = (
            (arrivals[0], arrivals[-1]) if len(arrivals) > 1 else None
        )
        return result


def _compare(root: Path, reference: Path) -> dict:
    samples = _rows(root / "samples.jsonl")
    ticks = _rows(root / "inputs.jsonl")
    reference_samples = {
        row["race_time"]: row for row in _rows(reference / "samples.jsonl")
    }
    reference_ticks = _rows(reference / "inputs.jsonl")
    pairs = [
        (reference_samples[row["race_time"]], row)
        for row in samples
        if row["race_time"] in reference_samples
    ]
    comparison = {
        "identical_input_stream": ticks == reference_ticks,
        "exact_frames": sum(
            row["race_time"] == row["render_race_time"] for row in samples
        ),
        "matched_frames": len(pairs),
    }
    for key in ("position", "camera_position"):
        distances = [
            np.linalg.norm(np.array(expected[key]) - actual[key])
            for expected, actual in pairs
        ]
        if distances:
            comparison[key] = {
                "mean": float(np.mean(distances)),
                "max": float(np.max(distances)),
            }
    return comparison


def run(
    replay: Path,
    out: Path,
    *,
    instances: int,
    speeds: list[float],
    port: int = 8520,
    instance_base: int = 30,
    stagger: float = 8.0,
    reference: Path | None = None,
    width: int = 320,
    height: int = 240,
    camera: int = 1,
    offscreen: bool = True,
) -> list[dict]:
    """Record ``replay`` on every instance at every speed; one summary per speed.

    The full report, every run included, is written to
    ``out/matrix-<instances>i.json``.
    """
    if instances < 1:
        raise ValueError("instances must be positive")
    for speed in speeds:
        if not 1 <= speed <= 5:
            raise ValueError(f"speed {speed:g} is outside the supported 1-5")

    layout = detect()
    install.install(layout)
    prepared = runner.plan([replay], layout=layout, strip_intros=True)
    if prepared.skipped or len(prepared.jobs) != 1:
        raise RuntimeError(prepared.skipped or "replay did not produce one job")
    original = prepared.jobs[0]
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"matrix-{instances}i.json"

    round_barrier = threading.Barrier(instances)
    records: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        job = replace(
            original,
            script_name=f"matrix_{instances}_{index}.txt",
            output_name=f"instance-{index}",
        )
        session = _TimedSession(
            port=port + index,
            instance_id=instance_base + index,
            layout=layout,
            width=width,
            height=height,
            camera=camera,
            offscreen=offscreen,
        )
        try:
            session.start()
            session.prepare()
            for speed in speeds:
                session.set_speed(speed)
                round_barrier.wait(timeout=180)
                started = time.monotonic()
                result = runner.run_job(
                    session, job, out / f"{instances}i-{speed:g}x", timeout=300
                )
                ended = time.monotonic()
                row = {
                    **asdict(result),
                    "instances": instances,
                    "speed": speed,
                    "worker": index,
                    "started": started,
                    "ended": ended,
                    "capture_window": session.capture_window,
                }
                with lock:
                    records.append(row)
                    print(json.dumps(row), flush=True)
                round_barrier.wait(timeout=180)
        except Exception as exc:
            with lock:
                errors.append(f"instance {index}: {type(exc).__name__}: {exc}")
            round_barrier.abort()
        finally:
            session.close()

    threads = []
    for index in range(instances):
        thread = threading.Thread(target=worker, args=(index,))
        thread.start()
        threads.append(thread)
        if index + 1 < instances:
            time.sleep(stagger)
    for thread in threads:
        thread.join()

    if errors:
        report_path.write_text(
            json.dumps({"runs": records, "errors": errors}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError("; ".join(errors))

    if reference is None and instances == 1 and 1.0 in speeds:
        reference = out / "1i-1x" / "instance-0"
    summaries = []
    for speed in speeds:
        group = [row for row in records if row["speed"] == speed]
        capture_windows = [
            row["capture_window"] for row in group if row["capture_window"]
        ]
        summary = {
            "instances": instances,
            "speed": speed,
            "passed": sum(row["status"] == "ok" for row in group),
            "runs": len(group),
            "finish_times": [row["finish_time"] for row in group],
            "wall_seconds": max(row["ended"] for row in group)
            - min(row["started"] for row in group),
        }
        if capture_windows:
            capture_seconds = max(w[1] for w in capture_windows) - min(
                w[0] for w in capture_windows
            )
            summary["aggregate_capture_speed"] = (
                instances * original.replay.race_time / 1000 / capture_seconds
            )
        if reference is not None:
            summary["comparisons"] = [
                _compare(
                    out / f"{instances}i-{speed:g}x" / row["output_name"],
                    reference,
                )
                for row in group
            ]
        summaries.append(summary)
    report_path.write_text(
        json.dumps({"runs": records, "summaries": summaries}, indent=2),
        encoding="utf-8",
    )
    return summaries
