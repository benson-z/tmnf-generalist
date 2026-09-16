"""Benchmark full-replay correctness across instance-count/speed combinations."""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from tmnf_collect import collect, install
from tmnf_collect.paths import detect
from tmnf_collect.session import Session


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TimedSession(Session):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--speeds", default="1,2,3,4,5,8,10")
    parser.add_argument("--port", type=int, default=8520)
    parser.add_argument("--instance-base", type=int, default=30)
    parser.add_argument("--stagger", type=float, default=8.0)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    speeds = [float(value) for value in args.speeds.split(",")]
    if args.instances < 1:
        parser.error("--instances must be positive")

    layout = detect()
    install.install(layout)
    prepared = collect.plan([args.replay], layout=layout, strip_intros=True)
    if prepared.skipped or len(prepared.jobs) != 1:
        raise RuntimeError(prepared.skipped or "replay did not produce one job")
    original = prepared.jobs[0]
    args.out.mkdir(parents=True, exist_ok=True)

    round_barrier = threading.Barrier(args.instances)
    records: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        job = replace(
            original,
            script_name=f"matrix_{args.instances}_{index}.txt",
            output_name=f"instance-{index}",
        )
        session = TimedSession(
            port=args.port + index,
            instance_id=args.instance_base + index,
            layout=layout,
            width=160,
            height=120,
            speed=1,
        )
        try:
            session.start()
            session.prepare()
            for speed in speeds:
                session._command(f"set speed {speed}")
                session._ctrl.configure(frame_barrier=speed > 1)
                round_barrier.wait(timeout=180)
                started = time.monotonic()
                result = collect.run_job(
                    session,
                    job,
                    args.out / f"{args.instances}i-{speed:g}x",
                    timeout=300,
                )
                ended = time.monotonic()
                row = {
                    **asdict(result),
                    "instances": args.instances,
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
            session._command("set speed 1")
        except Exception as exc:
            with lock:
                errors.append(f"instance {index}: {type(exc).__name__}: {exc}")
            round_barrier.abort()
        finally:
            session.close()

    threads = []
    for index in range(args.instances):
        thread = threading.Thread(target=worker, args=(index,), daemon=False)
        thread.start()
        threads.append(thread)
        if index + 1 < args.instances:
            time.sleep(args.stagger)
    for thread in threads:
        thread.join()

    if errors:
        report = {"runs": records, "errors": errors}
        (args.out / f"matrix-{args.instances}i.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        raise RuntimeError("; ".join(errors))

    reference = args.reference
    if reference is None and args.instances == 1 and 1.0 in speeds:
        reference = args.out / "1i-1x" / "instance-0"
    summaries = []
    for speed in speeds:
        group = [row for row in records if row["speed"] == speed]
        capture_windows = [
            row["capture_window"] for row in group if row["capture_window"]
        ]
        summary = {
            "instances": args.instances,
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
                args.instances * original.replay.race_time / 1000 / capture_seconds
            )
        if reference is not None:
            comparisons = []
            for row in group:
                run_root = (
                    args.out
                    / f"{args.instances}i-{speed:g}x"
                    / row["output_name"]
                )
                comparisons.append(_compare(run_root, reference))
            summary["comparisons"] = comparisons
        summaries.append(summary)
    report = {"runs": records, "summaries": summaries}
    (args.out / f"matrix-{args.instances}i.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
