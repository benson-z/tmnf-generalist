"""What a collection setting is worth, end to end.

`matrix` answers whether a setting reproduces a replay. This answers the other
half: it runs the *production* `collect` command over the same fixed set of
replays under several speed/lane settings, times each one, then verifies what
landed on disk and compares the camera track against the first condition.

Each condition runs as its own `tmnf-collect collect` subprocess, so it picks
up the same config file a real collection would; only speed and lane count
(and any extra flags given) are overridden.

Throughput is reported as recorded gameplay seconds per wall second, which is
the only figure that translates into "hours of training data per hour of
machine". Correctness is reported beside it because a fast setting that loses
runs is slower, not faster.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

from ..collect import launcher
from ..tools.verify import check_dataset

CLI = [sys.executable, "-m", "tmnf_collect"]


def parse_condition(text: str) -> tuple[float, int]:
    """``"2:8"`` -> speed 2, 8 lanes."""
    speed_text, sep, lanes_text = text.partition(":")
    if not sep:
        raise ValueError(f"condition {text!r} should be speed:lanes, e.g. 2:8")
    speed, lanes = float(speed_text), int(lanes_text)
    if lanes < 1:
        raise ValueError(f"condition {text!r} needs at least one lane")
    return speed, lanes


def _run_collect(
    replays: Path,
    out: Path,
    *,
    speed: float,
    lanes: int,
    global_args: list[str],
    collect_args: list[str],
) -> tuple[dict, float]:
    command = [
        *CLI, *global_args, "collect", str(replays),
        "--out", str(out),
        "--speed", f"{speed:g}",
        # Both, so the condition wins whichever lane mode the config selects.
        "--processes", str(lanes),
        "--instances", str(lanes),
        "--no-resume",
        *collect_args,
    ]
    # collect exits 1 when any run errored, which is a result to measure, not a
    # reason to stop; only a missing index means the condition never ran.
    (out / "index.json").unlink(missing_ok=True)
    started = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True)
    wall = time.monotonic() - started
    if completed.returncode != 0 and not (out / "index.json").is_file():
        raise RuntimeError(
            f"collect failed ({completed.returncode}) at {speed:g}x with "
            f"{lanes} lanes:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    return index, wall


def _camera_track(root: Path) -> dict[str, dict[int, tuple]]:
    """Camera pose per map per race time, for comparing two conditions."""
    track: dict[str, dict[int, tuple]] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        samples = directory / "samples.jsonl"
        if not samples.is_file():
            continue
        track[directory.name] = {
            row["race_time"]: (
                tuple(row["camera_position"]),
                tuple(row["camera_yaw_pitch_roll"]),
                tuple(row["position"]),
            )
            for row in (
                json.loads(line)
                for line in samples.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    return track


def _compare_cameras(root: Path, reference: Path) -> dict:
    here, there = _camera_track(root), _camera_track(reference)
    camera_gaps: list[float] = []
    car_gaps: list[float] = []
    maps = 0
    for name, rows in here.items():
        other = there.get(name)
        if not other:
            continue
        maps += 1
        for race_time, (camera, _rot, car) in rows.items():
            if race_time not in other:
                continue
            ref_camera, _ref_rot, ref_car = other[race_time]
            camera_gaps.append(math.dist(camera, ref_camera))
            car_gaps.append(math.dist(car, ref_car))
    if not camera_gaps:
        return {"maps": maps, "points": 0}
    return {
        "maps": maps,
        "points": len(camera_gaps),
        "camera_mean_m": round(sum(camera_gaps) / len(camera_gaps), 4),
        "camera_max_m": round(max(camera_gaps), 4),
        "car_mean_m": round(sum(car_gaps) / len(car_gaps), 6),
        "car_max_m": round(max(car_gaps), 6),
        "car_exact_points": sum(1 for gap in car_gaps if gap == 0),
    }


def _measure(index: dict, wall: float, out: Path) -> dict:
    results = index["results"]
    ok = [r for r in results if r["status"] == "ok"]
    gameplay_ms = sum(r["finish_time"] or 0 for r in ok)
    report = check_dataset(out)
    checks = report.pop("checks")
    # Launching eight games costs about the same 45s whatever the speed, and on
    # a corpus of a few hundred maps it is amortised away. The busy figure is
    # the same throughput measured over the longest instance's recording time
    # only, which is what a long collection converges on.
    busy: dict[int, float] = {}
    for result in results:
        busy[result["instance"]] = busy.get(result["instance"], 0.0) + result["seconds"]
    longest = max(busy.values(), default=0.0)
    return {
        "wall_seconds": round(wall, 1),
        "collect_seconds": index["seconds"],
        "replays": index["replays_found"],
        "ok": len(ok),
        "ok_first_try": index["ok_first_try"],
        "by_status": index["by_status"],
        "attempts": sum(r["attempts"] for r in results),
        "dropped_sample_points": sum(r["dropped"] for r in results),
        "gameplay_seconds": round(gameplay_ms / 1000, 1),
        # The headline: recorded gameplay per second of wall clock.
        "throughput_x": round(gameplay_ms / 1000 / wall, 2),
        "busy_seconds": round(longest, 1),
        "busy_throughput_x": round(gameplay_ms / 1000 / longest, 2) if longest else 0.0,
        "verify_passed": report["passed"],
        "verify_failed": report["failed"],
        "verify_problems": [
            {"run": c.name, "problems": c.problems} for c in checks if not c.ok
        ],
        "frames": report["total_frames"],
        "frames_on_their_own_tick": report["frames_on_their_own_tick"],
        "max_frame_lag_ms": report["max_frame_lag_ms"],
    }


def run(
    replays: Path,
    out_root: Path,
    conditions: list[tuple[float, int]],
    *,
    global_args: list[str] | None = None,
    collect_args: list[str] | None = None,
) -> dict:
    """Collect ``replays`` once per (speed, lanes) condition and measure each.

    The first condition is the camera reference for the others. The summary is
    also written to ``out_root/benchmark.json``.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    measured: dict[str, dict] = {}
    reference_dir: Path | None = None
    for speed, lanes in conditions:
        name = f"{speed:g}x-{lanes}i"
        out = out_root / name
        print(f"=== {name} ===", flush=True)
        try:
            index, wall = _run_collect(
                replays,
                out,
                speed=speed,
                lanes=lanes,
                global_args=global_args or [],
                collect_args=collect_args or [],
            )
        finally:
            # A failed condition must not leave games running into the next.
            launcher.kill_all()
        entry = _measure(index, wall, out)
        entry["speed"] = speed
        entry["instances"] = lanes
        if reference_dir is None:
            reference_dir = out
            entry["camera_vs_reference"] = "reference"
        else:
            entry["camera_vs_reference"] = _compare_cameras(out, reference_dir)
        measured[name] = entry
        print(json.dumps(entry, indent=2), flush=True)

    summary = {
        "replays": str(replays),
        "reference": next(iter(measured), None),
        "conditions": measured,
    }
    (out_root / "benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def format_table(summary: dict) -> str:
    header = (
        f"{'condition':12} {'wall':>7} {'ok':>8} {'gameplay':>9} "
        f"{'x-real':>7} {'x-busy':>7} {'verify':>8} {'on-tick':>9} "
        f"{'cam vs ref':>11}"
    )
    lines = [header, "-" * len(header)]
    for name, entry in summary["conditions"].items():
        camera = entry["camera_vs_reference"]
        camera_text = (
            "reference" if camera == "reference"
            else f"{camera.get('camera_mean_m', float('nan')):.3f} m"
        )
        on_tick = (
            f"{entry['frames_on_their_own_tick'] / entry['frames']:.1%}"
            if entry["frames"] else "-"
        )
        checked = entry["verify_passed"] + entry["verify_failed"]
        lines.append(
            f"{name:12} {entry['wall_seconds']:6.0f}s "
            f"{entry['ok']:3d}/{entry['replays']:<4d} "
            f"{entry['gameplay_seconds']:8.0f}s {entry['throughput_x']:6.2f}x "
            f"{entry['busy_throughput_x']:6.2f}x "
            f"{entry['verify_passed']:3d}/{checked:<4d} "
            f"{on_tick:>9} {camera_text:>11}"
        )
    return "\n".join(lines)
