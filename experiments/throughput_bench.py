"""End-to-end throughput benchmark for the frame-barrier branch.

`speed_instance_matrix.py` answers whether a setting reproduces a replay. This
answers the other half: what a setting is worth. It runs the *production*
collect command over the same fixed set of replays under several
speed/instance settings, times each one, then verifies what landed on disk and
compares the camera track against the natural-speed condition.

Throughput is reported as recorded gameplay seconds per wall second, which is
the only figure that translates into "hours of training data per hour of
machine". Correctness is reported beside it because a fast setting that loses
runs is slower, not faster.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from tmnf_collect.verify import check_dataset

REPO = Path(__file__).resolve().parent.parent
# `tmnf_collect.cli` has no __main__ guard, so `-m` imports it and exits.
CLI = [sys.executable, "-c", "from tmnf_collect.cli import main; raise SystemExit(main())"]


def _run_collect(
    replays: Path, out: Path, *, speed: float, instances: int, extra: list[str]
) -> tuple[dict, float]:
    command = [
        *CLI, "collect", str(replays),
        "--out", str(out),
        "--speed", str(speed),
        "--instances", str(instances),
        "--no-resume",
        *extra,
    ]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    wall = time.monotonic() - started
    if completed.returncode != 0:
        raise SystemExit(
            f"collect failed ({completed.returncode}) for speed {speed} "
            f"instances {instances}:\n{completed.stdout[-4000:]}\n"
            f"{completed.stderr[-4000:]}"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replays", help="folder of replays to record in every condition"
    )
    parser.add_argument("--out", default="out/throughput-bench")
    parser.add_argument(
        "--condition", action="append", default=None,
        help="speed:instances, repeatable; the first one is the camera reference",
    )
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args(argv)

    conditions = args.condition or ["1:6", "2:8", "1:8"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    extra = ["--width", str(args.width), "--height", str(args.height)]

    measured: dict[str, dict] = {}
    reference_dir: Path | None = None
    for condition in conditions:
        speed_text, _, instance_text = condition.partition(":")
        speed, instances = float(speed_text), int(instance_text)
        name = f"{speed:g}x-{instances}i"
        out = out_root / name
        print(f"=== {name} ===", flush=True)
        index, wall = _run_collect(
            Path(args.replays), out, speed=speed, instances=instances, extra=extra
        )
        entry = _measure(index, wall, out)
        entry["speed"] = speed
        entry["instances"] = instances
        if reference_dir is None:
            reference_dir = out
            entry["camera_vs_reference"] = "reference"
        else:
            entry["camera_vs_reference"] = _compare_cameras(out, reference_dir)
        measured[name] = entry
        print(json.dumps(entry, indent=2), flush=True)
        subprocess.run(
            [*CLI, "kill"],
            cwd=REPO, capture_output=True, text=True,
        )

    summary = {
        "replays": str(args.replays),
        "reference": conditions[0],
        "conditions": measured,
    }
    (out_root / "benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    header = (
        f"{'condition':12} {'wall':>7} {'ok':>8} {'gameplay':>9} "
        f"{'x-real':>7} {'x-busy':>7} {'verify':>8} {'on-tick':>9} "
        f"{'cam vs ref':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, entry in measured.items():
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
        print(
            f"{name:12} {entry['wall_seconds']:6.0f}s "
            f"{entry['ok']:3d}/{entry['replays']:<4d} "
            f"{entry['gameplay_seconds']:8.0f}s {entry['throughput_x']:6.2f}x "
            f"{entry['busy_throughput_x']:6.2f}x "
            f"{entry['verify_passed']:3d}/{checked:<4d} "
            f"{on_tick:>9} {camera_text:>11}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
