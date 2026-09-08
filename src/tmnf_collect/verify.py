"""Checking a recorded dataset is actually usable.

A run can look fine in the collector's summary and still be unusable on disk:
frames missing, rows out of order, a gap where the game skipped a sample. This
re-reads what was written and checks it against what the replay promised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunCheck:
    name: str
    problems: list[str] = field(default_factory=list)
    rows: int = 0
    frames: int = 0
    expected_ms: int | None = None
    recorded_ms: int | None = None
    duration_ms: int = 0
    dropped: int = 0
    max_frame_lag_ms: int = 0
    exact_frames: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def check_run(directory: Path, *, period_ms: int = 50) -> RunCheck:
    result = RunCheck(name=directory.name)

    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result.problems.append(f"meta.json unreadable: {exc}")
        return result

    result.expected_ms = meta.get("replay_race_time_ms")
    result.recorded_ms = meta.get("recorded_finish_time_ms")
    result.dropped = meta.get("dropped_sample_points", 0)

    if meta.get("status") != "ok":
        result.problems.append(f"status is {meta.get('status')!r}")
    if result.expected_ms != result.recorded_ms:
        result.problems.append(
            f"finish time {result.recorded_ms} != replay {result.expected_ms}"
        )
    if result.dropped:
        result.problems.append(f"{result.dropped} sample points were dropped")

    try:
        lines = (directory / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as exc:
        result.problems.append(f"samples.jsonl unreadable: {exc}")
        return result

    rows = [json.loads(line) for line in lines if line.strip()]
    result.rows = len(rows)
    if not rows:
        result.problems.append("no samples")
        return result

    times = [r["race_time"] for r in rows]
    result.duration_ms = times[-1] - times[0]

    if times[0] != 0:
        result.problems.append(f"first sample is at {times[0]} ms, not 0")
    if any(r["i"] != i for i, r in enumerate(rows)):
        result.problems.append("row indices are not contiguous")

    gaps = {b - a for a, b in zip(times, times[1:])}
    if gaps - {period_ms}:
        result.problems.append(
            f"sample gaps other than {period_ms} ms: {sorted(gaps)}"
        )

    # The run has to cover the replay, right up to the finish. Only worth
    # saying for a run that claims to have reproduced it.
    if result.expected_ms is not None and result.expected_ms == result.recorded_ms:
        missing = result.expected_ms - times[-1]
        if not 0 <= missing <= period_ms:
            result.problems.append(
                f"last sample at {times[-1]} ms leaves {missing} ms "
                f"before the {result.expected_ms} ms finish"
            )

    lags = [r["render_race_time"] - r["race_time"] for r in rows]
    result.max_frame_lag_ms = max(lags)
    result.exact_frames = sum(1 for lag in lags if lag == 0)
    if min(lags) < 0:
        result.problems.append("a frame was drawn before its own tick")

    frame_files = {p.name for p in (directory / "frames").glob("*") if p.is_file()}
    result.frames = len(frame_files)
    missing_frames = [
        r["frame"] for r in rows if Path(r["frame"]).name not in frame_files
    ]
    if missing_frames:
        result.problems.append(
            f"{len(missing_frames)} rows point at missing frames "
            f"(e.g. {missing_frames[0]})"
        )
    if result.frames != result.rows:
        result.problems.append(
            f"{result.frames} frames for {result.rows} rows"
        )

    return result


def check_dataset(root: Path, *, period_ms: int = 50) -> dict:
    runs = [
        check_run(d, period_ms=period_ms)
        for d in sorted(p for p in root.iterdir() if p.is_dir())
    ]
    failed = [r for r in runs if not r.ok]
    return {
        "root": str(root),
        "runs": len(runs),
        "passed": len(runs) - len(failed),
        "failed": len(failed),
        "total_rows": sum(r.rows for r in runs),
        "total_frames": sum(r.frames for r in runs),
        "recorded_seconds": round(sum(r.duration_ms for r in runs) / 1000, 1),
        "frames_on_their_own_tick": sum(r.exact_frames for r in runs),
        "max_frame_lag_ms": max((r.max_frame_lag_ms for r in runs), default=0),
        "checks": runs,
    }
