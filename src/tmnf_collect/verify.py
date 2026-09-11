"""Checking a recorded dataset is actually usable.

A run can look fine in the collector's summary and still be unusable on disk:
frames missing, rows out of order, a gap where the game skipped a sample. This
re-reads what was written and checks it against what the replay promised.

`clean_dataset` is the acting half of the same check: it takes the runs that
fail and moves them out of the dataset, so what is left is what can be trained
on.
"""

from __future__ import annotations

import json
import shutil
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
    ticks: int = 0

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

    # The 100Hz input stream is checked the same way as the samples: it has to
    # be continuous, start at 0 and reach the finish, and agree with the 20Hz
    # rows wherever the two land on the same instant.
    tick_path = directory / "inputs.jsonl"
    if tick_path.is_file():
        try:
            ticks = [
                json.loads(line)
                for line in tick_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as exc:
            result.problems.append(f"inputs.jsonl unreadable: {exc}")
            ticks = []
        result.ticks = len(ticks)
        if ticks:
            tick_times = [t["race_time"] for t in ticks]
            if tick_times[0] != 0:
                result.problems.append(
                    f"first input tick is at {tick_times[0]} ms, not 0"
                )
            tick_gaps = {b - a for a, b in zip(tick_times, tick_times[1:])}
            if tick_gaps - {10}:
                result.problems.append(
                    f"input tick gaps other than 10 ms: {sorted(tick_gaps)}"
                )
            if result.expected_ms is not None and tick_times[-1] != result.expected_ms:
                result.problems.append(
                    f"input ticks stop at {tick_times[-1]} ms, "
                    f"not the {result.expected_ms} ms finish"
                )
            keys_at = {
                t["race_time"]: (t["up"], t["down"], t["left"], t["right"])
                for t in ticks
            }
            disagreed = sum(
                1
                for r in rows
                if r["race_time"] in keys_at
                and keys_at[r["race_time"]]
                != (r["up"], r["down"], r["left"], r["right"])
            )
            if disagreed:
                result.problems.append(
                    f"{disagreed} samples disagree with the input tick at the "
                    "same race time"
                )
        elif rows:
            result.problems.append("inputs.jsonl is empty")

    # Frames live end to end in one file; the rows say where each one is. That
    # has to tile the file exactly: a gap means a frame was written that no row
    # claims, an overlap means two rows share pixels.
    blob = directory / "frames.bin"
    size = blob.stat().st_size if blob.is_file() else 0
    result.frames = sum(1 for r in rows if r.get("frame_bytes"))
    if not blob.is_file():
        result.problems.append("frames.bin is missing")
    else:
        expected = 0
        for row in rows:
            if row.get("frame_offset") != expected:
                result.problems.append(
                    f"row {row['i']} starts at {row.get('frame_offset')} in "
                    f"frames.bin, expected {expected}"
                )
                break
            expected += row.get("frame_bytes", 0)
        else:
            if expected != size:
                result.problems.append(
                    f"frames.bin is {size} bytes, rows account for {expected}"
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
        "total_input_ticks": sum(r.ticks for r in runs),
        "recorded_seconds": round(sum(r.duration_ms for r in runs) / 1000, 1),
        "frames_on_their_own_tick": sum(r.exact_frames for r in runs),
        "max_frame_lag_ms": max((r.max_frame_lag_ms for r in runs), default=0),
        "checks": runs,
    }


def _reason(directory: Path) -> str:
    """A folder name to file a failed run under."""
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # No meta at all: the run died before it could write one.
        return "no_meta"
    status = meta.get("status") or "unknown"
    return status if status != "ok" else "incomplete"


def clean_dataset(
    root: Path, *, period_ms: int = 50, delete: bool = False, dry_run: bool = False
) -> dict:
    """Move every run that fails ``check_run`` out of the dataset.

    Training reads whatever directories are present, so a half-written run is
    not merely noise -- it is frames whose labels stop matching partway through.
    Failures go to a *sibling* folder by default rather than being deleted, both
    because the collector's own summary is a log and not a live index, and
    because a desync is worth keeping to look at.

    Clearing them out also matters before re-collecting: a run directory is
    reused, and its frames are numbered, so a shorter second attempt would leave
    the tail of the first one behind.
    """
    report = check_dataset(root, period_ms=period_ms)
    checks = report.pop("checks")
    rejected_root = root.parent / f"{root.name}.rejected"

    removed: list[dict] = []
    for check in checks:
        if check.ok:
            continue
        directory = root / check.name
        reason = _reason(directory)
        entry = {
            "run": check.name,
            "reason": reason,
            "rows": check.rows,
            "frames": check.frames,
            "problems": check.problems,
        }
        if not delete:
            destination = rejected_root / reason / check.name
            entry["moved_to"] = str(destination)
        removed.append(entry)

        if dry_run:
            continue
        if delete:
            shutil.rmtree(directory)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            directory.replace(destination)

    summary = {
        "root": str(root),
        "runs": report["runs"],
        "kept": report["passed"],
        "removed": len(removed),
        "action": "deleted" if delete else "moved",
        "rejected_dir": None if delete else str(rejected_root),
        "dry_run": dry_run,
        "kept_rows": sum(c.rows for c in checks if c.ok),
        "kept_seconds": round(
            sum(c.duration_ms for c in checks if c.ok) / 1000, 1
        ),
        "details": removed,
    }
    if not dry_run and removed:
        (root / "clean.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary
