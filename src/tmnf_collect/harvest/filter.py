"""Sorting replays by the input device they were driven with.

Keyboard and pad are different control regimes, not different styles: in TMNF a
key press is instant full lock, while a pad emits a continuous value. Mixed into
one training set the same corner carries contradictory labels, so a corpus should
be one or the other.

This runs as a pass of its own, before any recording, and needs no game: pygbx
reads each replay's ghost directly, about 10 ms a file. A replay it cannot read
is set aside as unreadable rather than guessed at.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..common import replays as replay_files

KEYBOARD = "keyboard"
PAD = "pad"
NO_INPUTS = "no_inputs"
TOO_LONG = "too_long"
UNREADABLE = "unreadable"

def classify_replay(path: Path) -> str | None:
    """Classify a replay by reading its ghost. None if it cannot be read."""
    from pygbx import Gbx, GbxType

    try:
        # Hand it the bytes, not the path: pygbx opens a path and never closes
        # it, and Windows will not rename a file while a handle is open -- which
        # is exactly what sorting the folder needs to do next.
        ghost = Gbx(path.read_bytes()).get_class_by_id(GbxType.CTN_GHOST)
    except Exception:
        return None
    if ghost is None:
        return None

    entries = getattr(ghost, "control_entries", None)
    if not entries:
        return NO_INPUTS
    names = {entry.event_name for entry in entries}
    if "Steer" in names:
        return PAD
    if names & {"SteerLeft", "SteerRight"}:
        return KEYBOARD
    return UNREADABLE


@dataclass
class FilterResult:
    counts: dict[str, int] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)
    moved: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def filter_replays(
    folder: Path,
    *,
    want: str = KEYBOARD,
    max_seconds: float | None = 180.0,
    dry_run: bool = False,
) -> FilterResult:
    """Sort a folder of replays by input device, in one pass, before collecting.

    Also drops runs longer than ``max_seconds``: recording cost is linear in
    race time while the learning signal is roughly per corner, so a long run
    buys far less than the same minutes spent on several short ones.

    Anything rejected is moved into a *sibling* folder rather than deleted, so a
    later `collect` over the folder sees only the replays that belong --
    collection walks subdirectories, so a subfolder would still be found -- and
    nothing is thrown away.
    """
    started = time.monotonic()
    result = FilterResult()

    rejected_root = folder.parent / f"{folder.name}.rejected"
    paths = replay_files.discover_replays(folder)
    if not paths:
        return result

    verdicts: dict[Path, str] = {}
    for path in paths:
        # Length comes from the uncompressed header, so check it before doing
        # any ghost parsing.
        try:
            race_time = replay_files.read_replay(path).race_time
        except replay_files.ReplayError:
            verdicts[path] = UNREADABLE
            continue
        if max_seconds is not None and race_time > max_seconds * 1000:
            verdicts[path] = TOO_LONG
            continue

        verdicts[path] = classify_replay(path) or UNREADABLE

    for path in paths:
        kind = verdicts.get(path, UNREADABLE)
        result.counts[kind] = result.counts.get(kind, 0) + 1
        if kind == want:
            result.kept.append(str(path))
            continue

        destination = rejected_root / kind / path.name
        result.moved.append(
            {"replay": str(path), "kind": kind, "moved_to": str(destination)}
        )
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)

    result.seconds = round(time.monotonic() - started, 1)
    if not dry_run:
        (folder / "filter.json").write_text(
            json.dumps(
                {
                    "want": want,
                    "counts": result.counts,
                    "kept": len(result.kept),
                    "moved": result.moved,
                    "seconds": result.seconds,
                    "rejected_dir": str(rejected_root),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return result
