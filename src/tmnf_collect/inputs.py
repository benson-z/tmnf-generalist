"""Sorting replays by the input device they were driven with.

Keyboard and pad are different control regimes, not different styles: in TMNF a
key press is instant full lock, while a pad emits a continuous value. Mixed into
one training set the same corner carries contradictory labels, so a corpus should
be one or the other.

This runs as a pass of its own, before any recording, and takes whichever of two
routes is available.

If `pygbx` is installed it reads the replay's ghost directly and needs no game at
all: about 10 ms per replay. Otherwise it falls back to asking a running game to
`dump_inputs` each file, which is correct but costs a game launch plus roughly a
second per replay. Both were checked against the same replays and agreed on
every one.

pygbx is not a hard dependency because it pulls in `python-lzo`, whose newest
wheels are cp311 and would pin this project to Python 3.11. Install it if the
corpus is large enough for the speed to matter:

    uv pip install pygbx
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import replays as replay_files
from . import staging
from .paths import Layout, detect
from .session import NoInputsError, Session, SessionError

KEYBOARD = "keyboard"
PAD = "pad"
NO_INPUTS = "no_inputs"
TOO_LONG = "too_long"
UNREADABLE = "unreadable"

# TMInterface writes an analog `steer <value>` line for a pad replay, and
# `press left` / `press right` for a keyboard one.
_ANALOG = re.compile(r"^\s*[\d.:]+(?:-[\d.:]+)?\s+steer\s+-?\d+", re.MULTILINE)
_DIGITAL = re.compile(
    r"^\s*[\d.:]+(?:-[\d.:]+)?\s+(?:press|rel)\s+(?:up|down|left|right)",
    re.MULTILINE,
)


def classify_replay(path: Path) -> str | None:
    """Classify a replay by reading its ghost, without a game.

    Returns None when pygbx is unavailable or cannot read the file, so the
    caller can fall back to asking the game.
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError:
        return None

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


def classify_script(text: str) -> str:
    """Which device produced the run this input script came from."""
    if not text.strip():
        return NO_INPUTS
    if _ANALOG.search(text):
        return PAD
    if _DIGITAL.search(text):
        return KEYBOARD
    return UNREADABLE


@dataclass
class FilterResult:
    counts: dict[str, int] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)
    moved: list[dict] = field(default_factory=list)
    read_offline: int = 0  # classified without launching the game
    seconds: float = 0.0


def filter_replays(
    folder: Path,
    *,
    want: str = KEYBOARD,
    max_seconds: float | None = 180.0,
    port: int = 8477,
    layout: Layout | None = None,
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
    layout = layout or detect()
    result = FilterResult()

    rejected_root = folder.parent / f"{folder.name}.rejected"
    paths = replay_files.discover_replays(folder)
    if not paths:
        return result

    # Read what can be read without a game; only the leftovers need one.
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

        kind = classify_replay(path)
        if kind is not None:
            verdicts[path] = kind
    result.read_offline = len(verdicts)
    undecided = [p for p in paths if p not in verdicts]

    if undecided:
        # Stage before the game starts: it indexes its Tracks folder at startup
        # and will not see files added later.
        staged = {}
        for path in undecided:
            try:
                staged[path] = staging.stage_replay(path, layout)
            except OSError:
                verdicts[path] = UNREADABLE

        session = Session(port=port, layout=layout)
        session.start()
        try:
            session.prepare(speed=1.0)
            for index, path in enumerate(undecided):
                if path not in staged:
                    continue
                try:
                    script = session.dump_inputs(
                        staged[path], f"tmnf_filter_{index}.txt"
                    )
                    verdicts[path] = classify_script(
                        script.read_text(encoding="utf-8")
                    )
                except NoInputsError:
                    verdicts[path] = NO_INPUTS
                except (SessionError, OSError):
                    verdicts[path] = UNREADABLE
        finally:
            session.close()

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
                    "read_offline": result.read_offline,
                    "seconds": result.seconds,
                    "rejected_dir": str(rejected_root),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return result
