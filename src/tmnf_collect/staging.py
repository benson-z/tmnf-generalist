"""Putting maps, replays and input scripts where the game can find them.

TMInterface's ``map``, ``validate_replay`` and ``load`` commands resolve paths
relative to the user's ``Tracks/Challenges``, ``Tracks/Replays`` and
``Documents/TMInterface/Scripts`` folders, so anything we want the game to open
has to be copied under those roots first.  Everything we stage goes into a
single ``tmnf-collect`` subfolder so it is obvious what belongs to this tool.

The game indexes its Tracks folder at startup, so files must be staged *before*
the instance that will use them is launched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import mediatracker
from .paths import Layout, detect

STAGE_DIR = "tmnf-collect"

# Used once per instance to get the game out of the main menu, where the `map`
# command only queues.  Any replay whose map the game can find will do; a stock
# campaign one is always present.
BOOTSTRAP_TRACK = "A01-Race.Challenge.Gbx"
BOOTSTRAP_REPLAY = "A01-Race.Replay.gbx"
_CAMPAIGN_WHITE = Path(
    r"C:\Program Files (x86)\TmNationsForever\GameData\Tracks\Campaigns\Nations\White"
)


def challenges_dir(layout: Layout) -> Path:
    return layout.challenges_dir / STAGE_DIR


def replays_dir(layout: Layout) -> Path:
    return layout.replays_dir / STAGE_DIR


def stage_challenge(
    source: Path, layout: Layout | None = None, *, strip_intro: bool = False
) -> str:
    """Copy a ``.Challenge.Gbx`` in and return the name the ``map`` command wants.

    With ``strip_intro`` the staged copy has its MediaTracker clips removed,
    which is what makes the intro flythrough go away. The map keeps its UID, so
    it is still the map the replay was driven on.
    """
    layout = layout or detect()
    target_dir = challenges_dir(layout)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name

    if strip_intro:
        # Size is how a plain copy decides it is up to date, and a stripped map
        # is a different size from its source, so mark it instead.
        marker = target.with_suffix(".stripped")
        if not target.exists() or not marker.exists():
            try:
                removed = mediatracker.strip_file(source, target)
                marker.write_text(str(removed), encoding="utf-8")
            except mediatracker.MediaTrackerError:
                shutil.copyfile(source, target)  # keep the map, keep the intro
        return f"{STAGE_DIR}/{source.name}"

    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
    return f"{STAGE_DIR}/{source.name}"


def stage_replay(source: Path, layout: Layout | None = None) -> str:
    """Copy a ``.Replay.Gbx`` in and return the name ``validate_replay`` wants."""
    layout = layout or detect()
    target_dir = replays_dir(layout)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
    return f"{STAGE_DIR}/{source.name}"


def write_script(name: str, body: str, layout: Layout | None = None) -> str:
    """Write an input script and return the name the ``load`` command wants."""
    layout = layout or detect()
    layout.scripts_dir.mkdir(parents=True, exist_ok=True)
    (layout.scripts_dir / name).write_text(body, encoding="utf-8")
    return name


def stage_bootstrap(layout: Layout | None = None) -> tuple[str, str]:
    """Stage the menu-escape replay and its map. Returns (replay, challenge)."""
    layout = layout or detect()
    challenge_source = _CAMPAIGN_WHITE / BOOTSTRAP_TRACK
    replay_source = _CAMPAIGN_WHITE / BOOTSTRAP_REPLAY
    if not challenge_source.is_file() or not replay_source.is_file():
        raise FileNotFoundError(
            f"stock campaign files not found under {_CAMPAIGN_WHITE}; "
            "set TMNF_CAMPAIGN_DIR or stage a bootstrap replay manually"
        )

    challenge = stage_challenge(challenge_source, layout)

    # The game only lists replays with the exact `.Replay.Gbx` suffix.
    replay_dir = replays_dir(layout)
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_target = replay_dir / "A01-Race.Replay.Gbx"
    if not replay_target.exists():
        shutil.copyfile(replay_source, replay_target)
    return f"{STAGE_DIR}/{replay_target.name}", challenge
