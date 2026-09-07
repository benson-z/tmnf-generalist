"""Reading replay and challenge files, and matching one to the other.

A ``.Replay.Gbx`` names the map it was driven on only by UID, so to re-drive it
we have to find the matching ``.Challenge.Gbx`` on disk.

Both file types start with an uncompressed XML header that carries everything
we need -- the map UID, the map name, and the replay's finish time -- so
nothing here decompresses a Gbx body.  That matters beyond simplicity: the map
UID the ghost record stores does not decode to the same string as the one in
the challenge header, and matching those two would silently never hit.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .paths import Layout, detect

# Where stock maps live; user maps are found under the Tracks folder.
CAMPAIGN_ROOT = Path(
    r"C:\Program Files (x86)\TmNationsForever\GameData\Tracks\Campaigns"
)

_HEADER = re.compile(rb"<header\b.*?</header>", re.S)
_HEADER_SCAN_BYTES = 65536


class ReplayError(RuntimeError):
    pass


def read_gbx_header(path: Path) -> ET.Element | None:
    """Parse the XML header block at the front of a Gbx file."""
    try:
        with path.open("rb") as handle:
            head = handle.read(_HEADER_SCAN_BYTES)
    except OSError:
        return None
    match = _HEADER.search(head)
    if match is None:
        return None
    try:
        return ET.fromstring(match.group().decode("utf-8", "replace"))
    except ET.ParseError:
        return None


@dataclass(frozen=True)
class ReplayInfo:
    """What a replay file tells us before we ever load the game."""

    path: Path
    map_uid: str
    race_time: int  # ms; the time our re-driven run has to reproduce
    respawns: int = 0
    stunt_score: int = 0
    exe_version: str = ""
    cp_times: list[int] = field(default_factory=list)

    @property
    def has_inputs(self) -> bool:
        """Only replays that finished carry the inputs that produced them."""
        return self.race_time > 0


def read_replay(path: Path) -> ReplayInfo:
    """Read a ``.Replay.Gbx``'s header."""
    header = read_gbx_header(path)
    if header is None or header.get("type") != "replay":
        raise ReplayError(f"{path.name} is not a readable replay file")

    challenge = header.find("challenge")
    times = header.find("times")
    uid = challenge.get("uid") if challenge is not None else None
    if not uid:
        raise ReplayError(f"{path.name} names no map")

    def number(element: ET.Element | None, key: str) -> int:
        if element is None:
            return 0
        try:
            return int(element.get(key, "0"))
        except ValueError:
            return 0

    return ReplayInfo(
        path=path,
        map_uid=uid,
        race_time=number(times, "best"),
        respawns=number(times, "respawns"),
        stunt_score=number(times, "stuntscore"),
        exe_version=header.get("exever", ""),
    )


def read_challenge(path: Path) -> tuple[str, str] | None:
    """Return a challenge's (uid, name), or None if it will not parse."""
    header = read_gbx_header(path)
    if header is None or header.get("type") != "challenge":
        return None
    ident = header.find("ident")
    if ident is None or not ident.get("uid"):
        return None
    return ident.get("uid", ""), ident.get("name", path.stem)


class ChallengeIndex:
    """UID -> challenge file, over the user's Tracks folder and stock campaigns.

    Cached on disk, since a well-stocked Tracks folder holds thousands of maps.
    """

    def __init__(self, layout: Layout | None = None, cache: Path | None = None):
        self.layout = layout or detect()
        self.cache_path = cache or (
            self.layout.tmi_dir / "tmnf-collect-mapindex.json"
        )
        self._by_uid: dict[str, str] = {}

    def roots(self) -> list[Path]:
        return [self.layout.challenges_dir, CAMPAIGN_ROOT]

    def load(self) -> bool:
        try:
            self._by_uid = json.loads(self.cache_path.read_text("utf-8"))
            return True
        except (OSError, ValueError):
            return False

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._by_uid, indent=1), encoding="utf-8"
        )

    def build(self) -> int:
        """Scan every challenge file. Returns how many maps were indexed."""
        found: dict[str, str] = {}
        for root in self.roots():
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if not path.name.lower().endswith(".challenge.gbx"):
                    continue
                parsed = read_challenge(path)
                if parsed:
                    found.setdefault(parsed[0], str(path))
        self._by_uid = found
        self.save()
        return len(found)

    def ensure(self) -> None:
        if not self._by_uid and not self.load():
            self.build()

    def find(self, map_uid: str) -> Path | None:
        """Locate a map by UID, rescanning once if it is not already known."""
        self.ensure()
        hit = self._by_uid.get(map_uid)
        if hit and Path(hit).is_file():
            return Path(hit)
        # A map added since the last scan, or a stale cache entry.
        self.build()
        hit = self._by_uid.get(map_uid)
        return Path(hit) if hit and Path(hit).is_file() else None


def discover_replays(root: Path) -> list[Path]:
    """Every replay under ``root``, sorted for a stable queue order."""
    if root.is_file():
        return [root]
    # Not a glob: the game writes both ".Replay.Gbx" and ".Replay.gbx", and
    # pathlib's case handling for that differs by platform.
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name.lower().endswith(".replay.gbx")
    )
