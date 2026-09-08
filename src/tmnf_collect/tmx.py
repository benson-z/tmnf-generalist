"""Fetching missing maps from TrackMania Exchange.

A replay names its map only by UID, and a replay downloaded from TMX is
usually accompanied by nothing at all -- so for any corpus that did not come
off this machine, most maps will be missing. TMX can be searched by that same
UID, which makes the lookup exact rather than a guess at the map's name.

This is opt-in (``--fetch-maps``): it reaches out to a third-party site and
writes files into the game's Tracks folder.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .replays import read_challenge

API = "https://tmnf.exchange/api/tracks"
DOWNLOAD = "https://tmnf.exchange/trackgbx/{track_id}"
USER_AGENT = "tmnf-collect/0.1 (dataset collection tool)"
TIMEOUT = 30.0


class TmxError(RuntimeError):
    pass


@dataclass(frozen=True)
class TmxTrack:
    track_id: int
    name: str


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TmxError(f"{url}: {exc}") from exc


def find_by_uid(map_uid: str) -> TmxTrack | None:
    """Look a map up by its UID. Returns None if TMX does not have it."""
    query = urllib.parse.urlencode(
        {"fields": "TrackId,TrackName", "uid": map_uid}
    )
    try:
        payload = json.loads(_get(f"{API}?{query}"))
    except ValueError as exc:
        raise TmxError(f"unreadable response for {map_uid}: {exc}") from exc

    results = payload.get("Results") or []
    if not results:
        return None
    first = results[0]
    return TmxTrack(track_id=int(first["TrackId"]), name=first.get("TrackName", ""))


def download(track: TmxTrack, into: Path, map_uid: str) -> Path:
    """Download a map and check it is really the one we asked for.

    The file is named after its TMX id rather than its title: map names carry
    spaces and formatting codes, and the game's console commands take these
    paths as bare words.
    """
    into.mkdir(parents=True, exist_ok=True)
    target = into / f"tmx-{track.track_id}.Challenge.Gbx"
    if target.is_file():
        parsed = read_challenge(target)
        if parsed and parsed[0] == map_uid:
            return target

    data = _get(DOWNLOAD.format(track_id=track.track_id))
    if not data.startswith(b"GBX"):
        raise TmxError(
            f"TMX returned {len(data)} bytes for track {track.track_id} that "
            "are not a Gbx file"
        )

    partial = target.with_suffix(".part")
    partial.write_bytes(data)
    parsed = read_challenge(partial)
    if not parsed or parsed[0] != map_uid:
        got = parsed[0] if parsed else "unreadable"
        partial.unlink(missing_ok=True)
        raise TmxError(
            f"track {track.track_id} has UID {got}, expected {map_uid}"
        )
    partial.replace(target)
    return target


def fetch_map(map_uid: str, into: Path) -> Path | None:
    """Find and download the map with this UID, or None if TMX lacks it."""
    track = find_by_uid(map_uid)
    if track is None:
        return None
    return download(track, into, map_uid)
