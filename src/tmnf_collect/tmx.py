"""Finding and fetching maps and replays from TrackMania Exchange.

Two jobs live here.

*Filling a gap*: a replay names its map only by UID, and a replay downloaded
from TMX rarely arrives with the map beside it, so `fetch_map` resolves that UID.

*Building a corpus*: `harvest` goes the other way. It picks maps by quality, asks
TMX for each map's replay leaderboard, and downloads a demonstration for each.
Selecting maps first means the map and its replay always match, which removes the
pairing problem entirely.

Everything here is opt-in: it reaches a third-party site and writes files into the
game's Tracks folder.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .replays import ReplayError, read_challenge, read_replay

BASE = "https://tmnf.exchange"
TRACKS_API = f"{BASE}/api/tracks"
REPLAYS_API = f"{BASE}/api/replays"
TRACK_GBX = f"{BASE}/trackgbx/{{track_id}}"
REPLAY_GBX = f"{BASE}/recordgbx/{{replay_id}}"

USER_AGENT = "tmnf-collect/0.1 (dataset collection tool)"
TIMEOUT = 30.0
PAGE = 100  # the API's maximum page size
COURTESY_DELAY = 0.2  # seconds between requests

# Ordering ids the API understands. Awards descending is the useful one: it puts
# well-built, well-received maps first, which is a far better quality signal than
# anything derivable from a map's own metadata.
ORDER_AWARDS_DESC = 6

TRACK_FIELDS = "TrackId,TrackName,Authors,AuthorTime,Awards,Tags,Difficulty"
REPLAY_FIELDS = "ReplayId,ReplayTime,User.Name"

# The API returns tags as bare ids and serves no table for them, but the site's
# own front end carries one: `enumTrackTagDesc` in /js/meta.js, which this list
# matches entry for entry. The ids are indices into that array, so 0 is the tag
# "Race" and not an absent tag -- a map with no tag comes back with an empty
# list. Race is also the catch-all most maps carry, so it says little on its own.
# Ids the site adds later render as "tag-<n>" rather than a guess.
TAG_NAMES = {
    0: "Race",
    1: "Stunt",
    2: "Maze",
    3: "Offroad",
    4: "Multilap",
    5: "FullSpeed",
    6: "LOL",
    7: "Tech",
    8: "SpeedTech",
    9: "RPG",
    10: "PressForward",
    11: "Trial",
    12: "Grass",
    13: "Story",
    14: "Nascar",
    15: "Speedfun",
    16: "Endurance",
    17: "Altered Nadeo",
    18: "Transitional",
}


def tag_name(tag_id: int) -> str:
    return TAG_NAMES.get(tag_id, f"tag-{tag_id}")


def tag_names(tags: tuple[int, ...]) -> str:
    return ", ".join(tag_name(tag) for tag in tags) if tags else "untagged"


def parse_tags(text: str) -> tuple[int, ...]:
    """Turn ``"LOL,PressForward"`` or ``"6,10"`` into tag ids.

    Names are matched case-insensitively and ignoring spaces and underscores,
    because the site itself spells one of them both ways.
    """
    if not text:
        return ()
    lookup = {
        name.lower().replace(" ", "").replace("_", ""): tag_id
        for tag_id, name in TAG_NAMES.items()
    }
    found: list[int] = []
    for part in text.split(","):
        word = part.strip()
        if not word:
            continue
        if word.isdigit():
            found.append(int(word))
            continue
        key = word.lower().replace(" ", "").replace("_", "")
        if key not in lookup:
            raise TmxError(
                f"unknown tag {word!r}; known tags are "
                + ", ".join(TAG_NAMES[i] for i in sorted(TAG_NAMES))
            )
        found.append(lookup[key])
    return tuple(dict.fromkeys(found))


class TmxError(RuntimeError):
    pass


@dataclass(frozen=True)
class TmxTrack:
    track_id: int
    name: str
    author_time: int = 0
    awards: int = 0
    difficulty: int = 0
    tags: tuple[int, ...] = ()


@dataclass(frozen=True)
class TmxReplay:
    replay_id: int
    time_ms: int
    user: str


@dataclass
class Harvested:
    """One map plus the replay chosen to demonstrate it."""

    track: TmxTrack
    replay: TmxReplay
    map_path: Path
    replay_path: Path
    is_author_run: bool


@dataclass
class HarvestResult:
    picked: list[Harvested] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    tracks_considered: int = 0


# ----------------------------------------------------------------- transport


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TmxError(f"{url}: {exc}") from exc
    finally:
        time.sleep(COURTESY_DELAY)


def _get_json(url: str) -> dict:
    try:
        return json.loads(_get(url))
    except ValueError as exc:
        raise TmxError(f"unreadable response from {url}: {exc}") from exc


# -------------------------------------------------------------------- search


def _track_from(row: dict) -> TmxTrack:
    return TmxTrack(
        track_id=int(row["TrackId"]),
        name=row.get("TrackName", ""),
        author_time=int(row.get("AuthorTime") or 0),
        awards=int(row.get("Awards") or 0),
        difficulty=int(row.get("Difficulty") or 0),
        tags=tuple(row.get("Tags") or ()),
    )


def search_tracks(
    *,
    limit: int,
    min_author_time: int | None = None,
    max_author_time: int | None = None,
    min_awards: int = 0,
    include_tags: tuple[int, ...] = (),
    exclude_tags: tuple[int, ...] = (),
) -> list[TmxTrack]:
    """Maps ordered by award count, best first.

    ``min_awards`` is applied here rather than by the API, which ignores an
    awards filter; because the ordering is awards-descending, dropping below the
    threshold means every later page is below it too, so the walk can stop.

    Tags are filtered here too -- the API takes no tag parameter, and the one it
    appears to take is ignored. A map matches ``include_tags`` if it carries any
    of them, and is dropped if it carries any of ``exclude_tags``. Well-awarded
    maps predate TMX's multi-tag support and carry exactly one tag each, so in
    practice the two are simple opposites.
    """
    found: list[TmxTrack] = []
    after: int | None = None

    while len(found) < limit:
        params = {
            "fields": TRACK_FIELDS,
            "count": PAGE,
            "order1": ORDER_AWARDS_DESC,
        }
        if min_author_time is not None:
            params["authorTimeMin"] = min_author_time
        if max_author_time is not None:
            params["authorTimeMax"] = max_author_time
        if after is not None:
            params["after"] = after

        payload = _get_json(f"{TRACKS_API}?{urllib.parse.urlencode(params)}")
        rows = payload.get("Results") or []
        if not rows:
            break

        for row in rows:
            track = _track_from(row)
            if track.awards < min_awards:
                return found  # ordering guarantees nothing better follows
            if exclude_tags and set(track.tags) & set(exclude_tags):
                continue
            if include_tags and not set(track.tags) & set(include_tags):
                continue
            found.append(track)
            if len(found) >= limit:
                break

        after = int(rows[-1]["TrackId"])
        if not payload.get("More"):
            break

    return found


def find_by_uid(map_uid: str) -> TmxTrack | None:
    """Look a map up by its UID. Returns None if TMX does not have it."""
    query = urllib.parse.urlencode({"fields": TRACK_FIELDS, "uid": map_uid})
    results = _get_json(f"{TRACKS_API}?{query}").get("Results") or []
    return _track_from(results[0]) if results else None


def track_tags(
    track_ids: list[int], *, batch: int = 50
) -> dict[int, tuple[int, ...]]:
    """Tag ids for many tracks at once.

    The ``id`` filter takes a comma-separated list, so a corpus of a few hundred
    maps costs a handful of requests rather than one each.
    """
    found: dict[int, tuple[int, ...]] = {}
    for start in range(0, len(track_ids), batch):
        chunk = track_ids[start : start + batch]
        query = urllib.parse.urlencode(
            {
                "fields": "TrackId,Tags",
                "count": len(chunk),
                "id": ",".join(str(i) for i in chunk),
            }
        )
        for row in _get_json(f"{TRACKS_API}?{query}").get("Results") or []:
            found[int(row["TrackId"])] = tuple(row.get("Tags") or ())
    return found


def track_replays(track_id: int, *, limit: int = PAGE) -> list[TmxReplay]:
    """A map's replay leaderboard, fastest first."""
    query = urllib.parse.urlencode(
        {"trackId": track_id, "fields": REPLAY_FIELDS, "count": limit}
    )
    rows = _get_json(f"{REPLAYS_API}?{query}").get("Results") or []
    replays = [
        TmxReplay(
            replay_id=int(row["ReplayId"]),
            time_ms=int(row.get("ReplayTime") or 0),
            user=(row.get("User") or {}).get("Name", ""),
        )
        for row in rows
    ]
    return sorted(replays, key=lambda r: r.time_ms)


# ------------------------------------------------------------------ download


def download_track(
    track: TmxTrack, into: Path, *, expect_uid: str | None = None
) -> tuple[Path, str]:
    """Download a map. Returns its path and its own UID.

    Named after the TMX id rather than the map's title: real map names carry
    spaces and formatting codes, and the game's console commands take these
    paths as bare words.
    """
    into.mkdir(parents=True, exist_ok=True)
    target = into / f"tmx-{track.track_id}.Challenge.Gbx"

    if not target.is_file():
        data = _get(TRACK_GBX.format(track_id=track.track_id))
        if not data.startswith(b"GBX"):
            raise TmxError(
                f"track {track.track_id}: {len(data)} bytes that are not a Gbx"
            )
        partial = target.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(target)

    parsed = read_challenge(target)
    if not parsed:
        target.unlink(missing_ok=True)
        raise TmxError(f"track {track.track_id} did not parse as a challenge")
    if expect_uid is not None and parsed[0] != expect_uid:
        target.unlink(missing_ok=True)
        raise TmxError(
            f"track {track.track_id} has UID {parsed[0]}, expected {expect_uid}"
        )
    return target, parsed[0]


def download_replay(replay: TmxReplay, into: Path, *, expect_uid: str) -> Path:
    """Download a replay and check it belongs to the map we think it does."""
    into.mkdir(parents=True, exist_ok=True)
    target = into / f"tmx-{replay.replay_id}.Replay.Gbx"

    if not target.is_file():
        data = _get(REPLAY_GBX.format(replay_id=replay.replay_id))
        if not data.startswith(b"GBX"):
            raise TmxError(
                f"replay {replay.replay_id}: {len(data)} bytes that are not a Gbx"
            )
        partial = target.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(target)

    try:
        info = read_replay(target)
    except ReplayError as exc:
        # The site occasionally serves a file that is not a replay. Drop it
        # and move on: one bad download must not end a harvest of thousands.
        target.unlink(missing_ok=True)
        raise TmxError(f"replay {replay.replay_id}: {exc}") from exc
    if info.map_uid != expect_uid:
        target.unlink(missing_ok=True)
        raise TmxError(
            f"replay {replay.replay_id} is for map {info.map_uid}, not {expect_uid}"
        )
    return target


def fetch_map(map_uid: str, into: Path) -> Path | None:
    """Find and download the map with this UID, or None if TMX lacks it."""
    track = find_by_uid(map_uid)
    if track is None:
        return None
    return download_track(track, into, expect_uid=map_uid)[0]


# ------------------------------------------------------------------- harvest


# A run far slower than the map's best is not a demonstration of driving it.
# Author validation runs in particular can be a leisurely lap: one sampled map
# has an author time of 199 s against a 89 s record, another 190 s against 15 s.
SLOW_FACTOR = 1.5


def choose_replay(
    replays: list[TmxReplay], track: TmxTrack, *, prefer: str
) -> TmxReplay | None:
    """Pick which run on a map to learn from.

    ``median`` is the default: a competent mid-leaderboard run. Records are
    edge-of-control and nearly identical to each other, which gives poor state
    coverage for a behaviour-cloning prior, while the map author's own
    validation lap is often far too slow to be worth imitating.
    """
    if not replays:
        return None

    fastest = replays[0].time_ms
    usable = [r for r in replays if r.time_ms <= fastest * SLOW_FACTOR] or [
        replays[0]
    ]

    if prefer == "best":
        return usable[0]
    if prefer == "author":
        author = [r for r in replays if r.time_ms == track.author_time]
        if author:
            return author[0]
        return usable[-1]  # closest in spirit: the slowest still-credible run
    return usable[len(usable) // 2]


def harvest(
    *,
    maps_into: Path,
    replays_into: Path,
    limit: int,
    min_author_time: int | None = None,
    max_author_time: int | None = None,
    min_awards: int = 0,
    include_tags: tuple[int, ...] = (),
    exclude_tags: tuple[int, ...] = (),
    prefer: str = "median",
    dry_run: bool = False,
) -> HarvestResult:
    """Select maps on TMX and download a demonstration for each."""
    result = HarvestResult()

    # Over-fetch: some maps have no replays and drop out.
    tracks = search_tracks(
        limit=limit * 3,
        min_author_time=min_author_time,
        max_author_time=max_author_time,
        min_awards=min_awards,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )
    result.tracks_considered = len(tracks)

    for track in tracks:
        if len(result.picked) >= limit:
            break
        try:
            replays = track_replays(track.track_id)
        except TmxError as exc:
            result.skipped.append({"track": track.track_id, "reason": str(exc)})
            continue

        replay = choose_replay(replays, track, prefer=prefer)
        if replay is None:
            result.skipped.append(
                {
                    "track": track.track_id,
                    "name": track.name,
                    "reason": "no replays on its leaderboard",
                }
            )
            continue

        if dry_run:
            result.picked.append(
                Harvested(
                    track=track,
                    replay=replay,
                    map_path=Path(),
                    replay_path=Path(),
                    is_author_run=replay.time_ms == track.author_time,
                )
            )
            continue

        try:
            map_path, map_uid = download_track(track, maps_into)
            replay_path = download_replay(
                replay, replays_into, expect_uid=map_uid
            )
        except TmxError as exc:
            result.skipped.append({"track": track.track_id, "reason": str(exc)})
            continue

        result.picked.append(
            Harvested(
                track=track,
                replay=replay,
                map_path=map_path,
                replay_path=replay_path,
                is_author_run=replay.time_ms == track.author_time,
            )
        )

    return result
