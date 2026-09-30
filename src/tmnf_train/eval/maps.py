"""Eval maps: stock campaign maps, corpus maps, or any TMX track.

``B01-Race`` is looked up among the stock campaigns. ``corpus:tmx-1016190``
names a run of the training corpus: its map file (staged locally by the
collector) and its recorded trajectory, used only as a diagnostic reference
line. ``tmx:10460245`` is a track id on tmnf.exchange, downloaded once into
the game's Tracks folder with the collector's TMX client (Gbx and UID
checked). Medal times are read from the map file's header either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmnf_collect.common.paths import detect
from tmnf_collect.common.replays import CAMPAIGN_ROOT, read_gbx_header

CORPUS_PREFIX = "corpus:"
TMX_PREFIX = "tmx:"


@dataclass(frozen=True)
class MapInfo:
    name: str  # as in the map header (may carry TrackMania formatting codes)
    uid: str
    path: Path
    author_ms: int
    gold_ms: int
    silver_ms: int
    bronze_ms: int
    label: str = ""  # short, filename-safe: "B01-Race" or "tmx-1016190"
    split: str = "campaign"  # campaign | train | val
    demo: np.ndarray | None = None  # (N, 3) positions of the corpus run, 20 Hz


def find_map(name: str) -> Path:
    """A stock campaign map by name, e.g. ``B01-Race``."""
    hits = sorted(CAMPAIGN_ROOT.rglob(f"{name}.Challenge.Gbx"))
    if not hits:
        raise FileNotFoundError(f"{name}.Challenge.Gbx not under {CAMPAIGN_ROOT}")
    return hits[0]


def read_map(path: Path) -> MapInfo:
    header = read_gbx_header(path)
    if header is None or header.get("type") != "challenge":
        raise ValueError(f"{path} is not a readable challenge")
    ident, times = header.find("ident"), header.find("times")
    if ident is None or times is None:
        raise ValueError(f"{path} has no ident/times in its header")
    return MapInfo(
        name=ident.get("name", path.stem),
        uid=ident.get("uid", ""),
        path=path,
        author_ms=int(times.get("authortime")),
        gold_ms=int(times.get("gold")),
        silver_ms=int(times.get("silver")),
        bronze_ms=int(times.get("bronze")),
        label=path.name.split(".")[0],
    )


def fetch_tmx(track_id: int) -> Path:
    """A TMX track's map file, downloading it on first use."""
    from tmnf_collect.harvest import tmx

    into = detect().challenges_dir / "tmnf-collect"
    path, _uid = tmx.download_track(tmx.TmxTrack(track_id=track_id, name=""), into)
    return path


def resolve(spec: str, corpus: str, val_runs: set[str] | None = None) -> MapInfo:
    """A map spec from the eval config -> MapInfo (with a demo line for corpus maps)."""
    if spec.startswith(TMX_PREFIX):
        track_id = int(spec[len(TMX_PREFIX):])
        info = read_map(fetch_tmx(track_id))
        return MapInfo(**{**info.__dict__, "label": f"tmx-{track_id}", "split": "tmx"})
    if not spec.startswith(CORPUS_PREFIX):
        return read_map(find_map(spec))
    run = spec[len(CORPUS_PREFIX):]
    run_dir = Path(corpus) / run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    # The collector recorded on another machine; its map file name is what
    # carries over (it stages maps under Tracks/Challenges/tmnf-collect).
    map_file = detect().challenges_dir / "tmnf-collect" / Path(meta["map_file"].replace("\\", "/")).name
    if not map_file.is_file():
        raise FileNotFoundError(f"{spec}: map file {map_file} is not on this machine")
    info = read_map(map_file)
    demo = np.array([json.loads(l)["position"] for l in (run_dir / "samples.jsonl").open(encoding="utf-8") if l.strip()])
    split = "val" if val_runs is not None and run in val_runs else "train"
    return MapInfo(**{**info.__dict__, "label": run, "split": split, "demo": demo})


class DemoLine:
    """Distance from the car to a corpus run's recorded trajectory.

    A diagnostic for corpus maps only, never used on B01 and never a training
    signal. The polyline is densified to ~0.5 m so nearest-point distance is a
    good stand-in for distance to the line.
    """

    def __init__(self, demo: np.ndarray, spacing_m: float = 0.5):
        pts = [demo[:1]]
        for a, b in zip(demo[:-1], demo[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing_m)))
            pts.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
        self.points = np.concatenate(pts)
        # Fraction of the demo's length at each densified point.
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.fraction = arc / max(arc[-1], 1e-6)

    def nearest(self, pos) -> tuple[float, float]:
        """(distance in m, fraction of the demo line at the nearest point)."""
        d = np.linalg.norm(self.points - np.asarray(pos)[None], axis=1)
        k = int(d.argmin())
        return float(d[k]), float(self.fraction[k])
