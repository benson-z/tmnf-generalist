"""What a collected corpus is actually made of, by run length.

Total recorded hours is the number everyone quotes, but it hides the shape that
matters: an hour of thirty-second maps is far more track geometry, and far more
corners per frame, than an hour spent on four long ones. This reads the runs on
disk and draws that distribution.

Lengths come from the samples themselves rather than the replay's advertised
time, so what is plotted is what was recorded.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import tmx

# Harvested maps are filed under their TMX id, which is the only handle a
# recorded run keeps on where it came from.
_TMX_MAP = re.compile(r"tmx-(\d+)\.", re.IGNORECASE)
_TAG_CACHE = "tags.json"

_BG = (16, 16, 20)
_DIM = (70, 74, 82)
_TEXT = (236, 238, 242)
_MUTED = (150, 156, 168)
_BAR = (90, 200, 250)


@dataclass(frozen=True)
class Run:
    name: str
    seconds: float
    status: str
    track_id: int | None = None


def read_runs(root: Path, *, only_ok: bool = True) -> tuple[list[Run], list[str]]:
    """Every run in a dataset with its recorded length.

    Returns the runs and the names of directories that carried no readable
    length -- an errored run that never got as far as writing its meta.
    """
    runs: list[Run] = []
    unreadable: list[str] = []

    for directory in sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        try:
            meta = json.loads(
                (directory / "meta.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            unreadable.append(directory.name)
            continue

        status = meta.get("status", "?")
        if only_ok and status != "ok":
            continue

        # Prefer the finish time the game reported; fall back to counting the
        # samples, which is all an unfinished run has.
        finish = meta.get("recorded_finish_time_ms")
        if finish is None:
            samples = meta.get("samples") or 0
            period = meta.get("period_ms") or 50
            finish = max(0, samples - 1) * period
        found = _TMX_MAP.search(Path(meta.get("map_file") or "").name)
        runs.append(
            Run(
                directory.name,
                finish / 1000,
                status,
                int(found.group(1)) if found else None,
            )
        )

    return runs, unreadable


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return values[index]


def summarize(runs: list[Run], *, bin_seconds: float = 10.0) -> dict:
    lengths = sorted(run.seconds for run in runs)
    total = sum(lengths)

    bins: list[dict] = []
    if lengths:
        last = int(lengths[-1] // bin_seconds)
        counts = [0] * (last + 1)
        for length in lengths:
            counts[min(last, int(length // bin_seconds))] += 1
        bins = [
            {
                "from": round(index * bin_seconds, 3),
                "to": round((index + 1) * bin_seconds, 3),
                "count": count,
            }
            for index, count in enumerate(counts)
        ]

    return {
        "runs": len(runs),
        "total_seconds": round(total, 1),
        "total_hours": round(total / 3600, 3),
        "mean_seconds": round(total / len(lengths), 2) if lengths else 0.0,
        "min_seconds": round(lengths[0], 2) if lengths else 0.0,
        "p25_seconds": round(percentile(lengths, 0.25), 2),
        "median_seconds": round(percentile(lengths, 0.50), 2),
        "p75_seconds": round(percentile(lengths, 0.75), 2),
        "p90_seconds": round(percentile(lengths, 0.90), 2),
        "max_seconds": round(lengths[-1], 2) if lengths else 0.0,
        "bin_seconds": bin_seconds,
        "bins": bins,
    }


def tag_counts(root: Path, runs: list[Run], *, refresh: bool = False) -> dict:
    """How the corpus splits across TMX's map tags.

    A map carries any number of tags, so these count to more than the number of
    runs; what they answer is what kind of driving the corpus is teaching -- a
    pile of LOL or PressForward maps is time spent learning nothing about racing.

    Tags are not in the recorded run, so they come from TMX once and are cached
    beside the dataset.
    """
    cache_path = root / _TAG_CACHE
    known: dict[int, tuple[int, ...]] = {}
    if not refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            known = {
                int(key): tuple(value)
                for key, value in (cached.get("tracks") or {}).items()
            }
        except (OSError, ValueError):
            known = {}

    wanted = {run.track_id for run in runs if run.track_id is not None}
    missing = sorted(wanted - known.keys())
    fetched = 0
    error = ""
    if missing:
        try:
            known.update(tmx.track_tags(missing))
            fetched = len(missing)
            cache_path.write_text(
                json.dumps(
                    {"tracks": {str(k): list(v) for k, v in known.items()}},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except (tmx.TmxError, OSError) as exc:
            error = str(exc)

    counts: Counter[str] = Counter()
    seconds: Counter[str] = Counter()
    untagged = 0
    unknown = 0
    for run in runs:
        tags = known.get(run.track_id) if run.track_id is not None else None
        if tags is None:
            unknown += 1
            continue
        if not tags:
            untagged += 1
            continue
        for tag in tags:
            counts[tmx.tag_name(tag)] += 1
            seconds[tmx.tag_name(tag)] += run.seconds

    return {
        "runs": len(runs),
        "tagged_runs": len(runs) - unknown - untagged,
        "untagged_runs": untagged,
        "unknown_runs": unknown,  # not from TMX, or the lookup did not answer
        "fetched_from_tmx": fetched,
        "error": error,
        "tags": [
            {
                "tag": tag,
                "runs": count,
                "seconds": round(seconds[tag], 1),
            }
            for tag, count in counts.most_common()
        ],
    }


def format_tags(summary: dict, *, width: int = 32) -> str:
    """Tag counts as text, in the same shape as the histogram.

    The bars are drawn from recorded time rather than run count, because time is
    what a tag actually costs and what the training set is made of; run counts
    sit beside them and the two do not always agree, short maps being unevenly
    spread across tags.
    """
    tags = summary["tags"]
    if not tags:
        return "no tags known for this dataset"

    longest = max(entry["seconds"] for entry in tags) or 1
    total = sum(entry["seconds"] for entry in tags) or 1
    lines = [
        f"map tags ({summary['tagged_runs']} of {summary['runs']} runs tagged, "
        f"{format_hours(total)} in total; a map can carry several)"
    ]
    for entry in tags:
        bar = "#" * round(width * entry["seconds"] / longest)
        lines.append(
            f"  {entry['tag']:14} {entry['runs']:4d} runs "
            f"{format_hours(entry['seconds']):>12} "
            f"{entry['seconds'] / total:5.1%} |{bar}"
        )
    tail = []
    if summary["untagged_runs"]:
        tail.append(f"{summary['untagged_runs']} carry no tag")
    if summary["unknown_runs"]:
        tail.append(f"{summary['unknown_runs']} not from TMX")
    if tail:
        lines.append("  " + ", ".join(tail))
    if summary["error"]:
        lines.append(f"  tag lookup failed: {summary['error']}")
    return "\n".join(lines)


def format_hours(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def format_short(seconds: float) -> str:
    """The same duration in as few characters as it can be read in."""
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m"
    return f"{total}s"


def histogram(summary: dict, *, width: int = 48) -> str:
    """The distribution as text, for reading in a terminal."""
    bins = summary["bins"]
    if not bins:
        return "no runs"

    tallest = max(b["count"] for b in bins) or 1
    hours = summary["total_hours"]
    lines = [
        f"{summary['runs']} runs, {format_hours(summary['total_seconds'])} "
        f"of driving ({hours:.2f} h)",
        f"min {summary['min_seconds']:.1f}s   "
        f"p25 {summary['p25_seconds']:.1f}s   "
        f"median {summary['median_seconds']:.1f}s   "
        f"p75 {summary['p75_seconds']:.1f}s   "
        f"max {summary['max_seconds']:.1f}s",
        "",
    ]
    for entry in bins:
        bar = "#" * round(width * entry["count"] / tallest)
        lines.append(
            f"{entry['from']:6.0f}-{entry['to']:4.0f}s "
            f"{entry['count']:4d} |{bar}"
        )
    return "\n".join(lines)


def render_png(
    summary: dict, path: Path, *, size: tuple[int, int] = (960, 520)
) -> Path:
    """The same histogram as an image, for keeping alongside a dataset."""
    from PIL import Image, ImageDraw

    from .video import _font

    bins = summary["bins"]
    image = Image.new("RGB", size, _BG)
    draw = ImageDraw.Draw(image)
    title = _font(20)
    label = _font(13)

    draw.text(
        (28, 20),
        f"run length distribution  --  {summary['runs']} runs, "
        f"{format_hours(summary['total_seconds'])}",
        font=title,
        fill=_TEXT,
    )
    draw.text(
        (28, 48),
        f"median {summary['median_seconds']:.1f}s    "
        f"mean {summary['mean_seconds']:.1f}s    "
        f"range {summary['min_seconds']:.1f}-{summary['max_seconds']:.1f}s",
        font=label,
        fill=_MUTED,
    )

    top = 92
    tags = (summary.get("tag_counts") or {}).get("tags") or []
    if tags:
        draw.text(
            (28, 68),
            "tags:  "
            + "   ".join(
                f"{e['tag']} {e['runs']} / {format_short(e['seconds'])}"
                for e in tags[:6]
            ),
            font=label,
            fill=_MUTED,
        )
        top = 112

    left, right, bottom = 60, size[0] - 28, size[1] - 52
    draw.line([(left, bottom), (right, bottom)], fill=_DIM, width=1)
    draw.line([(left, top), (left, bottom)], fill=_DIM, width=1)
    if not bins:
        return _save(image, path)

    tallest = max(b["count"] for b in bins) or 1
    step = (right - left) / len(bins)
    for index, entry in enumerate(bins):
        height = (bottom - top) * entry["count"] / tallest
        x0 = left + index * step + 1
        x1 = x0 + step - 2
        if entry["count"]:
            draw.rectangle([x0, bottom - height, x1, bottom], fill=_BAR)
            draw.text(
                (x0, bottom - height - 15), str(entry["count"]), font=label, fill=_MUTED
            )
        # Only every other tick when bars get narrow, or the axis is a smear.
        if step > 34 or index % 2 == 0:
            draw.text(
                (x0, bottom + 8), f"{entry['from']:.0f}", font=label, fill=_MUTED
            )

    draw.text(
        (left, bottom + 28),
        f"run length (seconds, {summary['bin_seconds']:.0f}s bins)",
        font=label,
        fill=_MUTED,
    )
    return _save(image, path)


def _save(image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
