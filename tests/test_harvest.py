"""Tests for the quota harvest, against a fake TMX: no network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tmnf_collect.harvest import filter as filter_mod
from tmnf_collect.harvest import tmx


def _row(track_id: int, awards: int, tags: list[int]) -> dict:
    return {"TrackId": track_id, "TrackName": f"t{track_id}", "AuthorTime": 40000,
            "Awards": awards, "Tags": tags}


@pytest.fixture
def fake_tmx(monkeypatch, tmp_path):
    """300 maps over 3 pages, awards falling, tags cycling Race/Tech/Offroad.

    Replays on map ids divisible by 5 are pad on the median run and keyboard on
    its neighbour; on ids divisible by 11 the neighbour was driven on an older
    version of the map; ids divisible by 7 have no runs at all.
    """
    tags = [0, 7, 3]
    rows = [_row(i, 1000 - i, [tags[i % 3]]) for i in range(1, 301)]
    calls = {"search": 0, "replays": 0, "replay_dl": 0, "map_dl": 0}

    def get_json(url):
        assert "api/tracks" in url
        calls["search"] += 1
        after = int(url.split("after=")[1].split("&")[0]) if "after=" in url else 0
        page = [r for r in rows if r["TrackId"] > after][: tmx.PAGE]
        return {"Results": page, "More": page[-1]["TrackId"] < rows[-1]["TrackId"] if page else False}

    def track_replays(track_id, limit=tmx.PAGE):
        calls["replays"] += 1
        if track_id % 7 == 0:
            return []
        return [tmx.TmxReplay(track_id * 10 + k, 40000 + 100 * k, f"u{k}") for k in range(5)]

    def download_replay(replay, into, *, expect_uid):
        calls["replay_dl"] += 1
        into.mkdir(parents=True, exist_ok=True)
        path = into / f"tmx-{replay.replay_id}.Replay.Gbx"
        path.write_text(str(replay.replay_id))
        return path

    def classify(path):
        rid = int(path.read_text())
        track, k = divmod(rid, 10)
        return filter_mod.PAD if track % 5 == 0 and k == 2 else filter_mod.KEYBOARD

    def download_track(track, into, *, expect_uid=None):
        calls["map_dl"] += 1
        into.mkdir(parents=True, exist_ok=True)
        path = into / f"tmx-{track.track_id}.Challenge.Gbx"
        path.write_text("map")
        return path, f"uid{track.track_id}"

    def read_replay(path):
        track, k = divmod(int(path.read_text()), 10)
        return SimpleNamespace(map_uid="old" if track % 11 == 0 and k != 2 else f"uid{track}")

    monkeypatch.setattr(tmx, "_get_json", get_json)
    monkeypatch.setattr(tmx, "track_replays", track_replays)
    monkeypatch.setattr(tmx, "download_replay", download_replay)
    monkeypatch.setattr(tmx, "download_track", download_track)
    monkeypatch.setattr(tmx, "read_replay", read_replay)
    monkeypatch.setattr(filter_mod, "classify_replay", classify)
    paths = SimpleNamespace(maps=tmp_path / "maps", replays=tmp_path / "corpus3",
                            manifest=tmp_path / "corpus3.harvest.jsonl")
    return calls, paths


def _run(paths, **kw):
    args = dict(maps_into=paths.maps, replays_into=paths.replays, manifest=paths.manifest,
                quotas={7: 10, 3: 5}, max_scanned=10_000, log=lambda _: None)
    args.update(kw)
    return tmx.harvest_quota(**args)


def test_quota_stops_when_full_and_skips_other_tags(fake_tmx):
    calls, paths = fake_tmx
    state = _run(paths)
    assert state.stop_reason == "all quotas full"
    assert state.filled == {7: 10, 3: 5}
    assert calls["search"] == 1  # never paged past what it needed
    rows = [json.loads(line) for line in paths.manifest.read_text().splitlines()]
    assert {r["tag"] for r in rows} <= {7, 3}  # Race maps never touched
    assert calls["map_dl"] == 15


def test_pad_median_falls_back_to_neighbour(fake_tmx):
    calls, paths = fake_tmx
    _run(paths, quotas={7: 20})
    rows = {r["track"]: r for r in map(json.loads, paths.manifest.read_text().splitlines())}
    pad_maps = [r for t, r in rows.items() if t % 5 == 0 and r["outcome"] == "kept"]
    assert pad_maps and all(r["skipped"]["pad"] and r["replay"] % 10 != 2 for r in pad_maps)
    assert (paths.replays.parent / "corpus3.rejected" / "pad").is_dir()
    assert all(r["outcome"] == "no_replays" for t, r in rows.items() if t % 7 == 0)


def test_max_scanned_caps_the_walk(fake_tmx):
    calls, paths = fake_tmx
    state = _run(paths, quotas={7: 1000}, max_scanned=150)
    assert state.stop_reason == "scanned 150 maps"
    assert calls["search"] == 2


def test_resume_and_existing_maps_are_skipped(fake_tmx):
    calls, paths = fake_tmx
    paths.maps.mkdir()
    (paths.maps / "tmx-2.Challenge.Gbx").write_text("from an earlier corpus")
    first = _run(paths, quotas={7: 3})
    assert 2 not in [json.loads(x)["track"] for x in paths.manifest.read_text().splitlines()]
    downloads = calls["map_dl"]
    second = _run(paths, quotas={7: 6})  # resumes: 3 already filled
    assert first.filled[7] == 3 and second.filled[7] == 6
    assert calls["map_dl"] - downloads == 3


def test_dry_run_only_searches(fake_tmx):
    calls, paths = fake_tmx
    state = _run(paths, dry_run=True)
    assert state.filled == {7: 10, 3: 5}
    assert calls["replays"] == calls["replay_dl"] == calls["map_dl"] == 0
    assert not paths.manifest.exists()


def test_busy_site_backs_off_then_gives_up(monkeypatch):
    import urllib.error

    sleeps = []
    monkeypatch.setattr(tmx.time, "sleep", sleeps.append)

    def busy(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "busy", {}, None)

    monkeypatch.setattr(tmx.urllib.request, "urlopen", busy)
    with pytest.raises(tmx.TmxError):
        tmx._get("https://example.invalid/x")
    waits = [s for s in sleeps if s != tmx.COURTESY_DELAY]
    assert waits == [tmx.BACKOFF_S, 2 * tmx.BACKOFF_S, 4 * tmx.BACKOFF_S]


def test_have_ids_are_skipped(fake_tmx):
    calls, paths = fake_tmx
    state = _run(paths, quotas={7: 3}, have={4, 10})  # Tech maps are 1, 4, 7, 10, 13, ...
    tracks = [json.loads(x)["track"] for x in paths.manifest.read_text().splitlines()]
    assert tracks == [1, 7, 13, 16]  # 7 has no runs
    assert state.filled[7] == 3 and state.outcomes["already_have"] == 2


def test_old_version_runs_are_skipped_and_mapless_runs_leave_no_map(fake_tmx):
    calls, paths = fake_tmx
    _run(paths, quotas={0: 40, 7: 40, 3: 40}, max_scanned=120)
    rows = {r["track"]: r for r in map(json.loads, paths.manifest.read_text().splitlines())}
    # 55 and 110: pad median, then two neighbours on an old version -> nothing usable
    for t in (55, 110):
        assert rows[t]["outcome"] == "no_usable_run"
        assert rows[t]["skipped"] == {"pad": [t * 10 + 2], "old_version": [t * 10 + 3, t * 10 + 1]}
        assert not (paths.maps / f"tmx-{t}.Challenge.Gbx").exists()
    # 22: median fine (k=2 is never old), kept
    assert rows[22]["outcome"] == "kept" and rows[22]["replay"] == 222
    kept = {t for t, r in rows.items() if r["outcome"] == "kept"}
    assert {int(p.name[4:-14]) for p in paths.maps.iterdir()} == kept
    assert {int(p.name[4:-11]) // 10 for p in paths.replays.iterdir()} == kept
