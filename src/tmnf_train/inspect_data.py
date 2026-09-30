"""Step 1: look at a recorded corpus before building anything on it.

Reads every run directory written by ``tmnf-collect collect`` and reports the
layout, how the three streams (frames, 20 Hz samples, 100 Hz input ticks) line
up, run lengths, map repeats, per-key press/gap histograms in ticks, byte
counts, and a measured H.265 re-encode projection.

    python -m tmnf_train.inspect_data Z:/application_storage/tmnf-ml/train_data/corpus2 \
        --out docs/data_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

KEYS = ("up", "down", "left", "right")




def _press_stats(bits: np.ndarray) -> tuple[list[int], list[int]]:
    """Press durations and inter-press gaps (both in ticks) of one key."""
    b = bits.astype(np.int8)
    edges = np.diff(np.concatenate([[0], b, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    presses = (ends - starts).tolist()
    gaps = (starts[1:] - ends[:-1]).tolist()
    return presses, gaps


def inspect_run(run_dir: str) -> dict:
    d = Path(run_dir)
    out: dict = {"name": d.name, "files": sorted(p.name for p in d.iterdir())}
    out["bytes"] = {p.name: p.stat().st_size for p in d.iterdir()}
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        out["status"] = "no_meta"
        return out
    out["status"] = meta.get("status")
    out["map_uid"] = meta.get("map_uid")
    out["map_file"] = Path(meta.get("map_file") or "").name
    out["replay_respawns"] = meta.get("replay_respawns")
    out["finish_ms"] = meta.get("recorded_finish_time_ms")
    out["meta_keys"] = sorted(meta)

    rows = [json.loads(l) for l in (d / "samples.jsonl").open(encoding="utf-8") if l.strip()]
    ticks = [json.loads(l) for l in (d / "inputs.jsonl").open(encoding="utf-8") if l.strip()]
    out["n_samples"] = len(rows)
    out["n_ticks"] = len(ticks)
    if rows:
        out["sample_keys"] = sorted(rows[0])
    if ticks:
        out["tick_keys"] = sorted(ticks[0])
    st = np.array([r["race_time"] for r in rows])
    tt = np.array([t["race_time"] for t in ticks])
    out["sample_i_is_index"] = bool(all(r["i"] == k for k, r in enumerate(rows)))
    out["sample_times_are_50i"] = bool(len(st) and np.array_equal(st, 50 * np.arange(len(st))))
    out["tick_times_are_10j"] = bool(len(tt) and np.array_equal(tt, 10 * np.arange(len(tt))))
    out["render_lag_nonzero"] = int(sum(r["render_race_time"] != r["race_time"] for r in rows))
    out["last_sample_ms"] = int(st[-1]) if len(st) else None
    out["last_tick_ms"] = int(tt[-1]) if len(tt) else None

    # Do sample keys equal the tick keys at the same race time?
    tick_by_time = {t["race_time"]: t for t in ticks}
    mism = 0
    for r in rows:
        t = tick_by_time.get(r["race_time"])
        if t is None or any(t[k] != r[k] for k in KEYS):
            mism += 1
    out["sample_tick_key_mismatch"] = mism

    bits = {k: np.array([t[k] for t in ticks], dtype=bool) for k in KEYS}
    out["presses"] = {}
    out["gaps"] = {}
    for k in KEYS:
        p, g = _press_stats(bits[k])
        out["presses"][k] = p
        out["gaps"][k] = g
    out["both_lr_ticks"] = int((bits["left"] & bits["right"]).sum())
    out["both_ud_ticks"] = int((bits["up"] & bits["down"]).sum())
    out["n_ticks_total"] = len(ticks)
    # How much steering is lost if a 50 ms step is labelled by the key at its
    # first tick instead of by majority over its 5 ticks.
    n5 = len(ticks) // 5
    if n5:
        mix = 0
        for k in KEYS:
            blk = bits[k][: n5 * 5].reshape(n5, 5).sum(1)
            mix += int(((blk > 0) & (blk < 5)).sum())
        out["mixed_50ms_blocks"] = mix
        out["n_50ms_blocks"] = n5

    # Respawns: large position jumps between consecutive 20 Hz samples.
    pos = np.array([r["position"] for r in rows], dtype=np.float64)
    if len(pos) > 1:
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        vel = np.array([r["velocity"] for r in rows], dtype=np.float64)
        expect = np.linalg.norm(vel[:-1], axis=1) * 0.05
        jumps = np.flatnonzero(step > np.maximum(10.0, 3 * expect + 5))
        out["position_jumps"] = [int(j) for j in jumps]
        out["max_speed_kmh"] = int(max(r["speed_kmh"] for r in rows))
        # Stationary stretches (speed < 5 km/h) after the first second.
        slow = np.array([r["speed_kmh"] < 5 for r in rows])
        out["slow_frames_after_1s"] = int(slow[20:].sum())
    cps = [r["checkpoints"] for r in rows]
    out["checkpoints_final"] = cps[-1] if cps else None
    out["finished_rows"] = int(sum(r["finished"] for r in rows))
    out["first_finished_row"] = next((k for k, r in enumerate(rows) if r["finished"]), None)
    return out


def _hist(values: list[int], edges: list[int]) -> dict:
    counts, _ = np.histogram(values, bins=edges + [10**9])
    labels = [f"{a}-{b - 1}" if b - a > 1 else f"{a}" for a, b in zip(edges, edges[1:] + [10**9])]
    labels[-1] = f">={edges[-1]}"
    return dict(zip(labels, counts.tolist()))


def _pct(values: list[int]) -> dict:
    if not values:
        return {}
    a = np.asarray(values)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 2),
        **{f"p{q}": float(np.percentile(a, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
        "max": int(a.max()),
        "frac_1_tick": round(float((a == 1).mean()), 4),
        "frac_lt_5_ticks": round(float((a < 5).mean()), 4),
    }


def ffprobe_video(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries",
         "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_packets,has_b_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)["streams"][0]
    kf = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    keys = [i for i, f in enumerate(kf) if f.startswith("K")]
    info["keyframe_indices_head"] = keys[:6]
    info["keyframe_spacing"] = sorted(set(np.diff(keys).tolist())) if len(keys) > 1 else []
    return info


def reencode_projection(runs: list[Path], sample_n: int, gop: int) -> dict:
    """Re-encode a few runs with x265 and compare sizes (no files kept)."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for run in runs[:sample_n]:
            src = run / "frames.mkv"
            dst = Path(tmp) / f"{run.name}.mkv"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an",
                 "-c:v", "libx265", "-preset", "medium", "-crf", "20",
                 "-x265-params", f"log-level=error:scenecut=0:open-gop=0:keyint={gop}:min-keyint={gop}:bframes=0",
                 "-pix_fmt", "yuv420p", str(dst)],
                check=True,
            )
            rows.append((src.stat().st_size, dst.stat().st_size))
    a = sum(r[0] for r in rows)
    b = sum(r[1] for r in rows)
    return {"runs": len(rows), "source_bytes": a, "reencoded_bytes": b, "ratio": round(b / a, 3)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/data_report.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--probe", type=int, default=20, help="runs to ffprobe")
    ap.add_argument("--reencode", type=int, default=6, help="runs to trial re-encode")
    args = ap.parse_args(argv)

    dirs = sorted(p for p in args.corpus.iterdir() if p.is_dir())
    with ProcessPoolExecutor(args.workers) as ex:
        per = list(ex.map(inspect_run, [str(d) for d in dirs], chunksize=8))

    ok = [r for r in per if r.get("status") == "ok"]
    report: dict = {"corpus": str(args.corpus), "run_dirs": len(dirs)}
    report["status"] = dict(Counter(r.get("status") for r in per))
    report["file_sets"] = {",".join(k): v for k, v in Counter(tuple(r["files"]) for r in per).items()}
    report["meta_keys"] = ok[0]["meta_keys"]
    report["sample_keys"] = ok[0]["sample_keys"]
    report["tick_keys"] = ok[0]["tick_keys"]

    total_bytes = sum(sum(r["bytes"].values()) for r in per)
    by_file = Counter()
    for r in per:
        by_file.update(r["bytes"])
    ok_bytes = sum(sum(r["bytes"].values()) for r in ok)
    report["bytes"] = {"total_all_dirs": total_bytes, "ok_runs": ok_bytes, "by_file": dict(by_file)}

    report["alignment"] = {
        "runs_sample_i_is_index": sum(r["sample_i_is_index"] for r in ok),
        "runs_sample_times_50i": sum(r["sample_times_are_50i"] for r in ok),
        "runs_tick_times_10j": sum(r["tick_times_are_10j"] for r in ok),
        "runs_with_render_lag": sum(r["render_lag_nonzero"] > 0 for r in ok),
        "runs_ticks_eq_5x_samples_minus_4": sum(r["n_ticks"] == 5 * r["n_samples"] - 4 for r in ok),
        "runs_last_tick_eq_last_sample": sum(r["last_tick_ms"] == r["last_sample_ms"] for r in ok),
        "runs_last_sample_eq_finish": sum(r["last_sample_ms"] == r["finish_ms"] for r in ok),
        "runs_finish_multiple_of_50": sum(r["finish_ms"] % 50 == 0 for r in ok),
        "sample_rows_key_mismatch_vs_tick": sum(r["sample_tick_key_mismatch"] for r in ok),
        "finished_rows_per_run": dict(Counter(r["finished_rows"] for r in ok)),
    }

    secs = np.array([r["finish_ms"] / 1000 for r in ok])
    report["run_length_s"] = {
        "runs": len(ok), "total_hours": round(float(secs.sum()) / 3600, 3),
        "min": float(secs.min()), "p5": float(np.percentile(secs, 5)), "p25": float(np.percentile(secs, 25)),
        "median": float(np.median(secs)), "p75": float(np.percentile(secs, 75)),
        "p95": float(np.percentile(secs, 95)), "max": float(secs.max()),
        "hist_10s": _hist(secs.astype(int).tolist(), list(range(0, 190, 10))),
    }
    report["total_frames_ok"] = int(sum(r["n_samples"] for r in ok))
    report["total_ticks_ok"] = int(sum(r["n_ticks"] for r in ok))

    uids = Counter(r["map_uid"] for r in ok)
    files = Counter(r["map_file"] for r in ok)
    report["maps"] = {
        "distinct_uids": len(uids), "uid_repeat_hist": dict(Counter(uids.values())),
        "distinct_map_files": len(files), "file_repeat_hist": dict(Counter(files.values())),
        "b01_race_present": any("B01" in f for f in files),
    }
    report["respawns"] = {
        "replay_respawns_hist": dict(Counter(r["replay_respawns"] for r in ok)),
        "runs_with_position_jumps": sum(bool(r.get("position_jumps")) for r in ok),
        "position_jumps_total": sum(len(r.get("position_jumps", [])) for r in ok),
        "examples": [(r["name"], r["position_jumps"][:5]) for r in ok if r.get("position_jumps")][:10],
    }
    report["slow_frames_after_1s_total"] = sum(r.get("slow_frames_after_1s", 0) for r in ok)

    edges = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200, 500]
    keys_report = {}
    for k in KEYS:
        presses = [x for r in ok for x in r["presses"][k]]
        gaps = [x for r in ok for x in r["gaps"][k]]
        held = sum(sum(r["presses"][k]) for r in ok)
        keys_report[k] = {
            "held_fraction": round(held / report["total_ticks_ok"], 4),
            "press_ticks": _pct(presses), "press_hist": _hist(presses, edges),
            "gap_ticks": _pct(gaps), "gap_hist": _hist(gaps, edges),
        }
    report["keys"] = keys_report
    report["both_left_right_ticks"] = sum(r["both_lr_ticks"] for r in ok)
    report["both_up_down_ticks"] = sum(r["both_ud_ticks"] for r in ok)
    report["mixed_50ms_blocks_frac"] = round(
        sum(r.get("mixed_50ms_blocks", 0) for r in ok) / max(1, 4 * sum(r.get("n_50ms_blocks", 0) for r in ok)), 4)

    probe_dirs = [args.corpus / r["name"] for r in ok[:: max(1, len(ok) // max(1, args.probe))]][: args.probe]
    report["video_probe"] = [dict(run=d.name, **ffprobe_video(d / "frames.mkv")) for d in probe_dirs]
    report["video_frames_match_rows"] = sum(
        int(p["nb_read_packets"]) == next(r["n_samples"] for r in ok if r["name"] == p["run"])
        for p in report["video_probe"])
    if args.reencode:
        report["reencode_trial"] = reencode_projection(probe_dirs, args.reencode, gop=20)
        report["reencode_trial"]["projected_total_frames_bytes"] = int(
            by_file["frames.mkv"] * report["reencode_trial"]["ratio"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("video_probe",)}, indent=1))


if __name__ == "__main__":
    main()
