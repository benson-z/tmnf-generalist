"""Deliberate reversing in eval rollouts, from the per-step logs.

    python scripts/reverse_rate.py <eval dir> [<eval dir> ...]

A reverse is brake held without gas while the car moves backwards
(``local_speed`` forward < -1 m/s) for at least 0.5 s; a car bounced off a
wall with gas held does not count. Per map: rollouts that reversed, how many
of them took a checkpoint afterwards, and how many finished. Needs step logs
with ``local_speed`` (evals from 2026-09-30 on).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("evals", nargs="+")
ap.add_argument("--min-s", type=float, default=0.5, help="shortest reverse that counts")
args = ap.parse_args()


def reverses(rows: list[dict]) -> list[list[dict]]:
    """Runs of reversing steps (at most one missed 50 ms step inside a run)."""
    spells, run = [], []
    for rw in rows:
        if not (rw["name"].startswith(".B") and rw["local_speed"][2] < -1):
            continue
        if run and rw["t"] - run[-1]["t"] > 100:
            spells.append(run)
            run = []
        run.append(rw)
    if run:
        spells.append(run)
    return [s for s in spells if s[-1]["t"] - s[0]["t"] >= args.min_s * 1000 - 50]


for ev in map(Path, args.evals):
    for md in sorted(p for p in ev.iterdir() if (p / "rollouts.jsonl").exists()):
        ro = {r["rollout"]: r for r in map(json.loads, open(md / "rollouts.jsonl"))}
        n_ro = n_rev = n_cp = n_fin = 0
        for sf in sorted((md / "steps").glob("*.jsonl")):
            rows = [json.loads(line) for line in open(sf)]
            if not rows or "local_speed" not in rows[0]:
                print(f"{ev.name} {md.name}: no local_speed in the step logs; skipped")
                break
            n_ro += 1
            spells = reverses(rows)
            if not spells:
                continue
            r = ro.get(int(sf.stem.rsplit("_r", 1)[1]), {})
            end = spells[-1][-1]["t"] / 1000
            n_rev += 1
            n_cp += any(ct > end for ct in r.get("checkpoint_times_s") or [])
            n_fin += bool(r.get("finished"))
            for s in spells:
                print(f"    {sf.stem[-3:]} {s[0]['t'] / 1000:6.1f}-{s[-1]['t'] / 1000:6.1f} s, "
                      f"back up to {-min(rw['local_speed'][2] for rw in s) * 3.6:.0f} km/h; "
                      f"end {r.get('end_reason')}, {r.get('checkpoints')} CPs")
        else:
            print(f"{ev.name} {md.name}: {n_rev}/{n_ro} rollouts reversed; {n_cp} took a CP after, {n_fin} finished")
