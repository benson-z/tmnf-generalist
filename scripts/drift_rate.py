"""Drift behaviour in eval rollouts, from the per-step logs.

    python scripts/drift_rate.py <eval dir> [<eval dir> ...]

Per eval (all maps or --map): drift taps per minute (brake turning on while
steering, above 150 km/h), share of steps with brake above 100 km/h, and the
share of steps above 150 km/h where the car slides (velocity heading from
positions vs yaw, over 10 deg). Demos (corpus2): 7.7 taps/min, 4.5% brake.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("evals", nargs="+")
ap.add_argument("--map", default=None, help="map label, e.g. tmx-10460245")
args = ap.parse_args()

for ev in args.evals:
    taps = steps_fast = brake_fast = slide = n150 = 0
    minutes = 0.0
    files = sorted(Path(ev).glob(f"{args.map or '*'}/steps/*.jsonl"))
    for f in files:
        rows = [json.loads(line) for line in f.open() if line.strip()]
        if len(rows) < 3:
            continue
        a = np.array([r["action"] for r in rows])
        spd = np.array([r["speed"] for r in rows], float)
        pos = np.array([r["pos"] for r in rows], float)
        yaw = np.array([r["yaw"] for r in rows], float)
        br = (a // 3) % 2 == 1
        st = a % 3 != 1
        on = br[1:] & ~br[:-1] & st[1:] & (spd[1:] > 150)
        taps += int(on.sum())
        minutes += len(rows) * 0.05 / 60
        fast = spd > 100
        steps_fast += int(fast.sum())
        brake_fast += int((br & fast).sum())
        d = pos[1:] - pos[:-1]
        head = np.arctan2(d[:, 0], d[:, 2])
        sl = np.abs(np.degrees(np.angle(np.exp(1j * (head - yaw[1:])))))
        m = (spd[1:] > 150) & (np.hypot(d[:, 0], d[:, 2]) > 0.5)
        n150 += int(m.sum())
        slide += int((sl[m] > 10).sum())
    print(f"{Path(ev).name}: rollouts {len(files)}, {minutes:.1f} min driven; "
          f"drift taps/min {taps / max(minutes, 1e-9):.2f}; brake share >100 km/h {brake_fast / max(steps_fast, 1):.3f}; "
          f"sliding share >150 km/h {slide / max(n150, 1):.3f}")
