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
ap.add_argument("--demos", type=int, default=0, help="also measure N random corpus runs ($TMNF_STORAGE)")
args = ap.parse_args()


def tap_timing(a: np.ndarray, spd: np.ndarray) -> tuple[list[int], list[int]]:
    """Drift taps (brake on while steering, >150 km/h): their lengths in
    steps, and steps since the current steer direction was turned in
    (one-step gaps filled)."""
    br = (a // 3) % 2 == 1
    st = (a % 3).copy()
    # Sampling flicker: a one-step gap in a brake hold or a steer direction
    # counts as held, so a flicker neither splits a tap nor restarts a turn.
    for t in range(1, len(a) - 1):
        if not br[t] and br[t - 1] and br[t + 1]:
            br[t] = True
        if st[t] != st[t - 1] and st[t - 1] == st[t + 1]:
            st[t] = st[t - 1]
    lens, lags = [], []
    for t in range(1, len(a)):
        if br[t] and not br[t - 1] and st[t] != 1 and spd[t] > 150:
            k = t
            while k < len(a) and br[k]:
                k += 1
            lens.append(k - t)
            j = t
            while j > 0 and st[j - 1] == st[t]:
                j -= 1
            lags.append(t - j)
    return lens, lags


def summary(lens: list[int], lags: list[int]) -> str:
    if not lens:
        return "no taps"
    L, G = np.array(lens), np.array(lags)
    return (f"tap length median {np.median(L):.0f} steps (1 step {np.mean(L == 1):.2f}, >=4 {np.mean(L >= 4):.2f}); "
            f"turn-in to tap median {np.median(G):.0f} steps (0-1 {np.mean(G <= 1):.2f}, 5-10 {np.mean((G >= 5) & (G <= 10)):.2f})")

for ev in args.evals:
    taps = steps_fast = brake_fast = slide = n150 = 0
    all_lens, all_lags = [], []
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
        ln, lg = tap_timing(a, spd)
        all_lens += ln
        all_lags += lg
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
    print("   ", summary(all_lens, all_lags))

if args.demos:
    import os
    import random

    root = Path(os.path.expanduser(os.environ["TMNF_STORAGE"])) / "train_data" / "corpus2"
    idx = Path(os.path.expanduser(os.environ["TMNF_STORAGE"])) / "work" / "index"
    runs = sorted(p.stem for p in idx.glob("*.npz"))
    random.seed(0)
    lens, lags = [], []
    for name in random.sample(runs, min(args.demos, len(runs))):
        z = np.load(idx / f"{name}.npz")
        a, spd = z["action"], z["speed"]
        ok = a >= 0
        ln, lg = tap_timing(np.where(ok, a, 7), spd)  # unlabelled -> gas, no brake, no steer
        lens += ln
        lags += lg
    print(f"demos ({args.demos} runs):", summary(lens, lags))
