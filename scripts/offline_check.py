"""Offline timing checks of a checkpoint on the validation runs (no game).

    TMNF_STORAGE=~/tmnf-ml .venv/bin/python scripts/offline_check.py <ckpt.pt> [<ckpt.pt> ...]

For each checkpoint, over every validation run (non-overlapping windows, as
the earlier Windows analysis did):

  steer onset / release   mean P(steer in the demo's direction) around the
                          demo's steering onsets and releases, and the first
                          step k where it passes 0.5 (demo onset is k = 0;
                          v2_soft e4: about +2 on both)
  drift taps              mean P(brake) at T=1 around the demo's drift taps
                          (brake + steer starting above 150 km/h, slip > 10
                          deg within 0.5 s; v2_soft e4: 0.21 at the tap,
                          0.56 one step later, median 0.11 at the tap)

With an action-chunk head, the same curves are repeated for the chunk
head's t+1 and t+2 predictions acting at step t (what ``eval.action_source
chunk1/chunk2`` would do). Results go to stdout and, as JSON, next to the
checkpoint's run (``<run>/offline/<ckpt>.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tmnf_train.config import load  # noqa: E402
from tmnf_train.data.dataset import Observation, RunFrames, frame_indices  # noqa: E402
from tmnf_train.data.index import load_manifest, load_run  # noqa: E402
from tmnf_train.policy_model import ModelPolicy  # noqa: E402

BRAKE = np.array([(a // 3) % 2 == 1 for a in range(12)])
GAS_BRAKE = np.array([a // 6 == 1 and (a // 3) % 2 == 1 for a in range(12)])
KS = range(-4, 7)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def predict(model, cfg, name: str, n: int, rf: RunFrames, lab: dict, dev) -> dict[str, np.ndarray]:
    """Per-frame 12-way probabilities: policy head, and chunk head steps 1..2."""
    T = cfg.data.window
    heads = {"policy": np.full((n, 12), np.nan, np.float32)}
    if model.n_chunk:
        for j in range(min(2, model.n_chunk)):
            heads[f"chunk{j + 1}"] = np.full((n, 12), np.nan, np.float32)
    fr = rf.load(name, n)
    starts = list(range(0, n - T + 1, T))
    for i in range(0, len(starts), 16):
        ss = starts[i:i + 16]
        idx = np.stack([frame_indices(s, cfg.data) for s in ss])
        obs = Observation(torch.from_numpy(np.asarray(fr[idx])).to(dev),
                          torch.from_numpy(np.stack([lab["speed"][s:s + T] for s in ss])).float().to(dev))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(obs)
        pol = softmax(out["logits"].float().cpu().numpy())
        ch = softmax(out["chunk_logits"].float().cpu().numpy()) if "chunk_logits" in out else None
        for b, s in enumerate(ss):
            heads["policy"][s:s + T] = pol[b]
            for j in range(len(heads) - 1):
                heads[f"chunk{j + 1}"][s:s + T] = ch[b, :, j]
    return heads


def demo_motion(corpus: str, name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    smp = [json.loads(line) for line in open(Path(corpus) / name / "samples.jsonl", encoding="utf-8") if line.strip()][:n]
    vel = np.array([s["velocity"] for s in smp])
    yaw = np.array([s["yaw_pitch_roll"][0] for s in smp])
    slip = np.abs(np.degrees(np.angle(np.exp(1j * (np.arctan2(vel[:, 0], vel[:, 2]) - yaw)))))
    return slip, np.array([s["speed_kmh"] for s in smp])


def first_over(curve: dict[int, float], thr: float = 0.5):
    ks = [k for k in sorted(curve) if curve[k] > thr]
    return ks[0] if ks else None


def check(ck: str, cfg, manifest, dev) -> dict:
    policy = ModelPolicy.from_checkpoint(ck, device=str(dev))
    model = policy.model
    rf = RunFrames(cfg.data)
    runs = [r for r in manifest["runs"] if r["val"]]
    onset: dict[str, dict[int, list]] = {}
    release: dict[str, dict[int, list]] = {}
    taps: dict[str, dict[int, list]] = {}
    tap_gas: dict[str, dict[int, list]] = {}
    demo_gas: list[float] = []
    for r in runs:
        name, n = r["name"], r["n"]
        lab = load_run(cfg.data, name)
        heads = predict(model, cfg, name, n, rf, lab, dev)
        slip, spd = demo_motion(cfg.data.corpus, name, n)
        a = lab["action"]
        st = np.where(a >= 0, a % 3, -1)
        br = (a >= 0) & ((a // 3) % 2 == 1)
        for h, P in heads.items():
            on, off, tp = (onset.setdefault(h, {k: [] for k in KS}), release.setdefault(h, {k: [] for k in KS}),
                           taps.setdefault(h, {k: [] for k in KS}))
            tg = tap_gas.setdefault(h, {k: [] for k in KS})
            ok = ~np.isnan(P).any(1)
            for t in range(4, n - 7):
                if not ok[t - 4:t + 7].all():
                    continue
                d = st[t]
                if d in (0, 2) and (st[t - 3:t] == 1).all() and (st[t:t + 3] == d).all():
                    for k in KS:
                        on[k].append(P[t + k, d::3].sum())
                if d == 1 and st[t - 1] in (0, 2) and (st[t - 3:t] == st[t - 1]).all() and (st[t:t + 3] == 1).all():
                    for k in KS:
                        off[k].append(P[t + k, st[t - 1]::3].sum())
                if (br[t] and not br[t - 1] and st[t] in (0, 2) and spd[t] > 150 and t + 11 <= n
                        and slip[t:t + 11].max() > 10):
                    for k in KS:
                        tp[k].append(P[t + k, BRAKE].sum())
                        tg[k].append(P[t + k, GAS_BRAKE].sum() / max(P[t + k, BRAKE].sum(), 1e-6))
                    if h == "policy":
                        k_end = t
                        while k_end < n and br[k_end]:
                            k_end += 1
                        demo_gas.append(float((a[t:k_end] // 6 == 1).mean()))
    res = {"checkpoint": Path(ck).stem, "heads": {}, "demo_gas_in_tap": float(np.mean(demo_gas)) if demo_gas else None}
    for h in onset:
        on = {k: float(np.mean(v)) for k, v in onset[h].items()}
        off = {k: float(np.mean(v)) for k, v in release[h].items()}
        tp = {k: float(np.mean(v)) for k, v in taps[h].items()}
        res["heads"][h] = {
            "onsets": len(onset[h][0]), "releases": len(release[h][0]), "taps": len(taps[h][0]),
            "onset_curve": on, "onset_cross_step": first_over(on),
            # Release: the held direction's probability falling below 0.5.
            "release_curve": off, "release_cross_step": next((k for k in sorted(off) if off[k] < 0.5), None),
            "tap_curve": tp, "tap_p_brake_at_tap": tp[0], "tap_p_brake_next": tp[1],
            "tap_p_brake_median_at_tap": float(np.median(taps[h][0])) if taps[h][0] else None,
            # Of the brake probability at a demo tap, the share that keeps gas held.
            "tap_p_gas_given_brake": {k: float(np.mean(v)) for k, v in tap_gas[h].items()},
        }
    return res


def main() -> None:
    cfg = load(REPO / "configs/v2_chunk_path3d_c23.yaml", ["data.loader_workers=4"])
    manifest = load_manifest(cfg.data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ck in sys.argv[1:]:
        res = check(ck, cfg, manifest, dev)
        print(f"\n== {res['checkpoint']}")
        for h, v in res["heads"].items():
            print(f"  [{h}] onsets {v['onsets']}, releases {v['releases']}, taps {v['taps']}")
            print(f"    steer onset crosses 0.5 at k={v['onset_cross_step']}  "
                  f"{ {k: round(x, 3) for k, x in v['onset_curve'].items()} }")
            print(f"    release drops below 0.5 at k={v['release_cross_step']}  "
                  f"{ {k: round(x, 3) for k, x in v['release_curve'].items()} }")
            print(f"    P(brake) at tap {v['tap_p_brake_at_tap']:.3f}, next step {v['tap_p_brake_next']:.3f}, "
                  f"median at tap {v['tap_p_brake_median_at_tap']:.3f}")
            g = v["tap_p_gas_given_brake"]
            print(f"    P(gas | brake) at tap {g[0]:.3f}, k=1..3 {g[1]:.3f} {g[2]:.3f} {g[3]:.3f} "
                  f"(demo holds gas in {res['demo_gas_in_tap']:.0%} of tap steps)")
        out = Path(ck).parents[1] / "offline" / f"{res['checkpoint']}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
