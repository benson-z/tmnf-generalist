"""Per-run labels, built once from samples.jsonl / inputs.jsonl.

Alignment (verified over all 2483 ok runs in corpus2, see docs/DATA_REPORT.md):
sample row i is frame i of frames.mkv and sits at race time 50*i; input tick j
sits at race time 10*j. Sample i's keys equal tick 5i's keys. The action for
frame i is the 50 ms step the policy would hold after seeing it: ticks
5i .. 5i+4, labelled by majority (``actions.from_ticks``). The last frame has
fewer than five ticks after it when the run finished off the 50 ms grid, so it
gets no action label.

Everything a label is computed from (position, orientation, velocity) stays in
here. The only per-step observation is ``speed`` (the HUD speed, km/h).

Output: ``<work_dir>/index/<run>.npz`` plus ``<work_dir>/index/manifest.json``
with the run list, lengths, train/val split and target normalisation stats.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import actions
from ..config import DataConfig

FPS = 20
TICKS_PER_FRAME = 5


def heading(yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit forward and lateral vectors in the ground (x, z) plane.

    TMNF's yaw of -pi/2 faces -x (checked against the velocity of cars leaving
    the start line). Lateral is forward rotated by -90 degrees; which side is
    "positive" does not matter as long as mirroring negates it.
    """
    fwd = np.stack([np.sin(yaw), np.cos(yaw)], -1)
    lat = np.stack([np.cos(yaw), -np.sin(yaw)], -1)
    return fwd, lat


def respawn_jumps(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Frame indices k where the car teleported between frame k and k+1."""
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    expect = np.linalg.norm(vel[:-1], axis=1) / FPS
    return np.flatnonzero(step > np.maximum(10.0, 3 * expect + 5))


def labels_for_run(run_dir: Path, cfg: DataConfig) -> dict | None:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        return None
    rows = [json.loads(l) for l in (run_dir / "samples.jsonl").open(encoding="utf-8") if l.strip()]
    ticks = [json.loads(l) for l in (run_dir / "inputs.jsonl").open(encoding="utf-8") if l.strip()]
    n = len(rows)
    assert all(r["race_time"] == 50 * i for i, r in enumerate(rows)), run_dir
    assert all(t["race_time"] == 10 * j for j, t in enumerate(ticks)), run_dir

    # -- action labels: majority over the five ticks of each 50 ms step -------
    key = {k: np.array([t[k] for t in ticks], dtype=bool) for k in ("up", "down", "left", "right")}
    act = np.full(n, -1, dtype=np.int64)
    # Soft labels: how many of the step's 5 ticks fell in each of the 12 actions.
    tick_actions = np.zeros((n, actions.N_ACTIONS), dtype=np.uint8)
    full = min(n, len(ticks) // TICKS_PER_FRAME)
    if full:
        blk = {k: v[: full * 5].reshape(full, 5) for k, v in key.items()}
        act[:full] = actions.from_ticks(blk["up"], blk["down"], blk["left"], blk["right"])
        per_tick = actions.per_tick(blk["up"], blk["down"], blk["left"], blk["right"])  # (full, 5)
        for j in range(TICKS_PER_FRAME):
            np.add.at(tick_actions, (np.arange(full), per_tick[:, j]), 1)

    speed = np.array([r["speed_kmh"] for r in rows], dtype=np.float32)
    pos = np.array([r["position"] for r in rows], dtype=np.float64)
    vel = np.array([r["velocity"] for r in rows], dtype=np.float64)
    yaw = np.array([r["yaw_pitch_roll"][0] for r in rows], dtype=np.float64)

    # -- valid frames: drop the neighbourhood of respawns --------------------
    valid = act >= 0
    jumps = respawn_jumps(pos, vel)
    if len(jumps) and cfg.drop_respawn_runs:
        return {"dropped": "respawn", "name": run_dir.name}
    for k in jumps:
        lo = max(0, k + 1 - int(cfg.respawn_mask_before_s * FPS))
        hi = min(n, k + 1 + int(cfg.respawn_mask_after_s * FPS))
        valid[lo:hi] = False
    if meta.get("replay_respawns", 0) and cfg.drop_respawn_runs:
        return {"dropped": "respawn", "name": run_dir.name}

    # -- future path: waypoints in the car's own frame, and speed profile ----
    fwd, lat = heading(yaw)
    hs = [int(round(h * FPS)) for h in cfg.waypoint_horizons_s]
    # (lateral m, forward m, speed km/h[, height change m])
    path = np.zeros((n, len(hs), 4 if cfg.path_height else 3), dtype=np.float32)
    path_ok = np.zeros((n, len(hs)), dtype=bool)
    seg_break = np.zeros(n, dtype=np.int64)  # segment id, bumped at each respawn
    for k in jumps:
        seg_break[k + 1:] += 1
    for j, h in enumerate(hs):
        idx = np.arange(n - h)
        d = pos[idx + h][:, [0, 2]] - pos[idx][:, [0, 2]]
        path[idx, j, 0] = (d * lat[idx]).sum(1)
        path[idx, j, 1] = (d * fwd[idx]).sum(1)
        path[idx, j, 2] = speed[idx + h]
        if cfg.path_height:
            path[idx, j, 3] = pos[idx + h, 1] - pos[idx, 1]  # y is up
        path_ok[idx, j] = seg_break[idx + h] == seg_break[idx]

    # -- fixed-horizon progress: arc length along the run's own trajectory ---
    H = int(round(cfg.progress_horizon_s * FPS))
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    if len(jumps):
        step[jumps] = 0.0
    arc = np.concatenate([[0.0], np.cumsum(step)])
    progress = np.zeros(n, dtype=np.float32)
    progress_ok = np.zeros(n, dtype=bool)
    idx = np.arange(max(0, n - H))
    progress[idx] = arc[idx + H] - arc[idx]
    progress_ok[idx] = seg_break[idx + H] == seg_break[idx]

    return {
        "name": run_dir.name,
        "n": n,
        "action": act,
        "tick_actions": tick_actions,
        "valid": valid,
        "speed": speed,
        "path": path,
        "path_ok": path_ok,
        "progress": progress,
        "progress_ok": progress_ok,
    }


def _job(args: tuple[str, str, dict]) -> dict | None:
    run_dir, out_dir, cfg_dict = args
    cfg = DataConfig(**cfg_dict)
    lab = labels_for_run(Path(run_dir), cfg)
    if lab is None or "dropped" in lab:
        return lab
    np.savez(Path(out_dir) / f"{lab['name']}.npz", **{k: v for k, v in lab.items() if k not in ("name", "n")})
    ok = lab["valid"]
    return {
        "name": lab["name"], "n": lab["n"],
        # Per horizon: (H, 3) sums over frames with that horizon available.
        "path_sum": (lab["path"].astype(np.float64) * lab["path_ok"][..., None]).sum(0).tolist(),
        "path_sq": (lab["path"].astype(np.float64) ** 2 * lab["path_ok"][..., None]).sum(0).tolist(),
        "path_cnt": lab["path_ok"].sum(0).tolist(),
        "prog_sum": float(lab["progress"][lab["progress_ok"]].sum()),
        "prog_sq": float((lab["progress"][lab["progress_ok"]].astype(np.float64) ** 2).sum()),
        "prog_cnt": int(lab["progress_ok"].sum()),
        "actions": np.bincount(lab["action"][ok], minlength=actions.N_ACTIONS).tolist(),
    }


def is_val(name: str, fraction: float) -> bool:
    """Deterministic run-level split by name hash."""
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraction


def index_dir(cfg: DataConfig) -> Path:
    return Path(cfg.work_dir) / "index"


def build(cfg: DataConfig, log=print, workers: int = 16) -> dict:
    out = index_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in Path(cfg.corpus).iterdir() if p.is_dir())
    from dataclasses import asdict

    jobs = [(str(r), str(out), asdict(cfg)) for r in runs if (r / "meta.json").exists()]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=8))
    kept = [r for r in res if r and "dropped" not in r]
    dropped = [r["name"] for r in res if r and "dropped" in r]
    cnt = np.sum([r["path_cnt"] for r in kept], 0)[:, None]
    ps = np.sum([r["path_sum"] for r in kept], 0) / cnt
    pq = np.sum([r["path_sq"] for r in kept], 0) / cnt
    pcnt = sum(r["prog_cnt"] for r in kept)
    pm = sum(r["prog_sum"] for r in kept) / pcnt
    pv = sum(r["prog_sq"] for r in kept) / pcnt - pm**2
    act = np.sum([r["actions"] for r in kept], 0)
    manifest = {
        "corpus": cfg.corpus,
        "waypoint_horizons_s": cfg.waypoint_horizons_s,
        "progress_horizon_s": cfg.progress_horizon_s,
        "runs": [{"name": r["name"], "n": r["n"], "val": is_val(r["name"], cfg.val_fraction)} for r in kept],
        "dropped_respawn_runs": dropped,
        # Lateral is zero-mean by mirror symmetry; forward and speed are not.
        "path_mean": [[0.0, *m[1:]] for m in ps.tolist()],
        "path_height": cfg.path_height,
        "path_std": np.sqrt(np.maximum(pq - ps**2, 1e-6)).tolist(),
        "progress_mean": pm,
        "progress_std": float(np.sqrt(pv)),
        "action_counts": act.tolist(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"indexed {len(kept)} runs ({sum(r['val'] for r in manifest['runs'])} val), "
        f"dropped {len(dropped)} with respawns; frames {sum(r['n'] for r in kept)}")
    return manifest


def load_manifest(cfg: DataConfig) -> dict:
    return json.loads((index_dir(cfg) / "manifest.json").read_text())


def load_run(cfg: DataConfig, name: str) -> dict[str, np.ndarray]:
    with np.load(index_dir(cfg) / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}
