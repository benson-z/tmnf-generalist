"""Loader throughput: frames per second delivered to the GPU, uint8 -> device."""

from __future__ import annotations

import time

import torch

from ..config import Config
from .dataset import WindowLoader, split_runs
from .frames import nvdec_available
from .index import load_manifest


def run(cfg: Config, batches: int = 60) -> dict:
    manifest = load_manifest(cfg.data)
    train, _ = split_runs(manifest, cfg.data)
    loader = WindowLoader(cfg, train, manifest, epoch=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    it = iter(loader)
    obs, _ = next(it)  # worker start-up excluded
    frames_per_window = obs.frames.shape[1]
    t = time.perf_counter()
    n = 0
    for k in range(batches):
        obs, lab = next(it)
        obs.frames.to(dev, non_blocking=True)
        n += obs.frames.shape[0]
    torch.cuda.synchronize() if dev.type == "cuda" else None
    dt = time.perf_counter() - t
    return {
        "source": cfg.data.source,
        "decoder": "nvdec" if cfg.data.decoder != "cpu" and nvdec_available() else "cpu",
        "input_hw": list(obs.frames.shape[2:4]),
        "workers": cfg.data.loader_workers,
        "micro_batch": cfg.train.micro_batch,
        "windows_per_s": round(n / dt, 1),
        "frames_per_s": round(n * frames_per_window / dt, 1),
        "supervised_frames_per_s": round(n * cfg.data.window / dt, 1),
    }
