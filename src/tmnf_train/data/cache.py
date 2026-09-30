"""Optional uint8 memmap cache of every run at the training resolution.

``data.source: memmap`` reads it instead of decoding video. Built with the
same ``frames.to_input`` as every other path. Sizes for corpus2 (2.06M
frames): 320x240 is ~475 GB, 160x120 ~119 GB, so it goes on Z: and is only
worth it at the reduced resolution.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import Config, DataConfig
from . import frames as frames_mod
from .dataset import input_hw
from .index import load_manifest


def cache_dir(cfg: DataConfig) -> Path:
    h, w = input_hw(cfg)
    crop = f"_crop{cfg.crop_rows[0]}-{cfg.crop_rows[1]}" if cfg.crop_rows else ""
    return Path(cfg.work_dir) / "cache" / f"{w}x{h}{crop}"


def build(cfg: Config, log=print, workers: int = 8) -> None:
    d = cfg.data
    out = cache_dir(d)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_manifest(d)["runs"]
    h, w = input_hw(d)
    total = sum(r["n"] for r in runs) * h * w * 3
    log(f"cache {out}: {len(runs)} runs, {total / 1e9:.1f} GB")

    def one(r: dict) -> None:
        path = out / f"{r['name']}.u8"
        if path.exists() and path.stat().st_size == r["n"] * h * w * 3:
            return
        src = frames_mod.decode_run(Path(d.corpus) / r["name"] / "frames.mkv", d.decoder, expected=r["n"])
        tmp = path.with_suffix(".tmp")
        frames_mod.to_input(src, d.downsample, d.crop_rows).tofile(tmp)
        tmp.replace(path)

    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, runs)):
            if k % 100 == 0:
                log(f"  {k}/{len(runs)}")
