"""Contiguous 16-frame windows, as batches, from the corpus.

The observation / label boundary is explicit here: a batch is an
``(Observation, Labels)`` pair. ``Observation`` carries only frames and the
per-step HUD speed; everything derived from car state (actions, future path,
progress) lives in ``Labels`` and never reaches the model's forward pass.

How an epoch is read:

* An epoch plan (seeded by ``(seed, epoch)``) permutes the training runs, gives
  each a random phase, and cuts it into non-overlapping windows, so an epoch is
  one pass over the data. Each window is mirrored with ``mirror_prob``.
* Runs are grouped into blocks of ``BLOCK_RUNS``; windows are shuffled within
  a block. A block's runs are decoded whole, one run per worker item (the
  source is H.265 with a GOP of 40, so whole-run decoding avoids re-decoding
  GOP prefixes per window).
* Batch order is a pure function of the plan, so resuming at optimizer step
  ``k`` skips exactly the windows already consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import actions
from ..config import Config, DataConfig
from . import frames as frames_mod
from . import index as index_mod

BLOCK_RUNS = 12


class Observation(NamedTuple):
    """Everything the model may see. Nothing else is allowed in here."""

    frames: torch.Tensor  # (B, C + T + 1, H, W, 3) uint8: context frames, then prev frame, then T window frames
    speed: torch.Tensor  # (B, T) float32 km/h, HUD speed at each window frame


class Labels(NamedTuple):
    """Training targets, from car state and inputs. Never a model input."""

    action: torch.Tensor  # (B, T) int64, -1 where unlabelled
    action_soft: torch.Tensor  # (B, T, 12) float32: share of the step's 5 ticks per action (0 where unlabelled)
    path: torch.Tensor  # (B, T, K, 3) float32 normalised (lateral, forward, speed)
    path_ok: torch.Tensor  # (B, T, K) bool
    progress: torch.Tensor  # (B, T) float32 normalised
    progress_ok: torch.Tensor  # (B, T) bool
    chunk_soft: torch.Tensor  # (B, T, N, 12) float32: soft actions of steps t+1 .. t+N (N = data.action_chunk)
    chunk_ok: torch.Tensor  # (B, T, N) bool


OBSERVATION_FIELDS = ("frames", "speed")


@dataclass(frozen=True)
class WindowSpec:
    run: str
    start: int
    mirror: bool


def frame_indices(start: int, cfg: DataConfig) -> np.ndarray:
    """Source frame indices for one window: context, prev, then T frames."""
    s = cfg.frame_stride
    ctx = [start - o for o in cfg.context_offsets]
    idx = np.array(ctx + [start - s] + [start + k * s for k in range(cfg.window)])
    return np.clip(idx, 0, None)


def window_span(cfg: DataConfig) -> int:
    return (cfg.window - 1) * cfg.frame_stride + 1


def epoch_plan(runs: list[dict], cfg: DataConfig, seed: int, epoch: int) -> list[list[WindowSpec]]:
    """Blocks of windows for one epoch; deterministic in (seed, epoch)."""
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(len(runs))
    span = window_span(cfg)
    step = cfg.window * cfg.frame_stride
    blocks: list[list[WindowSpec]] = []
    for b in range(0, len(order), BLOCK_RUNS):
        block: list[WindowSpec] = []
        for r in order[b : b + BLOCK_RUNS]:
            run = runs[r]
            phase = int(rng.integers(0, step))
            for start in range(phase, run["n"] - span + 1, step):
                block.append(WindowSpec(run["name"], start, bool(rng.random() < cfg.mirror_prob)))
        rng.shuffle(block)
        blocks.append(block)
    if cfg.windows_per_epoch:
        kept, total = [], 0
        for blk in blocks:
            if total >= cfg.windows_per_epoch:
                break
            kept.append(blk[: cfg.windows_per_epoch - total])
            total += len(kept[-1])
        blocks = kept
    return blocks


def mirror_window(frames: np.ndarray, action: np.ndarray, path: np.ndarray,
                  soft: np.ndarray | None = None):
    """Horizontal flip: image left-right, steer labels swapped, lateral negated.

    ``soft`` (..., 12) per-action tick shares are permuted the same way.
    Returns (frames, action, path) or, with ``soft``, (frames, action, path, soft).
    """
    frames = frames[..., :, ::-1, :]
    act = np.where(action >= 0, actions.MIRROR[np.clip(action, 0, None)], action)
    path = path.copy()
    path[..., 0] = -path[..., 0]
    if soft is None:
        return np.ascontiguousarray(frames), act, path
    # MIRROR is an involution, so gathering by it moves mass from a to MIRROR[a].
    return np.ascontiguousarray(frames), act, path, soft[..., actions.MIRROR]


class RunFrames:
    """Frames of one run at the training resolution, from video or memmap."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def cache_path(self, name: str) -> Path:
        from .cache import cache_dir

        return cache_dir(self.cfg) / f"{name}.u8"

    def load(self, name: str, n: int) -> np.ndarray:
        cfg = self.cfg
        if cfg.source == "memmap":
            h, w = input_hw(cfg)
            return np.memmap(self.cache_path(name), dtype=np.uint8, mode="r", shape=(n, h, w, 3))
        return frames_mod.decode_run_input(Path(cfg.corpus) / name / "frames.mkv", cfg.downsample, cfg.crop_rows,
                                           cfg.decoder, expected=n)


def input_hw(cfg: DataConfig) -> tuple[int, int]:
    top, bottom = cfg.crop_rows or (0, frames_mod.SRC_H)
    return (bottom - top) // cfg.downsample, frames_mod.SRC_W // cfg.downsample


class Normaliser:
    def __init__(self, manifest: dict):
        self.path_mean = np.array(manifest["path_mean"], dtype=np.float32)
        self.path_std = np.array(manifest["path_std"], dtype=np.float32)
        self.prog_mean = float(manifest["progress_mean"])
        self.prog_std = float(manifest["progress_std"])


class RunWindows(Dataset):
    """Item i = all windows of one run in a block, fully materialised."""

    def __init__(self, cfg: DataConfig, manifest: dict, items: list[tuple[str, int, list[WindowSpec]]]):
        self.cfg = cfg
        self.items = items
        self.norm = Normaliser(manifest)
        self.frames = RunFrames(cfg)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        name, n, specs = self.items[i]
        cfg = self.cfg
        lab = index_mod.load_run(cfg, name)
        all_frames = self.frames.load(name, n)
        out = {"frames": [], "speed": [], "action": [], "action_soft": [], "path": [], "path_ok": [], "progress": [], "progress_ok": [],
               "chunk_soft": [], "chunk_ok": [], "key": []}
        n_lab = len(lab["valid"])
        ahead = np.arange(1, cfg.action_chunk + 1)
        for spec in specs:
            fidx = frame_indices(spec.start, cfg)
            tidx = fidx[len(cfg.context_offsets) + 1 :]
            fr = np.asarray(all_frames[fidx])
            act = np.where(lab["valid"][tidx], lab["action"][tidx], -1)
            soft = lab["tick_actions"][tidx].astype(np.float32) / 5.0 * lab["valid"][tidx, None]
            path = (lab["path"][tidx] - self.norm.path_mean) / self.norm.path_std
            # Future steps are consecutive 50 ms steps whatever the frame stride.
            fut = tidx[:, None] + ahead[None]
            fc = np.clip(fut, 0, n_lab - 1)
            cok = (fut < n_lab) & lab["valid"][fc]
            csoft = lab["tick_actions"][fc].astype(np.float32) / 5.0 * cok[..., None]
            if spec.mirror:
                fr, act, path, soft = mirror_window(fr, act, path, soft)
                csoft = csoft[..., actions.MIRROR]
            out["frames"].append(fr)
            out["speed"].append(lab["speed"][tidx])
            out["action"].append(act)
            out["action_soft"].append(soft)
            out["path"].append(path.astype(np.float32))
            out["path_ok"].append(lab["path_ok"][tidx] & lab["valid"][tidx, None])
            out["progress"].append(((lab["progress"][tidx] - self.norm.prog_mean) / self.norm.prog_std).astype(np.float32))
            out["progress_ok"].append(lab["progress_ok"][tidx] & lab["valid"][tidx])
            out["chunk_soft"].append(csoft)
            out["chunk_ok"].append(cok)
            out["key"].append((spec.run, spec.start, spec.mirror))
        keys = out.pop("key")
        batch = {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}
        batch["keys"] = keys
        return batch


def to_pair(chunk: dict, sel: list[int] | slice) -> tuple[Observation, Labels]:
    obs = Observation(frames=chunk["frames"][sel], speed=chunk["speed"][sel].float())
    lab = Labels(
        action=chunk["action"][sel], action_soft=chunk["action_soft"][sel], path=chunk["path"][sel], path_ok=chunk["path_ok"][sel],
        progress=chunk["progress"][sel], progress_ok=chunk["progress_ok"][sel],
        chunk_soft=chunk["chunk_soft"][sel], chunk_ok=chunk["chunk_ok"][sel],
    )
    return obs, lab


def _identity(x):
    return x


class WindowLoader:
    """Yields micro-batches ``(Observation, Labels)`` for one epoch, in plan order."""

    def __init__(self, cfg: Config, runs: list[dict], manifest: dict, epoch: int, *, skip_windows: int = 0,
                 micro_batch: int | None = None, workers: int | None = None):
        self.cfg = cfg
        self.dcfg = cfg.data
        self.manifest = manifest
        self.blocks = epoch_plan(runs, cfg.data, cfg.train.seed, epoch)
        self.n_by_name = {r["name"]: r["n"] for r in runs}
        self.micro = micro_batch or cfg.train.micro_batch
        self.skip = skip_windows
        self.workers = cfg.data.loader_workers if workers is None else workers

    def __len__(self) -> int:
        return sum(len(b) for b in self.blocks) // self.micro

    @property
    def total_windows(self) -> int:
        return len(self) * self.micro

    def __iter__(self) -> Iterator[tuple[Observation, Labels]]:
        # Work items: (block, run) pairs in block order; a block is emitted once
        # all of its runs have arrived.
        items = []
        skip = self.skip
        blocks = []
        for blk in self.blocks:
            if skip >= len(blk):
                skip -= len(blk)
                continue
            blocks.append((blk, skip))
            skip = 0
        for blk, _ in blocks:
            by_run: dict[str, list[WindowSpec]] = {}
            for spec in blk:
                by_run.setdefault(spec.run, []).append(spec)
            for name, specs in by_run.items():
                items.append((name, self.n_by_name[name], specs))
        ds = RunWindows(self.dcfg, self.manifest, items)
        dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=self.workers,
                        prefetch_factor=self.dcfg.prefetch_batches if self.workers else None,
                        persistent_workers=False, collate_fn=_identity, pin_memory=False)
        # Windows are emitted in plan order. A run's chunk is pulled from the
        # workers only when one of its windows is first due, so the handoff of
        # a block's decoded runs (~2 GB at 320x240) is spread over the block
        # instead of stalling the GPU at every block boundary.
        pool: dict[tuple[str, int, bool], tuple[dict, int]] = {}
        it = iter(dl)
        order: list[tuple[str, int, bool]] = []
        skipped: set[tuple[str, int, bool]] = set()
        for blk, skip_in in blocks:
            keys = [(s.run, s.start, s.mirror) for s in blk]
            skipped.update(keys[:skip_in])  # consumed before a resume
            order.extend(keys[skip_in:])
        batch: list[tuple[dict, int]] = []
        for key in order:
            while key not in pool:
                chunk = next(it)
                for j, k2 in enumerate(chunk["keys"]):
                    if k2 not in skipped:
                        pool[k2] = (chunk, j)
            batch.append(pool.pop(key))
            if len(batch) == self.micro:
                yield _collate(batch)
                batch = []
        # A final partial micro-batch is dropped so every step has the same size.


def _collate(parts: list[tuple[dict, int]]) -> tuple[Observation, Labels]:
    fields = ("frames", "speed", "action", "action_soft", "path", "path_ok", "progress", "progress_ok",
              "chunk_soft", "chunk_ok")
    stacked = {f: torch.stack([c[f][j] for c, j in parts]) for f in fields}
    return to_pair(stacked, slice(None))


def run_tags(corpus: str) -> dict[str, set[str]]:
    """TMX tag names per run, from the corpus's cached tags.json."""
    import json
    import re

    from tmnf_collect.harvest import tmx

    root = Path(corpus)
    cached = json.loads((root / "tags.json").read_text(encoding="utf-8"))["tracks"]
    out: dict[str, set[str]] = {}
    for d in root.iterdir():
        meta = d / "meta.json"
        if not meta.is_file():
            continue
        name = Path(json.loads(meta.read_text(encoding="utf-8"))["map_file"].replace("\\", "/")).name
        m = re.match(r"tmx-(\d+)\.", name)
        ids = cached.get(m.group(1), []) if m else []
        out[d.name] = {tmx.tag_name(t) for t in ids}
    return out


def split_runs(manifest: dict, cfg: DataConfig | None = None) -> tuple[list[dict], list[dict]]:
    runs = manifest["runs"]
    if cfg is not None and cfg.exclude_tags:
        tags = run_tags(cfg.corpus)
        drop = set(cfg.exclude_tags)
        runs = [r for r in runs if not (tags.get(r["name"], set()) & drop)]
    train = [r for r in runs if not r["val"]]
    val = [r for r in runs if r["val"]]
    return train, val


def val_batches(cfg: Config, manifest: dict, n_windows: int) -> list[tuple[Observation, Labels]]:
    """A fixed validation set: the same windows every time (seeded, no mirroring)."""
    from dataclasses import replace

    _, val = split_runs(manifest, cfg.data)
    dcfg = replace(cfg.data, mirror_prob=0.0, windows_per_epoch=n_windows)
    vcfg = replace(cfg, data=dcfg)
    loader = WindowLoader(vcfg, val, manifest, epoch=10**6, workers=min(4, cfg.data.loader_workers))
    return list(loader)
