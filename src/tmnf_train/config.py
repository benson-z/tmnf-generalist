"""One config file drives data, model, training and evaluation.

Nested dataclasses loaded from YAML. Unknown keys are an error, as in
tmnf-collect's own config, so a typo cannot silently do nothing.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STORAGE = Path("Z:/application_storage/tmnf-ml")
# Paths in configs and checkpoints are written against the Windows NAS drive.
# With TMNF_STORAGE set (e.g. ~/tmnf-ml on Linux, /tmnf-ml in the ser5
# container), that prefix is swapped for it when a config is loaded, so the
# same config files work on every machine.
_STORAGE_PREFIX = "Z:/application_storage/tmnf-ml"


def storage_path(p: str) -> str:
    """``p`` with the Windows storage prefix swapped for $TMNF_STORAGE, if set."""
    root = os.environ.get("TMNF_STORAGE")
    if not root:
        return p
    q = p.replace("\\", "/")
    if q.lower().startswith(_STORAGE_PREFIX.lower()):
        return str(Path(os.path.expanduser(root)) / q[len(_STORAGE_PREFIX):].lstrip("/"))
    return p


def canonical_path(p: str) -> str:
    """The inverse of :func:`storage_path`: a local storage path in its
    machine-independent ``Z:/application_storage/tmnf-ml/...`` form, so it can
    be handed to another machine (e.g. the policy server)."""
    root = os.environ.get("TMNF_STORAGE")
    if root:
        base = Path(os.path.expanduser(root)).resolve()
        try:
            rel = Path(p).resolve().relative_to(base)
        except ValueError:
            try:  # a symlinked subtree (e.g. runs/ -> NAS) resolves elsewhere
                rel = Path(os.path.abspath(p)).relative_to(Path(os.path.abspath(os.path.expanduser(root))))
            except ValueError:
                return p
        return f"{_STORAGE_PREFIX}/{rel.as_posix()}"
    return p.replace("\\", "/")


@dataclass
class DataConfig:
    corpus: str = str(STORAGE / "train_data" / "corpus2")
    # Where derived data (labels index, optional uint8 memmap cache) is written.
    work_dir: str = str(STORAGE / "work")
    # Source resolution is 320x240; the input is derived from it by exact
    # integer downsampling (factor 1 or 2).
    downsample: int = 1
    # Optional horizon crop in *source* pixel rows: [top, bottom). null = none.
    crop_rows: list[int] | None = None
    window: int = 16  # frames per training window
    frame_stride: int = 1  # step between frames of a window, in 20 Hz frames
    # Optional extra strided context frames before the window's first frame,
    # e.g. [20, 40, 60] -> t-20, t-40, t-60. Off by default.
    context_offsets: list[int] = field(default_factory=list)
    # Action chunking: also label each frame with the soft actions of the next
    # N 50 ms steps (t+1 .. t+N), for an auxiliary head. 0 disables it.
    action_chunk: int = 0
    # "video" decodes H.265 on the fly (NVDEC when available); "memmap" reads a
    # uint8 cache at the training resolution built by `tmnf-train cache`.
    source: str = "video"
    # auto | nvdec | cpu. CPU decoding feeds the loader faster than NVDEC: with
    # 8 workers it gave 1,411 frames/s of training against 1,200 with NVDEC and
    # 6 workers, and no data wait (laptop, 2026-10-01). Past ~6k frames/s the
    # per-frame CPU work (download, rgb24, piping) is the limit, not the decoder.
    decoder: str = "cpu"
    mirror_prob: float = 0.5
    val_fraction: float = 0.02  # held-out runs, for a sanity-check loss only
    drop_respawn_runs: bool = True
    # Leave out runs on maps carrying any of these TMX tags (from the
    # tags.json that `tmnf-collect stats` caches beside the corpus), in both
    # the training and validation splits. E.g. [FullSpeed].
    exclude_tags: list[str] = field(default_factory=list)
    respawn_mask_before_s: float = 2.0
    respawn_mask_after_s: float = 1.0
    # Future path aux: waypoints at these horizons (s), in the car's frame.
    waypoint_horizons_s: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    progress_horizon_s: float = 4.0
    # Add each waypoint's height change (m, world up) as a 4th path channel,
    # after lateral, forward and speed. Needs an index built with it (use a
    # separate work_dir: the manifest's path statistics change shape).
    path_height: bool = False
    windows_per_epoch: int | None = None  # null = one pass over all windows at stride `window`
    loader_workers: int = 8  # 12 ran out of shared memory and hung training
    prefetch_batches: int = 4


@dataclass
class ModelConfig:
    channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    blocks_per_stage: int = 2
    tokens_per_frame: int = 8
    width: int = 512
    layers: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    speed_scale_kmh: float = 300.0
    groupnorm: bool = True
    # joint: one 12-way softmax over gas x brake x steer.
    # chain: P(steer) * P(brake | steer) * P(gas | steer, brake), all
    # conditionals from one linear layer (3 + 3 + 6 = 12 outputs). The joint
    # distribution is the same family; brake gets its own loss weight
    # (train.w_brake). The head still returns 12-way log-probabilities.
    policy_head: str = "joint"


@dataclass
class TrainConfig:
    run_dir: str = str(STORAGE / "runs")
    run_name: str = "baseline"
    seed: int = 0
    epochs: int = 2
    batch_size: int = 32  # effective batch (windows per optimizer step)
    micro_batch: int = 8  # windows per forward pass; grads accumulate
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp: str = "bf16"  # bf16 | fp16 | none
    grad_checkpoint: bool = False
    # Policy target. hard: the per-key majority action of the 5 ticks.
    # soft: the share of the 5 ticks that fell in each of the 12 actions, so a
    # 20 ms steer tap is a 40/60 target instead of being rounded away.
    label_mode: str = "hard"
    # Loss weights. Each target is normalised so the terms start comparable.
    w_policy: float = 1.0
    w_path: float = 0.5
    w_progress: float = 0.5
    w_chunk: float = 0.5  # used only when data.action_chunk > 0
    # model.policy_head chain only: weight of the P(brake | steer) term within
    # the policy loss. 1.0 makes the policy loss exactly the joint CE.
    w_brake: float = 1.0
    # Exponential moving average of the weights, updated every optimizer step.
    # Every checkpoint gets an ``<name>_ema.pt`` twin holding it. 0 disables.
    ema_decay: float = 0.0
    log_every: int = 50
    val_every_steps: int = 2000
    val_windows: int = 512
    ckpt_every_steps: int = 5000  # in addition to one per epoch
    eval_every_epochs: int = 1  # K; 0 disables in-training eval
    # inline: training pauses and runs the eval itself (same GPU).
    # external: training only writes checkpoints; `tmnf-train eval-watch`,
    # typically in the ROCm environment on the iGPU, evaluates them.
    eval_mode: str = "inline"
    # torch.compile the CNN encoder and transformer blocks (needs Triton; on
    # Windows the triton-windows package). Compiled in place, so checkpoints
    # are unchanged; only training uses it, eval builds a plain model.
    compile: bool = False
    # Stop the path head's gradient at the shared features: it still learns to
    # read the future off them (and can be drawn), but no longer shapes them.
    # Trains the trunk exactly as if the path head were removed.
    path_detach: bool = False
    # Keep the CNN's activations in NHWC (channels_last) layout for cuDNN.
    # Measured 0.9x alone (GroupNorm), and with compile the gradients were
    # wrong (torch 2.11 / triton-windows 3.6); leave off. Still wrong on Linux
    # (torch 2.11, 2026-10-01): in bf16, compile + channels_last gives a
    # whole-model gradient cosine of 0.958 against the plain model, with the
    # error in the encoder's GroupNorm residual blocks; fp32, or either one
    # alone, is correct. Eager channels_last is slower than compiled NCHW.
    channels_last: bool = False
    # Seeds are always fixed. True also forces deterministic cuDNN kernels
    # (bitwise-repeatable, measured ~2x slower); False lets cuDNN autotune.
    deterministic: bool = False


@dataclass
class EvalConfig:
    # Stock campaign maps by name, or corpus runs as "corpus:<run name>"
    # (their recorded trajectory gives a distance-from-line diagnostic).
    maps: list[str] = field(default_factory=lambda: ["B01-Race"])
    corpus: str = str(STORAGE / "train_data" / "corpus2")
    off_line_m: float = 10.0  # corpus maps: "left the line" threshold
    rollouts: int = 4
    temperature: float = 0.3  # 0 = greedy (then one rollout is enough)
    # Per-control temperatures, e.g. {steer: 0.5, brake: 1.0, gas: 0.3}. When
    # set, the action is sampled in stages from the 12-way distribution: steer
    # from its marginal, then brake given steer, then gas given both, each
    # with its own temperature. ``temperature`` is then unused.
    temperature_controls: dict[str, float] | None = None
    # Which head picks the action: policy, or chunkK = the action-chunk head's
    # prediction for step t+K (a model trained with data.action_chunk >= K).
    action_source: str = "policy"
    seed: int = 1234
    timeout_s: float = 60.0  # race time; B01 bronze is 39.68 s
    # Per map the race timeout is max(timeout_s, this x the map's author time),
    # and the per-rollout wall cap is raised to match.
    timeout_author_factor: float = 1.5
    stationary_s: float = 3.0
    stationary_kmh: float = 10.0  # a car grinding on a wall reads 4-6 km/h
    lanes: int = 1  # game instances in parallel
    speed: float = 2.0  # game speed between held steps
    port: int = 8600
    offscreen: bool = False  # game windows stay on screen
    max_attempts: int = 3  # per rollout, across game crashes
    # Wall-clock budget for a whole eval (all maps), game launches included.
    # When it runs out no new rollout starts, and a rollout still driving ends
    # as "time_limit". Rollouts are interleaved across maps, so a cut still
    # covers every map. null = no limit.
    max_wall_s: float | None = 600.0
    # Wall-clock cap for one rollout; it ends as "time_limit". null = none.
    rollout_max_wall_s: float | None = 120.0
    step_wall_timeout_s: float = 30.0  # no message this long = game hung
    out_dir: str = str(STORAGE / "eval")
    # Keep videos of only the most recent N checkpoints; null keeps all.
    keep_last_n_videos: int | None = None
    overlay: bool = True
    # Draw the path head's prediction (0.5-3 s ahead) on eval videos as a
    # car-wide ribbon on the road (in 3-D for models trained with
    # data.path_height), colored by the planned speed change (green faster,
    # white the same, red slower), with a halo for the predicted lateral
    # uncertainty.
    overlay_path: bool = True
    # Eval videos are upscaled by this integer factor (each pixel a block)
    # before the path and the info strip are drawn, so text and lines are
    # sharp. 1 = the game's 320x240.
    video_scale: int = 2
    # kv_cache: rolling KV cache (each frame and each token computed once).
    # recompute: CNN tokens cached per frame, the transformer re-run over the
    # last window each step; exactly what training saw.
    inference: str = "kv_cache"
    # auto (cuda if present, else cpu) | cuda | cpu | directml. directml runs
    # the exported model under ONNX Runtime on any DX12 GPU, e.g. the iGPU.
    device: str = "auto"
    # DXGI adapter index for directml. On the dev laptop 0 is the RTX 5070 Ti
    # and 1 the Radeon iGPU (checked: 1 never shows up in nvidia-smi).
    dml_device_id: int = 1
    # device remote: the policy runs in `tmnf-train serve-policy` on another
    # machine (host:port); frames go over TCP, actions come back.
    policy_url: str = "127.0.0.1:9555"
    random_gas_prob: float = 0.8  # random-policy smoke test only
    random_brake_prob: float = 0.1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    # -- derived --------------------------------------------------------------
    @property
    def input_hw(self) -> tuple[int, int]:
        top, bottom = self.data.crop_rows or (0, 240)
        f = self.data.downsample
        if (bottom - top) % f or 320 % f:
            raise ValueError("crop/downsample must divide exactly")
        return (bottom - top) // f, 320 // f


def _fill(cls, values: dict[str, Any], where: str):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")
    return cls(**values)


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        raw.setdefault(section, {})[name] = yaml.safe_load(value)
    sections = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig, "eval": EvalConfig}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown config section(s): {sorted(unknown)}")
    cfg = Config(**{k: _fill(c, raw.get(k, {}), k) for k, c in sections.items()})
    localize(cfg)
    cfg.input_hw  # validate
    return cfg


def localize(cfg: Config) -> Config:
    """Rewrite the storage paths of ``cfg`` in place for this machine."""
    for section, name in (("data", "corpus"), ("data", "work_dir"), ("train", "run_dir"),
                          ("eval", "corpus"), ("eval", "out_dir")):
        obj = getattr(cfg, section)
        setattr(obj, name, storage_path(getattr(obj, name)))
    return cfg


def dump(cfg: Config) -> str:
    return json.dumps(cfg.to_dict(), indent=1)
