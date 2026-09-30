"""One config file drives data, model, training and evaluation.

Nested dataclasses loaded from YAML. Unknown keys are an error, as in
tmnf-collect's own config, so a typo cannot silently do nothing.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STORAGE = Path("Z:/application_storage/tmnf-ml")


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
    decoder: str = "auto"  # auto | nvdec | cpu
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
    windows_per_epoch: int | None = None  # null = one pass over all windows at stride `window`
    loader_workers: int = 6
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
    # wrong (torch 2.11 / triton-windows 3.6); leave off.
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
    # Draw the path head's predicted waypoints (0.5-3 s ahead) on eval videos,
    # with a ring for the predicted lateral uncertainty.
    overlay_path: bool = True
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
    cfg.input_hw  # validate
    return cfg


def dump(cfg: Config) -> str:
    return json.dumps(cfg.to_dict(), indent=1)
