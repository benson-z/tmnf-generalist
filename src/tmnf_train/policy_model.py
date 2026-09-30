"""A trained model as an eval-harness policy.

Sees exactly what training saw: the captured 320x240 frame put through
``frames.to_input`` (same crop and integer downsampling), paired with the
previous model frame, plus the HUD speed. Nothing else from the game.

Without context frames each step runs ``Model.step`` on a rolling KV cache
(each frame encoded once). With context frames the last window is recomputed
from a frame history every step.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import Config, load
from .data import frames as frames_mod
from .data.dataset import Observation, frame_indices
from .model import Model

_LOCK = threading.Lock()


def path_aux(mean: np.ndarray, logvar: np.ndarray, norm: dict | None, horizons: list[float]) -> dict | None:
    """The path head's output for one frame in metres / km/h.

    ``mean``/``logvar`` are (K, 3) in normalised units: lateral (left +),
    forward, future speed. Returned for drawing and logging only.
    """
    if norm is None:
        return None
    m = np.asarray(norm["path_mean"], np.float32)
    s = np.asarray(norm["path_std"], np.float32)
    return {
        "horizons_s": list(horizons),
        "mean": (mean * s + m).tolist(),
        "std": (np.exp(0.5 * logvar) * s).tolist(),
    }


def sample_action(logits: torch.Tensor, temperature: float, gen: torch.Generator,
                  controls: dict[str, float] | None = None) -> tuple[int, np.ndarray]:
    """Pick one of the 12 actions from 12-way logits.

    Without ``controls``: softmax(logits / temperature), sampled (argmax at
    temperature 0). With ``controls`` ({steer, brake, gas} temperatures): the
    same 12-way distribution sampled in stages, steer from its marginal, then
    brake given the chosen steer, then gas given both, each stage sharpened by
    its own temperature (0 = argmax for that stage). Returns the action and the
    distribution it was drawn from.
    """
    logits = logits.float().cpu()
    if controls is None:
        if temperature <= 0:
            probs = torch.softmax(logits, -1)
            return int(probs.argmax()), probs.numpy()
        probs = torch.softmax(logits / temperature, -1)
        return int(torch.multinomial(probs, 1, generator=gen)), probs.numpy()

    p = torch.softmax(logits, -1).double().view(2, 2, 3)  # [gas, brake, steer]
    t_s, t_b, t_g = (controls.get(k, 1.0) for k in ("steer", "brake", "gas"))
    q_s = _temper(p.sum((0, 1)), t_s)
    q_b = torch.stack([_temper(p[:, :, s].sum(0), t_b) for s in range(3)], -1)  # [brake, steer]
    q_g = torch.stack([torch.stack([_temper(p[:, b, s], t_g) for s in range(3)], -1)
                       for b in range(2)], 1)  # [gas, brake, steer]
    s = int(torch.multinomial(q_s.float(), 1, generator=gen))
    b = int(torch.multinomial(q_b[:, s].float(), 1, generator=gen))
    g = int(torch.multinomial(q_g[:, b, s].float(), 1, generator=gen))
    # The distribution actually sampled from: q(s) q(b|s) q(g|s,b).
    joint = q_s[None, None, :] * q_b[None] * q_g
    return g * 6 + b * 3 + s, joint.flatten().float().numpy()


def _temper(w: torch.Tensor, t: float) -> torch.Tensor:
    """Normalise ``w`` and sharpen it by temperature ``t`` (0 = one-hot argmax)."""
    w = w / w.sum().clamp(min=1e-30)
    if t <= 0:
        q = torch.zeros_like(w)
        q[int(w.argmax())] = 1.0
        return q
    q = w.clamp(min=1e-30) ** (1.0 / t)
    return q / q.sum()


def load_policy(path: str, cfg_eval, id: str | None = None):
    """A checkpoint as an eval policy on the configured eval device."""
    if cfg_eval.device == "remote":
        from .policy_server import RemotePolicy

        return RemotePolicy(path, cfg_eval, id=id)
    if cfg_eval.device == "directml":
        from .onnx_policy import OnnxPolicy

        assert cfg_eval.temperature_controls is None and cfg_eval.action_source == "policy", \
            "per-control temperatures and action_source are not implemented for directml"
        return OnnxPolicy.from_checkpoint(path, id=id, provider="dml", device_id=cfg_eval.dml_device_id)
    policy = ModelPolicy.from_checkpoint(path, id=id, inference=cfg_eval.inference, device=cfg_eval.device)
    policy.set_sampling(cfg_eval.temperature_controls, cfg_eval.action_source)
    return policy


class ModelPolicy:
    def __init__(self, model: Model, cfg: Config, id: str, device: torch.device | None = None,
                 inference: str | None = None, norm: dict | None = None):
        self.model = model
        self.cfg = cfg
        self.id = id
        # Target normalisation (from the checkpoint or manifest), to report the
        # path head in metres. Without it episodes expose no aux output.
        self.norm = norm
        self.inference = inference or cfg.eval.inference
        self.device = device or next(model.parameters()).device
        self.temperature_controls: dict[str, float] | None = None
        self.action_source = "policy"

    def set_sampling(self, temperature_controls: dict[str, float] | None, action_source: str = "policy") -> None:
        if temperature_controls is not None:
            unknown = set(temperature_controls) - {"steer", "brake", "gas"}
            if unknown:
                raise ValueError(f"unknown control(s) in temperature_controls: {sorted(unknown)}")
        if action_source != "policy":
            k = int(action_source.removeprefix("chunk")) if action_source.startswith("chunk") else 0
            if not 1 <= k <= self.model.n_chunk:
                raise ValueError(f"action_source {action_source!r} needs an action-chunk head of length >= "
                                 f"{max(k, 1)}; this model has {self.model.n_chunk}")
        self.temperature_controls = dict(temperature_controls) if temperature_controls else None
        self.action_source = action_source

    @classmethod
    def from_checkpoint(cls, path: str, id: str | None = None, inference: str | None = None,
                        device: str = "auto") -> "ModelPolicy":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # Keys a newer config no longer has (e.g. the old eval.map) are dropped;
        # only the data and model sections matter to a checkpoint.
        base = load(None)
        sections = {}
        for k, v in ck["config"].items():
            section_cls = type(getattr(base, k))
            known = {f.name for f in dataclasses.fields(section_cls)}
            sections[k] = section_cls(**{kk: vv for kk, vv in v.items() if kk in known})
        cfg = Config(**sections)
        model = Model(cfg, ck["n_horizons"])
        model.load_state_dict(ck["model"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)  # "cuda" is also ROCm/HIP in a ROCm build
        model.to(dev).eval()
        return cls(model, cfg, id or Path(path).stem, dev, inference, ck.get("norm"))

    def episode(self, seed: int, temperature: float) -> "ModelEpisode":
        return ModelEpisode(self, seed, temperature)


class ModelEpisode:
    def __init__(self, policy: ModelPolicy, seed: int, temperature: float):
        self.p = policy
        self.gen = torch.Generator().manual_seed(seed)
        self.temperature = temperature
        d = policy.cfg.data
        self.stride = d.frame_stride
        # Enough history for context offsets and the previous-frame pairing.
        need = max([self.stride * d.window + self.stride] + [o + self.stride * d.window for o in d.context_offsets])
        self.history: deque[np.ndarray] = deque(maxlen=need + 1)
        self.speeds: deque[float] = deque(maxlen=need + 1)
        self.cache = None
        self.tokens: deque[torch.Tensor] = deque(maxlen=d.window)
        self.n = 0
        self.last = (0, None)
        self.aux: dict | None = None  # path head at the last model frame, metres

    def _input(self, frame: np.ndarray) -> np.ndarray:
        d = self.p.cfg.data
        return frames_mod.to_input(frame, d.downsample, d.crop_rows)

    @torch.no_grad()
    def act(self, frame: np.ndarray, speed_kmh: float) -> tuple[int, np.ndarray | None]:
        x = self._input(frame)
        self.history.append(x)
        self.speeds.append(speed_kmh)
        k = self.n
        self.n += 1
        if k % self.stride:  # between model frames the last action is held
            return self.last
        model, dev = self.p.model, self.p.device
        with _LOCK, torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if model.n_ctx == 0:
                prev = self.history[-1 - self.stride] if len(self.history) > self.stride else x
                u8 = torch.from_numpy(np.stack([prev, x]))[None].to(dev)
                pair = model.preprocess(u8)[:, 0]
                sp = torch.tensor([speed_kmh], device=dev)
                if self.p.inference == "recompute":
                    self.tokens.append(model.encode_step(pair, sp))
                    out = model.core_last(torch.stack(list(self.tokens), 1))
                else:
                    out, self.cache = model.step(pair, sp, self.cache)
                out = {k: v[0, 0] for k, v in out.items()}
            else:
                out = self._full_window(model, dev)
            if self.p.action_source == "policy":
                logits = out["logits"]
            else:  # chunkK: the chunk head's prediction for step t+K
                logits = out["chunk_logits"][int(self.p.action_source.removeprefix("chunk")) - 1]
        self.aux = path_aux(out["path_mean"].float().cpu().numpy(), out["path_logvar"].float().cpu().numpy(),
                            self.p.norm, self.p.cfg.data.waypoint_horizons_s)
        self.last = sample_action(logits, self.temperature, self.gen, self.p.temperature_controls)
        return self.last

    def _full_window(self, model: Model, dev) -> dict:
        """Recompute the window ending at the newest frame (context-frame models)."""
        d = self.p.cfg.data
        hist = list(self.history)
        cur = len(hist) - 1
        start = cur - (d.window - 1) * self.stride
        # Indices into the history. Until the history is full, index 0 is the
        # episode's first frame, so negative indices clamp to it as in training;
        # once full it holds every frame a window can reach.
        idx = frame_indices(start, d)
        frames = torch.from_numpy(np.stack([hist[i] for i in idx]))[None].to(dev)
        sp_idx = idx[len(d.context_offsets) + 1 :]
        speeds = torch.tensor([[list(self.speeds)[i] for i in sp_idx]], device=dev, dtype=torch.float32)
        out = model(Observation(frames=frames, speed=speeds))
        return {k: v[0, -1] for k, v in out.items()}
