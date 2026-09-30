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


def load_policy(path: str, cfg_eval, id: str | None = None):
    """A checkpoint as an eval policy on the configured eval device."""
    if cfg_eval.device == "directml":
        from .onnx_policy import OnnxPolicy

        return OnnxPolicy.from_checkpoint(path, id=id, provider="dml", device_id=cfg_eval.dml_device_id)
    return ModelPolicy.from_checkpoint(path, id=id, inference=cfg_eval.inference, device=cfg_eval.device)


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
            logits = out["logits"]
        self.aux = path_aux(out["path_mean"].float().cpu().numpy(), out["path_logvar"].float().cpu().numpy(),
                            self.p.norm, self.p.cfg.data.waypoint_horizons_s)
        logits = logits.float().cpu()
        if self.temperature <= 0:
            probs = torch.softmax(logits, -1)
            a = int(probs.argmax())
        else:
            probs = torch.softmax(logits / self.temperature, -1)
            a = int(torch.multinomial(probs, 1, generator=self.gen))
        self.last = (a, probs.numpy())
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
