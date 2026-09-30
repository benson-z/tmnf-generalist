"""A checkpoint as an eval policy on ONNX Runtime, e.g. DirectML on the iGPU.

PyTorch has no Windows build for the Radeon 890M (gfx1150), but DirectML runs
on any DX12 GPU. The model is exported as two graphs, which are exactly the
``recompute`` inference path of ``policy_model.ModelEpisode``:

    encoder   prev, cur (1, H, W, 3) uint8, speed (1,)  ->  tokens (1, K, width)
    core      tokens of the last window frames, left-padded (1, window, K, width),
              first real frame index  ->  logits (1, 12)

so each frame is encoded once and the transformer sees exactly what training
did. Context-frame models are not supported here (use a torch device).

Exports are cached next to the checkpoint's eval output under
``<work_dir>/onnx/<checkpoint>/``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import frames as frames_mod
from .model import Model
from .policy_model import ModelPolicy, path_aux


class _Encoder(torch.nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.m = model

    def forward(self, prev_u8: torch.Tensor, cur_u8: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        pair = self.m.preprocess(torch.stack([prev_u8, cur_u8], 1))[:, 0]
        return self.m._tokens(self.m.encode(pair[:, None]), speed[:, None])[:, 0]


class _Core(torch.nn.Module):
    """The transformer over a fixed ``window`` of frames, left-padded.

    A fixed shape runs ~4x faster under DirectML than a dynamic one. Frames
    before ``first`` are padding: no real token attends to them (and padding
    attends only to itself, so nothing becomes NaN). Rotary positions depend
    only on offsets, so the last frame's output equals ``core_last`` over the
    real frames alone.
    """

    def __init__(self, model: Model):
        super().__init__()
        self.m = model

    def forward(self, tokens: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        m = self.m
        b, t = tokens.shape[:2]
        frame_of = torch.arange(t, device=tokens.device).repeat_interleave(m.K)
        real = frame_of >= first
        causal = frame_of[None, :] <= frame_of[:, None]
        mask = causal & (real[None, :] | (frame_of[None, :] == frame_of[:, None]))
        x = tokens.flatten(1, 2)
        for blk in m.blocks:
            x, _ = blk(x, frame_of, mask)
        out = m.heads(x[:, -m.K:].view(b, 1, m.K, -1))
        return out["logits"][:, 0], out["path_mean"][:, 0], out["path_logvar"][:, 0]


def export(model: Model, cfg: Config, out_dir: Path) -> tuple[Path, Path]:
    """Write encoder.onnx and core.onnx (fp32) if not already there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_p, core_p = out_dir / "encoder.onnx", out_dir / "core.onnx"
    if enc_p.exists() and core_p.exists():
        return enc_p, core_p
    model = model.float().cpu().eval()
    h, w = cfg.input_hw
    u8 = torch.randint(0, 255, (1, h, w, 3), dtype=torch.uint8)
    with torch.no_grad():
        torch.onnx.export(_Encoder(model), (u8, u8, torch.tensor([100.0])), str(enc_p.with_suffix(".tmp")),
                          input_names=["prev", "cur", "speed"], output_names=["tokens"],
                          opset_version=17, dynamo=False)
        tok = torch.randn(1, cfg.data.window, model.K, cfg.model.width)
        torch.onnx.export(_Core(model), (tok, torch.tensor(3)), str(core_p.with_suffix(".tmp")),
                          input_names=["tokens", "first"], output_names=["logits", "path_mean", "path_logvar"],
                          opset_version=17, dynamo=False)
    enc_p.with_suffix(".tmp").replace(enc_p)
    core_p.with_suffix(".tmp").replace(core_p)
    return enc_p, core_p


def _session(path: Path, provider: str, device_id: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if provider == "dml":
        # DirectML wants sequential execution and no memory pattern.
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), opts, providers=providers)


class OnnxPolicy:
    """Shared by every eval lane. DirectML does not support concurrent Run
    calls into one session (two lanes doing so got the device suspended,
    887A0005), so all inference goes through ``run`` under one lock, and a
    lost device rebuilds the sessions once before giving up."""

    def __init__(self, cfg: Config, enc_path: Path, core_path: Path, id: str, provider: str = "dml",
                 device_id: int = 0, norm: dict | None = None):
        self.cfg = cfg
        self.id = id
        self.norm = norm
        self.paths = (enc_path, core_path)
        self.provider, self.device_id = provider, device_id
        self.lock = threading.Lock()
        self._load()
        self.device = f"onnx-{provider}:{device_id}"

    def _load(self) -> None:
        self.enc = _session(self.paths[0], self.provider, self.device_id)
        self.core = _session(self.paths[1], self.provider, self.device_id)

    def run(self, enc_feed: dict, core_feed_fn) -> tuple[np.ndarray, list[np.ndarray]]:
        """Encoder then core, serialised; ``core_feed_fn(tokens)`` builds the core feed."""
        with self.lock:
            for attempt in (1, 2):
                try:
                    tok = self.enc.run(None, enc_feed)[0]
                    return tok, self.core.run(None, core_feed_fn(tok))
                except Exception as exc:  # onnxruntime raises its own Fail/RuntimeException types
                    if attempt == 2 or "887A0005" not in str(exc) and "device" not in str(exc).lower():
                        raise
                    self._load()  # the device was reset: fresh sessions, try once more

    @classmethod
    def from_checkpoint(cls, path: str, id: str | None = None, provider: str = "dml", device_id: int = 0,
                        cache_root: Path | None = None) -> "OnnxPolicy":
        mp = ModelPolicy.from_checkpoint(path, id=id, inference="recompute", device="cpu")
        if mp.model.n_ctx:
            raise ValueError("ONNX eval supports models without context frames")
        root = cache_root or Path(mp.cfg.data.work_dir) / "onnx"
        # "_v2": the core graph also returns the path head.
        enc_p, core_p = export(mp.model, mp.cfg, root / f"{Path(path).stem}_v2")
        return cls(mp.cfg, enc_p, core_p, mp.id, provider, device_id, mp.norm)

    def episode(self, seed: int, temperature: float) -> "OnnxEpisode":
        return OnnxEpisode(self, seed, temperature)


class OnnxEpisode:
    """Same observation handling and sampling as ``ModelEpisode`` in recompute mode."""

    def __init__(self, policy: OnnxPolicy, seed: int, temperature: float):
        self.p = policy
        self.gen = torch.Generator().manual_seed(seed)
        self.temperature = temperature
        d = policy.cfg.data
        self.stride = d.frame_stride
        self.history: deque[np.ndarray] = deque(maxlen=self.stride + 1)
        self.tokens: deque[np.ndarray] = deque(maxlen=d.window)
        self.n = 0
        self.last = (0, None)
        self.infer_s = 0.0
        self.aux: dict | None = None

    def act(self, frame: np.ndarray, speed_kmh: float) -> tuple[int, np.ndarray | None]:
        d = self.p.cfg.data
        x = frames_mod.to_input(frame, d.downsample, d.crop_rows)
        self.history.append(x)
        k = self.n
        self.n += 1
        if k % self.stride:
            return self.last
        prev = self.history[0] if len(self.history) > self.stride else x
        t = time.perf_counter()
        window = self.tokens.maxlen

        def core_feed(tok: np.ndarray) -> dict:
            stacked = np.stack([*self.tokens, tok], 1)[:, -window:]
            first = window - stacked.shape[1]
            if first:
                stacked = np.concatenate([np.zeros((1, first, *stacked.shape[2:]), stacked.dtype), stacked], 1)
            return {"tokens": stacked, "first": np.array(first, np.int64)}

        tok, outs = self.p.run({"prev": prev[None], "cur": x[None], "speed": np.array([speed_kmh], np.float32)},
                               core_feed)
        self.tokens.append(tok)
        logits, pmean, plogvar = (o[0] for o in outs)
        self.infer_s += time.perf_counter() - t
        self.aux = path_aux(pmean, plogvar, self.p.norm, d.waypoint_horizons_s)
        logits = torch.from_numpy(np.ascontiguousarray(logits)).float()
        if self.temperature <= 0:
            probs = torch.softmax(logits, -1)
            a = int(probs.argmax())
        else:
            probs = torch.softmax(logits / self.temperature, -1)
            a = int(torch.multinomial(probs, 1, generator=self.gen))
        self.last = (a, probs.numpy())
        return self.last
