"""Pixels + speed -> 12-way action, with two auxiliary heads.

Per frame: the RGB frame concatenated with the previous frame (6 channels) goes
through an IMPALA-style CNN (32/64/64, GroupNorm, no BatchNorm) whose stride-2
stem brings 320x240 down to 160x120 before any trunk conv runs. Learned queries
attention-pool the final feature map to ``tokens_per_frame`` tokens; the HUD
speed is embedded and added to them.

Core: a causal transformer over the window's frame tokens. Attention is
block-causal: a token sees every token of its own frame and of earlier frames.
Positions are rotary over the frame index, so at inference the cache can slide:
a rolling KV cache of the last ``window`` frames keeps exactly the context the
model was trained with (``Model.step``), and each frame is encoded once.

Heads read the mean of a frame's output tokens:
  policy    12 logits (``model.policy_head: joint``), or 12 log-probabilities
            built as P(steer) P(brake|steer) P(gas|steer,brake) (``chain``)
  path      waypoints (lateral, forward) + speed at each horizon, as mean and
            log-variance (heteroscedastic, i.e. uncertainty-weighted, NLL)
  progress  arc length over the next H seconds (normalised), scalar

``forward`` accepts an ``Observation`` and nothing else; car state cannot get in.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .actions import N_ACTIONS
from .config import Config
from .data.dataset import Observation


# ----------------------------------------------------------------- encoder

def _gn(c: int, on: bool) -> nn.Module:
    return nn.GroupNorm(8, c) if on else nn.Identity()


class ResBlock(nn.Module):
    def __init__(self, c: int, gn: bool):
        super().__init__()
        self.n1, self.c1 = _gn(c, gn), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = _gn(c, gn), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        y = self.c1(F.relu(self.n1(x)))
        y = self.c2(F.relu(self.n2(y)))
        return x + y


class ImpalaStage(nn.Module):
    def __init__(self, cin: int, cout: int, blocks: int, gn: bool):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(cout, gn) for _ in range(blocks)])

    def forward(self, x):
        return self.blocks(self.pool(self.conv(x)))


class FrameEncoder(nn.Module):
    """(N, 6, H, W) -> (N, tokens, width)."""

    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        ch = m.channels
        # Stride-2 stem: the only layer that runs at full input resolution.
        self.stem = nn.Conv2d(6, ch[0], 3, stride=2, padding=1)
        stages, cin = [], ch[0]
        for c in ch:
            stages.append(ImpalaStage(cin, c, m.blocks_per_stage, m.groupnorm))
            cin = c
        self.stages = nn.Sequential(*stages)
        self.out_norm = _gn(cin, m.groupnorm)
        h, w = cfg.input_hw
        for _ in range(len(ch) + 1):
            h, w = (h + 1) // 2, (w + 1) // 2
        self.grid = (h, w)
        self.proj = nn.Linear(cin, m.width)
        self.pos = nn.Parameter(torch.randn(1, h * w, m.width) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, m.tokens_per_frame, m.width) * 0.02)
        self.pool = nn.MultiheadAttention(m.width, m.heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(m.width)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(self.out_norm(self.stages(x)))
        n, c, h, w = x.shape
        feats = self.proj(x.flatten(2).transpose(1, 2)) + self.pos
        q = self.queries.expand(n, -1, -1)
        tok, _ = self.pool(q, feats, feats, need_weights=False)
        return self.pool_norm(tok + q)


# ------------------------------------------------------------- transformer

def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding. x: (B, H, L, D), pos: (L,) frame index (float)."""
    # In float32: at inference absolute positions grow to thousands of frames,
    # and relative-position invariance must hold there too.
    d = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(d, device=x.device, dtype=torch.float32) / d))
    ang = pos[:, None].float() * freq[None]
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2 = xf[..., :d], xf[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class Block(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.n2 = nn.LayerNorm(width)
        hid = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hid), nn.GELU(), nn.Linear(hid, width))
        self.drop = dropout

    def attend(self, x, pos, mask, past=None):
        b, l, w = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, l, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        if past is not None:
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, l, w)), (k, v)

    def forward(self, x, pos, mask, past=None):
        a, kv = self.attend(x, pos, mask, past)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, kv


def chain_logp(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The chain-rule policy head: 12 raw outputs -> 12-way log-probabilities.

    ``raw[..., 0:3]`` are steer logits, ``raw[..., 3:6]`` the brake logit given
    each steer value, ``raw[..., 6:12]`` the gas logit given each (steer, brake)
    pair (index ``2 * steer + brake``). Returns the joint log-probabilities in
    the action layout (``gas * 6 + brake * 3 + steer``) and log P(brake | steer)
    as (..., 2, 3) indexed [brake, steer], for the brake loss weight.
    """
    raw = raw.float()
    steer = F.log_softmax(raw[..., 0:3], -1)  # (..., 3)
    bl = raw[..., 3:6]
    brake = torch.stack([F.logsigmoid(-bl), F.logsigmoid(bl)], -2)  # (..., b, s)
    gl = raw[..., 6:12].unflatten(-1, (3, 2)).transpose(-1, -2)  # (..., b, s)
    gas = torch.stack([F.logsigmoid(-gl), F.logsigmoid(gl)], -3)  # (..., g, b, s)
    joint = steer[..., None, None, :] + brake[..., None, :, :] + gas
    return joint.flatten(-3), brake


def block_causal_mask(frame_of: torch.Tensor, frame_of_keys: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean (Lq, Lk) mask, True = may attend: key frame <= query frame."""
    fk = frame_of if frame_of_keys is None else frame_of_keys
    return fk[None, :] <= frame_of[:, None]


class Model(nn.Module):
    def __init__(self, cfg: Config, n_horizons: int):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.T = cfg.data.window
        self.K = m.tokens_per_frame
        self.n_ctx = len(cfg.data.context_offsets)
        self.encoder = FrameEncoder(cfg)
        self.speed_embed = nn.Sequential(nn.Linear(1, m.width), nn.GELU(), nn.Linear(m.width, m.width))
        self.token_embed = nn.Parameter(torch.randn(1, 1, self.K, m.width) * 0.02)
        self.ctx_embed = nn.Parameter(torch.randn(max(1, self.n_ctx), m.width) * 0.02)
        self.blocks = nn.ModuleList(Block(m.width, m.heads, m.mlp_ratio, m.dropout) for _ in range(m.layers))
        self.norm = nn.LayerNorm(m.width)
        self.n_h = n_horizons
        self.policy_head = m.policy_head
        if self.policy_head not in ("joint", "chain"):
            raise ValueError(f"model.policy_head must be joint or chain, not {self.policy_head!r}")
        self.policy = nn.Linear(m.width, N_ACTIONS)  # chain: 3 + 3 + 6 raw outputs
        self.path = nn.Linear(m.width, n_horizons * 3 * 2)  # mean + log-variance
        self.progress = nn.Linear(m.width, 1)
        self.n_chunk = cfg.data.action_chunk
        self.chunk = nn.Linear(m.width, self.n_chunk * N_ACTIONS) if self.n_chunk else None
        self.grad_checkpoint = cfg.train.grad_checkpoint
        self.channels_last = cfg.train.channels_last
        self.path_detach = cfg.train.path_detach

    # -- per-frame encoding ------------------------------------------------
    def preprocess(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """(B, F, H, W, 3) uint8 -> (B, F-1, 6, H, W): each frame with its predecessor."""
        x = frames_u8.permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
        return torch.cat([x[:, 1:], x[:, :-1]], 2)

    def encode(self, pairs: torch.Tensor) -> torch.Tensor:
        """(B, F, 6, H, W) -> (B, F, K, width). Each frame encoded exactly once."""
        b, f = pairs.shape[:2]
        x = pairs.flatten(0, 1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        tok = self.encoder(x)
        return tok.view(b, f, self.K, -1)

    def _tokens(self, frame_tok: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        s = self.speed_embed((speed / self.cfg.model.speed_scale_kmh).unsqueeze(-1))
        return frame_tok + self.token_embed + s.unsqueeze(2)

    def heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.norm(h.mean(2))  # (B, T, width)
        path_in = pooled.detach() if self.path_detach else pooled
        path = self.path(path_in).view(*pooled.shape[:2], self.n_h, 3, 2)
        raw = self.policy(pooled)
        out = {}
        if self.policy_head == "chain":
            # Log-probabilities are valid logits: softmax/CE leave them as is.
            out["logits"], out["brake_logp"] = chain_logp(raw)
        else:
            out["logits"] = raw.float()
        out |= {
            "path_mean": path[..., 0].float(),
            "path_logvar": path[..., 1].float().clamp(-8, 8),
            "progress": self.progress(pooled).squeeze(-1).float(),
        }
        if self.chunk is not None:
            out["chunk_logits"] = self.chunk(pooled).view(*pooled.shape[:2], self.n_chunk, N_ACTIONS).float()
        return out

    # -- training: whole windows -------------------------------------------
    def forward(self, obs: Observation) -> dict[str, torch.Tensor]:
        assert isinstance(obs, Observation), "the model only accepts an Observation"
        c = self.n_ctx
        # Context frames are encoded on their own (paired with themselves);
        # window frames are paired with the frame before them.
        win = self.preprocess(obs.frames[:, c:])  # (B, T, 6, H, W)
        b, t = win.shape[:2]
        tok = self._tokens(self.encode(win), obs.speed)  # (B, T, K, W)
        pos = torch.arange(t, device=tok.device).repeat_interleave(self.K)
        frame_of = pos.clone()
        x = tok.flatten(1, 2)
        if c:
            # Context frames have no loaded predecessor: paired with themselves.
            cf = obs.frames[:, :c].permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
            ctx = self.encode(torch.cat([cf, cf], 2))
            ctx = (ctx + self.ctx_embed[:c, None]).flatten(1, 2)
            x = torch.cat([ctx, x], 1)
            # Context sits before the window: visible to all window tokens.
            frame_of = torch.cat([torch.full((c * self.K,), -1, device=x.device), frame_of])
            pos = torch.cat([torch.full((c * self.K,), -1, device=x.device), pos])
        mask = block_causal_mask(frame_of)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x, _ = checkpoint(blk, x, pos, mask, use_reentrant=False)
            else:
                x, _ = blk(x, pos, mask)
        h = x[:, c * self.K :].view(b, t, self.K, -1)
        return self.heads(h)

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def encode_step(self, pair: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """One frame's tokens, speed included: (B, 6, H, W), (B,) -> (B, K, width)."""
        return self._tokens(self.encode(pair[:, None]), speed[:, None])[:, 0]

    @torch.no_grad()
    def core_last(self, tokens: torch.Tensor) -> dict:
        """Exact recompute: the transformer over a window of already-encoded
        frames (B, t, K, width); heads for the last frame. Matches the last
        position of a training window bit for bit (up to float error)."""
        b, t = tokens.shape[:2]
        pos = torch.arange(t, device=tokens.device).repeat_interleave(self.K)
        mask = block_causal_mask(pos)
        x = tokens.flatten(1, 2)
        for blk in self.blocks:
            x, _ = blk(x, pos, mask)
        return self.heads(x[:, -self.K :].view(b, 1, self.K, -1))

    # -- inference: one frame at a time, rolling KV cache ------------------
    @torch.no_grad()
    def step(self, pair: torch.Tensor, speed: torch.Tensor, cache: dict | None) -> tuple[dict, dict]:
        """pair: (B, 6, H, W) float, speed: (B,). Returns head outputs for this frame.

        The cache keeps the last ``window`` frames' keys/values. Rotary
        positions make attention depend only on frame offsets, so the oldest
        frame can be dropped as a new one arrives. For the first ``window``
        frames this equals the training forward exactly. After that it is
        sliding-window attention: a retained frame's deeper-layer keys were
        computed when it could still see frames that have since left the
        window, so outputs drift slightly from a fresh window (measured ~0.02
        in logits on a small random model). ``core_last`` is the exact
        alternative (eval ``inference: recompute``).
        """
        assert self.n_ctx == 0, "KV-cached stepping is for models without context frames"
        if cache is None:
            cache = {"t": 0, "kv": [None] * len(self.blocks)}
        t = cache["t"]
        tok = self._tokens(self.encode(pair[:, None]), speed[:, None])  # (B,1,K,W)
        x = tok.flatten(1, 2)
        pos = torch.full((self.K,), t, device=x.device)
        new_kv = []
        for blk, past in zip(self.blocks, cache["kv"]):
            x, (k, v) = blk(x, pos, None, past)
            keep = self.T * self.K
            new_kv.append((k[:, :, -keep:], v[:, :, -keep:]))
        h = x.view(x.shape[0], 1, self.K, -1)
        return self.heads(h), {"t": t + 1, "kv": new_kv}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
