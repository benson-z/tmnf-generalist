"""Training: behaviour cloning with two auxiliary heads, single GPU.

    tmnf-train train [--resume]

Layout under ``<train.run_dir>/<train.run_name>/``:
    config.json            the resolved config
    metrics.jsonl          one log for the run: train, val, eval rows
    checkpoints/*.pt       every checkpoint, never deleted
    eval/<ckpt>/...        in-game rollouts (videos, metrics) per checkpoint

Loss = w_policy * CE(policy) + w_path * NLL(path) + w_progress * MSE(progress)
(+ w_chunk * CE(chunk) with action chunking). With the chain policy head, the
policy term is CE + (w_brake - 1) * CE(brake | steer): the joint CE splits
exactly into steer, brake-given-steer and gas-given-both parts, and w_brake
reweights the middle one.

With ``train.ema_decay`` > 0 an exponential moving average of the weights is
kept alongside, and every checkpoint ``<name>.pt`` gets an ``<name>_ema.pt``
twin in the same format holding those weights (loadable as any checkpoint).
Path and progress targets are normalised by corpus statistics and the path NLL
starts at log-variance 0, so all three terms start at comparable magnitudes
(CE ~ ln 12 = 2.5, the others ~0.5-1).

Validation loss is a sanity check only; checkpoints are chosen by watching
the eval videos.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data.dataset import Labels, Observation, WindowLoader, split_runs, val_batches
from .data.index import load_manifest
from .model import Model, n_params


# ------------------------------------------------------------------ losses

def losses(out: dict, lab: Labels, label_mode: str = "hard", w_brake: float = 1.0) -> dict[str, torch.Tensor]:
    act = lab.action
    if label_mode == "soft":
        m = (act >= 0).flatten().float()
        ce_all = F.cross_entropy(out["logits"].flatten(0, 1), lab.action_soft.flatten(0, 1), reduction="none")
        ce = (ce_all * m).sum() / m.sum().clamp(min=1)
    else:
        ce = F.cross_entropy(out["logits"].flatten(0, 1), act.flatten(), ignore_index=-1)
    if "brake_logp" in out and w_brake != 1.0:
        ce = ce + (w_brake - 1.0) * brake_ce(out["brake_logp"], lab, label_mode)

    ok = lab.path_ok.unsqueeze(-1).expand_as(lab.path).float()
    mu, lv = out["path_mean"], out["path_logvar"]
    nll = 0.5 * (lv + (lab.path - mu) ** 2 * torch.exp(-lv))
    path = (nll * ok).sum() / ok.sum().clamp(min=1)

    pok = lab.progress_ok.float()
    prog = (0.5 * (out["progress"] - lab.progress) ** 2 * pok).sum() / pok.sum().clamp(min=1)
    ls = {"policy": ce, "path": path, "progress": prog}
    if "chunk_logits" in out:
        # Soft cross-entropy against each future step's tick shares.
        cok = lab.chunk_ok.float()
        ce_c = -(lab.chunk_soft * F.log_softmax(out["chunk_logits"], -1)).sum(-1)
        ls["chunk"] = (ce_c * cok).sum() / cok.sum().clamp(min=1)
    return ls


def brake_ce(brake_logp: torch.Tensor, lab: Labels, label_mode: str) -> torch.Tensor:
    """-sum q(steer, brake) log P(brake | steer): the chain head's brake term.

    ``brake_logp`` is (B, T, 2, 3) indexed [brake, steer]; q is the target's
    (steer, brake) marginal, from the soft or the majority label.
    """
    act = lab.action
    m = (act >= 0).float()
    if label_mode == "soft":
        q = lab.action_soft
    else:
        q = F.one_hot(act.clamp(min=0), 12).float()
    q_bs = q.unflatten(-1, (2, 2, 3)).sum(-3)  # (B, T, brake, steer)
    per = -(q_bs * brake_logp).sum((-1, -2))
    return (per * m).sum() / m.sum().clamp(min=1)


class Ema:
    """fp32 exponential moving average of a model's floating-point state."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model) -> None:
        sd = model.state_dict()
        keys = list(self.shadow)
        torch._foreach_lerp_([self.shadow[k] for k in keys], [sd[k].detach().float() for k in keys], 1.0 - self.decay)

    def state_dict_like(self, model) -> dict:
        """The model's state dict with the averaged weights swapped in."""
        sd = model.state_dict()
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v) for k, v in sd.items()}


@torch.no_grad()
def metrics(out: dict, lab: Labels, norm: dict) -> dict[str, float]:
    act = lab.action
    m = act >= 0
    pred = out["logits"].argmax(-1)
    g_p, b_p, s_p = pred // 6, (pred // 3) % 2, pred % 3
    g_t, b_t, s_t = act // 6, (act // 3) % 2, act % 3
    r = {
        "acc": (pred == act)[m].float().mean().item(),
        "acc_gas": (g_p == g_t)[m].float().mean().item(),
        "acc_brake": (b_p == b_t)[m].float().mean().item(),
        "acc_steer": (s_p == s_t)[m].float().mean().item(),
        # Comparable across label modes: CE against the majority label.
        "ce_hard": F.cross_entropy(out["logits"].flatten(0, 1), act.flatten(), ignore_index=-1).item(),
    }
    # The rare classes, which overall accuracy hides (95% of labels are gas +
    # a steer choice). Recall = of frames with that label, how many predicted.
    for name, sel, hit in (
        ("steer_left", s_t == 0, s_p == 0), ("steer_none", s_t == 1, s_p == 1), ("steer_right", s_t == 2, s_p == 2),
        ("brake", b_t == 1, b_p == 1), ("no_gas", g_t == 0, g_p == 0),
    ):
        sel = sel & m
        if sel.any():
            r[f"recall_{name}"] = hit[sel].float().mean().item()
    # Steering changes: frames whose steer label differs from the previous frame's.
    change = torch.zeros_like(m)
    change[:, 1:] = (s_t[:, 1:] != s_t[:, :-1]) & m[:, 1:] & m[:, :-1]
    if change.any():
        r["acc_steer_at_change"] = (s_p == s_t)[change].float().mean().item()
    std = torch.tensor(norm["path_std"], device=act.device)
    ok = lab.path_ok
    err = ((out["path_mean"] - lab.path) * std).abs()  # metres / km/h
    if ok.any():
        r["path_lat_mae_m"] = err[..., 0][ok].mean().item()
        r["path_fwd_mae_m"] = err[..., 1][ok].mean().item()
        r["path_speed_mae_kmh"] = err[..., 2][ok].mean().item()
    pok = lab.progress_ok
    if pok.any():
        r["progress_mae_m"] = ((out["progress"] - lab.progress).abs()[pok].mean() * norm["progress_std"]).item()
    return r


# --------------------------------------------------------------- utilities

def seed_all(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


class Prefetcher:
    """Moves the next few micro-batches to the GPU on a background thread.

    Pinning and the host-to-device copy run on a side CUDA stream while the
    current step computes, so the training loop only waits when the loader
    itself is behind. ``wait_s`` accumulates that waiting.
    """

    def __init__(self, loader, dev, depth: int = 3):
        import queue
        import threading

        self.dev = dev
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.wait_s = 0.0
        self.stream = torch.cuda.Stream() if dev.type == "cuda" else None
        self._error: BaseException | None = None

        def work():
            try:
                for obs, lab in loader:
                    if self.stream is not None:
                        with torch.cuda.stream(self.stream):
                            obs, lab = to_device(obs, lab, dev)
                            ev = torch.cuda.Event()
                            ev.record(self.stream)
                    else:
                        ev = None
                    self.q.put((obs, lab, ev))
            except BaseException as exc:  # surfaced in the training thread
                self._error = exc
            finally:
                self.q.put(None)

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def __iter__(self):
        while True:
            t = time.perf_counter()
            item = self.q.get()
            self.wait_s += time.perf_counter() - t
            if item is None:
                if self._error is not None:
                    raise self._error
                return
            obs, lab, ev = item
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)
                # Tensors made on the side stream are now used on this one.
                for tensor in (*obs, *lab):
                    tensor.record_stream(torch.cuda.current_stream())
            yield obs, lab


def to_device(obs: Observation, lab: Labels, dev) -> tuple[Observation, Labels]:
    if dev.type == "cuda":  # pinned, so the frame copy overlaps compute
        obs = Observation(obs.frames.pin_memory(), obs.speed)
    obs = Observation(*(t.to(dev, non_blocking=True) for t in obs))
    lab = Labels(*(t.to(dev, non_blocking=True) for t in lab))
    return obs, lab


def amp_ctx(cfg: Config, dev):
    if cfg.train.amp == "none" or dev.type != "cuda":
        return torch.autocast(dev.type, enabled=False)
    dtype = torch.bfloat16 if cfg.train.amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def lr_at(step: int, total: int, cfg: Config) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    p = (step - t.warmup_steps) / max(1, total - t.warmup_steps)
    return t.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def build(cfg: Config, manifest: dict, dev):
    model = Model(cfg, len(manifest["waypoint_horizons_s"])).to(dev)
    if cfg.train.channels_last:
        model.encoder.to(memory_format=torch.channels_last)
    if cfg.train.compile:
        # In place (Module.compile), so state_dict keys stay the same.
        model.encoder.compile()
        for blk in model.blocks:
            blk.compile()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "pos" not in n and "queries" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n and "pos" not in n and "queries" not in n)]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay}, {"params": other, "weight_decay": 0.0}],
        lr=cfg.train.lr, betas=(0.9, 0.95), fused=dev.type == "cuda",
    )
    return model, opt


class file_lock:
    """Exclusive lock via O_EXCL on a sibling file; stale after 30 s."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        import os

        deadline = time.time() + 30
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    self.path.unlink(missing_ok=True)  # a crashed holder
                    deadline = time.time() + 30
                time.sleep(0.05)

    def __exit__(self, *exc):
        import os

        os.close(self.fd)
        self.path.unlink(missing_ok=True)


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, row: dict) -> None:
        row = {"time": round(time.time(), 1), **row}
        # The trainer and an external eval watcher both append to this one
        # log; a lock file keeps their lines whole.
        with file_lock(self.path.with_suffix(".lock")):
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        brief = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "time"}
        print(json.dumps(brief), flush=True)


def save_ckpt(path: Path, model, opt, cfg: Config, manifest: dict, state: dict, ema: Ema | None = None) -> None:
    base = {
        "config": cfg.to_dict(),
        "n_horizons": len(manifest["waypoint_horizons_s"]),
        "norm": {k: manifest[k] for k in ("path_mean", "path_std", "progress_mean", "progress_std")},
        "state": state,
    }
    if ema is not None:
        # The twin first: a checkpoint that exists always has its EMA twin.
        ema_path = path.with_name(path.stem + "_ema.pt")
        tmp = ema_path.with_suffix(".tmp")
        torch.save({**base, "model": ema.state_dict_like(model), "ema_of": path.name, "ema_decay": ema.decay}, tmp)
        tmp.replace(ema_path)
    tmp = path.with_suffix(".tmp")
    torch.save({
        **base, "model": model.state_dict(), "opt": opt.state_dict(),
        **({"ema": ema.shadow} if ema is not None else {}),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()},
    }, tmp)
    tmp.replace(path)


def latest_ckpt(ckpt_dir: Path) -> Path | None:
    # Names carry the step as _s<7 digits>; the epoch-end one of a step wins.
    def key(p: Path) -> tuple[int, int]:
        m = re.search(r"_s(\d{7})", p.stem)
        return (int(m.group(1)) if m else -1, p.stem.endswith("_end"))

    cks = sorted((p for p in ckpt_dir.glob("*.pt") if not p.stem.endswith("_ema")), key=key)
    return cks[-1] if cks else None


def run_eval(model, cfg: Config, ckpt_name: str, run_dir: Path, log: RunLog, manifest: dict) -> None:
    from dataclasses import replace

    from .eval.harness import evaluate
    from .policy_model import ModelPolicy

    model.eval()
    try:
        ecfg = replace(cfg.eval, out_dir=str(run_dir / "eval"), corpus=cfg.data.corpus)
        norm = {k: manifest[k] for k in ("path_mean", "path_std", "progress_mean", "progress_std")}
        summary = evaluate(ModelPolicy(model, cfg, ckpt_name, norm=norm), ecfg, checkpoint=ckpt_name, log=print)
        log.write({"kind": "eval", "checkpoint": ckpt_name, **summary})
    except Exception as exc:  # the game must never take training down
        log.write({"kind": "eval_error", "checkpoint": ckpt_name, "error": repr(exc)})
    finally:
        model.train()


# -------------------------------------------------------------------- main

def run(cfg: Config, *, resume: bool = False, max_steps: int | None = None) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.train.run_dir) / cfg.train.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "metrics.jsonl")
    manifest = load_manifest(cfg.data)
    norm = {k: manifest[k] for k in ("path_std", "progress_std")}
    train_runs, _ = split_runs(manifest, cfg.data)

    seed_all(cfg.train.seed, cfg.train.deterministic)
    model, opt = build(cfg, manifest, dev)
    accum = cfg.train.batch_size // cfg.train.micro_batch
    assert accum * cfg.train.micro_batch == cfg.train.batch_size, "batch_size must be a multiple of micro_batch"
    steps_per_epoch = WindowLoader(cfg, train_runs, manifest, 0).total_windows // cfg.train.batch_size
    total_steps = steps_per_epoch * cfg.train.epochs

    ema = Ema(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    state = {"step": 0, "epoch": 0, "windows_in_epoch": 0}
    if resume and (ck := latest_ckpt(ckpt_dir)) is not None:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        # Saved optimizer state has the layout the params had then; with
        # channels_last toggled, fused Adam needs them to match again.
        for group in opt.param_groups:
            for p in group["params"]:
                for k, t in opt.state.get(p, {}).items():
                    if torch.is_tensor(t) and t.shape == p.shape and t.stride() != p.stride():
                        opt.state[p][k] = torch.empty_like(p, dtype=t.dtype).copy_(t)
        state = blob["state"]
        if ema is not None:
            if "ema" in blob:
                ema.shadow = {k: v.to(dev) for k, v in blob["ema"].items()}
            else:  # resuming a run that had no EMA: start it from here
                ema = Ema(model, cfg.train.ema_decay)
        torch.set_rng_state(blob["rng"]["torch"])
        torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
        np.random.set_state(blob["rng"]["numpy"])
        random.setstate(blob["rng"]["python"])
        log.write({"kind": "resume", "from": ck.name, **state})
    else:
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
        log.write({"kind": "start", "params": n_params(model), "encoder_params": n_params(model.encoder),
                   "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "accum": accum,
                   "input_hw": list(cfg.input_hw), "train_runs": len(train_runs)})

    vb = val_batches(cfg, manifest, cfg.train.val_windows)
    w = {"policy": cfg.train.w_policy, "path": cfg.train.w_path, "progress": cfg.train.w_progress,
         "chunk": cfg.train.w_chunk}

    def validate() -> dict:
        model.eval()
        tot: dict[str, float] = {}
        cnt: dict[str, int] = {}
        with torch.no_grad(), amp_ctx(cfg, dev):
            for obs, lab in vb:
                obs, lab = to_device(obs, lab, dev)
                out = model(obs)
                # Some metrics (rare-class recalls) are absent from some batches.
                for k, v in {**{f"loss_{k}": v.item() for k, v in losses(out, lab, cfg.train.label_mode, cfg.train.w_brake).items()}, **metrics(out, lab, norm)}.items():
                    tot[k] = tot.get(k, 0.0) + v
                    cnt[k] = cnt.get(k, 0) + 1
        model.train()
        return {k: tot[k] / cnt[k] for k in tot}

    def checkpoint(tag: str) -> str:
        name = f"{cfg.train.run_name}_e{state['epoch']:02d}_s{state['step']:07d}{tag}"
        save_ckpt(ckpt_dir / f"{name}.pt", model, opt, cfg, manifest, dict(state), ema)
        return name

    model.train()
    t0, seen = time.time(), 0
    running: dict[str, float] = {}
    done = False
    while state["epoch"] < cfg.train.epochs and not done:
        loader = WindowLoader(cfg, train_runs, manifest, state["epoch"], skip_windows=state["windows_in_epoch"])
        feed = Prefetcher(loader, dev)
        micro = 0
        wait_mark = 0.0
        for obs, lab in feed:
            with amp_ctx(cfg, dev):
                out = model(obs)
                ls = losses(out, lab, cfg.train.label_mode, cfg.train.w_brake)
                loss = sum(w[k] * v for k, v in ls.items()) / accum
            loss.backward()
            # Kept on the GPU: a .item() here would sync every micro-step.
            for k, v in ls.items():
                running[k] = running.get(k, 0.0) + v.detach() / accum
            micro += 1
            seen += obs.frames.shape[0]
            state["windows_in_epoch"] += obs.frames.shape[0]
            if micro % accum:
                continue
            lr = lr_at(state["step"], total_steps, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            state["step"] += 1
            step = state["step"]
            if step % cfg.train.log_every == 0:
                dt = time.time() - t0
                row = {"kind": "train", "step": step, "epoch": state["epoch"], "lr": lr, "grad_norm": gnorm.item(),
                       **{f"loss_{k}": v.item() / cfg.train.log_every for k, v in running.items()},
                       **{f"batch_{k}": v for k, v in metrics(out, lab, norm).items()},
                       "windows_per_s": seen / dt, "frames_per_s": seen * cfg.data.window / dt,
                       # Share of the interval the loop spent waiting for data.
                       "data_wait_frac": (feed.wait_s - wait_mark) / dt,
                       "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0}
                log.write(row)
                running, t0, seen = {}, time.time(), 0
                wait_mark = feed.wait_s
            if step % cfg.train.val_every_steps == 0:
                log.write({"kind": "val", "step": step, **validate()})
            if step % cfg.train.ckpt_every_steps == 0:
                checkpoint("")
            if max_steps is not None and step >= max_steps:
                done = True
                break
        if done:
            checkpoint("_stop")
            break
        state["epoch"] += 1
        state["windows_in_epoch"] = 0
        log.write({"kind": "val", "step": state["step"], "epoch_end": state["epoch"], **validate()})
        name = checkpoint("_end")
        k = cfg.train.eval_every_epochs
        if k and state["epoch"] % k == 0:
            if cfg.train.eval_mode == "inline":
                run_eval(model, cfg, name, run_dir, log, manifest)
            else:  # `tmnf-train eval-watch` picks the checkpoint up
                log.write({"kind": "eval_queued", "checkpoint": name})


def measure_vram(cfg: Config, steps: int = 10) -> dict:
    """Peak VRAM for one optimizer step at the configured micro-batch, synthetic data."""
    dev = torch.device("cuda")
    seed_all(cfg.train.seed, cfg.train.deterministic)
    manifest = load_manifest(cfg.data)
    model, opt = build(cfg, manifest, dev)
    h, w = cfg.input_hw
    b, t, c = cfg.train.micro_batch, cfg.data.window, len(cfg.data.context_offsets)
    k = len(manifest["waypoint_horizons_s"])
    n = cfg.data.action_chunk
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 2):  # two warm-up steps (cuDNN autotuning) are not timed
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.time()
        obs = Observation(frames=torch.randint(0, 255, (b, c + t + 1, h, w, 3), dtype=torch.uint8, device=dev),
                          speed=torch.rand(b, t, device=dev) * 300)
        lab = Labels(action=torch.randint(0, 12, (b, t), device=dev),
                     action_soft=torch.softmax(torch.randn(b, t, 12, device=dev), -1), path=torch.randn(b, t, k, 3, device=dev),
                     path_ok=torch.ones(b, t, k, dtype=torch.bool, device=dev), progress=torch.randn(b, t, device=dev),
                     progress_ok=torch.ones(b, t, dtype=torch.bool, device=dev),
                     chunk_soft=torch.softmax(torch.randn(b, t, n, 12, device=dev), -1),
                     chunk_ok=torch.ones(b, t, n, dtype=torch.bool, device=dev))
        with amp_ctx(cfg, dev):
            ls = losses(model(obs), lab, cfg.train.label_mode)
            loss = sum(ls.values())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    return {"input_hw": [h, w], "micro_batch": b, "window": t, "params_m": round(n_params(model) / 1e6, 2),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "sec_per_micro_step": round(dt, 3), "model_frames_per_s": round(b * t / dt, 1)}
