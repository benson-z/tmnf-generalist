"""Evaluate a training run's checkpoints from a separate process.

    tmnf-train eval-watch --config configs/v2_chunk_path3d_c23.yaml

With ``train.eval_mode: external`` the trainer writes a checkpoint at the end
of every epoch and carries on; this watcher evaluates each epoch-end
checkpoint whose epoch is a multiple of ``train.eval_every_epochs``, in order,
and appends the summary to the run's ``metrics.jsonl`` (the same single log,
under a lock). Run it in the ROCm environment and the model and eval stay off
the GPU that is training; see README-train.md.

Evaluated checkpoints are remembered by the ``eval/<checkpoint>/summary.json``
they leave behind, so restarting the watcher does not redo them. It exits by
itself once the final epoch's checkpoint has been evaluated.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path

from ..config import Config
from .harness import evaluate

_END = re.compile(r"_e(\d+)_s\d+_end$")


def pending(cfg: Config, run_dir: Path) -> list[tuple[int, Path]]:
    k = max(1, cfg.train.eval_every_epochs)
    out = []
    for ck in sorted((run_dir / "checkpoints").glob("*_end.pt")):
        m = _END.search(ck.stem)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch % k == 0 and not (run_dir / "eval" / ck.stem / "summary.json").exists():
            out.append((epoch, ck))
    return sorted(out)


def watch(cfg: Config, *, poll_s: float = 60.0, once: bool = False, log=print) -> None:
    from ..policy_model import load_policy
    from ..train import RunLog

    run_dir = Path(cfg.train.run_dir) / cfg.train.run_name
    runlog = RunLog(run_dir / "metrics.jsonl")
    ecfg = replace(cfg.eval, out_dir=str(run_dir / "eval"), corpus=cfg.data.corpus)
    log(f"watching {run_dir / 'checkpoints'} (eval every {cfg.train.eval_every_epochs} epoch(s), "
        f"device {cfg.eval.device})")
    while True:
        todo = pending(cfg, run_dir)
        # A checkpoint can be seen before its writer has renamed it into place;
        # only complete .pt files match the glob, so it is safe to load.
        for epoch, ck in todo:
            log(f"evaluating {ck.name}")
            policy = load_policy(str(ck), ecfg, id=ck.stem)
            try:
                summary = evaluate(policy, ecfg, checkpoint=ck.stem, log=log)
                runlog.write({"kind": "eval", "checkpoint": ck.stem, "device": str(policy.device), **summary})
            except Exception as exc:  # keep watching; the game must not end the watch
                runlog.write({"kind": "eval_error", "checkpoint": ck.stem, "error": repr(exc)})
            del policy
            if epoch >= cfg.train.epochs:
                return
        if once:
            return
        time.sleep(poll_s)
