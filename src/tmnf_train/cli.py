"""tmnf-train: data inspection, caching, training and in-game evaluation.

    tmnf-train inspect <corpus>                 step 1 data report
    tmnf-train eval --random                    harness smoke test, random policy
    tmnf-train eval --checkpoint <ckpt.pt>      evaluate a trained checkpoint
    tmnf-train index                            build the label index
    tmnf-train cache                            optional uint8 memmap cache
    tmnf-train bench-loader                     loader frames/sec
    tmnf-train train                            train (evals every K epochs)
    tmnf-train eval-watch                       evaluate a run's checkpoints as they appear
    tmnf-train dashboard                        live progress dashboard on localhost
    tmnf-train serve-policy                     run eval policies for a harness on another machine

Every command takes ``--config file.yaml`` and ``--set section.key=value``.
"""

from __future__ import annotations

import argparse
import json
import sys


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--set", action="append", default=[], metavar="section.key=value")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tmnf-train")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("rest", nargs=argparse.REMAINDER)

    p = sub.add_parser("eval")
    _common(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--random", action="store_true")
    g.add_argument("--checkpoint")
    p.add_argument("--id", help="checkpoint id used in file names")

    for name in ("index", "cache", "bench-loader", "train"):
        p = sub.add_parser(name)
        _common(p)
        if name == "bench-loader":
            p.add_argument("--batches", type=int, default=60)
        if name == "train":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--max-steps", type=int, default=None)
    p = sub.add_parser("vram")
    _common(p)

    p = sub.add_parser("render-paths", help="re-render eval videos with predicted vs actual path")
    p.add_argument("target", help="an eval folder, a map folder inside it, or one rollout .mp4")

    p = sub.add_parser("dashboard", help="live progress dashboard")
    _common(p)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve on every interface (no auth)")
    p.add_argument("--follow", action="store_true", help="show whichever run was updated most recently")

    p = sub.add_parser("eval-watch", help="evaluate a run's epoch checkpoints as they appear")
    _common(p)
    p.add_argument("--poll", type=float, default=60.0, help="seconds between checks")
    p.add_argument("--once", action="store_true", help="evaluate what is there, then exit")

    p = sub.add_parser("serve-policy", help="serve checkpoints to a remote eval harness (eval.device: remote)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9555)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")

    args = ap.parse_args(argv)
    if args.cmd == "serve-policy":
        from .policy_server import serve

        serve(args.host, args.port, args.device)
        return
    if args.cmd == "render-paths":
        from pathlib import Path

        from .eval.render import render

        render(Path(args.target))
        return
    if args.cmd == "inspect":
        from . import inspect_data

        inspect_data.main(args.rest)
        return

    from . import config as config_mod

    cfg = config_mod.load(args.config, args.set)

    if args.cmd == "eval":
        from .eval.harness import evaluate

        if args.random:
            from .eval.policies import RandomPolicy

            policy = RandomPolicy(cfg.eval.random_gas_prob, cfg.eval.random_brake_prob, id=args.id or "random")
        else:
            from .policy_model import load_policy

            # Eval settings come from --config/--set; the model's own from the checkpoint.
            policy = load_policy(args.checkpoint, cfg.eval, id=args.id)
        summary = evaluate(policy, cfg.eval)
        print(json.dumps(summary, indent=1))
    elif args.cmd == "index":
        from .data import index

        index.build(cfg.data, log=print)
    elif args.cmd == "cache":
        from .data import cache

        cache.build(cfg, log=print)
    elif args.cmd == "bench-loader":
        from .data import bench

        print(json.dumps(bench.run(cfg, args.batches), indent=1))
    elif args.cmd == "train":
        from . import train

        train.run(cfg, resume=args.resume, max_steps=args.max_steps)
    elif args.cmd == "dashboard":
        from .dashboard.server import serve

        serve(cfg, port=args.port, host=args.host, follow=args.follow)
    elif args.cmd == "eval-watch":
        from .eval.watch import watch

        watch(cfg, poll_s=args.poll, once=args.once)
    elif args.cmd == "vram":
        from . import train

        print(json.dumps(train.measure_vram(cfg), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
