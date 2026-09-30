"""Overnight experiment queue: training here, in-game evals on ser5.

    TMNF_STORAGE=~/tmnf-ml .venv/bin/python scripts/overnight.py

Two lanes that never wait on each other:
  train  one trainer at a time on this machine's GPU, in queue order
  eval   one `tmnf-train eval` at a time in ser5's `tmnf` container, with the
         policy served from this machine (`tmnf-train serve-policy`)

The queue is ``$TMNF_STORAGE/overnight/queue.json``, re-read every loop so
jobs can be appended while it runs:

  {"train": [{"id": "E1", "config": "configs/v2_chunk.yaml", "set": [...],
              "auto_evals": [{"epoch": 2, "tag": "sanity", "set": [...]},
                             {"epoch": "final", "tag": "n12", "ema": true, "set": [...]}]}],
   "eval":  [{"id": "E0", "run": "v2_soft", "checkpoint": "v2_soft_e04_s0015478_end",
              "eval_id": "...", "config": "configs/v2.yaml", "set": [...]}]}

An eval job waits until its checkpoint exists. ``auto_evals`` enqueue evals
of a training run's epoch-end checkpoints (and their ``_ema`` twins) as they
appear. Finished jobs are recorded in ``done.json``; each eval's summary is
appended to the run's metrics.jsonl as an ``eval`` row, like eval-watch does.

Nothing new starts after ``--no-new-after`` (HH:MM); at ``--stop-at`` every
trainer, eval and game this queue started is stopped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import traceback
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tmnf_train.config import canonical_path  # noqa: E402

STORAGE = Path(os.path.expanduser(os.environ["TMNF_STORAGE"]))
QDIR = STORAGE / "overnight"
RUNS = STORAGE / "runs"
SER5 = "ser5.lan"
CONTAINER_STORAGE = "/tmnf-ml"
PY = str(REPO / ".venv" / "bin" / "python")


def now() -> dt.datetime:
    return dt.datetime.now()


def at(hhmm: str) -> dt.datetime:
    """The next occurrence of HH:MM (today or tomorrow)."""
    h, m = map(int, hhmm.split(":"))
    t = now().replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now() - dt.timedelta(hours=12) else t + dt.timedelta(days=1)


def log(msg: str) -> None:
    line = f"{now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        with (RUNS / "overnight.log").open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def append_metrics(run: str, row: dict) -> None:
    from tmnf_train.train import RunLog

    try:
        RunLog(RUNS / run / "metrics.jsonl").write(row)
    except OSError as exc:
        log(f"could not append to {run}/metrics.jsonl: {exc!r}")


# --------------------------------------------------------------- train lane

class TrainLane:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.job: dict | None = None
        self.started = 0.0
        self.stalled = False
        self.retries: dict[str, int] = {}

    def finished_run(self) -> bool:
        """Whether the current job's final epoch-end checkpoint exists."""
        from tmnf_train.config import load

        run = self.run_name(self.job)
        epochs = load(REPO / self.job["config"], self.job.get("set", [])).train.epochs
        return any((RUNS / run / "checkpoints").glob(f"{run}_e{epochs:02d}_s*_end.pt"))

    def check_stall(self, stall_s: float) -> None:
        """Kill a trainer whose metrics log has gone quiet (e.g. a loader
        worker lost a batch and the loop waits for it forever); the queue then
        restarts the same job with --resume."""
        if not self.busy() or time.time() - self.started < stall_s:
            return
        try:
            quiet = time.time() - (RUNS / self.run_name(self.job) / "metrics.jsonl").stat().st_mtime
        except OSError:
            return
        if quiet < stall_s:
            return
        log(f"train {self.job['id']}: no log line for {quiet:.0f}s; killing it to resume")
        self.stalled = True
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        subprocess.run(["pkill", "-9", "-f", f"tmnf_train train --config {self.job['config']}"])

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stray_trainer(self) -> list[tuple[int, str]]:
        """Trainers already running: (pid, command line)."""
        out = subprocess.run(["pgrep", "-af", "tmnf_train train"], capture_output=True, text=True).stdout
        rows = [(int(line.split(" ", 1)[0]), line.split(" ", 1)[1]) for line in out.splitlines() if "--config" in line]
        # Loader workers share the command line; keep the oldest (lowest pid) one.
        return sorted(rows)[:1]

    def adopt(self, pid: int, job: dict) -> None:
        """Track a trainer started by an earlier session of this queue."""
        log(f"train {job['id']}: adopting running trainer pid {pid}")
        self.proc = _Adopted(pid)
        self.job = job
        self.started = time.time()

    def start(self, job: dict) -> None:
        run = self.run_name(job)
        resume = (RUNS / run / "checkpoints").is_dir() and any((RUNS / run / "checkpoints").glob("*.pt"))
        cmd = [PY, "-u", "-m", "tmnf_train", "train", "--config", job["config"]]
        for s in job.get("set", []):
            cmd += ["--set", s]
        if resume:
            cmd.append("--resume")
        out = (RUNS / f"{run}_stdout.log").open("a")
        log(f"train {job['id']}: start {run}{' (resume)' if resume else ''}: {shlex.join(cmd[3:])}")
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
        self.job = job
        self.started = time.time()

    @staticmethod
    def run_name(job: dict) -> str:
        for s in job.get("set", []):
            if s.startswith("train.run_name="):
                return s.split("=", 1)[1]
        sys.path.insert(0, str(REPO / "src"))
        from tmnf_train.config import load

        return load(REPO / job["config"]).train.run_name

    def stop(self) -> None:
        if self.busy():
            log(f"train {self.job['id']}: stopping (time)")
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class _Adopted:
    """Enough of Popen for a process this one did not start."""

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            self.returncode = "?"
            return self.returncode

    def terminate(self):
        try:
            os.kill(self.pid, 15)
        except ProcessLookupError:
            pass

    def wait(self, timeout=None):
        end = time.time() + (timeout or 1e9)
        while self.poll() is None:
            if time.time() > end:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(1)
        return self.returncode

    def kill(self):
        try:
            os.kill(self.pid, 9)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------- eval lane

def container_path(z_path: str) -> str:
    return z_path.replace("Z:/application_storage/tmnf-ml", CONTAINER_STORAGE)


class EvalLane:
    def __init__(self, policy_url: str):
        self.policy_url = policy_url
        self.proc: subprocess.Popen | None = None
        self.job: dict | None = None

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def summary_path(self, job: dict) -> Path:
        return RUNS / job["run"] / "eval" / job["eval_id"] / "summary.json"

    def start(self, job: dict) -> None:
        ck = canonical_path(str(RUNS / job["run"] / "checkpoints" / f"{job['checkpoint']}.pt"))
        out_dir = canonical_path(str(RUNS / job["run"] / "eval"))
        args = ["--config", job.get("config", "configs/v2.yaml"), "--checkpoint", container_path(ck),
                "--id", job["eval_id"],
                "--set", "eval.device=remote", "--set", f"eval.policy_url={self.policy_url}",
                "--set", f"eval.out_dir={out_dir}", "--set", "eval.offscreen=false",
                # 8600/8601 are held by orphaned collector workers in the container.
                "--set", "eval.port=8700"]
        for s in job.get("set", []):
            args += ["--set", s]
        logname = f"{CONTAINER_STORAGE}/runs/{job['run']}_{job['eval_id']}.log"
        script = (
            "#!/bin/bash\nset -o pipefail\ncd /work\n"
            f"export TMNF_STORAGE={CONTAINER_STORAGE}\n"
            f"exec /home/tmnf/venv/bin/python -u -m tmnf_train eval {shlex.join(args)} > {shlex.quote(logname)} 2>&1\n"
        )
        jobdir = RUNS / "overnight_jobs"  # on the NAS, so the container can read it
        jobdir.mkdir(parents=True, exist_ok=True)
        (jobdir / f"{job['id']}.sh").write_text(script)
        log(f"eval {job['id']}: start {job['eval_id']} ({job['run']}/{job['checkpoint']})")
        self.proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", SER5,
             f"docker exec tmnf bash {CONTAINER_STORAGE}/runs/overnight_jobs/{job['id']}.sh"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.job = job
        self.started = time.time()

    def adopt_running(self, queue: dict) -> bool:
        """Track an eval started by an earlier session of this queue (its ssh is still up)."""
        out = subprocess.run(["pgrep", "-af", "overnight_jobs/"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            m = re.search(r"overnight_jobs/(\S+)\.sh", line)
            job = next((j for j in queue["eval"] if m and j["id"] == m.group(1)), None)
            if job is not None and line.split(" ", 1)[1].startswith("ssh"):
                log(f"eval {job['id']}: adopting running eval (ssh pid {line.split()[0]})")
                self.proc = _Adopted(int(line.split()[0]))
                self.job = job
                self.started = time.time()
                return True
        return False

    def finish(self) -> bool:
        """Record the finished job. True if it produced a summary."""
        job = self.job
        rc = self.proc.returncode
        s = read_json(self.summary_path(job), None)
        wall = round(time.time() - self.started)
        if s is None:
            log(f"eval {job['id']}: no summary (rc {rc}, {wall}s)")
            append_metrics(job["run"], {"kind": "eval_error", "checkpoint": job["eval_id"], "error": f"rc {rc}"})
            return False
        brief = {m: (v.get("finish_rate"), (v.get("checkpoints") or {}).get("mean")) for m, v in s.get("maps", {}).items()}
        log(f"eval {job['id']}: done in {wall}s: {brief}")
        append_metrics(job["run"], {"kind": "eval", "checkpoint": job["eval_id"], "device": "remote", **s})
        return True

    def stop(self) -> None:
        if self.busy() or self.job is not None:
            log("eval: stopping the eval and games in the container (time)")
        stop_container()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


def stop_container() -> None:
    subprocess.run(["ssh", "-o", "BatchMode=yes", SER5,
                    "docker exec tmnf bash -c 'pkill -INT -f \"tmnf_train eval\"; sleep 20; "
                    "pkill -f \"tmnf_train eval\"; pkill -f TmForever; pkill -f TMLoader; true'"],
                   capture_output=True, timeout=120)


# ------------------------------------------------------------------ queue

def expand_auto_evals(queue: dict, done: dict) -> None:
    """Enqueue evals of training checkpoints that have appeared."""
    known = {j["id"] for j in queue["eval"]}
    for tj in queue["train"]:
        try:
            run = TrainLane.run_name(tj)
        except Exception as exc:
            log(f"train {tj['id']}: cannot read its config ({exc!r}); skipping its auto evals")
            continue
        ck_dir = RUNS / run / "checkpoints"
        if not ck_dir.is_dir():
            continue
        from tmnf_train.config import load

        epochs = load(REPO / tj["config"], tj.get("set", [])).train.epochs
        for ae in tj.get("auto_evals", []):
            epoch = epochs if ae["epoch"] == "final" else int(ae["epoch"])
            for ck in sorted(ck_dir.glob(f"{run}_e{epoch:02d}_s*_end.pt")):
                names = [ck.stem] + ([ck.stem + "_ema"] if ae.get("ema") else [])
                for name in names:
                    jid = f"{tj['id']}-{name}-{ae['tag']}"
                    if jid in known or jid in done:
                        continue
                    queue["eval"].append({
                        "id": jid, "run": run, "checkpoint": name, "eval_id": f"{name}_{ae['tag']}",
                        "config": tj["config"], "set": ae.get("set", []), "auto": True})
                    known.add(jid)
                    log(f"queued eval {jid}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-url", default="10.0.0.201:9555")
    ap.add_argument("--no-new-after", default="07:50", help="no new eval starts after this (HH:MM)")
    ap.add_argument("--no-train-after", default="07:15", help="no new training starts after this")
    ap.add_argument("--stop-at", default="08:15", help="stop everything (HH:MM)")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--stall-s", type=float, default=600.0, help="restart a trainer silent this long")
    args = ap.parse_args()

    QDIR.mkdir(parents=True, exist_ok=True)
    lock = (QDIR / "overnight.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another overnight.py holds the lock")
    stop_at, no_new, no_train = at(args.stop_at), at(args.no_new_after), at(args.no_train_after)
    log(f"overnight queue up; no new train after {no_train:%H:%M}, no new eval after {no_new:%H:%M}, "
        f"stop at {stop_at:%H:%M}")

    train, ev = TrainLane(), EvalLane(args.policy_url)
    first = True
    while True:
        try:
            if step(args, train, ev, stop_at, no_new, no_train, first):
                return
        except Exception:  # a bad queue entry or a flaky NAS must not end the night
            log("error in the queue loop:\n" + traceback.format_exc())
        first = False
        time.sleep(args.poll)


def step(args, train: TrainLane, ev: EvalLane, stop_at, no_new, no_train, first: bool) -> bool:
    """One pass over both lanes. True when the queue is finished."""
    queue = read_json(QDIR / "queue.json", {"train": [], "eval": []})
    queue.setdefault("train", [])
    queue.setdefault("eval", [])
    done = read_json(QDIR / "done.json", {})

    if now() >= stop_at:
        train.stop()
        ev.stop()
        log("stop time reached; queue stopped")
        return True
    if first and ev.job is None:
        ev.adopt_running(queue)

    # train lane
    train.check_stall(args.stall_s)
    if train.job is not None and not train.busy():
        rc = train.proc.returncode
        if train.stalled:
            log(f"train {train.job['id']}: stalled trainer stopped; resuming it")
            train.stalled = False
        elif not train.finished_run() and train.retries.get(train.job["id"], 0) < 3:
            # Died before its last epoch (e.g. the NAS mount dropped under it).
            train.retries[train.job["id"]] = train.retries.get(train.job["id"], 0) + 1
            log(f"train {train.job['id']}: exited rc {rc} before its final checkpoint; resuming it "
                f"(retry {train.retries[train.job['id']]})")
        else:
            done[train.job["id"]] = {"kind": "train", "rc": rc, "at": f"{now():%H:%M}"}
            log(f"train {train.job['id']}: exited rc {rc}")
        train.job = None
    if train.job is None and now() < no_train:
        nxt = next((j for j in queue["train"] if j["id"] not in done and not j.get("hold")), None)
        if nxt is not None:
            stray = train.stray_trainer()
            if stray and f"--config {nxt['config']}" in stray[0][1]:
                train.adopt(stray[0][0], nxt)
            elif stray:
                log("a trainer is already running outside the queue; waiting")
            else:
                train.start(nxt)

    # eval lane
    expand_auto_evals(queue, done)
    write_json(QDIR / "queue.json", queue)
    if ev.job is not None and not ev.busy():
        ok = ev.finish()
        done[ev.job["id"]] = {"kind": "eval", "ok": ok, "at": f"{now():%H:%M}"}
        ev.job = None
    if ev.job is None and now() < no_new:
        for j in queue["eval"]:
            if j["id"] in done or j.get("hold"):
                continue
            if ev.summary_path(j).exists():  # done in an earlier session
                done[j["id"]] = {"kind": "eval", "ok": True, "at": "earlier"}
                continue
            if (RUNS / j["run"] / "checkpoints" / f"{j['checkpoint']}.pt").exists():
                ev.start(j)
                break
    write_json(QDIR / "done.json", done)

    if (train.job is None and ev.job is None and now() >= no_new):
        log("nothing running and nothing may start; queue finished")
        return True
    return False


if __name__ == "__main__":
    main()
