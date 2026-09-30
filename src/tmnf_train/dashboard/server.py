"""A live progress dashboard for one training run, served on localhost.

    tmnf-train dashboard --config configs/v2.yaml [--port 8765]

Standard library only. Background samplers keep ~1 hour of hardware history:

* NVIDIA GPU: ``nvidia-smi`` every 2 s (utilisation, memory, temperature,
  power, SM clock).
* Every GPU, CPU and RAM: one long-lived PowerShell ``Get-Counter -Continuous``
  (5 s). Per-adapter utilisation is the sum of that adapter's 3D and compute
  engine counters, capped at 100; the adapter whose dedicated memory matches
  nvidia-smi is the NVIDIA one, any other active adapter is shown as the iGPU.

The page polls JSON endpoints; run files are read fresh on each request, so
the dashboard can be started or restarted at any time without affecting the
run. Eval videos are served from the run's eval folder for in-browser playback.
Binds to 127.0.0.1 unless ``--host`` says otherwise (``0.0.0.0`` for every
interface). There is no authentication: anyone who can reach the port can see
the run's metrics and eval videos. Every endpoint is read-only.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..config import Config

HISTORY_S = 3600
PAGE = Path(__file__).with_name("index.html")


# ------------------------------------------------------------------ samplers

class NvidiaSampler(threading.Thread):
    FIELDS = ("utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw", "clocks.sm")

    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.period = period
        self.samples: deque = deque(maxlen=int(HISTORY_S / period))
        self.name = ""

    def run(self) -> None:
        try:
            self.name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return
        while True:
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()[0]
                vals = [float(v) if v.strip() not in ("[N/A]", "N/A") else None for v in out.split(",")]
                self.samples.append({"t": time.time(), **dict(zip(("util", "mem_used", "mem_total", "temp",
                                                                    "power", "clock"), vals))})
            except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                pass
            time.sleep(self.period)


class CounterSampler(threading.Thread):
    """Windows performance counters via one continuous PowerShell process."""

    SCRIPT = (
        "$c = '\\GPU Engine(*engtype_3D)\\Utilization Percentage',"
        "'\\GPU Engine(*engtype_Compute*)\\Utilization Percentage',"
        "'\\GPU Adapter Memory(*)\\Dedicated Usage',"
        "'\\Processor(_Total)\\% Processor Time','\\Memory\\Available MBytes';"
        "Get-Counter -Counter $c -SampleInterval 5 -Continuous -ErrorAction SilentlyContinue | ForEach-Object {"
        " ($_.CounterSamples | ForEach-Object { @{p=$_.Path; v=$_.CookedValue} }) | ConvertTo-Json -Compress }"
    )
    _LUID = re.compile(r"luid_(0x[0-9a-f]+_0x[0-9a-f]+)", re.I)

    def __init__(self, nvidia: NvidiaSampler):
        super().__init__(daemon=True)
        self.nvidia = nvidia
        self.samples: deque = deque(maxlen=HISTORY_S // 5)
        self.total_ram_mb = self._total_ram()

    @staticmethod
    def _total_ram() -> float:
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                                 capture_output=True, text=True, timeout=30).stdout
            return float(out.strip()) / 2**20
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0.0

    def run(self) -> None:
        while True:
            try:
                proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", self.SCRIPT],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.strip()
                    if line.startswith("["):
                        try:
                            self._ingest(json.loads(line))
                        except (ValueError, KeyError):
                            pass
            except OSError:
                pass
            time.sleep(5)  # PowerShell went away; start it again

    def _ingest(self, rows: list[dict]) -> None:
        util: dict[str, float] = {}
        mem: dict[str, float] = {}
        cpu = avail = None
        for r in rows:
            p, v = r["p"].lower(), float(r["v"] or 0)
            m = self._LUID.search(p)
            if "gpu engine" in p and m:
                util[m.group(1)] = util.get(m.group(1), 0.0) + v
            elif "gpu adapter memory" in p and m:
                mem[m.group(1)] = v / 2**20
            elif "processor" in p:
                cpu = v
            elif "available mbytes" in p:
                avail = v
        # Which adapter is the NVIDIA one: dedicated memory closest to nvidia-smi's.
        nv_mem = self.nvidia.samples[-1]["mem_used"] if self.nvidia.samples else None
        nv_luid = None
        if nv_mem is not None and mem:
            nv_luid = min(mem, key=lambda k: abs(mem[k] - nv_mem))
        others = {k: min(100.0, v) for k, v in util.items() if k != nv_luid}
        igpu = max(others.values()) if others else None
        self.samples.append({
            "t": time.time(), "cpu": cpu,
            "ram_used_gb": (self.total_ram_mb - avail) / 1024 if avail is not None and self.total_ram_mb else None,
            "ram_total_gb": self.total_ram_mb / 1024 if self.total_ram_mb else None,
            "igpu_util": igpu,
            "nvidia_util_counters": min(100.0, util.get(nv_luid, 0.0)) if nv_luid else None,
        })


# ------------------------------------------------------------------- run state

def _processes() -> dict:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='TmForever.exe'\" "
             "| Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        rows = json.loads(out) if out else []
        rows = rows if isinstance(rows, list) else [rows]
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    # A venv's python.exe is a launcher that starts the real interpreter with
    # the same command line; count only processes whose parent is not one of them.
    pids = {r.get("ProcessId") for r in rows}
    cmd = [(r.get("Name") or "", r.get("CommandLine") or "") for r in rows
           if not (r.get("ParentProcessId") in pids and (r.get("Name") or "").lower() == "python.exe")]
    return {
        "trainer": sum(1 for n, c in cmd if "tmnf_train" in c and " train" in c),
        "eval_watch": sum(1 for n, c in cmd if "tmnf_train" in c and "eval-watch" in c),
        "games": sum(1 for n, c in cmd if n.lower() == "tmforever.exe"),
    }


class RunState:
    def __init__(self, cfg: Config, follow: bool = False):
        self.cfg = cfg
        self.follow = follow
        self._fixed = Path(cfg.train.run_dir) / cfg.train.run_name
        self._proc = {"t": 0.0, "v": {}}

    @property
    def run_dir(self) -> Path:
        """The configured run, or with ``follow`` the run whose metrics log
        changed most recently (so a queue of runs stays on screen)."""
        if not self.follow:
            return self._fixed
        logs = list(Path(self.cfg.train.run_dir).glob("*/metrics.jsonl"))
        return max(logs, key=lambda p: p.stat().st_mtime).parent if logs else self._fixed

    def run_config(self) -> dict:
        try:
            return json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.cfg.to_dict()

    def rows(self) -> list[dict]:
        path = self.run_dir / "metrics.jsonl"
        out = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue  # a line being written right now
        except OSError:
            pass
        return out

    def processes(self) -> dict:
        if time.time() - self._proc["t"] > 10:
            self._proc = {"t": time.time(), "v": _processes()}
        return self._proc["v"]

    def status(self) -> dict:
        rows = self.rows()
        starts = [r for r in rows if r.get("kind") == "start"]
        train = [r for r in rows if r.get("kind") == "train"]
        total = starts[-1].get("total_steps") if starts else None
        spe = starts[-1].get("steps_per_epoch") if starts else None
        last = train[-1] if train else {}
        # Rate from the last few logged intervals, in steps per second.
        rate = None
        recent = [r for r in train[-6:]]
        if len(recent) >= 2 and recent[-1]["time"] > recent[0]["time"]:
            rate = (recent[-1]["step"] - recent[0]["step"]) / (recent[-1]["time"] - recent[0]["time"])
        step = last.get("step", 0)
        eta = (total - step) / rate if rate and total else None
        ckpts = sorted(p.name for p in (self.run_dir / "checkpoints").glob("*.pt"))
        rc = self.run_config()
        return {
            "run": self.run_dir.name, "run_dir": str(self.run_dir),
            "step": step, "total_steps": total, "steps_per_epoch": spe,
            "epoch": (step / spe) if spe else None, "epochs": rc["train"]["epochs"],
            "frames_per_s": last.get("frames_per_s"), "data_wait_frac": last.get("data_wait_frac"),
            "peak_vram_gb": last.get("peak_vram_gb"), "lr": last.get("lr"),
            "last_log_age_s": time.time() - last["time"] if last else None,
            "eta_s": eta, "checkpoints": ckpts, "processes": self.processes(),
            "label_mode": rc["train"].get("label_mode", "hard"),
        }

    def evals(self) -> list[dict]:
        out = []
        root = self.run_dir / "eval"
        for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            summary = d / "summary.json"
            newest = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=d.stat().st_mtime)
            entry = {"checkpoint": d.name, "done": summary.exists(), "maps": {},
                     # No summary and nothing written for 15 min: an eval that was stopped.
                     "stale": not summary.exists() and time.time() - newest > 900}
            if summary.exists():
                try:
                    entry.update(json.loads(summary.read_text(encoding="utf-8")))
                except ValueError:
                    pass
            for m in sorted(p for p in d.iterdir() if p.is_dir()):
                vids = sorted(v.name for v in (m / "videos").glob("*.mp4")) if (m / "videos").is_dir() else []
                entry.setdefault("videos", {})[m.name] = vids
                if m.name not in entry["maps"]:  # in progress: count finished rollouts
                    rj = m / "rollouts.jsonl"
                    n = sum(1 for _ in rj.open(encoding="utf-8")) if rj.exists() else 0
                    entry["maps"][m.name] = {"label": m.name, "in_progress": True, "rollouts": n}
            out.append(entry)
        return out


# ----------------------------------------------------------------------- http

def make_handler(state: RunState, nv: NvidiaSampler, counters: CounterSampler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _json(self, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            since = float(q.get("since", ["0"])[0])
            if url.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/status":
                self._json(state.status())
            elif url.path == "/api/hw":
                self._json({
                    "nvidia_name": nv.name,
                    "nvidia": [s for s in nv.samples if s["t"] > since],
                    "counters": [s for s in counters.samples if s["t"] > since],
                })
            elif url.path == "/api/metrics":
                keep = ("train", "val", "eval", "eval_error", "resume", "start")
                self._json([r for r in state.rows() if r.get("kind") in keep])
            elif url.path == "/api/evals":
                self._json(state.evals())
            elif url.path.startswith("/video/"):
                self._video(unquote(url.path[len("/video/"):]))
            else:
                self.send_error(404)

        def _video(self, rel: str) -> None:
            root = (state.run_dir / "eval").resolve()
            path = (root / rel).resolve()
            if root not in path.parents or path.suffix != ".mp4" or not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else 0
                end = int(b) if b else size - 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionError, OSError):
                        return
                    left -= len(chunk)

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        import sys

        # A browser that navigates away mid-response is routine, not an error.
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(cfg: Config, port: int = 8765, host: str = "127.0.0.1", follow: bool = False, log=print) -> None:
    nv = NvidiaSampler()
    nv.start()
    counters = CounterSampler(nv)
    counters.start()
    state = RunState(cfg, follow)
    server = _Server((host, port), make_handler(state, nv, counters))
    if host in ("0.0.0.0", "::"):
        import socket

        addrs = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
        log(f"dashboard for {state.run_dir} on port {port}, all interfaces: "
            + ", ".join(f"http://{a}:{port}/" for a in ["127.0.0.1", *addrs]))
    else:
        log(f"dashboard for {state.run_dir} on http://{host}:{port}/")
    server.serve_forever()
