"""Run the eval policy on one machine and the game on another.

    tmnf-train serve-policy --port 9555            on the machine with the GPU
    eval.device: remote, eval.policy_url: host:port  on the machine with the game

The game box (the ser5 Wine container) has no GPU that can run the model at
20 Hz, so the harness there sends each captured frame over TCP and gets the
action back; the model runs here exactly as it would locally (the same
``ModelPolicy`` / ``ModelEpisode``, so the same seed gives the same actions).

Checkpoint paths travel in their machine-independent ``Z:/application_storage/
tmnf-ml/...`` form (see ``config.canonical_path``) and each side maps them
onto its own storage with ``TMNF_STORAGE``.

Wire format, both directions: ``u32 header length, JSON header, u32 body
length, body`` (little-endian). One TCP connection per episode:

    {"op": "open", checkpoint, seed, temperature, temperature_controls,
     action_source, inference}                     -> {"id", "device"}
    {"op": "act", "speed", "shape"} + RGB bytes    -> {"action", "aux"} + float32 probs
    {"op": "info", checkpoint, ...}                -> {"id", "device", "n_chunk"}  (then closes)
"""

from __future__ import annotations

import copy
import json
import socket
import socketserver
import struct
import threading
import time
from collections import OrderedDict

import numpy as np

from .config import canonical_path, storage_path

_HDR = struct.Struct("<I")


# ------------------------------------------------------------------ wire

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, body: bytes = b"") -> None:
    h = json.dumps(header).encode()
    sock.sendall(_HDR.pack(len(h)) + h + _HDR.pack(len(body)) + body)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    (n,) = _HDR.unpack(_recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    (m,) = _HDR.unpack(_recv_exact(sock, 4))
    return header, _recv_exact(sock, m) if m else b""


# ---------------------------------------------------------------- server

class _Policies:
    """Loaded checkpoints, most recently used last; a few stay resident."""

    def __init__(self, device: str, keep: int = 3):
        self.device = device
        self.keep = keep
        self.lock = threading.Lock()
        self.cache: OrderedDict[tuple[str, str], object] = OrderedDict()

    def get(self, checkpoint: str, inference: str):
        from .policy_model import ModelPolicy

        path = storage_path(checkpoint)
        key = (path, inference)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            policy = ModelPolicy.from_checkpoint(path, inference=inference, device=self.device)
            self.cache[key] = policy
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
            return policy


def _configured(policies: _Policies, req: dict):
    """The requested checkpoint with this request's sampling settings.

    A shallow copy shares the loaded model, so two lanes can sample the same
    checkpoint differently at the same time."""
    base = policies.get(req["checkpoint"], req.get("inference", "kv_cache"))
    policy = copy.copy(base)
    policy.set_sampling(req.get("temperature_controls"), req.get("action_source", "policy"))
    if req.get("id"):
        policy.id = req["id"]
    return policy


def serve(host: str = "0.0.0.0", port: int = 9555, device: str = "auto", log=print) -> None:
    policies = _Policies(device)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            sock: socket.socket = self.request
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            peer = f"{self.client_address[0]}:{self.client_address[1]}"
            episode = None
            steps, t0 = 0, time.monotonic()
            try:
                while True:
                    try:
                        req, body = recv_msg(sock)
                    except ConnectionError:
                        return
                    op = req.get("op")
                    try:
                        if op == "info":
                            policy = _configured(policies, req)
                            send_msg(sock, {"id": policy.id, "device": str(policy.device),
                                            "n_chunk": policy.model.n_chunk})
                        elif op == "open":
                            policy = _configured(policies, req)
                            episode = policy.episode(int(req["seed"]), float(req["temperature"]))
                            log(f"[{peer}] open {req['checkpoint']} seed={req['seed']} T={req['temperature']} "
                                f"controls={req.get('temperature_controls')} source={req.get('action_source', 'policy')}")
                            send_msg(sock, {"id": policy.id, "device": str(policy.device)})
                        elif op == "act":
                            if episode is None:
                                raise RuntimeError("act before open")
                            frame = np.frombuffer(body, dtype=np.uint8).reshape(req["shape"])
                            action, probs = episode.act(frame, float(req["speed"]))
                            steps += 1
                            send_msg(sock, {"action": int(action), "aux": episode.aux},
                                     np.asarray(probs, np.float32).tobytes() if probs is not None else b"")
                        else:
                            raise ValueError(f"unknown op {op!r}")
                    except (ConnectionError, OSError):
                        raise
                    except Exception as exc:  # reported to the client, connection kept
                        send_msg(sock, {"error": repr(exc)})
            except (ConnectionError, OSError):
                pass
            finally:
                if steps:
                    dt = time.monotonic() - t0
                    log(f"[{peer}] closed after {steps} steps ({steps / max(dt, 1e-9):.1f} steps/s)")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((host, port), Handler) as srv:
        log(f"serving policies on {host}:{port} (device {device})")
        srv.serve_forever()


# ---------------------------------------------------------------- client

def _address(url: str) -> tuple[str, int]:
    host, _, port = url.rpartition(":")
    return host, int(port)


class RemotePolicy:
    """An eval policy whose model runs in ``tmnf-train serve-policy`` elsewhere."""

    def __init__(self, checkpoint: str, cfg_eval, id: str | None = None, timeout_s: float = 60.0):
        from pathlib import Path

        self.addr = _address(cfg_eval.policy_url)
        self.timeout_s = timeout_s
        self.request = {
            "checkpoint": canonical_path(checkpoint),
            "inference": cfg_eval.inference,
            "temperature_controls": cfg_eval.temperature_controls,
            "action_source": cfg_eval.action_source,
            "id": id or Path(checkpoint).stem,
        }
        # Load it on the server now, so a bad path or setting fails here and
        # not inside the first rollout.
        with self._connect() as sock:
            send_msg(sock, {"op": "info", **self.request})
            reply, _ = recv_msg(sock)
        if "error" in reply:
            raise RuntimeError(f"policy server {cfg_eval.policy_url}: {reply['error']}")
        self.id = reply["id"]
        self.device = f"remote:{cfg_eval.policy_url}/{reply['device']}"

    def _connect(self) -> socket.socket:
        sock = socket.create_connection(self.addr, timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def episode(self, seed: int, temperature: float) -> "RemoteEpisode":
        return RemoteEpisode(self, seed, temperature)


class RemoteEpisode:
    def __init__(self, policy: RemotePolicy, seed: int, temperature: float):
        self.sock = policy._connect()
        self.aux: dict | None = None
        send_msg(self.sock, {"op": "open", "seed": int(seed), "temperature": float(temperature), **policy.request})
        reply, _ = recv_msg(self.sock)
        if "error" in reply:
            self.sock.close()
            raise RuntimeError(f"policy server: {reply['error']}")

    def act(self, frame: np.ndarray, speed_kmh: float) -> tuple[int, np.ndarray | None]:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        send_msg(self.sock, {"op": "act", "speed": float(speed_kmh), "shape": list(frame.shape)}, frame.tobytes())
        reply, body = recv_msg(self.sock)
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        self.aux = reply.get("aux")
        probs = np.frombuffer(body, dtype=np.float32).copy() if body else None
        return int(reply["action"]), probs

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def __del__(self) -> None:
        self.close()
