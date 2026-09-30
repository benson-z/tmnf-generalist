"""Closed-loop eval rollouts on a list of maps, each recorded to its own video.

The game is driven through the collection harness (``tmnf_collect``): the same
launcher, plugin, session sequence and frame capture. The plugin's drive mode
holds every 20 Hz sample tick until the policy answers the captured frame with
an action, which is then held for the five physics ticks up to the next sample
-- the same 50 ms step the labels are built on.

A rollout ends at the finish, at a race-time timeout, or once the car has been
effectively stationary for a few seconds. The harness never respawns or
restarts for the policy. A game that crashes or hangs is killed and relaunched,
and the rollout is re-run from scratch with the same seed.

Maps are stock campaign maps (``B01-Race``) or corpus runs
(``corpus:tmx-1016190``). On corpus maps each rollout also logs its distance
from that run's recorded trajectory, a diagnostic that separates "cannot
generalise to a new map" from "cannot recover once off the line".

Outputs under ``<out_dir>/<checkpoint id>/``:
    summary.json                 per-map summaries
    <map label>/rollouts.jsonl   one metrics row per rollout
    <map label>/summary.json     spread across that map's rollouts
    <map label>/videos/<ckpt>_<map>_r<k>_<outcome>.mp4
    <map label>/steps/<ckpt>_<map>_r<k>.jsonl   per-step action, probabilities, speed
    <map label>/replays/         any run the game autosaved (a finish)
"""

from __future__ import annotations

import json
import queue
import shutil
import statistics
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tmnf_collect.collect import install, staging
from tmnf_collect.collect.protocol import (
    EV_CHECKPOINT,
    EV_FINISH,
    EV_GAMESTATE,
    Event,
    Sample,
    Tick,
)
from tmnf_collect.collect.session import Session, SessionError
from tmnf_collect.common.frames import HEVC, FrameWriteError, VideoEncoder
from tmnf_collect.common.paths import detect
from tmnf_collect.common.replays import ReplayError, read_replay

from .. import actions
from ..config import EvalConfig
from .maps import DemoLine, MapInfo, resolve
from .projection import draw_path
from .policies import Policy

OVERLAY_H = 28
HIDDEN_SUFFIX = ".tmnf-train-hidden"


# ----------------------------------------------------------------- ghosts

def _autosaves(layout, map_uid: str) -> list[Path]:
    """The game's personal-best replays for a map, which it shows as a ghost.

    They live in Tracks/Replays/Autosaves, which every instance shares. Matched
    by the map UID in each replay's header rather than by file name, since map
    names carry formatting codes and characters the game rewrites.
    """
    folder = layout.replays_dir / "Autosaves"
    if not folder.is_dir():
        return []
    found = []
    for p in folder.iterdir():
        if p.is_file() and p.name.lower().endswith(".replay.gbx"):
            try:
                if read_replay(p).map_uid == map_uid:
                    found.append(p)
            except ReplayError:
                continue
    return found


def restore_ghosts(layout) -> None:
    """Put back any autosaves an earlier (possibly crashed) eval set aside."""
    folder = layout.replays_dir / "Autosaves"
    if folder.is_dir():
        for p in folder.glob(f"*{HIDDEN_SUFFIX}"):
            target = p.with_name(p.name[: -len(HIDDEN_SUFFIX)])
            if target.exists():  # a newer one was written meanwhile; keep the user's
                target.unlink()
            p.rename(target)


def hide_ghosts(layout, map_uid: str) -> list[Path]:
    """Rename the map's autosaves so no ghost drives next to the policy.

    The training corpus was recorded without ghosts. Renamed in place (the game
    only lists .Replay.gbx), and put back by ``restore_ghosts``.
    """
    hidden = []
    for p in _autosaves(layout, map_uid):
        target = p.with_name(p.name + HIDDEN_SUFFIX)
        p.rename(target)
        hidden.append(target)
    return hidden


def collect_new_ghosts(layout, map_uid: str, dest: Path) -> list[str]:
    """Move autosaves a rollout created (the policy finished) out of the game's
    reach, into the eval folder, so the next rollout has no ghost either."""
    moved = []
    for p in _autosaves(layout, map_uid):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), dest)
        moved.append(str(dest))
    return moved


class GameHung(RuntimeError):
    pass


@dataclass
class RolloutResult:
    checkpoint: str
    rollout: int
    seed: int
    temperature: float
    map: str
    end_reason: str  # finish | timeout | stationary | time_limit | harness_error
    finished: bool = False
    finish_time_s: float | None = None
    time_over_author: float | None = None
    checkpoints: int = 0
    checkpoint_times_s: list[float] = field(default_factory=list)
    elapsed_race_s: float = 0.0
    final_position: list[float] | None = None
    final_speed_kmh: int = 0
    steps: int = 0
    attempts: int = 1
    wall_s: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)
    # Ticks whose applied keys differ from the action sent for their step.
    # Should be 0: it is how the harness proves the actions reach the car.
    action_tick_mismatch: int = 0
    ticks_checked: int = 0
    video: str | None = None
    error: str | None = None
    # Corpus maps only: distance from the recorded run's trajectory (m).
    offline_mean_m: float | None = None
    offline_max_m: float | None = None
    time_to_leave_line_s: float | None = None  # first time > eval.off_line_m away
    demo_fraction_reached: float | None = None  # furthest point along the demo line before first leaving it


# ----------------------------------------------------------------- video

def _font(size: int):
    for name in ("consola.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT = None


def overlay(frame: np.ndarray, race_ms: int, speed: int, action: int, prob: float | None) -> np.ndarray:
    """The frame with a strip underneath: time, speed, chosen action and keys."""
    global _FONT
    if _FONT is None:
        _FONT = _font(14)
    h, w, _ = frame.shape
    canvas = Image.new("RGB", (w, h + OVERLAY_H), (16, 16, 20))
    canvas.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(canvas)
    p = f" p={prob:.2f}" if prob is not None else ""
    d.text((4, h + 6), f"{race_ms / 1000:6.2f}s {speed:3d}km/h a{action:02d}{p}", font=_FONT, fill=(236, 238, 242))
    up, down, left, right = actions.keys(action)
    x0 = w - 4 * 22 - 4
    for k, (label, on, col) in enumerate(
        (("<", left, (90, 200, 250)), ("^", up, (90, 200, 250)), ("v", down, (250, 110, 110)), (">", right, (90, 200, 250)))
    ):
        box = (x0 + k * 22, h + 4, x0 + k * 22 + 19, h + 23)
        d.rectangle(box, fill=col if on else None, outline=col if on else (70, 74, 82), width=1)
        d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font=_FONT,
               fill=(16, 16, 20) if on else (150, 156, 168), anchor="mm")
    return np.asarray(canvas)


def _bgra(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., 0] = rgb[..., 2]
    out[..., 1] = rgb[..., 1]
    out[..., 2] = rgb[..., 0]
    out[..., 3] = 255
    return out.tobytes()


def sample_rgb(sample: Sample) -> np.ndarray:
    """The plugin's BGRA capture as an RGB array, as the dataset stores it."""
    bgra = np.frombuffer(sample.pixels, dtype=np.uint8).reshape(sample.height, sample.width, 4)
    return np.ascontiguousarray(bgra[..., 2::-1])


# --------------------------------------------------------------- rollout

def run_rollout(
    session: Session,
    staged: str,
    policy: Policy,
    result: RolloutResult,
    cfg: EvalConfig,
    map_info: MapInfo,
    video_dir: Path,
    steps_dir: Path,
    line: DemoLine | None = None,
    deadline: float | None = None,
) -> RolloutResult:
    """Drive one rollout on an already-running game. Raises on game trouble."""
    ctrl = session.controller
    assert ctrl is not None
    episode = policy.episode(result.seed, result.temperature)
    stem = f"{result.checkpoint}_{map_info.label}_r{result.rollout:02d}"
    tmp_video = video_dir / f".{stem}.partial.mp4"
    encoder = VideoEncoder(tmp_video, fps=1000 / session.period_ms, codec=HEVC)
    steps_path = steps_dir / f"{stem}.jsonl"
    offline: list[float] = []
    reached = 0.0
    steps_file = steps_path.open("w", encoding="utf-8")

    stationary_limit = int(round(cfg.stationary_s * 1000))
    # Long maps get time to finish: 1.5x the author time when that is longer.
    timeout_ms = int(round(max(cfg.timeout_s * 1000, cfg.timeout_author_factor * map_info.author_ms)))
    # Real time runs longer than race time (inference, rendering): 1.5x margin.
    wall_cap = max(cfg.rollout_max_wall_s, 1.5 * timeout_ms / 1000) if cfg.rollout_max_wall_s else None
    sent: dict[int, int] = {}  # sample race time -> action mask sent
    counts: Counter[str] = Counter()
    slow_since: int | None = 0
    last: Sample | None = None
    started = time.monotonic()
    try:
        session.start_driven_run(staged)
        while True:
            message = ctrl.poll(timeout=cfg.step_wall_timeout_s)
            if message is None:
                raise GameHung(f"no message for {cfg.step_wall_timeout_s:.0f}s")
            if isinstance(message, Event):
                if message.kind == EV_GAMESTATE:
                    session.game_state = message.arg
                elif message.kind == EV_CHECKPOINT and last is not None:
                    if message.arg > result.checkpoints:
                        result.checkpoints = message.arg
                        result.checkpoint_times_s.append(message.race_time / 1000)
                elif message.kind == EV_FINISH and last is not None:
                    result.end_reason = "finish"
                    result.finished = True
                    result.finish_time_s = message.race_time / 1000
                    result.time_over_author = message.race_time / map_info.author_ms
                    result.elapsed_race_s = message.race_time / 1000
                    break
                continue
            if isinstance(message, Tick):
                step = message.race_time - message.race_time % session.period_ms
                if step in sent and message.race_time != step:
                    mask = message.up | message.down << 1 | message.left << 2 | message.right << 3
                    result.ticks_checked += 1
                    result.action_tick_mismatch += int(mask != sent[step])
                continue
            if not isinstance(message, Sample):
                continue

            t = message.race_time
            if last is None and t != 0:
                raise SessionError(f"driven run started at race time {t}, not 0")
            last = message
            result.elapsed_race_s = t / 1000
            result.final_position = [round(v, 3) for v in message.position]
            result.final_speed_kmh = message.display_speed
            dist = None
            if line is not None:
                dist, frac = line.nearest(message.position)
                offline.append(dist)
                if result.time_to_leave_line_s is None:
                    # Progress counts only until the car first leaves the line:
                    # rejoining it later (maps often loop back past the start)
                    # is not progress.
                    if dist <= cfg.off_line_m:
                        reached = max(reached, frac)
                    else:
                        result.time_to_leave_line_s = t / 1000

            if message.display_speed >= cfg.stationary_kmh:
                slow_since = None
            elif slow_since is None:
                slow_since = t
            if slow_since is not None and t - slow_since >= stationary_limit:
                result.end_reason = "stationary"
                break
            if t >= timeout_ms:
                result.end_reason = "timeout"
                break
            now = time.monotonic()
            if (deadline is not None and now >= deadline) or (
                wall_cap and now - started >= wall_cap
            ):
                result.end_reason = "time_limit"  # a wall-clock limit, not the race timeout
                break

            frame = sample_rgb(message)
            action, probs = episode.act(frame, float(message.display_speed))
            up, down, left, right = actions.keys(action)
            ctrl.act(up, down, left, right)
            sent[t] = int(up) | int(down) << 1 | int(left) << 2 | int(right) << 3
            counts[actions.name(action)] += 1
            result.steps += 1

            p = float(probs[action]) if probs is not None else None
            aux = getattr(episode, "aux", None)  # path head, metres; None for the random policy
            pose = {
                "position": list(message.position), "yaw_pitch_roll": list(message.yaw_pitch_roll),
                "camera_position": list(message.camera_position),
                "camera_yaw_pitch_roll": list(message.camera_yaw_pitch_roll), "camera_fov": message.camera_fov,
            }
            drawn = frame
            if aux is not None and cfg.overlay_path:
                img = draw_path(Image.fromarray(frame), pose, np.asarray(aux["mean"])[:, :2],
                                np.asarray(aux["std"])[:, 0], aux["horizons_s"])
                drawn = np.asarray(img)
            shown = overlay(drawn, t, message.display_speed, action, p) if cfg.overlay else drawn
            encoder.add(_bgra(shown), shown.shape[1], shown.shape[0])
            steps_file.write(json.dumps({
                "t": t, "speed": message.display_speed, "action": action, "name": actions.name(action),
                "pos": [round(v, 2) for v in message.position],
                "yaw": round(message.yaw_pitch_roll[0], 4),
                # The game's own slide flag and car-space velocity (m/s), to
                # measure drifting against the demos' samples.
                "sliding": bool(message.sliding),
                "local_speed": [round(v, 2) for v in message.local_speed],
                "cam": [[round(v, 3) for v in message.camera_position],
                        [round(v, 4) for v in message.camera_yaw_pitch_roll], round(message.camera_fov, 2)],
                **({"path": {"mean": np.round(aux["mean"], 2).tolist(), "std": np.round(aux["std"], 2).tolist(),
                             "horizons_s": aux["horizons_s"]}} if aux is not None else {}),
                **({"off_line_m": round(dist, 2)} if dist is not None else {}),
                "p": None if probs is None else [round(float(x), 4) for x in probs],
            }) + "\n")
    finally:
        steps_file.close()
        try:
            session.stop_driven_run()
        except (OSError, ConnectionError, SessionError):
            pass
        try:
            encoder.close()
        except FrameWriteError:
            tmp_video.unlink(missing_ok=True)
            raise

    result.action_counts = dict(counts)
    result.wall_s = round(time.monotonic() - started, 2)
    if offline:
        result.offline_mean_m = round(float(np.mean(offline)), 2)
        result.offline_max_m = round(float(np.max(offline)), 2)
        result.demo_fraction_reached = round(1.0 if result.finished else reached, 3)
    outcome = result.end_reason
    if result.finished:
        outcome += f"-{result.finish_time_s:.2f}s"
    else:
        outcome += f"-cp{result.checkpoints}"
    final = video_dir / f"{stem}_{outcome}.mp4"
    if tmp_video.exists():
        tmp_video.replace(final)
        result.video = str(final)
    session.leave_map()
    return result


# ----------------------------------------------------------------- lanes

def _new_session(cfg: EvalConfig, lane: int) -> Session:
    session = Session(
        port=cfg.port + lane, instance_id=lane, width=320, height=240, period_ms=50,
        hide_ui=True, camera=1, speed=cfg.speed, offscreen=cfg.offscreen,
    )
    try:
        session.start()
        session.prepare()
    except BaseException:
        session.close()
        raise
    return session


@dataclass
class MapJob:
    """One map's share of an eval: where it is staged and where output goes."""

    info: MapInfo
    staged: str
    out: Path
    line: DemoLine | None

    @property
    def video_dir(self) -> Path:
        return self.out / "videos"

    @property
    def steps_dir(self) -> Path:
        return self.out / "steps"


def _lane(
    lane: int,
    jobs: queue.Queue,
    results: list[RolloutResult],
    lock: threading.Lock,
    policy: Policy,
    cfg: EvalConfig,
    log,
    deadline: float | None = None,
) -> None:
    session: Session | None = None
    layout = detect()
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return  # budget spent: leave the remaining rollouts unstarted
            try:
                mj, base = jobs.get_nowait()
            except queue.Empty:
                return
            result = None
            for attempt in range(1, cfg.max_attempts + 1):
                if attempt > 1 and deadline is not None and time.monotonic() >= deadline:
                    break  # no relaunch-and-retry once the budget is spent
                try:
                    if session is None:
                        session = _new_session(cfg, lane)
                    fresh = RolloutResult(**{**asdict(base), "attempts": attempt})
                    try:
                        result = run_rollout(session, mj.staged, policy, fresh, cfg, mj.info,
                                             mj.video_dir, mj.steps_dir, mj.line, deadline)
                    finally:
                        # The game autosaves a finished run as a PB ghost.
                        collect_new_ghosts(layout, mj.info.uid, mj.out / "replays" /
                                           f"{base.checkpoint}_{mj.info.label}_r{base.rollout:02d}.Replay.gbx")
                    break
                except Exception as exc:  # game crash, hang, protocol trouble
                    log(f"[lane {lane}] {mj.info.label} r{base.rollout:02d} attempt {attempt} failed: {exc!r}; "
                        "relaunching game")
                    if session is not None:
                        session.close()
                        session = None
                    result = RolloutResult(**{**asdict(base), "attempts": attempt,
                                              "end_reason": "harness_error", "error": repr(exc)})
            assert result is not None
            with lock:
                results.append(result)
                with (mj.out / "rollouts.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
            extra = ""
            if result.offline_mean_m is not None:
                extra = (f" off-line mean={result.offline_mean_m}m left@{result.time_to_leave_line_s}s "
                         f"reached={result.demo_fraction_reached}")
            log(f"[lane {lane}] {mj.info.label} r{result.rollout:02d} {result.end_reason} "
                f"t={result.elapsed_race_s:.2f}s cp={result.checkpoints} steps={result.steps} "
                f"wall={result.wall_s}s mismatch={result.action_tick_mismatch}/{result.ticks_checked}{extra}")
    finally:
        if session is not None:
            session.close()


def _spread(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {"mean": round(statistics.fmean(xs), 3), "std": round(statistics.pstdev(xs), 3),
            "min": round(min(xs), 3), "max": round(max(xs), 3)}


def summarize(results: list[RolloutResult], map_info: MapInfo, planned: int | None = None) -> dict:
    # A rollout cut by the eval's time limit says nothing about how it would
    # have ended, so it is left out of the rates and spreads (but counted).
    done = [r for r in results if r.end_reason not in ("harness_error", "time_limit")]
    fin = [r for r in done if r.finished]
    out = {
        "map": map_info.name,
        "label": map_info.label,
        "split": map_info.split,
        "author_s": map_info.author_ms / 1000, "gold_s": map_info.gold_ms / 1000,
        "silver_s": map_info.silver_ms / 1000, "bronze_s": map_info.bronze_ms / 1000,
        "rollouts": len(results),
        "rollouts_planned": planned if planned is not None else len(results),
        "rollouts_not_started": (planned - len(results)) if planned is not None else 0,
        "cut_by_time_limit": sum(r.end_reason == "time_limit" for r in results),
        "harness_errors": sum(r.end_reason == "harness_error" for r in results),
        "finish_rate": round(len(fin) / len(done), 3) if done else None,
        "end_reasons": dict(Counter(r.end_reason for r in results)),
        "finish_time_s": _spread([r.finish_time_s for r in fin]),
        "time_over_author": _spread([r.time_over_author for r in fin]),
        "checkpoints": _spread([float(r.checkpoints) for r in done]),
        "elapsed_race_s": _spread([r.elapsed_race_s for r in done]),
        "action_tick_mismatch": sum(r.action_tick_mismatch for r in results),
        "ticks_checked": sum(r.ticks_checked for r in results),
    }
    if map_info.demo is not None:
        out["offline_mean_m"] = _spread([r.offline_mean_m for r in done if r.offline_mean_m is not None])
        out["time_to_leave_line_s"] = _spread([r.time_to_leave_line_s for r in done
                                               if r.time_to_leave_line_s is not None])
        out["never_left_line"] = sum(r.time_to_leave_line_s is None for r in done)
        out["demo_fraction_reached"] = _spread([r.demo_fraction_reached for r in done
                                                if r.demo_fraction_reached is not None])
    return out


def prune_videos(root: Path, keep: int | None) -> None:
    """Delete videos of all but the most recent ``keep`` checkpoints."""
    if keep is None:
        return
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    for d in dirs[: max(0, len(dirs) - keep)]:
        for videos in d.rglob("videos"):
            shutil.rmtree(videos, ignore_errors=True)


def _val_runs(corpus: str) -> set[str]:
    """Held-out run names, from the label index manifest if it has been built."""
    try:
        from ..config import DataConfig
        from ..data.index import load_manifest

        manifest = load_manifest(DataConfig(corpus=corpus))
        return {r["name"] for r in manifest["runs"] if r["val"]}
    except (OSError, ValueError, KeyError):
        return set()


def evaluate(policy: Policy, cfg: EvalConfig, *, checkpoint: str | None = None, log=print) -> dict:
    """``cfg.rollouts`` rollouts of ``policy`` on each of ``cfg.maps``.

    Returns ``{"checkpoint", "temperature", "maps": {label: summary}}``.
    """
    ckpt = checkpoint or policy.id
    layout = detect()
    install.install(layout)
    val_runs = _val_runs(cfg.corpus)
    root = Path(cfg.out_dir) / ckpt

    n = cfg.rollouts
    temperature = cfg.temperature
    if temperature <= 0 and n > 1:
        log("temperature 0 is greedy and deterministic; running a single rollout per map")
        n = 1

    map_jobs: list[MapJob] = []
    for spec in cfg.maps:
        info = resolve(spec, cfg.corpus, val_runs)
        # Staged (intro stripped) before any game launches: games index maps at startup.
        staged = staging.stage_challenge(info.path, layout, strip_intro=True)
        out = root / info.label
        mj = MapJob(info, staged, out, DemoLine(info.demo) if info.demo is not None else None)
        for d in (mj.video_dir, mj.steps_dir):
            d.mkdir(parents=True, exist_ok=True)
        (out / "rollouts.jsonl").unlink(missing_ok=True)
        (out / "eval_config.json").write_text(json.dumps(
            {**asdict(cfg), "checkpoint": ckpt, "map_spec": spec, "map_file": str(info.path),
             "split": info.split}, indent=1))
        map_jobs.append(mj)

    # Interleaved by rollout index, so lanes spread over maps and an
    # interrupted eval still has something on every map.
    jobs: queue.Queue = queue.Queue()
    for k in range(n):
        for mj in map_jobs:
            jobs.put((mj, RolloutResult(checkpoint=ckpt, rollout=k, seed=cfg.seed * 1000 + k,
                                        temperature=temperature, map=mj.info.label, end_reason="pending")))
    results: list[RolloutResult] = []
    lock = threading.Lock()
    started = time.monotonic()
    deadline = started + cfg.max_wall_s if cfg.max_wall_s else None

    restore_ghosts(layout)
    hidden = sum(len(hide_ghosts(layout, mj.info.uid)) for mj in map_jobs)
    if hidden:
        log(f"set aside {hidden} personal-best ghost(s) during eval")
    try:
        lanes = [
            threading.Thread(target=_lane, args=(lane, jobs, results, lock, policy, cfg, log, deadline),
                             daemon=True)
            for lane in range(min(cfg.lanes, jobs.qsize()))
        ]
        for k, t in enumerate(lanes):
            t.start()
            if k + 1 < len(lanes):
                time.sleep(5.0)  # launches staggered, as the collector does
        for t in lanes:
            t.join()
    finally:
        restore_ghosts(layout)

    wall = time.monotonic() - started
    if deadline is not None and jobs.qsize():
        log(f"eval time limit ({cfg.max_wall_s:.0f}s) reached: {jobs.qsize()} rollout(s) not started")
    summary = {"checkpoint": ckpt, "temperature": temperature, "wall_s": round(wall, 1),
               "max_wall_s": cfg.max_wall_s, "maps": {}}
    # An eval of extra maps for an already-evaluated checkpoint adds to its
    # summary rather than replacing it.
    try:
        previous = json.loads((root / "summary.json").read_text())
        summary["maps"].update({k: v for k, v in previous.get("maps", {}).items()
                                if k not in {mj.info.label for mj in map_jobs}})
        summary["wall_s"] = round(previous.get("wall_s", 0) + wall, 1)
    except (OSError, ValueError):
        pass
    for mj in map_jobs:
        mine = sorted((r for r in results if r.map == mj.info.label), key=lambda r: r.rollout)
        s = summarize(mine, mj.info, planned=n)
        (mj.out / "summary.json").write_text(json.dumps(s, indent=1))
        summary["maps"][mj.info.label] = s
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    prune_videos(Path(cfg.out_dir), cfg.keep_last_n_videos)
    return summary
