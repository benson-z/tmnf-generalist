"""Turning a pile of replay files into a dataset.

Each replay is re-driven as a normal race and sampled at 20 Hz.  The replay's
own finish time is the correctness check: if the run we recorded finishes at a
different time, the inputs did not reproduce and the run is marked bad rather
than quietly kept.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common import replays
from ..common.frames import (
    LOSSLESS,
    VIDEO_CODECS,
    FrameWriteError,
    keyframe_interval,
)
from ..common.paths import Layout, detect
from ..common.replays import ChallengeIndex, ReplayError, ReplayInfo
from . import install, staging
from .dataset import RunWriter
from .session import NoInputsError, Session, SessionError


class MapFetchError(Exception):
    """A map fetcher tried and failed, as opposed to finding no such map."""


# Downloads the map with this UID into the folder, or returns None if the
# source does not have it. Collection takes one as a parameter rather than
# importing a source, so it does not care where maps come from.
MapFetcher = Callable[[str, Path], Path | None]


@dataclass
class Job:
    """One replay to record, with everything the game needs already staged."""

    replay: ReplayInfo
    challenge_path: Path
    staged_replay: str
    staged_challenge: str
    script_name: str
    output_name: str
    requeued: int = 0  # times another instance died holding this job
    # time.monotonic() when its files were staged. A game only sees maps staged
    # before it launched; 0 means staged before any game started.
    staged_at: float = 0.0


@dataclass
class JobResult:
    output_name: str
    replay: str
    map_uid: str
    status: str  # ok | no_inputs | dropped_frames | not_driven |
    #             restarted | unfinished | time_mismatch | error
    expected_time: int | None = None
    finish_time: int | None = None
    samples: int = 0
    dropped: int = 0
    preroll_restarts: int = 0
    attempts: int = 1  # how many times the job had to be driven
    seconds: float = 0.0
    instance: int = 0
    detail: str = ""


@dataclass
class JobProgress:
    """A snapshot for a live collection display."""

    instance: int | None
    output_name: str = ""
    state: str = "starting"  # starting | preparing | recording | idle | done
    race_time: int = 0
    expected_time: int = 0
    attempt: int = 1
    completed: int = 0
    total: int = 0


@dataclass(frozen=True)
class LaneConfig:
    """One game lane hosted by a collector subprocess."""

    instance_id: int
    port: int
    startup_delay: float


@dataclass
class Plan:
    """What we can and cannot record, worked out before any game is launched."""

    jobs: list[Job] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def _safe_name(path: Path) -> str:
    stem = path.name
    for suffix in (".Replay.Gbx", ".replay.gbx", ".Gbx", ".gbx"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)


@dataclass
class StagingProgress:
    """How far `plan` has got through the replays."""

    total: int
    done: int = 0
    fetched: int = 0  # maps downloaded from TMX
    skipped: int = 0
    already_done: int = 0  # recorded ok by an earlier collect; not staged


def plan(
    replay_paths: list[Path],
    *,
    layout: Layout | None = None,
    index: ChallengeIndex | None = None,
    fetch_map: MapFetcher | None = None,
    strip_intros: bool = False,
    on_step: Callable[[StagingProgress], None] | None = None,
    on_job: Callable[[Job], None] | None = None,
    is_done: Callable[[str], bool] | None = None,
    stop: threading.Event | None = None,
) -> Plan:
    """Resolve maps and stage every file.

    Staging happens here, before any instance starts, because the game builds
    its Tracks index at startup and will not see files added later. With maps
    fetched from TMX that is about two replays a second, so a corpus of
    thousands spends the better part of an hour here; ``on_step`` hears about
    every replay. ``on_job`` gets each job as soon as it is staged, which is
    what lets the process coordinator start recording before staging ends.
    ``is_done`` names the runs to leave out without fetching their maps again.
    """
    layout = layout or detect()
    index = index or ChallengeIndex(layout)
    result = Plan()
    step = StagingProgress(total=len(replay_paths))

    for path in replay_paths:
        if stop is not None and stop.is_set():
            break
        if is_done is not None and is_done(_safe_name(path)):
            step.already_done += 1
        else:
            before = len(result.jobs)
            _plan_one(path, result, step, layout, index, fetch_map, strip_intros)
            if len(result.jobs) > before:
                result.jobs[-1].staged_at = time.monotonic()
                if on_job is not None:
                    on_job(result.jobs[-1])
        step.done += 1
        step.skipped = len(result.skipped)
        if on_step is not None:
            on_step(step)

    return result


def _plan_one(
    path: Path,
    result: Plan,
    step: StagingProgress,
    layout: Layout,
    index: ChallengeIndex,
    fetch_map: MapFetcher | None,
    strip_intros: bool,
) -> None:
    try:
        info = replays.read_replay(path)
    except ReplayError as exc:
        result.skipped.append({"replay": str(path), "reason": str(exc)})
        return

    if not info.has_inputs:
        result.skipped.append(
            {
                "replay": str(path),
                "reason": "replay did not finish, so it carries no inputs",
            }
        )
        return

    challenge = index.find(info.map_uid)
    if challenge is None and fetch_map is not None:
        # A replay names its map only by UID, and one downloaded from TMX
        # rarely arrives with the map beside it.
        try:
            challenge = fetch_map(info.map_uid, staging.challenges_dir(layout))
        except MapFetchError as exc:
            result.skipped.append(
                {
                    "replay": str(path),
                    "reason": f"could not fetch map {info.map_uid}: {exc}",
                }
            )
            return
        if challenge is not None:
            step.fetched += 1
    if challenge is None:
        reason = f"no local map with UID {info.map_uid}"
        if fetch_map is None:
            reason += " (try --fetch-maps)"
        else:
            reason += " and the map source does not have it either"
        result.skipped.append({"replay": str(path), "reason": reason})
        return

    name = _safe_name(path)
    result.jobs.append(
        Job(
            replay=info,
            challenge_path=challenge,
            staged_replay=staging.stage_replay(path, layout),
            staged_challenge=staging.stage_challenge(
                challenge, layout, strip_intro=strip_intros
            ),
            script_name=f"tmnf_collect_{name}.txt",
            output_name=name,
        )
    )


def run_job(
    session: Session,
    job: Job,
    out_root: Path,
    *,
    codec: str = LOSSLESS,
    timeout: float = 600.0,
    on_progress: Callable[[int], None] | None = None,
) -> JobResult:
    """Record one replay on an already-running instance."""
    started = time.monotonic()
    result = JobResult(
        output_name=job.output_name,
        replay=str(job.replay.path),
        map_uid=job.replay.map_uid,
        status="error",
        expected_time=job.replay.race_time,
    )

    try:
        try:
            session.dump_inputs(job.staged_replay, job.script_name)
        except NoInputsError as exc:
            result.status = "no_inputs"
            result.detail = str(exc)
            result.seconds = round(time.monotonic() - started, 2)
            return result

        with RunWriter(
            out_root / job.output_name,
            codec=codec,
            fps=1000 / session.period_ms,
        ) as writer:
            def add_sample(sample) -> None:
                writer.add(sample)
                if on_progress is not None:
                    on_progress(sample.race_time)

            run = session.record_map_run(
                job.staged_challenge,
                job.script_name,
                timeout=timeout,
                expected_ms=job.replay.race_time,
                on_sample=add_sample,
                on_reset=writer.reset,
                on_tick=writer.add_tick,
            )
            result.samples = run.sample_count
            result.dropped = run.dropped
            result.preroll_restarts = run.restarts
            result.finish_time = run.finish_time

            if not run.driving:
                result.status = "not_driven"
                result.detail = (
                    f"the inputs never reached the car after {run.restarts + 1} "
                    f"attempt(s); it stayed on the start line "
                    f"(restart observed: {run.armed})"
                )
            elif not run.clean_start:
                result.status = "restarted"
                result.detail = (
                    "the first frame is not at race time 0, so the recording "
                    "misses the start of the run"
                )
            elif not run.finished:
                result.status = "unfinished"
                result.detail = (
                    "the re-driven inputs did not reproduce the replay: the run "
                    f"passed {job.replay.race_time} ms without finishing"
                    if run.overran
                    else "the run never crossed the finish line"
                )
            elif run.finish_time != job.replay.race_time:
                result.status = "time_mismatch"
                result.detail = (
                    f"re-driven run finished at {run.finish_time} ms but the "
                    f"replay says {job.replay.race_time} ms"
                )
            elif run.dropped:
                # The game could not draw a frame for some sample points, so
                # the sequence has holes in it even though the run itself was
                # correct. Usually transient: too many instances rendering at
                # once.
                result.status = "dropped_frames"
                result.detail = (
                    f"{run.dropped} sample point(s) never got a frame, so the "
                    "20 Hz sequence has gaps"
                )
            else:
                result.status = "ok"

            writer.write_meta(
                {
                    "replay": str(job.replay.path),
                    "map_uid": job.replay.map_uid,
                    "map_file": str(job.challenge_path),
                    "replay_respawns": job.replay.respawns,
                    "replay_race_time_ms": job.replay.race_time,
                    "recorded_finish_time_ms": run.finish_time,
                    "status": result.status,
                    "detail": result.detail,
                    "samples": run.sample_count,
                    "input_ticks": writer.ticks,
                    "tick_period_ms": 10,  # the simulation's own step
                    "camera": session.camera,
                    "speed": session.speed,
                    "frame_barrier": session.speed > 1,
                    "dropped_sample_points": run.dropped,
                    "arming_retries": run.restarts,
                    "driving": run.driving,
                    "restart_observed": run.armed,
                    "clean_start": run.clean_start,
                    "period_ms": session.period_ms,
                    "frame_size": [session.width, session.height],
                    "frame_codec": codec,
                    **(
                        {
                            "keyframe_interval": keyframe_interval(
                                1000 / session.period_ms
                            )
                        }
                        if codec in VIDEO_CODECS
                        else {}
                    ),
                    "hide_ui": session.hide_ui,
                    "reset_camera": True,  # the plugin always does now
                }
            )
    except FrameWriteError as exc:
        # The encoder, not the game: the run is retried like any other error
        # and the lane carries on. Checked before OSError, which a broken pipe
        # to ffmpeg would otherwise be taken for, marking a healthy game lost.
        result.status = "error"
        result.detail = f"frame encoder failed: {exc}"
        _mark_failed(out_root / job.output_name, result.detail)
    except (SessionError, OSError, ConnectionError) as exc:
        result.status = "error"
        result.detail = str(exc)
        if isinstance(exc, (OSError, ConnectionError)):
            session.lost = True

    try:
        # The medal screen blocks the next map load, so always step back out.
        session.leave_map()
    except (SessionError, OSError, ConnectionError) as exc:
        if result.status == "ok":
            result.status = "error"
            result.detail = f"recorded, but could not leave the map: {exc}"

    result.seconds = round(time.monotonic() - started, 2)
    return result


def _mark_failed(run_dir: Path, detail: str) -> None:
    """Correct a meta.json written as ok before the frames failed to close.

    Resume trusts that status, so an ok left in place would skip this replay
    for good with its frames missing.
    """
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    meta["status"] = "error"
    meta["detail"] = detail
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def already_done(out_root: Path, job: Job | str) -> bool:
    """Has this replay already been recorded successfully?"""
    name = job if isinstance(job, str) else job.output_name
    meta_path = out_root / name / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return meta.get("status") == "ok"


# Worth re-driving on the same instance. "unfinished" is deliberately absent:
# re-driving a desync was measured to be bit-identical three times over -- same
# checkpoint times, same crash position -- so a retry only costs another run.
RETRY_STATUSES = (
    "restarted",
    "not_driven",
    "error",
    "time_mismatch",
    "dropped_frames",
)

def _claim(claims: Path, job: Job) -> bool:
    """Take ownership of one map, across processes.

    Several collectors share one dataset so that the queue stays dynamic: maps
    run from ten seconds to over a minute, and a share dealt out in advance
    strands one collector on the long ones while the others idle. Creating the
    claim file is the atomic step -- O_EXCL either makes it or tells us somebody
    already has this map -- so no two instances record the same run whichever
    process they belong to.
    """
    try:
        handle = os.open(
            claims / f"{job.output_name}.claim",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError:
        return False
    except OSError:
        return True  # cannot arbitrate; better to record twice than not at all
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    return True


def _release(claims: Path, job: Job) -> None:
    """Give a map back after the instance holding it died."""
    try:
        (claims / f"{job.output_name}.claim").unlink()
    except OSError:
        pass


def _execute_job(
    session: Session,
    job: Job,
    out_root: Path,
    *,
    instance_id: int,
    codec: str,
    retries: int,
    live_progress: Callable[[JobProgress], None] | None,
) -> JobResult:
    """Drive one job, including the one same-instance retry policy."""
    attempts = 1

    def announce_preparing() -> None:
        if live_progress is not None:
            live_progress(
                JobProgress(
                    instance=instance_id,
                    output_name=job.output_name,
                    state="preparing",
                    expected_time=job.replay.race_time,
                    attempt=attempts,
                )
            )

    def sample_progress(race_time: int) -> None:
        if live_progress is not None:
            live_progress(
                JobProgress(
                    instance=instance_id,
                    output_name=job.output_name,
                    state="recording",
                    race_time=race_time,
                    expected_time=job.replay.race_time,
                    attempt=attempts,
                )
            )

    announce_preparing()
    result = run_job(
        session,
        job,
        out_root,
        codec=codec,
        on_progress=sample_progress if live_progress is not None else None,
    )
    if result.status in RETRY_STATUSES and retries and not _game_gone(session):
        attempts += 1
        announce_preparing()
        result = run_job(
            session,
            job,
            out_root,
            codec=codec,
            on_progress=sample_progress if live_progress is not None else None,
        )
    result.attempts = attempts
    result.instance = instance_id
    return result


def _game_gone(session: Session) -> bool:
    return (
        session.lost
        or session.instance is None
        or not session.instance.is_alive()
    )


# A game that dies on a map is restarted and the lane carries on; this many in
# a row says the trouble is not the maps, and the lane gives up.
MAX_CONSECUTIVE_CRASHES = 3


def _worker(
    pending: queue.Queue,
    out_root: Path,
    *,
    claims: Path | None = None,
    all_jobs: list[Job] | None = None,
    instance_id: int,
    port: int,
    layout: Layout,
    width: int,
    height: int,
    period_ms: int,
    hide_ui: bool,
    hide_console: bool,
    camera: int | None,
    speed: float,
    offscreen: bool,
    codec: str,
    retries: int,
    capture_log: bool,
    all_done: threading.Barrier | None,
    failures: list[str],
    deadline: float | None,
    results: list[JobResult],
    progress: Callable[[JobResult], None] | None,
    live_progress: Callable[[JobProgress], None] | None,
) -> None:
    """Run one game instance, taking jobs off the queue until it is empty.

    Work is claimed rather than dealt out. Maps vary from ten seconds to over a
    minute, so a fixed share hands one instance a run of long ones and leaves
    everybody waiting on it; pulling the next available map instead keeps every
    instance busy until there is genuinely nothing left.
    """

    def new_session() -> Session:
        session = Session(
            port=port,
            instance_id=instance_id,
            layout=layout,
            width=width,
            height=height,
            period_ms=period_ms,
            hide_ui=hide_ui,
            camera=camera,
            speed=speed,
            offscreen=offscreen,
        )
        session.start()
        session.prepare(hide_console=hide_console)
        return session

    session = new_session()
    try:
        while True:
            # A time budget stops us *claiming* work, never mid-run: the map
            # in progress always finishes, so nothing half-written is left.
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                job = pending.get_nowait()
            except queue.Empty:
                # This process has offered every map it knows about, but the
                # claim files are what actually say who has what: another
                # collector may have handed one back after an instance died.
                # Parking here would strand that map, since the other process
                # already passed over it. Terminates: the rescan only finds
                # maps with no claim, and claiming one removes it.
                if claims is None:
                    break
                unclaimed = [
                    other
                    for other in (all_jobs or [])
                    if not (claims / f"{other.output_name}.claim").exists()
                ]
                if not unclaimed:
                    break
                for other in unclaimed:
                    pending.put(other)
                continue
            if claims is not None and not _claim(claims, job):
                if live_progress is not None:
                    live_progress(
                        JobProgress(
                            instance=None,
                            output_name=job.output_name,
                            state="done",
                        )
                    )
                continue  # another collector is recording this one
            result = _execute_job(
                session,
                job,
                out_root,
                instance_id=instance_id,
                codec=codec,
                retries=retries,
                live_progress=live_progress,
            )

            # A dead game fails every job it is handed, instantly. With a
            # shared queue that is not just this instance's problem: left
            # alone it would claim the rest of the queue and fail all of it.
            if result.status == "error" and (
                session.lost or not session.instance.is_alive()
            ):
                if job.requeued < 1:
                    job.requeued += 1
                    if claims is not None:
                        _release(claims, job)
                    pending.put(job)  # a healthy instance can still record it
                    if live_progress is not None:
                        live_progress(
                            JobProgress(
                                instance=instance_id,
                                output_name=job.output_name,
                                state="idle",
                            )
                        )
                else:
                    results.append(result)
                    if live_progress is not None:
                        live_progress(
                            JobProgress(
                                instance=instance_id,
                                output_name=job.output_name,
                                state="done",
                                expected_time=job.replay.race_time,
                                attempt=result.attempts,
                            )
                        )
                    if progress is not None:
                        progress(result)
                failures.append(
                    f"instance {instance_id}: lost its plugin connection on "
                    f"{job.output_name} ({result.detail[:60]})"
                )
                break

            results.append(result)
            if live_progress is not None:
                live_progress(
                    JobProgress(
                        instance=instance_id,
                        output_name=job.output_name,
                        state="done",
                        race_time=result.finish_time or 0,
                        expected_time=job.replay.race_time,
                        attempt=result.attempts,
                    )
                )
            if progress is not None:
                progress(result)

        if capture_log:
            _save_log(session, out_root, instance_id)

        # Out of work, but not closing yet -- see the barrier below. An idle
        # instance still renders, and the fps throttle is off so it renders
        # flat out, taking GPU from the instances still recording. Stop drawing
        # instead: nothing is being captured from this one any more.
        try:
            session._command("set draw_game false", settle=0.0)
        except (OSError, ConnectionError, SessionError):
            pass
    except BaseException:
        # This instance is not going to reach the barrier, so release everyone
        # waiting on it rather than holding them for the timeout.
        if all_done is not None:
            all_done.abort()
        raise
    finally:
        # Closing a game closes its window, and Windows hands the foreground to
        # the next top-level window -- another instance, which is very likely
        # still recording, and a recording that changes focus drops frames.
        # Shards finish at different times, so every early finisher would
        # disturb the ones still going. Waiting here until the whole queue is
        # done means all that focus churn happens with nothing left to spoil.
        if all_done is not None:
            try:
                all_done.wait(timeout=3600)
            except (threading.BrokenBarrierError, RuntimeError):
                pass  # another instance died; close anyway
        session.close()


def _process_lane(
    lane: LaneConfig,
    commands,
    events,
    *,
    out_root: Path,
    layout: Layout,
    width: int,
    height: int,
    period_ms: int,
    hide_ui: bool,
    hide_console: bool,
    camera: int | None,
    speed: float,
    offscreen: bool,
    codec: str,
    retries: int,
    capture_log: bool,
) -> None:
    """Run one game lane inside a spawned collector process."""
    session: Session | None = None
    current_job: Job | None = None
    result: JobResult | None = None
    drawing = True
    crashes = 0

    def emit(kind: str, payload=None) -> None:
        events.put((kind, lane.instance_id, payload))

    def launch() -> Session:
        # The game indexes Tracks as it starts, so the coordinator needs to
        # know when that was to tell which staged maps this game can load.
        # time.monotonic() is one clock across processes.
        emit("launching", time.monotonic())
        started = Session(
            port=lane.port,
            instance_id=lane.instance_id,
            layout=layout,
            width=width,
            height=height,
            period_ms=period_ms,
            hide_ui=hide_ui,
            camera=camera,
            speed=speed,
            offscreen=offscreen,
        )
        started.start()
        started.prepare(hide_console=hide_console)
        return started

    try:
        if lane.startup_delay:
            time.sleep(lane.startup_delay)
        session = launch()
        emit("ready")

        while True:
            kind, payload = commands.get()
            if kind == "stop":
                break
            if kind == "restart":
                # Maps staged since this game started are invisible to it
                # until it starts again.
                session.close()
                session = None
                session = launch()
                drawing = True
                emit("ready")
                continue
            if kind == "idle":
                if drawing:
                    session._command("set draw_game false", settle=0.0)
                    drawing = False
                continue
            if kind != "job":
                continue

            if not drawing:
                session._command("set draw_game true", settle=0.0)
                drawing = True
            current_job = payload
            result = _execute_job(
                session,
                current_job,
                out_root,
                instance_id=lane.instance_id,
                codec=codec,
                retries=retries,
                live_progress=lambda update: emit("progress", update),
            )
            if result.status == "error" and _game_gone(session):
                # Some maps crash the game outright, stripped or not, and
                # would crash any other game handed them too. So the map is
                # not passed on: it is recorded as the reason, and this lane
                # restarts its game rather than dropping out for the rest of
                # the collection.
                crashes += 1
                result.status = "game_crashed"
                result.detail = f"the game died on this map: {result.detail}"
                if crashes >= MAX_CONSECUTIVE_CRASHES:
                    emit("fatal", (current_job, result, result.detail))
                    return
                emit("result", (current_job, result))
                current_job = None
                result = None
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                session = launch()
                drawing = True
                emit("ready")
                continue
            crashes = 0
            emit("result", (current_job, result))
            current_job = None
            result = None
            emit("ready")
    except BaseException as exc:
        emit(
            "fatal",
            (
                current_job,
                result,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            ),
        )
    finally:
        if session is not None:
            if capture_log:
                _save_log(session, out_root, lane.instance_id)
            try:
                session.close()
            except Exception:
                pass
        emit("stopped")


def _collector_subprocess(
    lanes: list[LaneConfig],
    command_queues: list,
    events,
    worker_options: dict,
) -> None:
    """Host one or more game lanes in a child Python process."""
    threads = [
        threading.Thread(
            target=_process_lane,
            args=(lane, commands, events),
            kwargs=worker_options,
            name=f"tmnf-process-lane-{lane.instance_id}",
        )
        for lane, commands in zip(lanes, command_queues)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _run_process_pool(
    jobs: list[Job],
    out_root: Path,
    *,
    process_count: int,
    instance_base: int,
    port: int,
    layout: Layout,
    width: int,
    height: int,
    period_ms: int,
    hide_ui: bool,
    hide_console: bool,
    camera: int | None,
    speed: float,
    offscreen: bool,
    codec: str,
    retries: int,
    capture_log: bool,
    stagger: float,
    deadline: float | None,
    progress: Callable[[JobResult], None] | None,
    live_progress: Callable[[JobProgress], None] | None,
    incoming: queue.Queue | None = None,
    staging_done: threading.Event | None = None,
) -> tuple[list[JobResult], list[str], int]:
    """Coordinate subprocess game lanes from one authoritative scheduler.

    With ``incoming``, staging is still going on: jobs arrive on that queue
    while games record, and ``jobs`` grows as they do. A game only sees maps
    staged before it launched, so a lane is only given those, and one that has
    run out while newer maps wait is restarted to pick them up. Restarts wait
    for a batch of new maps and are spaced like the initial launches, so the
    games are not relaunched for every map, nor all at once.
    """
    pending: list[Job] = list(jobs)

    def pull() -> None:
        if incoming is None:
            return
        while True:
            try:
                job = incoming.get_nowait()
            except queue.Empty:
                return
            jobs.append(job)
            pending.append(job)

    def staging_open() -> bool:
        return staging_done is not None and not staging_done.is_set()

    lane_count = max(1, process_count)
    if incoming is None:
        lane_count = min(len(jobs), lane_count)
    else:
        # Launch once every lane has a map to start on, not before: a game
        # started now would see nothing and need a restart straight away.
        while len(jobs) < lane_count and staging_open():
            pull()
            time.sleep(0.2)
        pull()
        lane_count = min(len(jobs), lane_count)
        if not lane_count:
            return [], [], 0

    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    lanes = [
        LaneConfig(
            instance_id=instance_base + index,
            port=port + index,
            startup_delay=stagger * index,
        )
        for index in range(lane_count)
    ]
    # process_count is both the number of collector subprocesses and, in this
    # mode, the number of game lanes. Keeping them 1:1 isolates socket readers,
    # encoders and game failures from one another.
    commands = {lane.instance_id: context.Queue() for lane in lanes}
    worker_options = {
        "out_root": out_root,
        "layout": layout,
        "width": width,
        "height": height,
        "period_ms": period_ms,
        "hide_ui": hide_ui,
        "hide_console": hide_console,
        "camera": camera,
        "speed": speed,
        "offscreen": offscreen,
        "codec": codec,
        "retries": retries,
        "capture_log": capture_log,
    }

    processes: list[tuple[multiprocessing.Process, list[int]]] = []
    try:
        for lane in lanes:
            process = context.Process(
                target=_collector_subprocess,
                args=(
                    [lane],
                    [commands[lane.instance_id]],
                    events,
                    worker_options,
                ),
                name=f"tmnf-collector-{lane.instance_id}",
            )
            process.start()
            processes.append((process, [lane.instance_id]))
    except BaseException:
        for process, _hosted in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        for command_queue in commands.values():
            command_queue.close()
        events.close()
        raise

    # When each lane's game last launched; see Job.staged_at.
    launched_at: dict[int, float] = {}
    last_launch = time.monotonic() + stagger * (len(lanes) - 1)
    restart_batch = max(8, len(lanes))
    active: dict[int, Job] = {}
    ready: set[int] = set()
    seen: set[int] = set()
    dead: set[int] = set()
    stopped: set[int] = set()
    idle_sent: set[int] = set()
    results: list[JobResult] = []
    failures: list[str] = []
    lane_ids = {lane.instance_id for lane in lanes}

    def finish_result(job: Job, result: JobResult) -> None:
        results.append(result)
        if live_progress is not None:
            live_progress(
                JobProgress(
                    instance=result.instance,
                    output_name=job.output_name,
                    state="done",
                    race_time=result.finish_time or 0,
                    expected_time=job.replay.race_time,
                    attempt=result.attempts,
                )
            )
        if progress is not None:
            progress(result)

    def clear_lane(instance_id: int) -> None:
        if live_progress is not None:
            live_progress(JobProgress(instance=instance_id, state="idle"))

    def loadable(instance_id: int) -> list[Job]:
        launched = launched_at.get(instance_id, 0.0)
        return [job for job in pending if job.staged_at < launched]

    def assign_ready() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            return
        unseen = lane_ids - seen - dead
        for instance_id in sorted(tuple(ready)):
            # Reserve one initial job for every lane that is still launching;
            # otherwise the first game can drain a short queue before the last
            # staggered game has even reached its menu.
            if len(pending) <= len(unseen):
                break
            choices = loadable(instance_id)
            if not choices:
                continue
            # Longest first is the standard greedy bound on the slow tail.
            job = max(choices, key=lambda item: item.replay.race_time)
            pending.remove(job)
            ready.remove(instance_id)
            idle_sent.discard(instance_id)
            active[instance_id] = job
            commands[instance_id].put(("job", job))

    def restart_starved() -> None:
        nonlocal last_launch
        for instance_id in sorted(tuple(ready)):
            if not pending or loadable(instance_id):
                continue
            # Everything pending was staged after this game launched.
            if staging_open() and len(pending) < restart_batch:
                continue
            if time.monotonic() - last_launch < stagger:
                return
            ready.discard(instance_id)
            idle_sent.discard(instance_id)
            last_launch = time.monotonic()
            commands[instance_id].put(("restart", None))

    try:
        while True:
            pull()
            assign_ready()
            live_lanes = lane_ids - dead
            budget_done = deadline is not None and time.monotonic() >= deadline
            if not budget_done:
                restart_starved()
            more_coming = staging_open() and not budget_done
            if not active and (
                (not pending and not more_coming) or budget_done or not live_lanes
            ):
                # Wait until every still-live staggered launcher has checked in,
                # then shut all games down together so closing one window cannot
                # steal focus from a run that is still recording.
                if live_lanes <= seen:
                    break

            for instance_id in ready:
                if instance_id not in idle_sent and (
                    not loadable(instance_id) or budget_done
                ):
                    commands[instance_id].put(("idle", None))
                    idle_sent.add(instance_id)

            try:
                kind, instance_id, payload = events.get(timeout=0.5)
            except queue.Empty:
                # A hard child crash cannot put a fatal event. Detect it here
                # and requeue whichever map its lane owned.
                for process, hosted in processes:
                    if process.exitcode is None:
                        continue
                    for hosted_id in hosted:
                        if hosted_id in dead or hosted_id in stopped:
                            continue
                        dead.add(hosted_id)
                        seen.add(hosted_id)
                        clear_lane(hosted_id)
                        job = active.pop(hosted_id, None)
                        detail = (
                            f"collector process exited with code "
                            f"{process.exitcode}"
                        )
                        if job is not None:
                            if job.requeued < 1 and lane_ids - dead:
                                job.requeued += 1
                                pending.append(job)
                            else:
                                finish_result(
                                    job,
                                    JobResult(
                                        output_name=job.output_name,
                                        replay=str(job.replay.path),
                                        map_uid=job.replay.map_uid,
                                        status="error",
                                        expected_time=job.replay.race_time,
                                        instance=hosted_id,
                                        detail=detail,
                                    ),
                                )
                        failures.append(
                            f"instance {hosted_id}: {detail}"
                        )
                continue

            if kind == "progress":
                if live_progress is not None:
                    live_progress(payload)
            elif kind == "launching":
                launched_at[instance_id] = payload
                last_launch = max(last_launch, payload)
            elif kind == "ready":
                seen.add(instance_id)
                ready.add(instance_id)
            elif kind == "result":
                job, result = payload
                active.pop(instance_id, None)
                finish_result(job, result)
            elif kind == "fatal":
                reported_job, result, detail = payload
                seen.add(instance_id)
                dead.add(instance_id)
                ready.discard(instance_id)
                clear_lane(instance_id)
                job = active.pop(instance_id, None) or reported_job
                failures.append(f"instance {instance_id}: {detail[:500]}")
                crashed_game = result is not None and result.status == "game_crashed"
                if (
                    job is not None
                    and not crashed_game  # it would crash the next game too
                    and job.requeued < 1
                    and lane_ids - dead
                ):
                    job.requeued += 1
                    pending.append(job)
                elif job is not None:
                    if result is None:
                        result = JobResult(
                            output_name=job.output_name,
                            replay=str(job.replay.path),
                            map_uid=job.replay.map_uid,
                            status="error",
                            expected_time=job.replay.race_time,
                            instance=instance_id,
                            detail=detail.splitlines()[0],
                        )
                    finish_result(job, result)
            elif kind == "stopped":
                stopped.add(instance_id)
    finally:
        for instance_id in lane_ids - dead:
            commands[instance_id].put(("stop", None))
        for process, hosted in processes:
            process.join(timeout=240)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                failures.append(
                    f"collector process for instance(s) {hosted} did not stop"
                )
        for command_queue in commands.values():
            command_queue.close()
        events.close()

    return results, failures, len(pending)


def _save_log(session: Session, out_root: Path, instance_id: int) -> None:
    try:
        (out_root / f"gamelog_{instance_id}.txt").write_text(
            session.console_log(), encoding="utf-8"
        )
    except (OSError, ConnectionError):
        pass


def collect(
    replay_root: Path,
    out_root: Path,
    *,
    port: int = 8477,
    width: int = 320,
    height: int = 240,
    period_ms: int = 50,
    hide_ui: bool = True,
    hide_console: bool = True,
    camera: int | None = None,
    speed: float = 1.0,
    offscreen: bool = True,
    instance_base: int = 0,
    claims: bool = False,
    budget_hours: float | None = None,
    settings: dict | None = None,
    strip_intros: bool = False,
    fetch_map: MapFetcher | None = None,
    limit: int | None = None,
    codec: str = LOSSLESS,
    capture_log: bool = False,
    retries: int = 1,
    instances: int = 1,
    processes: int = 0,
    resume: bool = True,
    stagger: float = 8.0,
    progress: Callable[[JobResult], None] | None = None,
    live_progress: Callable[[JobProgress], None] | None = None,
    staging_progress: Callable[[StagingProgress], None] | None = None,
) -> dict:
    """Record every replay under ``replay_root``.

    Game instances run in parallel, each on its own port. With ``processes``
    greater than zero, a parent coordinator assigns jobs to that many isolated
    collector subprocesses; otherwise ``instances`` worker threads are used.

    The game indexes Tracks at startup, so maps are staged before the games
    that play them launch. Normally that means everything first. Fetching maps
    from TMX is slow enough (about two replays a second) that with
    ``fetch_map`` and ``processes`` staging runs in the background instead,
    and games start, and restart, on what has been staged so far.
    """
    if not 1 <= speed <= 5:
        raise ValueError("speed must be between 1 and 5")
    if processes < 0:
        raise ValueError("processes cannot be negative")
    requested_lanes = processes or instances
    if requested_lanes < 1:
        raise ValueError("instances/processes must provide at least one game lane")
    if speed > 2 and requested_lanes != 1:
        raise ValueError("speed above 2 requires exactly one game instance")
    if processes and claims:
        raise ValueError(
            "the built-in process coordinator does not use --claims; "
            "remove --claims when --processes is set"
        )
    started = time.monotonic()
    layout = detect()
    install.install(layout)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = replays.discover_replays(replay_root)
    if limit is not None:
        paths = paths[:limit]

    step = StagingProgress(total=len(paths))

    def on_step(progress_so_far: StagingProgress) -> None:
        nonlocal step
        step = progress_so_far
        if staging_progress is not None:
            staging_progress(progress_so_far)

    is_done = (lambda name: already_done(out_root, name)) if resume else None
    stream = bool(processes and fetch_map)
    incoming: queue.Queue | None = None
    staging_done = threading.Event()
    stop_staging = threading.Event()
    staging_errors: list[str] = []
    prepared = Plan()
    jobs: list[Job] = []

    if stream:
        incoming = queue.Queue()

        def stage_all() -> None:
            nonlocal prepared
            try:
                prepared = plan(
                    paths,
                    layout=layout,
                    fetch_map=fetch_map,
                    strip_intros=strip_intros,
                    on_step=on_step,
                    on_job=incoming.put,
                    is_done=is_done,
                    stop=stop_staging,
                )
            except Exception as exc:
                staging_errors.append(
                    f"staging stopped: {type(exc).__name__}: {exc}"
                )
            finally:
                staging_done.set()

        stager = threading.Thread(target=stage_all, name="tmnf-staging")
        stager.start()
        instances = max(1, min(requested_lanes, len(paths))) if paths else 0
    else:
        prepared = plan(
            paths,
            layout=layout,
            fetch_map=fetch_map,
            strip_intros=strip_intros,
            on_step=on_step,
            is_done=is_done,
        )
        jobs = prepared.jobs
        staging_done.set()
        instances = max(1, min(requested_lanes, len(jobs))) if jobs else 0

    progress_lock = threading.Lock()
    progress_completed: set[str] = set()

    def report_live(update: JobProgress) -> None:
        if live_progress is None:
            return
        with progress_lock:
            if update.state == "done":
                progress_completed.add(update.output_name)
            update.completed = len(progress_completed)
            update.total = len(jobs)
            live_progress(update)

    report_live(JobProgress(instance=None, total=len(jobs)))
    deadline = started + budget_hours * 3600 if budget_hours else None
    results: list[JobResult] = []
    failures: list[str] = []
    not_started = 0

    # Shared with any other collector process writing into this dataset.
    claims_dir: Path | None = None
    if claims:
        claims_dir = out_root / ".claims"
        claims_dir.mkdir(parents=True, exist_ok=True)

    if processes and (jobs or stream):
        try:
            results, failures, not_started = _run_process_pool(
                jobs,
                out_root,
                process_count=instances,
                instance_base=instance_base,
                port=port,
                layout=layout,
                width=width,
                height=height,
                period_ms=period_ms,
                hide_ui=hide_ui,
                hide_console=hide_console,
                camera=camera,
                speed=speed,
                offscreen=offscreen,
                codec=codec,
                retries=retries,
                capture_log=capture_log,
                stagger=stagger,
                deadline=deadline,
                progress=progress,
                live_progress=report_live,
                incoming=incoming,
                staging_done=staging_done,
            )
        finally:
            # Out of budget or out of lanes: nothing left will record the
            # rest, so do not keep downloading it.
            stop_staging.set()
            if stream:
                stager.join()
        failures.extend(staging_errors)
        jobs = prepared.jobs
    else:
        # Released only when the whole queue is drained, so no window closes
        # while another instance is still recording.
        all_done = threading.Barrier(instances) if instances > 1 else None
        threads: list[threading.Thread] = []
        per_instance: list[list[JobResult]] = []

        # Longest first is the standard greedy bound on the slow tail.
        pending: queue.Queue = queue.Queue()
        for job in sorted(jobs, key=lambda item: -item.replay.race_time):
            pending.put(job)

        for index in range(instances):
            collected: list[JobResult] = []
            per_instance.append(collected)

            def target(
                collected: list[JobResult] = collected,
                index: int = index,
            ) -> None:
                try:
                    _worker(
                        pending,
                        out_root,
                        instance_id=instance_base + index,
                        port=port + index,
                        layout=layout,
                        width=width,
                        height=height,
                        period_ms=period_ms,
                        hide_ui=hide_ui,
                        hide_console=hide_console,
                        camera=camera,
                        speed=speed,
                        offscreen=offscreen,
                        codec=codec,
                        claims=claims_dir,
                        all_jobs=jobs,
                        retries=retries,
                        capture_log=capture_log,
                        all_done=all_done,
                        failures=failures,
                        deadline=deadline,
                        results=collected,
                        progress=progress,
                        live_progress=report_live,
                    )
                except Exception as exc:
                    report_live(
                        JobProgress(
                            instance=instance_base + index,
                            state="idle",
                        )
                    )
                    failures.append(f"instance {index}: {exc}")

            thread = threading.Thread(target=target, name=f"tmnf-instance-{index}")
            thread.start()
            threads.append(thread)
            if index + 1 < instances:
                time.sleep(stagger)

        for thread in threads:
            thread.join()
        for collected in per_instance:
            results.extend(collected)
        not_started = pending.qsize()

    summary = {
        "replays_found": len(paths),
        "instances": instances,
        "worker_processes": instances if processes else 0,
        "skipped_already_done": step.already_done,
        # Left on the queue when the time budget ran out; a later collect
        # over the same folder resumes with exactly these.
        "not_started": not_started,
        "budget_hours": budget_hours,
        "recorded": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        # An "ok" that needed several goes still means the bug fired, so these
        # are reported separately rather than folded into the pass count.
        "ok_first_try": sum(
            1 for r in results if r.status == "ok" and r.attempts == 1
        ),
        "jobs_retried": sum(1 for r in results if r.attempts > 1),
        "arming_retries_total": sum(r.preroll_restarts for r in results),
        "by_status": {
            status: sum(1 for r in results if r.status == status)
            for status in sorted({r.status for r in results})
        },
        # What this dataset was actually produced with, so it does not depend on
        # what the config file happened to say on the day.
        "settings": settings or {},
        "instance_failures": failures,
        "skipped": prepared.skipped,
        "seconds": round(time.monotonic() - started, 1),
        "results": [asdict(r) for r in results],
        "out_root": str(out_root),
    }
    # Collectors sharing a dataset each write their own index, or the second
    # one silently overwrites the first and half the run has no record of what
    # it did. The instance ids are already disjoint, so they name the file.
    name = "index.json" if not claims else f"index-{instance_base}.json"
    (out_root / name).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
