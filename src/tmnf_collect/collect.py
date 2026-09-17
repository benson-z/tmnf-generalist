"""Turning a pile of replay files into a dataset.

Each replay is re-driven as a normal race and sampled at 20 Hz.  The replay's
own finish time is the correctness check: if the run we recorded finishes at a
different time, the inputs did not reproduce and the run is marked bad rather
than quietly kept.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import install, replays, staging, tmx
from .dataset import RunWriter
from .frames import LOSSLESS
from .paths import Layout, detect
from .replays import ChallengeIndex, ReplayError, ReplayInfo
from .session import NoInputsError, Session, SessionError


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


def plan(
    replay_paths: list[Path],
    *,
    layout: Layout | None = None,
    index: ChallengeIndex | None = None,
    fetch_maps: bool = False,
    strip_intros: bool = False,
) -> Plan:
    """Resolve maps and stage every file.

    Staging happens here, before any instance starts, because the game builds
    its Tracks index at startup and will not see files added later.
    """
    layout = layout or detect()
    index = index or ChallengeIndex(layout)
    result = Plan()

    for path in replay_paths:
        try:
            info = replays.read_replay(path)
        except ReplayError as exc:
            result.skipped.append({"replay": str(path), "reason": str(exc)})
            continue

        if not info.has_inputs:
            result.skipped.append(
                {
                    "replay": str(path),
                    "reason": "replay did not finish, so it carries no inputs",
                }
            )
            continue

        challenge = index.find(info.map_uid)
        if challenge is None and fetch_maps:
            # A replay names its map only by UID, and one downloaded from TMX
            # rarely arrives with the map beside it.
            try:
                challenge = tmx.fetch_map(
                    info.map_uid, staging.challenges_dir(layout)
                )
            except tmx.TmxError as exc:
                result.skipped.append(
                    {
                        "replay": str(path),
                        "reason": f"could not fetch map {info.map_uid}: {exc}",
                    }
                )
                continue
        if challenge is None:
            reason = f"no local map with UID {info.map_uid}"
            if not fetch_maps:
                reason += " (try --fetch-maps)"
            else:
                reason += " and TMX does not have it either"
            result.skipped.append({"replay": str(path), "reason": reason})
            continue

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

    return result


def run_job(
    session: Session,
    job: Job,
    out_root: Path,
    *,
    codec: str = LOSSLESS,
    timeout: float = 600.0,
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
        ) as writer:
            run = session.record_map_run(
                job.staged_challenge,
                job.script_name,
                timeout=timeout,
                expected_ms=job.replay.race_time,
                on_sample=writer.add,
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
                    "hide_ui": session.hide_ui,
                }
            )
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


def already_done(out_root: Path, job: Job) -> bool:
    """Has this replay already been recorded successfully?"""
    meta_path = out_root / job.output_name / "meta.json"
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
    codec: str,
    retries: int,
    capture_log: bool,
    all_done: threading.Barrier | None,
    failures: list[str],
    deadline: float | None,
    results: list[JobResult],
    progress: Callable[[JobResult], None] | None,
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
                continue  # another collector is recording this one
            attempts = 1
            result = run_job(
                session, job, out_root, codec=codec
            )
            if result.status in RETRY_STATUSES and retries:
                attempts += 1
                result = run_job(
                    session,
                    job,
                    out_root,
                    codec=codec,
                )
            result.attempts = attempts
            result.instance = instance_id

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
                else:
                    results.append(result)
                    if progress is not None:
                        progress(result)
                failures.append(
                    f"instance {instance_id}: lost its plugin connection on "
                    f"{job.output_name} ({result.detail[:60]})"
                )
                break

            results.append(result)
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
    instance_base: int = 0,
    claims: bool = False,
    budget_hours: float | None = None,
    settings: dict | None = None,
    strip_intros: bool = False,
    fetch_maps: bool = False,
    limit: int | None = None,
    codec: str = LOSSLESS,
    capture_log: bool = False,
    retries: int = 1,
    instances: int = 1,
    resume: bool = True,
    stagger: float = 8.0,
    progress: Callable[[JobResult], None] | None = None,
) -> dict:
    """Record every replay under ``replay_root``.

    ``instances`` game instances run in parallel, each on its own port, taking
    the next map off a shared queue as they become free.  Every file they need is staged first, because the
    game only indexes its Tracks folder at startup.
    """
    if not 1 <= speed <= 5:
        raise ValueError("speed must be between 1 and 5")
    if speed > 2 and instances != 1:
        raise ValueError("speed above 2 requires exactly one game instance")
    started = time.monotonic()
    layout = detect()
    install.install(layout)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = replays.discover_replays(replay_root)
    if limit is not None:
        paths = paths[:limit]
    prepared = plan(
        paths, layout=layout, fetch_maps=fetch_maps, strip_intros=strip_intros
    )

    jobs = prepared.jobs
    resumed = 0
    if resume:
        keep = [job for job in jobs if not already_done(out_root, job)]
        resumed = len(jobs) - len(keep)
        jobs = keep

    instances = max(1, min(instances, len(jobs))) if jobs else 0
    # Released only when the whole queue is drained, so no window closes while
    # another instance is still recording.
    all_done = threading.Barrier(instances) if instances > 1 else None
    deadline = started + budget_hours * 3600 if budget_hours else None
    results: list[JobResult] = []
    threads: list[threading.Thread] = []
    per_instance: list[list[JobResult]] = []
    failures: list[str] = []

    # Shared with any other collector process writing into this dataset.
    claims_dir: Path | None = None
    if claims:
        claims_dir = out_root / ".claims"
        claims_dir.mkdir(parents=True, exist_ok=True)

    # Longest first. Every instance is idle from the moment it runs out of
    # work until the slowest one finishes, so what matters is not starting a
    # 70s map last: claimed late it runs alone while everything else waits at
    # the barrier. Handing out the long maps while there is still short work to
    # fill in behind them is the standard greedy bound on that tail.
    pending: queue.Queue = queue.Queue()
    for job in sorted(jobs, key=lambda j: -j.replay.race_time):
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
                )
            except Exception as exc:  # one instance dying must not sink the rest
                failures.append(f"instance {index}: {exc}")

        thread = threading.Thread(target=target, name=f"tmnf-instance-{index}")
        thread.start()
        threads.append(thread)
        # Two loaders starting at the same moment tread on each other.
        if index + 1 < instances:
            time.sleep(stagger)

    for thread in threads:
        thread.join()
    for collected in per_instance:
        results.extend(collected)

    summary = {
        "replays_found": len(paths),
        "instances": instances,
        "skipped_already_done": resumed,
        # Left on the queue when the time budget ran out; a later collect
        # over the same folder resumes with exactly these.
        "not_started": pending.qsize(),
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
