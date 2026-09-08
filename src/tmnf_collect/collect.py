"""Turning a pile of replay files into a dataset.

Each replay is re-driven as a normal race and sampled at 20 Hz.  The replay's
own finish time is the correctness check: if the run we recorded finishes at a
different time, the inputs did not reproduce and the run is marked bad rather
than quietly kept.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import install, replays, staging
from .dataset import RunWriter
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
        if challenge is None:
            result.skipped.append(
                {
                    "replay": str(path),
                    "reason": f"no local map with UID {info.map_uid}",
                }
            )
            continue

        name = _safe_name(path)
        result.jobs.append(
            Job(
                replay=info,
                challenge_path=challenge,
                staged_replay=staging.stage_replay(path, layout),
                staged_challenge=staging.stage_challenge(challenge, layout),
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
    image_format: str = "jpeg",
    quality: int = 90,
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
            image_format=image_format,
            quality=quality,
        ) as writer:
            run = session.record_map_run(
                job.staged_challenge,
                job.script_name,
                timeout=timeout,
                expected_ms=job.replay.race_time,
                on_sample=writer.add,
                on_reset=writer.reset,
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
                    "dropped_sample_points": run.dropped,
                    "arming_retries": run.restarts,
                    "driving": run.driving,
                    "restart_observed": run.armed,
                    "clean_start": run.clean_start,
                    "period_ms": session.period_ms,
                    "frame_size": [session.width, session.height],
                    "force_render": session.force_render,
                    "hide_ui": session.hide_ui,
                }
            )
    except (SessionError, OSError, ConnectionError) as exc:
        result.status = "error"
        result.detail = str(exc)

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

def _worker(
    shard: list[Job],
    out_root: Path,
    *,
    instance_id: int,
    port: int,
    layout: Layout,
    width: int,
    height: int,
    period_ms: int,
    speed: float,
    force_render: bool,
    hide_ui: bool,
    unfocused_fps_limit: bool,
    hide_console: bool,
    image_format: str,
    quality: int,
    retries: int,
    capture_log: bool,
    results: list[JobResult],
    progress: Callable[[JobResult], None] | None,
) -> None:
    """Run one game instance over its share of the queue."""

    def new_session() -> Session:
        session = Session(
            port=port,
            instance_id=instance_id,
            layout=layout,
            width=width,
            height=height,
            period_ms=period_ms,
            force_render=force_render,
            hide_ui=hide_ui,
        )
        session.start()
        session.prepare(
            speed=speed,
            unfocused_fps_limit=unfocused_fps_limit,
            hide_console=hide_console,
        )
        return session

    session = new_session()
    try:
        for job in shard:
            attempts = 1
            result = run_job(
                session, job, out_root, image_format=image_format, quality=quality
            )
            if result.status in RETRY_STATUSES and retries:
                attempts += 1
                result = run_job(
                    session,
                    job,
                    out_root,
                    image_format=image_format,
                    quality=quality,
                )
            result.attempts = attempts
            result.instance = instance_id
            results.append(result)
            if progress is not None:
                progress(result)

        if capture_log:
            _save_log(session, out_root, instance_id)
    finally:
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
    speed: float = 1.0,
    force_render: bool = False,
    hide_ui: bool = True,
    unfocused_fps_limit: bool = False,
    hide_console: bool = True,
    limit: int | None = None,
    image_format: str = "jpeg",
    quality: int = 90,
    capture_log: bool = False,
    retries: int = 1,
    instances: int = 1,
    resume: bool = True,
    stagger: float = 8.0,
    progress: Callable[[JobResult], None] | None = None,
) -> dict:
    """Record every replay under ``replay_root``.

    ``instances`` game instances run in parallel, each on its own port and its
    own share of the queue.  Every file they need is staged first, because the
    game only indexes its Tracks folder at startup.
    """
    started = time.monotonic()
    layout = detect()
    install.install(layout)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = replays.discover_replays(replay_root)
    if limit is not None:
        paths = paths[:limit]
    prepared = plan(paths, layout=layout)

    jobs = prepared.jobs
    resumed = 0
    if resume:
        keep = [job for job in jobs if not already_done(out_root, job)]
        resumed = len(jobs) - len(keep)
        jobs = keep

    instances = max(1, min(instances, len(jobs))) if jobs else 0
    results: list[JobResult] = []
    threads: list[threading.Thread] = []
    shard_results: list[list[JobResult]] = []
    failures: list[str] = []

    for index in range(instances):
        shard = jobs[index::instances]  # round robin, so lengths stay even
        collected: list[JobResult] = []
        shard_results.append(collected)

        def target(
            shard: list[Job] = shard,
            collected: list[JobResult] = collected,
            index: int = index,
        ) -> None:
            try:
                _worker(
                    shard,
                    out_root,
                    instance_id=index,
                    port=port + index,
                    layout=layout,
                    width=width,
                    height=height,
                    period_ms=period_ms,
                    speed=speed,
                    force_render=force_render,
                    hide_ui=hide_ui,
                    unfocused_fps_limit=unfocused_fps_limit,
                    hide_console=hide_console,
                    image_format=image_format,
                    quality=quality,
                    retries=retries,
                    capture_log=capture_log,
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
    for collected in shard_results:
        results.extend(collected)

    summary = {
        "replays_found": len(paths),
        "instances": instances,
        "skipped_already_done": resumed,
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
        "instance_failures": failures,
        "skipped": prepared.skipped,
        "seconds": round(time.monotonic() - started, 1),
        "results": [asdict(r) for r in results],
        "out_root": str(out_root),
    }
    (out_root / "index.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
