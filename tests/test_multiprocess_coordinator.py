from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tmnf_collect.collect import (
    Job,
    JobProgress,
    JobResult,
    _run_process_pool,
)
from tmnf_collect.paths import Layout
from tmnf_collect.replays import ReplayInfo


def fake_collector(lanes, command_queues, events, _worker_options) -> None:
    """Spawn-safe stand-in for a collector/game process."""
    lane = lanes[0]
    commands = command_queues[0]
    events.put(("ready", lane.instance_id, None))
    while True:
        kind, payload = commands.get()
        if kind == "stop":
            events.put(("stopped", lane.instance_id, None))
            return
        if kind != "job":
            continue
        job = payload
        events.put(
            (
                "progress",
                lane.instance_id,
                JobProgress(
                    instance=lane.instance_id,
                    output_name=job.output_name,
                    state="recording",
                    race_time=job.replay.race_time // 2,
                    expected_time=job.replay.race_time,
                ),
            )
        )
        result = JobResult(
            output_name=job.output_name,
            replay=str(job.replay.path),
            map_uid=job.replay.map_uid,
            status="ok",
            expected_time=job.replay.race_time,
            finish_time=job.replay.race_time,
            instance=lane.instance_id,
        )
        events.put(("result", lane.instance_id, (job, result)))
        events.put(("ready", lane.instance_id, None))


def crashing_collector(lanes, command_queues, events, worker_options) -> None:
    lane = lanes[0]
    if lane.instance_id != 20:
        fake_collector(lanes, command_queues, events, worker_options)
        return
    commands = command_queues[0]
    events.put(("ready", lane.instance_id, None))
    while True:
        kind, _payload = commands.get()
        if kind == "job":
            os._exit(7)
        if kind == "stop":
            return


def fake_layout(root: Path) -> Layout:
    return Layout(
        tmloader_exe=root / "TMLoader.exe",
        tmloader_root=root,
        game="TmForever",
        profile="default",
        tmi_dir=root / "TMInterface",
        plugins_dir=root / "TMInterface" / "Plugins",
        scripts_dir=root / "TMInterface" / "Scripts",
        user_dir=root / "TmForever",
        tracks_dir=root / "TmForever" / "Tracks",
        replays_dir=root / "TmForever" / "Tracks" / "Replays",
        challenges_dir=root / "TmForever" / "Tracks" / "Challenges",
    )


class MultiprocessCoordinatorTests(unittest.TestCase):
    def make_jobs(self, root: Path, count: int) -> list[Job]:
        return [
            Job(
                replay=ReplayInfo(
                    path=root / f"map-{index}.Replay.Gbx",
                    map_uid=f"uid-{index}",
                    race_time=10_000 + index * 1_000,
                ),
                challenge_path=root / f"map-{index}.Challenge.Gbx",
                staged_replay=f"map-{index}.Replay.Gbx",
                staged_challenge=f"map-{index}.Challenge.Gbx",
                script_name=f"map-{index}.txt",
                output_name=f"map-{index}",
            )
            for index in range(count)
        ]

    def test_assigns_each_job_once_across_spawned_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = self.make_jobs(root, 7)
            final: list[JobResult] = []
            live: list[JobProgress] = []
            results, failures, not_started = _run_process_pool(
                jobs,
                root / "out",
                process_count=3,
                instance_base=10,
                port=9000,
                layout=fake_layout(root),
                width=320,
                height=240,
                period_ms=50,
                hide_ui=True,
                hide_console=True,
                camera=1,
                speed=2,
                offscreen=True,
                codec="qoi-zstd",
                retries=1,
                capture_log=False,
                stagger=0,
                deadline=None,
                progress=final.append,
                live_progress=live.append,
                subprocess_target=fake_collector,
            )

        self.assertEqual(failures, [])
        self.assertEqual(not_started, 0)
        self.assertEqual(len(results), 7)
        self.assertEqual(len(final), 7)
        self.assertEqual(
            {result.output_name for result in results},
            {f"map-{index}" for index in range(7)},
        )
        self.assertTrue({result.instance for result in results} <= {10, 11, 12})
        self.assertTrue(any(update.state == "recording" for update in live))
        self.assertEqual(sum(update.state == "done" for update in live), 7)

    def test_requeues_a_job_after_a_worker_process_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, failures, not_started = _run_process_pool(
                self.make_jobs(root, 3),
                root / "out",
                process_count=2,
                instance_base=20,
                port=9100,
                layout=fake_layout(root),
                width=320,
                height=240,
                period_ms=50,
                hide_ui=True,
                hide_console=True,
                camera=1,
                speed=2,
                offscreen=True,
                codec="qoi-zstd",
                retries=1,
                capture_log=False,
                stagger=0,
                deadline=None,
                progress=None,
                live_progress=None,
                subprocess_target=crashing_collector,
            )

        self.assertEqual(not_started, 0)
        self.assertEqual(len(results), 3)
        self.assertEqual({result.status for result in results}, {"ok"})
        self.assertEqual(len(failures), 1)
        self.assertIn("exited with code 7", failures[0])


if __name__ == "__main__":
    unittest.main()
