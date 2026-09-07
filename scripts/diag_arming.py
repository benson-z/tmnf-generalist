"""Dev helper: why does the second map on an instance not drive?

Runs two replays through one session step by step, reporting whether the car
moved, then dumps the in-game console so the command order is visible.

    uv run python scripts/diag_arming.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from tmnf_collect import install, replays, staging  # noqa: E402
from tmnf_collect.paths import detect  # noqa: E402
from tmnf_collect.protocol import Event, Sample  # noqa: E402
from tmnf_collect.session import Session  # noqa: E402

layout = detect()
install.install(layout)
index = replays.ChallengeIndex(layout)

paths = replays.discover_replays(Path("testdata/replays"))[:2]
jobs = []
for path in paths:
    info = replays.read_replay(path)
    challenge = index.find(info.map_uid)
    jobs.append(
        {
            "name": path.stem,
            "replay": staging.stage_replay(path, layout),
            "challenge": staging.stage_challenge(challenge, layout),
            "script": f"diag_{path.stem}.txt",
            "expected": info.race_time,
        }
    )

session = Session(port=8477, layout=layout)
session.start()
try:
    session.prepare(speed=1.0)
    for n, job in enumerate(jobs):
        print(f"\n===== job {n}: {job['name']} =====", flush=True)

        script = session.dump_inputs(job["replay"], job["script"])
        print(f"script {script.name}: {script.stat().st_size} bytes", flush=True)

        if session.game_state != 32:
            session.leave_map()
        session.load_map(job["challenge"])
        print("map loaded, race live", flush=True)

        session._command(f"load {job['script']}", settle=0.5)
        session._command("press delete")
        got_reset = session._wait_for_event(5, timeout=15.0)  # EV_RUN_RESET
        print(f"reset seen: {got_reset}", flush=True)

        session.controller.configure(collect=True)
        # Run each job all the way to the finish, like the real collector
        # does, so the medal screen and everything after it is in play.
        moved, seen, first, last, finished = False, 0, None, None, False
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            msg = session.controller.poll(timeout=1.0)
            if isinstance(msg, Sample):
                seen += 1
                first = first if first is not None else msg.race_time
                last = msg
                if msg.display_speed > 0 or msg.up:
                    moved = True
            elif isinstance(msg, Event):
                if msg.kind == 3:  # EV_FINISH
                    finished = True
                    print(f"FINISH at {msg.race_time} (expected {job['expected']})", flush=True)
                    break
            if seen > 40 and not moved:
                print("car never moved; giving up on this job", flush=True)
                break
        session.controller.configure(collect=False)
        print(
            f"samples={seen} first_t={first} last_t={last.race_time if last else None} "
            f"MOVED={moved} FINISHED={finished}",
            flush=True,
        )
finally:
    log = ""
    try:
        log = session.console_log()
    except Exception as exc:
        print("log capture failed:", exc)
    Path("out").mkdir(exist_ok=True)
    Path("out/diag_arming.log").write_text(log, encoding="utf-8")
    print(f"\nconsole log: {len(log)} chars -> out/diag_arming.log", flush=True)
    session.close()
