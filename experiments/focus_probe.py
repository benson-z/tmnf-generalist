"""Does hiding the game windows make replays reproduce more reliably?

Reproduction failures come and go between identical runs -- 36/36, 31/36, 36/36
on the same 36 maps -- and different maps fail each time, which is the signature
of something outside the map. Window focus is the suspect: the collector never
touches a window, but Windows hands the foreground around when windows appear
and close, and anyone using the machine moves it too.

Built for repeated trials rather than one measurement. The games are launched
once and kept, and each pass records the same maps again, alternating between
conditions so that drift over the session cannot be mistaken for an effect.
A pass is about a minute; a whole benchmark is not.

    experiments/focus_probe.py --maps testdata/parallel36 --passes 3

Minimising uses SW_SHOWMINNOACTIVE, which does not give the window the
foreground on the way down -- SW_MINIMIZE would, which is the very thing being
tested.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import queue
import threading
import time
from ctypes import wintypes
from pathlib import Path

from tmnf_collect import collect as collect_mod
from tmnf_collect import install
from tmnf_collect.paths import detect
from tmnf_collect.session import Session

user32 = ctypes.WinDLL("user32", use_last_error=True)
SW_SHOWMINNOACTIVE = 7
SW_SHOWNOACTIVATE = 4
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SM_CXVIRTUALSCREEN = 78


def windows_for(pid: int) -> list[int]:
    """Top-level visible windows belonging to one process."""
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


def place(pids: list[int], condition: str) -> int:
    """Put every game window where this condition wants it.

    "minimized" stops a Direct3D 9 game rendering, which starves the frame
    barrier and makes even input extraction fail, so "offscreen" is the usable
    way to get the windows out of the way: the window keeps rendering because
    it is neither minimised nor occluded, it just is not anywhere the user can
    click it. Neither call activates the window -- doing so would cause the
    focus change being investigated.
    """
    right = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    touched = 0
    for pid in pids:
        for hwnd in windows_for(pid):
            if condition == "minimized":
                user32.ShowWindow(hwnd, SW_SHOWMINNOACTIVE)
            elif condition == "offscreen":
                user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
                user32.SetWindowPos(
                    hwnd, 0, right + 64, 0, 0, 0,
                    SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
                )
            else:
                user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
                user32.SetWindowPos(
                    hwnd, 0, 40 * (pid % 20), 40 * (pid % 10), 0, 0,
                    SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
                )
            touched += 1
    return touched


def foreground_pid() -> int:
    hwnd = user32.GetForegroundWindow()
    owner = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
    return owner.value


def run_pass(sessions: list[Session], jobs: list, out_root: Path) -> dict:
    """Record every job once, spread across the running instances."""
    pending: queue.Queue = queue.Queue()
    for job in sorted(jobs, key=lambda j: -j.replay.race_time):
        pending.put(job)

    results: list = []
    lock = threading.Lock()

    def worker(session: Session) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            result = collect_mod.run_job(session, job, out_root)
            with lock:
                results.append(result)

    threads = [threading.Thread(target=worker, args=(s,)) for s in sessions]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - started

    ok = [r for r in results if r.status == "ok"]
    return {
        "ok": len(ok),
        "total": len(results),
        "seconds": round(wall, 1),
        "failed": [(r.output_name, r.status) for r in results if r.status != "ok"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=Path("testdata/parallel36"))
    parser.add_argument("--only", default=None,
                        help="comma-separated substrings; default is a known-flaky set")
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("out/focus-probe"))
    parser.add_argument("--conditions", default="normal,minimized")
    args = parser.parse_args()

    # Maps seen to fail at least once across earlier 36-map runs. Starting from
    # a set with a real failure rate is what makes a short probe informative.
    flaky = args.only.split(",") if args.only else [
        "tmx-1014712", "tmx-10251861", "tmx-1092896",
        "tmx-11066600", "tmx-1133918", "tmx-1089685", "tmx-1118342",
    ]
    paths = [
        p for p in sorted(args.maps.iterdir())
        if p.suffix.lower() == ".gbx" and any(f in p.name for f in flaky)
    ]
    if not paths:
        parser.error(f"no maps under {args.maps} matched {flaky}")

    layout = detect()
    install.install(layout)
    plan = collect_mod.plan(paths, layout=layout, strip_intros=True)
    print(f"{len(plan.jobs)} maps, {args.instances} instances at {args.speed:g}x, "
          f"{args.passes} passes per condition")

    sessions: list[Session] = []
    try:
        for index in range(args.instances):
            session = Session(
                port=8477 + index, instance_id=index, layout=layout,
                width=320, height=240, speed=args.speed,
            )
            session.start()
            session.prepare()
            sessions.append(session)
            time.sleep(1.0)
        pids = [s.controller.hello.pid for s in sessions]
        print(f"game pids {pids}")

        conditions = args.conditions.split(",")
        history: dict[str, list[dict]] = {c: [] for c in conditions}
        for index in range(args.passes):
            for condition in conditions:          # alternate, so drift cancels
                touched = place(pids, condition)
                time.sleep(1.0)
                outcome = run_pass(sessions, plan.jobs, args.out / condition)
                outcome["windows_set"] = touched
                outcome["foreground_was_a_game"] = foreground_pid() in pids
                history[condition].append(outcome)
                print(f"  pass {index + 1} {condition:10} "
                      f"{outcome['ok']}/{outcome['total']} ok  "
                      f"{outcome['seconds']:5.1f}s  {outcome['failed']}")
    finally:
        place(
            [s.controller.hello.pid for s in sessions
             if s.controller and s.controller.hello],
            "normal",
        )
        for session in sessions:
            try:
                session.close()
            except Exception:
                pass

    print()
    for condition, passes in history.items():
        ok = sum(p["ok"] for p in passes)
        total = sum(p["total"] for p in passes)
        seconds = sum(p["seconds"] for p in passes)
        print(f"{condition:10} {ok}/{total} reproduced  ({ok / total:.0%})  "
              f"{seconds / len(passes):.1f}s a pass")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
