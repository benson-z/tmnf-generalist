"""One game instance, driven from Python.

A :class:`Session` owns a launched game plus the socket its plugin dials back
on, and knows the sequence needed to get from the main menu into a running race
on a chosen map.  It is the unit the replay queue hands work to.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import gamelog, launcher, staging
from .controller import Controller
from .launcher import GameInstance
from .paths import Layout, detect
from .protocol import (
    EV_FINISH,
    EV_GAMESTATE,
    EV_RUN_RESET,
    EV_RUN_START,
    Event,
    Sample,
)

# TM::GameState values we care about.
STATE_MENUS = 32
STATE_LOCAL_INIT = 128
STATE_LOCAL_RACE = 512
STATE_VALIDATION = 262144  # TM::GameState::Unknown1, where validate_replay lands


class SessionError(RuntimeError):
    pass


def _is_moving(sample: Sample) -> bool:
    """Is the car being driven, rather than parked on the start line?"""
    return (
        sample.display_speed > 0
        or sample.up
        or sample.down
        or abs(sample.gas) > 0
        or abs(sample.steer) > 0
    )


@dataclass
class RunResult:
    """What came out of one recorded run."""

    samples: list[Sample]  # empty when the caller streamed them instead
    finished: bool
    finish_time: int | None
    sample_count: int = 0
    dropped: int = 0
    restarts: int = 0  # arming attempts that produced a stationary car
    clean_start: bool = True  # False if the run did not begin at race time 0
    driving: bool = False  # the inputs actually reached the car


class Session:
    def __init__(
        self,
        *,
        port: int,
        instance_id: int = 0,
        layout: Layout | None = None,
        width: int = 320,
        height: int = 240,
        period_ms: int = 50,
        force_render: bool = False,
    ) -> None:
        self.layout = layout or detect()
        self.port = port
        self.instance_id = instance_id
        self.width = width
        self.height = height
        self.period_ms = period_ms
        self.force_render = force_render

        self.controller: Controller | None = None
        self.instance: GameInstance | None = None
        self.game_state = 0  # unknown until the game reports one

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, timeout: float = 180.0) -> None:
        """Launch the game and wait for its plugin to connect."""
        self.controller = Controller(self.port)
        self.instance = launcher.launch(
            port=self.port, instance_id=self.instance_id, layout=self.layout
        )
        self.controller.accept(timeout=timeout)

        # The plugin connects from Render(), which starts running while the
        # game is still initialising. Commands issued that early are either
        # dropped or crash the game, so wait until it reports the menu.
        if not self._wait_for_state(lambda s: s == STATE_MENUS, timeout):
            raise SessionError(
                f"game never reached the menu (state={self.game_state})"
            )
        self._drain(2.0)

        self.controller.configure(
            collect=False,
            period_ms=self.period_ms,
            width=self.width,
            height=self.height,
            force_render=self.force_render,
        )

    def close(self) -> None:
        if self.controller is not None:
            self.controller.close()
            self.controller = None
        if self.instance is not None:
            self.instance.terminate()
            self.instance = None

    def __enter__(self) -> Session:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- helpers -----------------------------------------------------------

    @property
    def _ctrl(self) -> Controller:
        if self.controller is None:
            raise SessionError("session not started")
        return self.controller

    def _command(self, text: str, *, settle: float = 0.15) -> None:
        """Send one console command and let the game consume it.

        TMInterface runs the commands it receives in a single frame as a stack,
        so several sent back to back come out in reverse order. Spacing them
        over frames keeps them in the order they were written.
        """
        self._ctrl.command(text)
        if settle:
            time.sleep(settle)

    def _drain(self, seconds: float) -> list[Event]:
        """Read messages for a while, tracking game state. Discards frames."""
        events: list[Event] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            message = self._ctrl.poll(timeout=max(0.05, deadline - time.monotonic()))
            if isinstance(message, Event):
                events.append(message)
                if message.kind == EV_GAMESTATE:
                    self.game_state = message.arg
        return events

    def _wait_for_event(self, kind: int, timeout: float) -> bool:
        """Poll until the plugin reports one specific event."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self._ctrl.poll(timeout=max(0.05, deadline - time.monotonic()))
            if isinstance(message, Event):
                if message.kind == EV_GAMESTATE:
                    self.game_state = message.arg
                elif message.kind == kind:
                    return True
        return False

    def _wait_for_state(
        self, wanted: Callable[[int], bool], timeout: float
    ) -> bool:
        """Poll until the game reports a state matching ``wanted``."""
        if wanted(self.game_state):
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self._ctrl.poll(timeout=max(0.05, deadline - time.monotonic()))
            if isinstance(message, Event) and message.kind == EV_GAMESTATE:
                self.game_state = message.arg
                if wanted(message.arg):
                    return True
        return False

    # -- getting into a map ------------------------------------------------

    def load_map(
        self,
        staged_name: str,
        *,
        timeout: float = 90.0,
        intro_timeout: float = 120.0,
        settle: float = 0.5,
    ) -> None:
        """Load a staged challenge as a normal race.

        Works straight from the main menu. It does *not* work while a
        validation result dialog is up, which is why nothing here validates
        replays: inputs are extracted from the replay file instead.
        """
        self.game_state = 0
        self._command(f"map {staged_name}")
        if not self._wait_for_state(lambda s: s == STATE_LOCAL_RACE, timeout):
            raise SessionError(
                f"map {staged_name} did not start a race "
                f"(state={self.game_state})"
            )
        # LocalRace is reported as soon as the map starts loading, but the
        # intro flythrough and the countdown still have to play out and any
        # input sent into that window is swallowed. The first run step at a
        # non-negative race time is the game telling us the race is really live.
        if not self._wait_for_event(EV_RUN_START, timeout=intro_timeout):
            raise SessionError(
                f"race never started on {staged_name} "
                f"(state={self.game_state})"
            )
        self._drain(settle)

    def leave_map(self, *, timeout: float = 60.0) -> None:
        """Return to the menu after a run.

        A finished race parks on the medal screen, and while that is up the
        game will not load another map, so this has to be done between jobs
        rather than relying on the next `map` to take.
        """
        self._command("exit_map")
        if self._wait_for_state(lambda s: s == STATE_MENUS, timeout / 2):
            return
        # The medal screen can swallow the first attempt; dismiss it and retry.
        self._command("press delete")
        self._command("exit_map")
        if not self._wait_for_state(lambda s: s == STATE_MENUS, timeout / 2):
            raise SessionError(
                f"could not get back to the menu (state={self.game_state})"
            )

    def prepare(self, *, speed: float = 1.0) -> None:
        """Settings a collecting instance always wants."""
        for command in (
            # Otherwise the game throttles itself whenever its window is not
            # focused, which a headless collector's windows never are.
            "set unfocused_fps_limit false",
            # TMInterface watches loaded scripts and rewinds the run when it
            # thinks one changed, which restarts a recording mid-flight. The
            # nofinish half can also suppress the finish we detect runs by.
            "set autorewind false",
            "set autorewind_nofinish false",
            # This is what actually injects the loaded inputs.
            "set execute_commands true",
            "set skip_map_load_screens true",
            "set draw_game true",
            "set countdown_speed 5",
            f"set speed {speed}",
        ):
            self._command(command)

    def dump_inputs(
        self, staged_replay: str, script_name: str, *, timeout: float = 60.0
    ) -> Path:
        """Extract a replay's inputs into a loadable script.

        TMInterface reads the ghost's inputs straight out of the file, so this
        needs no validation run and does not disturb whatever is on screen.
        """
        target = self.layout.scripts_dir / script_name
        if target.exists():
            target.unlink()
        self._command(f"dump_inputs {staged_replay} {script_name}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if target.is_file() and target.stat().st_size > 0:
                return target
            self._drain(0.25)
        raise SessionError(
            f"dump_inputs produced no script for {staged_replay}; "
            "the replay may not have finished the race"
        )

    def console_log(self) -> str:
        """The in-game TMInterface console, via its copy_log command."""
        self._command("copy_log", settle=0.8)
        return gamelog.read_clipboard_text()

    # -- recording ---------------------------------------------------------

    def record_map_run(
        self,
        staged_challenge: str,
        script_name: str,
        *,
        max_samples: int = 20000,
        timeout: float = 600.0,
        arm_attempts: int = 3,
        on_sample: Callable[[Sample], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> RunResult:
        """Load a map, make TMInterface replay ``script_name``, record the run.

        Loading a script does not by itself arm playback: the race has to
        restart afterwards for the inputs to be injected. That restart is
        sometimes swallowed, and the symptom is a car that just sits on the
        start line, so this checks that the car is actually driving and asks
        again if it is not, rather than recording a stationary run.
        """
        if self.game_state != STATE_MENUS:
            # A finished race parks on the medal screen, and no map loads while
            # that is up.
            self.leave_map()
        self.load_map(staged_challenge)
        self._command(f"load {script_name}", settle=0.5)

        attempt = 0
        while True:
            attempt += 1
            self._command("press delete")
            armed = self._wait_for_event(EV_RUN_RESET, timeout=15.0)

            self._ctrl.configure(collect=True)
            try:
                run = self._record_attempt(
                    max_samples=max_samples,
                    timeout=timeout,
                    on_sample=on_sample,
                )
            finally:
                self._ctrl.configure(collect=False)

            run = RunResult(
                samples=run.samples,
                finished=run.finished,
                finish_time=run.finish_time,
                sample_count=run.sample_count,
                dropped=run.dropped,
                restarts=attempt - 1,
                clean_start=run.clean_start and armed,
                driving=run.driving,
            )
            if run.driving or attempt >= arm_attempts:
                return run

            # The inputs never got injected; throw the stationary frames away
            # and ask for the restart again.
            if on_reset is not None:
                on_reset()

    # How long the car is given to show any sign of being driven. Every replay
    # we re-drive starts by accelerating, so a car still stationary after this
    # is one whose inputs were never armed.
    ARMING_GRACE_MS = 1500

    def _record_attempt(
        self,
        *,
        max_samples: int,
        timeout: float,
        on_sample: Callable[[Sample], None] | None,
    ) -> RunResult:
        """Record one attempt, bailing out early if the car never moves."""
        controller = self._ctrl
        samples: list[Sample] = []
        count = 0
        last_sample: Sample | None = None
        finished = False
        finish_time: int | None = None
        started = False
        clean_start = True
        driving = False
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            message = controller.poll(
                timeout=max(0.05, deadline - time.monotonic())
            )

            if isinstance(message, Event):
                if message.kind == EV_GAMESTATE:
                    self.game_state = message.arg
                elif message.kind == EV_FINISH and started:
                    finished = True
                    finish_time = message.race_time
                    break
                continue

            if isinstance(message, Sample):
                if not started:
                    started = True
                    clean_start = message.race_time == 0

                if not driving and _is_moving(message):
                    driving = True
                if (
                    not driving
                    and message.race_time >= self.ARMING_GRACE_MS
                ):
                    break  # never armed; caller retries

                count += 1
                last_sample = message
                if on_sample is None:
                    samples.append(message)
                else:
                    on_sample(message)
                if count >= max_samples:
                    break
                continue

            if started and count:
                break  # nothing more is coming

        return RunResult(
            samples=samples,
            finished=finished,
            finish_time=finish_time,
            sample_count=count,
            dropped=last_sample.dropped if last_sample else 0,
            clean_start=clean_start,
            driving=driving,
        )
