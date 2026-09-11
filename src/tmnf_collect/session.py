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

from . import gamelog, launcher, staging, userdirs
from .controller import Controller
from .launcher import GameInstance
from .paths import Layout, detect
from .protocol import (
    EV_FINISH,
    EV_GAMESTATE,
    EV_PRERACE,
    EV_RUN_RESET,
    EV_RUN_START,
    Event,
    Sample,
    Tick,
)

# TM::GameState values we care about.
STATE_MENUS = 32
STATE_LOCAL_INIT = 128
STATE_LOCAL_RACE = 512
STATE_VALIDATION = 262144  # TM::GameState::Unknown1, where validate_replay lands


class SessionError(RuntimeError):
    pass


class NoInputsError(SessionError):
    """The replay carries no inputs, so there is nothing to re-drive.

    Plenty of replays are like this -- the game only stores the inputs for
    runs it considers validatable -- and it is not a fault to recover from.
    """


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
    armed: bool = True  # the restart before the run was observed
    overran: bool = False  # ran past the replay's time without finishing


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
        hide_ui: bool = True,
        isolate_user_dir: bool = True,
        camera: int = 1,
        focus_before_run: bool = True,
    ) -> None:
        self.layout = layout or detect()
        self.port = port
        self.instance_id = instance_id
        self.width = width
        self.height = height
        self.period_ms = period_ms
        self.force_render = force_render
        self.hide_ui = hide_ui
        self.isolate_user_dir = isolate_user_dir
        self.camera = camera
        self.focus_before_run = focus_before_run

        self.controller: Controller | None = None
        self.instance: GameInstance | None = None
        self.game_state = 0  # unknown until the game reports one
        # Set once the plugin's socket has gone. The game process usually
        # survives it, so liveness of the process says nothing about whether
        # this instance can still record.
        self.lost = False
        self._console_hidden = False

    # -- lifecycle ---------------------------------------------------------

    def start(self, *, timeout: float = 180.0) -> None:
        """Launch the game and wait for its plugin to connect."""
        profile = None
        if self.isolate_user_dir:
            # Its own copy of the profile the game keys inputs off, so parallel
            # instances cannot clobber each other's bindings.
            profile = userdirs.setup(self.layout, self.instance_id)

        self.controller = Controller(self.port)
        self.instance = launcher.launch(
            port=self.port,
            instance_id=self.instance_id,
            layout=self.layout,
            profile=profile,
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
            hide_ui=self.hide_ui,
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
        if self.camera:
            # `cam` is a console command, so unlike a keystroke it reaches a
            # background instance. Sent per run rather than once, because
            # whether it survives a map change is not worth assuming.
            self._command(f"cam {self.camera}", settle=0.0)
        if settle:
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

    def prepare(
        self,
        *,
        speed: float = 1.0,
        unfocused_fps_limit: bool = False,
        hide_console: bool = True,
    ) -> None:
        """Settings a collecting instance always wants."""
        for command in (
            # Otherwise the game throttles itself whenever its window is not
            # focused, which a headless collector's windows never are. Leaving
            # it on costs throughput but paces rendering normally.
            f"set unfocused_fps_limit {str(unfocused_fps_limit).lower()}",
            # TMInterface watches loaded scripts and rewinds the run when it
            # thinks one changed, which restarts a recording mid-flight. The
            # nofinish half can also suppress the finish we detect runs by.
            "set autorewind false",
            "set autorewind_nofinish false",
            # This is what actually injects the loaded inputs.
            "set execute_commands true",
            # Makes the console say when an input could not be injected,
            # which is the only way a wedged instance announces itself.
            "set log_bot true",
            "set skip_map_load_screens true",
            "set draw_game true",
            # A sped-up countdown can skip the tick carrying the run's first
            # input, which leaves the car parked for the whole run.
            "set countdown_speed 1",
            f"set speed {speed}",
            # Input dumps in plain milliseconds; the driving timeline parses
            # them, and TMInterface loads either format itself.
            "set format_decimal_time false",
        ):
            self._command(command)

        # The console is only a toggle, and it starts visible, so this is done
        # once per instance. Off by default: it flickers while the game renders
        # unlocked, and it is one less thing drawing over the game.
        if hide_console and not self._console_hidden:
            self._command("toggle_console")
            self._console_hidden = True

    def dump_inputs(
        self, staged_replay: str, script_name: str, *, timeout: float = 25.0
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
        # The game says "The replay contained no inputs" and writes nothing.
        raise NoInputsError(
            f"{staged_replay} contains no inputs to replay"
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
        map_timeout: float = 120.0,
        expected_ms: int | None = None,
        on_sample: Callable[[Sample], None] | None = None,
        on_reset: Callable[[], None] | None = None,
        on_tick: Callable[[Tick], None] | None = None,
    ) -> RunResult:
        """Load a map, make TMInterface replay ``script_name``, record the run.

        Loading a script does not arm playback by itself; the race has to
        restart afterwards. The restart is checked by watching for the car to
        actually move, because an instance that has lost its key bindings
        restarts happily and then just sits on the start line.
        """
        if self.game_state != STATE_MENUS:
            # A finished race parks on the medal screen, and no map loads while
            # that is up.
            self.leave_map()

        self.load_map(staged_challenge, intro_timeout=map_timeout)
        self._command(f"load {script_name}", settle=0.5)

        attempt = 0
        while True:
            attempt += 1
            if self.focus_before_run:
                # An instance that is not the foreground window has no input
                # bindings for TMInterface to drive the car through, and the
                # run silently plays out with the car parked on the start line
                # ("no binding for Accelerate found"). Activating the window
                # before arming is what makes parallel instances work; the
                # bindings then survive losing focus again for the rest of the
                # run.
                self._ctrl.focus()
                time.sleep(0.4)
            self._command("press delete")
            armed = self._wait_for_event(EV_RUN_RESET, timeout=15.0)

            self._ctrl.configure(collect=True)
            try:
                run = self._record_attempt(
                    max_samples=max_samples,
                    timeout=timeout,
                    expected_ms=expected_ms,
                    on_sample=on_sample,
                    on_reset=on_reset,
                    on_tick=on_tick,
                )
            finally:
                self._ctrl.configure(collect=False)

            run.restarts += attempt - 1
            run.armed = armed
            if run.driving or attempt >= arm_attempts:
                return run

            # The inputs never reached the car. That is usually a game instance
            # whose key bindings have gone missing, which no amount of
            # restarting fixes, so the caller is expected to give up before
            # long and relaunch.
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
        expected_ms: int | None,
        on_sample: Callable[[Sample], None] | None,
        on_reset: Callable[[], None] | None = None,
        on_tick: Callable[[Tick], None] | None = None,
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
        overran = False
        restarts = 0
        prev_time: int | None = None

        # A run that desyncs never reaches the finish line and the race clock
        # just keeps going. The replay says how long it should take, so give it
        # a margin and then stop, rather than recording minutes of a crashed car.
        race_limit = (
            None
            if expected_ms is None
            else expected_ms + max(5000, expected_ms // 5)
        )

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

            if isinstance(message, Tick):
                # Inputs at 100Hz. The tick at race time 0 arrives just
                # before the sample for the same instant, so this cannot wait
                # for the run to be marked started or it would lose it; a
                # restart is handled by the writer discarding what it holds.
                if on_tick is not None:
                    on_tick(message)
                continue

            if isinstance(message, Sample):
                if prev_time is not None and message.race_time <= prev_time:
                    # The race went back to the start mid-recording. What was
                    # written so far belongs to an abandoned attempt, and
                    # keeping it would splice two runs into one file.
                    restarts += 1
                    count = 0
                    samples.clear()
                    last_sample = None
                    started = False
                    driving = False
                    if on_reset is not None:
                        on_reset()
                prev_time = message.race_time

                if not started:
                    started = True
                    clean_start = message.race_time == 0

                if not driving and _is_moving(message):
                    driving = True
                if not driving and message.race_time >= self.ARMING_GRACE_MS:
                    break  # the inputs never reached the car; caller retries

                if race_limit is not None and message.race_time > race_limit:
                    overran = True
                    break  # this run is not going to finish

                count += 1
                last_sample = message
                if on_sample is None:
                    samples.append(message)
                else:
                    on_sample(message)
                if count >= max_samples:
                    break
                continue

            # Nothing arrived before the poll timed out.
            if started and count:
                break

        return RunResult(
            samples=samples,
            finished=finished,
            finish_time=finish_time,
            sample_count=count,
            dropped=last_sample.dropped if last_sample else 0,
            clean_start=clean_start,
            driving=driving,
            overran=overran,
            restarts=restarts,
        )
