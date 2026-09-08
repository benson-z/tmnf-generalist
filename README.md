# tmnf-collect

Scriptable data collection from TrackMania Nations Forever for ML training:
screenshots, keyboard inputs and car telemetry at 20 Hz, clocked off game
physics ticks, with no human in the loop.

## How it works

Replays are **re-driven as a normal race**, not played back in the replay
viewer. The viewer and the validation screen both hijack the camera; a real run
does not. So for each `.Replay.Gbx`:

1. read the replay's map UID and finish time from its Gbx header,
2. find the matching `.Challenge.Gbx` on disk by UID and stage both files,
3. `dump_inputs <replay>` to get the exact inputs as a TMInterface script,
4. `map <challenge>` + `load <script>` so TMInterface drives the car,
5. sample the run from inside the game.

The replay's own finish time is the correctness check: a re-driven run that
finishes at a different millisecond did not reproduce, and is marked
`time_mismatch` rather than quietly kept. A run that never finishes at all is
marked `unfinished`. Not every replay re-drives: a long one can diverge and
leave the car crashed somewhere, so recording stops once the race clock passes
the replay's own time by a margin instead of waiting out the timeout.

Re-driving is deterministic, so this is not worth retrying: the same replay
diverges at the same instant every time. B08-Endurance was re-driven three
times and produced identical checkpoint times, an identical crash position and
an identical sample count on all three. The extracted script is a faithful,
repeatable copy of *itself* but not a bit-exact copy of the original ghost.

It is not about run length. Four replays from 116 s to 177 s -- one of them 73%
longer than the one that fails -- all reproduced to the millisecond. Whatever
makes a replay unreproducible is a property of that replay, not of how long it
is, and the finish-time check is what catches it.

Nothing in the pipeline validates a replay in-game. `validate_replay` works,
but it ends on a modal "this replay is valid" dialog, and while that dialog is
up the `map` command stops loading anything. `dump_inputs` reads a replay file
directly and needs no validation run at all.

Sampling lives in an AngelScript plugin loaded by TMInterface:

* `OnRunStep(simManager)` fires once per 10 ms physics tick, so every 5th tick
  is a 20 Hz sample point. The clock is game time, never wall time.
* `Graphics::CaptureScreenshot()` is only legal inside `Render()`, so a sample
  point stashes the tick's telemetry and the next frame the game draws captures
  it. Each sample records both the tick its telemetry came from and the tick
  the game had reached when the frame was drawn, so any lag between the two is
  visible in the data rather than assumed to be zero.
* `Graphics::ForceGameRender()` can drive that render from the tick instead
  (`--force-render`), removing the lag, but it shifts the chase camera's follow
  distance. See "What forced rendering does to the camera" below. Off by
  default.
* Frames (BGRA), input state and telemetry go out over a `Net::Socket` to the
  Python controller, which writes the dataset. There is no file-write API in
  the plugin sandbox, so the socket is the only way out.
* The in-game speedometer, clock and checkpoint widgets are switched off for
  the duration of a run (`ToggleRaceInterface`), so frames are the bare game
  view. The race interface comes back by itself across map loads, so it is
  re-applied when collection is armed and again when the run starts. Pass
  `--show-ui` to keep it.

The Python side owns the queue: it resolves and stages files, launches the
game, feeds replays to it one after another, and writes the dataset.

## Requirements

* TrackMania Nations Forever and TrackMania ModLoader (TMLoader), with a
  profile that has TMInterface 2.2+ enabled.
* Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/).

## Usage

```bash
uv sync
uv run tmnf-collect paths           # show what was detected on this machine
uv run tmnf-collect install-plugin  # copy the plugin into TMInterface
uv run tmnf-collect launch          # start one instance, already logged in
uv run tmnf-collect smoke           # launch, drive a fixed script, save frames
uv run tmnf-collect camera-check    # measure what forced rendering does
uv run tmnf-collect verify <dir>    # check a recorded dataset on disk
uv run tmnf-collect video <run>     # replay one run as annotated video
uv run tmnf-collect kill            # stop every running instance
```

Record every replay in a folder:

```bash
uv run tmnf-collect collect path/to/replays --out out/dataset

# several games at once, one port each
uv run tmnf-collect collect path/to/replays --out out/dataset --instances 4
```

Re-running the same command skips replays that already recorded successfully,
so an interrupted collection resumes where it stopped. Pass `--no-resume` to
re-record everything.

Check what landed on disk, independently of what the collector reported:

```bash
uv run tmnf-collect verify out/dataset
```

That re-reads every run and fails it on anything a training set would trip
over: a finish time that does not match the replay, a first sample that is not
at race time 0, a gap other than 50 ms, non-contiguous row indices, rows
pointing at frames that are not there, or a frame drawn before its own tick.

To check the frames and the labels are actually in step, watch one run back
with its telemetry drawn on:

```bash
uv run tmnf-collect video out/dataset/A07-Race.Replay --out out/A07.mp4
```

Every frame gets its race time, speed, resolved steer/gas/brake and the four
keys as they were held on that tick. Numbers can agree with each other and
still be shifted against the pictures; this is the check that catches that.
Needs `ffmpeg` on PATH.

Each replay becomes `out/dataset/<replay name>/` holding `frames/NNNNNN.jpg`,
`samples.jsonl` (one row per frame) and `meta.json` (what was expected, what
was recorded, and whether they agree). `index.json` at the top lists every
replay, its status, and why anything was skipped. Add `--log` to also save the
in-game TMInterface console to `gamelog.txt`, which is the fastest way to see
why a run misbehaved.

### Driving an instance from Python

```python
from tmnf_collect.session import Session

with Session(port=8477, width=320, height=240) as session:
    session.prepare(speed=1.0)
    session.dump_inputs("tmnf-collect/A01-Race.Replay.Gbx", "a01.txt")
    result = session.record_map_run(
        "tmnf-collect/A01-Race.Challenge.Gbx", "a01.txt"
    )
    print(result.sample_count, result.finish_time, result.driving)
    session.leave_map()
```

Each `Sample` carries the frame (BGRA bytes), the race time it belongs to, the
key state and analog gas/steer, car position, velocity and yaw/pitch/roll,
in-game speed, checkpoint count, and the camera pose. Pass `on_sample=` to
`record_map_run` to stream frames straight to disk; a 320x240 frame is 300 kB,
so holding a whole run in memory costs hundreds of megabytes.

### Launching without a human

`TMLoader.exe run <game> <profile> <args>` starts the game with the profile's
mods, exits immediately, and passes `<args>` through to `TmForever.exe`.
Account selection is skipped by TMInterface's `autologin` variable. The
launcher appends `/tmnfml_token=... /tmnfml_port=... /tmnfml_id=...`, which the
plugin reads back with `IO::GetCommandLineArgs()` so each instance dials the
right controller socket.

TMLoader passes that whole string to the game as one quoted argument, so the
game's own switches (`/file=`, and anything else) cannot be smuggled in this
way; the plugin scans the joined command line instead of matching argv entries.

## Timing quirks worth knowing

These all cost real debugging time and are handled in code:

* The game drops, or crashes on, commands sent before it reaches the main menu,
  so `Session.start()` waits for the menu before issuing anything.
* TMInterface runs the commands it receives in one frame as a *stack*, so
  commands sent back to back execute in reverse. `Session` spaces them out.
* Maps must be staged into the user Tracks folder *before* the instance that
  will load them is launched; the game indexes that folder at startup.
* `LocalRace` is reported while the map intro is still playing, and input sent
  into that window is swallowed. `Session.load_map` waits for the first run
  step at a non-negative race time instead.
* A finished race parks on the medal screen, and no map will load while it is
  up, so `Session.leave_map` steps back to the menu between runs.
* `autorewind` and `autorewind_nofinish` rewind a run when they think its
  script changed, and can suppress the finish. Both are turned off before
  recording.
* The map UID stored in a ghost record does not decode to the same string as
  the one in the challenge header, so both are read from the Gbx header XML.
* `Net::Socket::Connect` returns false in TMInterface 2.2.1 even when the
  connection is established, so the handshake write decides instead.
* `OnGameStateChanged` only fires on a *transition*, and the plugin often
  connects after the game has already reached the menu, so the plugin reports
  its current state on connect too. Without that, a controller waiting for the
  menu waits for an event that already happened.
* An instance that is not the foreground window has no input bindings, so
  nothing can drive the car. See below.
* `SimulationManager::SetInputState` crashes the game when called from
  `OnRunStep` during a normal race; it is for simulation contexts only.
* Do not take screenshots of, or click on, an instance while it is collecting.
  Activating a window steals focus from the instance that is driving, and that
  is enough to make it drop frames or desync the run outright.
* Some replays store no inputs at all (`dump_inputs` says so), and there is
  nothing to re-drive; those are reported `no_inputs` and skipped.

## Status

- [x] **1 - Launch.** Path detection, headless launch past account selection,
      per-instance arguments, process tracking, teardown.
- [x] **2 - Sampling bridge.** AngelScript plugin plus Python controller;
      20 Hz frames, inputs, telemetry and camera pose clocked off `OnRunStep`.
      Verified: 200 consecutive samples, every race-time gap exactly 50 ms.
- [x] **3 - Replay pipeline.** Replay to input script, map resolution by UID,
      auto load, finish detection, dataset writing, and queueing many replays
      through one instance. Verified: three stock campaign replays re-driven to
      their exact finish times (24540 / 16250 / 18750 ms), every gap 50 ms, no
      drops, frames and rows equal.
- [x] **4 - Throughput.** Several instances in parallel on their own ports,
      user directories and share of the queue, and resume.
      Verified: six replays through three instances, all six
      passing first try and reproducing their exact finish times, 1.8x faster
      than one instance, plus a resume run that correctly skipped all six.
- [x] **5 - Verified on unseen replays.** Twelve fresh replays across all five
      Nations campaigns and every map type (Race, Acrobatic, Speed, Obstacle,
      Endurance), on three instances: 10 recorded and reproduced their exact
      finish times first try, with 0 retries; 1 was correctly rejected as a
      desync and 1 as carrying no inputs. `verify` passes all 10 on disk --
      8840 rows, 8840 frames, 8744 of them drawn on their own tick, worst frame
      lag 10 ms. A further four long replays (116 s to 177 s) all reproduced
      exactly: 11358 more rows, 11358 frames, 4/4 passing verify.

## How fast one instance can go

Sampling is clocked on game time, so raising `speed` does not change *what*
gets recorded, but the frames still have to come from somewhere. With natural
rendering the game decides when it draws, and past a certain speed it outruns
itself. Measured with `smoke --samples 200` on this machine:

| game speed | dropped sample points | gaps | frames drawn on the sample's own tick |
|---|---|---|---|
| 1x | 0 | all 50 ms | 199 / 200 |
| 3x | 0 | all 50 ms | 89 / 200 |
| 5x | 2 | 50 and 100 ms | 55 / 200 |

So 1x is exact, and beyond that frames start being drawn a tick or two late
until sample points are missed outright. Throughput comes from running several
instances rather than from raising the speed of one. Every sample carries
`render_race_time` and a running `dropped` count, so this is checkable in any
dataset rather than having to be trusted.

Recording still checks that the car is actually moving, and reloads the map (or
reloads the map if it is not, so a stationary run can never be written out as
if it were real.

## Why parallel instances used to record stationary cars

Running several instances at once used to leave some runs with the car parked
on the start line for the whole race. The game's own console said why:

```
Failed to execute input: no binding for Accelerate found.
Accelerate at 0.00s
```

TMInterface replays a script by simulating the game's *bound keys*, and an
instance that is not the foreground window has no input bindings, so every
injected input is discarded. The race restarts happily, the script is loaded,
`execute_commands` is true -- and nothing reaches the car. Nothing in this tool
writes bindings, which is exactly why it took a while to find: the state that
goes missing is the game's, not ours.

Two things it was *not*, both ruled out by experiment rather than argument:

* Not shared user data. Giving every instance its own `/userdir` (its own
  `Profiles`, `Config` and `Scores`, with `Tracks` shared through a junction)
  did not reduce the binding errors at all.
* Not the sped-up countdown skipping the first input. Setting
  `countdown_speed` back to 1 made no difference either.

The fix is for each instance to activate its own window
(`Graphics::FocusGameWindow`) just before a run is armed. The bindings then
survive the window losing focus again for the rest of the run, so instances can
still overlap. Measured over the same six replays on three instances:

| | first-try passes | retries | instance relaunches | binding errors | wall clock |
|---|---|---|---|---|---|
| before | 4 / 6 | 2 | 2 | 46 | 149 s |
| after | 6 / 6 | 0 | 0 | 0 | 113 s |

Recovering from a wedged instance cost most of the parallel speed-up, which is
why fixing this made three instances 1.8x faster than one (113 s vs 209 s)
rather than barely faster. With the cause fixed there is nothing left for an
instance relaunch to rescue, so that recovery path has been removed: a job that
fails is reported and the queue moves on.

## What forced rendering does to the camera

`Graphics::ForceGameRender()` does **not** hijack or break the camera: every
frame comes out of the normal chase camera. `tmnf-collect camera-check` drives
one deterministic script three times on one instance (forced at 1x, natural
frames at 1x, forced at 5x) and compares images at identical race times. The
car reproduces exactly (0.00 m difference), so anything left is the camera:

| comparison | camera angle | camera position | mean pixel diff |
|---|---|---|---|
| forced 1x vs natural 1x | max 0.28 rad | mean 0.75 m, max 1.69 m | 22.8 / 255 |
| forced 1x vs forced 5x | max 0.03 rad | mean 0.87 m, max 1.89 m | 17.0 / 255 |

The camera's aim is stable under speed-up (under 2 degrees), so speeding up
collection does not change what the camera looks at. What moves is the eye
position: the chase camera's follow distance is smoothed per rendered frame, so
it settles differently depending on render cadence, by up to about 1.9 m. That
is a normal chase-cam view either way, not a broken one, but frames are not
bit-reproducible across game speeds. Natural rendering is the default for this
reason. The camera pose is recorded on every sample either way, so the
variation is measurable rather than invisible.
