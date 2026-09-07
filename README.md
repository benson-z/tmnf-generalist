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
`time_mismatch` rather than quietly kept.

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
- [x] **4 - Throughput.** Several instances in parallel on their own ports
      and their own share of the queue, resume, and instance recycling when one
      wedges. Verified: six replays through two instances, all six reproducing
      their exact finish times, then a resume run that correctly skipped all
      six.

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

Loading a script does not by itself make TMInterface replay it; the race has to
restart afterwards, and that restart is occasionally swallowed, leaving the car
parked on the start line. Recording therefore checks that the car is actually
moving and asks again if it is not, and a job that still will not start gets a
freshly launched instance. This happens more often with several instances
running at once, which is why the recovery exists rather than a longer sleep.

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
