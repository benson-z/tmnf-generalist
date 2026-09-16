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
* The game is never asked to render on demand. Above 1x, the plugin pauses at a
  sample tick and holds that exact simulation state until the next regular
  frame. If the game already queued another physics step, the plugin restores
  the held state while preserving input transitions the script already issued.
  `Render()` captures the frame and resumes simulation.
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
* Python 3.11 — pinned, because `pygbx` needs `python-lzo` and its newest
  wheels are cp311. Plus [uv](https://docs.astral.sh/uv/).

## Usage

```bash
uv sync
uv run tmnf-collect paths           # show what was detected on this machine
uv run tmnf-collect install-plugin  # copy the plugin into TMInterface
uv run tmnf-collect launch          # start one instance, already logged in
uv run tmnf-collect smoke           # launch, drive a fixed script, save frames
uv run tmnf-collect verify <dir>    # check a recorded dataset on disk
uv run tmnf-collect clean <dir>     # move failed runs out of a dataset
uv run tmnf-collect stats <dir>     # graph run lengths and map tags
uv run tmnf-collect video <run>     # replay one run as annotated video
uv run tmnf-collect harvest         # pick maps on TMX and fetch demos
uv run tmnf-collect filter <dir>    # sort replays by input device
uv run tmnf-collect kill            # stop every running instance
```

## Settings

`tmnf-collect.yaml` in the working directory sets defaults for every command, so
a collection run is one line again:

```bash
uv run tmnf-collect collect testdata/corpus --out out/corpus
```

Flags still win over the file, `--config path.yaml` reads another one, and an
unknown key or command section is an error rather than a shrug -- a typo that
silently does nothing is the failure the file exists to prevent. Every run
records what it actually used in `meta.json`, so a dataset does not depend on
what the file happened to say that day.

Record every replay in a folder:

```bash
uv run tmnf-collect collect path/to/replays --out out/dataset

# several games at once, one port each
uv run tmnf-collect collect path/to/replays --out out/dataset --instances 4
```

Re-running the same command skips replays that already recorded successfully,
so an interrupted collection resumes where it stopped. Pass `--no-resume` to
re-record everything.

### Building a corpus from TMX

`harvest` picks maps and fetches a demonstration for each, so the map and its
replay always match and there is no UID search:

```bash
uv run tmnf-collect harvest --limit 200 --out testdata/corpus --dry-run
uv run tmnf-collect harvest --limit 200 --out testdata/corpus     --exclude-tags LOL,PressForward,RPG,Trial,Maze
uv run tmnf-collect collect testdata/corpus --out out/corpus --instances 3
```

`--tags` keeps only maps carrying one of the tags named, `--exclude-tags` drops
maps carrying any of them; both take names or ids, and the harvest summary
reports what it picked by tag, with the gameplay seconds each tag will cost to
record (`--dry-run` gives that budget without downloading anything). Neither filters by default, because the tag says
less than it looks: well-awarded maps predate TMX's multi-tag support and carry
exactly one tag each, and for about half of them that tag is the catch-all
`Race`. What the filters are good for is keeping the genuinely unhelpful
categories out -- a LOL or PressForward map teaches nothing about racing.

Maps are ordered by **award count**, which is the only real quality signal TMX
exposes and filters out broken and troll maps cheaply. `--min-seconds` /
`--max-seconds` bound the author time, because a long map costs proportionally
more to record and is mostly straight-line holding.

`--prefer` chooses which run on a map to learn from:

* `median` (default) — a competent mid-leaderboard run. Records are
  edge-of-control and near-identical to each other, which is poor state coverage
  for a cloning prior.
* `best` — the fastest run.
* `author` — the map maker's validation lap. Often *far* too slow to imitate:
  sampled maps had author times of 199 s against an 89 s record, and 190 s
  against 15 s. Runs more than 1.5x the map's best are dropped whatever the
  setting.

Downloads are verified, not trusted: a fetched map must report the UID that was
asked for, and a fetched replay must name the map it was fetched for.

### Sorting replays by input device

Keyboard and pad are different control regimes rather than different styles: in
TMNF a key press is instant full lock, while a pad emits a continuous value.
Mixed into one training set, the same corner carries contradictory labels.

`filter` sorts a folder in one pass, before any recording:

```bash
uv run tmnf-collect filter testdata/corpus --inputs keyboard --max-seconds 180
uv run tmnf-collect collect testdata/corpus --out out/corpus --instances 3
```

It reads each replay's ghost with `pygbx` and looks at the control entries: a
`Steer` event means an analog device, `SteerLeft`/`SteerRight` means a keyboard,
and no entries at all means the replay stores no inputs. No game is involved and
it runs at roughly 10 ms per replay -- ten replays sort in 0.1 s.

Any replay pygbx cannot read falls back to asking a running game to
`dump_inputs` it, and the script is classified the same way (analog `steer`
lines versus `press left`/`press right`). Both routes were checked against the
same ten replays and agreed on every one; the fallback costs a game launch plus
about a second per replay.

`--max-seconds` drops long runs in the same pass, read from the replay header
before any parsing. Recording cost is linear in race time while the learning
signal is roughly per corner, so one three-minute run buys much less than three
one-minute runs.

Rejects are **moved to a sibling folder** (`testdata/corpus.rejected/pad/…`),
not deleted and not tucked into a subfolder -- collection walks subdirectories,
so a subfolder would still be picked up. A `filter.json` report is left behind.

**Expect to lose a lot.** Of eight replays harvested from the top of TMX's
award rankings, five were pad and only four keyboard. Fast players on
well-known maps often use a pad, so over-harvest by roughly 2x if the corpus is
to be keyboard-only.

### Filling in a missing map

A replay names its map only by UID, and one downloaded from TMX almost never
arrives with the map beside it. `--fetch-maps` looks any missing UID up on
tmnf.exchange and downloads the `.Challenge.Gbx` before launching:

```bash
uv run tmnf-collect collect path/to/replays --out out/dataset --fetch-maps
```

The lookup is by UID, so it is exact rather than a guess at the map's title,
and the downloaded file is rejected unless its own UID matches. It is opt-in
because it reaches a third-party site and writes into the game's Tracks
folder. Without it, a replay whose map is missing is skipped and says so.

Check what landed on disk, independently of what the collector reported:

```bash
uv run tmnf-collect verify out/dataset
```

That re-reads every run and fails it on anything a training set would trip
over: a finish time that does not match the replay, a first sample that is not
at race time 0, a gap other than 50 ms, non-contiguous row indices, rows
pointing at frames that are not there, or a frame drawn before its own tick.

Anything that fails is still on disk, and training reads whatever directories
are there, so take the failures out before using the dataset:

```bash
uv run tmnf-collect clean out/dataset --dry-run   # say what would go
uv run tmnf-collect clean out/dataset
```

Failed runs move to a sibling `out/dataset.rejected/<status>/` rather than being
deleted -- a desync is worth keeping to look at, and `--delete` is there if it
is not. Clearing them also matters before re-collecting: a run directory is
reused and its frames are numbered, so a shorter second attempt would otherwise
leave the tail of the first one behind.

Then look at what the corpus is made of:

```bash
uv run tmnf-collect stats out/dataset
uv run tmnf-collect stats out/dataset --bin 5 --png out/lengths.png
```

Total hours is the number everyone quotes and it hides the shape that matters.
`stats` draws the run-length histogram with its median and quartiles, and
breaks the corpus down by TMX map tag, in runs and in recorded time -- a corpus
that is a third LOL and PressForward maps is time spent learning something other
than racing. The tag bars are drawn from time rather than run count, since time
is what a tag costs and what the training set is made of. Tags are
not recorded in a run, so they are fetched from TMX once and cached in
`tags.json` beside the dataset; `--no-tags` skips the lookup entirely.

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
`samples.jsonl` (one row per frame), `inputs.jsonl` (one row per 10 ms physics
tick) and `meta.json` (what was expected, what
was recorded, and whether they agree). `index.json` at the top lists every
replay, its status, and why anything was skipped. Add `--log` to also save the
in-game TMInterface console to `gamelog.txt`, which is the fastest way to see
why a run misbehaved.

### Driving an instance from Python

```python
from tmnf_collect.session import Session

with Session(port=8477, width=320, height=240, speed=1.0) as session:
    session.prepare()
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

## Inputs are recorded at 100 Hz, frames at 20 Hz

The simulation steps every 10 ms and `inputs.jsonl` keeps every step: keys,
resolved gas/brake/steer, and speed. Frames stay at 20 Hz because they are what
costs disk and GPU.

This is not just finer labels. Measured on two maps, **74-84% of keyboard
transitions land between 20 Hz sample points**, so a frame-rate label stream
either loses a tap or attributes it to the wrong moment -- and TMNF keyboard
driving is largely short counter-steer taps. `verify` checks the stream is
continuous at 10 ms, starts at 0, reaches the finish, and agrees with the 20 Hz
rows wherever the two coincide.

## Map intros are removed from the map, not skipped in the game

The intro flythrough averaged 20 s a map against 41 s of actual driving, and
some intros never end without a keypress at all, which would strand those maps.
Nothing in the game skips it from outside: TMInterface's `press` injects into
the race input system, which does not exist during an intro (it logs "you are
currently not in a race", then "Respawn at invalids"), `PostMessage` and
`AttachThreadInput` never reach a DirectInput game, and `set speed` does not
apply to intros. A real keystroke works but only reaches the foreground window,
which cannot survive parallel instances.

So the clips come out of the staged copy of the map instead --
`mediatracker.py`, on by default, `--keep-intros` to opt out. The map keeps its
UID and its blocks, so it is still the map its replay was driven on. Checked on
465 maps: all stripped, all UIDs preserved, 27.6 MB of MediaTracker removed, one
map being 959 KB of intro out of 981 KB. It also removes the in-race clips that
would otherwise hijack the chase camera mid-run.

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
      exactly: 11358 more rows, 11358 frames, 4/4 passing verify. A TMX replay
      whose map was not on the machine was recorded end to end via
      `--fetch-maps`, reproducing its 43950 ms finish exactly.

## Faster collection without forced rendering

Pass `--speed 5` to run physics up to five times faster. At every 20 Hz sample
point, the plugin pauses simulation and holds that exact state for the next
regular frame. Physics ticks already queued in the current game-loop iteration
are rewound, while input transitions the script issued on those ticks are
carried forward. `Render()` captures the image and resumes simulation.

```bash
uv run tmnf-collect collect testdata/corpus --out out/fast --speed 5 --instances 1
```

Three full-replay comparisons (13.9 s, 29.1 s and 40.5 s) produced 1,673/1,673
frames on their own tick, identical 100 Hz input records, exact car positions,
and finish times identical to the source replays. Their capture intervals ran
4.2-4.8x real time. Mean camera-position difference from a natural 1x reference
was 0.13-0.30 m; the worst isolated difference was 2.82 m on a custom map. Use
1x when matching the human-speed camera matters more than throughput. A 20x
experiment reached 5.4-10.6x but changed one replay's finish by 20 ms, so the
supported setting is capped at 5x. Every sample still carries
`render_race_time` and the run still fails if any sample point is dropped.

Fast mode requires one instance. Under a two-instance 5x load, a replay that
passes alone finished 20 ms late on both attempts: contention caused queued
ticks to cross enough barriers that save-state restoration accumulated physics
drift. One 5x instance is already faster than the measured six-instance 1x
aggregate (4.2-4.8x versus 3.3-3.8x), so `collect` rejects `--speed` above 1
when `--instances` is not 1.

Recording also checks that the car is actually moving, and re-arms the run if
it is not, so a stationary run can never be written out as if it were real.

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

## Why the camera stays on the natural render path

Forcing a render (`Graphics::ForceGameRender()`) looked like the direct way to
speed collection up. It was measured against natural rendering on one
deterministic script, with the car reproducing to 0.00 m so anything left is the
camera:

| comparison | camera position |
|---|---|
| natural vs natural | mean 0.05 m |
| forced vs forced, same speed | mean 0.05 m |
| natural vs forced | mean 1.04 m, max 2.15 m |
| forced 1x vs forced 5x | mean 0.96 m |

Read-only inspection of the installed TMInterface 2.2.1 code found the concrete
cause: `ForceGameRender()` calls the same internal routine exposed as
`ResetCamera()` before it draws. Skipping only that call in a reversible process
experiment reduced the 1x error from 1.00 m to 0.07 m, confirming that this
reset—not Windows painting or screenshot timing—causes most of the shift. The
supported speedup therefore waits for ordinary `Render()` callbacks and does
not patch TMInterface or request an OS repaint. The camera pose is recorded on
every sample so the remaining variation is visible.
