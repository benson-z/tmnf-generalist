# tmnf-collect: how it works and why

Every technique on the collection path, organised by pipeline stage:
`harvest → filter → collect → verify`, with `stats`, `clean` and `video` for
looking at what came out. The README covers how to use it; this is the
reasoning and the measurements behind it.

Source layout under `src/tmnf_collect/`:

- `collect/` drives the game: launching instances, the plugin bridge,
  re-driving replays and writing runs to disk.
- `harvest/` builds the replay corpus (`harvest`, `filter`). Nothing in
  `collect/` imports it; the CLI hands collection a TMX map fetcher when
  `--fetch-maps` is set.
- `tools/` works on a recorded dataset: `verify`, `clean`, `stats`, `video`.
- `bench/` measures settings: `bench-matrix`, `bench-throughput`.
- `common/` is shared: Gbx header reading, the frame codec, map intro
  stripping, install and host-OS discovery.
- `plugin/TMNFCollect.as` is the AngelScript plugin TMInterface loads.

## 0. The core idea: re-drive, don't play back

Replays are **re-driven as a normal race**, not played back in the replay
viewer. The viewer and the validation screen both hijack the camera; a real run
does not. For each `.Replay.Gbx`:

1. read the map UID and finish time from its Gbx header,
2. find the matching `.Challenge.Gbx` by UID and stage both,
3. `dump_inputs <replay>` to get the exact inputs as a TMInterface script,
4. `map <challenge>` + `load <script>` so TMInterface drives the car,
5. sample the run from inside the game.

- **The replay's finish time is the correctness bar**, exact to the
  millisecond. A mismatch is `time_mismatch`; never finishing is `unfinished`.
- **Re-driving is deterministic, so a desync is not retried.** B08-Endurance
  was re-driven three times with identical checkpoint times, crash position
  and sample count. The extracted script is a faithful copy of *itself*, not
  a bit-exact copy of the original ghost.
- **It is not run length.** Four replays from 116 s to 177 s, one 73% longer
  than a failing one, all reproduced exactly. Unreproducibility is a property
  of the replay; the finish-time check is what catches it. Roughly one in ten
  harvested replays fails this way.
- **Nothing validates a replay in-game.** `validate_replay` ends on a modal
  dialog during which `map` loads nothing; `dump_inputs` reads the file
  directly and needs no validation run.
- Some replays store no inputs at all; they are `no_inputs` and skipped.

## 1. Acquisition — `harvest` (`harvest/tmx.py`)

- **Map-first, not replay-first.** Pick maps, then take a replay *from that
  map's leaderboard*, so map and replay always match by construction.
- **Awards as the quality signal.** The only real one TMX exposes, and it
  filters out broken and troll maps cheaply. `/api/tracks` ordered
  awards-descending; `min_awards` applied client-side with an early stop.
- **Author-time bounds** (`--min-seconds` / `--max-seconds`): a long map costs
  proportionally more to record and is mostly straight-line holding.
- **Tag filtering client-side** (`--tags` / `--exclude-tags`) because the API
  silently ignores its tag parameter. Names come from TMX's own
  `enumTrackTagDesc`; unknown ids render as `tag-N`. Well-awarded maps predate
  multi-tagging and carry exactly one tag, about half of them the catch-all
  `Race`, so tags are good for excluding LOL/PressForward-style maps and not
  much else. `--dry-run` reports the gameplay seconds per tag without
  downloading.
- **`--prefer median` of the credible leaderboard** (runs within 1.5× of the
  map's best). Author laps are often far too slow (199 s against an 89 s
  record; 190 s against 15 s); records are edge-of-control and near
  identical, poor state coverage for a cloning prior. `best` and `author`
  remain available.
- **Over-fetch 3×** because some maps have empty leaderboards.
- Downloads are verified, not trusted: `GBX` magic, the fetched map reports
  the UID asked for, the replay names the map it was fetched for. Written to
  `.part` then renamed; named `tmx-<id>` because console commands take paths
  as bare words. User agent and courtesy delay on every request.

## 2. Filtering — `filter` (`harvest/filter.py`)

- **Input device read offline from the ghost** with pygbx, ~10 ms a file: a
  `Steer` control entry means pad, `SteerLeft`/`SteerRight` means keyboard,
  none means no inputs. Keyboard is instant full lock and a pad is
  continuous, so mixing them puts contradictory labels on the same corner.
  Bytes are passed rather than the path, because pygbx never closes the handle
  and Windows will not rename an open file.
- **No game involved.** pygbx is a hard dependency; a replay it cannot read is
  treated as unreadable. (An earlier fallback that asked a running game to
  `dump_inputs` it was removed as not worth a launch.)
- **Duration cap** read from the uncompressed header before any parsing: cost
  is linear in race time, learning signal is roughly per corner.
- **Rejects moved to a sibling folder** `<folder>.rejected/<kind>/`, not a
  subfolder, which `collect`'s recursive discovery would still find. A
  `filter.json` report is left behind.
- **Expect heavy losses.** Of eight top-awarded replays, five were pad. Fast
  players on well-known maps often use a pad, so over-harvest ~2×.

## 3. Staging (`collect/staging.py`, `common/mediatracker.py`, `common/replays.py`)

- **UID → file index** over the Tracks folder and stock campaigns, cached on
  disk, rebuilt on a miss. The map UID in a ghost record does not decode to
  the same string as the challenge header's, so both are read from the Gbx
  header XML.
- **`--fetch-maps`** looks a missing UID up on TMX and rejects the download
  unless its own UID matches. Opt-in because it reaches a third-party site and
  writes into the game's Tracks folder.
- **Maps staged before the game that plays them launches** — the game indexes
  its Tracks folder at startup only (a map fetched after launch fails `map`
  with the game still in the menu; a late replay's `dump_inputs` works).
  Normally that means everything first. With `--fetch-maps` under the process
  coordinator, fetching runs ~0.5 replays/s (2518 replays ≈ 90 min), so
  staging runs in a background thread instead: games launch once each lane
  has a map, a lane is only given maps staged before its game launched, and a
  lane that runs out restarts its game once 8 newer maps wait (or staging has
  finished), spaced like the initial launches. 30 replays with no map on disk:
  games up with 11 staged, 29/30 ok, the other a desync. Resume is checked
  before fetching, so a re-run does not download finished maps again.
- **MediaTracker clips stripped from the staged copy** (`--keep-intros` to
  opt out). Intros averaged 20 s a map against 41 s of driving, and some never
  end without a keypress. Nothing skips them from outside: TMInterface's
  `press` has no race input system to inject into during an intro,
  `PostMessage`/`AttachThreadInput` never reach a DirectInput game, `set
  speed` does not apply, and a real keystroke only reaches the foreground
  window. So: decompress the LZO body, replace chunk `0x03043021`'s three
  inline node refs (intro, in-race, end-race) with nulls through to the next
  ascending chunk id, recompress. UID and blocks live elsewhere, so the map is
  still the one the replay was driven on; freed node indices become gaps,
  which is safe because GBX writes each index explicitly. Also removes the
  in-race clips that hijack the chase camera. Checked on 465 maps: all
  stripped, all UIDs preserved, 27.6 MB removed (one map was 959 KB of intro
  out of 981 KB). A marker file tracks staleness.

## 4. Launching and instance isolation (`collect/launcher.py`, `collect/userdirs.py`)

- **`TMLoader.exe run <game> <profile> <args>`** starts the game with the
  profile's mods and passes `<args>` through. TMInterface's `autologin` skips
  account selection.
- **One TMLoader profile per instance**, regenerated from `default.yaml` on
  every start (so a version pin there reaches every instance), adding only
  `/userdir=<path>`.
- **Per-instance user directory**: `Config` and `Profiles` mirrored from
  `Documents/TmForever` on every start (one place to change any setting),
  `Scores` copied once, `Tracks` a junction so staged files are visible
  everywhere.
- **Magenta car, body only.** `skins/Magenta.zip` (solid DXT1 `Diffuse.dds`
  and `Icon.dds`, no `Details.dds`, so wheels and suspension stay stock) is
  installed with the plugin into `Skins/Vehicles/StadiumCar`, and
  `Skins/Vehicles` is linked into each instance (not all of `Skins`: the game
  makes its own there first). The seeded profile selects it in chunk
  `0x0308C043`, which names the zip and pins its MD5, so the zip must stay
  byte-identical.
- Token, port and instance id go on the command line
  (`/tmnfml_token= /tmnfml_port= /tmnfml_id=`) and the plugin dials *back* to
  the controller. TMLoader wraps every argument into one quoted string, so
  the plugin scans the joined command line from `IO::GetCommandLineArgs()`,
  and the game's own switches cannot be passed this way.
- **Offscreen by default.** Windows stay shown so Direct3D keeps producing
  frames, but sit beyond the virtual desktop where they cannot be covered or
  clicked. `--no-offscreen` for debugging.

## 5. Session control (`collect/session.py`)

- **Wait for the menu before any command** — commands sent while the game
  initialises are dropped or crash it. The plugin reports its current state
  on connect, because `OnGameStateChanged` only fires on transitions and the
  plugin often connects after the menu is up.
- **Commands spaced across frames** — TMInterface runs same-frame commands as
  a stack, last in first out.
- `prepare()`: fps throttle off, `autorewind` and `autorewind_nofinish` off
  (they rewind mid-recording and can suppress the finish), `execute_commands`
  on, `log_bot` on (the only way a wedged instance announces itself), load
  screens skipped, `countdown_speed 1`, configurable game speed, console
  hidden once.
- **Map load waits for the first non-negative tick**, not the race state —
  `LocalRace` is reported while the intro still plays, and input sent then is
  swallowed.
- **Camera issued per run** (`cam 1`) once the race is live.
- `leave_map` clears the medal screen, which otherwise blocks the next load.

## 6. Arming and recording a run (`collect/session.py`, `collect/runner.py`)

- **Focus before arming** (`Graphics::FocusGameWindow`). TMInterface replays a
  script by simulating the game's *bound keys*, and an instance that has never
  been foreground has no bindings: the console says `no binding for
  Accelerate found` and the car sits on the line. Bindings then survive losing
  focus for the rest of the run. Ruled out by experiment first: shared user
  data (separate `/userdir`s did not help) and the fast countdown
  (`countdown_speed 1` did not help). On six replays over three instances:

  | | first-try passes | retries | relaunches | binding errors | wall clock |
  |---|---|---|---|---|---|
  | before | 4 / 6 | 2 | 2 | 46 | 149 s |
  | after | 6 / 6 | 0 | 0 | 0 | 113 s |

  That made three instances 1.8× faster than one (113 s vs 209 s), and the
  instance-relaunch recovery path was removed as having nothing left to rescue.
- `load <script>`, then `press delete` to restart; wait for `EV_RUN_RESET`;
  **verify the car actually moves** within a grace window, else re-arm (up to
  three attempts), so a stationary run is never written out.
- **Restart detection**: race time going backwards means an abandoned attempt;
  everything written is discarded rather than spliced.
- **Overrun guard**: a run past `expected + max(5 s, 20%)` is stopped as a
  desync rather than recorded to the timeout.
- `restarted`, `not_driven`, `error`, `time_mismatch` and `dropped_frames` are
  retried on the same instance; `unfinished` is not (see §0).
- **`game_crashed`**: some TMX maps kill TMNF under Wine as they load,
  stripped or not (tmx-4347297 and tmx-5760368 did, every time). Such a map
  is not retried and not handed to another lane, where it killed a healthy
  game too; the lane restarts its own game and carries on. Three crashes in
  a row and the lane gives up. Before this, two such replays took four of
  eight lanes out of a collection.
- **Encoder failures are the run's, not the game's.** A broken pipe to ffmpeg
  is an `OSError`, which used to mark the game lost; it is `FrameWriteError`
  now, and ffmpeg is restarted if it dies before its first frame.

## 7. Sampling in the plugin (`plugin/TMNFCollect.as`)

- **Clocked on physics ticks** — `OnRunStep` at 10 ms; every fifth tick at
  `race_time ≥ 0` is a 20 Hz sample point. Game time, never wall time.
- **Capture on the game's next natural render.** `CaptureScreenshot` is only
  legal inside `Render()`, so the tick stashes telemetry and the next drawn
  frame captures it. Each sample records both its tick (`race_time`) and the
  tick the game had reached when drawn (`render_race_time`), so lag is visible
  rather than assumed zero. A pending sample the game never draws increments
  `dropped`, and a run with any dropped points is rejected.
- **Frame barrier above 1×**, held the way Linesight does: at a sample tick,
  rewind to the state the game is already in (changes no physics, drops the
  ticks the loop had queued), put that tick's inputs back, then speed 0.
  `Render()` captures and restores the speed. The earlier `Running=false`
  hold let one more tick run on nearly every sample point; rewinding it and
  re-applying inputs shifted transitions by a tick, and 3 of 20 replays
  diverged at 2×.
- **Camera reset at every sample point** (`ResetCamera`, or the barrier's
  rewind). The chase camera is smoothed on wall-clock time, so byte-identical
  car trajectories still differed by 1.2 m mean camera position at 2× under
  Wine (0.45 m with eight instances; 0.05 m on Windows). With the reset,
  16678 of 16680 samples across two runs had a byte-identical camera pose; the
  other two were finish ticks, where the game switches camera. The camera
  then sits at a fixed offset rather than lagging on acceleration, so
  **RL-time inference must reset the camera the same way.**
- **No forced rendering.** `Graphics::ForceGameRender()` looked like the
  direct speed-up, but it calls the same routine as `ResetCamera()` before
  drawing, and on the natural-camera path that moved the camera ~1 m:

  | comparison | camera position |
  |---|---|
  | natural vs natural | mean 0.05 m |
  | forced vs forced, same speed | mean 0.05 m |
  | natural vs forced | mean 1.04 m, max 2.15 m |
  | forced 1× vs forced 5× | mean 0.96 m |

  Skipping only that call in a process experiment cut the 1× error from 1.00 m
  to 0.07 m. The collector waits for ordinary `Render()` callbacks and patches
  nothing in TMInterface.
- **Inputs at 100 Hz**: one `MSG_TICK` per physics step carrying a 4-bit key
  mask, two socket writes — kept lean because a single failed write
  disconnects the plugin for good. 74–84% of keyboard transitions fall between
  20 Hz sample points, so the frame-rate stream alone loses or misplaces most
  taps. Analog values and speed come with the 20 Hz samples.
- Telemetry per sample: position, velocity, yaw/pitch/roll, local speed,
  camera pose and FOV, keys and analog inputs, checkpoints, finished, sliding,
  gearbox.
- **HUD off** via `ToggleRaceInterface`, re-applied when armed and at run
  start since map loads restore it (`--show-ui` to keep it).
- Outbound `Net::Socket` (the sandbox has no file-write API), reconnect every
  second from `Render()`, the one callback that runs in the menus.
  `Connect()`'s return value is ignored because it reports false on success in
  2.2.1; the handshake write decides.
- `SimulationManager::SetInputState` crashed the game when called from
  `OnRunStep` in a normal race; the barrier calls it only immediately after
  `RewindToState`.

## 8. Controller and writer (`collect/controller.py`, `collect/protocol.py`, `collect/dataset.py`, `common/frames.py`)

- Little-endian framed protocol with a version check on the hello.
- **The socket reader never blocks.** Frames and ticks go to *unbounded*
  queues on their own writer threads. A bounded queue stalled the reader under
  load, the plugin's writes backed up and failed, and instances disconnected.
- **4 MB receive buffer**, set on the listener before `listen()` so accepted
  sockets inherit it. The plugin writes a whole 307 KB frame in one call
  against Windows' 64 KB default, so a write only succeeded if the reader
  happened to be draining right then, and a failed write is fatal to the
  plugin. On 36 maps with two collectors of four instances at 2×:

  | buffer | ok | instances lost | gameplay / wall clock |
  |---|---|---|---|
  | 64 KB | 33/36 | 4 | 4.74× |
  | 4 MB | 36/36 | 0 | 7.17× |
- **One `frames.bin` per run**, offset and length in `samples.jsonl` — no
  200,000-file corpus to read back.
- **Frames lossless, QOI + zstd-6.** 70.0 KB a frame at 3669 frames/s on
  eight threads, against 66.9 KB at 1054/s for lossless WebP and 75.9 KB at
  835/s for PNG. Both stages release the GIL, so encoding never starves a
  lane's socket reader. Lossless because JPEG artifacts vary with content, so
  the same corner recorded twice would differ visibly. Costs ~5.8 GB per
  recorded hour (JPEG q80: 0.85 GB).
- **Default is now `hevc`: H.265 CRF 18, preset veryfast**, one ffmpeg
  process per run writing `frames.mkv`, one video frame per row, no offsets
  in the rows. 7.0 KB a frame (~0.5 GB per recorded hour) at 33.6 dB RGB
  PSNR; eight encoders do 404 frames/s against the 320 that eight lanes at 2×
  need (`medium`: 268). Closed GOP every 2 s, scene cuts off, for random
  access at +1.5% size. On Linux, `hevc-vaapi` does the same on the GPU
  at QP 24 (equal size and PSNR): 20 replays in 137.0 s against 152.8 s for
  x265 and 138.8 s lossless, because x265 takes CPU the games need. Gives up lossless comparability across
  re-collections; `qoi-zstd` remains selectable.
- `inputs.jsonl` holds the ticks, `meta.json` expected against recorded and
  the settings used.
- `reset()` discards a partial attempt across both queues, in order; a reused
  run directory is truncated so a shorter retry leaves nothing behind.

## 9. Scheduling (`collect/runner.py`)

- **`--processes N`: one coordinator, N isolated collector subprocesses**, one
  game each. Workers announce when ready and receive one map at a time
  directly over multiprocessing queues; no claim files. Separate processes
  because one interpreter was the wall, not the hardware: eight socket readers
  moving 307 KB a sample past one GIL. (Two processes of four recorded eleven
  replays in 78.7 s against 94.8 s for one of eight.) `--processes 0
  --instances N` is the older threaded mode.
- **Longest map first.** Every lane idles from when it runs out of work until
  the slowest finishes, so a long map handed out last runs alone; longest
  first is the standard greedy bound on that tail. One initial map is
  reserved per lane still launching, so the first game up cannot drain a
  short queue.
- **Idle lanes stop drawing** (`set draw_game false`) so they take no GPU from
  lanes still recording.
- **All games close together at the end**: a closing window hands the
  foreground to a still-recording instance, which then drops frames.
- **Dead-instance handling**: socket lost or process gone means the in-flight
  map is requeued once to a healthy lane and that lane stops. Without this a
  dead instance failed jobs in 0.0 s and drained the whole queue.
- **Several independent collectors** (`--claims`, legacy): maps are claimed
  through `O_EXCL` files under the dataset, idle instances rescan the claims
  rather than park (so a map handed back after a crash is picked up), and
  each collector writes `index-<instance_base>.json` instead of overwriting
  one `index.json`. `--instance-base` keeps their user directories and
  profiles apart. `install` skips an unchanged plugin and replaces the file
  atomically, because a later collector rewriting it under running games made
  them fail to compile it.
- Launches staggered; resume skips runs whose `meta.json` says ok;
  `--budget-hours` stops assigning maps, never mid-run.
- Live progress: one bar per instance driven by the race clock, plus an
  aggregate; redirected output stays line-oriented.

## 10. Verification and hygiene (`tools/`)

- `verify` re-reads what is on disk: status, finish time against the replay,
  dropped points, first sample at 0, contiguous indices, 50 ms gaps only,
  coverage to the finish, no frame drawn before its tick, ticks continuous at
  10 ms from 0 to the finish and agreeing with the samples wherever they
  coincide, and `frames.bin` tiled exactly by the rows (or, for `hevc`,
  `frames.mkv` holding exactly one frame per row).
- `clean` moves failing runs to `<dataset>.rejected/<status>/` (a desync is
  worth looking at); `--delete` if not. Do it before training, which reads
  every directory, and before re-collecting.
- `stats`: run-length histogram with median and quartiles, and time per TMX
  tag (time, not run count, is what a tag costs). Tags fetched in batches and
  cached in `tags.json` beside the dataset; `--no-tags` skips it.
- `video`: race time, speed, resolved steer/gas/brake and the held keys drawn
  on every frame — the check that numbers are not shifted against pictures.

## 11. Configuration (`config.py`, `tmnf-collect.yaml`)

- Per-command sections applied via `set_defaults`, so flag > file > built-in.
  Unknown keys and sections are errors. Each committed value carries a note on
  how it was measured. `meta.json` records what a run actually used.
- `TMNF_CONFIG` selects another file; the container uses
  `docker/tmnf-collect.yaml`.

## 12. Linux under Wine (`docker/`)

The controller's Windows-only parts (process discovery, window placement,
clipboard, junctions, `Program Files`) are behind `hostos.IS_WINDOWS` in
`common/hostos.py`, `common/paths.py`, `collect/launcher.py`,
`collect/gamelog.py` and `collect/userdirs.py`. The container runs sway on the
render node (no host display or X server) plus a noVNC console. Measured on a
Ryzen 5 5560U (6 cores, Vega iGPU):

- **Wine's own Direct3D 9, not DXVK.** Through DXVK every lit surface renders
  black. PC3 shaders also render black on Mesa, and the launcher benchmark
  picks them, so `docker/seed/` carries a known-good system config, profile
  and score file for a fresh prefix. Windowed 640x480, Minimum Quality.
- **One Wine virtual desktop per instance.** Wine keeps the foreground window
  per desktop, which is what TMInterface's input injection keys off. On one
  shared desktop a lane's second map failed to arm every time.
- Instances are tiled, not overlapped: the compositor stops sending frames to
  a covered window.
- `autologin` is `1` (the account-less profile); `kill` also removes the Wine
  desktops; the console log is read through the shared Wayland clipboard, so
  `--log` is only trustworthy with one lane.

## Measured limits

| lever | result |
|---|---|
| speed above 1× without barrier | 1.5× lost 10 of 36 runs to dropped frames |
| 5× with barrier, one instance | 3/3 full replays exact; 1,673/1,673 frames on their tick; 4.2–4.8× real time |
| above 5×, one instance | non-monotonic: 6×/10× passed, 8×/15×/20× failed; 20× moved a finish by 20 ms |
| 2× with 2–8 instances (Windows) | 31/31 exact across two maps; 8 instances ≈15.8× aggregate |
| above 2× with several instances | scheduling-sensitive 20 ms finish slips; one instance required |
| 2× with 8 lanes (Linux, 5560U) | 20/20 reproduced, every frame on its tick, 5.9× aggregate; 10 lanes drop frames |
| forced rendering | camera moves ~1 m (resets before drawing) |
| fps throttle | slower and slightly less accurate |
| minimum graphics settings | 78.7% → 99.6% of frames drawn on their own tick |

`bench-matrix` checks whether a setting still reproduces a replay exactly;
`bench-throughput` times the real `collect` under each setting and verifies
the output. Run the matrix several times before trusting an edge setting:
failures there come and go by one physics step.
