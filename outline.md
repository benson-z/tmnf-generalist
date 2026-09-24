# tmnf-collect: every technique on the collection path

Grounded in `main` as of the "keep only the measured path" commit. Organised by
pipeline stage: `harvest → filter → collect → verify`, with `stats`, `clean`
and `video` for looking at what came out.

## 1. Acquisition — `harvest` (`harvest/tmx.py`)

- **Map-first, not replay-first.** Pick maps, then take a replay *from that
  map's leaderboard*, so map and replay always match by construction — no UID
  search, no missing maps.
- **Awards as the quality signal.** `/api/tracks` ordered awards-descending;
  `min_awards` applied client-side with an early stop, since the ordering
  guarantees nothing better follows.
- **Tag filtering client-side** (`--tags` / `--exclude-tags`) because the API
  silently ignores its tag parameter. Names come from TMX's own
  `enumTrackTagDesc`; ids not in that table render as `tag-N` rather than a
  guess. Well-awarded maps predate multi-tagging and carry exactly one tag.
- **Median of the credible leaderboard** — runs within 1.5× of the map's best
  — rather than the author lap (often 2–13× too slow) or the record
  (edge-of-control, poor state coverage).
- **Over-fetch 3×** because some maps have empty leaderboards.
- Downloads check the `GBX` magic and that the replay's embedded map UID
  matches the map; written to `.part` then renamed; named `tmx-<id>` because
  console commands take paths as bare words. A user agent and a courtesy delay
  on every request.

## 2. Filtering — `filter` (`harvest/filter.py`)

- **Input device read offline from the ghost** with pygbx, ~10 ms a file: a
  `Steer` control entry means pad, `SteerLeft`/`SteerRight` means keyboard.
  Keyboard is instant full lock and a pad is continuous, so mixing them puts
  contradictory labels on the same corner. Bytes are passed rather than the
  path, because pygbx never closes the handle and Windows will not rename an
  open file.
- **Duration cap** read from the uncompressed replay header before any
  parsing: cost is linear in race time, learning signal is roughly per corner.
- **Rejects moved to a sibling folder** `<folder>.rejected/<kind>/` — a
  subfolder would still be found by `collect`'s recursive discovery.

## 3. Staging (`collect/staging.py`, `common/mediatracker.py`, `common/replays.py`)

- **UID → file index** over the Tracks folder and stock campaigns, cached on
  disk, rebuilt on a miss.
- **Everything staged before any game launches** — the game indexes its
  Tracks folder at startup only.
- **MediaTracker clips stripped from the staged copy.** The LZO body is
  decompressed, chunk `0x03043021`'s three inline node refs (intro, in-race,
  end-race) are replaced with nulls through to the next ascending chunk id,
  and the body recompressed. The UID lives in the header and the blocks in
  earlier chunks, so the map is byte-identical where it matters; freed node
  indices become gaps, which is safe because GBX writes each index
  explicitly. Removes the intro flythrough (~20 s a map; some never end
  without a keypress) and the in-race clips that hijack the camera. Checked
  on 465 maps: all stripped, all UIDs preserved. A marker file tracks
  staleness, since a stripped file's size cannot be compared to its source.
- **Inputs extracted with `dump_inputs`**, which reads the ghost from the
  file — no validation run, so no modal dialog blocking the next map load.

## 4. Instance isolation (`collect/launcher.py`, `collect/userdirs.py`)

- **One TMLoader profile per instance**, regenerated from `default.yaml` on
  every start (so a version pin there reaches every instance), adding only
  `/userdir=<path>`.
- **Per-instance user directory**: `Config` and `Profiles` are **mirrored
  from `Documents/TmForever` on every start** — one place to change any
  setting — `Scores` copied once, `Tracks` a junction so staged files are
  visible everywhere. Sharing a profile is what made instances lose their
  key bindings.
- Token, port and instance id go on the command line and the plugin dials
  *back* to the controller. It scans the joined command line because
  TMLoader wraps every argument into one quoted string.

## 5. Session control (`collect/session.py`)

- **Wait for the menu before any command** — commands sent while the game
  initialises are dropped or crash it. The plugin reports its current state
  on connect, because `OnGameStateChanged` only fires on transitions.
- **Commands spaced across frames** — TMInterface runs same-frame commands as
  a stack, last in first out.
- `prepare()`: fps throttle off, `autorewind` off (it rewinds a recording
  mid-flight), `execute_commands` on, `log_bot` on (the only way a wedged
  instance announces itself), load screens skipped, `countdown_speed 1` (a
  fast countdown skips the tick carrying the first input), configurable game
  speed from 1× to 5×, console hidden once.
- **Map load waits for the first non-negative tick**, not just the race
  state — LocalRace is reported while the map is still loading.
- **Camera issued per run** (`cam 1`) once the race is live, because whether
  it survives a map change is not worth assuming.

## 6. Arming and recording a run (`collect/session.py`)

- **Focus before arming.** An instance that has never been the foreground
  window has no input bindings and the car sits at the line; bindings then
  persist for the rest of the run.
- `load <script>`, then `press delete` to restart; wait for `EV_RUN_RESET`;
  **verify the car actually moves** within a grace window, else re-arm (up
  to three attempts).
- **Restart detection**: race time going backwards means an abandoned
  attempt, and everything written is discarded rather than spliced.
- **Overrun guard**: a run past `expected + max(5 s, 20%)` is a desync and is
  stopped rather than recorded to the timeout.
- **The correctness bar is the replay's own finish time**, exact to the
  millisecond. A run that does not match is marked, never quietly kept.
  `unfinished` is deliberately not retried — re-driving a desync was
  measured bit-identical three times over.
- `leave_map` clears the medal screen, which otherwise blocks the next load.

## 7. Sampling in the plugin (`plugin/TMNFCollect.as`)

- **Clocked on physics ticks** — `OnRunStep` at 10 ms; every fifth tick at
  `race_time ≥ 0` is a 20 Hz sample point. Game time, never wall time.
- **Capture on the game's next natural render.** `CaptureScreenshot` is only
  legal inside `Render()`, so the tick stashes telemetry and the next drawn
  frame captures it. Above 1×, a barrier holds the sample state until that
  callback and preserves input edges consumed by any queued extra tick. Each
  sample records both times. A pending sample the game never draws increments
  `dropped`, and a run with any dropped points is rejected.
- **No forced rendering.** TMInterface's forced-render path explicitly resets
  the race camera. The frame barrier uses the regular render path; measured at
  5× it preserved exact cars, inputs, frame ticks and replay finish times.
- **Inputs at 100 Hz**: one `MSG_TICK` per physics step carrying a 4-bit key
  mask, two socket writes — kept lean because a single failed write
  disconnects the plugin for good. 74–84% of keyboard transitions fall between
  20 Hz sample points, so the frame-rate stream alone loses or misplaces most
  taps.
- Telemetry per sample: position, velocity, yaw/pitch/roll, local speed,
  camera pose and FOV, keys and analog inputs, checkpoints, finished, sliding,
  gearbox.
- **HUD off** via `ToggleRaceInterface`, re-applied at each run start since
  map loads restore it.
- Outbound `Net::Socket`, reconnect attempted every second from `Render()`
  (the one callback that runs in the menus); `Connect()`'s return value is
  ignored because it reports false on success.

## 8. Controller and writer (`collect/controller.py`, `collect/protocol.py`, `collect/dataset.py`)

- Little-endian framed protocol with a version check on the hello.
- **The socket reader never blocks.** Frames and ticks go to *unbounded*
  queues on their own writer threads. A bounded queue looked like sensible
  back-pressure and was not: under load it stalled the reader, the plugin's
  writes backed up and failed, and instances disconnected.
- **One `frames.bin` per run**, JPEG 4:2:0 q80, offset and length in
  `samples.jsonl` — 0.067 ms against 0.647 ms per frame for one file each,
  and no 200,000-file corpus to read back. `inputs.jsonl` holds the ticks,
  `meta.json` what was expected against what was recorded.
- `reset()` discards a partial attempt across both queues, in order.

## 9. Scheduling (`collect/runner.py`)

- **Shared work queue** — instances claim the next map as they free up, so
  no instance becomes the tail everyone waits on.
- **Barrier at the end**: no instance closes until the queue is drained,
  because a closing window hands the foreground to a still-recording
  instance, which then drops frames. Waiting instances **stop drawing** so
  they take no GPU. A crashed worker aborts the barrier rather than holding
  the others for the timeout.
- **Dead-instance handling**: socket lost or process gone means the job is
  requeued once and that worker stops claiming. Without this a dead instance
  failed jobs in 0.0 s and drained the whole queue.
- Launches staggered; resume skips runs whose `meta.json` says ok;
  `index.json` records the effective settings so a dataset describes itself.

## 10. Verification and hygiene (`tools/verify.py`, `tools/stats.py`, `tools/video.py`)

- `verify` re-reads what is on disk: status, finish time against the replay,
  dropped points, first sample at 0, contiguous indices, 50 ms gaps only,
  coverage to the finish, no frame drawn before its tick, ticks continuous at
  10 ms from 0 to the finish and agreeing with the samples wherever they
  coincide, and `frames.bin` tiled exactly by the rows.
- `clean` moves failing runs to `<dataset>.rejected/<reason>/` — never
  deletes.
- `stats`: run-length distribution and time per TMX tag, tags fetched in
  batches by id and cached beside the dataset.
- `video`: telemetry drawn over each frame, the check for labels and pixels
  being in step.

## 11. Configuration (`config.py`, `tmnf-collect.yaml`)

- Per-command sections applied via `set_defaults`, so flag > file > built-in.
  Unknown keys and unknown sections are errors, not shrugs. The committed
  file carries the measured defaults.

## Measured limits, for the record

| lever | result |
|---|---|
| game speed above 1× without barrier | 1.5× lost 10 of 36 runs to dropped frames |
| 5× with natural-frame barrier | 3/3 full replays exact; capture interval 4.2-4.8× real time |
| 2× plus parallel instances | 31/31 exact across 2-8 instances and two maps; 8 instances reached 15.8× aggregate |
| above 2× plus parallel instances | scheduling-sensitive 20 ms finish slips; kept single-instance only |
| above 5× with one instance | non-monotonic: 6×/10× passed while 8×/15×/20× failed |
| forced rendering | camera moves ~1 m because TMInterface resets it before drawing |
| fps throttle | slower and slightly less accurate |
| instances and speed | 8 at 2× passed 16/16 synchronized and 7/7 reproducible mixed runs; about 15.8× aggregate |
| minimum graphics settings | 78.7% → 99.6% of frames drawn on their own tick |
