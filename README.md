# tmnf-collect

Unattended data collection from TrackMania Nations Forever for imitation
learning. Give it a folder of replays and it re-drives each one in the game,
recording what the driver saw and what they pressed:

* **frames** at 20 Hz, 320x240, lossless,
* **key inputs** at 100 Hz (every physics tick),
* **car telemetry and camera pose** on every frame,
* **a magenta car** (body only; see [skins/](skins/make_magenta.py)), the
  same in every run, so the car never takes on a replay's own paint,

all clocked off the game's physics, not wall time. Every run is checked against
the replay's own finish time to the millisecond, so a run that did not
reproduce is flagged instead of quietly ending up in the dataset.

It can also build the replay corpus for you from
[TMX](https://tmnf.exchange): pick well-rated maps, download a mid-leaderboard
run for each, and keep only keyboard drivers.

How it works, and why it works that way, is in [outline.md](outline.md).
Training and evaluating a driving model on what it records is in
[README-train.md](README-train.md); the corpus itself is described in
[docs/DATA_REPORT.md](docs/DATA_REPORT.md).

## Requirements

**Windows:** TrackMania Nations Forever, TrackMania ModLoader (TMLoader) with a
profile that has TMInterface 2.2+ enabled, Python 3.11+ and
[uv](https://docs.astral.sh/uv/).

**Linux:** just Docker and a GPU render node (`/dev/dri`). The game, TMLoader
and TMInterface run under Wine inside the container. See [Linux](#linux) below.

## Quick start (Windows)

```bash
uv sync
uv run tmnf-collect paths           # check what was detected on this machine
uv run tmnf-collect install-plugin  # copy the plugin into TMInterface
uv run tmnf-collect smoke           # launch, drive a fixed script, save frames
```

If `smoke` writes frames to `out/smoke`, the setup works.

## Collecting a dataset

The usual pipeline is harvest → filter → collect → verify → clean:

```bash
# 1. fetch maps and one replay each from TMX (--dry-run to preview the budget)
uv run tmnf-collect harvest --out testdata/corpus

# 2. keep keyboard runs up to 3 minutes; the rest move to testdata/corpus.rejected/
uv run tmnf-collect filter testdata/corpus

# 3. record everything
uv run tmnf-collect collect testdata/corpus --out out/corpus

# 4. check what landed on disk, then move failed runs aside
uv run tmnf-collect verify out/corpus
uv run tmnf-collect clean out/corpus
```

Collection resumes: re-running the same `collect` command skips replays that
already recorded successfully. `--budget-hours N` stops handing out new maps
after N hours, so a long corpus can be done over several sittings.

Expect losses along the way. About half of harvested replays are pad rather
than keyboard, and roughly one in ten will not reproduce in-game, so harvest
more than you need. The committed defaults already do (`limit: 400`).

Your own replays work too: point `collect` at any folder of `.Replay.Gbx` files.
If their maps are not installed, add `--fetch-maps` to download them from TMX
by UID.

### Looking at the result

```bash
uv run tmnf-collect stats out/corpus                          # run lengths and time per map tag
uv run tmnf-collect video out/corpus/A07-Race.Replay --out a07.mp4  # one run with telemetry drawn on
```

`video` is the quickest way to confirm that frames and labels are in step. It
needs `ffmpeg` on PATH.

## What a dataset looks like

```
out/corpus/
  index.json               every replay, its status, and why anything was skipped
  <replay name>/
    frames.mkv             every frame, H.265 CRF 18, one video frame per row,
                           a keyframe every 2 s (meta.json: keyframe_interval)
                           (frames.bin, QOI + zstd back to back, with
                           frame_codec: qoi-zstd)
    samples.jsonl          one row per frame: race time (and, for qoi-zstd, its
                           offset in frames.bin),
                           keys, analog gas/steer, position, velocity, orientation,
                           speed, checkpoints, camera pose
    inputs.jsonl           one row per 10 ms physics tick: the keys held
    meta.json              expected vs. recorded, and the settings used
```

A run's status is `ok`, or says why not: `time_mismatch` (finished at a
different millisecond from the replay), `unfinished`, `no_inputs`,
`game_crashed` (the map crashes the game, so it is set aside rather than
retried), and so on.
Add `--log` to `collect` to also save the in-game TMInterface console to
`gamelog.txt`, which is the fastest way to see why a run misbehaved.

## Settings

`tmnf-collect.yaml` in the working directory sets the defaults for every
command, and command-line flags override it. `--config other.yaml` reads a
different file. Unknown keys are an error, so a typo cannot silently do
nothing. The committed file holds settings measured on the collection machine,
each with a note on how it was measured. The main ones:

| setting | default | |
|---|---|---|
| `processes` | 8 | game instances running in parallel |
| `speed` | 2 | simulation speed; above 2x needs a single instance (max 5x) |
| `camera` | 1 | race camera; use the same one at inference time |
| `width` / `height` | 320x240 | half of the 640x480 render |
| `show_ui` | false | hide the speedometer and clock from frames |
| `keep_intros` | false | strip map intro clips from the staged copy |

To re-measure speed and parallelism on new hardware, use `bench-matrix` (does
a setting still reproduce a replay exactly?) and `bench-throughput` (how fast is
it end to end?).

While collecting, avoid clicking on or screenshotting the game windows. Taking
focus away from a running instance can make it drop frames or desync.

## Linux

`docker/` runs the whole rig on a Linux host with nothing but Docker installed:
the game under Wine, a headless compositor on the GPU, and a browser console to
watch it.

```bash
cd docker
cp .env.example .env           # console password, /dev/dri gids, dataset dir
mkdir dl                       # put tmnationsforever_setup.exe and TMLoader-latest.zip here
docker compose up -d --build   # the first start installs the game into the prefix
docker exec -w /work tmnf uv sync
docker exec -w /work tmnf uv run tmnf-collect smoke --out /out/smoke
docker exec -w /work tmnf uv run tmnf-collect collect /out/harvest --out /out/dataset
```

The console is at `http://<host>:6080/vnc.html?autoconnect=1&resize=scale`,
behind the password from `.env`. It shows every game instance tiled on one
desktop. Super+Return opens a terminal and Super+t opens the TMLoader UI.
The Wine prefix and game data persist in `docker/data/home`, and datasets go to
`docker/data/out` (or `TMNF_OUT`). The container reads
`docker/tmnf-collect.yaml`, which holds the settings measured there.

## All commands

| command | |
|---|---|
| `paths` | show the detected game, TMLoader and TMInterface locations |
| `install-plugin` | copy the plugin into TMInterface |
| `launch` / `kill` | start one instance / stop every instance |
| `smoke` | end-to-end check with a fixed script |
| `harvest` | pick maps on TMX and download a replay for each |
| `filter` | sort replays by input device and length |
| `collect` | record replays into a dataset |
| `verify` / `clean` | check a dataset / move failed runs aside |
| `stats` / `video` | summarise a dataset / render one run as video |
| `bench-matrix` / `bench-throughput` | measure settings on this machine |

`uv run tmnf-collect <command> --help` lists each command's options.
