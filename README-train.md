# tmnf-train: behaviour cloning from pixels

Trains a policy that drives TrackMania Nations Forever from the rendered frame
and the HUD speed alone, on data recorded by `tmnf-collect`, and evaluates it
closed-loop in the real game on Nadeo's **B01-Race**, recording every rollout
to video. No RL, no ghost conditioning, no per-map fine-tuning.

Everything large lives under one storage root. Config paths that start with
`$TMNF_STORAGE` resolve to the `TMNF_STORAGE` environment variable, or to
`data/` in the working directory if it is unset; the commands below assume it
is set. The layout:

```
train_data/corpus2/      the recorded corpus (input, read-only)
work/index/              per-run labels + manifest (148 MB), built by `index`
work/cache/<WxH>/        optional uint8 memmap cache, built by `cache`
runs/<run_name>/         config.json, metrics.jsonl, checkpoints/, eval/
eval/<checkpoint>/       standalone evals: rollouts.jsonl, summary.json, videos/, steps/
```

## Reproduce the baseline

```bash
uv sync --extra train
```

```bash
uv run tmnf-train inspect "$TMNF_STORAGE"/train_data/corpus2 --out docs/data_report.json
```

```bash
uv run tmnf-train index
```

```bash
uv run tmnf-train eval --random --id smoke_random
```

```bash
uv run tmnf-train train --config configs/baseline.yaml
```

`configs/v2_chunk_path3d_c23.yaml` is the config behind the demos: soft policy labels,
an 8-step action chunk head, path targets with height, 4 epochs, trained on
corpus2 + corpus3.

```bash
uv run tmnf-train train --config configs/v2_chunk_path3d_c23.yaml
```

`train` resumes with `--resume` (from the newest checkpoint, mid-epoch
included). Evaluate any checkpoint on its own with:

```bash
uv run tmnf-train eval --config configs/v2_chunk_path3d_c23.yaml --checkpoint "$TMNF_STORAGE"/runs/v2_chunk_path3d_c23/checkpoints/<name>.pt
```

`configs/baseline.yaml` is the one file that sets resolution
(`data.downsample`, `data.crop_rows`), window length, frame stride, tokens per
frame, model size, epochs, eval frequency (`train.eval_every_epochs`) and the
aux loss weights; anything can be overridden with `--set section.key=value`.
Other commands: `bench-loader` (loader throughput), `vram` (peak VRAM for one
optimizer step), `cache` (build the memmap cache).

Before the first eval on Windows: the game profile must have the magenta skin
selected (Profile > Vehicles > last page), because the corpus was recorded
with it. Instances copy the profile from `Documents/TmForever` at every
launch.

Tests: `uv run --with pytest pytest tests/test_train.py`.

## Measured on this machine

RTX 5070 Ti Laptop (12 GB), Ryzen AI 9 HX 370, 27.6 GB RAM.

| | 320x240 (default) | 160x120 |
|---|---|---|
| loader, video source, NVDEC, 6 workers | **2650 frames/s** (156 windows/s) | decode-bound, same order |
| model fwd+bwd only, micro-batch 8, bf16 | 1170 frames/s | 3150 frames/s |
| peak VRAM, micro-batch 8 (default) | **2.92 GB allocated / 3.60 GB reserved** | 1.14 / 1.33 GB |
| peak VRAM, micro-batch 16 | 5.45 / 6.72 GB | |
| end-to-end training, default config | **~790 frames/s** (750–825; ~45 min per epoch) | |

"frames" are supervised window frames delivered to the GPU (16 per window, plus
one previous frame). Default effective batch is 32 windows = 4 micro-batches
of 8 with gradient accumulation; `train.micro_batch` changes the micro-batch
without changing the effective batch. One epoch of corpus2 is 3868 optimizer
steps (123,776 windows, ~2 M frames).

End to end, training is bound by the main process and the CPU the decoders
share with it, not by the GPU (~70% busy). The baseline itself averaged ~600
frames/s because it shared the machine with other work; 2 epochs plus two
in-game evals took 117 minutes.

## Baseline result (configs/baseline.yaml, 2 epochs)

| after | val policy CE | val acc | acc steer | B01 eval (4 rollouts, T=0.3) |
|---|---|---|---|---|
| epoch 1 | 0.650 | 73.8% | 77.0% | 4x stationary at 8–11 s, 0 checkpoints |
| epoch 2 | 0.594 | 75.7% | 78.2% | 3x stationary (9–27 s), 1x timeout (60 s), 0 checkpoints |

It does drive: it launches, holds the first straight at ~160 km/h, and
steers for the first corner, but it runs wide, leaves the track and either
pins itself on a wall holding gas (p≈1.0) or wanders off-course on the dirt.
It never reaches checkpoint 1. The epoch-2 checkpoint with exact
`recompute` inference and the same seeds does the same (4x stationary,
9–46 s), so the KV-cache approximation is not what stops it. Videos:
`$TMNF_STORAGE/runs/baseline/eval/<checkpoint>/videos/`.

## Watching a run

```bash
uv run tmnf-train dashboard --config configs/v2_chunk_path3d_c23.yaml
```

Serves a live page on http://127.0.0.1:8765/ (standard library only; add
`--host 0.0.0.0` to reach it from other machines, with no authentication):
progress, ETA, throughput and data-wait; NVIDIA utilisation, memory,
temperature and power (`nvidia-smi`, 2 s); iGPU utilisation, CPU and RAM
(Windows performance counters, 5 s); train/val loss curves and the steer,
brake and no-gas metrics; and the eval table per checkpoint and map, whose
links play each rollout's video in the browser. Hardware history is kept in
memory for an hour, so it starts empty when the dashboard starts. It only
reads the run's files, so it can be started and stopped at any time.

With `train.eval_mode: external` the trainer only writes checkpoints, and a
second process evaluates them as they appear:

```bash
uv run tmnf-train eval-watch --config configs/v2_chunk_path3d_c23.yaml
```

`eval.device: directml` runs the model through ONNX Runtime on the iGPU
(`eval.dml_device_id`, 1 on this laptop), because PyTorch has no Windows build
for the Radeon 890M. The game itself renders on whichever GPU Windows assigns
`TmForever.exe` (Settings > System > Display > Graphics).

## Seeing the path head

Eval videos draw the path head's predicted waypoints (0.5–3 s ahead, cyan
near to orange far) on every frame, with a ring for the predicted lateral
uncertainty (`eval.overlay_path`). Each step's prediction and camera pose go
into `steps/*.jsonl`, and

```bash
uv run tmnf-train render-paths "$TMNF_STORAGE"/runs/v2_chunk_path3d_c23/eval/<checkpoint>
```

re-renders every rollout at 2x as `<video>_paths.mp4` with the path the car
actually took over the next 3 s as a white line, and predicted vs actual
speed 3 s ahead. The projection uses the camera pose sent with each frame;
its conventions were calibrated on corpus2 (see `eval/projection.py`).
Waypoints are 2-D, so they are drawn at the car's current height.

## Eval maps and diagnostics

`eval.maps` lists stock campaign maps by name (`B01-Race`) and corpus runs as
`corpus:<run name>`. v2 evaluates B01 plus two training-split maps
(`tmx-1016190`, `tmx-4912708`) and two held-out ones (`tmx-2655171`,
`tmx-948303`), 8 rollouts each, on 2 game lanes. On a corpus map every step
logs the distance to that run's recorded trajectory, and each rollout reports
`offline_mean_m`, `time_to_leave_line_s` (first time more than
`eval.off_line_m` = 10 m away) and `demo_fraction_reached` (furthest point
along the recorded line reached while on it). These are diagnostics on corpus
maps only; B01 keeps just finish/time/checkpoints. If the model fails the
training maps as badly as B01, the problem is recovery, not generalisation.

The game shows a personal-best autosave of the map as a ghost, which the
corpus never had. Eval sets aside every autosave whose header names an eval
map (renamed in place, restored afterwards, including after a crashed eval),
and moves any run the game autosaves during eval into `<map>/replays/`.

## Policy labels and metrics

`train.label_mode: hard` (baseline) trains on the per-key majority action of
each 50 ms step. `soft` (v2) trains on the share of the step's 5 ticks that
fell in each of the 12 actions, so a 20 ms steer tap becomes a 0.4 target
instead of being rounded away; mirroring moves that mass between left and
right. `ce_hard` (cross-entropy against the majority label) is logged in both
modes so runs stay comparable. Since 95% of labels are gas + a steer choice,
validation also logs recall for left / none / right steer, brake and no-gas,
and steer accuracy on frames where the steer label changes.

## How it fits together

**Eval harness** (`src/tmnf_train/eval/`). The collector's plugin gained a
*drive* mode (protocol v6): every 20 Hz sample tick is held — the same
rewind-and-speed-0 barrier the collector uses — until the controller answers
the captured frame with `CMD_ACTION`, which sets the four keys for the next
five physics ticks. So the policy sees exactly the frames the collector would
have recorded (same capture path, same camera reset), and each action spans
the same 50 ms as its training label. Every rollout counts the ticks whose
applied keys differ from the action sent (`action_tick_mismatch`); it has been
0 on every rollout so far.

A rollout ends on the finish event, a race-time timeout (`eval.timeout_s`,
60 s; B01 bronze is 39.68 s) or `eval.stationary_s` (3 s) below
`eval.stationary_kmh` (10 km/h; a car pushing into a wall reads 4–6 km/h).
There is no respawn. Medal times are read from the map file's header
(author 26.20 s, gold 27.80, silver 31.62, bronze 39.68). A crash or hang
(no message for 30 s) kills the game, relaunches it and re-runs the rollout
with the same seed, up to `eval.max_attempts`. Rollout `k` samples with seed
`eval.seed * 1000 + k` at `eval.temperature` (0 = greedy, then only one
rollout is run). Videos are H.265 via the collector's `VideoEncoder`, with a
strip under the frame showing race time, speed, action index and probability,
and the four keys. `eval.keep_last_n_videos` prunes old checkpoints' videos;
it is `null` (keep all).

**Data** (`src/tmnf_train/data/`). `index` turns each run's JSON into labels
once: the action for frame *i* is the per-key majority over ticks 5i..5i+4;
future waypoints (lateral, forward in the car's heading frame, at 0.5–3 s) and
the future speed; arc length over the next `progress_horizon_s` (4 s) along the
run's own trajectory. Runs with respawns are dropped (16). 2% of runs, by name
hash, are held out for the validation loss. The loader decodes whole runs
(NVDEC via ffmpeg), applies `frames.to_input` (optional horizon crop, then
exact integer box downsampling — the same function eval uses), cuts
non-overlapping 16-frame windows with a random phase per epoch, mirrors half
(image flipped, left/right steer swapped, lateral waypoints negated) and
shuffles within blocks of 12 runs. Batch order is a pure function of
`(seed, epoch)`, which is what makes mid-epoch resume exact.

**Model** (`src/tmnf_train/model.py`, 27.2 M parameters). Per frame: RGB +
previous frame (6 channels) → stride-2 stem (320x240 → 160x120) → IMPALA
stages 32/64/64 with GroupNorm → 20x15 map → 8 attention-pooled tokens, plus a
speed embedding. 16 frames → 128 tokens → 8-layer, width-512 block-causal
transformer with rotary frame positions. Heads on the mean of each frame's
tokens: 12-way policy, path (mean + log-variance, Gaussian NLL) and progress
(MSE), all supervised at all 16 positions. `Observation` (frames, speed) is
the model's only input; labels are a separate `Labels` tuple, and
`tests/test_train.py` asserts the boundary.

Inference: `eval.inference: kv_cache` (default) runs one frame per step on a
rolling 16-frame KV cache; `recompute` caches each frame's CNN tokens and
re-runs the transformer over the last 16 frames, which matches training
exactly. See "Spec notes" for why both exist.

## Assumptions

* "Labels assigned by majority within each window" is read as: within each
  50 ms step (5 ticks), each of gas, brake and steer is on if held for ≥3 of 5
  ticks; a tick with left and right both held counts as no steer.
* The step's action starts at the frame's tick: a frame at race time *t* is
  labelled by ticks *t*..*t*+40, and in eval the action is applied from the
  same tick. The recorded tick at *t* itself still shows the previous action
  (TMInterface reports inputs before the rewind); ticks *t*+10..*t*+40 are
  verified per rollout.
* "Speed" is the HUD display speed (`speed_kmh` / `DisplaySpeed`), km/h,
  divided by 300 before embedding.
* The previous frame for a window's first frame is the frame before it (or the
  frame itself at race start); context frames (t-20/40/60) are paired with
  themselves.
* The future path is 2-D in the ground plane (lateral, forward) plus future
  speed; height is ignored. The progress target is truncated (masked) within
  4 s of the finish.
* Countdown and post-finish frames: the recorder already starts at race time 0
  and stops at the finish, so nothing had to be dropped; respawn runs are
  dropped whole by default (masking is the alternative).
* The existing `frames.mkv` already are H.265 with a closed GOP of 40, so they
  are used as they are rather than re-encoded (7.5% smaller, a second lossy
  generation).
* Eval runs one game instance at 2x between held steps; `eval.lanes` runs more.

## Spec notes: what I think is wrong or worth deciding

1. **The corpus is not what the brief says**: 2483 distinct maps and 28.8 h,
   not ~1200 maps and ~15 h. No map repeats, and B01-Race (like every Nadeo
   map) is not in it, so eval measures generalisation to an unseen map.
2. **KV cache vs. 16-frame windows**: training sees fresh 16-frame windows,
   but a rolling KV cache is sliding-window attention: once it slides, the
   retained frames' deeper-layer keys were computed with context the training
   window never has. Output drift was ~0.02 in logits on a small test model.
   `eval.inference: recompute` removes the mismatch at the cost of a 128-token
   transformer pass per step (still one CNN pass per frame). I kept KV cache as
   the default because the brief asks for it, but for "can it drive at all" I
   would use `recompute`.
3. **Video retention** conflicts between step 2 ("keep the most recent N
   checkpoints' videos, delete older ones") and step 5 ("keep every
   checkpoint's videos"). Per your follow-up, nothing is deleted by default.
4. **Tokens per window**: 16 frames x 8 tokens = 128 is at the top of the
   64–128 range; context frames add 8 each (152 with three).
5. **Steering below the action rate**: ~28% of steer presses are shorter than
   10 ticks and 3.2% of (step, key) pairs are mixed, so a 50 ms categorical
   quantises a real part of how keyboard players steer. The action set cannot
   express a 20 ms tap.
6. **Label imbalance**: 98% of ticks have gas held and 95% of labels are
   gas + {left, none, right}. Accuracy on gas/brake is uninformative; watch
   `acc_steer`.
7. **HUD speed is not deterministic**: two rollouts with identical actions
   differed by ±1 km/h in the displayed speed on 4 of 287 steps (the game
   smooths it on render timing). Random-policy rollouts reproduce exactly;
   a model rollout can diverge from that jitter. Using speed from velocity
   would be deterministic, but would not be what the HUD shows.
8. **Stationary cutoff** had to be 10 km/h, not ~0: a car pinned against a wall
   jitters at 4–6 km/h and would otherwise idle to the timeout.
9. **The action at tick *t*** of each step is reported by TMInterface before
   the new keys are applied, so exact tick-level alignment of eval vs. the
   recorded label could be off by one 10 ms tick; not measurable with the
   current plugin hooks.
10. **Eval on one map, 4 rollouts** gives a very coarse finish-rate signal;
    that is fine for watching videos, but the logged metrics will be noisy.

## Changes outside `tmnf_train`

* `plugin/TMNFCollect.as` and `collect/protocol.py`: drive mode, `CMD_ACTION`,
  protocol v6 (collection is unaffected; reinstall happens automatically).
* `collect/controller.py`: `act()`, `configure(drive=...)`.
* `collect/session.py`: `start_driven_run()` / `stop_driven_run()`.
* `collect/launcher.py`: fixed an undefined `WNDENUMPROC` that broke
  `offscreen` on Windows.
* `pyproject.toml`: `tmnf-train` entry point now `tmnf_train.cli:main`.
