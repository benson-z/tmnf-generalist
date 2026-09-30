# corpus2: data inspection report (step 1)

Source: `Z:\application_storage\tmnf-ml\train_data\corpus2`. Produced by
`python -m tmnf_train inspect <corpus> --out docs/data_report.json`, which reads
every run directory. The full numbers are in [data_report.json](data_report.json).

## Headline differences from the brief

| | brief | on disk |
|---|---|---|
| distinct maps | ~1200 | **2483**, each driven exactly once |
| total driving | ~15 h | **28.8 h** (2483 ok runs) |
| B01-Race in the corpus | — | no (no Nadeo campaign map is in it), so eval is on an unseen map |

## Layout and formats

```
corpus2/
  index.json                  collector's report: settings used, every job's status
  tmx-<id>/                   one directory per replay (2518)
    frames.mkv                H.265, 320x240, 20 fps, one video frame per sample row
    samples.jsonl             one JSON row per frame (20 Hz)
    inputs.jsonl              one JSON row per physics tick (100 Hz)
    meta.json                 per-run settings and outcome
```

Status: 2483 `ok`, 25 `unfinished` (desynced re-drives), 10 without `meta.json`
or `frames.mkv` (game crashed on the map). Only `ok` runs are used.

All runs were recorded by the Linux/Wine collector: `frame_codec: hevc-vaapi`
(constant QP 24), `camera: 1`, `hide_ui: true`, `speed: 2.0`, `frame_barrier: true`,
`reset_camera: true`, `period_ms: 50`, `frame_size: [320, 240]`,
`keyframe_interval: 40`, magenta car skin.

**frames.mkv** (ffprobe, 20 runs sampled): `hevc` Main, `yuv420p`, 320x240,
20/1 fps, no B-frames, keyframes at exactly every 40th frame (closed GOP),
packet count equal to the number of sample rows in 20/20.

**samples.jsonl** fields: `i`, `race_time` (ms), `render_race_time`, `speed_kmh`
(HUD display speed, int), `up down left right` (keys at that tick), `gas`,
`steer` (raw analog, 0 for keyboard), `gas_f brake_f steer_f` (resolved
inputs), `position[3]`, `velocity[3]` (m/s, world), `yaw_pitch_roll[3]`,
`local_speed[3]`, `camera_position[3]`, `camera_yaw_pitch_roll[3]`,
`camera_fov`, `checkpoints`, `finished`, `sliding`, `gearbox`, `dropped_so_far`.

**inputs.jsonl** fields: `race_time`, `up`, `down`, `left`, `right`.

**meta.json** fields: `replay`, `map_uid`, `map_file`, `replay_respawns`,
`replay_race_time_ms`, `recorded_finish_time_ms`, `status`, `detail`,
`samples`, `input_ticks`, `tick_period_ms`, `camera`, `speed`,
`frame_barrier`, `dropped_sample_points`, `arming_retries`, `driving`,
`restart_observed`, `clean_start`, `period_ms`, `frame_size`, `frame_codec`,
`keyframe_interval`, `hide_ui`, `reset_camera`.

## Keys and alignment (checked on all 2483 ok runs)

* Sample row `i` has `i == row index` and `race_time == 50*i` in every run; video
  frame `i` is row `i` (packet count == row count).
* Tick row `j` has `race_time == 10*j` in every run, contiguous from 0.
* `render_race_time == race_time` for every sample: each frame shows the state
  at its own tick (no render lag).
* Keys in sample `i` equal the keys of tick `5i` in every row of every run (0 mismatches).
* The streams start at race time 0 (after the countdown). The ticks run to the
  finish tick inclusive; the samples stop at the last 50 ms multiple ≤ the finish,
  so the last frame has 0–4 ticks after it. `finished` is never true in any
  sample row. So: **no countdown or post-finish frames exist in the data.**
* Label used for frame `i`: ticks `5i..5i+4` (the 50 ms the policy holds after
  seeing frame `i`); a frame without all five ticks gets no action label.

## Run length and repeats

28.8 h over 2483 runs; min 21.7 s, p25 35.6 s, median 41.6 s, p75 46.7 s,
p95 57.7 s, max 73.1 s (the harvest was capped at 25–75 s author times).

| seconds | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 |
|---|---|---|---|---|---|---|
| runs | 233 | 786 | 1082 | 308 | 67 | 7 |

2,074,121 frames and 10,365,626 ticks. **No map appears more than once**
(2483 distinct UIDs and 2483 distinct map files).

Respawns: 16 replays report respawns (`replay_respawns` 1–5); position jumps
(> 10 m between frames, well beyond velocity) are found in 6 of them. The
pipeline drops those 16 runs by default (`drop_respawn_runs`), otherwise it
masks 2 s before / 1 s after each jump.

## Keys: press durations and inter-press gaps, in 10 ms ticks

Held fraction of all ticks: up 98.3%, down 4.4%, left 26.5%, right 27.3%.
Ticks with left and right both held: 2,277 (0.02%). Ticks with up and down both
held: 398,698 (3.8%) — brake is mostly applied *with* gas held.

| | presses | p5 | p25 | median | p75 | p95 | max | < 5 ticks |
|---|---|---|---|---|---|---|---|---|
| up press | 10,457 | 13 | 131 | 525 | 1146 | 4117 | 7309 | 0.6% |
| up gap | 7,976 | 4 | 8 | 13 | 24 | 55 | 1300 | 6.7% |
| down press | 12,622 | 5 | 11 | 26 | 55 | 92 | 4891 | 2.8% |
| down gap | 10,910 | 13 | 88 | 306 | 474 | 864 | 4884 | 0.3% |
| left press | 71,890 | 5 | 9 | 14 | 33 | 177 | 7062 | 4.0% |
| left gap | 69,409 | 6 | 10 | 24 | 83 | 464 | 2783 | 2.5% |
| right press | 71,223 | 5 | 9 | 15 | 35 | 182 | 6045 | 3.9% |
| right gap | 68,742 | 6 | 11 | 24 | 83 | 471 | 2488 | 2.5% |

Histograms (count of presses / gaps with that length):

| ticks | up press | up gap | down press | down gap | left press | left gap | right press | right gap |
|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 35 | 13 | 0 | 97 | 80 | 126 | 84 |
| 2 | 16 | 83 | 40 | 6 | 296 | 205 | 268 | 197 |
| 3 | 20 | 192 | 117 | 16 | 1050 | 613 | 1047 | 600 |
| 4 | 24 | 222 | 184 | 14 | 1452 | 836 | 1319 | 803 |
| 5 | 44 | 317 | 308 | 26 | 2366 | 1531 | 2248 | 1370 |
| 6-7 | 85 | 795 | 815 | 83 | 6601 | 5121 | 6307 | 4627 |
| 8-9 | 96 | 975 | 1012 | 139 | 8402 | 6787 | 8015 | 6478 |
| 10-14 | 311 | 1788 | 1912 | 383 | 17053 | 11039 | 16201 | 11284 |
| 15-19 | 259 | 965 | 1053 | 328 | 7901 | 5419 | 7805 | 5572 |
| 20-29 | 407 | 1202 | 1292 | 495 | 7313 | 6303 | 7487 | 6494 |
| 30-49 | 501 | 909 | 2261 | 688 | 6436 | 7490 | 6848 | 7545 |
| 50-99 | 644 | 448 | 3221 | 631 | 5871 | 8522 | 6203 | 8430 |
| 100-199 | 605 | 41 | 385 | 914 | 4002 | 6096 | 4231 | 5882 |
| 200-499 | 2028 | 3 | 3 | 4754 | 3029 | 6368 | 3086 | 6343 |
| >=500 | 5411 | 1 | 6 | 2433 | 21 | 2999 | 32 | 3033 |

What this means for a 50 ms (5-tick) action: the median steer tap is 14–15
ticks (3 steps), and ~28% of steer presses are shorter than 10 ticks, so
steering is often a sub-step duty cycle. 3.2% of (5-tick block, key) pairs are
mixed (key held for 1–4 of the 5 ticks); majority labelling rounds those.

Resulting label distribution over the 2.06 M training frames (12 actions):
gas+none 45.7%, gas+right 24.7%, gas+left 24.1%, gas+brake (any steer) 3.9%,
no gas (any) 1.7%. The policy head's floor is therefore dominated by the
left/none/right choice.

## Bytes on disk and H.265

| | bytes |
|---|---|
| all run dirs | 17.81 GB |
| ok runs | 17.68 GB |
| frames.mkv | 15.76 GB (7.6 KB/frame) |
| samples.jsonl | 1.23 GB |
| inputs.jsonl | 0.82 GB |

The frames are **already H.265, one file per run, closed GOP of 40**, which is
inside the requested 20–50 range. A trial re-encode of 6 runs with x265
(`-preset medium -crf 20`, keyint 20, no B-frames) came out at 0.925x the
size, projecting **~14.6 GB** for all frames. That is a 1.2 GB saving bought
with a second generation of lossy coding, so the pipeline reads the existing
files as they are and does not re-encode. The label index derived from the
JSON is 148 MB.

A uint8 memmap cache (optional, `tmnf-train cache`) would be ~477 GB at
320x240 or ~119 GB at 160x120.
