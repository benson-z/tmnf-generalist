# Rendering experiments

Branch: `codex/decouple-render-ticks`.

`render_probe.py` drives the same six-second script on A01 and compares natural
1x rendering with the supported 5x frame barrier. It saves camera/car
transforms, selected PNGs, and a summary under the requested output directory.
It uses one owned game instance (ID 19, port 8499).

```powershell
.venv/Scripts/python.exe experiments/render_probe.py --out out/render-probe
```

The plugin had an unfinished forced-render experiment when this branch was
created. Its undeclared `p_dropped` assignment was removed; the real dropped
frame counter retains its original meaning. Camera pose is now measured in
Render, next to screenshot capture, for all comparisons.

## Findings so far

1. More forced renders do not eliminate the camera difference. The original
   `out/render-probe` run compared natural rendering twice, sample-point forced
   renders at 1x/5x, and forced renders every tick. All 120 frames matched their
   ticks and all car positions matched exactly. Natural repeat mean camera
   difference was 0.023 m; every forced schedule remained around 1.0 m away.
2. `SimulationOnly` plus sample-point forced rendering runs the capture interval
   at 9–10x real time, but retains the camera mismatch (`out/render-probe2`).
   Map-loading/countdown overhead is excluded from the capture speed measurement.
3. Read-only disassembly reveals that **ForceGameRender explicitly resets the
   camera**. In the installed TMInterface 2.2.1, the function at RVA `0x842d0`
   calls through the pointer at RVA `0x2502e8` at instruction RVA `0x8437f`.
   The separately registered `ResetCamera` implementation at RVA `0x9d380`
   resolves the same object and jumps through exactly that pointer. This is
   stronger evidence than the earlier hypothesis about render frequency alone.
4. Skipping that one call reduces the mean difference to 0.070 m at 1x and
   0.196 m at 5x (`out/render-probe3`). Simulation-only still has about 0.55 m
   error at 20 Hz and 0.52 m with forced renders at 100 Hz. Removing the reset
   addresses a real cause but does not by itself reproduce every natural camera
   update in simulation-only mode.
5. Pausing simulation at a sample point is not initially a hard boundary: ticks
   already queued for the current game-loop iteration still execute. Saving the
   sample state, restoring it on those callbacks, and carrying forward input
   edges consumed by the script scheduler makes it one. At requested 5x, three
   complete replays finished at the source millisecond with identical 100 Hz
   input streams, exact car positions, and 1,673/1,673 frames on their own tick.
   Capture ran 4.2-4.8x real time. This is the supported `--speed 5` path.
6. Requested 20x is beyond the reliable input/simulation range. One of three
   replays finished 20 ms late and its input stream differed, despite every
   image reporting the intended sample tick. Production therefore accepts only
   speeds from 1 through 5.
7. Parallel fast mode has a reliable 2x envelope. `speed_instance_matrix.py`
   ran A07 at 1x through 20x with one instance and at several speeds with 2,
   3, 4, and 6 instances. Higher speeds were scheduling-sensitive: apparently
   successful settings failed in other rounds by one 20 ms physics step. In
   contrast, 2x passed 31/31 concurrent runs across instance counts 2, 3, 4,
   6, and 8, including eight A07 and eight C03 runs. Eight instances delivered
   15.8x aggregate capture speed with identical inputs, exact car positions,
   and exact finish times. Production permits parallel instances through 2x;
   speeds above 2x still require one instance.
   An end-to-end eight-replay production run then passed all seven known-good
   replays on their first attempt. B08, which is known not to reproduce at
   natural speed either, was correctly reported unfinished.
8. A single instance can sometimes run well above 5x, but it is not reliable.
   A07 passed at 6x and 10x while failing at 8x, 15x, and 20x in the same
   sweep. The non-monotonic result points to scheduling luck rather than a safe
   higher limit, so production remains capped at 5x.

Run a matrix with:

```powershell
.venv/Scripts/python.exe experiments/speed_instance_matrix.py `
  testdata/replays12/A07-Race.Replay.gbx `
  --out out/speed-instance-matrix --instances 8 --speeds 2 `
  --reference out/replay-render-probe/natural1/A07-Race.Replay
```

`render_patch.py` changes only the owned process's memory, checks the expected
instruction (including relocated pointer), restores the original memory
protection, and restores the instruction on exit. It does not modify DLL files.
It is a version-specific research tool, not a portable rendering API.

`inspect_render.py` reproduces the read-only disassembly. Its optional dependencies
are isolated in `out/render-tools` (`uv pip install --target out/render-tools capstone pefile`).

## Relevant API documentation

- [ForceGameRender](https://donadigo.com/tminterface/plugins/api/Graphics/ForceGameRender):
  invokes Render and supports screenshot capture during simulation-only runs.
- [SimulationOnly](https://donadigo.com/tminterface/plugins/api/global/SimulationManager/set_SimulationOnly):
  skips other game systems; plugin must keep servicing commands through forced renders.
- [Running](https://donadigo.com/tminterface/plugins/api/global/SimulationManager/set_Running):
  pauses simulation while rendering continues; Render can resume simulation.
- [ResetCamera](https://donadigo.com/tminterface/plugins/api/global/SimulationManager/ResetCamera):
  removes temporal camera movement.
- [Resync](https://donadigo.com/tminterface/plugins/api/global/SimulationManager/Resync):
  updates performance timers to avoid catch-up simulation after blocking work.

Windows repaint messages request painting through the application's window
procedure; they do not specify a game physics tick or guarantee a camera update.
[Microsoft WM_PAINT documentation](https://learn.microsoft.com/en-us/windows/win32/gdi/wm-paint).
An engine-level pause/resume experiment is therefore being tried before an OS
repaint experiment. Linux/Wine has not been tested; changing graphics backends
would not remove the explicit reset inside the same TMInterface DLL.
