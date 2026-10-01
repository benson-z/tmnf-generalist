"""Re-render eval rollouts with the path head's prediction against what happened.

    tmnf-train render-paths <eval folder, map folder, or one rollout video>

For each frame, draws the waypoints the model predicted at that frame
(a car-wide ribbon on the road colored near -> far, haloed by +-1 sigma) and the
path the car actually took over the next 3 s (white line), both from that
frame's camera. Writes ``<video stem>_paths.mp4`` next to the original, at 2x
so the dots are legible. Needs rollouts recorded with path logging (the
``cam`` and ``path`` fields in ``steps/*.jsonl``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tmnf_collect.common.frames import find_ffmpeg

from .projection import draw_path, draw_trail

SCALE = 2
TRAIL_S = 3.0
FPS = 20


def _pose(row: dict) -> dict:
    cam_pos, cam_ypr, fov = row["cam"]
    return {"position": row["pos"], "yaw_pitch_roll": [row["yaw"], 0.0, 0.0],
            "camera_position": cam_pos, "camera_yaw_pitch_roll": cam_ypr, "camera_fov": fov}


def render_rollout(video: Path, steps: Path, out: Path | None = None) -> Path | None:
    rows = [json.loads(l) for l in steps.open(encoding="utf-8") if l.strip()]
    if not rows or "cam" not in rows[0]:
        return None  # recorded before path logging
    out = out or video.with_name(video.stem + "_paths.mp4")
    probe = subprocess.run([find_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    w, h = (int(v) for v in probe.stdout.strip().split(","))
    raw = subprocess.run([find_ffmpeg(), "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    positions = np.array([r["pos"] for r in rows], dtype=np.float64)
    ahead = int(TRAIL_S * FPS)

    enc = subprocess.Popen(
        [find_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w * SCALE}x{h * SCALE}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         str(out)], stdin=subprocess.PIPE)
    assert enc.stdin is not None
    try:
        for i, frame in enumerate(frames[: len(rows)]):
            img = Image.fromarray(frame).resize((w * SCALE, h * SCALE), Image.NEAREST)
            pose = _pose(rows[i])
            path = rows[i].get("path")
            if path:
                mean = np.asarray(path["mean"])
                img = draw_path(img, pose, mean[:, :2], np.asarray(path["std"])[:, 0], path["horizons_s"],
                                scale=SCALE, height=mean[:, 3] if mean.shape[1] > 3 else None)
            d = ImageDraw.Draw(img)
            draw_trail(d, pose, positions[i + 1 : i + 1 + ahead], scale=SCALE)
            if path:
                # Predicted vs actual speed at the furthest horizon.
                k = len(path["horizons_s"]) - 1
                j = i + int(round(path["horizons_s"][k] * FPS))
                actual = rows[j]["speed"] if j < len(rows) else None
                d.text((6, 6), f"pred {path['horizons_s'][k]:.1f}s: {mean[k, 2]:.0f} km/h"
                       + (f" (actual {actual})" if actual is not None else ""), fill=(255, 255, 255))
            d.text((6, 20), "ribbon: predicted path (halo = +-1 sigma)   white line: actual next 3 s",
                   fill=(220, 220, 220))
            enc.stdin.write(img.tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    return out


def render(target: Path, log=print) -> list[Path]:
    """Render every rollout under an eval folder, a map folder, or one video."""
    videos = [target] if target.suffix == ".mp4" else sorted(
        v for v in target.rglob("videos/*.mp4") if not v.stem.endswith("_paths"))
    done = []
    for v in videos:
        # <map>/videos/<ckpt>_<map>_rXX_<outcome>.mp4  ->  <map>/steps/<ckpt>_<map>_rXX.jsonl
        stem = v.stem.rsplit("_", 1)[0]
        steps = v.parent.parent / "steps" / f"{stem}.jsonl"
        if not steps.exists():
            log(f"skip {v.name}: no steps log")
            continue
        out = render_rollout(v, steps)
        if out is None:
            log(f"skip {v.name}: recorded before path logging")
        else:
            log(f"wrote {out}")
            done.append(out)
    return done
