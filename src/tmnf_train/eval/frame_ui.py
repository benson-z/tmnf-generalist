"""What an eval video frame shows: the game frame, upscaled, with the path
head's prediction drawn into it and an info strip underneath.

Everything is drawn from one row of the rollout's steps log
(``steps/*.jsonl``), so a rollout recorded without overlays can be redrawn
offline exactly as the harness would have drawn it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import actions
from .projection import draw_path

OVERLAY_H = 28  # strip height at scale 1

_FONTS: dict[int, object] = {}
_BG, _TEXT, _DIM, _TRACK = (16, 16, 20), (236, 238, 242), (150, 156, 168), (44, 47, 54)


def _font(size: int):
    # Consolas/Arial on Windows; DejaVu in the Linux container.
    if size not in _FONTS:
        _FONTS[size] = ImageFont.load_default()
        for name in ("consola.ttf", "DejaVuSansMono.ttf", "arial.ttf", "DejaVuSans.ttf"):
            try:
                _FONTS[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
    return _FONTS[size]


def _confidence_color(p: float) -> tuple[int, int, int]:
    """Red (unsure) through amber to green (sure) for the chosen action's probability."""
    stops = ((0.0, (235, 80, 80)), (0.5, (245, 190, 60)), (1.0, (90, 210, 120)))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p <= p1:
            a = (max(p, p0) - p0) / (p1 - p0)
            return tuple(int(round((1 - a) * u + a * v)) for u, v in zip(c0, c1))
    return stops[-1][1]


def overlay(frame: np.ndarray, race_ms: int, speed: int, checkpoint: int, action: int,
            prob: float | None, scale: int = 1) -> np.ndarray:
    """The frame with a strip underneath: time, speed, checkpoints reached, a bar
    for the chosen action's probability, and the keys it holds. ``scale`` is
    how far the frame was upscaled from the game's 320x240; the strip grows
    with it."""
    font = _font(13 * scale)
    u = scale
    h, w, _ = frame.shape
    strip = OVERLAY_H * u
    canvas = Image.new("RGB", (w, h + strip), _BG)
    canvas.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(canvas)
    mid = h + strip // 2
    text = f"{race_ms / 1000:6.2f}s {speed:3d}km/h CP{checkpoint}"
    d.text((4 * u, mid), text, font=font, fill=_TEXT, anchor="lm")

    key_w = 20 * u
    x0 = w - 4 * key_w - 3 * u
    up, down, left, right = actions.keys(action)
    for k, (label, on, col) in enumerate(
        (("<", left, (90, 200, 250)), ("^", up, (90, 200, 250)), ("v", down, (250, 110, 110)), (">", right, (90, 200, 250)))
    ):
        box = (x0 + k * key_w, h + 5 * u, x0 + (k + 1) * key_w - 3 * u, h + strip - 5 * u)
        d.rectangle(box, fill=col if on else None, outline=col if on else (70, 74, 82), width=u)
        d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font=font,
               fill=_BG if on else _DIM, anchor="mm")

    if prob is not None:
        # Confidence: how much of the policy's mass was on the action it took.
        bx0, bx1 = int(4 * u + d.textlength(text, font=font)) + 8 * u, x0 - 8 * u
        if bx1 - bx0 >= 16 * u:
            by0, by1 = mid - 4 * u, mid + 4 * u
            d.rectangle((bx0, by0, bx1, by1), fill=_TRACK)
            fill_x = bx0 + round(float(np.clip(prob, 0.0, 1.0)) * (bx1 - bx0))
            if fill_x > bx0:
                d.rectangle((bx0, by0, fill_x, by1), fill=_confidence_color(prob))
            half = (bx0 + bx1) // 2  # tick at p = 0.5
            d.line((half, by0 - 2 * u, half, by1 + 2 * u), fill=_DIM, width=u)
    return np.asarray(canvas)


def row_pose(row: dict) -> dict:
    """The car and camera pose of a steps-log row, as ``draw_path`` takes it."""
    cam_pos, cam_ypr, fov = row["cam"]
    return {"position": row["pos"], "yaw_pitch_roll": [row["yaw"], 0.0, 0.0],
            "camera_position": cam_pos, "camera_yaw_pitch_roll": cam_ypr, "camera_fov": fov}


def compose(frame: np.ndarray, row: dict, scale: int = 2, strip: bool = True, path: bool = True) -> np.ndarray:
    """One video frame: ``frame`` (the game's RGB capture) upscaled by an integer
    ``scale`` (each pixel a scale x scale block), the predicted path drawn at
    that resolution, and the info strip under it."""
    scale = max(1, int(scale))
    h, w, _ = frame.shape
    img = Image.fromarray(frame)
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    pred = row.get("path")
    if path and pred:
        mean = np.asarray(pred["mean"])
        img = draw_path(img, row_pose(row), mean[:, :2], np.asarray(pred["std"])[:, 0], pred["horizons_s"],
                        scale=scale, frame_w=w, frame_h=h, height=mean[:, 3] if mean.shape[1] > 3 else None,
                        speed=mean[:, 2], speed_now=row["speed"])
    out = np.asarray(img)
    if not strip:
        return out
    probs = row.get("p")
    prob = float(probs[row["action"]]) if probs is not None else None
    return overlay(out, row["t"], row["speed"], row.get("cp", 0), row["action"], prob, scale=scale)
