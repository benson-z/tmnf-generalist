"""Rendering a recorded run back into a video, with its labels drawn on.

This is a sanity check you can watch: if the inputs and speed drawn on each
frame do not match what the car is visibly doing, the frames and the telemetry
are not lined up, however good the numbers look.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

STEER_FULL = 65536  # analog steer range is [-65536, 65536]

_BG = (16, 16, 20)
_DIM = (70, 74, 82)
_TEXT = (236, 238, 242)
_MUTED = (150, 156, 168)
_ON = (90, 200, 250)
_BRAKE = (250, 110, 110)


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError(
            "ffmpeg not found on PATH; install it or add it to PATH to render video"
        )
    return found


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("consola.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_time(ms: int) -> str:
    minutes, rest = divmod(max(0, ms), 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def _key_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    on: bool,
    font: ImageFont.ImageFont,
    colour: tuple[int, int, int] = _ON,
) -> None:
    fill = colour if on else None
    draw.rectangle(box, fill=fill, outline=colour if on else _DIM, width=2)
    text_colour = _BG if on else _MUTED
    left, top, right, bottom = box
    draw.text(
        ((left + right) / 2, (top + bottom) / 2),
        label,
        font=font,
        fill=text_colour,
        anchor="mm",
    )


def _frame_at(blob, row: dict) -> io.BytesIO:
    """Pull one frame out of the run's single frames file."""
    blob.seek(row["frame_offset"])
    return io.BytesIO(blob.read(row["frame_bytes"]))


def render_frame(
    image: Image.Image, row: dict, *, scale: int, panel: int
) -> Image.Image:
    """One dataset frame, scaled up with a telemetry panel underneath."""
    width, height = image.width * scale, image.height * scale
    view = image.resize((width, height), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height + panel), _BG)
    canvas.paste(view, (0, 0))
    draw = ImageDraw.Draw(canvas)

    big = _font(max(15, panel // 4))
    small = _font(max(11, panel // 7))

    top = height + panel // 8
    draw.text((14, top), _format_time(row["race_time"]), font=big, fill=_TEXT)
    draw.text(
        (14, top + panel // 3),
        f"{row['speed_kmh']:>3} km/h",
        font=big,
        fill=_TEXT,
    )
    draw.text(
        (14, top + panel // 3 + panel // 4),
        f"cp {row['checkpoints']}",
        font=small,
        fill=_MUTED,
    )

    # Speed bar, 0..500 km/h.
    bar_x, bar_w = 190, max(120, width // 3)
    bar_y, bar_h = top + 6, max(10, panel // 8)
    draw.rectangle(
        (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline=_DIM, width=2
    )
    filled = int(bar_w * min(1.0, row["speed_kmh"] / 500.0))
    if filled > 2:
        draw.rectangle(
            (bar_x + 2, bar_y + 2, bar_x + filled, bar_y + bar_h - 2), fill=_ON
        )

    # Steering: centre-anchored bar. Keyboard replays leave the raw analog
    # value at 0, so this follows the steer the car actually resolved.
    steer = max(-1.0, min(1.0, row["steer_f"]))
    steer_y = bar_y + bar_h + max(8, panel // 8)
    centre = bar_x + bar_w / 2
    draw.rectangle(
        (bar_x, steer_y, bar_x + bar_w, steer_y + bar_h), outline=_DIM, width=2
    )
    draw.line(
        (centre, steer_y, centre, steer_y + bar_h), fill=_MUTED, width=1
    )
    if abs(steer) > 0.01:
        end = centre + steer * (bar_w / 2 - 2)
        draw.rectangle(
            (min(centre, end), steer_y + 2, max(centre, end), steer_y + bar_h - 2),
            fill=_ON,
        )
    draw.text(
        (bar_x + bar_w + 12, steer_y - 2),
        f"steer {steer:>+5.2f}   gas {row['gas_f']:.2f}   brake {row['brake_f']:.2f}",
        font=small,
        fill=_MUTED,
    )

    # The four keys, as they were held on this tick.
    size = max(26, panel // 3)
    gap = 6
    right_edge = width - 14
    base_y = height + panel // 2 - size // 2
    boxes = [
        ("<", row["left"], right_edge - 4 * size - 3 * gap, _ON),
        ("^", row["up"], right_edge - 3 * size - 2 * gap, _ON),
        ("v", row["down"], right_edge - 2 * size - gap, _BRAKE),
        (">", row["right"], right_edge - size, _ON),
    ]
    for label, on, x, colour in boxes:
        _key_box(
            draw, (x, base_y, x + size, base_y + size), label, on, big, colour
        )

    return canvas


def render_run(
    run_dir: Path,
    out_path: Path,
    *,
    fps: int = 20,
    scale: int = 3,
    panel: int = 96,
    limit: int | None = None,
) -> dict:
    """Encode one recorded run into an annotated video."""
    ffmpeg = find_ffmpeg()
    rows = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"{run_dir} has no samples")

    blob = (run_dir / "frames.bin").open("rb")
    first = Image.open(_frame_at(blob, rows[0]))
    size = (first.width * scale, first.height * scale + panel)

    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{size[0]}x{size[1]}",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for row in rows:
            with Image.open(_frame_at(blob, row)) as image:
                frame = render_frame(
                    image.convert("RGB"), row, scale=scale, panel=panel
                )
            process.stdin.write(frame.tobytes())
    finally:
        blob.close()
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with {code}")

    return {
        "run": run_dir.name,
        "frames": len(rows),
        "fps": fps,
        "size": f"{size[0]}x{size[1]}",
        "seconds": round(len(rows) / fps, 2),
        "output": str(out_path),
        "bytes": out_path.stat().st_size,
    }
