"""Drawing the path head's waypoints into the game's frames.

The sample the plugin sends with every frame carries the camera pose, so
world points project exactly. Conventions, calibrated on recorded
trajectories projected onto their own frames:

* y is up; a yaw of -pi/2 faces -x (forward = (sin yaw, 0, cos yaw)).
* The chase camera looks down at the car: forward = (sin yaw cos pitch,
  -sin pitch, cos yaw cos pitch).
* ``camera_fov`` (75 for camera 1) is the **vertical** field of view (91.5
  degrees horizontal at 4:3). Settled on a flat straight of tmx-10460245: with
  the horizontal reading, the car's actual next 3 s rose off the road into the
  stands; with the vertical one it lies on the road surface to the far end.
* Image x decreases along up x forward, i.e. the car-frame "lateral" axis of
  the labels ((cos yaw, 0, -sin yaw)) points to the car's left. Checked on
  1469 sustained single-key steering segments: 1459 agree.

Waypoints are lateral/forward in the ground plane plus, for models trained
with ``data.path_height``, a height change (world up) relative to the car's
origin. Without the height channel they are drawn at the car's height.
Nothing here is a model input; it only draws on videos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


def camera_basis(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.array([np.sin(yaw) * np.cos(pitch), -np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    r = np.cross(np.array([0.0, 1.0, 0.0]), f)
    r /= np.linalg.norm(r)
    return f, r, np.cross(f, r)


def focal_px(fov_deg: float, h: int = 240) -> float:
    """Focal length in pixels from the (vertical) camera FOV."""
    return (h / 2) / np.tan(np.radians(fov_deg) / 2)


def project(points: np.ndarray, cam_pos, cam_ypr, fov_deg: float, w: int = 320, h: int = 240):
    """World points (N, 3) -> image (N, 2) and depth (N,); NaN behind the camera."""
    f, r, u = camera_basis(cam_ypr[0], cam_ypr[1])
    d = np.asarray(points, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    z = d @ f
    fp = focal_px(fov_deg, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = np.stack([w / 2 - fp * (d @ r) / z, h / 2 - fp * (d @ u) / z], 1)
    xy[z <= 0.5] = np.nan
    return xy, z, fp


def car_frame_to_world(path_lf: np.ndarray, car_pos, car_yaw: float) -> np.ndarray:
    """(K, 2) lateral (left +), forward metres -> world (K, 3) at the car's height.
    A third column, if present, is the height change (m, world up)."""
    fwd = np.array([np.sin(car_yaw), 0.0, np.cos(car_yaw)])
    lat = np.array([np.cos(car_yaw), 0.0, -np.sin(car_yaw)])
    world = np.asarray(car_pos)[None] + path_lf[:, :1] * lat[None] + path_lf[:, 1:2] * fwd[None]
    if path_lf.shape[1] > 2:
        world[:, 1] += path_lf[:, 2]
    return world


# near -> far along the prediction: cyan to amber, readable on tarmac and grass
_NEAR, _FAR = np.array([80, 220, 255]), np.array([255, 185, 40])


def time_color(a: float) -> tuple[int, int, int]:
    """Color for a point a fraction ``a`` of the way to the last horizon."""
    a = float(np.clip(a, 0.0, 1.0))
    return tuple(int(v) for v in (1 - a) * _NEAR + a * _FAR)


def _catmull_rom(ctrl: np.ndarray, per_segment: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Smooth curve through the (K, D) control points: (M, D) points and their
    fractional control index (0 = first control point)."""
    p = np.concatenate([2 * ctrl[:1] - ctrl[1:2], ctrl, 2 * ctrl[-1:] - ctrl[-2:-1]])
    out, idx = [ctrl[:1]], [np.zeros(1)]
    t = np.linspace(0, 1, per_segment + 1)[1:, None]
    for i in range(len(ctrl) - 1):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)
        out.append(seg)
        idx.append(i + t[:, 0])
    return np.concatenate(out), np.concatenate(idx)


def speed_color(v: float, v_now: float) -> tuple[int, int, int]:
    """Planned speed against the current one: green faster, white the same, red
    slower (full color at a 40 km/h change)."""
    d = float(np.clip((v - v_now) / 40.0, -1.0, 1.0))
    end = _FASTER if d >= 0 else _SLOWER
    return tuple(int(x) for x in (1 - abs(d)) * _SAME + abs(d) * end)


_SAME, _FASTER, _SLOWER = np.array([240, 240, 240]), np.array([80, 230, 120]), np.array([250, 70, 70])


@dataclass
class PathCurve:
    """A predicted path as a smooth curve in the car's frame, sampled densely.

    ``points`` (M, 3) are lateral (left +), forward, height (m); ``t`` the time
    ahead of each sample, ``sigma`` its lateral std, ``speed`` its planned speed
    (km/h, from the current speed at the car), ``normal`` the ground-plane
    normal to the curve.
    """

    points: np.ndarray
    t: np.ndarray
    sigma: np.ndarray
    speed: np.ndarray | None
    normal: np.ndarray

    def at(self, seconds: float) -> int:
        return int(np.argmin(np.abs(self.t - seconds)))


def path_curve(mean_lf: np.ndarray, std_lat: np.ndarray | None, horizons: list[float] | None,
               height: np.ndarray | None = None, speed: np.ndarray | None = None,
               speed_now: float | None = None, per_segment: int = 12) -> PathCurve:
    mean_lf = np.asarray(mean_lf, dtype=np.float64)
    k = len(mean_lf)
    h = np.zeros(k) if height is None else np.asarray(height, dtype=np.float64).reshape(k)
    wp = np.concatenate([mean_lf, h[:, None]], 1)
    curve, idx = _catmull_rom(np.concatenate([np.zeros((1, 3)), wp]), per_segment)
    knots = np.arange(k + 1)
    times = np.concatenate([[0.0], horizons if horizons is not None else np.arange(1, k + 1)])
    sig = np.zeros(k) if std_lat is None else np.asarray(std_lat, dtype=np.float64)
    spd = None
    if speed is not None:
        spd = np.interp(idx, knots, np.concatenate([[speed_now if speed_now is not None else speed[0]], speed]))
    tan = np.gradient(curve[:, :2], axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    normal = np.zeros_like(curve)
    normal[:, 0], normal[:, 1] = tan[:, 1], -tan[:, 0]
    return PathCurve(curve, np.interp(idx, knots, times), np.interp(idx, knots, np.concatenate([[0.0], sig])),
                     spd, normal)


TICKS_S = (1.0, 2.0, 3.0)  # white lines across the ribbon at these times ahead
CAR_HALF_WIDTH_M = 0.9
# The car's body in its own frame as boxes of (lateral, forward, up) ranges,
# metres about its origin: low and full length, plus the tall rear (cockpit
# and wing). Padded so they still cover the car when it rolls or pitches
# (only yaw is logged). Fitted by eye on chase-camera frames of B01 and
# tmx-10460245.
CAR_BOXES_M = (((-1.2, 1.2), (-2.3, 2.3), (-0.4, 0.6)),
               ((-1.2, 1.2), (-2.3, 0.2), (-0.4, 1.3)))


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points (monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return np.asarray(pts)

    def half(seq):
        out: list[tuple[float, float]] = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    return np.asarray(half(pts) + half(reversed(pts)))


def car_keepout(pose: dict, frame_w: int = 320, frame_h: int = 240) -> list[np.ndarray]:
    """The car's outline in the image: one convex polygon (frame pixels) per box
    of ``CAR_BOXES_M`` projected through the camera, skipping any not in view."""
    polys = []
    for (x0, x1), (z0, z1), (y0, y1) in CAR_BOXES_M:
        corners = np.array([[x, z, y] for x in (x0, x1) for z in (z0, z1) for y in (y0, y1)])
        world = car_frame_to_world(corners, pose["position"], pose["yaw_pitch_roll"][0])
        xy, _, _ = project(world, pose["camera_position"], pose["camera_yaw_pitch_roll"], pose["camera_fov"],
                           w=frame_w, h=frame_h)
        if np.isfinite(xy).all():
            polys.append(_hull(xy))
    return polys


def draw_path(img: Image.Image, pose: dict, mean_lf: np.ndarray, std_lat: np.ndarray | None = None,
              horizons: list[float] | None = None, scale: float = 1.0, band_alpha: int = 28,
              frame_w: int = 320, frame_h: int = 240, height: np.ndarray | None = None,
              speed: np.ndarray | None = None, speed_now: float | None = None,
              keep_out_car: bool = True, fill_alpha: int = 110) -> Image.Image:
    """Draw a predicted path (K, 2 lateral/forward metres) onto ``img`` as a
    car-wide ribbon lying on the road.

    Translucent fill with solid edges, white cross lines at ``TICKS_S``, and a
    faint halo widened by +-1 sigma of the predicted lateral position.
    Colored near -> far; with ``speed`` (K,) and ``speed_now`` (km/h), by the
    planned speed change instead (green faster, white the same, red slower).
    ``height`` (K,) is
    each waypoint's predicted height change (m, world up); without it the
    ribbon lies at the car's height. With ``keep_out_car`` nothing is drawn
    over the car itself (see ``car_keepout``), so the ribbon runs under it. ``frame_w/h`` is the game
    frame's size before ``scale`` (the image may carry an info strip below it).
    Returns the composited image.
    """
    c = path_curve(mean_lf, std_lat, horizons, height, speed, speed_now)

    def to_img(pts: np.ndarray) -> np.ndarray:
        world = car_frame_to_world(pts, pose["position"], pose["yaw_pitch_roll"][0])
        xy, _, _ = project(world, pose["camera_position"], pose["camera_yaw_pitch_roll"], pose["camera_fov"],
                           w=frame_w, h=frame_h)
        return xy * scale

    def ok(*pts) -> bool:
        return all(np.isfinite(q).all() for q in pts)

    def strip(left: np.ndarray, right: np.ndarray, fill) -> None:
        for i in range(len(left) - 1):
            quad = np.stack([left[i], left[i + 1], right[i + 1], right[i]])
            if ok(quad):
                d.polygon([tuple(q) for q in quad], fill=fill(i))

    if c.speed is not None and speed_now is not None:
        cols = [speed_color(v, speed_now) for v in c.speed]
    else:
        cols = [time_color(t / max(c.t[-1], 1e-6)) for t in c.t]
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if std_lat is not None:  # Offsets along the curve's normal in the ground plane; height unchanged.
        halo = c.normal * (CAR_HALF_WIDTH_M + c.sigma)[:, None]
        strip(to_img(c.points + halo), to_img(c.points - halo), lambda i: (255, 255, 255, band_alpha))
    edge = c.normal * CAR_HALF_WIDTH_M
    left, right = to_img(c.points + edge), to_img(c.points - edge)
    strip(left, right, lambda i: (*cols[i], fill_alpha))
    w = max(1, int(round(scale)))
    for side in (left, right):
        for i in range(len(side) - 1):
            if ok(side[i], side[i + 1]):
                d.line([tuple(side[i]), tuple(side[i + 1])], fill=(*cols[i], 255), width=w)
    for tick in TICKS_S:
        i = c.at(tick)
        if abs(c.t[i] - tick) < 0.1 and ok(left[i], right[i]):
            d.line([tuple(left[i]), tuple(right[i])], fill=(255, 255, 255, 230), width=w)
    if keep_out_car:
        for poly in car_keepout(pose, frame_w, frame_h):
            d.polygon([tuple(q) for q in poly * scale], fill=(0, 0, 0, 0))
    return Image.alpha_composite(base, layer).convert("RGB")


def draw_trail(draw: ImageDraw.ImageDraw, pose: dict, future_world: np.ndarray, scale: float = 1.0,
               color: tuple[int, int, int] = (255, 255, 255), w: int = 320, h: int = 240) -> None:
    """Draw where the car actually went (world points) from this sample's camera."""
    if not len(future_world):
        return
    xy, _, _ = project(future_world, pose["camera_position"], pose["camera_yaw_pitch_roll"], pose["camera_fov"],
                       w=w, h=h)
    pts = [tuple(p * scale) for p in xy if np.isfinite(p).all()]
    if len(pts) > 1:
        draw.line(pts, fill=color, width=max(1, int(scale)))
