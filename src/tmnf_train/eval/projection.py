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

Waypoints are 2-D, so they are drawn at the height of the car's origin.
Nothing here is a model input; it only draws on videos.
"""

from __future__ import annotations

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
    """(K, 2) lateral (left +), forward metres -> world (K, 3) at the car's height."""
    fwd = np.array([np.sin(car_yaw), 0.0, np.cos(car_yaw)])
    lat = np.array([np.cos(car_yaw), 0.0, -np.sin(car_yaw)])
    return np.asarray(car_pos)[None] + path_lf[:, :1] * lat[None] + path_lf[:, 1:2] * fwd[None]


# near -> far along the prediction: cyan to amber, readable on tarmac and grass
_NEAR, _FAR = np.array([80, 220, 255]), np.array([255, 185, 40])


def time_colour(a: float) -> tuple[int, int, int]:
    """Colour for a point a fraction ``a`` of the way to the last horizon."""
    a = float(np.clip(a, 0.0, 1.0))
    return tuple(int(v) for v in (1 - a) * _NEAR + a * _FAR)


def _catmull_rom(ctrl: np.ndarray, per_segment: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Smooth curve through the control points: (M, 2) points and their
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


def draw_path(img: Image.Image, pose: dict, mean_lf: np.ndarray, std_lat: np.ndarray | None = None,
              horizons: list[float] | None = None, scale: float = 1.0, band_alpha: int = 60,
              frame_w: int = 320, frame_h: int = 240) -> Image.Image:
    """Draw a predicted path (K, 2 lateral/forward metres) onto ``img``.

    A smooth curve from the car through the waypoints, coloured near -> far,
    small ticks at each waypoint, and a translucent band of +-1 sigma of the
    predicted lateral position around it. ``frame_w/h`` is the game frame's
    size before ``scale`` (the image may carry an info strip below it).
    Returns the composited image.
    """
    k = len(mean_lf)
    ctrl = np.concatenate([np.zeros((1, 2)), np.asarray(mean_lf, dtype=np.float64)])
    curve, idx = _catmull_rom(ctrl)
    times = np.concatenate([[0.0], horizons if horizons is not None else np.arange(1, k + 1)])
    t_of = np.interp(idx, np.arange(k + 1), times)
    frac = t_of / max(times[-1], 1e-6)
    sig = np.interp(idx, np.arange(k + 1), np.concatenate([[0.0], std_lat])) if std_lat is not None else None

    def to_img(lf: np.ndarray) -> np.ndarray:
        world = car_frame_to_world(lf, pose["position"], pose["yaw_pitch_roll"][0])
        xy, _, _ = project(world, pose["camera_position"], pose["camera_yaw_pitch_roll"], pose["camera_fov"],
                           w=frame_w, h=frame_h)
        return xy * scale

    centre = to_img(curve)
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if sig is not None:
        # Offset along the curve's normal in the ground plane.
        tan = np.gradient(curve, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        normal = np.stack([tan[:, 1], -tan[:, 0]], 1)
        left, right = to_img(curve + normal * sig[:, None]), to_img(curve - normal * sig[:, None])
        for i in range(len(curve) - 1):
            quad = np.stack([left[i], left[i + 1], right[i + 1], right[i]])
            if np.isfinite(quad).all():
                d.polygon([tuple(q) for q in quad], fill=(*time_colour(frac[i]), band_alpha))
    w = max(2, int(round(2 * scale)))
    for i in range(len(centre) - 1):
        a, b = centre[i], centre[i + 1]
        if np.isfinite(a).all() and np.isfinite(b).all():
            d.line([tuple(a), tuple(b)], fill=(*time_colour(frac[i]), 255), width=w)
    wp = to_img(np.asarray(mean_lf, dtype=np.float64))
    rr = 2.2 * scale
    for j, (x, y) in enumerate(wp):
        if np.isfinite([x, y]).all():
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*time_colour((j + 1) / k), 255),
                      outline=(0, 0, 0, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def draw_trail(draw: ImageDraw.ImageDraw, pose: dict, future_world: np.ndarray, scale: float = 1.0,
               colour: tuple[int, int, int] = (255, 255, 255), w: int = 320, h: int = 240) -> None:
    """Draw where the car actually went (world points) from this sample's camera."""
    if not len(future_world):
        return
    xy, _, _ = project(future_world, pose["camera_position"], pose["camera_yaw_pitch_roll"], pose["camera_fov"],
                       w=w, h=h)
    pts = [tuple(p * scale) for p in xy if np.isfinite(p).all()]
    if len(pts) > 1:
        draw.line(pts, fill=colour, width=max(1, int(scale)))
