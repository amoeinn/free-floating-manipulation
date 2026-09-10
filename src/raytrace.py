"""An exact ray tracer over analytic primitives, used to make ground truth.

This exists so that phase 4 has images whose camera poses are known rather
than estimated, and whose geometry is known in closed form rather than
sampled. Everything it returns is exact to floating point: the depth of a
surface, its normal, and whether a pixel saw anything at all. That matters
because the splatting forward pass will be checked against references that
have to share this camera model exactly, and a camera convention that is only
written down in one place is a convention nobody has checked.

Frames, stated once and not repeated.

- World frame is right handed, metres.
- Camera frame follows the OpenCV convention: `+x` right across the image,
  `+y` down, `+z` forward along the optical axis. A point in front of the
  camera has positive z.
- A pose is `(R, t)` mapping world to camera: `x_cam = R @ x_world + t`. The
  camera centre in world coordinates is therefore `C = -R.T @ t`.
- Pixel `(u, v)` is a column and a row, with `(0, 0)` at the centre of the top
  left pixel. A pixel's ray direction in camera coordinates is
  `normalize([(u - cx) / fx, (v - cy) / fy, 1])`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit vectors along `axis`, leaving exact zeros alone."""
    length = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(length > 0, v / np.where(length > 0, length, 1.0), v)


def rotation(rpy: Sequence[float]) -> np.ndarray:
    """Rotation matrix from roll pitch yaw, composed Rz Ry Rx.

    Same convention as `kinematics.rpy_to_matrix`, so a pose written here
    means what it means elsewhere in the repository.
    """
    r, p, y = (float(a) for a in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World to camera `(R, t)` for a camera at `eye` looking at `target`.

    Returns the OpenCV convention above: the camera's `+z` row points from the
    eye toward the target, `+y` points down in the image.
    """
    eye = np.asarray(eye, float)
    forward = normalize(np.asarray(target, float) - eye)
    axis = np.cross(forward, np.asarray(up, float))
    # A view direction parallel to `up` leaves no way to orient the image, and
    # the arithmetic does not complain: `right` comes out as the zero vector,
    # the rotation matrix is singular, every ray is degenerate and the render
    # is uniformly black. That looks like an empty scene rather than a broken
    # camera, so it is refused here instead.
    if np.linalg.norm(axis) < 1e-9:
        raise ValueError(
            f"look_at: the up vector {np.asarray(up, float)} is parallel to the "
            f"view direction {np.round(forward, 6)}, so the camera roll is "
            "undefined. Pick an up vector that is not along the line of sight.")
    right = normalize(axis)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])          # rows are the camera axes in world
    return R, -R @ eye


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray                                  # world to camera
    t: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    @classmethod
    def with_fov(cls, width: int, height: int, fov_x_deg: float,
                 R: np.ndarray, t: np.ndarray) -> "Camera":
        fx = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_x_deg))
        return cls(width, height, fx, fx, (width - 1) / 2.0, (height - 1) / 2.0, R, t)

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points to pixel coordinates and camera-frame depth.

        Returns `(uv, z)`. Points at or behind the camera come back with their
        depth so a caller can reject them; no clipping happens here.
        """
        cam = points_world @ self.R.T + self.t
        z = cam[..., 2]
        safe = np.where(np.abs(z) > 1e-12, z, 1e-12)
        u = self.fx * cam[..., 0] / safe + self.cx
        v = self.fy * cam[..., 1] / safe + self.cy
        return np.stack([u, v], axis=-1), z

    def rays(self) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit direction in world coordinates, one per pixel.

        Shapes are `(3,)` and `(height, width, 3)`.
        """
        v, u = np.meshgrid(np.arange(self.height, dtype=float),
                           np.arange(self.width, dtype=float), indexing="ij")
        d_cam = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy,
                          np.ones_like(u)], axis=-1)
        return self.centre, normalize(d_cam @ self.R)   # R.T applied on the right


@dataclass
class Primitive:
    """A convex solid with a pose and an albedo.

    Subclasses implement `_hit_local`, which works in the primitive's own
    frame; the pose handling is shared so that a new shape cannot get it
    wrong in a new way.
    """
    position: np.ndarray
    rpy: Sequence[float]
    albedo: np.ndarray
    name: str = "primitive"
    _R: np.ndarray = field(init=False)

    def __post_init__(self):
        self.position = np.asarray(self.position, float)
        self.albedo = np.asarray(self.albedo, float)
        self._R = rotation(self.rpy)               # local to world

    def hit(self, origin: np.ndarray, direction: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest positive intersection distance and world normal per ray.

        Misses come back as `inf` distance and a zero normal.
        """
        o = (origin - self.position) @ self._R     # world to local
        d = direction @ self._R
        distance, normal_local = self._hit_local(o, d)
        return distance, normal_local @ self._R.T

    def _hit_local(self, o, d):
        raise NotImplementedError


@dataclass
class Box(Primitive):
    half_extents: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        self.half_extents = np.asarray(self.half_extents, float)

    def _hit_local(self, o, d):
        """Slab method. `o` broadcasts against `d`, which is `(..., 3)`."""
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            lo = (-self.half_extents - o) * inv
            hi = (self.half_extents - o) * inv
        # A direction component of exactly zero gives inf or nan here. The ray
        # is parallel to that slab, so it constrains nothing: push its near
        # bound to -inf and its far bound to +inf rather than letting a nan
        # propagate into the reduction.
        near = np.nan_to_num(np.minimum(lo, hi), nan=-np.inf)
        far = np.nan_to_num(np.maximum(lo, hi), nan=+np.inf)
        t_near = near.max(axis=-1)
        t_far = far.min(axis=-1)

        entering = t_near > 1e-9
        t = np.where(entering, t_near, t_far)
        missed = (t_near > t_far) | (t <= 1e-9) | ~np.isfinite(t)

        # The normal comes from whichever slab supplied the bound that was
        # taken: the largest near bound on the way in, the smallest far bound
        # on the way out.
        axis = np.where(entering, near.argmax(axis=-1), far.argmin(axis=-1))
        n = np.zeros(np.broadcast_shapes(o.shape, d.shape))
        np.put_along_axis(n, axis[..., None], 1.0, axis=-1)
        point = o + np.where(missed, 0.0, t)[..., None] * d
        n = n * np.sign(np.take_along_axis(point, axis[..., None], axis=-1))
        return np.where(missed, np.inf, t), np.where(missed[..., None], 0.0, n)


@dataclass
class Cylinder(Primitive):
    """A capped cylinder along its own local z."""
    radius: float = 1.0
    half_height: float = 1.0

    def _hit_local(self, o, d):
        # A ray parallel to the axis divides by zero on the caps and a miss
        # multiplies inf by zero on the side; both are handled below by the
        # validity masks, so the warnings are noise.
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._hit_local_unguarded(o, d)

    def _hit_local_unguarded(self, o, d):
        ox, oy = o[..., 0], o[..., 1]
        dx, dy = d[..., 0], d[..., 1]
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - self.radius ** 2
        disc = b * b - 4 * a * c
        best = np.full(d.shape[:-1], np.inf)
        normal = np.zeros(d.shape)

        ok = (disc > 0) & (a > 1e-16)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        for root in (-1.0, 1.0):
            t = np.where(ok, (-b + root * sq) / (2 * np.where(a > 1e-16, a, 1.0)), np.inf)
            z = o[..., 2] + t * d[..., 2]
            valid = ok & (t > 1e-9) & (np.abs(z) <= self.half_height) & (t < best)
            n = np.stack([o[..., 0] + t * dx, o[..., 1] + t * dy,
                          np.zeros_like(t)], axis=-1) / self.radius
            best = np.where(valid, t, best)
            normal = np.where(valid[..., None], n, normal)

        for sign in (-1.0, 1.0):
            t = np.where(np.abs(d[..., 2]) > 1e-12,
                         (sign * self.half_height - o[..., 2]) / d[..., 2], np.inf)
            p = o + t[..., None] * d
            inside = p[..., 0] ** 2 + p[..., 1] ** 2 <= self.radius ** 2
            valid = inside & (t > 1e-9) & (t < best)
            n = np.zeros(d.shape)
            n[..., 2] = sign
            best = np.where(valid, t, best)
            normal = np.where(valid[..., None], n, normal)

        return best, normal


@dataclass
class Scene:
    primitives: list
    light_direction: np.ndarray                    # world, points FROM the light
    light_intensity: float = 1.0

    def __post_init__(self):
        self.light_direction = normalize(np.asarray(self.light_direction, float))

    def intersect(self, origin, direction):
        """Nearest hit over all primitives: distance, normal, albedo, index."""
        shape = direction.shape[:-1]
        best = np.full(shape, np.inf)
        normal = np.zeros(direction.shape)
        albedo = np.zeros(direction.shape)
        index = np.full(shape, -1, dtype=int)
        for i, prim in enumerate(self.primitives):
            t, n = prim.hit(origin, direction)
            closer = t < best
            best = np.where(closer, t, best)
            normal = np.where(closer[..., None], n, normal)
            albedo = np.where(closer[..., None], prim.albedo, albedo)
            index = np.where(closer, i, index)
        return best, normal, albedo, index

    def render(self, camera: Camera, shadows: bool = True) -> dict:
        """Lambertian shading under one hard directional light, no ambient.

        Returns the image, the depth along the optical axis, a hit mask, the
        surface normals and which primitive each pixel saw. Unlit surface is
        black and so is the background, which is the point: an orbital scene
        has no fill light, and the reconstruction problem is harder for it.
        """
        origin, direction = camera.rays()
        distance, normal, albedo, index = self.intersect(origin, direction)
        hit = np.isfinite(distance)

        to_light = -self.light_direction
        lambert = np.clip(np.einsum("...i,i->...", normal, to_light), 0.0, None)

        if shadows:
            point = origin + np.where(hit, distance, 0.0)[..., None] * direction
            shadow_origin = point + 1e-4 * normal
            lit = np.zeros(lambert.shape, dtype=bool)
            flat_o = shadow_origin.reshape(-1, 3)
            occluder = np.full(flat_o.shape[0], np.inf)
            for prim in self.primitives:
                t, _ = prim.hit(flat_o, np.broadcast_to(to_light, flat_o.shape).copy())
                occluder = np.minimum(occluder, t)
            lit = ~np.isfinite(occluder).reshape(lambert.shape)
            lambert = lambert * lit

        image = albedo * (self.light_intensity * lambert)[..., None]
        # Depth along the optical axis, which is what a projection compares to,
        # not the distance along the ray.
        forward = camera.R[2]
        axis_depth = distance * np.einsum("...i,i->...", direction, forward)
        return {
            "image": np.where(hit[..., None], np.clip(image, 0.0, 1.0), 0.0),
            "depth": np.where(hit, axis_depth, 0.0),
            "hit": hit,
            "normal": normal,
            "primitive": index,
        }


def approach_trajectory(n_views: int, start_range: float, end_range: float,
                        sweep_deg: float, elevation_deg: float,
                        target: np.ndarray = None, fov_deg: float = 40.0,
                        width: int = 128, height: int = 128) -> list:
    """Cameras closing on a target while sweeping around it.

    A servicer's approach is not an orbit at fixed radius and it is not a
    straight line either: range closes while the relative bearing walks
    around, which is what gives a reconstruction its baseline. Elevation
    rises over the same arc so the views are not coplanar, because coplanar
    views leave the depth of a symmetric object unconstrained.
    """
    target = np.zeros(3) if target is None else np.asarray(target, float)
    cameras = []
    for k in range(n_views):
        s = k / max(n_views - 1, 1)
        radius = start_range + s * (end_range - start_range)
        azimuth = np.deg2rad(sweep_deg) * s
        elevation = np.deg2rad(elevation_deg) * np.sin(np.pi * s)
        eye = target + radius * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ])
        R, t = look_at(eye, target, np.array([0.0, 0.0, 1.0]))
        cameras.append(Camera.with_fov(width, height, fov_deg, R, t))
    return cameras
