"""Fitting Gaussians to approach imagery, with the target's geometry withheld.

The premise only holds if nothing here reads the client's shape. The
initialisation is a visual hull carved from the silhouettes and the known
camera poses, which is information the servicer has; `target.py` is never
imported. The one thing taken from outside the images is a generous bounding
box, and that follows from the approach range and the field of view rather
than from the object.

There is no adaptive densification or pruning. A fixed set of Gaussians is
placed by the hull and then optimised. That is a real limitation and is
reported as one rather than worked around.
"""

from __future__ import annotations

import numpy as np
import torch

from .splatting import Gaussians


def carve_visual_hull(images, poses, intrinsics, resolution=64, extent=2.6,
                      threshold=1e-3):
    """Voxels no silhouette rules out, plus a colour for each.

    A voxel survives only if every view that can see it puts it inside the
    silhouette. That is the standard visual hull: it cannot recover concavity
    and it will bridge the gap between the panels and the bus, which is
    exactly the sort of error the fit then has to remove.
    """
    # Callers hold the views as a torch tensor for the fit and as numpy here.
    # `torch.max(axis=-1)` returns a pair rather than an array, so coerce once
    # at the boundary instead of failing several lines later.
    images = np.asarray(images)
    poses = np.asarray(poses)
    fx, fy, cx, cy, width, height = intrinsics
    axis = np.linspace(-extent, extent, resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)

    occupied = np.ones(points.shape[0], bool)
    colour_sum = np.zeros((points.shape[0], 3))
    colour_count = np.zeros(points.shape[0])

    for image, pose in zip(images, poses):
        R, t = pose[:9].reshape(3, 3), pose[9:]
        cam = points @ R.T + t
        z = cam[:, 2]
        u = (fx * cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + cx).round().astype(int)
        v = (fy * cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + cy).round().astype(int)
        inside = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        lum = np.zeros(points.shape[0])
        lum[inside] = image[v[inside], u[inside]].max(axis=-1)
        # Outside the frame a voxel is unconstrained by this view, not carved.
        occupied &= (~inside) | (lum > threshold)
        lit = inside & (lum > threshold)
        colour_sum[lit] += image[v[lit], u[lit]]
        colour_count[lit] += 1

    keep = occupied & (colour_count > 0)
    colours = colour_sum[keep] / colour_count[keep][:, None]
    return points[keep], colours, 2 * extent / (resolution - 1)


def initialise(images, poses, intrinsics, n_gaussians, resolution=64,
               extent=2.6, seed=0, dtype=torch.float32):
    """Gaussians placed on the visual hull, coloured by what the views saw."""
    points, colours, voxel = carve_visual_hull(images, poses, intrinsics,
                                               resolution, extent)
    if points.shape[0] == 0:
        raise SystemExit("the visual hull carved away every voxel, so the "
                         "silhouette threshold or the bounding box is wrong")
    rng = np.random.default_rng(seed)
    pick = rng.choice(points.shape[0], size=min(n_gaussians, points.shape[0]),
                      replace=points.shape[0] < n_gaussians)
    means = points[pick] + rng.normal(0.0, voxel * 0.25, size=(pick.size, 3))

    quats = rng.normal(size=(pick.size, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    scale = voxel * 0.6
    return Gaussians(
        means=torch.as_tensor(means, dtype=dtype),
        log_scales=torch.full((pick.size, 3), float(np.log(scale)), dtype=dtype),
        quats=torch.as_tensor(quats, dtype=dtype),
        logit_opacity=torch.full((pick.size,), float(np.log(0.1 / 0.9)), dtype=dtype),
        colors=torch.as_tensor(np.clip(colours[pick], 1e-3, 1.0), dtype=dtype),
    ).requires_grad_(True), points.shape[0]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def learning_rates(gaussians, scene_scale=1.0):
    """Per group rates. Means move in metres, colours in [0, 1], so one rate
    for all of them would either freeze the geometry or destroy the colour."""
    return [
        {"params": [gaussians.means], "lr": 1.6e-3 * scene_scale},
        {"params": [gaussians.log_scales], "lr": 8e-3},
        {"params": [gaussians.quats], "lr": 2e-3},
        {"params": [gaussians.logit_opacity], "lr": 5e-2},
        {"params": [gaussians.colors], "lr": 1e-2},
    ]
