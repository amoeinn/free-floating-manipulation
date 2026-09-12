"""Gaussian sets built so that specific errors can show themselves.

These are not generic fixtures. Each one exists because some check is
worthless without it, and the discriminating property is asserted rather than
assumed: a fixture that cannot express the bug a test exists to catch makes
that test pass against the bug.

Project C has hit that three times. Five hundred configurations of a robot
whose inertial frames were all axis aligned could not express the forward
kinematics bug. Isotropic Gaussians cannot express a covariance rotation bug.
An occlusion stack built in ascending depth order made removing the depth sort
a literal no operation.

They live here rather than inside a script so that the verification scripts
and the test suite share one definition. Two copies of a scene whose whole
purpose is a subtle discriminating property is exactly the thing that drifts.
"""

from __future__ import annotations

import numpy as np
import torch

from .splatting import Gaussians

DTYPE = torch.float64
PANEL_AXIS = np.array([0.0, 1.0, 0.0])


def random_quaternions(n, generator, dtype=DTYPE):
    q = torch.randn(n, 4, dtype=dtype, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)


def anisotropic_rotated(n, spread=0.5, scale_range=(0.02, 0.09), opacity=0.35,
                        centre=(0.0, 0.0, 0.0), seed=0, anisotropy=3.0,
                        dtype=DTYPE):
    """Gaussians that are elongated and rotated, never near spherical.

    The anisotropy ratio is forced rather than sampled, because a random draw
    occasionally produces a nearly spherical covariance and a check run on
    that set cannot tell a rotation error from a correct rotation.
    """
    g = torch.Generator().manual_seed(seed)
    means = torch.as_tensor(centre, dtype=dtype) + spread * (
        torch.rand(n, 3, dtype=dtype, generator=g) - 0.5)
    lo, hi = scale_range
    base = lo + (hi - lo) * torch.rand(n, 1, dtype=dtype, generator=g)
    ratios = torch.stack([torch.ones(n, dtype=dtype),
                          torch.full((n,), 1.0 / anisotropy, dtype=dtype),
                          torch.full((n,), 1.0 / (anisotropy ** 0.5), dtype=dtype)],
                         dim=1)
    return Gaussians(
        means=means,
        log_scales=torch.log(base * ratios),
        quats=random_quaternions(n, g, dtype),
        logit_opacity=torch.full((n,), float(np.log(opacity / (1 - opacity))),
                                 dtype=dtype),
        colors=torch.rand(n, 3, dtype=dtype, generator=g) * 0.8 + 0.2,
    )


def discrimination(gaussians, camera):
    """How far this set is from isotropic and from axis aligned.

    Returns `(worst axis ratio, largest off diagonal as a fraction of the
    diagonal)`. A set near 1.0 and near 0.0 cannot express a covariance
    rotation error at all.
    """
    scales = gaussians.scales
    ratio = (scales.max(dim=1).values / scales.min(dim=1).values).min().item()
    R = camera.R.to(gaussians.means.dtype)
    cov_cam = R @ gaussians.covariance() @ R.T
    diagonal = torch.diagonal(cov_cam, dim1=-2, dim2=-1)
    off = (cov_cam - torch.diag_embed(diagonal)).abs()
    off_ratio = (off.amax(dim=(1, 2)) / diagonal.abs().max(dim=1).values).max().item()
    return ratio, off_ratio


def occlusion_stack(opacity=0.25, depths=(2.4, 3.0, 3.6), extent=0.05,
                    dtype=DTYPE):
    """Differently coloured Gaussians along one line of sight, out of order.

    Built explicitly, because the point is a case where depth order decides
    the pixel colour. They are returned deliberately scrambled: the first
    version of this returned them already sorted, which made removing the
    depth sort a no operation and undetectable by construction.

    The camera is assumed on +x at 3 m, so varying world x walks them along
    the view axis while they stay on top of each other in the image.
    """
    n = len(depths)
    order = [2, 0, 1][:n]
    means = torch.zeros(n, 3, dtype=dtype)
    means[:, 0] = 3.0 - torch.as_tensor([depths[i] for i in order], dtype=dtype)
    colors = torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]],
                          dtype=dtype)[:n]
    return Gaussians(
        means=means,
        log_scales=torch.log(torch.full((n, 3), extent, dtype=dtype)
                             * torch.tensor([1.0, 0.6, 1.6], dtype=dtype)),
        quats=random_quaternions(n, torch.Generator().manual_seed(77), dtype),
        logit_opacity=torch.full((n,), float(np.log(opacity / (1 - opacity))),
                                 dtype=dtype),
        colors=colors)


def axial_slab(n=4, lateral=2.2, along=0.45, across=0.06, opacity=0.06,
               dtype=DTYPE):
    """Gaussians elongated along the view axis and far off centre.

    The perspective terms of the projection Jacobian, `-fx x / z^2`, couple
    extent along the view axis into image displacement. They therefore carry
    no weight unless a Gaussian is both long in depth and off axis, which is
    why random test sets could not detect their removal.
    """
    g = torch.Generator().manual_seed(5)
    means = torch.zeros(n, 3, dtype=dtype)
    means[:, 0] = torch.linspace(-0.4, 0.4, n, dtype=dtype)
    means[:, 1] = lateral + 0.15 * torch.linspace(-1, 1, n, dtype=dtype)
    quats = torch.zeros(n, 4, dtype=dtype)
    quats[:, 0] = 1.0                       # long axis stays on the view axis
    scales = torch.tensor([[along, across, across]], dtype=dtype).repeat(n, 1)
    return Gaussians(means, torch.log(scales), quats,
                     torch.full((n,), float(np.log(opacity / (1 - opacity))),
                                dtype=dtype),
                     torch.rand(n, 3, dtype=dtype, generator=g) * 0.8 + 0.2)


def flip_matrix():
    """180 degrees about the panel axis, the target's near degeneracy."""
    k = PANEL_AXIS
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + 2 * K @ K
