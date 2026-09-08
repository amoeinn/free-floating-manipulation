"""Estimating the target's pose against the model fitted in step 3.

The pose is carried as a body to world rigid transform applied to the model
rather than to the camera, because the camera model is shared with the
reference implementations and is the last thing that should acquire a second
convention. Rotation is an axis angle vector, so the parameterisation is
minimal and has no constraint for an optimiser to violate.

Rotating a Gaussian set is exact rather than approximate: a world rotation `R`
turns the covariance into `(R R_q) S S^T (R R_q)^T`, which is the same set with
the rotation composed onto each quaternion. Nothing is resampled and no
information is lost, which matters because the round trip check below requires
that rendering at a pose and recovering it be limited by the estimator alone.
"""

from __future__ import annotations

import numpy as np
import torch

from .splatting import Gaussians, render_tiled


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues, with the small angle branch written so autograd survives it.

    `norm()` is not differentiable at zero, and an optimiser started at the
    identity sits exactly there, so the angle is floored before dividing.
    """
    theta = v.norm().clamp_min(1e-12)
    k = v / theta
    K = torch.stack([
        torch.stack([torch.zeros_like(k[0]), -k[2], k[1]]),
        torch.stack([k[2], torch.zeros_like(k[0]), -k[0]]),
        torch.stack([-k[1], k[0], torch.zeros_like(k[0])]),
    ])
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def matrix_to_axis_angle(R) -> np.ndarray:
    R = np.asarray(R, float)
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2 * np.sin(angle)) * angle


def quaternion_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """`(w, x, y, z)` from a rotation matrix, by the largest diagonal branch."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = torch.sqrt(t + 1.0) * 2
        return torch.stack([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                            (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return torch.stack([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                            (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return torch.stack([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                            0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return torch.stack([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                        (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a` composed onto every row of `b`, both `(w, x, y, z)`."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def transform(model: Gaussians, rotvec: torch.Tensor, translation: torch.Tensor,
              centre: torch.Tensor = None) -> Gaussians:
    """The model rigidly moved. Rotation is about `centre`, the model centroid
    by default, so that a rotation parameter does not also translate."""
    R = axis_angle_to_matrix(rotvec)
    c = model.means.mean(dim=0) if centre is None else centre
    q = quaternion_from_matrix(R)
    return Gaussians(
        means=(model.means - c) @ R.T + c + translation,
        log_scales=model.log_scales,
        quats=quaternion_multiply(q, model.quats),
        logit_opacity=model.logit_opacity,
        colors=model.colors,
    )


def photometric_loss(model, rotvec, translation, camera, image, centre=None):
    moved = transform(model, rotvec, translation, centre)
    return (render_tiled(moved, camera) - image).abs().mean()


def silhouette_loss(model, rotvec, translation, camera, target_mask, centre=None):
    """The independent comparison: shape only, shading discarded.

    Compares accumulated coverage against an observed mask. Coverage is
    opacity, not brightness: the first version of this thresholded the
    rendered image, which called every face turned away from the light empty
    and left the loss almost flat under rotation. That is why the renderer
    now returns coverage separately.
    """
    moved = transform(model, rotvec, translation, centre)
    _, coverage = render_tiled(moved, camera, return_coverage=True)
    return (coverage - target_mask).abs().mean()


def register(model, camera, image, init_rotvec, init_translation, loss_fn,
             iterations=120, lr_rot=0.02, lr_trans=0.01, centre=None):
    """Optimise 6 DoF against one frame. Returns the pose and a trace."""
    rot = torch.as_tensor(init_rotvec, dtype=model.means.dtype).clone().requires_grad_(True)
    tr = torch.as_tensor(init_translation, dtype=model.means.dtype).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [rot], "lr": lr_rot},
                            {"params": [tr], "lr": lr_trans}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations)
    trace = []
    for _ in range(iterations):
        opt.zero_grad()
        loss = loss_fn(model, rot, tr, camera, image, centre)
        loss.backward()
        opt.step()
        sched.step()
        trace.append(loss.item())
    return rot.detach(), tr.detach(), trace


def pose_error(rot_a, trans_a, rot_b, trans_b):
    """Geodesic rotation error in degrees and translation error in mm."""
    Ra = axis_angle_to_matrix(torch.as_tensor(rot_a, dtype=torch.float64)).numpy()
    Rb = axis_angle_to_matrix(torch.as_tensor(rot_b, dtype=torch.float64)).numpy()
    cos = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos)), 1000 * float(np.linalg.norm(
        np.asarray(trans_a, float) - np.asarray(trans_b, float)))
