"""Acquisition, the flip test and a tracking step, as library functions.

These procedures were written inside `examples/acquisition.py` and
`examples/handover.py`, which was fine while only those scripts ran them. The
mission executive needs the same procedures, and a second copy of an algorithm
whose whole point is a subtle failure mode is the last thing this project
should carry. So they moved here and the scripts call them.

That makes this a refactor and it is held to the numbers already committed:
22 percent of single attempts land in the true basin, 87 percent at eight
restarts, and the flip test reads 3.84x at a correct pose and 0.26x at a
flipped one where the model was fitted.

Nothing here decides policy. The window the flip test is valid in, the restart
budget, and what to do about a repair are the executive's business; these just
compute.
"""

from __future__ import annotations

import numpy as np
import torch

from .splatting import Gaussians
from .tracking import (axis_angle_to_matrix, matrix_to_axis_angle,
                       photometric_loss, register, silhouette_loss)

PANEL_AXIS = np.array([0.0, 1.0, 0.0])
ACQUIRE_ITERATIONS = 120
ACQUIRE_LR_ROT = 0.08
ACQUIRE_LR_TRANS = 0.02
TRACK_ITERATIONS = 25
TRACK_LR_ROT = 0.02
TRACK_LR_TRANS = 0.010


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def flip_matrix():
    """180 degrees about the panel axis, the near degeneracy of this target.

    It maps the bus onto itself and swaps the two identical panels. Only the
    dish and the boom break it, 4.4 percent of the object's pixels, which is
    why geometry alone barely rejects it and the baked lighting does.
    """
    K = skew(PANEL_AXIS)
    return np.eye(3) + 2 * K @ K


def uniform_so3(n, rng):
    """Uniform rotations, by normalised Gaussian quaternions."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def register_from(model, camera, target, R0, loss_fn, iterations, lr_rot, lr_trans,
                  centre=None):
    """One registration from one starting attitude. Returns (R, final loss)."""
    origin = torch.zeros(3, dtype=model.means.dtype) if centre is None else centre
    rot, _, trace = register(model, camera, target, matrix_to_axis_angle(R0),
                             np.zeros(3), loss_fn, iterations=iterations,
                             lr_rot=lr_rot, lr_trans=lr_trans, centre=origin)
    R = axis_angle_to_matrix(
        torch.as_tensor(rot.numpy(), dtype=torch.float64)).numpy()
    return R, trace[-1]


def acquire_pose(model, camera, image, restarts=8, iterations=ACQUIRE_ITERATIONS,
                 seed=31, rng=None, on_restart=None):
    """Best of `restarts` photometric registrations from uniform SO(3).

    The lowest final loss wins, and that is only sound for the photometric
    loss: its true basin holds the lowest value anywhere, so minimising over
    restarts converges on the truth. The silhouette loss does not have that
    property, its global minimum sits in a wrong basin, and best of k
    therefore gets monotonically worse with more restarts. This function is
    photometric for that reason and takes no loss argument.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    best_loss, best_R, used = np.inf, None, 0
    for R0 in uniform_so3(int(restarts), rng):
        R, loss = register_from(model, camera, image, R0, photometric_loss,
                                iterations, ACQUIRE_LR_ROT, ACQUIRE_LR_TRANS)
        used += 1
        if loss < best_loss:
            best_loss, best_R = loss, R
        if on_restart is not None:
            on_restart(used, float(best_loss))
    return best_R, float(best_loss), used


def flip_test(model, camera, image, R):
    """Is the pose held, or its 180 degree flip, the better explanation?

    One extra render. Returns `(repair, ratio)` where ratio is the loss at the
    flipped pose over the loss at the pose held: above one keeps the pose,
    below one says the flip explains the image better.

    Only meaningful near the body frame sun angle the model was fitted at.
    Measured: 3.84x at a correct pose and 0.26x at a flipped one at zero
    degrees, 1.69x and 0.59x by 15, exactly 1.00x at 28, and inverted past 40,
    where it would repair a correct pose into the flip. Deciding whether the
    window is open is the caller's job, not this function's.
    """
    dtype = model.means.dtype
    origin = torch.zeros(3, dtype=dtype)
    with torch.no_grad():
        here = photometric_loss(
            model, torch.as_tensor(matrix_to_axis_angle(R), dtype=dtype),
            origin, camera, image, origin).item()
        there = photometric_loss(
            model, torch.as_tensor(matrix_to_axis_angle(R @ flip_matrix()), dtype=dtype),
            origin, camera, image, origin).item()
    ratio = there / max(here, 1e-12)
    return ratio < 1.0, ratio


def body_frame_sun_angle_deg(R, sun_world):
    """How far the sun has moved in the body frame since the model was fitted.

    The model absorbed the shading it saw on approach, when the sun was fixed
    in the body frame. A turning body moves it, and this is that angle. It is
    computed from the pose estimate and the known sun direction, so it needs
    no ground truth: a servicer knows where the sun is.
    """
    sun = np.asarray(sun_world, float)
    sun = sun / np.linalg.norm(sun)
    body = np.asarray(R, float).T @ sun
    return float(np.degrees(np.arccos(np.clip(float(body @ sun), -1.0, 1.0))))


def track_step(model, camera, mask, R, iterations=TRACK_ITERATIONS):
    """One silhouette tracking step, warm started from the current belief.

    Silhouette because it ignores shading and therefore holds while the baked
    model decays. It cannot acquire, and it will hold a wrong pose exactly as
    stably as a right one, which is why acquisition and verification happen
    before anything calls this.
    """
    R_new, _ = register_from(model, camera, mask, R, silhouette_loss,
                             iterations, TRACK_LR_ROT, TRACK_LR_TRANS)
    return R_new
