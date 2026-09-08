"""3D Gaussian splatting forward pass, in torch, differentiable end to end.

Written rather than imported. `PLAN.md` records why: every usable
implementation is either copyleft, non-commercial, or CUDA only, and beyond
the licences a dependency would remove the content of the phase and leave
nothing to disagree with.

What this computes, and what it approximates, stated here because the second
half is what the verification had to be designed around. The pass is not a
volume renderer. It makes four approximations, each of which
`examples/verify_splatting.py` isolates and measures:

1. **A local affine approximation of the projection.** The 2D covariance uses
   the Jacobian of the perspective map evaluated at the Gaussian's mean, so it
   is exact only at the mean and degrades with angular extent and off axis
   position. Under an orthographic camera the map is linear and the
   approximation disappears, which is what makes an exact reference possible.
2. **The depth integral is collapsed.** Each Gaussian contributes one alpha
   per pixel rather than its density integrated along the ray.
3. **One global sort.** Gaussians are ordered once by the depth of their
   means, so overlapping footprints whose true ordering varies across a pixel
   get a single order.
4. **`1 - a` in place of `exp(-tau)`.** The compositing operator agrees with
   transmittance only to first order in the per Gaussian alpha.

Frames are `raytrace`'s and are not restated: world to camera `(R, t)`,
OpenCV axes, pixel `(u, v)` a column and a row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TWO_PI = 2.0 * np.pi


def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """`(N, 4)` quaternions as `(w, x, y, z)` to `(N, 3, 3)` rotations.

    Normalised here rather than assumed, because a fit moves the quaternion
    off the unit sphere at every step and a silently unnormalised quaternion
    scales the covariance instead of rotating it.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


@dataclass
class Gaussians:
    """A set of 3D Gaussians, in the parameters a fit actually moves.

    Scales are held as logs and opacity as a logit so that gradient steps
    cannot drive either out of its valid range, which is the usual
    parameterisation and matters once step 3 starts optimising.
    """
    means: torch.Tensor          # (N, 3) world
    log_scales: torch.Tensor     # (N, 3)
    quats: torch.Tensor          # (N, 4) as (w, x, y, z)
    logit_opacity: torch.Tensor  # (N,)
    colors: torch.Tensor         # (N, 3)

    def __len__(self):
        return self.means.shape[0]

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacity)

    def parameters(self):
        return [self.means, self.log_scales, self.quats, self.logit_opacity, self.colors]

    def requires_grad_(self, flag: bool = True):
        for p in self.parameters():
            p.requires_grad_(flag)
        return self

    def covariance(self) -> torch.Tensor:
        """`(N, 3, 3)` world covariance, `R S S^T R^T`."""
        R = quaternion_to_rotation(self.quats)
        RS = R * self.scales[:, None, :]
        return RS @ RS.transpose(-1, -2)


@dataclass
class TorchCamera:
    """The `raytrace.Camera` conventions, in torch, for the differentiable path."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: torch.Tensor              # (3, 3) world to camera
    t: torch.Tensor              # (3,)
    orthographic: bool = False

    @classmethod
    def from_raytrace(cls, cam, dtype=torch.float64, orthographic: bool = False):
        return cls(cam.width, cam.height, float(cam.fx), float(cam.fy),
                   float(cam.cx), float(cam.cy),
                   torch.as_tensor(cam.R, dtype=dtype),
                   torch.as_tensor(cam.t, dtype=dtype), orthographic)

    def pixel_grid(self, dtype, device):
        v, u = torch.meshgrid(torch.arange(self.height, dtype=dtype, device=device),
                              torch.arange(self.width, dtype=dtype, device=device),
                              indexing="ij")
        return torch.stack([u, v], dim=-1)          # (H, W, 2)

    def ray_directions(self, dtype, device) -> torch.Tensor:
        """Unit world directions per pixel, matching `raytrace.Camera.rays`."""
        uv = self.pixel_grid(dtype, device)
        if self.orthographic:
            d = torch.zeros(self.height, self.width, 3, dtype=dtype, device=device)
            d[..., 2] = 1.0
        else:
            d = torch.stack([(uv[..., 0] - self.cx) / self.fx,
                             (uv[..., 1] - self.cy) / self.fy,
                             torch.ones_like(uv[..., 0])], dim=-1)
        world = d @ self.R.to(dtype)
        return world / world.norm(dim=-1, keepdim=True)

    def ray_origins(self, dtype, device) -> torch.Tensor:
        """Per pixel ray origins. A pinhole shares one; an orthographic camera
        does not, which is the whole reason its projection is linear."""
        centre = -self.R.to(dtype).T @ self.t.to(dtype)
        if not self.orthographic:
            return centre.expand(self.height, self.width, 3)
        uv = self.pixel_grid(dtype, device)
        offset = torch.stack([(uv[..., 0] - self.cx) / self.fx,
                              (uv[..., 1] - self.cy) / self.fy,
                              torch.zeros_like(uv[..., 0])], dim=-1)
        return centre + offset @ self.R.to(dtype)


def projection_jacobian(mean_cam: torch.Tensor, camera: TorchCamera) -> torch.Tensor:
    """`(N, 2, 3)` Jacobian of the projection, at each Gaussian's mean.

    Orthographic drops the perspective divide, so the Jacobian is constant and
    the affine approximation below stops being an approximation at all.
    """
    N = mean_cam.shape[0]
    J = mean_cam.new_zeros(N, 2, 3)
    if camera.orthographic:
        J[:, 0, 0] = camera.fx
        J[:, 1, 1] = camera.fy
        return J
    z = mean_cam[:, 2]
    J[:, 0, 0] = camera.fx / z
    J[:, 1, 1] = camera.fy / z
    J[:, 0, 2] = -camera.fx * mean_cam[:, 0] / z ** 2
    J[:, 1, 2] = -camera.fy * mean_cam[:, 1] / z ** 2
    return J


def project(gaussians: Gaussians, camera: TorchCamera):
    """Means, 2D covariances and depths in the image.

    Returns `(mu_2d (N,2), cov_2d (N,2,2), depth (N,), mean_cam (N,3))`.
    """
    dtype = gaussians.means.dtype
    R, t = camera.R.to(dtype), camera.t.to(dtype)
    mean_cam = gaussians.means @ R.T + t
    depth = mean_cam[:, 2]
    if camera.orthographic:
        u = camera.fx * mean_cam[:, 0] + camera.cx
        v = camera.fy * mean_cam[:, 1] + camera.cy
    else:
        u = camera.fx * mean_cam[:, 0] / depth + camera.cx
        v = camera.fy * mean_cam[:, 1] / depth + camera.cy
    cov_cam = R @ gaussians.covariance() @ R.T
    J = projection_jacobian(mean_cam, camera)
    cov_2d = J @ cov_cam @ J.transpose(-1, -2)
    return torch.stack([u, v], dim=-1), cov_2d, depth, mean_cam


def render(gaussians: Gaussians, camera: TorchCamera, dilation: float = 0.0,
           background: float = 0.0, return_parts: bool = False,
           sort_by_depth: bool = True):
    """The image, by projection, depth sort and front to back compositing.

    `sort_by_depth` exists so that the verification can turn the depth sort
    off and confirm which checks notice. It has no other use and should stay
    true everywhere else.

    `dilation` adds a fraction of a pixel to the 2D covariance diagonal. Real
    implementations set it so that sub pixel Gaussians do not vanish between
    samples; it is zero by default here so that a comparison against the
    volume reference measures the method rather than an anti aliasing hack.
    """
    dtype, device = gaussians.means.dtype, gaussians.means.device
    mu, cov, depth, _ = project(gaussians, camera)
    if dilation:
        cov = cov + dilation * torch.eye(2, dtype=dtype, device=device)

    order = (torch.argsort(depth) if sort_by_depth
             else torch.arange(depth.shape[0], device=depth.device))
    mu, cov, depth = mu[order], cov[order], depth[order]
    opacity = gaussians.opacity[order]
    colors = gaussians.colors[order]

    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    inv = torch.stack([
        torch.stack([cov[:, 1, 1], -cov[:, 0, 1]], -1),
        torch.stack([-cov[:, 1, 0], cov[:, 0, 0]], -1),
    ], dim=-2) / det.clamp_min(1e-20)[:, None, None]

    uv = camera.pixel_grid(dtype, device)           # (H, W, 2)
    d = uv[None] - mu[:, None, None, :]             # (N, H, W, 2)
    power = -0.5 * (d[..., 0] ** 2 * inv[:, None, None, 0, 0]
                    + 2 * d[..., 0] * d[..., 1] * inv[:, None, None, 0, 1]
                    + d[..., 1] ** 2 * inv[:, None, None, 1, 1])

    # Behind the camera, or degenerate, contributes nothing. Masking the
    # exponent rather than the alpha keeps the gradient finite.
    valid = (depth > 1e-6) & (det > 1e-20)
    power = torch.where(valid[:, None, None], power, torch.full_like(power, -1e30))
    alpha = (opacity[:, None, None] * torch.exp(power)).clamp(0.0, 1.0 - 1e-7)

    transmittance = torch.cumprod(1.0 - alpha, dim=0)
    ahead = torch.cat([torch.ones_like(transmittance[:1]), transmittance[:-1]], dim=0)
    weight = alpha * ahead                          # (N, H, W)
    image = (weight[..., None] * colors[:, None, None, :]).sum(0)
    image = image + background * transmittance[-1][..., None]
    if return_parts:
        return image, {"alpha": alpha, "weight": weight, "order": order,
                       "mu": mu, "cov": cov, "depth": depth,
                       "transmittance": transmittance}
    return image
