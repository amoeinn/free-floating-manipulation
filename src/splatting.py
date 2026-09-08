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


def _footprint(cov_2d, cutoff_sigmas):
    """Half width and half height in pixels of a Gaussian's cutoff box."""
    return cutoff_sigmas * torch.sqrt(torch.stack(
        [cov_2d[:, 0, 0], cov_2d[:, 1, 1]], dim=-1).clamp_min(0.0))


def render(gaussians: Gaussians, camera: TorchCamera, dilation: float = 0.0,
           background: float = 0.0, return_parts: bool = False,
           sort_by_depth: bool = True, cutoff_sigmas: float = None):
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
    if cutoff_sigmas is not None:
        # The same truncation the tiled path gets for free by only visiting
        # the tiles a Gaussian reaches. Applied here too, so that the two can
        # be compared at machine precision rather than across a difference
        # nobody has quantified.
        power = torch.where(power > -0.5 * cutoff_sigmas ** 2, power,
                            torch.full_like(power, -1e30))
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


def render_tiled(gaussians: Gaussians, camera: TorchCamera, tile: int = 16,
                 cutoff_sigmas: float = 3.0, dilation: float = 0.0,
                 background: float = 0.0, return_coverage: bool = False):
    """The same image, visiting only the tiles each Gaussian actually reaches.

    `render` evaluates every Gaussian at every pixel, which is `N x H x W`
    and was measured at 2.9 s per forward and backward for 3000 Gaussians at
    128 px. That is a property of the implementation and not of the method,
    and letting it set the compute budget would report the wrong limit. This
    does the identical arithmetic over a tile grid, so the cost falls to the
    footprints the Gaussians occupy.

    It is not a second method and must not be treated as one:
    `verify_tiled_matches_dense` in `examples/fit_target.py` holds it to the
    dense path at the same cutoff, which is the only thing that makes it safe
    to fit with.
    """
    dtype, device = gaussians.means.dtype, gaussians.means.device
    H, W = camera.height, camera.width
    mu, cov, depth, _ = project(gaussians, camera)
    if dilation:
        cov = cov + dilation * torch.eye(2, dtype=dtype, device=device)

    order = torch.argsort(depth)
    mu, cov, depth = mu[order], cov[order], depth[order]
    opacity = gaussians.opacity[order]
    colors = gaussians.colors[order]

    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    live = (depth > 1e-6) & (det > 1e-20)
    inv = torch.stack([
        torch.stack([cov[:, 1, 1], -cov[:, 0, 1]], -1),
        torch.stack([-cov[:, 1, 0], cov[:, 0, 0]], -1),
    ], dim=-2) / det.clamp_min(1e-20)[:, None, None]

    half = _footprint(cov, cutoff_sigmas).detach()
    lo = (mu.detach() - half)
    hi = (mu.detach() + half)

    n_x, n_y = (W + tile - 1) // tile, (H + tile - 1) // tile
    tile_u = torch.arange(n_x, device=device) * tile
    tile_v = torch.arange(n_y, device=device) * tile
    # (n_y, n_x, N): does this Gaussian's cutoff box touch this tile?
    overlap_u = (hi[:, 0][None, :] >= tile_u[:, None]) & (lo[:, 0][None, :] < tile_u[:, None] + tile)
    overlap_v = (hi[:, 1][None, :] >= tile_v[:, None]) & (lo[:, 1][None, :] < tile_v[:, None] + tile)
    touches = overlap_v[:, None, :] & overlap_u[None, :, :] & live[None, None, :]

    uv_all = camera.pixel_grid(dtype, device)
    rows, cover_rows = [], []
    for iy in range(n_y):
        row, cover_row = [], []
        for ix in range(n_x):
            idx = torch.nonzero(touches[iy, ix], as_tuple=False).squeeze(-1)
            patch = uv_all[iy * tile:(iy + 1) * tile, ix * tile:(ix + 1) * tile]
            if idx.numel() == 0:
                row.append(patch.new_full((*patch.shape[:2], 3), background))
                cover_row.append(patch.new_zeros(patch.shape[:2]))
                continue
            m, i2, op, col = mu[idx], inv[idx], opacity[idx], colors[idx]
            d = patch[None] - m[:, None, None, :]
            power = -0.5 * (d[..., 0] ** 2 * i2[:, None, None, 0, 0]
                            + 2 * d[..., 0] * d[..., 1] * i2[:, None, None, 0, 1]
                            + d[..., 1] ** 2 * i2[:, None, None, 1, 1])
            power = torch.where(power > -0.5 * cutoff_sigmas ** 2, power,
                                torch.full_like(power, -1e30))
            a = (op[:, None, None] * torch.exp(power)).clamp(0.0, 1.0 - 1e-7)
            T = torch.cumprod(1.0 - a, dim=0)
            ahead = torch.cat([torch.ones_like(T[:1]), T[:-1]], dim=0)
            img = ((a * ahead)[..., None] * col[:, None, None, :]).sum(0)
            row.append(img + background * T[-1][..., None])
            # Accumulated coverage, which is opacity and not brightness. A
            # face turned away from the light is black and fully covered, so
            # thresholding the image would call it empty.
            cover_row.append(1.0 - T[-1])
        rows.append(torch.cat(row, dim=1))
        cover_rows.append(torch.cat(cover_row, dim=1))
    image = torch.cat(rows, dim=0)[:H, :W]
    if return_coverage:
        return image, torch.cat(cover_rows, dim=0)[:H, :W]
    return image
