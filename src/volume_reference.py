"""Layer C: the same Gaussians rendered as a volume, by brute force ray marching.

This exists to disagree with `splatting.render`. It shares no code and no
derivation with it: there is no projection, no 2D covariance, no Jacobian, no
depth sort and no `over` operator anywhere below. It samples a genuine 3D
density field along each pixel ray and integrates the emission absorption
equation, which is the quantity splatting approximates.

The two parameterisations have to be tied together once, and that step is the
only thing the two share, so it is stated rather than buried. The field is

    sigma(x) = sum_i rho_i exp(-0.5 (x - mu_i)^T Sigma_i^-1 (x - mu_i))

and the line integral of one Gaussian along a ray of unit direction `d` is
available in closed form:

    integral = G_perp * sqrt(2 pi / (d^T Sigma^-1 d))

where `G_perp` is the Gaussian evaluated at its closest approach to the ray,
which is the exact quantity that splatting's projected 2D Gaussian
approximates. Choosing

    rho_i = alpha_i * sqrt(d^T Sigma_i^-1 d / (2 pi))

therefore makes one isolated Gaussian contribute optical depth exactly
`alpha_i G_perp`. With that calibration the two renderers compute the same
thing, and any difference is one of splatting's four named approximations or a
bug, rather than a units mismatch. `verify_splatting.py` checks the
calibration itself before using it.

What remains different by construction, and is what the regime table measures:

- splatting uses `1 - alpha G_2d`, this uses `exp(-alpha G_perp)`; they agree
  to first order, so the gap is second order in the per Gaussian alpha
- splatting evaluates `G_2d` through the affine approximation, this evaluates
  `G_perp` exactly
- splatting sorts once by mean depth, this integrates in true depth order at
  every sample
"""

from __future__ import annotations

import numpy as np
import torch

TWO_PI = 2.0 * np.pi


def bounding_sphere(gaussians, sigmas: float = 4.0):
    """Centre and radius enclosing every Gaussian out to `sigmas`."""
    means = gaussians.means.detach()
    reach = gaussians.scales.detach().max(dim=1).values * sigmas
    centre = means.mean(dim=0)
    radius = ((means - centre).norm(dim=1) + reach).max()
    return centre, radius


def _sphere_span(origins, directions, centre, radius):
    """Entry and exit distance along each ray, and whether it hits at all."""
    oc = origins - centre
    b = (oc * directions).sum(-1)
    c = (oc * oc).sum(-1) - radius ** 2
    disc = b * b - c
    hit = disc > 0
    root = torch.sqrt(disc.clamp_min(0.0))
    return (-b - root), (-b + root), hit


def ray_march(gaussians, camera, samples: int = 256, sigmas: float = 4.0,
              chunk: int = 256, background: float = 0.0):
    """Render by sampling the density field along every pixel ray.

    `samples` is the number of quadrature points across the span of the
    bounding sphere. It is the only accuracy knob, and halving the step is how
    `verify_splatting.py` separates quadrature error from a genuine difference
    between the two renderers.
    """
    dtype, device = gaussians.means.dtype, gaussians.means.device
    means = gaussians.means
    cov = gaussians.covariance()
    inv_cov = torch.linalg.inv(cov)                       # (N, 3, 3)
    opacity = gaussians.opacity
    colors = gaussians.colors

    centre, radius = bounding_sphere(gaussians, sigmas)
    origins = camera.ray_origins(dtype, device).reshape(-1, 3)
    directions = camera.ray_directions(dtype, device).reshape(-1, 3)

    image = torch.zeros(origins.shape[0], 3, dtype=dtype, device=device)
    for lo in range(0, origins.shape[0], chunk):
        o = origins[lo:lo + chunk]                        # (C, 3)
        d = directions[lo:lo + chunk]
        t0, t1, hit = _sphere_span(o, d, centre, radius)
        t0 = torch.where(hit, t0.clamp_min(0.0), torch.zeros_like(t0))
        t1 = torch.where(hit, t1, torch.zeros_like(t1))
        step = (t1 - t0) / samples

        # rho depends on the ray direction only, through d^T Sigma^-1 d.
        dAd = torch.einsum("ci,nij,cj->cn", d, inv_cov, d).clamp_min(1e-20)
        rho = opacity[None, :] * torch.sqrt(dAd / TWO_PI)  # (C, N)

        # Midpoint rule. Sampling at cell centres rather than edges makes the
        # quadrature second order, so halving the step should quarter the
        # error, which is what the convergence check looks for.
        s = (torch.arange(samples, dtype=dtype, device=device) + 0.5)
        t = t0[:, None] + s[None, :] * step[:, None]      # (C, S)
        points = o[:, None, :] + t[..., None] * d[:, None, :]

        delta = points[:, :, None, :] - means[None, None, :, :]        # (C,S,N,3)
        power = -0.5 * torch.einsum("csni,nij,csnj->csn", delta, inv_cov, delta)
        density = rho[:, None, :] * torch.exp(power)                   # (C,S,N)

        sigma = density.sum(-1)                                        # (C,S)
        # Density weighted colour, the emission of the emission absorption
        # model. Where the field is empty the colour is irrelevant because
        # the absorption below is zero.
        emission = torch.einsum("csn,nk->csk", density, colors)
        colour = emission / sigma.clamp_min(1e-30)[..., None]

        tau = sigma * step[:, None]
        absorb = 1.0 - torch.exp(-tau)
        transmit = torch.cumprod(torch.exp(-tau), dim=1)
        ahead = torch.cat([torch.ones_like(transmit[:, :1]), transmit[:, :-1]], dim=1)
        chunk_image = (colour * (absorb * ahead)[..., None]).sum(1)
        chunk_image = chunk_image + background * transmit[:, -1:]
        image[lo:lo + chunk] = torch.where(hit[:, None], chunk_image,
                                           torch.full_like(chunk_image, background))
    return image.reshape(camera.height, camera.width, 3)


def closest_approach_gaussian(gaussians, origins, directions):
    """`G_perp`, the exact Gaussian value at each ray's closest approach.

    The quantity splatting's projected 2D Gaussian is an approximation of.
    Returns `(rays, N)`.
    """
    inv_cov = torch.linalg.inv(gaussians.covariance())
    p = origins[:, None, :] - gaussians.means[None, :, :]          # (C, N, 3)
    dAd = torch.einsum("ci,nij,cj->cn", directions, inv_cov, directions)
    dAp = torch.einsum("ci,nij,cnj->cn", directions, inv_cov, p)
    pAp = torch.einsum("cni,nij,cnj->cn", p, inv_cov, p)
    return torch.exp(-0.5 * (pAp - dAp ** 2 / dAd.clamp_min(1e-20)))


def line_integral(gaussians, origins, directions):
    """Closed form `integral of exp(-0.5 Mahalanobis) dt` along each ray."""
    inv_cov = torch.linalg.inv(gaussians.covariance())
    dAd = torch.einsum("ci,nij,cj->cn", directions, inv_cov, directions).clamp_min(1e-20)
    return closest_approach_gaussian(gaussians, origins, directions) * torch.sqrt(TWO_PI / dAd)
