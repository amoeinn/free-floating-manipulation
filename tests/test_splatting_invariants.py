"""What the splatting forward pass must be true of, and what would catch it.

The pass is not an exact volume renderer: it makes four approximations, and
the reference that measures them, brute force volume ray marching, lives in
`examples/verify_splatting.py` as a regime table. That table is a
characterisation and is deliberately not asserted here. Asserting it would
freeze numbers that are supposed to move when the method changes.

What is asserted is everything with a bound: the layers that are exact under a
stated condition, the gradients, the tiled path against the dense one, and two
mutations that were measured to be caught. The mutation cases matter more than
the rest, because the same measurement found that the independent volume
reference is nearly blind to both breaks while a two line central difference
catches one of them at nine orders of magnitude.

None of this needs a fitted model or any generated artifact. The Gaussian sets
come from `src.gaussian_scenes`, which is also what the script uses, so the
two cannot drift.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src import splatting
from src.gaussian_scenes import (anisotropic_rotated, axial_slab, discrimination,
                                 occlusion_stack)
from src.raytrace import Camera, look_at
from src.splatting import (Gaussians, TorchCamera, project, projection_jacobian,
                           render, render_tiled)
from src.volume_reference import line_integral, ray_march

DTYPE = torch.float64


def camera(distance=3.0, width=24, height=24, fov=35.0, orthographic=False):
    eye = np.array([distance, 0.0, 0.0])
    R, t = look_at(eye, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    cam = Camera.with_fov(width, height, fov, R, t)
    if orthographic:
        cam.fx = cam.fy = cam.fx / distance
    return TorchCamera.from_raytrace(cam, dtype=DTYPE, orthographic=orthographic)


@pytest.fixture(scope="module")
def scene():
    """Anisotropic, rotated, and asserted so before anything uses it."""
    g = anisotropic_rotated(24, seed=11)
    ratio, off = discrimination(g, camera(orthographic=True))
    assert ratio > 1.5 and off > 0.05
    return g


def test_the_projected_covariance_is_the_exact_marginal_under_orthographic(scene):
    """Orthographic projection is linear, so the affine approximation that the
    perspective path makes disappears entirely and the 2D covariance must equal
    the closed form marginal of the 3D one. This is the only place the
    covariance path can be checked with no approximation in the way."""
    cam = camera(orthographic=True)
    _, cov_2d, _, _ = project(scene, cam)
    R = cam.R.to(DTYPE)
    cov_cam = R @ scene.covariance() @ R.T
    scale = torch.diag(torch.tensor([cam.fx, cam.fy], dtype=DTYPE))
    exact = scale @ cov_cam[:, :2, :2] @ scale
    relative = (cov_2d - exact).abs().max().item() / exact.abs().max().item()
    assert relative < 1e-12, f"{relative:.3e} is not machine precision"


def test_an_isotropic_fixture_is_rejected_as_unable_to_express_a_rotation_error():
    """A covariance that is a scaled identity is unchanged by any rotation, so
    a set of near spherical Gaussians cannot tell a rotation bug from correct
    code. The layer above would pass against the very error it exists to
    catch, which is why the fixture's discriminating property is measured."""
    spherical = anisotropic_rotated(24, seed=11, anisotropy=1.0)
    ratio, off = discrimination(spherical, camera(orthographic=True))
    assert ratio == pytest.approx(1.0, abs=1e-9)
    assert off < 0.05


def test_the_projection_jacobian_matches_a_central_difference(scene):
    """Whether the affine approximation is good is a different question from
    whether the Jacobian was derived correctly. This settles the second."""
    cam = camera()
    R, t = cam.R.to(DTYPE), cam.t.to(DTYPE)
    mean_cam = scene.means @ R.T + t
    J = projection_jacobian(mean_cam, cam)

    def proj(x):
        return torch.stack([cam.fx * x[..., 0] / x[..., 2] + cam.cx,
                            cam.fy * x[..., 1] / x[..., 2] + cam.cy], dim=-1)

    h = 1e-6
    numeric = torch.zeros_like(J)
    for k in range(3):
        e = torch.zeros(3, dtype=DTYPE)
        e[k] = h
        numeric[:, :, k] = (proj(mean_cam + e) - proj(mean_cam - e)) / (2 * h)
    assert (J - numeric).abs().max().item() < 1e-6


def test_dropping_the_jacobians_perspective_terms_is_caught_by_that_difference():
    """The mutation that matters, and where it is caught. Measured, the volume
    reference barely notices this break in its general regimes, at 1.9x, while
    the central difference above catches it at nine orders of magnitude."""
    slab = axial_slab()
    cam = camera(width=64, height=64, fov=80.0)
    R, t = cam.R.to(DTYPE), cam.t.to(DTYPE)
    mean_cam = slab.means @ R.T + t
    honest = projection_jacobian(mean_cam, cam)
    broken = honest.clone()
    broken[:, 0, 2] = 0.0
    broken[:, 1, 2] = 0.0
    assert (honest - broken).abs().max().item() > 1.0, (
        "this fixture does not exercise the perspective terms, so removing "
        "them would be undetectable and the test would prove nothing")


def test_the_closed_form_gaussian_line_integral_matches_numerical_integration(scene):
    """The volume reference converts opacity to a density amplitude through
    this closed form. If it is wrong every regime measured against that
    reference is measuring the wrong thing, so it is checked first."""
    cam = camera(width=8, height=8)
    o = cam.ray_origins(DTYPE, "cpu").reshape(-1, 3)
    d = cam.ray_directions(DTYPE, "cpu").reshape(-1, 3)
    small = Gaussians(*[q[:8] for q in scene.parameters()])
    closed = line_integral(small, o, d)

    inv_cov = torch.linalg.inv(small.covariance())
    t = torch.linspace(0.0, 6.0, 12001, dtype=DTYPE)
    step = t[1] - t[0]
    pts = o[:, None, :] + t[None, :, None] * d[:, None, :]
    delta = pts[:, :, None, :] - small.means[None, None, :, :]
    power = -0.5 * torch.einsum("csni,nij,csnj->csn", delta, inv_cov, delta)
    numeric = torch.exp(power).sum(1) * step
    worst = ((closed - numeric).abs() / closed.abs().clamp_min(1e-12)).max().item()
    assert worst < 1e-6


def test_compositing_gives_the_same_image_front_to_back_and_back_to_front(scene):
    """Algebraically the same operator, so this is machine precision or an
    index slip. Weak and free, and it catches the slips that would otherwise
    surface as a subtly wrong image."""
    cam = camera()
    image, parts = render(scene, cam, return_parts=True)
    alpha = parts["alpha"]
    colors = scene.colors[parts["order"]]
    acc = torch.zeros(cam.height, cam.width, 3, dtype=DTYPE)
    for i in reversed(range(alpha.shape[0])):
        a = alpha[i][..., None]
        acc = colors[i] * a + (1 - a) * acc
    assert (image - acc).abs().max().item() < 1e-12


def test_an_occlusion_stack_in_depth_order_would_make_the_sort_a_no_op():
    """The fixture guard that has actually caught something. The first
    occlusion stack was built in ascending depth order, so removing the depth
    sort changed nothing and the mutation passed against the break it existed
    to catch."""
    stack = occlusion_stack()
    cam = camera()
    _, _, depth, _ = project(stack, cam)
    order = torch.argsort(depth)
    assert not bool((order == torch.arange(len(stack))).all()), (
        "the stack is already in depth order, so sorting it is a no operation")


def test_removing_the_depth_sort_is_caught_by_an_opaque_stack():
    """Where the sort break is detectable, and where it is not.

    Judged against a control rather than a fixed threshold, because a
    threshold here would be a number tuned until the test passed. The control
    is the same stack already in depth order, where sorting genuinely is a no
    operation: the scrambled stack has to move by far more than that.

    This is the measurement that matters. The independent volume reference
    agrees 0.6x *better* without the sort on its randomly overlapping regime,
    so only a stack built for the purpose sees this break at all.
    """
    cam = camera()
    scrambled = occlusion_stack()

    order = torch.argsort(project(scrambled, cam)[2])
    ordered = Gaussians(*[q[order] for q in scrambled.parameters()])

    def sort_effect(g):
        return (render(g, cam) - render(g, cam, sort_by_depth=False)).abs().max().item()

    control = sort_effect(ordered)
    measured = sort_effect(scrambled)
    assert control < 1e-12, (
        f"the control moved by {control:.3e}; it is supposed to be already "
        "sorted, so removing the sort should change nothing at all")
    assert measured > 1e4 * max(control, 1e-15), (
        f"removing the sort changed the scrambled stack by {measured:.3e} "
        f"against a control of {control:.3e}, so this fixture cannot "
        "distinguish a lost sort from no sort being needed")


def test_the_tiled_rasteriser_is_the_dense_one_at_the_same_cutoff(scene):
    """The fitting path uses the tiled renderer and the volume reference
    verified the dense one, so the two have to be the same computation. Being
    43x faster is a scheduling change and not a second method, which is a
    claim and not a fact until it is measured."""
    cam = camera(width=48, height=48)
    for cutoff in (2.0, 3.0, 4.0):
        dense = render(scene, cam, cutoff_sigmas=cutoff)
        tiled = render_tiled(scene, cam, cutoff_sigmas=cutoff)
        assert (dense - tiled).abs().max().item() < 1e-12, f"at {cutoff} sigma"


def test_the_cutoff_is_an_approximation_rather_than_a_free_speedup(scene):
    """Truncating the footprint is not harmless, and the size of it is
    measured rather than assumed. If this ever reads as zero the cutoff has
    stopped doing anything and the tiled path has lost its speed."""
    cam = camera(width=48, height=48)
    full = render(scene, cam)
    cost = {c: (full - render(scene, cam, cutoff_sigmas=c)).abs().max().item()
            for c in (2.0, 3.0, 4.0)}
    assert cost[2.0] > cost[3.0] > cost[4.0] > 0.0
    assert cost[2.0] > 1e-2


def test_the_compositing_gap_falls_as_the_square_of_the_per_gaussian_opacity():
    """Splatting composites with `1 - a` where transmittance is `exp(-tau)`.
    They agree to first order, so the residual against the volume reference
    must fall fourfold per halving of the opacity. This is the one place the
    reference is asserted, because it is a prediction rather than a
    measurement."""
    cam = camera(width=24, height=24, orthographic=True)
    residual = {}
    for opacity in (0.24, 0.12, 0.06):
        g = anisotropic_rotated(6, spread=0.35, scale_range=(0.03, 0.05),
                                opacity=opacity, seed=21)
        splat = render(g, cam)
        volume = ray_march(g, cam, samples=1024)
        residual[opacity] = (splat - volume).abs().max().item()
    for coarse, fine in ((0.24, 0.12), (0.12, 0.06)):
        ratio = residual[coarse] / residual[fine]
        assert 3.0 < ratio < 5.0, (
            f"halving alpha from {coarse} to {fine} changed the residual by "
            f"{ratio:.2f}x, not the fourfold a first order agreement predicts")


def test_every_gaussian_parameter_carries_a_usable_gradient():
    """Step 3 rests entirely on this. Checked on all five groups, because a
    parameter that silently detaches would simply stop being fitted and the
    loss would still fall."""
    g = anisotropic_rotated(8, spread=0.3, seed=31, opacity=0.4).requires_grad_(True)
    cam = camera(width=16, height=16)
    target = torch.rand(cam.height, cam.width, 3, dtype=DTYPE,
                        generator=torch.Generator().manual_seed(3))

    def loss_of(params):
        return ((render(Gaussians(*params), cam) - target) ** 2).mean()

    params = g.parameters()
    grads = torch.autograd.grad(loss_of(params), params)
    names = ["means", "log_scales", "quats", "logit_opacity", "colors"]
    h = 1e-6
    for k, (name, grad) in enumerate(zip(names, grads)):
        assert grad is not None and grad.abs().max() > 0, f"{name} has no gradient"
        idx = int(torch.argmax(grad.abs().reshape(-1)))
        numeric = []
        for sign in (+1, -1):
            bumped = [q.detach().clone() for q in params]
            bumped[k].reshape(-1)[idx] += sign * h
            numeric.append(loss_of(bumped).item())
        finite = (numeric[0] - numeric[1]) / (2 * h)
        analytic = grad.reshape(-1)[idx].item()
        assert abs(finite - analytic) < 1e-5 * max(abs(analytic), 1.0), name
