"""Phase 4 step 2: the splatting forward pass against independent references.

Five layers, none of which checks the forward pass against a second call to
itself. The plan for these was fixed before the pass was written, because the
pass is not an exact volume renderer and a reference chosen afterwards tends
to be one that agrees.

  A  the 2D covariance, exactly, under an orthographic camera
  B  the perspective Jacobian against central differences
  C  the whole pass against brute force volume ray marching, by regime
  D  front to back compositing against an independent back to front pass
  E  gradients against central differences

Every test Gaussian is anisotropic and rotated, and that is asserted rather
than assumed. Isotropic Gaussians cannot express a covariance rotation bug,
in exactly the way that 500 configurations of an axis aligned robot could not
express the phase 1 forward kinematics bug. That guard is the reason this
file exists in this shape.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raytrace import Camera, look_at
from src.splatting import (Gaussians, TorchCamera, project, projection_jacobian,
                           quaternion_to_rotation, render)
from src.volume_reference import (closest_approach_gaussian, line_integral, ray_march)

DT = torch.float64
torch.manual_seed(20260907)


def random_quaternions(n, generator):
    q = torch.randn(n, 4, dtype=DT, generator=generator)
    return q / q.norm(dim=-1, keepdim=True)


def make_gaussians(n, spread=0.5, scale_range=(0.02, 0.09), opacity=0.35,
                   centre=(0.0, 0.0, 0.0), seed=0, anisotropy=3.0):
    g = torch.Generator().manual_seed(seed)
    means = torch.as_tensor(centre, dtype=DT) + spread * (
        torch.rand(n, 3, dtype=DT, generator=g) - 0.5)
    lo, hi = scale_range
    base = lo + (hi - lo) * torch.rand(n, 1, dtype=DT, generator=g)
    # Deliberately elongated: a ratio of `anisotropy` between the longest and
    # shortest axis, so the covariance is nothing like a scaled identity.
    ratios = torch.stack([torch.ones(n, dtype=DT),
                          torch.full((n,), 1.0 / anisotropy, dtype=DT),
                          torch.full((n,), 1.0 / (anisotropy ** 0.5), dtype=DT)], dim=1)
    scales = base * ratios
    return Gaussians(
        means=means,
        log_scales=torch.log(scales),
        quats=random_quaternions(n, g),
        logit_opacity=torch.full((n,), float(np.log(opacity / (1 - opacity))), dtype=DT),
        colors=torch.rand(n, 3, dtype=DT, generator=g) * 0.8 + 0.2,
    )


def camera_at(distance, width=32, height=32, fov=35.0, offset=(0.0, 0.0),
              orthographic=False, target=(0.0, 0.0, 0.0)):
    eye = np.array([distance, offset[0], offset[1]], float)
    R, t = look_at(eye, np.asarray(target, float), np.array([0.0, 0.0, 1.0]))
    cam = Camera.with_fov(width, height, fov, R, t)
    if orthographic:
        # Pixels per metre chosen to frame the same solid angle at `distance`.
        cam.fx = cam.fy = cam.fx / distance
    return TorchCamera.from_raytrace(cam, dtype=DT, orthographic=orthographic)


def guard_anisotropic_and_rotated(gaussians, camera, label):
    """The test set must be able to express the bugs the tests exist to catch."""
    scales = gaussians.scales
    ratio = (scales.max(dim=1).values / scales.min(dim=1).values).min().item()
    R = camera.R.to(DT)
    cov_cam = R @ gaussians.covariance() @ R.T
    diag = torch.diagonal(cov_cam, dim1=-2, dim2=-1).abs().max(dim=1).values
    off = (cov_cam - torch.diag_embed(torch.diagonal(cov_cam, dim1=-2, dim2=-1))).abs()
    off_ratio = (off.amax(dim=(1, 2)) / diag).max().item()
    print(f"  guard [{label}]: worst axis ratio {ratio:.2f} (want > 1.5), "
          f"largest off diagonal {off_ratio:.3f} of the diagonal (want > 0.05)")
    if ratio < 1.5 or off_ratio < 0.05:
        raise SystemExit("the test Gaussians are too close to isotropic or axis "
                         "aligned to express a covariance rotation error, so "
                         "these checks would pass against the bug they exist to catch")


def layer_a_orthographic_covariance():
    """Under orthographic projection the affine approximation is exact.

    The 2D covariance must then equal the exact marginal of the 3D Gaussian
    along the view axis, which is the top left 2x2 of the camera frame
    covariance scaled by the pixel scale. Closed form, so this is assertable
    to machine precision and isolates `R S S^T R^T`, the world to camera
    rotation and the marginalisation from everything else.
    """
    print("\nA. 2D covariance against the exact marginal, orthographic camera")
    g = make_gaussians(24, seed=11)
    cam = camera_at(3.0, orthographic=True)
    guard_anisotropic_and_rotated(g, cam, "layer A")
    _, cov_2d, _, _ = project(g, cam)
    R = cam.R.to(DT)
    cov_cam = R @ g.covariance() @ R.T
    scale = torch.diag(torch.tensor([cam.fx, cam.fy], dtype=DT))
    exact = scale @ cov_cam[:, :2, :2] @ scale
    err = (cov_2d - exact).abs().max().item()
    rel = err / exact.abs().max().item()
    print(f"  max |cov_2d - exact marginal|   {err:.3e}  ({rel:.3e} relative)")
    return rel


def layer_b_jacobian():
    """`J` against a central difference of the projection map itself.

    Whether the affine approximation is good is a different question from
    whether the Jacobian was derived correctly. This settles the second
    without relying on the first.
    """
    print("\nB. perspective Jacobian against central differences")
    g = make_gaussians(16, spread=0.8, seed=12)
    cam = camera_at(3.0)
    R, t = cam.R.to(DT), cam.t.to(DT)
    mean_cam = g.means @ R.T + t
    J = projection_jacobian(mean_cam, cam)

    def proj(x):
        return torch.stack([cam.fx * x[..., 0] / x[..., 2] + cam.cx,
                            cam.fy * x[..., 1] / x[..., 2] + cam.cy], dim=-1)

    h = 1e-6
    numeric = torch.zeros_like(J)
    for k in range(3):
        e = torch.zeros(3, dtype=DT)
        e[k] = h
        numeric[:, :, k] = (proj(mean_cam + e) - proj(mean_cam - e)) / (2 * h)
    err = (J - numeric).abs().max().item()
    print(f"  max |J - central difference|    {err:.3e} px/m")
    return err


def layer_d_compositing_order():
    """Front to back with transmittance against an independent back to front `over`.

    Algebraically the same operator, so this is machine precision or a bug.
    It is a weak check and it is free, and it catches the index slips that
    would otherwise show up as a subtly wrong image.
    """
    print("\nD. front to back compositing against an independent back to front pass")
    g = make_gaussians(20, seed=13, opacity=0.6)
    cam = camera_at(3.0)
    image, parts = render(g, cam, return_parts=True)

    alpha = parts["alpha"]                       # already depth sorted, front first
    colors = g.colors[parts["order"]]
    acc = torch.zeros(cam.height, cam.width, 3, dtype=DT)
    for i in reversed(range(alpha.shape[0])):    # back to front
        a = alpha[i][..., None]
        acc = colors[i] * a + (1 - a) * acc
    err = (image - acc).abs().max().item()
    print(f"  max |front-to-back - back-to-front|  {err:.3e}")
    return err


def layer_c_calibration_check():
    """The one derivation the two renderers share, checked before it is used.

    The volume reference converts opacity to a density amplitude using the
    closed form line integral of a 3D Gaussian. If that closed form is wrong,
    every regime below is measuring the wrong thing, so it is checked against
    a direct numerical integration along the ray first.
    """
    print("\nC0. the closed form line integral, against direct numerical integration")
    g = make_gaussians(8, seed=14)
    cam = camera_at(3.0, width=8, height=8)
    o = cam.ray_origins(DT, "cpu").reshape(-1, 3)
    d = cam.ray_directions(DT, "cpu").reshape(-1, 3)
    closed = line_integral(g, o, d)

    inv_cov = torch.linalg.inv(g.covariance())
    t = torch.linspace(0.0, 6.0, 24001, dtype=DT)
    step = (t[1] - t[0])
    pts = o[:, None, :] + t[None, :, None] * d[:, None, :]
    delta = pts[:, :, None, :] - g.means[None, None, :, :]
    power = -0.5 * torch.einsum("csni,nij,csnj->csn", delta, inv_cov, delta)
    numeric = torch.exp(power).sum(1) * step
    rel = ((closed - numeric).abs() / closed.abs().clamp_min(1e-12)).max().item()
    print(f"  worst relative difference       {rel:.3e}")
    return rel


def compare(gaussians, camera, samples, background=0.0):
    splat = render(gaussians, camera, background=background)
    volume = ray_march(gaussians, camera, samples=samples, background=background)
    resid = (splat - volume).abs().max().item()
    signal = splat.abs().max().item()
    return resid, signal, splat, volume


def layer_c_convergence():
    """Halving the step, so quadrature error can be told from a real difference.

    A residual that keeps falling as the step shrinks is the reference not yet
    converged. A residual that flattens is the two renderers genuinely
    differing, which is the thing worth measuring. Without this the regime
    table below would be uninterpretable.
    """
    print("\nC1. step halving, separated Gaussians on axis at low opacity")
    g = make_gaussians(6, spread=0.35, scale_range=(0.03, 0.05), opacity=0.05, seed=15)
    cam = camera_at(3.0, width=24, height=24)
    guard_anisotropic_and_rotated(g, cam, "layer C")
    print(f"  {'samples':>9} {'step (mm)':>11} {'max residual':>14} {'ratio':>8}")
    previous = None
    for samples in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        resid, _, _, _ = compare(g, cam, samples)
        span = 2 * (g.means.std(0).max().item() + 4 * g.scales.max().item())
        ratio = "" if previous is None else f"{previous / resid:8.2f}"
        print(f"  {samples:9d} {1000*span/samples:11.3f} {resid:14.3e} {ratio:>8}")
        previous = resid
    return previous


def layer_c_regimes():
    """Where the two agree, and how the gap grows as each approximation bites."""
    print("\nC2. regime table, 1024 samples per ray throughout")
    print(f"  {'regime':<44} {'max resid':>11} {'rel to peak':>12}  what it isolates")

    rows = []

    # 1. The assertable case: separated, on axis, low opacity, and made
    #    orthographic so the affine approximation is not in play at all.
    g = make_gaussians(6, spread=0.35, scale_range=(0.03, 0.05), opacity=0.03, seed=21)
    cam = camera_at(3.0, width=24, height=24, orthographic=True)
    r, s, _, _ = compare(g, cam, 1024)
    rows.append(("separated, orthographic, alpha 0.03", r, r / s, "nothing; must agree"))

    # 2. Opacity sweep, still orthographic and separated, so the only
    #    difference left is 1 - a against exp(-tau). Second order in alpha,
    #    so the residual should fall about fourfold per halving.
    for op in (0.24, 0.12, 0.06, 0.03):
        g = make_gaussians(6, spread=0.35, scale_range=(0.03, 0.05), opacity=op, seed=21)
        r, s, _, _ = compare(g, cam, 1024)
        rows.append((f"orthographic, separated, alpha {op:.2f}", r, r / s,
                     "the compositing operator"))

    # 3. Perspective, on axis: the affine approximation switches on.
    g = make_gaussians(6, spread=0.35, scale_range=(0.03, 0.05), opacity=0.06, seed=21)
    cam_p = camera_at(3.0, width=24, height=24)
    r, s, _, _ = compare(g, cam_p, 1024)
    rows.append(("perspective, on axis", r, r / s, "the affine approximation"))

    # 4. Off axis. The first version of this moved the camera but left it
    #    pointing at the cluster, so the Gaussians stayed dead centre in the
    #    image and the affine approximation was never stressed at all. What
    #    matters is where a Gaussian sits in the frame, so move the cluster.
    wide = camera_at(3.0, width=48, height=48, fov=60.0)
    for lateral in (0.0, 0.5, 1.0, 1.5):
        g_off = make_gaussians(6, spread=0.35, scale_range=(0.03, 0.05),
                               opacity=0.06, seed=21, centre=(0.0, lateral, 0.0))
        mu, _, _, _ = project(g_off, wide)
        frac = ((mu - torch.tensor([wide.cx, wide.cy], dtype=DT)).norm(dim=1)
                / (wide.width / 2)).max().item()
        angle = np.degrees(np.arctan2(lateral, 3.0))
        r, s_, _, _ = compare(g_off, wide, 1024)
        rows.append((f"perspective, {angle:4.1f} deg off axis "
                     f"({frac:.2f} of half frame)", r, r / s_,
                     "the affine approximation"))

    # 5. Overlapping in depth, where the single global sort and the collapsed
    #    depth integral both start to matter.
    for spread, label in ((0.10, "tight"), (0.04, "heavily")):
        g_ov = make_gaussians(10, spread=spread, scale_range=(0.05, 0.09),
                              opacity=0.20, seed=22)
        o = cam_p.ray_origins(DT, "cpu").reshape(-1, 3)
        d = cam_p.ray_directions(DT, "cpu").reshape(-1, 3)
        per_ray = (closest_approach_gaussian(g_ov, o, d) > 0.1).sum(1)
        seen = per_ray[per_ray > 0]
        if per_ray.max().item() < 2:
            raise SystemExit(f"the '{label} overlapping' case has no ray passing "
                             "near two Gaussians, so it does not test overlap")
        r, s_, _, _ = compare(g_ov, cam_p, 1024)
        rows.append((f"perspective, {label} overlapping "
                     f"({seen.double().mean().item():.1f} per ray, max "
                     f"{per_ray.max().item()})", r, r / s_,
                     "sort and depth collapse"))

    # 6. Angular extent, which turns out to be the axis that actually drives
    #    the affine error, and its interaction with off axis position. The
    #    off axis sweep above is flat because these Gaussians subtend under a
    #    degree; the approximation cannot fail on position alone.
    g_big = make_gaussians(6, spread=0.5, scale_range=(0.20, 0.35), opacity=0.06, seed=23)
    r, s_, _, _ = compare(g_big, cam_p, 1024)
    rows.append(("perspective, Gaussians 6x larger, on axis", r, r / s_,
                 "the affine approximation"))
    g_both = make_gaussians(6, spread=0.5, scale_range=(0.20, 0.35), opacity=0.06,
                            seed=23, centre=(0.0, 1.5, 0.0))
    r, s_, _, _ = compare(g_both, wide, 1024)
    rows.append(("perspective, 6x larger and 26.6 deg off axis", r, r / s_,
                 "the affine approximation"))

    for label, resid, rel, isolates in rows:
        print(f"  {label:<44} {resid:11.3e} {rel:12.3e}  {isolates}")
    return rows


def occlusion_stack(opacity=0.25, depths=(2.4, 3.0, 3.6), extent=0.05):
    """Gaussians stacked along one line of sight, differently coloured.

    Built explicitly, because the point is a case where depth order decides
    the pixel colour. The camera sits on +x at 3 m, so varying world x walks
    them along the view axis while they stay on top of each other in the
    image. They are returned deliberately out of depth order: the first
    version of this returned them already sorted, which made removing the
    sort a no operation and the mutation undetectable by construction.
    """
    n = len(depths)
    order = [2, 0, 1][:n]
    means = torch.zeros(n, 3, dtype=DT)
    means[:, 0] = 3.0 - torch.as_tensor([depths[i] for i in order], dtype=DT)
    colors = torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]],
                          dtype=DT)[:n]
    return Gaussians(
        means=means,
        log_scales=torch.log(torch.full((n, 3), extent, dtype=DT)
                             * torch.tensor([1.0, 0.6, 1.6], dtype=DT)),
        quats=random_quaternions(n, torch.Generator().manual_seed(77)),
        logit_opacity=torch.full((n,), float(np.log(opacity / (1 - opacity))), dtype=DT),
        colors=colors)


def axial_slab(n=4, lateral=2.2, along=0.45, across=0.06, opacity=0.06):
    """Gaussians elongated along the view axis, far off centre in the frame.

    The perspective terms of the Jacobian, `-fx x / z^2`, couple extent along
    the view axis into image displacement. They therefore carry no weight
    unless a Gaussian is both long in depth and off axis, which is why the
    earlier random test sets could not detect their removal.
    """
    g = torch.Generator().manual_seed(5)
    means = torch.zeros(n, 3, dtype=DT)
    means[:, 0] = torch.linspace(-0.4, 0.4, n, dtype=DT)
    means[:, 1] = lateral + 0.15 * torch.linspace(-1, 1, n, dtype=DT)
    quats = torch.zeros(n, 4, dtype=DT)
    quats[:, 0] = 1.0                       # long axis stays on the view axis
    scales = torch.tensor([[along, across, across]], dtype=DT).repeat(n, 1)
    return Gaussians(means, torch.log(scales), quats,
                     torch.full((n,), float(np.log(opacity / (1 - opacity))), dtype=DT),
                     torch.rand(n, 3, dtype=DT, generator=g) * 0.8 + 0.2)


def mutation_check():
    """Break the forward pass on purpose, and record which layer notices.

    A regime table full of small numbers proves nothing unless a real error
    would make them large. Two plausible breaks: dropping the two perspective
    terms of the Jacobian, which is a transcription slip, and compositing in
    input order, which is what a refactor does to a sort.

    The result is not that every layer catches everything. It is that each
    break is caught decisively by one layer and is nearly invisible to the
    others, and knowing which is which is the point of running this.
    """
    print("\nC3. mutation, and which layer catches which break")
    import src.splatting as sp

    ortho = camera_at(3.0, width=24, height=24, orthographic=True)
    persp = camera_at(3.0, width=24, height=24)
    wide = camera_at(3.0, width=64, height=64, fov=80.0)
    separated = make_gaussians(6, spread=0.35, scale_range=(0.03, 0.05),
                               opacity=0.03, seed=21)
    overlapping = make_gaussians(10, spread=0.10, scale_range=(0.05, 0.09),
                                 opacity=0.20, seed=22)
    stack = occlusion_stack()
    slab = axial_slab()

    _, _, stack_depth, _ = project(stack, persp)
    if bool((torch.argsort(stack_depth) == torch.arange(len(stack))).all()):
        raise SystemExit("the occlusion stack is already in depth order, so removing "
                         "the sort would be a no operation and this would pass "
                         "against the very break it exists to catch")

    original = sp.projection_jacobian

    def flat_jacobian(mean_cam, camera):
        J = original(mean_cam, camera).clone()
        J[:, 0, 2] = 0.0
        J[:, 1, 2] = 0.0
        return J

    def volume_residual(g, cam, sort=True, patched=False):
        sp.projection_jacobian = flat_jacobian if patched else original
        splat = sp.render(g, cam, sort_by_depth=sort)
        sp.projection_jacobian = original
        return (splat - ray_march(g, cam, samples=1024)).abs().max().item()

    def jacobian_residual(g, cam, patched=False):
        """Layer B's check, run on demand: J against central differences."""
        R, t = cam.R.to(DT), cam.t.to(DT)
        mean_cam = g.means @ R.T + t
        J = (flat_jacobian if patched else original)(mean_cam, cam)

        def proj(x):
            return torch.stack([cam.fx * x[..., 0] / x[..., 2] + cam.cx,
                                cam.fy * x[..., 1] / x[..., 2] + cam.cy], dim=-1)
        h = 1e-6
        numeric = torch.zeros_like(J)
        for k in range(3):
            e = torch.zeros(3, dtype=DT)
            e[k] = h
            numeric[:, :, k] = (proj(mean_cam + e) - proj(mean_cam - e)) / (2 * h)
        return (J - numeric).abs().max().item()

    print(f"  {'break':<28} {'layer and regime':<40} {'clean':>10} {'broken':>10} {'x':>7}")
    rows = [
        ("perspective terms of J", "B, Jacobian vs central differences",
         jacobian_residual(slab, wide), jacobian_residual(slab, wide, patched=True)),
        ("perspective terms of J", "C, orthographic separated",
         volume_residual(separated, ortho), volume_residual(separated, ortho, patched=True)),
        ("perspective terms of J", "C, small Gaussians on axis",
         volume_residual(separated, persp), volume_residual(separated, persp, patched=True)),
        ("perspective terms of J", "C, long in depth and 36 deg off axis",
         volume_residual(slab, wide), volume_residual(slab, wide, patched=True)),
        ("depth sort", "C, small Gaussians on axis",
         volume_residual(separated, persp), volume_residual(separated, persp, sort=False)),
        ("depth sort", "C, randomly overlapping",
         volume_residual(overlapping, persp), volume_residual(overlapping, persp, sort=False)),
        ("depth sort", "C, opaque stack on one line of sight",
         volume_residual(stack, persp), volume_residual(stack, persp, sort=False)),
    ]
    factors = {}
    for name, where, clean, broken in rows:
        factor = broken / clean if clean > 0 else float("inf")
        factors[(name, where)] = factor
        print(f"  {name:<28} {where:<40} {clean:10.3e} {broken:10.3e} {factor:7.1f}")

    best_j = max(f for (n, _), f in factors.items() if n.startswith("perspective"))
    best_sort = max(f for (n, _), f in factors.items() if n.startswith("depth"))
    print(f"\n  best detection: Jacobian {best_j:.0f}x, sort {best_sort:.1f}x")
    print("  neither break is visible in the general regimes. The Jacobian is caught "
          "by layer B\n  outright; the sort is caught only by a regime built for it, "
          "and only at a few times\n  over, because the regimes where ordering matters "
          "are also where the two renderers\n  genuinely differ most. Layer C "
          "characterises the approximations; layers A, B, D\n  and E are what pin the "
          "implementation.")
    if best_j < 100 or best_sort < 2.0:
        raise SystemExit("a deliberate break is not detected by any layer, so these "
                         "checks are not evidence of anything")
    return factors


def layer_e_gradients():
    """Autograd against central differences, on every parameter a fit moves."""
    print("\nE. gradients of the image against central differences")
    g = make_gaussians(8, spread=0.3, seed=31, opacity=0.4).requires_grad_(True)
    cam = camera_at(3.0, width=16, height=16)
    target = torch.rand(cam.height, cam.width, 3, dtype=DT, generator=
                        torch.Generator().manual_seed(3))

    def loss_of(params):
        gg = Gaussians(*params)
        return ((render(gg, cam) - target) ** 2).mean()

    params = g.parameters()
    loss = loss_of(params)
    grads = torch.autograd.grad(loss, params)

    names = ["means", "log_scales", "quats", "logit_opacity", "colors"]
    worst = 0.0
    h = 1e-6
    for k, (name, p, grad) in enumerate(zip(names, params, grads)):
        flat = p.detach().reshape(-1)
        picks = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(k))[:6]
        errs = []
        for idx in picks:
            numeric = []
            for sign in (+1, -1):
                bumped = [q.detach().clone() for q in params]
                bumped[k].reshape(-1)[idx] += sign * h
                numeric.append(loss_of(bumped).item())
            fd = (numeric[0] - numeric[1]) / (2 * h)
            errs.append(abs(fd - grad.reshape(-1)[idx].item()))
        scale = grad.abs().max().item()
        rel = max(errs) / max(scale, 1e-12)
        worst = max(worst, rel)
        print(f"  {name:16s} worst |autograd - fd|  {max(errs):.3e}  "
              f"({rel:.3e} of the largest gradient)")
    return worst


def main():
    print("phase 4 step 2: the splatting forward pass against independent references")
    print("all in float64; every test set anisotropic and rotated, asserted below")

    a = layer_a_orthographic_covariance()
    b = layer_b_jacobian()
    c0 = layer_c_calibration_check()
    d = layer_d_compositing_order()
    floor = layer_c_convergence()
    layer_c_regimes()
    mutations = mutation_check()
    e = layer_e_gradients()

    print("\nsummary")
    print(f"  A  2D covariance, orthographic, vs exact marginal   {a:.3e} relative")
    print(f"  B  Jacobian vs central differences                  {b:.3e} px/m")
    print(f"  C0 line integral closed form vs numerical           {c0:.3e} relative")
    print(f"  C1 converged residual against the volume reference  {floor:.3e}")
    print(f"  D  compositing order                                {d:.3e}")
    print(f"  E  gradients vs central differences                 {e:.3e} relative")
    for label, value, bound in (("A", a, 1e-12), ("B", b, 1e-6), ("C0", c0, 1e-6),
                                ("D", d, 1e-12), ("E", e, 1e-5)):
        if not value < bound:
            raise SystemExit(f"layer {label} did not meet its bound: {value:.3e} >= {bound:.0e}")
    print("\n  layers A, B, C0, D and E are within bound; layer C is a regime table "
          "and is read, not asserted")


if __name__ == "__main__":
    main()
