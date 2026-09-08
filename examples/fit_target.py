"""Phase 4 step 3: fitting Gaussians to the approach imagery, against a clock.

Reported as a scaling curve rather than a single best result. When a result
depends on a compute budget, the number that matters is quality against time,
because that is what says whether a longer run would have helped. Five
independent fits, each with its own schedule sized to its own budget, so each
row is what you would actually get with that much compute and not a checkpoint
of a longer run under a schedule it never had.

The fit sees images and camera poses. It never imports `target.py`. The
initialisation is a visual hull carved from the silhouettes, which is
information a servicer has, and the only outside quantity is a bounding box
that follows from the approach range and the field of view.

Quality is reported over the object rather than the whole frame. The target
covers about a quarter of the image and the rest is black, so a frame wide
PSNR mostly measures how well black is reproduced: it reads 26.6 dB at a point
where the object itself is at 21.4 dB.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fitting import initialise, learning_rates, psnr
from src.splatting import Gaussians, TorchCamera, render, render_tiled

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = (30, 60, 120, 240, 480, 960)
REPEAT_BUDGET = 240
REPEAT_SEEDS = (11, 12, 13)
N_GAUSSIANS = 6000
HOLDOUT_EVERY = 6


def load():
    d = np.load(ROOT / "data" / "approach" / "views.npz")
    intr = (float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
            int(d["width"]), int(d["height"]))
    cams = []
    for p in d["poses"]:
        holder = type("C", (), dict(width=intr[4], height=intr[5], fx=intr[0],
                                    fy=intr[1], cx=intr[2], cy=intr[3],
                                    R=p[:9].reshape(3, 3), t=p[9:]))()
        cams.append(TorchCamera.from_raytrace(holder, dtype=torch.float32))
    return d, intr, cams


def verify_tiled_matches_dense(gaussians, camera, cutoff=3.0):
    """The fit uses the tiled path, so it has to be the dense one.

    The dense path is what `verify_splatting.py` checked against the volume
    reference. The tiled path is 43x faster and is only a scheduling change,
    which is a claim and not a fact until it is measured, so it is measured
    here before a single optimiser step runs.
    """
    small = Gaussians(*[p.detach().double()[:400] for p in gaussians.parameters()])
    cam64 = TorchCamera(camera.width, camera.height, camera.fx, camera.fy,
                        camera.cx, camera.cy, camera.R.double(), camera.t.double())
    a = render(small, cam64, cutoff_sigmas=cutoff)
    b = render_tiled(small, cam64, cutoff_sigmas=cutoff)
    gap = (a - b).abs().max().item()
    print(f"  tiled against dense at the same cutoff   {gap:.3e}")
    if gap > 1e-12:
        raise SystemExit("the tiled path is not the dense path, so nothing fitted "
                         "with it inherits the verification of the forward pass")
    return gap


def fit(budget, images, cams, train, intr, seed=0, log=None):
    g, voxels = initialise(images, np.stack([np.concatenate([c.R.numpy().ravel(),
                                                             c.t.numpy()]) for c in cams]),
                           intr, N_GAUSSIANS, resolution=128, seed=seed)
    opt = torch.optim.Adam(learning_rates(g, scene_scale=2.0))
    # A cosine decay sized to this budget's own step count, estimated from a
    # short warm up. A schedule borrowed from a longer run would make the
    # short budgets look worse than they are.
    start = time.perf_counter()
    for _ in range(20):
        k = int(np.random.choice(train))
        opt.zero_grad()
        (render_tiled(g, cams[k]) - images[k]).abs().mean().backward()
        opt.step()
    rate = 20 / (time.perf_counter() - start)
    total = max(int(rate * budget), 40)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=0.0)

    steps = 20
    while time.perf_counter() - start < budget:
        k = int(np.random.choice(train))
        opt.zero_grad()
        (render_tiled(g, cams[k]) - images[k]).abs().mean().backward()
        opt.step()
        if steps < total:
            sched.step()
        steps += 1
    return g, steps, time.perf_counter() - start, voxels


def evaluate(g, images, cams, views, mask, parts, names):
    """Held out PSNR over the object, over the frame, and per part."""
    with torch.no_grad():
        preds = {k: render_tiled(g, cams[k]) for k in views}
    obj = np.mean([psnr(preds[k][mask[k]], images[k][mask[k]]) for k in views])
    full = np.mean([psnr(preds[k], images[k]) for k in views])
    per_part = {}
    for i, name in enumerate(names):
        vals = []
        for k in views:
            m = torch.as_tensor(parts[k] == i)
            if m.sum() > 20:
                vals.append(psnr(preds[k][m], images[k][m]))
        per_part[name] = float(np.mean(vals)) if vals else float("nan")
    return obj, full, per_part, preds


def main():
    torch.manual_seed(4); np.random.seed(4)
    d, intr, cams = load()
    images = torch.as_tensor(d["images"], dtype=torch.float32)
    mask = images.max(-1).values > 1e-3
    parts, names = d["primitive"], [str(n) for n in d["primitive_names"]]

    holdout = list(range(0, len(cams), HOLDOUT_EVERY))
    train = [k for k in range(len(cams)) if k not in holdout]
    print(f"phase 4 step 3: fitting under the 30 minute working default")
    print(f"  {len(train)} training views, {len(holdout)} held out, "
          f"{intr[4]}x{intr[5]}, {N_GAUSSIANS} Gaussians, CPU")

    seed_g, voxels = initialise(images.numpy(), d["poses"], intr, N_GAUSSIANS,
                                resolution=128, seed=0)
    print(f"  visual hull at 128^3 leaves {voxels} voxels, from silhouettes and "
          f"poses only")
    verify_tiled_matches_dense(seed_g, cams[0])
    base_obj, base_full, base_parts, _ = evaluate(seed_g, images, cams, holdout,
                                                  mask, parts, names)
    print(f"  the hull alone, before any fitting        {base_obj:.2f} dB on the object\n")

    print(f"  {'budget':>8} {'steps':>7} {'object dB':>10} {'frame dB':>9} "
          f"{'gain':>7} {'per doubling':>13}")
    rows = []
    previous = None
    for budget in BUDGETS:
        g, steps, elapsed, _ = fit(budget, images, cams, train, intr, seed=0)
        obj, full, pp, preds = evaluate(g, images, cams, holdout, mask, parts, names)
        gain = obj - base_obj
        per_doubling = "" if previous is None else f"{obj - previous:+13.2f}"
        print(f"  {budget:6d} s {steps:7d} {obj:10.2f} {full:9.2f} "
              f"{gain:+7.2f} {per_doubling:>13}")
        rows.append((budget, steps, obj, full, pp, preds if budget == BUDGETS[-1] else None))
        previous = obj

    print(f"\n  per part, held out views, at each budget")
    header = "  " + f"{'part':<16}" + "".join(f"{b:>9}s" for b in BUDGETS)
    print(header)
    for name in names:
        line = "  " + f"{name:<16}"
        for _, _, _, _, pp, _ in rows:
            line += f"{pp[name]:9.2f} " if np.isfinite(pp[name]) else f"{'n/a':>9} "
        print(line)

    # Every row above is one draw. The view order is random, so the return
    # per doubling carries scatter that has to be measured before any of it
    # is read as a trend. Repeats at one budget give its size.
    print(f"\n  repeat draws at {REPEAT_BUDGET} s, varying only the view sampling")
    repeats = [r[2] for r in rows if r[0] == REPEAT_BUDGET]
    for seed in REPEAT_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        g, steps, _, _ = fit(REPEAT_BUDGET, images, cams, train, intr, seed=0)
        obj, full, _, _ = evaluate(g, images, cams, holdout, mask, parts, names)
        repeats.append(obj)
        print(f"    seed {seed:3d}  {steps:6d} steps  object {obj:6.2f} dB  "
              f"frame {full:6.2f} dB")
    arr = np.array(repeats)
    print(f"    {len(arr)} draws span {arr.min():.2f} to {arr.max():.2f} dB, "
          f"sd {arr.std(ddof=1):.2f}")
    print(f"    a difference between two budgets therefore carries about "
          f"{arr.std(ddof=1)*np.sqrt(2):.2f} dB of noise")

    total = sum(BUDGETS) + REPEAT_BUDGET * len(REPEAT_SEEDS)
    print(f"\n  total fitting time {total} s, {total/60:.1f} min across "
          f"{len(BUDGETS) + len(REPEAT_SEEDS)} runs; the longest single run is "
          f"{max(BUDGETS)/60:.0f} min,\n  which is what the 30 minute working budget "
          f"governs")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="object")
        axes[0].plot([r[0] for r in rows], [r[3] for r in rows], "s--", label="whole frame")
        axes[0].axhline(base_obj, color="grey", ls=":", label="visual hull alone")
        axes[0].set_xscale("log"); axes[0].set_xlabel("fitting budget (s, log scale)")
        axes[0].set_ylabel("held out PSNR (dB)")
        axes[0].set_title("Quality against compute, five independent fits")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        for name in names:
            axes[1].plot(BUDGETS, [r[4][name] for r in rows], "o-", label=name)
        axes[1].set_xscale("log"); axes[1].set_xlabel("fitting budget (s, log scale)")
        axes[1].set_ylabel("held out PSNR (dB)")
        axes[1].set_title("Per part: where the fit succeeds and where it does not")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROOT / "docs" / "fit_scaling.png", dpi=130)
        print(f"  wrote docs/fit_scaling.png")

        preds = rows[-1][5]
        picks = holdout[:4]
        fig, axes = plt.subplots(2, len(picks), figsize=(2.4 * len(picks), 5))
        for j, k in enumerate(picks):
            axes[0, j].imshow(images[k].numpy()); axes[0, j].set_title(f"view {k}", fontsize=8)
            axes[1, j].imshow(preds[k].clamp(0, 1).numpy())
            for ax in (axes[0, j], axes[1, j]):
                ax.axis("off")
        axes[0, 0].set_ylabel("truth"); axes[1, 0].set_ylabel("fit")
        fig.suptitle(f"Held out views, {BUDGETS[-1]} s fit, {N_GAUSSIANS} Gaussians")
        fig.tight_layout()
        fig.savefig(ROOT / "docs" / "fit_holdout.png", dpi=130)
        print(f"  wrote docs/fit_holdout.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
