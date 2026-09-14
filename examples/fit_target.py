"""Phase 4 step 3: fitting Gaussians to the approach imagery, against a clock.

Reported as a scaling curve rather than a single best result. When a result
depends on a compute budget, the number that matters is quality against time,
because that is what says whether a longer run would have helped. One
independent fit per budget in BUDGETS, each with its own schedule sized to its
own budget, so each row is what you would actually get with that much compute
and not a checkpoint of a longer run under a schedule it never had.

The sweep costs about 43 minutes and writes its rows to `data/fit_scaling.json`
alongside the figures, so `--replot` redraws both figures from that file in
seconds without refitting. Everything either figure needs comes from that
record, which is also what keeps a figure's own labels from disagreeing with
the points on it.

The fit sees images and camera poses. It never imports `target.py`. The
initialisation is a visual hull carved from the silhouettes, which is
information a servicer has, and the only outside quantity is a bounding box
that follows from the approach range and the field of view.

Quality is reported over the object rather than the whole frame. The target
covers about a quarter of the image and the rest is black, so a frame wide
PSNR mostly measures how well black is reproduced: it reads 26.6 dB at a point
where the object itself is at 21.4 dB.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fitting import initialise, learning_rates, psnr
from src.splatting import Gaussians, TorchCamera, render, render_tiled

ROOT = Path(__file__).resolve().parent.parent

# The two figures are coupled through the last budget, and the coupling is not
# obvious from either one. fit_holdout.png takes its title and its images from
# the final row, so any run that extends BUDGETS also relabels the held out
# views: adding the 960 s point moved that figure from a 480 s fit to a 960 s
# fit without anything about the holdout code changing. The portfolio review
# document quotes both labels and the PSNR that goes with each, so a run that
# changes BUDGETS invalidates both of its figure captions and they have to be
# updated in the same change. Regenerating one of these figures without the
# other is not possible here, and is not meant to be.
BUDGETS = (30, 60, 120, 240, 480, 960)
REPEAT_BUDGET = 240
REPEAT_SEEDS = (11, 12, 13)
N_GAUSSIANS = 6000
HOLDOUT_EVERY = 6
RESULTS = ROOT / "data" / "fit_scaling.json"
PREDICTIONS = ROOT / "data" / "fit_scaling_preds.npz"


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


def build_record(rows, base_obj, names, voxels, repeats, holdout):
    """The whole of what both figures need, in one serialisable structure.

    Both the sweep and `--replot` draw from this and from nothing else. A
    figure label taken from a module constant can disagree with the points
    actually plotted, which is how fit_scaling.png came to be titled "five
    independent fits" while BUDGETS held six; a label taken from the record
    cannot.
    """
    return {
        "schema": 1,
        "n_gaussians": N_GAUSSIANS,
        "hull_voxels": int(voxels),
        "holdout_every": HOLDOUT_EVERY,
        "hull_object_psnr": float(base_obj),
        "part_names": list(names),
        "holdout_views": [int(k) for k in holdout],
        "rows": [
            {"budget_s": int(b), "steps": int(st), "object_psnr": float(o),
             "frame_psnr": float(f),
             "per_part": {n: float(pp[n]) for n in names}}
            for b, st, o, f, pp, _ in rows
        ],
        "repeats": {
            "budget_s": REPEAT_BUDGET,
            "extra_seeds": list(REPEAT_SEEDS),
            # The first draw is the sweep's own row at this budget, so the
            # list is one longer than extra_seeds.
            "object_psnr": [float(x) for x in repeats],
        },
        "predictions_file": PREDICTIONS.name,
    }


def save_record(record, preds, picks):
    """Rows to JSON, predicted pixels to a companion npz.

    The images do not belong in the JSON, which exists to be read: four held
    out predictions are three quarters of a megabyte of float32 and would bury
    the rows they sit next to. The record names the companion file so the
    replot path is driven by the JSON alone.
    """
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(PREDICTIONS, views=np.asarray(picks, dtype=np.int64),
                        images=np.stack([preds[k].clamp(0, 1).numpy() for k in picks]))
    print(f"  wrote {RESULTS.relative_to(ROOT)} and {PREDICTIONS.relative_to(ROOT)}")


def load_record():
    if not RESULTS.exists():
        raise SystemExit(f"no {RESULTS.relative_to(ROOT)}: run the sweep once to "
                         f"produce it, which takes about 43 minutes")
    record = json.loads(RESULTS.read_text())
    companion = RESULTS.parent / record["predictions_file"]
    if not companion.exists():
        raise SystemExit(f"{RESULTS.relative_to(ROOT)} names {record['predictions_file']}, "
                         f"which is missing; the held out figure cannot be drawn without it")
    d = np.load(companion)
    picks = [int(k) for k in d["views"]]
    return record, picks, {k: d["images"][i] for i, k in enumerate(picks)}


def plot(record, picks, preds, truth):
    """Both figures, from the record and nothing else."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, names = record["rows"], record["part_names"]
    budgets = [r["budget_s"] for r in rows]
    n = len(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(budgets, [r["object_psnr"] for r in rows], "o-", label="object")
    axes[0].plot(budgets, [r["frame_psnr"] for r in rows], "s--", label="whole frame")
    axes[0].axhline(record["hull_object_psnr"], color="grey", ls=":",
                    label="visual hull alone")
    axes[0].set_xscale("log"); axes[0].set_xlabel("fitting budget (s, log scale)")
    axes[0].set_ylabel("held out PSNR (dB)")
    axes[0].set_title(f"Quality against compute, {n} independent "
                      f"{'fit' if n == 1 else 'fits'}")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for name in names:
        axes[1].plot(budgets, [r["per_part"][name] for r in rows], "o-", label=name)
    axes[1].set_xscale("log"); axes[1].set_xlabel("fitting budget (s, log scale)")
    axes[1].set_ylabel("held out PSNR (dB)")
    axes[1].set_title("Per part: where the fit succeeds and where it does not")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "fit_scaling.png", dpi=130)
    plt.close(fig)
    print(f"  wrote docs/fit_scaling.png")

    # Title and images both come from the final row, which is the coupling
    # described at the top of this file.
    fig, axes = plt.subplots(2, len(picks), figsize=(2.4 * len(picks), 5))
    for j, k in enumerate(picks):
        axes[0, j].imshow(truth[k]); axes[0, j].set_title(f"view {k}", fontsize=8)
        axes[1, j].imshow(preds[k])
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    axes[0, 0].set_ylabel("truth"); axes[1, 0].set_ylabel("fit")
    fig.suptitle(f"Held out views, {rows[-1]['budget_s']} s fit, "
                 f"{record['n_gaussians']} Gaussians")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "fit_holdout.png", dpi=130)
    plt.close(fig)
    print(f"  wrote docs/fit_holdout.png")


def replot():
    """Redraw both figures from the last sweep. Seconds, not 43 minutes."""
    record, picks, preds = load_record()
    truth = np.load(ROOT / "data" / "approach" / "views.npz")["images"]
    print(f"  replotting {len(record['rows'])} budgets from "
          f"{RESULTS.relative_to(ROOT)}, no fitting")
    plot(record, picks, preds, {k: truth[k] for k in picks})


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

    # The record is written before anything is drawn, so a sweep that survives
    # the fitting is never lost to a plotting failure. Forty three minutes is
    # too expensive to spend twice on a matplotlib error.
    picks = holdout[:4]
    record = build_record(rows, base_obj, names, voxels, repeats, holdout)
    save_record(record, rows[-1][5], picks)
    try:
        plot(record, picks, {k: rows[-1][5][k].clamp(0, 1).numpy() for k in picks},
             {k: images[k].numpy() for k in picks})
    except ImportError:
        print(f"  matplotlib is absent, so no figures were drawn; the rows are "
              f"in {RESULTS.relative_to(ROOT)} and --replot will draw them")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replot", action="store_true",
                    help="redraw both figures from data/fit_scaling.json without refitting")
    ap.add_argument("--budgets", type=int, nargs="+", metavar="S",
                    help="override BUDGETS, in seconds; for exercising the path, "
                         "not for producing a result")
    args = ap.parse_args()
    if args.budgets:
        BUDGETS = tuple(args.budgets)
        REPEAT_BUDGET = BUDGETS[-1]
    if args.replot:
        replot()
    else:
        main()
