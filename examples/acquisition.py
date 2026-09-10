"""Phase 4 step 4b: acquiring lock without being told where the target is.

Every tracking run in this phase starts frame 0 at the truth. That is not a
detail: it means the tracking result assumes an acquisition step, and this is
the experiment that tests whether that step exists. Nothing here is warm
started. Orientations are drawn uniformly over SO(3), each is optimised on its
own, and where it lands is classified.

P4 predicted the silhouette tracker would flip and it did not, because a
tracker warm started from the previous frame steps a degree at a time and
never has to search as far as the flipped basin. Acquisition does have to. The
flip is rejected by the photometric loss by about 2 to 4x and by the
silhouette loss by only 1.1 to 1.3x, so the two should behave very differently
here, and that difference is the point of running both.

The result is reported as success against the number of initialisations,
because whether acquisition is possible at all is a weaker question than what
it costs.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fitting import load_model
from src.splatting import TorchCamera
from src.acquisition import (ACQUIRE_LR_ROT, ACQUIRE_LR_TRANS, flip_matrix,
                             register_from, uniform_so3)
from src.tracking import (axis_angle_to_matrix, matrix_to_axis_angle,
                          photometric_loss, register, silhouette_loss)

ROOT = Path(__file__).resolve().parent.parent
DT = torch.float32
ORIGIN = torch.zeros(3, dtype=DT)
PANEL_AXIS = np.array([0.0, 1.0, 0.0])
POOL = 40
ITERATIONS = 120
FRAME = 20
BASIN_DEG = 10.0


def load_sequence(name):
    d = np.load(ROOT / "data" / f"tumble_{name}.npz")
    c = d["camera"]
    holder = type("C", (), dict(width=int(d["width"]), height=int(d["height"]),
                                fx=float(d["fx"]), fy=float(d["fy"]),
                                cx=float(d["cx"]), cy=float(d["cy"]),
                                R=c[:9].reshape(3, 3), t=c[9:]))()
    return d, TorchCamera.from_raytrace(holder, dtype=DT)


def geodesic(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0)))


def flipped(R):
    return R @ flip_matrix()


def best_of_k(losses, correct, k, trials, rng):
    """Probability that the lowest loss among `k` restarts is in the true basin.

    Sampling with replacement from the pool, which assumes the restarts are
    independent draws from the same distribution. They are: each starts from
    its own uniform SO(3) sample and shares nothing with the others.
    """
    idx = rng.integers(0, len(losses), size=(trials, k))
    winner = idx[np.arange(trials), np.argmin(losses[idx], axis=1)]
    return float(correct[winner].mean())


def run(loss_name, loss_fn, target, camera, truth, model, rng):
    starts = uniform_so3(POOL, rng)
    spread = np.array([geodesic(R, truth) for R in starts])
    print(f"\n{loss_name}: {POOL} initialisations drawn uniformly over SO(3)")
    print(f"  start distance to truth: median {np.median(spread):.1f} deg, "
          f"min {spread.min():.1f}, max {spread.max():.1f}")
    if np.median(spread) < 90.0 or spread.min() > 60.0:
        raise SystemExit("the initialisations are clustered near the truth, so this "
                         "measures refinement rather than acquisition")

    losses, finals = [], []
    for R0 in starts:
        R, loss = register_from(model, camera, target, R0, loss_fn, ITERATIONS,
                                ACQUIRE_LR_ROT, ACQUIRE_LR_TRANS, centre=ORIGIN)
        finals.append(R)
        losses.append(loss)
    losses = np.array(losses)

    to_truth = np.array([geodesic(R, truth) for R in finals])
    to_flip = np.array([geodesic(R, flipped(truth)) for R in finals])
    correct = to_truth < BASIN_DEG
    flip = (~correct) & (to_flip < BASIN_DEG)
    other = ~(correct | flip)

    print(f"  where {POOL} independent runs landed")
    print(f"    true basin, within {BASIN_DEG:.0f} deg   {int(correct.sum()):3d} "
          f"({100*correct.mean():.0f}%)")
    print(f"    the 180 degree flip           {int(flip.sum()):3d} "
          f"({100*flip.mean():.0f}%)")
    print(f"    somewhere else                {int(other.sum()):3d} "
          f"({100*other.mean():.0f}%)")

    for label, sel in (("true basin", correct), ("flip", flip), ("other", other)):
        if sel.sum():
            print(f"    final loss, {label:<11} median {np.median(losses[sel]):.4e}, "
                  f"best {losses[sel].min():.4e}")
    if correct.sum():
        ranks = (losses < losses[correct].min()).sum()
        print(f"    runs scoring below the best correct run: {int(ranks)} "
              f"{'(so lowest loss picks the truth)' if ranks == 0 else '(so lowest loss does NOT pick the truth)'}")
    return losses, correct, flip


def main():
    torch.manual_seed(21)
    rng = np.random.default_rng(21)
    model = load_model(ROOT / "data" / "model.pt", dtype=DT)
    d, camera = load_sequence("short")
    truth = d["attitudes"][FRAME]

    print("acquisition without a warm start")
    print(f"  frame {FRAME} of the short sequence, body frame sun at "
          f"{d['sun_angle_deg'][FRAME]:.1f} deg, {len(model)} Gaussians")
    print("  every tracking run in this phase starts frame 0 at the truth, so the")
    print("  tracking result assumes this step. This is the test of it.")

    results = {}
    for name, fn, key in (("photometric", photometric_loss, "images"),
                          ("silhouette", silhouette_loss, "masks")):
        target = torch.as_tensor(d[key], dtype=DT)[FRAME]
        results[name] = run(name, fn, target, camera, truth, model, rng)

    print("\nwhat acquisition costs: probability the best of k restarts is correct")
    print(f"  {'restarts':>9} " + "".join(f"{n:>16}" for n in results))
    boot = np.random.default_rng(7)
    for k in (1, 2, 4, 8, 16, 32):
        line = f"  {k:9d} "
        for name in results:
            losses, correct, _ = results[name]
            line += f"{100*best_of_k(losses, correct, k, 4000, boot):15.0f}%"
        print(line)
    print(f"\n  each restart is {ITERATIONS} iterations; the cost of a given "
          f"success rate is that\n  number times the restarts needed to reach it")


if __name__ == "__main__":
    main()
