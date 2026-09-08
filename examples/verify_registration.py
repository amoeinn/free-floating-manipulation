"""Phase 4 step 4, check 1: the round trip through the renderer.

Built before the tracker so that estimator bugs stay separable from model
mismatch. The image here is rendered from the same model the estimator
optimises against, so the data matches the model exactly and there is nothing
left for a failure to be except a bug. If this passes and tracking real frames
fails, the fault is the model or the lighting, not the optimiser, and that
split is worth having on the first day rather than the third.

Ground truth poses exist in this project, but a check that only compares
against them measures accuracy. This measures whether the estimator can find a
pose it is guaranteed to be able to represent.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fitting import load_model
from src.splatting import TorchCamera, render_tiled
from src.tracking import (axis_angle_to_matrix, photometric_loss, pose_error,
                          register, silhouette_loss, transform)

ROOT = Path(__file__).resolve().parent.parent
DT = torch.float32


def dataset_camera(index=30):
    d = np.load(ROOT / "data" / "approach" / "views.npz")
    p = d["poses"][index]
    holder = type("C", (), dict(width=int(d["width"]), height=int(d["height"]),
                                fx=float(d["fx"]), fy=float(d["fy"]),
                                cx=float(d["cx"]), cy=float(d["cy"]),
                                R=p[:9].reshape(3, 3), t=p[9:]))()
    return TorchCamera.from_raytrace(holder, dtype=DT)


def main():
    torch.manual_seed(9)
    model = load_model(ROOT / "data" / "model.pt", dtype=DT)
    camera = dataset_camera()
    centre = model.means.mean(dim=0)
    print(f"round trip through the renderer, {len(model)} Gaussians, "
          f"{camera.width}x{camera.height}")
    print("  the image is rendered from the model being fitted, so anything "
          "that fails here is a bug\n")

    cases = [
        ("2 deg, 10 mm", np.deg2rad(2.0), 0.010),
        ("5 deg, 25 mm", np.deg2rad(5.0), 0.025),
        ("10 deg, 50 mm", np.deg2rad(10.0), 0.050),
        ("20 deg, 80 mm", np.deg2rad(20.0), 0.080),
        ("35 deg, 120 mm", np.deg2rad(35.0), 0.120),
    ]
    zero = torch.zeros(3, dtype=DT)
    image, coverage = render_tiled(transform(model, zero, zero, centre), camera,
                                   return_coverage=True)
    image, coverage = image.detach(), coverage.detach()
    targets = {"photometric": image, "silhouette": coverage}

    # Absolute losses, not ratios. Both are exactly zero at the truth here,
    # because the observation is a render of the model, so a ratio against the
    # baseline divides by zero and reports whatever the floating point noise
    # happens to be. The first version of this printed 5.8e9 and looked like a
    # sensitivity.
    print("  sensitivity: the loss itself, at the truth and under perturbation")
    print(f"    {'perturbation':<16} {'photometric':>13} {'silhouette':>13}")
    rng = np.random.default_rng(3)
    perturbations = []
    with torch.no_grad():
        print(f"    {'at the truth':<16} "
              f"{photometric_loss(model, zero, zero, camera, image, centre).item():13.3e} "
              f"{silhouette_loss(model, zero, zero, camera, coverage, centre).item():13.3e}")
        for label, angle, shift in cases:
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            rot = axis * angle
            tr = rng.normal(size=3) / np.sqrt(3) * shift
            perturbations.append((label, rot, tr))
            rt = torch.as_tensor(rot, dtype=DT); tt = torch.as_tensor(tr, dtype=DT)
            p = photometric_loss(model, rt, tt, camera, image, centre).item()
            s_ = silhouette_loss(model, rt, tt, camera, coverage, centre).item()
            print(f"    {label:<16} {p:13.3e} {s_:13.3e}")
            if p <= 0 or s_ <= 0:
                raise SystemExit(f"the {label} perturbation left one loss at zero, "
                                 "so recovery from it would prove nothing")
    print()

    for name, loss_fn in (("photometric", photometric_loss),
                          ("silhouette", silhouette_loss)):
        print(f"  {name} recovery")
        print(f"    {'perturbation':<16} {'start err':>18} {'recovered':>18} "
              f"{'loss drop':>11}")
        for label, start_rot, start_trans in perturbations:
            start_err = pose_error(start_rot, start_trans, np.zeros(3), np.zeros(3))
            with torch.no_grad():
                loss0 = loss_fn(model, torch.as_tensor(start_rot, dtype=DT),
                                torch.as_tensor(start_trans, dtype=DT),
                                camera, targets[name], centre).item()
            rot, tr, trace = register(model, camera, targets[name], start_rot,
                                      start_trans, loss_fn, iterations=150,
                                      lr_rot=0.03, lr_trans=0.015, centre=centre)
            err = pose_error(rot.numpy(), tr.numpy(), np.zeros(3), np.zeros(3))
            print(f"    {label:<16} {start_err[0]:7.2f} deg {start_err[1]:6.1f} mm"
                  f" {err[0]:8.3f} deg {err[1]:6.2f} mm "
                  f"{loss0/max(trace[-1],1e-12):10.1f}x")
        print()


if __name__ == "__main__":
    main()
