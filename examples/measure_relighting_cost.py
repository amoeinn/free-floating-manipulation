"""What baked lighting costs once the target turns under a fixed sun.

The fitted model absorbed the shading it saw during the approach, when the sun
was fixed in the target's body frame. A tumbling target moves the sun in that
frame, and the model has no way to represent it. This measures the size of
that error as a function of how far the body frame sun has moved, which is
what decides how short a tracking window has to be.

Rotating the target by R under a fixed camera and sun is the same as leaving
it still and pre-multiplying the camera by R while carrying the sun into the
body frame as R^T s. That equivalence is what lets the exact ray tracer supply
ground truth at every angle without rebuilding the scene.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fitting import load_model, psnr
from src.raytrace import Camera, Scene
from src.splatting import TorchCamera, render_tiled
from src.target import client_satellite, client_scene
from src.tracking import axis_angle_to_matrix

ROOT = Path(__file__).resolve().parent.parent
DT = torch.float32
ANGLES = (0, 2, 5, 10, 15, 20, 30, 45, 60, 90)


def main():
    d = np.load(ROOT / "data" / "approach" / "views.npz")
    model = load_model(ROOT / "data" / "model.pt", dtype=DT)
    base = client_scene()
    sun = base.light_direction
    index = 30
    p = d["poses"][index]
    R_cw, t_cw = p[:9].reshape(3, 3), p[9:]
    intr = dict(width=int(d["width"]), height=int(d["height"]), fx=float(d["fx"]),
                fy=float(d["fy"]), cx=float(d["cx"]), cy=float(d["cy"]))

    axis = np.array([0.31, -0.62, 0.72])
    axis /= np.linalg.norm(axis)

    print("cost of a model with lighting baked in, as the body frame sun moves")
    print(f"  view {index} of the approach set, {intr['width']}x{intr['height']}, "
          f"{len(model)} Gaussians")
    print(f"  at zero the model was fitted, so the number there is the fit's own "
          f"quality\n")
    print(f"  {'sun moved':>10} {'object dB':>10} {'lost':>7} "
          f"{'lit pixels changed':>20}")

    reference = None
    base_lit = None
    for deg in ANGLES:
        R = axis_angle_to_matrix(torch.as_tensor(axis * np.deg2rad(deg),
                                                 dtype=torch.float64)).numpy()
        scene = Scene(primitives=client_satellite(),
                      light_direction=R.T @ sun,
                      light_intensity=base.light_intensity)
        cam = Camera(R=R_cw @ R, t=t_cw, **intr)
        truth = scene.render(cam)["image"]

        holder = type("C", (), dict(R=R_cw @ R, t=t_cw, **intr))()
        with torch.no_grad():
            pred = render_tiled(model, TorchCamera.from_raytrace(holder, dtype=DT))
        target = torch.as_tensor(truth, dtype=DT)
        mask = target.max(-1).values > 1e-3
        value = psnr(pred[mask], target[mask])
        lit = (target.max(-1).values > 1e-2)
        if reference is None:
            reference, base_lit = value, lit
        changed = 100 * (lit ^ base_lit).float().sum().item() / max(base_lit.sum().item(), 1)
        print(f"  {deg:8d} deg {value:10.2f} {reference - value:7.2f} "
              f"{changed:18.1f}%")

    print("\n  the sun angle is the rotation angle, because the body turns under a "
          "fixed sun")
    print("  this is the cost of not making the model relightable")


if __name__ == "__main__":
    main()
