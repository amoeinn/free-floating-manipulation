"""Fit one model and save it, for everything in step 4 to track against.

`fit_target.py` reports the scaling curve and throws its models away. This
keeps one. The budget defaults to the largest point on that curve, where the
held out object PSNR was 33.88 dB, because tracking should be limited by the
tracking rather than by a model nobody would have shipped.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.fitting import initialise, learning_rates, psnr, save_model
from src.splatting import render_tiled

ROOT = Path(__file__).resolve().parent.parent


def main(budget=480.0, n_gaussians=6000, seed=4):
    from examples.fit_target import load, HOLDOUT_EVERY, evaluate
    torch.manual_seed(seed)
    np.random.seed(seed)
    d, intr, cams = load()
    images = torch.as_tensor(d["images"], dtype=torch.float32)
    mask = images.max(-1).values > 1e-3
    parts = d["primitive"]
    names = [str(n) for n in d["primitive_names"]]
    holdout = list(range(0, len(cams), HOLDOUT_EVERY))
    train = [k for k in range(len(cams)) if k not in holdout]

    g, voxels = initialise(images, d["poses"], intr, n_gaussians,
                           resolution=128, seed=0)
    opt = torch.optim.Adam(learning_rates(g, scene_scale=2.0))
    start = time.perf_counter()
    for _ in range(20):
        k = int(np.random.choice(train))
        opt.zero_grad()
        (render_tiled(g, cams[k]) - images[k]).abs().mean().backward()
        opt.step()
    total = max(int(20 / (time.perf_counter() - start) * budget), 40)
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

    obj, full, per_part, _ = evaluate(g, images, cams, holdout, mask, parts, names)
    out = ROOT / "data" / "model.pt"
    save_model(g, out, meta={"budget_s": budget, "steps": steps,
                             "holdout_object_psnr": obj,
                             "holdout_frame_psnr": full,
                             "n_gaussians": len(g), "hull_voxels": voxels})
    print(f"fitted {len(g)} Gaussians in {steps} steps over {budget:.0f} s")
    print(f"  held out object PSNR {obj:.2f} dB, frame {full:.2f} dB")
    print("  per part: " + ", ".join(f"{n} {per_part[n]:.2f}" for n in names))
    print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
