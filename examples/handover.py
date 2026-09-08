"""Phase 4 step 4c: photometric acquisition handed over to silhouette hold.

The composition the phase implies and had never run end to end. Photometric
acquires and cannot hold a long window, because the baked model decays as the
body frame sun sweeps. Silhouette holds indefinitely and cannot acquire,
because its global minimum is not the true pose. Neither alone is a tracker.

Run in two stages so each stays inside the working runtime budget:

    python examples/handover.py acquire
    python examples/handover.py track

The failure mode that matters is the one silhouette cannot report. Once
handed a wrong pose it will hold that pose exactly as stably as a right one,
so an acquisition that lands on the 180 degree flip produces a confident,
smooth, wrong track. The only place to catch it is at the handover, with the
photometric loss still available, and the check is one extra render: evaluate
the loss at the acquired pose and at its flip and keep the lower. If the
acquisition was correct the flip should score 2 to 4x worse; if it landed on
the flip, the test repairs it.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fitting import load_model
from src.splatting import TorchCamera
from src.tracking import (axis_angle_to_matrix, matrix_to_axis_angle,
                          photometric_loss, register, silhouette_loss)
from src.tumble import skew

ROOT = Path(__file__).resolve().parent.parent
DT = torch.float32
ORIGIN = torch.zeros(3, dtype=DT)
PANEL_AXIS = np.array([0.0, 1.0, 0.0])
TRIALS = 10
RESTARTS = 8          # the acquisition experiment says 8 gives 87 percent
ACQ_ITERATIONS = 120
TRACK_ITERATIONS = 25
BASIN_DEG = 10.0
STORE = ROOT / "data" / "handover.npz"

# Silhouette tracking warm started from the truth, measured in the previous
# run and quoted here as the bar this has to match to be worth anything.
BASELINE_WORST = 4.67
BASELINE_FINAL = 1.21


def load_sequence(name="long"):
    d = np.load(ROOT / "data" / f"tumble_{name}.npz")
    c = d["camera"]
    holder = type("C", (), dict(width=int(d["width"]), height=int(d["height"]),
                                fx=float(d["fx"]), fy=float(d["fy"]),
                                cx=float(d["cx"]), cy=float(d["cy"]),
                                R=c[:9].reshape(3, 3), t=c[9:]))()
    return d, TorchCamera.from_raytrace(holder, dtype=DT)


def geodesic(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0)))


def flip_matrix():
    K = skew(PANEL_AXIS)
    return np.eye(3) + 2 * K @ K


def uniform_so3(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def classify(R, truth):
    if geodesic(R, truth) < BASIN_DEG:
        return "correct"
    if geodesic(R, truth @ flip_matrix()) < BASIN_DEG:
        return "flip"
    return "other"


def acquire():
    torch.manual_seed(31)
    rng = np.random.default_rng(31)
    model = load_model(ROOT / "data" / "model.pt", dtype=DT)
    d, camera = load_sequence()
    truth = d["attitudes"][0]
    image = torch.as_tensor(d["images"], dtype=DT)[0]

    print(f"acquisition, {TRIALS} independent trials of {RESTARTS} restarts each")
    print(f"  frame 0 of the long sequence, photometric loss, "
          f"{ACQ_ITERATIONS} iterations per restart")
    print(f"  {RESTARTS} restarts is what the acquisition experiment says buys "
          f"87 percent\n")
    print(f"  {'trial':>5} {'landed':>8} {'err':>9} {'loss':>11} "
          f"{'flip test':>11} {'after test':>11} {'err':>9}")

    rows = []
    for trial in range(TRIALS):
        best_loss, best_rot = np.inf, None
        for R0 in uniform_so3(RESTARTS, rng):
            rot, tr, trace = register(model, camera, image, matrix_to_axis_angle(R0),
                                      np.zeros(3), photometric_loss,
                                      iterations=ACQ_ITERATIONS, lr_rot=0.08,
                                      lr_trans=0.02, centre=ORIGIN)
            if trace[-1] < best_loss:
                best_loss, best_rot = trace[-1], rot.numpy()

        R_acq = axis_angle_to_matrix(
            torch.as_tensor(best_rot, dtype=torch.float64)).numpy()
        before = classify(R_acq, truth)

        # One extra render: is the flipped pose a better explanation?
        R_alt = R_acq @ flip_matrix()
        rot_alt = matrix_to_axis_angle(R_alt)
        with torch.no_grad():
            loss_alt = photometric_loss(model, torch.as_tensor(rot_alt, dtype=DT),
                                        ORIGIN, camera, image, ORIGIN).item()
        ratio = loss_alt / max(best_loss, 1e-12)
        if ratio < 1.0:
            best_rot, R_acq = rot_alt, R_alt
        after = classify(R_acq, truth)
        rows.append((best_rot, before, after, best_loss, ratio))
        print(f"  {trial:5d} {before:>8} {geodesic(R_acq, truth):8.2f}d "
              f"{best_loss:11.4e} {ratio:10.2f}x {after:>11} "
              f"{geodesic(R_acq, truth):8.2f}d")

    np.savez(STORE, poses=np.array([r[0] for r in rows]),
             before=np.array([r[1] for r in rows]),
             after=np.array([r[2] for r in rows]),
             loss=np.array([r[3] for r in rows]),
             ratio=np.array([r[4] for r in rows]))

    before = np.array([r[1] for r in rows])
    after = np.array([r[2] for r in rows])
    print(f"\n  before the flip test: " + ", ".join(
        f"{k} {int((before == k).sum())}" for k in ("correct", "flip", "other")))
    print(f"  after the flip test:  " + ", ".join(
        f"{k} {int((after == k).sum())}" for k in ("correct", "flip", "other")))
    ratios = np.array([r[4] for r in rows])
    for k in ("correct", "flip", "other"):
        sel = before == k
        if sel.sum():
            print(f"    flip test ratio when acquisition landed {k:<8} "
                  f"median {np.median(ratios[sel]):.2f}x, "
                  f"range {ratios[sel].min():.2f} to {ratios[sel].max():.2f}")
    print(f"\n  wrote {STORE.name}; now run: python examples/handover.py track")


def track():
    torch.manual_seed(32)
    model = load_model(ROOT / "data" / "model.pt", dtype=DT)
    d, camera = load_sequence()
    truth = d["attitudes"]
    masks = torch.as_tensor(d["masks"], dtype=DT)
    dt = float(d["dt"])
    store = np.load(STORE, allow_pickle=False)
    poses, after = store["poses"], store["after"]

    # One representative of each observed outcome, the truth as the bar, and
    # a deliberately flipped start. No trial landed on the flip, so that case
    # is constructed rather than observed and is labelled as such: leaving it
    # out would leave the failure mode this experiment exists to probe
    # untested for want of a sample.
    picks = []
    for kind in ("correct", "other", "flip"):
        idx = np.where(after == kind)[0]
        if idx.size:
            best = idx[np.argmin(store["loss"][idx])]
            picks.append((f"acquired, {kind}", poses[best]))
    picks.append(("constructed flip", matrix_to_axis_angle(truth[0] @ flip_matrix())))
    picks.append(("started at the truth", matrix_to_axis_angle(truth[0])))

    print(f"handover: silhouette tracking over {len(truth)} frames, "
          f"{TRACK_ITERATIONS} iterations per frame")
    print(f"  the bar is silhouette warm started from the truth: worst "
          f"{BASELINE_WORST:.2f} deg, final {BASELINE_FINAL:.2f} deg\n")
    print(f"  {'start':<24} {'worst':>9} {'final':>9} {'median':>9} "
          f"{'past 20 deg':>12}")

    for label, rot0 in picks:
        rot, tr = np.asarray(rot0, float), np.zeros(3)
        errors = []
        for k in range(len(truth)):
            if k > 0:
                rot_t, tr_t, _ = register(model, camera, masks[k], rot, tr,
                                          silhouette_loss,
                                          iterations=TRACK_ITERATIONS,
                                          lr_rot=0.02, lr_trans=0.010, centre=ORIGIN)
                rot, tr = rot_t.numpy(), tr_t.numpy()
            R = axis_angle_to_matrix(torch.as_tensor(rot, dtype=torch.float64)).numpy()
            errors.append(geodesic(R, truth[k]))
        e = np.array(errors)
        print(f"  {label:<24} {e.max():8.2f}d {e[-1]:8.2f}d {np.median(e):8.2f}d "
              f"{int((e > 20).sum()):8d}/{len(e)}")

    print("\n  a wrong start is held exactly as stably as a right one, which is "
          "the point:\n  silhouette tracking cannot report that it is tracking the "
          "wrong pose")

    print("\n  detectability at the handover, from the acquisition stage")
    loss, before = store["loss"], store["before"]
    err_tight = loss < 1.0e-02
    print(f"    acquisition loss splits into two clusters: "
          f"{int(err_tight.sum())} runs at "
          f"{loss[err_tight].min():.3e} to {loss[err_tight].max():.3e}, "
          f"{int((~err_tight).sum())} at {loss[~err_tight].min():.3e} to "
          f"{loss[~err_tight].max():.3e}")
    print("    so the loss value itself, not the flip test, is what separates a "
          "converged\n    acquisition from a near miss. The flip test ratios for "
          "correct and other\n    overlap at 2.81x against 2.80x and cannot tell "
          "those two apart; it is only\n    diagnostic for the flip, where it "
          "would come back below 1.0.")


def detect():
    """Where the flip test is valid, tested on true positives.

    The ten acquisition trials never landed on the flip, so the repair path
    was never exercised on a case it should fire for. Claiming a detector
    works because it stayed quiet on negatives is not evidence. This runs it
    on a deliberately flipped pose at several points along the sequence.
    """
    model = load_model(ROOT / "data" / "model.pt", dtype=DT)
    d, camera = load_sequence()
    truth, images = d["attitudes"], torch.as_tensor(d["images"], dtype=DT)
    print("the flip test on true positives, and where it stops working")
    print("  ratio = photometric loss at the flipped pose over the loss at the "
          "pose held")
    print("  above 1 keeps the pose, below 1 repairs it to the flip\n")
    print(f"  {'frame':>5} {'sun':>6} {'held pose':>12} {'ratio':>8} {'verdict':>9}")
    for k in (0, 24, 48, 72, 96):
        for label, R in (("correct", truth[k]),
                         ("flipped", truth[k] @ flip_matrix())):
            with torch.no_grad():
                a = photometric_loss(model, torch.as_tensor(
                    matrix_to_axis_angle(R), dtype=DT), ORIGIN, camera,
                    images[k], ORIGIN).item()
                b = photometric_loss(model, torch.as_tensor(
                    matrix_to_axis_angle(R @ flip_matrix()), dtype=DT), ORIGIN,
                    camera, images[k], ORIGIN).item()
            r = b / max(a, 1e-12)
            print(f"  {k:5d} {float(d['sun_angle_deg'][k]):5.1f} {label:>12} "
                  f"{r:7.2f}x {'keep' if r >= 1 else 'REPAIR':>9}")
    print("\n  the test is decisive at acquisition, 3.84x against 0.26x with the "
          "sun where\n  the model was fitted, weakens to 1.69x against 0.59x by 15 "
          "degrees, is exactly\n  ambiguous at 28, and inverts beyond it: past 40 "
          "degrees it would repair a\n  correct pose into the flip. It is a "
          "handover time check and nothing else.")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "acquire"
    {"acquire": acquire, "track": track, "detect": detect}[stage]()
