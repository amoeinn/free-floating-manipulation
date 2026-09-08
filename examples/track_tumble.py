"""Phase 4 step 4: per frame registration, then a filter, on a tumbling target.

Two windows, for the reason recorded in PLAN.md. The photometric tracker runs
over 8 s, where the baked model is still worth something. The silhouette
tracker runs over 48 s, where the body frame sun has swept 50 degrees and the
photometric model is worthless, but where the polhode has travelled 0.47 of
its own magnitude and the Euler coupling is genuinely present.

The filter is graded on prediction error against lead time, not on tracking
error at the current frame, because a rendezvous needs where the fixture will
be on arrival. Three propagators are compared and the comparison is the whole
point: holding still, extrapolating at a constant body rate, and integrating
Euler's equations. If Euler does not beat constant rate, the dynamics are not
being exercised whatever the polhode says.
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
from src.tumble import integrate, skew

ROOT = Path(__file__).resolve().parent.parent
DT = torch.float32
ORIGIN = torch.zeros(3, dtype=DT)
PANEL_AXIS = np.array([0.0, 1.0, 0.0])


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
    """`R` composed with a 180 degree rotation about the panel axis."""
    K = skew(PANEL_AXIS)
    return R @ (np.eye(3) + 2 * K @ K)


def track(name, loss_name, loss_fn, iterations, lr_rot, lr_trans):
    d, camera = load_sequence(name)
    model = load_model(ROOT / "data" / "model.pt", dtype=DT)
    truth = d["attitudes"]
    sun = d["sun_angle_deg"]
    targets = (torch.as_tensor(d["images"], dtype=DT) if loss_name == "photometric"
               else torch.as_tensor(d["masks"], dtype=DT))

    print(f"\n{loss_name} tracking over the {name} window, {len(truth)} frames, "
          f"dt {float(d['dt']):.2f} s")
    print("  frame 0 is initialised at the truth; acquisition is not part of "
          "this and is\n  a separate problem")

    rot = np.zeros(3)
    tr = np.zeros(3)
    estimates, errors, flip_errors = [], [], []
    for k in range(len(truth)):
        if k > 0:
            rot_t, tr_t, _ = register(model, camera, targets[k], rot, tr, loss_fn,
                                      iterations=iterations, lr_rot=lr_rot,
                                      lr_trans=lr_trans, centre=ORIGIN)
            rot, tr = rot_t.numpy(), tr_t.numpy()
        R_est = axis_angle_to_matrix(torch.as_tensor(rot, dtype=torch.float64)).numpy()
        estimates.append(R_est)
        errors.append(geodesic(R_est, truth[k]))
        flip_errors.append(geodesic(R_est, flipped(truth[k])))
    return d, np.array(estimates), np.array(errors), np.array(flip_errors), sun


def report_track(name, errors, flip_errors, sun, dt):
    print(f"  {'time':>7} {'sun moved':>10} {'attitude err':>13} {'if flipped':>12}"
          f"  {'':<8}")
    step = max(len(errors) // 10, 1)
    for k in range(0, len(errors), step):
        tag = ""
        if flip_errors[k] < errors[k] - 5.0:
            tag = "FLIPPED"
        print(f"  {k*dt:6.1f}s {sun[k]:9.1f} {errors[k]:12.3f} deg "
              f"{flip_errors[k]:10.2f} deg  {tag}")
    lost = errors > 20.0
    flips = flip_errors < errors - 5.0
    print(f"  worst attitude error {errors.max():.2f} deg, final "
          f"{errors[-1]:.2f} deg")
    print(f"  frames past 20 deg: {int(lost.sum())} of {len(errors)}; "
          f"frames closer to the flipped pose: {int(flips.sum())}")
    return flips.any(), errors


def body_rate(estimates, k, dt, window):
    """Body rate at frame `k`, averaged over the preceding `window` steps.

    A two frame difference was the first version and it does not work. The
    attitude estimates carry a degree or two of error and the step is half a
    second, so a two frame difference produces a rate whose error is several
    deg/s against a true rate of 2.7: the propagator choice then cannot
    matter, because nothing downstream of that estimate is signal. Averaging
    trades that noise against the bias of assuming the rate is constant over
    the window, and the sweep below reports both sides of the trade.
    """
    lo = max(k - window, 0)
    steps = [matrix_to_axis_angle(estimates[j].T @ estimates[j + 1]) / dt
             for j in range(lo, k)]
    if not steps:
        return np.zeros(3)
    return np.mean(steps, axis=0)


def propagate(R, omega, dt, steps, inertia, mode):
    if mode == "hold":
        return R
    if mode == "constant rate":
        theta = np.linalg.norm(omega) * dt * steps
        if theta < 1e-12:
            return R
        K = skew(omega / np.linalg.norm(omega))
        return R @ (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K)
    att, _ = integrate(inertia, omega, dt=dt, steps=steps, R0=R)
    return att[-1]


def prediction_curve(estimates, truth, dt, inertia, leads, window):
    n = len(estimates)
    rows = {"hold": [], "constant rate": [], "Euler": []}
    for lead in leads:
        ahead = int(round(lead / dt))
        if ahead < 1 or ahead >= n - window - 1:
            continue
        errs = {k: [] for k in rows}
        for k in range(window, n - ahead):
            omega = body_rate(estimates, k, dt, window)
            target = truth[k + ahead]
            for mode in rows:
                errs[mode].append(geodesic(
                    propagate(estimates[k], omega, dt, ahead, inertia, mode), target))
        for key in rows:
            rows[key].append((lead, float(np.median(errs[key]))))
    return rows


def rate_quality(estimates, truth_rates, dt, window):
    """How far the smoothed rate is from the truth, in deg/s.

    Reported because the prediction curve is only informative once this is
    below the rate itself. It is a diagnostic, not an input: no propagator
    below ever sees the truth.
    """
    errs = []
    for k in range(window, len(estimates)):
        omega = body_rate(estimates, k, dt, window)
        # The tracker's rate is in the world frame; the truth is in the body.
        truth_world = truth_rates[k] @ np.asarray(estimates[k]).T
        errs.append(np.degrees(np.linalg.norm(omega - truth_world)))
    return float(np.median(errs))


def main():
    torch.manual_seed(11)

    d_s, est_s, err_s, flip_s, sun_s = track(
        "short", "photometric", photometric_loss, 40, 0.02, 0.010)
    report_track("short", err_s, flip_s, sun_s, float(d_s["dt"]))

    print("\n  degradation against body frame sun angle, which is what the baked "
          "model costs")
    bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 11)]
    print(f"    {'sun moved':<14} {'frames':>7} {'median err':>12} {'worst':>10}")
    for lo, hi in bins:
        sel = (sun_s >= lo) & (sun_s < hi)
        if sel.sum() == 0:
            continue
        print(f"    {f'{lo} to {hi} deg':<14} {int(sel.sum()):7d} "
              f"{np.median(err_s[sel]):9.3f} deg {err_s[sel].max():7.3f} deg")

    d_l, est_l, err_l, flip_l, sun_l = track(
        "long", "silhouette", silhouette_loss, 40, 0.02, 0.010)
    any_flip, _ = report_track("long", err_l, flip_l, sun_l, float(d_l["dt"]))

    print("\n  P4 said the silhouette tracker is the one that should flip.")
    print(f"  It {'did' if any_flip else 'did not'}.")

    if err_l.max() > 20.0:
        print("\n  the silhouette tracker did not hold lock over the long window, so "
              "neither\n  method exercises the tri-axial dynamics on this target and "
              "relighting is a\n  prerequisite for the question rather than an "
              "enhancement")
        return

    np.savez_compressed(ROOT / "data" / "track_estimates.npz",
                        short=est_s, long=est_l, err_short=err_s, err_long=err_l,
                        sun_short=sun_s, sun_long=sun_l)

    print("\n  the body rate, smoothed over a window, against the truth")
    print("  the prediction curve below is meaningless until this is well under "
          "the 2.7 deg/s\n  the target is actually turning")
    dt_l = float(d_l["dt"])
    truth_rates = d_l["rates"]
    print(f"    {'window':>8} {'span':>8} {'rate error':>13}")
    for window in (1, 2, 4, 8, 16, 24):
        q = rate_quality(est_l, truth_rates, dt_l, window)
        print(f"    {window:6d} f {window*dt_l:6.1f}s {q:10.3f} deg/s")

    print("\n  filter prediction error against lead time, long window")
    print("  the body rate comes from the tracker's own estimates, never from "
          "the truth")
    leads = [1.0, 2.0, 4.0, 8.0, 16.0]
    for window in (2, 8, 16):
        rows = prediction_curve(est_l, d_l["attitudes"], dt_l, d_l["inertia"],
                                leads, window)
        print(f"\n    rate smoothed over {window} frames ({window*dt_l:.1f} s), "
              f"rate error {rate_quality(est_l, truth_rates, dt_l, window):.3f} deg/s")
        print(f"    {'lead':>7} " + "".join(f"{k:>16}" for k in rows))
        for i in range(len(rows["hold"])):
            line = f"    {rows['hold'][i][0]:6.1f}s "
            for key in rows:
                line += f"{rows[key][i][1]:13.2f} deg"
            print(line)
        gain = [c[1] - e[1] for c, e in zip(rows["constant rate"], rows["Euler"])]
        print(f"    Euler beats constant rate by {min(gain):+.2f} to "
              f"{max(gain):+.2f} deg")


if __name__ == "__main__":
    main()
