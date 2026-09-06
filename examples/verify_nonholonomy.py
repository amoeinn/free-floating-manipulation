"""Phase 2c: a closed loop in joint space leaves the base rotated.

On a fixed base, returning every joint to its starting angle returns the
gripper to its starting pose. On a free-floating base it does not: the base
attitude depends on the whole history of arm motion, so a closed joint-space
loop produces a net base rotation. That rotation is the nonholonomy, and it
is why a fixed-base planner is invalid here.

The rotation has to be shown to be real, not the simulator's integration
drift. Two properties a drift artifact does not have:

  - it is independent of the timestep. Halving dt changes the measured
    rotation by less each time; it converges to a non-zero limit.
  - it reverses sign when the loop is traversed backward. R_backward is
    R_forward transposed, so composing the two returns almost exactly to
    the identity.

Momentum is also checked directly, and the analytic model is integrated
along the same path, before the loop amplitude and bus inertia are swept to
produce the attitude-excursion figure.

The mechanics live in `src/freeflight.py`; this script is the report.
Run `pytest tests/` for the same invariants as assertions.

Usage:
    python examples/verify_nonholonomy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import torch

from src.dynamics import FloatingBaseModel
from src.freeflight import (ARM_JOINTS, BUS_GYRATION_SQUARED, JointLoop,
                            build_model, load_panda,
                            disable_damping, integrate_base_rotation,
                            rotation_angle, set_bus, simulate_loop)

LOOP = JointLoop()
DTYPE = torch.float64
FIGURE = Path(__file__).resolve().parent.parent / "docs" / "attitude_excursion.png"


def part_nominal(body: int) -> bool:
    print("part 1: the nominal loop, and momentum along it")
    coarse = simulate_loop(body, LOOP, dt=1e-3, watch_momentum=True)
    fine = simulate_loop(body, LOOP, dt=2.5e-4, watch_momentum=True)
    vector = fine["vector"]
    print(f"  joints {LOOP.joints} swing +/-{LOOP.amplitude} rad and return")
    print(f"  net base rotation: {np.degrees(fine['angle']):.4f} deg "
          f"about [{vector[0]:+.3f} {vector[1]:+.3f} {vector[2]:+.3f}]")
    print(f"  joint closure: {fine['closure']:.2e} rad")

    # Motor torques are internal, so momentum should be exactly zero. What
    # is left is the stepper: the residual falls in proportion to dt, and
    # more solver iterations do not touch it.
    ratio = coarse["momentum"] / fine["momentum"]
    print(f"  peak momentum, dt=1.0e-3: {coarse['momentum']:.2e} of its scale")
    print(f"  peak momentum, dt=2.5e-4: {fine['momentum']:.2e}")
    print(f"  falls by {ratio:.1f}x for a 4x smaller step -> "
          f"{'first order in dt, conserved in the limit' if 3 < ratio < 5 else 'NOT A DISCRETISATION RESIDUAL'}\n")
    return 3 < ratio < 5


def part_timestep(body: int) -> bool:
    print("part 2: independent of timestep (a drift artifact would not be)")
    timesteps = [2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4]
    angles = []
    print(f"  {'dt':>10}  {'net rotation (deg)':>18}  {'change':>10}")
    for dt in timesteps:
        angle = np.degrees(simulate_loop(body, LOOP, dt=dt)["angle"])
        change = "" if not angles else f"{angle - angles[-1]:+.4f}"
        angles.append(angle)
        print(f"  {dt:>10.2e}  {angle:>18.4f}  {change:>10}")

    increments = np.abs(np.diff(angles))
    shrinking = bool(np.all(increments[1:] < increments[:-1]))
    spread = abs(angles[-1] - angles[0]) / abs(angles[-1])
    richardson = angles[-1] + (angles[-1] - angles[-2])
    passed = shrinking and spread < 0.02 and abs(richardson) > 1.0
    print(f"  increments shrink each halving: {shrinking}")
    print(f"  coarse-to-fine spread: {spread * 100:.2f}% of the value")
    print(f"  extrapolated limit: {richardson:.3f} deg, non-zero")
    print(f"  -> {'converges, not drift' if passed else 'LOOKS LIKE DRIFT'}\n")
    return passed


def part_reversal(body: int) -> bool:
    print("part 3: reverses sign when the loop runs backward")
    forward = simulate_loop(body, LOOP, dt=2.5e-4, direction=1)
    backward = simulate_loop(body, LOOP, dt=2.5e-4, direction=-1)

    residual = np.degrees(rotation_angle(forward["rotation"]
                                         @ backward["rotation"]))
    single = np.degrees(forward["angle"])
    ahead, behind = forward["vector"], backward["vector"]
    alignment = (np.dot(ahead, behind)
                 / (np.linalg.norm(ahead) * np.linalg.norm(behind)))

    print(f"  forward:  {single:.4f} deg about "
          f"[{ahead[0]:+.3f} {ahead[1]:+.3f} {ahead[2]:+.3f}]")
    print(f"  backward: {np.degrees(backward['angle']):.4f} deg about "
          f"[{behind[0]:+.3f} {behind[1]:+.3f} {behind[2]:+.3f}]")
    print(f"  axis alignment forward.backward: {alignment:+.3f} (want -1)")
    print(f"  R_forward R_backward is {residual:.4f} deg from identity, "
          f"vs {single:.2f} deg for one loop")
    passed = alignment < -0.98 and residual < 0.1 * single
    print(f"  -> {'sign reverses, not drift' if passed else 'DOES NOT REVERSE'}\n")
    return passed


def part_analytic(body: int) -> bool:
    print("part 4: analytic -H_b^-1 H_bm qdot integrated vs the simulation")
    model = build_model(body, dtype=DTYPE)
    truth_angle = np.degrees(simulate_loop(body, LOOP, dt=1.25e-4)["angle"])

    print(f"  {'frame':>8}  {'samples':>8}  {'net rotation (deg)':>18}  "
          f"{'vs sim':>10}")
    best = None
    for frame in ("body", "world"):
        for samples in (100, 400):
            angle = np.degrees(rotation_angle(
                integrate_base_rotation(model, LOOP, samples, frame)))
            error = abs(angle - truth_angle) / truth_angle
            print(f"  {frame:>8}  {samples:>8}  {angle:>18.4f}  {error:>9.2%}")
            if samples == 400 and (best is None or error < best[1]):
                best = (frame, error)

    passed = best[1] < 0.02
    print(f"  sim: {truth_angle:.4f} deg")
    print(f"  -> base twist integrates in the {best[0]} frame, "
          f"{best[1]:.2%} from the simulation "
          f"-> {'model matches' if passed else 'MODEL DISAGREES'}\n")
    return passed


def part_sweep(body: int) -> bool:
    print("part 5: amplitude and bus inertia sweep")
    amplitudes = np.array([0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
    bus_masses = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0])

    excursion = np.zeros((len(bus_masses), len(amplitudes)))
    for row, mass in enumerate(bus_masses):
        set_bus(body, float(mass))
        for column, amplitude in enumerate(amplitudes):
            loop = JointLoop(amplitude=float(amplitude))
            excursion[row, column] = np.degrees(
                simulate_loop(body, loop, dt=1e-3)["angle"])

    print(f"  {'bus kg':>8}  " + "  ".join(f"A={a:<4}" for a in amplitudes))
    for row, mass in enumerate(bus_masses):
        cells = "  ".join(f"{value:6.3f}" for value in excursion[row])
        print(f"  {mass:>8.0f}  {cells}")

    small = slice(0, 3)
    amplitude_slope = np.polyfit(np.log(amplitudes[small]),
                                 np.log(excursion[-1, small]), 1)[0]
    mass_slope = np.polyfit(np.log(bus_masses[-3:]),
                            np.log(excursion[-3:, 3]), 1)[0]
    passed = 1.7 < amplitude_slope < 2.3 and -1.2 < mass_slope < -0.8
    print(f"  small-amplitude log-log slope: {amplitude_slope:.2f} (expect 2)")
    print(f"  large-bus log-log slope:       {mass_slope:.2f} (expect -1)")
    print(f"  -> {'geometric phase, area x 1/inertia' if passed else 'UNEXPECTED SCALING'}")

    _write_figure(amplitudes, bus_masses, excursion)
    return passed


def _write_figure(amplitudes, bus_masses, excursion) -> None:
    FIGURE.parent.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(bus_masses)))
    for row, mass in enumerate(bus_masses):
        axis.loglog(amplitudes, excursion[row], "o-", color=colours[row],
                    label=f"{mass:.0f} kg bus")

    # Guide line, offset clear of the data: small loops enclose area ~ A^2.
    reference = 0.4 * excursion[-1, 0] * (amplitudes / amplitudes[0]) ** 2
    axis.loglog(amplitudes, reference, "k:", lw=1.2,
                label=r"slope 2 ($\propto$ enclosed area)")

    axis.set_xlabel("loop amplitude (rad, each of two joints)")
    axis.set_ylabel("net base rotation after one closed loop (deg)")
    axis.set_title("A closed joint-space loop rotates the free-flying base")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, loc="upper left")
    axis.text(0.98, 0.03,
              f"bus inertia $m({np.sqrt(BUS_GYRATION_SQUARED)}\\,"
              "\\mathrm{m})^2$; curves bend below slope 2\n"
              "past ~0.5 rad, where the loop is no longer small",
              transform=axis.transAxes, ha="right", va="bottom", fontsize=7.5,
              color="0.35")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=130)
    plt.close(figure)
    print(f"  wrote {FIGURE.relative_to(Path.cwd())}\n")


def main() -> None:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    body = load_panda(fixed_base=False)
    disable_damping(body)

    ok_momentum = part_nominal(body)
    ok_dt = part_timestep(body)
    ok_reverse = part_reversal(body)
    ok_analytic = part_analytic(body)
    ok_sweep = part_sweep(body)

    print(f"summary: momentum {'ok' if ok_momentum else 'FAIL'}, "
          f"timestep {'ok' if ok_dt else 'FAIL'}, "
          f"reversal {'ok' if ok_reverse else 'FAIL'}, "
          f"analytic {'ok' if ok_analytic else 'FAIL'}, "
          f"sweep {'ok' if ok_sweep else 'FAIL'}")
    p.disconnect()


if __name__ == "__main__":
    main()
