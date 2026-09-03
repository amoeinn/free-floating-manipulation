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

Momentum is also checked directly: total linear and angular momentum stay
at zero through the maneuver, since the motor torques are internal.

Then the analytic model (-H_b^-1 H_bm qdot, integrated) is checked against
the simulation over the whole loop, and the loop amplitude and bus inertia
are swept to produce the attitude-excursion figure.

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

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
LOOP_JOINTS = (1, 2)          # panda_joint2, panda_joint3: the arm's heavy pair
FINGER_JOINTS = [9, 10]
HOME = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0])
PERIOD = 2.0
DEFAULT_AMPLITUDE = 0.6
BUS_GYRATION_SQUARED = 0.25   # (0.5 m)^2, a metre-scale bus, not the Panda puck
FIGURE = Path(__file__).resolve().parent.parent / "docs" / "attitude_excursion.png"
DTYPE = torch.float64


# ------------------------------------------------------------- the joint loop

def loop_reference(phase: float, amplitude: float):
    """Joint angles on the loop at loop phase `phase` (radians).

    joint a traces a sine, joint b a raised cosine, so the pair walks a
    circle in its own plane and the path encloses area. Both angles and
    both rates are periodic, so the loop closes in position and velocity.
    """
    angles = HOME.copy()
    a, b = LOOP_JOINTS
    angles[a] = HOME[a] + amplitude * np.sin(phase)
    angles[b] = HOME[b] + amplitude * (1.0 - np.cos(phase))
    return angles


def loop_rates(phase: float, amplitude: float, phase_rate: float):
    rates = np.zeros(len(ARM_JOINTS))
    a, b = LOOP_JOINTS
    rates[a] = amplitude * np.cos(phase) * phase_rate
    rates[b] = amplitude * np.sin(phase) * phase_rate
    return rates


# ------------------------------------------------------------- rotation helpers

def rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    angle = rotation_angle(rotation)
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    return axis / np.linalg.norm(axis) * angle


# --------------------------------------------------------------- the simulation

def _world_inertia(diagonal, orientation_quaternion) -> np.ndarray:
    rotation = np.array(p.getMatrixFromQuaternion(orientation_quaternion)
                        ).reshape(3, 3)
    return rotation @ np.diag(diagonal) @ rotation.T


def total_momentum(body: int) -> tuple:
    """(linear, angular) momentum about the world origin, and a scale for each."""
    linear = np.zeros(3)
    angular = np.zeros(3)
    linear_scale = 0.0
    angular_scale = 0.0

    position, orientation = p.getBasePositionAndOrientation(body)
    velocity, omega = p.getBaseVelocity(body)
    info = p.getDynamicsInfo(body, -1)
    position = np.array(position)
    velocity = np.array(velocity)
    omega = np.array(omega)
    inertia = _world_inertia(info[2], orientation)
    linear += info[0] * velocity
    angular += info[0] * np.cross(position, velocity) + inertia @ omega
    linear_scale += info[0] * np.linalg.norm(velocity)
    angular_scale += np.linalg.norm(inertia @ omega)

    for link in range(p.getNumJoints(body)):
        info = p.getDynamicsInfo(body, link)
        if info[0] == 0.0:
            continue
        state = p.getLinkState(body, link, computeLinkVelocity=1)
        centre = np.array(state[0])
        velocity = np.array(state[6])
        omega = np.array(state[7])
        inertia = _world_inertia(info[2], state[1])
        linear += info[0] * velocity
        angular += info[0] * np.cross(centre, velocity) + inertia @ omega
        linear_scale += info[0] * np.linalg.norm(velocity)
        angular_scale += (np.linalg.norm(inertia @ omega)
                          + info[0] * np.linalg.norm(np.cross(centre, velocity)))

    return linear, angular, max(linear_scale, 1e-12), max(angular_scale, 1e-12)


def simulate_loop(body: int, amplitude: float, dt: float,
                  bus_mass: float = None, direction: int = 1,
                  revolutions: int = 1, watch_momentum: bool = False) -> dict:
    """Drive the loop with position control on a free base; report what moved."""
    if bus_mass is not None:
        p.changeDynamics(body, -1, mass=bus_mass,
                         localInertiaDiagonal=[bus_mass * BUS_GYRATION_SQUARED] * 3)
    p.resetBasePositionAndOrientation(body, [0, 0, 0], [0, 0, 0, 1])
    p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
    for joint, angle in zip(ARM_JOINTS, HOME):
        p.resetJointState(body, joint, float(angle), targetVelocity=0.0)
    for joint in FINGER_JOINTS:
        p.resetJointState(body, joint, 0.0, targetVelocity=0.0)

    p.setTimeStep(dt)
    steps = int(round(revolutions * PERIOD / dt))
    phase_rate = direction * 2.0 * np.pi / PERIOD
    momentum = 0.0

    for step in range(1, steps + 1):
        phase = phase_rate * (step * dt)
        targets = loop_reference(phase, amplitude)
        rates = loop_rates(phase, amplitude, phase_rate)
        p.setJointMotorControlArray(
            body, ARM_JOINTS, p.POSITION_CONTROL,
            targetPositions=list(targets), targetVelocities=list(rates),
            forces=[5.0e3] * len(ARM_JOINTS), positionGains=[1.0] * len(ARM_JOINTS))
        p.setJointMotorControlArray(
            body, FINGER_JOINTS, p.POSITION_CONTROL, targetPositions=[0.0, 0.0],
            forces=[1.0e3, 1.0e3])
        p.stepSimulation()

        if watch_momentum and step % 25 == 0:
            linear, angular, linear_scale, angular_scale = total_momentum(body)
            momentum = max(momentum,
                           np.linalg.norm(linear) / linear_scale,
                           np.linalg.norm(angular) / angular_scale)

    final_angles = np.array([p.getJointState(body, j)[0] for j in ARM_JOINTS])
    orientation = p.getBasePositionAndOrientation(body)[1]
    rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    return {
        "rotation": rotation,
        "angle": rotation_angle(rotation),
        "vector": rotation_vector(rotation),
        "closure": float(np.abs(final_angles - HOME).max()),
        "momentum": momentum,
    }


# ------------------------------------------------------ analytic loop integral

def integrate_base_rotation(model: FloatingBaseModel, amplitude: float,
                            samples: int, frame: str) -> np.ndarray:
    """Net base rotation from integrating omega_b = [-H_b^-1 H_bm qdot]_ang
    once around the loop, RK4 in the loop phase."""
    step = 2.0 * np.pi / samples

    def omega(phase: float) -> np.ndarray:
        angles = loop_reference(phase, amplitude)
        rates = loop_rates(phase, amplitude, 2.0 * np.pi / PERIOD)
        twist = model.base_velocity(torch.tensor(angles, dtype=DTYPE),
                                    torch.tensor(rates, dtype=DTYPE))
        # omega_b scales with phase_rate; integrating in phase divides it back
        return twist[3:].numpy() * PERIOD / (2.0 * np.pi)

    def exponential(vector: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(vector)
        if angle < 1e-15:
            return np.eye(3)
        unit = vector / angle
        cross = np.array([[0, -unit[2], unit[1]],
                          [unit[2], 0, -unit[0]],
                          [-unit[1], unit[0], 0]])
        return (np.eye(3) + np.sin(angle) * cross
                + (1 - np.cos(angle)) * cross @ cross)

    rotation = np.eye(3)
    for index in range(samples):
        phase = index * step
        first = omega(phase)
        middle = omega(phase + step / 2)
        last = omega(phase + step)
        increment = (first + 4 * middle + last) * step / 6
        if frame == "body":
            rotation = rotation @ exponential(increment)
        else:
            rotation = exponential(increment) @ rotation
    return rotation


# ------------------------------------------------------------------- the parts

def part_nominal(body: int) -> bool:
    print("part 1: the nominal loop, and momentum along it")
    coarse = simulate_loop(body, DEFAULT_AMPLITUDE, dt=1e-3,
                           watch_momentum=True)
    fine = simulate_loop(body, DEFAULT_AMPLITUDE, dt=2.5e-4,
                         watch_momentum=True)
    vector = fine["vector"]
    print(f"  joints {LOOP_JOINTS} swing +/-{DEFAULT_AMPLITUDE} rad and return")
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
        angle = np.degrees(simulate_loop(body, DEFAULT_AMPLITUDE, dt=dt)["angle"])
        change = "" if not angles else f"{angle - angles[-1]:+.4f}"
        angles.append(angle)
        print(f"  {dt:>10.2e}  {angle:>18.4f}  {change:>10}")

    increments = np.abs(np.diff(angles))
    shrinking = np.all(increments[1:] < increments[:-1])
    spread = abs(angles[-1] - angles[0]) / abs(angles[-1])
    richardson = angles[-1] + (angles[-1] - angles[-2])  # first-order extrapolate
    passed = shrinking and spread < 0.02 and abs(richardson) > 1.0
    print(f"  increments shrink each halving: {shrinking}")
    print(f"  coarse-to-fine spread: {spread * 100:.2f}% of the value")
    print(f"  extrapolated limit: {richardson:.3f} deg, non-zero")
    print(f"  -> {'converges, not drift' if passed else 'LOOKS LIKE DRIFT'}\n")
    return passed


def part_reversal(body: int) -> bool:
    print("part 3: reverses sign when the loop runs backward")
    forward = simulate_loop(body, DEFAULT_AMPLITUDE, dt=2.5e-4, direction=1)
    backward = simulate_loop(body, DEFAULT_AMPLITUDE, dt=2.5e-4, direction=-1)

    composed = forward["rotation"] @ backward["rotation"]
    residual = np.degrees(rotation_angle(composed))
    single = np.degrees(forward["angle"])
    forward_vector = forward["vector"]
    backward_vector = backward["vector"]
    alignment = (np.dot(forward_vector, backward_vector)
                 / (np.linalg.norm(forward_vector)
                    * np.linalg.norm(backward_vector)))

    print(f"  forward:  {single:.4f} deg about "
          f"[{forward_vector[0]:+.3f} {forward_vector[1]:+.3f} "
          f"{forward_vector[2]:+.3f}]")
    print(f"  backward: {np.degrees(backward['angle']):.4f} deg about "
          f"[{backward_vector[0]:+.3f} {backward_vector[1]:+.3f} "
          f"{backward_vector[2]:+.3f}]")
    print(f"  axis alignment forward.backward: {alignment:+.3f} (want -1)")
    print(f"  R_forward R_backward is {residual:.4f} deg from identity, "
          f"vs {single:.2f} deg for one loop")
    passed = alignment < -0.98 and residual < 0.1 * single
    print(f"  -> {'sign reverses, not drift' if passed else 'DOES NOT REVERSE'}\n")
    return passed


def part_analytic(body: int) -> bool:
    print("part 4: analytic -H_b^-1 H_bm qdot integrated vs the simulation")
    model = FloatingBaseModel(body, ARM_JOINTS, dtype=DTYPE)
    # Reset the base link mass the sweep's changeDynamics may have altered.
    p.changeDynamics(body, -1, mass=2.9,
                     localInertiaDiagonal=list(p.getDynamicsInfo(body, -1)[2]))

    truth = simulate_loop(body, DEFAULT_AMPLITUDE, dt=1.25e-4)
    truth_angle = np.degrees(truth["angle"])

    print(f"  {'frame':>8}  {'samples':>8}  {'net rotation (deg)':>18}  "
          f"{'vs sim':>10}")
    best = None
    for frame in ("body", "world"):
        for samples in (100, 400):
            rotation = integrate_base_rotation(model, DEFAULT_AMPLITUDE,
                                               samples, frame)
            angle = np.degrees(rotation_angle(rotation))
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
        for column, amplitude in enumerate(amplitudes):
            result = simulate_loop(body, float(amplitude), dt=1e-3,
                                   bus_mass=float(mass))
            excursion[row, column] = np.degrees(result["angle"])

    print(f"  {'bus kg':>8}  " + "  ".join(f"A={a:<4}" for a in amplitudes))
    for row, mass in enumerate(bus_masses):
        cells = "  ".join(f"{value:6.3f}" for value in excursion[row])
        print(f"  {mass:>8.0f}  {cells}")

    # Small-amplitude slope on a log-log plot should be about 2 (area ~ A^2),
    # and the bus-mass slope about -1.
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
              "bus inertia $m(0.5\\,\\mathrm{m})^2$; curves bend below slope 2\n"
              "past ~0.5 rad, where the loop is no longer small",
              transform=axis.transAxes, ha="right", va="bottom", fontsize=7.5,
              color="0.35")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=130)
    plt.close(figure)
    print(f"  wrote {FIGURE.relative_to(Path.cwd())}\n")


def disable_damping(body: int) -> None:
    """PyBullet applies 0.04 linear and 0.04 angular damping to every link
    by default. Damping is an external force: it bleeds momentum and the
    closed loop stops returning a clean geometric phase. Zero it."""
    for link in range(-1, p.getNumJoints(body)):
        p.changeDynamics(body, link, linearDamping=0.0, angularDamping=0.0,
                         jointDamping=0.0)


def main() -> None:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    body = p.loadURDF("franka_panda/panda.urdf", useFixedBase=False,
                      basePosition=[0, 0, 0])
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
